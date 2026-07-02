package worker

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"net/rpc"
	"sync"
	"time"

	"github.com/admwoo/distributed-ml-cluster/config"
	"github.com/admwoo/distributed-ml-cluster/datastore"
	"github.com/admwoo/distributed-ml-cluster/paramserver"
)

// --- Types ---

type Worker struct {
	// identities
	nodeID        int
	coordPeers    []string // coordinator addresses for heartbeat + re-election
	sidecarAddr   string   // "http://localhost:5000"
	datastoreAddr string
	
	lastHeartbeat time.Time // last time coordinator called our Heartbeat()

	mu            sync.Mutex
	listener      net.Listener
}

// --- RPC types ---

type ComputeGradientArgs struct{ Params paramserver.Params }
type ComputeGradientReply struct {
	Gradients []float64
	RowCount  int
	Loss      float64
}

type HeartbeatArgs struct{ NodeID int }
type HeartbeatReply struct{}

type EvaluateArgs struct{ Params paramserver.Params }
type EvaluateReply struct {
	Correct int
	Total   int
}

type PingArgs struct{}
type PingReply struct{}

// local stubs for outbound coordinator RPCs — avoids circular import
type coordHeartbeatArgs struct{ NodeID int }
type coordHeartbeatReply struct{}
type coordRequestElectionArgs struct{}
type coordRequestElectionReply struct{}

// --- Lifecycle ---

// Make starts the worker RPC server, waits for the sidecar, loads the local shard, and
// starts background goroutines. coordPeers are coordinator addresses for all nodes.
// listenAddr is the TCP address this worker listens on (e.g. "127.0.0.1:10085").
// sidecarAddr is the HTTP address of the local Python sidecar (e.g. "http://localhost:5000").
func Make(nodeID int, coordPeers []string, datastoreAddr string, listenAddr string, sidecarAddr string) *Worker {
	w := &Worker{
		nodeID:        nodeID,
		coordPeers:    coordPeers,
		sidecarAddr:   sidecarAddr,
		datastoreAddr: datastoreAddr,
	}

	rpcs := rpc.NewServer()
	rpcs.Register(w)

	addr := listenAddr
	l, e := net.Listen("tcp", addr)
	if e != nil {
		log.Fatal("worker: listen error: ", e)
	}
	w.listener = l

	// starts RPC server
	go func() {
		for {
			conn, err := l.Accept()
			if err != nil {
				return
			}
			go rpcs.ServeConn(conn)
		}
	}()
	// blocks until sidecar server returns 200, with timeout
	w.waitForSidecar()
	
	// send rpc to datastore to get shards
	rows := w.fetchShard()
	// POST shards to the sidecar
	w.loadShard(rows)

	// spawns failure detection goroutines
	go w.heartbeatSender()
	go w.coordinatorWatchdog()

	return w
}

func (w *Worker) Kill() {
	w.listener.Close()
}

// --- RPC handlers ---

// ComputeGradient receives current params, delegates to sidecar, and returns the
// gradient vector, the shard's row count, and the regularized loss.
func (w *Worker) ComputeGradient(args *ComputeGradientArgs, reply *ComputeGradientReply) error {
	gradients, rowCount, loss, err := w.requestGradients(args.Params)
	if err != nil {
		return fmt.Errorf("sidecar gradient error: %w", err)
	}
	reply.Gradients = gradients
	reply.RowCount = rowCount
	reply.Loss = loss
	return nil
}

// Heartbeat is called by the coordinator to signal it is alive.
func (w *Worker) Heartbeat(args *HeartbeatArgs, reply *HeartbeatReply) error {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.lastHeartbeat = time.Now()
	return nil
}

// Evaluate calls the local sidecar and returns correct/total counts for this shard.
func (w *Worker) Evaluate(args *EvaluateArgs, reply *EvaluateReply) error {
	correct, total, err := w.requestEvaluation(args.Params)
	if err != nil {
		return err
	}
	reply.Correct = correct
	reply.Total = total
	return nil
}

// Ping returns OK; used by new coordinator during recovery to check liveness.
func (w *Worker) Ping(args *PingArgs, reply *PingReply) error {
	return nil
}

// --- Goroutines ---

// heartbeatSender periodically calls Coordinator.Heartbeat on all coordinator peers
// so the coordinator knows this worker is alive.
func (w *Worker) heartbeatSender() {
	// worker broadcasts to all coordinators, all non-leaders ignore
	args := coordHeartbeatArgs{NodeID: w.nodeID}
	for {
		for _, peer := range w.coordPeers {
			call(peer, "Coordinator.Heartbeat", &args, &coordHeartbeatReply{})
		}
		time.Sleep(config.HeartBeatInterval * time.Second)
	}
}

// coordinatorWatchdog monitors the coordinator leader's heartbeat and triggers re-election on timeout.
func (w *Worker) coordinatorWatchdog() {
	timeout := time.Duration(config.HeartbeatTimeout) * time.Second
	for {
		time.Sleep(config.HeartBeatInterval * time.Second)

		w.mu.Lock()
		last := w.lastHeartbeat
		w.mu.Unlock()

		if !last.IsZero() && time.Since(last) > timeout {
			for _, peer := range w.coordPeers {
				call(peer, "Coordinator.RequestElection",
					&coordRequestElectionArgs{}, &coordRequestElectionReply{})
			}
		}
	}
}

// --- Sidecar communication ---

func (w *Worker) waitForSidecar() {
	for i := 0; i < 30; i++ {
		resp, err := http.Get(w.sidecarAddr + "/health")
		if err == nil && resp.StatusCode == 200 {
			return
		}
		time.Sleep(time.Second)
	}
	log.Fatal("worker: sidecar not ready after 30 seconds")
}

func (w *Worker) fetchShard() []datastore.Row {
	args := datastore.GetShardArgs{}
	reply := datastore.GetShardReply{}
	if !call(w.datastoreAddr, "DataStore.GetShard", &args, &reply) {
		log.Fatal("worker: failed to fetch shard from datastore")
	}
	return reply.Rows
}

type loadShardRequest struct {
	Rows []datastore.Row `json:"rows"`
}

func (w *Worker) loadShard(rows []datastore.Row) {
	body, _ := json.Marshal(loadShardRequest{Rows: rows})
	resp, err := http.Post(w.sidecarAddr+"/load_shard", "application/json", bytes.NewReader(body))
	if err != nil || resp.StatusCode != 200 {
		log.Fatal("worker: failed to load shard into sidecar")
	}
}

type gradientRequest struct {
	Params []float64 `json:"params"`
}

type gradientResponse struct {
	Gradients []float64 `json:"gradients"`
	RowCount  int       `json:"row_count"`
	Loss      float64   `json:"loss"`
}

func (w *Worker) requestGradients(params paramserver.Params) ([]float64, int, float64, error) {
	req := gradientRequest{Params: params.Weights}
	body, err := json.Marshal(req)
	if err != nil {
		return nil, 0, 0, err
	}
	resp, err := http.Post(w.sidecarAddr+"/gradient", "application/json", bytes.NewReader(body))
	if err != nil || resp.StatusCode != 200 {
		return nil, 0, 0, fmt.Errorf("sidecar unavailable on node %d", w.nodeID)
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, 0, 0, err
	}
	var gr gradientResponse
	if err := json.Unmarshal(data, &gr); err != nil {
		return nil, 0, 0, err
	}
	return gr.Gradients, gr.RowCount, gr.Loss, nil
}

type evaluateRequest struct {
	Params []float64 `json:"params"`
}

type evaluateResponse struct {
	Correct int `json:"correct"`
	Total   int `json:"total"`
}

func (w *Worker) requestEvaluation(params paramserver.Params) (int, int, error) {
	req := evaluateRequest{Params: params.Weights}
	body, err := json.Marshal(req)
	if err != nil {
		return 0, 0, err
	}
	resp, err := http.Post(w.sidecarAddr+"/evaluate", "application/json", bytes.NewReader(body))
	if err != nil || resp.StatusCode != 200 {
		return 0, 0, fmt.Errorf("sidecar evaluate failed on node %d", w.nodeID)
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return 0, 0, err
	}
	var er evaluateResponse
	if err := json.Unmarshal(data, &er); err != nil {
		return 0, 0, err
	}
	return er.Correct, er.Total, nil
}

type reloadShardRequest struct {
	Rows []datastore.Row `json:"rows"`
}

// reloadShard appends rescued replica rows to the sidecar's dataset after ActivateReplica.
func (w *Worker) reloadShard(rows []datastore.Row) error {
	body, err := json.Marshal(reloadShardRequest{Rows: rows})
	if err != nil {
		return err
	}
	resp, err := http.Post(w.sidecarAddr+"/reload_shard", "application/json", bytes.NewReader(body))
	if err != nil || resp.StatusCode != 200 {
		return fmt.Errorf("sidecar reload_shard failed on node %d", w.nodeID)
	}
	return nil
}

// --- Helpers ---

func call(srv string, name string, args any, reply any) bool {
	c, err := rpc.Dial("tcp", srv)
	if err != nil {
		return false
	}
	defer c.Close()
	return c.Call(name, args, reply) == nil
}
