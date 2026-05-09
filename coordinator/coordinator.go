package coordinator

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"net"
	"net/http"
	"net/rpc"
	"sync"
	"sync/atomic"
	"time"

	"github.com/admwoo/distributed-ml-cluster/config"
	"github.com/admwoo/distributed-ml-cluster/paramserver"
	"github.com/admwoo/distributed-ml-cluster/paxos"
)

// --- Types ---

type Coordinator struct {
	mu                sync.Mutex
	dead              int32
	me                int
	peers             []string // coordinator RPC addresses, indexed by nodeID
	paramPeers        []string // param server RPC addresses, indexed by nodeID
	workerPeers       []string // worker RPC addresses, indexed by nodeID
	px                *paxos.Paxos
	sidecarAddr       string
	isLeader          bool
	currentEpoch      int
	lastCheckpoint    int
	workerList        []int // ALL active node IDs: coordinator + param nodes + workers
	paramPrimaryID    int   // node ID of current param primary (-1 if unset)
	paramBackupID     int   // node ID of current param backup (-1 if unset)
	failedNodes       map[int]bool
	lastHeartbeat     map[int]time.Time
	heartbeatInterval time.Duration
	heartbeatTimeout  time.Duration
	recoveryTimeout   time.Duration
	electionInstance  int
	listener          net.Listener
}

// --- RPC types ---

type HeartbeatArgs struct{ NodeID int }
type HeartbeatReply struct{}

type GradientResultArgs struct {
	NodeID    int
	Gradients []float64
}
type GradientResultReply struct{}

type RequestElectionArgs struct{}
type RequestElectionReply struct{}

type PingArgs struct{}
type PingReply struct{}

// local stubs for outbound worker RPCs — avoids circular import
type workerComputeArgs struct{ Params paramserver.Params }
type workerComputeReply struct{ Gradients []float64 }

// internal type for concurrent gradient collection
type gradientResult struct {
	nodeID    int
	gradients []float64
	err       error
}

// --- Lifecycle ---

// Make starts the coordinator RPC server and launches the election+role goroutine.
// peers: coordinator addresses (e.g. "10.0.0.61:8081")
// paxosPeers: paxos addresses (e.g. "10.0.0.61:8080")
// workerPeers: worker RPC addresses indexed by nodeID (e.g. "127.0.0.1:9301")
// paramPeers: param server addresses indexed by nodeID; pass nil to derive from peers via paramAddr
// sidecarAddr: HTTP address of the local Python sidecar (e.g. "http://localhost:5000")
func Make(nodeID int, peers []string, paxosPeers []string, workerPeers []string, paramPeers []string, sidecarAddr string) *Coordinator {
	if paramPeers == nil {
		paramPeers = make([]string, len(peers))
		for i, p := range peers {
			paramPeers[i] = paramAddr(p)
		}
	}
	c := &Coordinator{
		me:                nodeID,
		peers:             peers,
		paramPeers:        paramPeers,
		workerPeers:       workerPeers,
		sidecarAddr:       sidecarAddr,
		paramPrimaryID:    -1,
		paramBackupID:     -1,
		failedNodes:       make(map[int]bool),
		lastHeartbeat:     make(map[int]time.Time),
		heartbeatInterval: config.HeartBeatInterval * time.Second,
		heartbeatTimeout:  config.HeartbeatTimeout * time.Second,
		recoveryTimeout:   config.RecoveryTimeout * time.Second,
	}

	c.px = paxos.Make(paxosPeers, nodeID)

	rpcs := rpc.NewServer()
	rpcs.Register(c)

	l, e := net.Listen("tcp", peers[nodeID])
	if e != nil {
		log.Fatal("coordinator: listen error: ", e)
	}
	c.listener = l

	go func() {
		for {
			conn, err := l.Accept()
			if err != nil {
				return
			}
			go rpcs.ServeConn(conn)
		}
	}()

	go c.startNode()

	return c
}

func (c *Coordinator) Kill() {
	atomic.StoreInt32(&c.dead, 1)
	c.listener.Close()
	c.px.Kill()
}

func (c *Coordinator) isdead() bool {
	return atomic.LoadInt32(&c.dead) != 0
}

// --- RPC handlers ---

// Heartbeat is called by any peer to signal liveness. Updates lastHeartbeat[nodeID].
func (c *Coordinator) Heartbeat(args *HeartbeatArgs, reply *HeartbeatReply) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.lastHeartbeat[args.NodeID] = time.Now()
	return nil
}

// GradientResult is called by workers to deliver a computed gradient vector.
func (c *Coordinator) GradientResult(args *GradientResultArgs, reply *GradientResultReply) error {
	return nil
}

// RequestElection triggers a new Paxos election on the next instance.
func (c *Coordinator) RequestElection(args *RequestElectionArgs, reply *RequestElectionReply) error {
	go c.triggerReElection()
	return nil
}

// Ping returns OK; used by new coordinator during recovery to check node liveness.
func (c *Coordinator) Ping(args *PingArgs, reply *PingReply) error {
	return nil
}

// --- Election and role transition ---

// startNode runs on every node at startup: elect a coordinator, then branch.
func (c *Coordinator) startNode() {
	winnerID := c.runElection()
	if winnerID == c.me {
		c.becomeCoordinator()
	} else {
		c.becomeWorker(winnerID)
	}
}

// runElection proposes own nodeID on the current electionInstance and polls until Paxos decides.
func (c *Coordinator) runElection() int {
	c.mu.Lock()
	seq := c.electionInstance
	c.mu.Unlock()

	c.px.Start(seq, c.me)

	for !c.isdead() {
		fate, val := c.px.Status(seq)
		if fate == paxos.Decided {
			return val.(int)
		}
		time.Sleep(20 * time.Millisecond)
	}
	return -1
}

func (c *Coordinator) becomeCoordinator() {
	c.mu.Lock()
	c.isLeader = true
	c.mu.Unlock()

	// wait for at least 2 healthy non-coordinator nodes to assign param roles
	deadline := time.Now().Add(c.recoveryTimeout)
	var candidates []int
	for time.Now().Before(deadline) && !c.isdead() {
		candidates = []int{}
		c.mu.Lock()
		for i := range c.peers {
			if i != c.me && c.isHealthy(i) {
				candidates = append(candidates, i)
			}
		}
		c.mu.Unlock()
		if len(candidates) >= 2 {
			break
		}
		time.Sleep(c.heartbeatInterval)
	}
	if c.isdead() {
		return
	}
	if len(candidates) < 2 {
		log.Fatal("coordinator: not enough healthy nodes after recovery timeout")
	}

	rand.Shuffle(len(candidates), func(i, j int) {
		candidates[i], candidates[j] = candidates[j], candidates[i]
	})

	c.mu.Lock()
	c.paramPrimaryID = candidates[0]
	c.paramBackupID = candidates[1]
	// all node IDs train — coordinator and param nodes are also workers
	c.workerList = make([]int, len(c.peers))
	for i := range c.peers {
		c.workerList[i] = i
	}
	primID := c.paramPrimaryID
	backID := c.paramBackupID
	c.mu.Unlock()

	call(c.paramPeers[primID], "ParamServer.SetRole",
		&paramserver.SetRoleArgs{Role: config.RoleParamPrimary, BackupAddr: c.paramPeers[backID]},
		&paramserver.SetRoleReply{})
	call(c.paramPeers[backID], "ParamServer.SetRole",
		&paramserver.SetRoleArgs{Role: config.RoleParamBackup, BackupAddr: ""},
		&paramserver.SetRoleReply{})

	go c.heartbeatSender()
	go c.epochLoop()
}

func (c *Coordinator) becomeWorker(coordID int) {
	c.mu.Lock()
	c.isLeader = false
	c.mu.Unlock()

	go c.coordinatorWatchdog(coordID)
}

// --- Goroutines ---

// heartbeatSender runs on coordinator only; broadcasts to all coordinator and worker peers.
func (c *Coordinator) heartbeatSender() {
	args := HeartbeatArgs{NodeID: c.me}
	for !c.isdead() {
		for i, peer := range c.peers {
			if i != c.me {
				call(peer, "Coordinator.Heartbeat", &args, &HeartbeatReply{})
			}
		}
		for i, peer := range c.workerPeers {
			if i != c.me {
				call(peer, "Worker.Heartbeat", &args, &HeartbeatReply{})
			}
		}
		time.Sleep(c.heartbeatInterval)
	}
}

// coordinatorWatchdog runs on non-coordinator nodes; sends own heartbeat and triggers
// re-election if the coordinator's heartbeat goes stale.
func (c *Coordinator) coordinatorWatchdog(coordID int) {
	coordAddr := c.peers[coordID]
	for !c.isdead() {
		call(coordAddr, "Coordinator.Heartbeat", &HeartbeatArgs{NodeID: c.me}, &HeartbeatReply{})

		c.mu.Lock()
		last := c.lastHeartbeat[coordID]
		c.mu.Unlock()

		if !last.IsZero() && time.Since(last) > c.heartbeatTimeout {
			c.triggerReElection()
			return
		}
		time.Sleep(c.heartbeatInterval)
	}
}

// triggerReElection increments the election instance and races Paxos again.
func (c *Coordinator) triggerReElection() {
	c.mu.Lock()
	c.electionInstance++
	c.mu.Unlock()

	winnerID := c.runElection()
	if winnerID == c.me {
		c.becomeCoordinator()
	} else {
		c.becomeWorker(winnerID)
	}
}

// --- Epoch loop ---

func (c *Coordinator) epochLoop() {
	for !c.isdead() {
		c.mu.Lock()
		epoch := c.currentEpoch
		workerCount := len(c.workerList)
		c.mu.Unlock()

		start := time.Now()
		c.postCheckpoint(epoch, "started", 0.0)

		params, err := c.fetchParams()
		if err != nil {
			c.runParamRecovery()
			continue
		}

		gradients, failedIDs := c.delegateAll(params)
		if len(failedIDs) > 0 {
			log.Printf("coordinator: epoch %d — %d/%d workers failed: %v", epoch, len(failedIDs), workerCount, failedIDs)
			if c.runWorkerRecovery(failedIDs) {
				continue
			}
		}
		if len(gradients) == 0 {
			log.Printf("coordinator: epoch %d — no gradients received, skipping", epoch)
			continue
		}

		log.Printf("coordinator: epoch %d — gradients from %d/%d workers", epoch, len(gradients), workerCount)

		averaged := aggregateGradients(gradients)
		norm := gradNorm(averaged)
		log.Printf("coordinator: epoch %d — gradient norm %.6f", epoch, norm)

		if err := c.writeParams(params, averaged); err != nil {
			c.runParamRecovery()
			continue
		}

		c.mu.Lock()
		primID := c.paramPrimaryID
		backID := c.paramBackupID
		c.mu.Unlock()

		if epoch%config.CheckpointInterval == 0 {
			call(c.paramPeers[primID], "ParamServer.Checkpoint",
				&paramserver.CheckpointArgs{}, &paramserver.CheckpointReply{})
			call(c.paramPeers[backID], "ParamServer.Checkpoint",
				&paramserver.CheckpointArgs{}, &paramserver.CheckpointReply{})
			log.Printf("coordinator: epoch %d — checkpoint written", epoch)
		}

		accuracy := c.evaluateAccuracy()
		c.postCheckpoint(epoch, "complete", accuracy)

		c.mu.Lock()
		c.lastCheckpoint = c.currentEpoch
		c.currentEpoch++
		c.mu.Unlock()

		log.Printf("coordinator: epoch %d complete in %v — accuracy %.4f (%.1f%%)",
			epoch, time.Since(start).Round(time.Millisecond), accuracy, accuracy*100)

		if norm < config.ConvergenceThreshold {
			log.Printf("coordinator: converged at epoch %d — grad norm %.6f, accuracy %.1f%%", epoch, norm, accuracy*100)
			c.postCheckpoint(epoch, "converged", accuracy)
			return
		}
		if epoch+1 >= config.MaxEpochs {
			log.Printf("coordinator: max epochs (%d) reached — final accuracy %.1f%%", config.MaxEpochs, accuracy*100)
			c.postCheckpoint(epoch, "max_epochs_reached", accuracy)
			return
		}
	}
}

// --- Epoch loop helpers ---

// delegateAll sends ComputeGradient to all workers concurrently and collects results.
// Returns collected gradient vectors and IDs of workers that failed or timed out.
func (c *Coordinator) delegateAll(params paramserver.Params) ([][]float64, []int) {
	c.mu.Lock()
	workerList := append([]int{}, c.workerList...)
	c.mu.Unlock()

	resultCh := make(chan gradientResult, len(workerList))

	for _, workerID := range workerList {
		go func(id int) {
			if id == c.me {
				g, err := c.callSidecar(params)
				resultCh <- gradientResult{id, g, err}
			} else {
				args := workerComputeArgs{Params: params}
				reply := workerComputeReply{}
				if !call(c.workerPeers[id], "Worker.ComputeGradient", &args, &reply) {
					resultCh <- gradientResult{id, nil, fmt.Errorf("rpc failed")}
					return
				}
				resultCh <- gradientResult{id, reply.Gradients, nil}
			}
		}(workerID)
	}

	timeout := time.After(time.Duration(config.GradientTimeout) * time.Second)
	var gradients [][]float64
	var failedIDs []int
	responded := make(map[int]bool)

loop:
	for len(responded) < len(workerList) {
		select {
		case r := <-resultCh:
			responded[r.nodeID] = true
			if r.err != nil {
				failedIDs = append(failedIDs, r.nodeID)
			} else {
				gradients = append(gradients, r.gradients)
			}
		case <-timeout:
			if config.SGDMode == config.SyncSGD {
				for _, id := range workerList {
					if !responded[id] {
						failedIDs = append(failedIDs, id)
					}
				}
			}
			break loop
		}
	}

	return gradients, failedIDs
}

// fetchParams reads current params from primary, falls back to backup, then LoadCheckpoint.
// Returns zero-initialized params if no checkpoint exists yet (epoch 0).
func (c *Coordinator) fetchParams() (paramserver.Params, error) {
	c.mu.Lock()
	primID := c.paramPrimaryID
	backID := c.paramBackupID
	paramPeers := append([]string{}, c.paramPeers...)
	c.mu.Unlock()

	args := paramserver.ReadParamsArgs{}
	reply := paramserver.ReadParamsReply{}

	if call(paramPeers[primID], "ParamServer.ReadParams", &args, &reply) {
		return initParams(reply.Params), nil
	}
	if call(paramPeers[backID], "ParamServer.ReadParams", &args, &reply) {
		return initParams(reply.Params), nil
	}

	cpArgs := paramserver.LoadCheckpointArgs{}
	cpReply := paramserver.LoadCheckpointReply{}
	for _, peer := range paramPeers {
		if call(peer, "ParamServer.LoadCheckpoint", &cpArgs, &cpReply) {
			return initParams(cpReply.Params), nil
		}
	}

	return paramserver.Params{}, fmt.Errorf("fetchParams: all param sources failed")
}

// initParams returns p unchanged if it has weights; otherwise returns zero-initialized params.
func initParams(p paramserver.Params) paramserver.Params {
	if len(p.Weights) > 0 {
		return p
	}
	return paramserver.Params{
		Weights: make([]float64, config.NumFeatures*config.NumClasses),
		Bias:    make([]float64, config.NumClasses),
	}
}

// writeParams applies an SGD update using old params and the averaged gradient, then writes to primary.
func (c *Coordinator) writeParams(old paramserver.Params, gradients []float64) error {
	weightGrads := gradients[:config.NumFeatures*config.NumClasses]
	biasGrads := gradients[config.NumFeatures*config.NumClasses:]

	newWeights := make([]float64, len(old.Weights))
	for i := range newWeights {
		newWeights[i] = old.Weights[i] - config.LearningRate*weightGrads[i]
	}
	newBias := make([]float64, len(old.Bias))
	for i := range newBias {
		newBias[i] = old.Bias[i] - config.LearningRate*biasGrads[i]
	}

	c.mu.Lock()
	primID := c.paramPrimaryID
	c.mu.Unlock()

	wArgs := paramserver.WriteParamsArgs{Params: paramserver.Params{Weights: newWeights, Bias: newBias}}
	wReply := paramserver.WriteParamsReply{}
	if !call(c.paramPeers[primID], "ParamServer.WriteParams", &wArgs, &wReply) {
		return fmt.Errorf("writeParams: RPC to primary failed")
	}
	return nil
}

type evalResult struct{ correct, total int }

// evaluateAccuracy asks every worker (and the coordinator's own sidecar) to evaluate
// their local shard, then returns the weighted average accuracy across all rows.
func (c *Coordinator) evaluateAccuracy() float64 {
	params, err := c.fetchParams()
	if err != nil {
		return 0.0
	}

	c.mu.Lock()
	workerList := append([]int{}, c.workerList...)
	c.mu.Unlock()

	ch := make(chan evalResult, len(workerList))

	for _, id := range workerList {
		go func(wid int) {
			if wid == c.me {
				correct, total, err := c.callSidecarEvaluate(params)
				if err != nil {
					ch <- evalResult{}
					return
				}
				ch <- evalResult{correct, total}
			} else {
				args := workerEvaluateArgs{Params: params}
				reply := workerEvaluateReply{}
				if !call(c.workerPeers[wid], "Worker.Evaluate", &args, &reply) {
					ch <- evalResult{}
					return
				}
				ch <- evalResult{reply.Correct, reply.Total}
			}
		}(id)
	}

	var totalCorrect, totalRows int
	for range workerList {
		r := <-ch
		totalCorrect += r.correct
		totalRows += r.total
	}
	if totalRows == 0 {
		return 0.0
	}
	return float64(totalCorrect) / float64(totalRows)
}

func (c *Coordinator) callSidecarEvaluate(params paramserver.Params) (int, int, error) {
	req := sidecarGradientReq{
		Weights: reshapeWeights(params.Weights),
		Bias:    params.Bias,
	}
	body, err := json.Marshal(req)
	if err != nil {
		return 0, 0, err
	}
	resp, err := monitorClient.Post(c.sidecarAddr+"/evaluate", "application/json", bytes.NewReader(body))
	if err != nil || resp.StatusCode != 200 {
		return 0, 0, fmt.Errorf("sidecar evaluate unavailable on coordinator %d", c.me)
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return 0, 0, err
	}
	var er sidecarEvaluateResp
	if err := json.Unmarshal(data, &er); err != nil {
		return 0, 0, err
	}
	return er.Correct, er.Total, nil
}

// --- Recovery stubs (implemented after happy path is verified) ---

func (c *Coordinator) runParamRecovery() {
	c.mu.Lock()
	epoch := c.currentEpoch
	c.mu.Unlock()
	log.Printf("coordinator: param recovery triggered on epoch %d (not yet implemented)", epoch)
}

func (c *Coordinator) runWorkerRecovery(failedIDs []int) bool {
	c.mu.Lock()
	epoch := c.currentEpoch
	c.mu.Unlock()
	log.Printf("coordinator: worker recovery for nodes %v on epoch %d (not yet implemented)", failedIDs, epoch)
	return true
}

// --- Sidecar communication ---

type sidecarGradientReq struct {
	Weights [][]float64 `json:"weights"`
	Bias    []float64   `json:"bias"`
}

type sidecarGradientResp struct {
	Gradients []float64 `json:"gradients"`
}

type sidecarEvaluateResp struct {
	Correct int `json:"correct"`
	Total   int `json:"total"`
}

type workerEvaluateArgs struct{ Params paramserver.Params }
type workerEvaluateReply struct{ Correct, Total int }

// callSidecar posts params to the coordinator's local sidecar and returns the gradient vector.
func (c *Coordinator) callSidecar(params paramserver.Params) ([]float64, error) {
	req := sidecarGradientReq{
		Weights: reshapeWeights(params.Weights),
		Bias:    params.Bias,
	}
	body, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}
	resp, err := http.Post(c.sidecarAddr+"/gradient", "application/json", bytes.NewReader(body))
	if err != nil || resp.StatusCode != 200 {
		return nil, fmt.Errorf("sidecar unavailable on coordinator node %d", c.me)
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	var gr sidecarGradientResp
	if err := json.Unmarshal(data, &gr); err != nil {
		return nil, err
	}
	return gr.Gradients, nil
}

// reshapeWeights converts a flat weight vector into a [NumClasses][NumFeatures] matrix.
func reshapeWeights(flat []float64) [][]float64 {
	matrix := make([][]float64, config.NumClasses)
	for i := range matrix {
		start := i * config.NumFeatures
		matrix[i] = flat[start : start+config.NumFeatures]
	}
	return matrix
}

// --- Monitor communication ---

type checkpointPost struct {
	Epoch    int     `json:"epoch"`
	Status   string  `json:"status"`
	Accuracy float64 `json:"accuracy"`
}

var monitorClient = &http.Client{Timeout: 200 * time.Millisecond}

// postCheckpoint posts epoch status to the monitor; non-fatal on failure.
func (c *Coordinator) postCheckpoint(epoch int, status string, accuracy float64) {
	if !config.MonitorEnabled {
		return
	}
	body, _ := json.Marshal(checkpointPost{epoch, status, accuracy})
	resp, err := monitorClient.Post("http://"+config.MonitorAddr+"/checkpoint", "application/json", bytes.NewReader(body))
	if err != nil {
		log.Printf("coordinator: monitor unreachable (%v)", err)
		return
	}
	resp.Body.Close()
}

// --- Gradient aggregation ---

func aggregateGradients(gradients [][]float64) []float64 {
	n := len(gradients)
	result := make([]float64, len(gradients[0]))
	for _, g := range gradients {
		for i, v := range g {
			result[i] += v / float64(n)
		}
	}
	return result
}

func gradNorm(g []float64) float64 {
	var sum float64
	for _, v := range g {
		sum += v * v
	}
	return math.Sqrt(sum)
}

// --- Helpers ---

func (c *Coordinator) isHealthy(nodeID int) bool {
	last, ok := c.lastHeartbeat[nodeID]
	if !ok {
		return false
	}
	return time.Since(last) <= c.heartbeatTimeout
}

// paramAddr derives the param server address from a coordinator peer address by substituting the port.
func paramAddr(coordAddr string) string {
	for i := len(coordAddr) - 1; i >= 0; i-- {
		if coordAddr[i] == ':' {
			return coordAddr[:i] + config.ParamPort
		}
	}
	return coordAddr
}

func call(srv string, name string, args any, reply any) bool {
	c, err := rpc.Dial("tcp", srv)
	if err != nil {
		return false
	}
	defer c.Close()
	return c.Call(name, args, reply) == nil
}
