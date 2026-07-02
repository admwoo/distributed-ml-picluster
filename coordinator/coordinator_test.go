package coordinator

import (
	"encoding/json"
	"net"
	"net/http"
	"net/http/httptest"
	"net/rpc"
	"testing"
	"time"

	"github.com/admwoo/distributed-ml-cluster/config"
	"github.com/admwoo/distributed-ml-cluster/paramserver"
)

// Test port ranges (no overlap with paxos: 9001-9005, paramserver: 9201-9202, datastore: 9101-9105)
//   coordinator RPC : 9301-9303
//   paxos           : 9311-9313
//   param server    : 9321-9323
//   fake workers    : 9331-9333

var (
	testCoordPeers  = []string{"127.0.0.1:9301", "127.0.0.1:9302", "127.0.0.1:9303"}
	testPaxosPeers  = []string{"127.0.0.1:9311", "127.0.0.1:9312", "127.0.0.1:9313"}
	testParamPeers  = []string{"127.0.0.1:9321", "127.0.0.1:9322", "127.0.0.1:9323"}
	testWorkerPeers = []string{"127.0.0.1:9331", "127.0.0.1:9332", "127.0.0.1:9333"}
)

func makeCoordCluster(t *testing.T, n int) []*Coordinator {
	t.Helper()
	coords := make([]*Coordinator, n)
	for i := range n {
		coords[i] = Make(i,
			testCoordPeers[:n],
			testPaxosPeers[:n],
			testWorkerPeers[:n],
			testParamPeers[:n],
			"http://localhost:5000", // not used in election test
		)
	}
	return coords
}

func killAll(coords []*Coordinator) {
	for _, c := range coords {
		c.Kill()
	}
	time.Sleep(100 * time.Millisecond)
}

// --- Unit tests ---

func TestAggregateGradients(t *testing.T) {
	gradients := [][]float64{
		{1.0, 2.0, 3.0},
		{3.0, 2.0, 1.0},
	}
	rowCounts := []int{50, 50}
	result := aggregateGradients(gradients, rowCounts)
	for i, v := range result {
		if v != 2.0 {
			t.Errorf("result[%d] = %f, want 2.0", i, v)
		}
	}
}

func TestAggregateGradientsIdentity(t *testing.T) {
	single := [][]float64{{1.0, 0.5, -1.0}}
	result := aggregateGradients(single, []int{50})
	for i, v := range result {
		if v != single[0][i] {
			t.Errorf("result[%d] = %f, want %f", i, v, single[0][i])
		}
	}
}

func TestAggregateGradientsWeighted(t *testing.T) {
	// worker A has 1 row, gradient [0]; worker B has 3 rows, gradient [4].
	// global mean should be (1*0 + 3*4) / 4 = 3.0, not the unweighted (0+4)/2 = 2.0.
	gradients := [][]float64{{0.0}, {4.0}}
	rowCounts := []int{1, 3}
	result := aggregateGradients(gradients, rowCounts)
	if result[0] != 3.0 {
		t.Errorf("weighted result = %f, want 3.0", result[0])
	}
}

// TestParamRecovery kills the param primary and verifies runParamRecovery promotes
// the surviving backup to primary, assigns a spare as the new backup, and re-seeds
// that spare with the survivor's params — all without a running epoch loop.
func TestParamRecovery(t *testing.T) {
	dir := t.TempDir()

	// three param servers: 0=primary, 1=backup, 2=spare
	ps := make([]*paramserver.ParamServer, 3)
	for i := range ps {
		ps[i] = paramserver.Make(testParamPeers[i], dir+"/ps"+string(rune('0'+i))+".json")
	}
	defer func() {
		for _, p := range ps {
			if p != nil {
				p.Kill()
			}
		}
	}()

	// assign roles and commit some params; the write replicates primary -> backup
	call(testParamPeers[0], "ParamServer.SetRole",
		&paramserver.SetRoleArgs{Role: config.RoleParamPrimary, BackupAddr: testParamPeers[1]},
		&paramserver.SetRoleReply{})
	call(testParamPeers[1], "ParamServer.SetRole",
		&paramserver.SetRoleArgs{Role: config.RoleParamBackup, BackupAddr: ""},
		&paramserver.SetRoleReply{})

	want := paramserver.Params{Weights: []float64{1, 2, 3}, Bias: []float64{4}}
	if !call(testParamPeers[0], "ParamServer.WriteParams",
		&paramserver.WriteParamsArgs{Params: want}, &paramserver.WriteParamsReply{}) {
		t.Fatal("initial WriteParams to primary failed")
	}

	// minimal coordinator: node 0 is the coordinator (and current primary) and dies.
	c := &Coordinator{
		me:             0,
		paramPeers:     testParamPeers[:3],
		paramPrimaryID: 0,
		paramBackupID:  1,
		workerList:     []int{0, 1, 2},
	}

	// kill the primary's param server, then recover
	ps[0].Kill()
	ps[0] = nil
	time.Sleep(50 * time.Millisecond)

	c.runParamRecovery()

	if c.paramPrimaryID != 1 {
		t.Errorf("paramPrimaryID = %d, want 1 (backup promoted)", c.paramPrimaryID)
	}
	if c.paramBackupID != 2 {
		t.Errorf("paramBackupID = %d, want 2 (spare assigned)", c.paramBackupID)
	}

	// the new backup (node 2) must have been re-seeded with the committed params
	reply := paramserver.ReadParamsReply{}
	if !call(testParamPeers[2], "ParamServer.ReadParams", &paramserver.ReadParamsArgs{}, &reply) {
		t.Fatal("ReadParams on new backup failed")
	}
	if len(reply.Params.Weights) != 3 || reply.Params.Weights[0] != 1 || reply.Params.Bias[0] != 4 {
		t.Errorf("new backup params = %+v, want %+v", reply.Params, want)
	}
}

func TestInitParams(t *testing.T) {
	c := &Coordinator{sidecarAddr: "http://127.0.0.1:59999"} // nothing listening here

	// non-empty params pass through unchanged
	full := paramserver.Params{Weights: []float64{1.0, 2.0, 3.0}}
	got, err := c.initParams(full)
	if err != nil {
		t.Fatalf("unexpected error on non-empty params: %v", err)
	}
	if len(got.Weights) != 3 || got.Weights[0] != 1.0 {
		t.Errorf("non-empty params modified: %+v", got)
	}

	// empty params with no reachable sidecar to seed from -> error
	if _, err := c.initParams(paramserver.Params{}); err == nil {
		t.Error("expected error seeding init params with no sidecar, got nil")
	}
}

func TestParamAddr(t *testing.T) {
	cases := []struct{ in, want string }{
		{"10.0.0.61:8081", "10.0.0.61" + config.ParamPort},
		{"127.0.0.1:9101", "127.0.0.1" + config.ParamPort},
	}
	for _, tc := range cases {
		got := paramAddr(tc.in)
		if got != tc.want {
			t.Errorf("paramAddr(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

// --- Election test ---

// TestElection verifies that exactly one of three coordinators wins the Paxos election.
func TestElection(t *testing.T) {
	coords := makeCoordCluster(t, 3)
	defer killAll(coords)

	deadline := time.Now().Add(3 * time.Second)
	var leaderCount int
	for time.Now().Before(deadline) {
		leaderCount = 0
		for _, c := range coords {
			c.mu.Lock()
			if c.isLeader {
				leaderCount++
			}
			c.mu.Unlock()
		}
		if leaderCount == 1 {
			break
		}
		time.Sleep(50 * time.Millisecond)
	}
	if leaderCount != 1 {
		t.Fatalf("expected exactly 1 leader after election, got %d", leaderCount)
	}
}

// --- Epoch loop integration test ---

// fakeWorker is a minimal RPC server that returns a fixed gradient for testing.
// Arg/reply types are exported so net/rpc's reflection-based method discovery accepts them.
type fakeWorker struct{ gradient []float64 }

type FakeComputeArgs struct{ Params paramserver.Params }
type FakeComputeReply struct {
	Gradients []float64
	RowCount  int
}
type FakeHeartbeatArgs struct{ NodeID int }
type FakeHeartbeatReply struct{}

func (fw *fakeWorker) ComputeGradient(args *FakeComputeArgs, reply *FakeComputeReply) error {
	reply.Gradients = append([]float64{}, fw.gradient...)
	reply.RowCount = 50
	return nil
}

func (fw *fakeWorker) Heartbeat(args *FakeHeartbeatArgs, reply *FakeHeartbeatReply) error {
	return nil
}

func startFakeWorker(t *testing.T, addr string, gradient []float64) net.Listener {
	t.Helper()
	rpcs := rpc.NewServer()
	// register under "Worker" so coordinator's "Worker.ComputeGradient" RPC resolves
	rpcs.RegisterName("Worker", &fakeWorker{gradient: gradient})
	l, err := net.Listen("tcp", addr)
	if err != nil {
		t.Fatalf("fake worker listen %s: %v", addr, err)
	}
	go func() {
		for {
			conn, err := l.Accept()
			if err != nil {
				return
			}
			go rpcs.ServeConn(conn)
		}
	}()
	return l
}

// fakeSidecar mimics the Flask sidecar's flat-param endpoints: /init_params seeds a
// zero vector the size of the gradient, /gradient returns the fixed gradient, and
// /evaluate returns fixed counts.
func fakeSidecar(t *testing.T, gradient []float64) *httptest.Server {
	t.Helper()
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/init_params":
			json.NewEncoder(w).Encode(map[string]any{"params": make([]float64, len(gradient))})
		case "/evaluate":
			json.NewEncoder(w).Encode(map[string]any{"correct": 40, "total": 50})
		default: // /gradient
			json.NewEncoder(w).Encode(map[string]any{"gradients": gradient, "row_count": 50, "loss": 0.5})
		}
	}))
	return ts
}

// TestEpochLoopHappyPath wires coordinator node 0 against real param servers and fake
// workers (including an httptest sidecar for the coordinator's own gradient), then
// verifies the epoch counter advances and params are updated via SGD.
func TestEpochLoopHappyPath(t *testing.T) {
	const n = 3
	dir := t.TempDir()

	// fixed gradient: 0.1 per element. Length is arbitrary now that params are an
	// opaque flat vector — it just has to match across init/gradient/SGD.
	const paramSize = 15
	grad := make([]float64, paramSize)
	for i := range grad {
		grad[i] = 0.1
	}

	// param servers on testParamPeers
	pss := make([]*paramserver.ParamServer, n)
	for i := range pss {
		pss[i] = paramserver.Make(testParamPeers[i], dir+"/ps"+string(rune('0'+i))+".json")
	}
	defer func() {
		for _, ps := range pss {
			ps.Kill()
		}
	}()

	// fake workers on testWorkerPeers (nodes 1 and 2; node 0 uses its own sidecar)
	fakeListeners := make([]net.Listener, n)
	for i := range fakeListeners {
		fakeListeners[i] = startFakeWorker(t, testWorkerPeers[i], grad)
	}
	defer func() {
		for _, l := range fakeListeners {
			l.Close()
		}
	}()

	// fake sidecar HTTP server for coordinator node 0's own gradient
	ts := fakeSidecar(t, grad)
	defer ts.Close()

	// 1-node paxos so node 0 trivially wins the election; full n-node coord/param/worker peers
	c := Make(0, testCoordPeers[:n], testPaxosPeers[:1], testWorkerPeers[:n], testParamPeers[:n], ts.URL)
	defer c.Kill()

	// simulate heartbeats from nodes 1 and 2 so becomeLeader considers them healthy
	go func() {
		for !c.isdead() {
			call(testCoordPeers[0], "Coordinator.Heartbeat", &HeartbeatArgs{NodeID: 1}, &HeartbeatReply{})
			call(testCoordPeers[0], "Coordinator.Heartbeat", &HeartbeatArgs{NodeID: 2}, &HeartbeatReply{})
			time.Sleep(200 * time.Millisecond)
		}
	}()

	// wait up to 8s for at least 2 epochs to complete
	deadline := time.Now().Add(8 * time.Second)
	for time.Now().Before(deadline) {
		c.mu.Lock()
		epoch := c.currentEpoch
		c.mu.Unlock()
		if epoch >= 2 {
			break
		}
		time.Sleep(200 * time.Millisecond)
	}

	c.mu.Lock()
	epoch := c.currentEpoch
	c.mu.Unlock()
	if epoch < 2 {
		t.Fatalf("epoch loop stalled at epoch %d (want >= 2)", epoch)
	}

	// verify params were written: weights should be non-zero after SGD (0 - 0.01*0.1 = -0.001)
	args := paramserver.ReadParamsArgs{}
	reply := paramserver.ReadParamsReply{}
	for _, ps := range pss[1:] { // param primary is one of nodes 1 or 2
		ps.ReadParams(&args, &reply)
		if len(reply.Params.Weights) > 0 {
			break
		}
	}
	if len(reply.Params.Weights) == 0 {
		t.Fatal("params never written after epoch loop ran")
	}
	for i, w := range reply.Params.Weights {
		if w == 0.0 {
			t.Errorf("weight[%d] = 0 after %d epochs (expected SGD update)", i, epoch)
			break
		}
	}
}
