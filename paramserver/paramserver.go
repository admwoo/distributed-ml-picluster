package paramserver

import (
	"encoding/json"
	"fmt"
	"log"
	"net"
	"net/rpc"
	"os"
	"sync"

	"github.com/admwoo/distributed-ml-cluster/config"
)

// --- Types ---

type Params struct {
	Weights []float64 // flat, length NumFeatures * NumClasses
	Bias    []float64 // length NumClasses
}

type ParamServer struct {
	mu             sync.Mutex
	role           string
	backupAddr     string
	params         Params
	checkpointPath string
	listener       net.Listener
}

// --- RPC types ---

type ReadParamsArgs struct{}
type ReadParamsReply struct{ Params Params }

type WriteParamsArgs struct{ Params Params }
type WriteParamsReply struct{}

type ReplicateParamsArgs struct{ Params Params }
type ReplicateParamsReply struct{}

type SetRoleArgs struct {
	Role       string
	BackupAddr string
}
type SetRoleReply struct{}

type CheckpointArgs struct{}
type CheckpointReply struct{}

type LoadCheckpointArgs struct{}
type LoadCheckpointReply struct{ Params Params }

// --- Lifecycle ---

// Make starts the param server with no role; coordinator assigns role via SetRole.
func Make(addr string, checkpointPath string) *ParamServer {
	ps := &ParamServer{checkpointPath: checkpointPath}

	rpcs := rpc.NewServer()
	rpcs.Register(ps)

	l, e := net.Listen("tcp", addr)
	if e != nil {
		log.Fatal("paramserver: listen error: ", e)
	}
	ps.listener = l

	go func() {
		for {
			conn, err := l.Accept()
			if err != nil {
				return
			}
			go rpcs.ServeConn(conn)
		}
	}()

	return ps
}

func (ps *ParamServer) Kill() {
	ps.listener.Close()
}

// --- RPC handlers ---

// ReadParams returns the current model parameters.
func (ps *ParamServer) ReadParams(args *ReadParamsArgs, reply *ReadParamsReply) error {
	ps.mu.Lock()
	defer ps.mu.Unlock()
	reply.Params = ps.params
	return nil
}

// WriteParams updates parameters; primary replicates to backup before confirming.
func (ps *ParamServer) WriteParams(args *WriteParamsArgs, reply *WriteParamsReply) error {
	ps.mu.Lock()

	if ps.role != config.RoleParamPrimary {
		ps.mu.Unlock()
		return fmt.Errorf("WriteParams called on non-primary node (role=%s)", ps.role)
	}
	repArgs := ReplicateParamsArgs{Params: args.Params}
	repReply := ReplicateParamsReply{}
	backupAddr := ps.backupAddr
	
	ps.mu.Unlock()

	ok := call(backupAddr, "ParamServer.ReplicateParams", &repArgs, &repReply)
	if !ok {
		return fmt.Errorf("replication to backup failed: %s", backupAddr)
	}

	ps.mu.Lock()
	defer ps.mu.Unlock()
	ps.params = args.Params

	return nil
}

// ReplicateParams is called by the primary to store params on the backup.
func (ps *ParamServer) ReplicateParams(args *ReplicateParamsArgs, reply *ReplicateParamsReply) error {
	ps.mu.Lock()
	defer ps.mu.Unlock()
	ps.params = args.Params

	return nil
}

// SetRole is called by the coordinator to assign or change this node's role.
func (ps *ParamServer) SetRole(args *SetRoleArgs, reply *SetRoleReply) error {
	ps.mu.Lock()
	defer ps.mu.Unlock()
	ps.role = args.Role
	ps.backupAddr = args.BackupAddr

	return nil
}

// Checkpoint writes current params to disk atomically.
func (ps *ParamServer) Checkpoint(args *CheckpointArgs, reply *CheckpointReply) error {
	ps.mu.Lock()
	params := ps.params
	ps.mu.Unlock()

	data, err := json.Marshal(params)
	if err != nil {
		return err
	}
	tmp := ps.checkpointPath + ".tmp"
	if err := os.WriteFile(tmp, data, 0644); err != nil {
		return err
	}
	return os.Rename(tmp, ps.checkpointPath)
}

// LoadCheckpoint reads the last persisted params from disk.
func (ps *ParamServer) LoadCheckpoint(args *LoadCheckpointArgs, reply *LoadCheckpointReply) error {
	data, err := os.ReadFile(ps.checkpointPath)
	if err != nil {
		return fmt.Errorf("no checkpoint at %s: %w", ps.checkpointPath, err)
	}
	return json.Unmarshal(data, &reply.Params)
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
