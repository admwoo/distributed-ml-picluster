package datastore

import (
	"encoding/csv"
	"fmt"
	"hash/fnv"
	"log"
	"net"
	"net/rpc"
	"os"
	"sort"
	"strconv"
	"sync"
)

const DefaultVNodes = 1

// --- Types ---

type Row struct {
	Features []float64
	Label    int
}

type RingEntry struct {
	pos    uint32
	nodeID int
}

type DataStore struct {
	mu            sync.Mutex
	nodeID        int
	peers         []string
	ring          []RingEntry
	vnodes        int
	primaryShard  []Row
	replicaShard  []Row
	replicaActive bool
	listener      net.Listener
}

// --- RPC types ---

type GetShardArgs struct{}
type GetShardReply struct{ Rows []Row }

type ReceiveReplicaArgs struct{ Rows []Row }
type ReceiveReplicaReply struct{}

type ActivateReplicaArgs struct{}
type ActivateReplicaReply struct{}

type GetRescuedShardArgs struct{}
type GetRescuedShardReply struct{ Rows []Row }

// --- Lifecycle ---

// Make loads the dataset, builds the hash ring, and pushes this node's shard to its right neighbor.
func Make(nodeID int, peers []string, dataPath string, vnodes int) *DataStore {
	ds := &DataStore{}
	ds.nodeID = nodeID
	ds.peers = peers
	ds.vnodes = vnodes
	ds.ring = ds.buildRing()

	allRows, err := loadCSV(dataPath)
	if err != nil {
		log.Fatal("datastore: failed to load dataset: ", err)
	}

	for i, row := range allRows {
		if ds.ownerOf(i) == nodeID {
			ds.primaryShard = append(ds.primaryShard, row)
		}
	}

	rpcs := rpc.NewServer()
	rpcs.Register(ds)

	l, e := net.Listen("tcp", peers[nodeID])
	if e != nil {
		log.Fatal("datastore: listen error: ", e)
	}
	ds.listener = l

	go func() {
		for {
			conn, err := l.Accept()
			if err != nil {
				return
			}
			go rpcs.ServeConn(conn)
		}
	}()

	rightNeighbor := (nodeID + 1) % len(peers)
	args := ReceiveReplicaArgs{Rows: ds.primaryShard}
	reply := ReceiveReplicaReply{}
	call(peers[rightNeighbor], "DataStore.ReceiveReplica", &args, &reply)

	return ds
}

// --- Hash ring ---

// buildRing constructs the sorted virtual-node ring; call once during Make.
func (ds *DataStore) buildRing() []RingEntry {
	// TODO: for each physical node 0..len(peers)-1 and each vnode index 0..vnodes-1,
	// hash the key fmt.Sprintf("%d-%d", nodeID, v) and append RingEntry{pos, nodeID}.
	// Sort the result by pos before returning.
	return nil
}

// ownerOf returns the physical node responsible for the given row index.
func (ds *DataStore) ownerOf(rowIndex int) int {
	// TODO: compute rowPos = fnv32(strconv.Itoa(rowIndex)), then walk ds.ring
	// clockwise to find the first entry with pos >= rowPos.
	// If none found, wrap around to ds.ring[0].
	// Return that entry's nodeID.
	return 0
}

// fnv32 hashes a string to a uint32 ring position.
func fnv32(key string) uint32 {
	h := fnv.New32a()
	h.Write([]byte(key))
	return h.Sum32()
}

// --- CSV loading ---

// loadCSV reads all rows; last column is label, remaining columns are features.
func loadCSV(path string) ([]Row, error) {
	// TODO: os.Open(path), csv.NewReader, reader.ReadAll(),
	// then for each record parse all but last column as float64 Features,
	// and the last column as int Label.
	return nil, nil
}

// --- RPC handlers ---

// GetShard returns this node's primary shard.
func (ds *DataStore) GetShard(args *GetShardArgs, reply *GetShardReply) error {
	ds.mu.Lock()
	defer ds.mu.Unlock()
	reply.Rows = ds.primaryShard
	return nil
}

// ReceiveReplica stores the shard pushed from the left neighbor on startup.
func (ds *DataStore) ReceiveReplica(args *ReceiveReplicaArgs, reply *ReceiveReplicaReply) error {
	ds.mu.Lock()
	defer ds.mu.Unlock()
	ds.replicaShard = args.Rows
	return nil
}

// ActivateReplica is called by the coordinator when the left neighbor has died.
func (ds *DataStore) ActivateReplica(args *ActivateReplicaArgs, reply *ActivateReplicaReply) error {
	ds.mu.Lock()
	defer ds.mu.Unlock()
	ds.replicaActive = true
	return nil
}

// GetRescuedShard returns the replica shard; only valid after ActivateReplica.
func (ds *DataStore) GetRescuedShard(args *GetRescuedShardArgs, reply *GetRescuedShardReply) error {
	ds.mu.Lock()
	defer ds.mu.Unlock()
	if !ds.replicaActive {
		return fmt.Errorf("replica not activated on node %d", ds.nodeID)
	}
	reply.Rows = ds.replicaShard
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

var _ = sort.Slice
var _ = strconv.Itoa
var _ = csv.NewReader
var _ = os.Open
