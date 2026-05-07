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
	"time"
)

const DefaultVNodes = 100 // Increase for larger dataset sizes

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
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if call(peers[rightNeighbor], "DataStore.ReceiveReplica", &args, &reply) {
			break
		}
		time.Sleep(20 * time.Millisecond)
	}

	return ds
}

// --- Hash ring ---

// buildRing constructs the sorted virtual-node ring; call once during Make.
func (ds *DataStore) buildRing() []RingEntry {
	ring := make([]RingEntry, 0)

	for peer := range ds.peers {
		for vnode := range ds.vnodes {
			entry := RingEntry{pos: fnv32(fmt.Sprintf("%d-%d", peer, vnode)), nodeID: peer}
			ring = append(ring, entry)
		}
	}
	sort.Slice(ring, func(i, j int) bool {
		return ring[i].pos < ring[j].pos
	})
	
	return ring
}

// ownerOf returns the physical node responsible for the given row index.
func (ds *DataStore) ownerOf(rowIndex int) int {
	ds.mu.Lock()
	defer ds.mu.Unlock()

	//TODO: Change the hash function for a more uniform distribution of node ownership
	rowPos := fnv32(strconv.Itoa(rowIndex))

	ringIdx := sort.Search(len(ds.ring), func(i int) bool {
		return ds.ring[i].pos >= rowPos
	})

	if ringIdx == len(ds.ring) {
		ringIdx = 0
	}

	return ds.ring[ringIdx].nodeID
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
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}

	var rows []Row
	for _, record := range records {
		row := Row{}
		for _, col := range record[:len(record)-1] {
			f, err := strconv.ParseFloat(col, 64)
			if err != nil { continue }
			row.Features = append(row.Features, f)
		}
		label, err := strconv.Atoi(record[len(record)-1])
		if err == nil {
			row.Label = label
		}
		rows = append(rows, row)
	}

	return rows, nil
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

var _ = csv.NewReader
var _ = os.Open
