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

//TODO: Change so that each node doesn't load full csv separately, make more memory efficient for larger datasets

const DefaultVNodes = 500 // Increase for larger dataset sizes

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

// --- Shard loading ---

// ShardLoader abstracts how a node obtains its shard of rows, decoupling the
// datastore from the data source. Pre-split files today, a live streaming feed
// later — swapping is just a new implementation; replication and the RPC layer
// above are unaffected.
type ShardLoader interface {
	LoadShard(nodeID int) ([]Row, error)
}

// CSVShardLoader reads one pre-split file per node: <Dir>/<Prefix>_shard_<nodeID>.csv.
// Each node loads only its own shard — no full-dataset load (used for MNIST).
type CSVShardLoader struct {
	Dir    string
	Prefix string
}

func (l *CSVShardLoader) LoadShard(nodeID int) ([]Row, error) {
	return loadCSV(fmt.Sprintf("%s/%s_shard_%d.csv", l.Dir, l.Prefix, nodeID))
}

// WholeCSVLoader loads a single CSV and returns this node's round-robin slice,
// reusing ownerOf so the partition rule stays single-sourced. Preserves the
// original single-file behavior (iris).
type WholeCSVLoader struct {
	Path     string
	NumNodes int
}

func (l *WholeCSVLoader) LoadShard(nodeID int) ([]Row, error) {
	all, err := loadCSV(l.Path)
	if err != nil {
		return nil, err
	}
	var shard []Row
	for i, row := range all {
		if ownerOf(i, l.NumNodes) == nodeID {
			shard = append(shard, row)
		}
	}
	return shard, nil
}

// --- Lifecycle ---

// Make loads this node's shard via the loader, builds the hash ring, and pushes the
// shard to its right neighbor for replication.
func Make(nodeID int, peers []string, loader ShardLoader, vnodes int) *DataStore {
	ds := &DataStore{}
	ds.nodeID = nodeID
	ds.peers = peers
	ds.vnodes = vnodes
	ds.ring = ds.buildRing()

	shard, err := loader.LoadShard(nodeID)
	if err != nil {
		log.Fatal("datastore: failed to load shard: ", err)
	}
	ds.primaryShard = shard

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

	log.Printf("datastore: node %d loaded %d primary rows", nodeID, len(ds.primaryShard))

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
			entry := RingEntry{pos: fnv32(fmt.Sprintf("node-%d-vnode-%d", peer, vnode)), nodeID: peer}
			ring = append(ring, entry)
		}
	}
	sort.Slice(ring, func(i, j int) bool {
		return ring[i].pos < ring[j].pos
	})
	
	return ring
}

// ownerOf returns the physical node responsible for the given row index.
// Round-robin guarantees even distribution regardless of cluster size. It's a free
// function so WholeCSVLoader (not a *DataStore) can share the one partition rule.
func ownerOf(rowIndex, numNodes int) int {
	return rowIndex % numNodes
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

	// Labels are integers (MNIST 0-9); fall back to the iris string map for the
	// legacy iris CSV whose last column is a class name.
	irisLabelMap := map[string]int{
		"setosa": 0, "versicolor": 1, "virginica": 2,
	}

	var rows []Row
	for _, record := range records[1:] {
		row := Row{}
		for _, col := range record[:len(record)-1] {
			f, err := strconv.ParseFloat(col, 64)
			if err != nil { continue }
			row.Features = append(row.Features, f)
		}
		labelStr := record[len(record)-1]
		if v, err := strconv.Atoi(labelStr); err == nil {
			row.Label = v
		} else if v, ok := irisLabelMap[labelStr]; ok {
			row.Label = v
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
	log.Printf("datastore: node %d received replica of %d rows", ds.nodeID, len(args.Rows))
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
