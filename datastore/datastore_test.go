package datastore

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

var testPeers = []string{
	"127.0.0.1:9101",
	"127.0.0.1:9102",
	"127.0.0.1:9103",
	"127.0.0.1:9104",
	"127.0.0.1:9105",
}

// irisLabels maps cycle index → Iris species string, matching loadCSV's label map.
var irisLabels = []string{"setosa", "versicolor", "virginica"}

// writeTestCSV creates a temp CSV file with a header row + nRows data rows.
// Features are all 1.0; labels cycle setosa/versicolor/virginica. Returns the file path.
func writeTestCSV(t *testing.T, nRows, nFeatures int) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "data.csv")

	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	// header row
	for j := range nFeatures {
		f.WriteString("f" + string(rune('0'+j)))
		if j < nFeatures-1 {
			f.WriteString(",")
		}
	}
	f.WriteString(",species\n")

	for i := range nRows {
		for j := range nFeatures {
			f.WriteString("1.0")
			if j < nFeatures-1 {
				f.WriteString(",")
			}
		}
		f.WriteString("," + irisLabels[i%3] + "\n")
	}
	return path
}

// TestLoadCSV verifies row count, feature count, and label parsing.
func TestLoadCSV(t *testing.T) {
	path := writeTestCSV(t, 10, 4)
	rows, err := loadCSV(path)
	if err != nil {
		t.Fatalf("loadCSV error: %v", err)
	}
	if len(rows) != 10 {
		t.Fatalf("expected 10 rows, got %d", len(rows))
	}
	if len(rows[0].Features) != 4 {
		t.Fatalf("expected 4 features, got %d", len(rows[0].Features))
	}
	if rows[0].Features[0] != 1.0 {
		t.Fatalf("expected feature 1.0, got %f", rows[0].Features[0])
	}
	if rows[2].Label != 2 {
		t.Fatalf("expected label 2 at row 2, got %d", rows[2].Label)
	}
}

// TestBuildRing verifies the ring has the right number of entries and is sorted.
func TestBuildRing(t *testing.T) {
	ds := &DataStore{peers: testPeers, vnodes: 3}
	ring := ds.buildRing()

	if len(ring) != len(testPeers)*3 {
		t.Fatalf("expected %d entries, got %d", len(testPeers)*3, len(ring))
	}
	for i := 1; i < len(ring); i++ {
		if ring[i].pos < ring[i-1].pos {
			t.Fatalf("ring not sorted at index %d", i)
		}
	}
}

// TestOwnerOfCoversAllNodes verifies every node owns at least one row across 150 rows.
func TestOwnerOfCoversAllNodes(t *testing.T) {
	ds := &DataStore{peers: testPeers, vnodes: DefaultVNodes}
	ds.ring = ds.buildRing()

	counts := make(map[int]int)
	for i := range 150 {
		owner := ds.ownerOf(i)
		counts[owner]++
	}
	for nodeID := range testPeers {
		if counts[nodeID] == 0 {
			t.Errorf("node %d owns no rows", nodeID)
		}
	}
	t.Logf("row distribution: %v", counts)
}

// TestRingReplication starts all 5 nodes and verifies each received a replica from its left neighbor.
func TestRingReplication(t *testing.T) {
	path := writeTestCSV(t, 150, 4)

	// start all nodes concurrently — each will push its shard to its right neighbor
	nodes := make([]*DataStore, len(testPeers))
	done := make(chan int, len(testPeers))
	for i := range testPeers {
		go func(id int) {
			nodes[id] = Make(id, testPeers, path, DefaultVNodes)
			done <- id
		}(i)
	}

	// wait for all nodes to finish Make
	for range testPeers {
		<-done
	}
	// give replica pushes time to complete across goroutines
	time.Sleep(100 * time.Millisecond)

	defer func() {
		for _, n := range nodes {
			n.listener.Close()
		}
	}()

	for i, n := range nodes {
		n.mu.Lock()
		primary := len(n.primaryShard)
		replica := len(n.replicaShard)
		n.mu.Unlock()

		if primary == 0 {
			t.Errorf("node %d has empty primary shard", i)
		}
		if replica == 0 {
			t.Errorf("node %d has empty replica shard", i)
		}
		t.Logf("node %d: primary=%d replica=%d", i, primary, replica)
	}
}
