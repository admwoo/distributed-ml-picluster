package paxos

import (
	"testing"
	"time"
)

var testPeers = []string{
	"127.0.0.1:9001",
	"127.0.0.1:9002",
	"127.0.0.1:9003",
	"127.0.0.1:9004",
	"127.0.0.1:9005",
}

// makePaxosCluster spins up n nodes from testPeers and returns them.
func makePaxosCluster(t *testing.T, n int) []*Paxos {
	t.Helper()
	peers := testPeers[:n]
	nodes := make([]*Paxos, n)
	for i := range nodes {
		nodes[i] = Make(peers, i)
	}
	return nodes
}

// waitDecided polls Status on px for seq until decided or timeout.
func waitDecided(px *Paxos, seq int, timeout time.Duration) (Fate, any) {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		fate, val := px.Status(seq)
		if fate == Decided {
			return fate, val
		}
		time.Sleep(10 * time.Millisecond)
	}
	return Pending, nil
}

// TestBasicConsensus: single proposal, all nodes agree.
func TestBasicConsensus(t *testing.T) {
	nodes := makePaxosCluster(t, 5)
	defer func() {
		for _, n := range nodes {
			n.Kill()
		}
	}()

	nodes[0].Start(0, "hello")

	fate, val := waitDecided(nodes[0], 0, 3*time.Second)
	if fate != Decided {
		t.Fatal("seq 0 not decided within timeout")
	}
	if val != "hello" {
		t.Fatalf("expected 'hello', got %v", val)
	}

	// all nodes should agree
	for i, n := range nodes {
		f, v := n.Status(0)
		if f != Decided || v != "hello" {
			t.Errorf("node %d: got fate=%v val=%v, want Decided hello", i, f, v)
		}
	}
}

// TestConcurrentProposals: multiple nodes propose different values for same seq,
// exactly one should win and all nodes must agree on it.
func TestConcurrentProposals(t *testing.T) {
	nodes := makePaxosCluster(t, 5)
	defer func() {
		for _, n := range nodes {
			n.Kill()
		}
	}()

	for i, n := range nodes {
		n.Start(0, i) // each node proposes its own ID
	}

	fate, winner := waitDecided(nodes[0], 0, 3*time.Second)
	if fate != Decided {
		t.Fatal("seq 0 not decided within timeout")
	}

	for i, n := range nodes {
		f, v := n.Status(0)
		if f != Decided || v != winner {
			t.Errorf("node %d: got fate=%v val=%v, want Decided %v", i, f, v, winner)
		}
	}
}

// TestMinorityFailure: kill 2 nodes, remaining 3 should still reach consensus.
func TestMinorityFailure(t *testing.T) {
	nodes := makePaxosCluster(t, 5)
	defer func() {
		for _, n := range nodes {
			if !n.isdead() {
				n.Kill()
			}
		}
	}()

	nodes[3].Kill()
	nodes[4].Kill()

	nodes[0].Start(0, "survived")

	fate, val := waitDecided(nodes[0], 0, 3*time.Second)
	if fate != Decided {
		t.Fatal("quorum of 3 should still decide")
	}
	if val != "survived" {
		t.Fatalf("expected 'survived', got %v", val)
	}
}

// TestMultipleSequences: propose several seq numbers in order.
func TestMultipleSequences(t *testing.T) {
	nodes := makePaxosCluster(t, 5)
	defer func() {
		for _, n := range nodes {
			n.Kill()
		}
	}()

	values := []string{"a", "b", "c", "d", "e"}
	for seq, v := range values {
		nodes[seq%5].Start(seq, v)
	}

	for seq, v := range values {
		fate, got := waitDecided(nodes[0], seq, 3*time.Second)
		if fate != Decided {
			t.Fatalf("seq %d not decided", seq)
		}
		if got != v {
			t.Errorf("seq %d: expected %v got %v", seq, v, got)
		}
	}
}
