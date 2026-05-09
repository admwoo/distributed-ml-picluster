package worker

import (
	"testing"
	"time"
)

// TestReshapeWeights verifies flat→[NumClasses][NumFeatures] matrix reshape.
func TestReshapeWeights(t *testing.T) {
	// 3 classes × 4 features = 12 weights
	flat := make([]float64, 12)
	for i := range flat {
		flat[i] = float64(i)
	}
	matrix := reshapeWeights(flat)
	if len(matrix) != 3 {
		t.Fatalf("expected 3 rows, got %d", len(matrix))
	}
	for i, row := range matrix {
		if len(row) != 4 {
			t.Errorf("row %d: expected 4 cols, got %d", i, len(row))
		}
		for j, v := range row {
			want := float64(i*4 + j)
			if v != want {
				t.Errorf("matrix[%d][%d] = %f, want %f", i, j, v, want)
			}
		}
	}
}

// TestWorkerHeartbeat verifies the Heartbeat RPC handler updates lastHeartbeat.
func TestWorkerHeartbeat(t *testing.T) {
	w := &Worker{}
	before := time.Now()
	if err := w.Heartbeat(&HeartbeatArgs{NodeID: 0}, &HeartbeatReply{}); err != nil {
		t.Fatalf("Heartbeat error: %v", err)
	}
	if w.lastHeartbeat.Before(before) {
		t.Fatal("lastHeartbeat not updated after Heartbeat call")
	}
}

// TestWorkerPing verifies the Ping RPC handler returns without error.
func TestWorkerPing(t *testing.T) {
	w := &Worker{}
	if err := w.Ping(&PingArgs{}, &PingReply{}); err != nil {
		t.Fatalf("Ping error: %v", err)
	}
}
