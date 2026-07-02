package worker

import (
	"testing"
	"time"
)

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
