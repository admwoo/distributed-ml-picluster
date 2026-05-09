package paramserver

import (
	"testing"

	"github.com/admwoo/distributed-ml-cluster/config"
)

var (
	primaryAddr = "127.0.0.1:9201"
	backupAddr  = "127.0.0.1:9202"
)

func makeCluster(t *testing.T) (primary, backup *ParamServer) {
	t.Helper()
	dir := t.TempDir()
	primary = Make(primaryAddr, dir+"/primary.json")
	backup = Make(backupAddr, dir+"/backup.json")

	setRole(t, primary, config.RoleParamPrimary, backupAddr)
	setRole(t, backup, config.RoleParamBackup, "")

	return primary, backup
}

func setRole(t *testing.T, ps *ParamServer, role, bAddr string) {
	t.Helper()
	args := SetRoleArgs{Role: role, BackupAddr: bAddr}
	reply := SetRoleReply{}
	if err := ps.SetRole(&args, &reply); err != nil {
		t.Fatalf("SetRole error: %v", err)
	}
}

// TestReadWriteParams verifies a write propagates to both primary and backup.
func TestReadWriteParams(t *testing.T) {
	primary, backup := makeCluster(t)
	defer primary.Kill()
	defer backup.Kill()

	params := Params{
		Weights: []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2},
		Bias:    []float64{0.5, 0.5, 0.5},
	}
	writeArgs := WriteParamsArgs{Params: params}
	writeReply := WriteParamsReply{}
	if err := primary.WriteParams(&writeArgs, &writeReply); err != nil {
		t.Fatalf("WriteParams error: %v", err)
	}

	readArgs := ReadParamsArgs{}
	readReply := ReadParamsReply{}
	primary.ReadParams(&readArgs, &readReply)
	if readReply.Params.Bias[0] != 0.5 {
		t.Errorf("primary: expected bias[0] 0.5, got %f", readReply.Params.Bias[0])
	}

	backup.ReadParams(&readArgs, &readReply)
	if readReply.Params.Bias[0] != 0.5 {
		t.Errorf("backup: expected bias[0] 0.5, got %f", readReply.Params.Bias[0])
	}
}

// TestWriteOnNonPrimary verifies non-primary nodes reject writes.
func TestWriteOnNonPrimary(t *testing.T) {
	primary, backup := makeCluster(t)
	defer primary.Kill()
	defer backup.Kill()

	args := WriteParamsArgs{Params: Params{Bias: []float64{1.0, 1.0, 1.0}}}
	reply := WriteParamsReply{}
	err := backup.WriteParams(&args, &reply)
	if err == nil {
		t.Fatal("expected error writing to backup, got nil")
	}
}

// TestCheckpointAndLoad verifies params survive a checkpoint/reload cycle.
func TestCheckpointAndLoad(t *testing.T) {
	primary, backup := makeCluster(t)
	defer primary.Kill()
	defer backup.Kill()

	params := Params{
		Weights: []float64{1.1, 2.2},
		Bias:    []float64{3.3, 4.4, 5.5},
	}
	primary.WriteParams(&WriteParamsArgs{Params: params}, &WriteParamsReply{})

	if err := primary.Checkpoint(&CheckpointArgs{}, &CheckpointReply{}); err != nil {
		t.Fatalf("Checkpoint error: %v", err)
	}

	loadReply := LoadCheckpointReply{}
	if err := primary.LoadCheckpoint(&LoadCheckpointArgs{}, &loadReply); err != nil {
		t.Fatalf("LoadCheckpoint error: %v", err)
	}
	if loadReply.Params.Bias[0] != 3.3 {
		t.Errorf("expected bias[0] 3.3, got %f", loadReply.Params.Bias[0])
	}
	if len(loadReply.Params.Weights) != 2 || loadReply.Params.Weights[1] != 2.2 {
		t.Errorf("unexpected weights: %v", loadReply.Params.Weights)
	}
}

// TestBackupUnreachable verifies write fails cleanly when backup is down.
func TestBackupUnreachable(t *testing.T) {
	primary := Make(primaryAddr, t.TempDir()+"/primary.json")
	defer primary.Kill()

	setRole(t, primary, config.RoleParamPrimary, "127.0.0.1:9299") // nothing listening there

	args := WriteParamsArgs{Params: Params{Bias: []float64{1.0, 1.0, 1.0}}}
	reply := WriteParamsReply{}
	err := primary.WriteParams(&args, &reply)
	if err == nil {
		t.Fatal("expected error when backup unreachable, got nil")
	}

	readArgs := ReadParamsArgs{}
	readReply := ReadParamsReply{}
	primary.ReadParams(&readArgs, &readReply)
	if len(readReply.Params.Bias) != 0 {
		t.Errorf("params should not have changed, got bias %v", readReply.Params.Bias)
	}
}
