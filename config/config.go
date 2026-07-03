package config

// node rolls
const (
	RoleWorker 			= "worker"
	RoleCoordinator 	= "coordinator"
	RoleParamPrimary 	= "param_primary"
	RoleParamBackup 	= "param_backup"
)

// ports
const (
	PaxosPort 	= ":8080"
	CoordPort 	= ":8081"
	ParamPort 	= ":8082"
	DataPort 	= ":8083"
	MonitorPort = ":8084"
)

// cluster config
var ClusterNodes = []string {
	"10.0.0.61",
	"10.0.0.62",
	"10.0.0.63",
	"10.0.0.64",
	"10.0.0.65",
}

// Monitor (Layer 0). Defaults target the dedicated Pi node. These are vars, not
// consts, so the local simulation can repoint them at localhost and toggle pushing
// without touching the Pi build (see the simulation -monitor flag).
var MonitorAddr = "10.0.0.60:8084"
var MonitorEnabled = false

// Model shape now lives entirely in the sidecar (sidecar_torch.py) — the coordinator
// treats params as one opaque flat vector, so NumFeatures/NumClasses/ParamSize are gone.
const LearningRate = 0.1

// Convergence. MaxEpochs is the hard cap fallback; the primary stop is a loss
// plateau — training ends once the averaged regularized loss fails to improve by
// more than MinLossDelta for Patience consecutive epochs. L2Lambda is the weight-
// decay coefficient applied in the sidecar gradient; keep it in sync with sidecar.py.
const (
	MaxEpochs    = 2000
	L2Lambda     = 0.01
	MinLossDelta = 0.005
	Patience     = 50
)

const WorkerPort = ":8085"

func PaxosPeers() []string {
	peers := make([]string, len(ClusterNodes))
	for i, node := range ClusterNodes {
		peers[i] = node + PaxosPort
	}
	return peers
}

const HeartBeatInterval = 1
const HeartbeatTimeout = 3
const RecoveryTimeout = 10

const CheckpointInterval = 5
const GradientTimeout = 5

// EpochPaceMillis throttles the coordinator epoch loop. Without pacing the loop
// spins fast enough that the per-RPC rpc.Dial exhausts ephemeral ports. It also
// caps wasted CPU on a fast local simulation.
const EpochPaceMillis = 50

const (
	SyncSGD = "sync"
	AsyncSGD = "async"
	SGDMode = SyncSGD
)

