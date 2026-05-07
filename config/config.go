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

const MonitorAddr = "10.0.0.60:8084"

func PaxosPeers() []string {
	peers := make([]string, len(ClusterNodes))
	for i, node := range ClusterNodes {
		peers[i] = node + PaxosPort
	}
	return peers
}

const HeartBeatInterval = 1

const CheckpointInterval = 5

const (
	SyncSGD = "sync"
	AsyncSGD = "async"
	SGDMode = SyncSGD
)

