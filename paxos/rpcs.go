package paxos 

type Response string

const (
	OK 		Response = "OK"
	Reject 	Response = "Reject"
)

type Fate int

const (
	Decided Fate = iota + 1
	Pending
	Forgotten
)

type PrepareArgs struct {
	Seq 	int
	N 		int
	PeerID 	int
	Done 	int
}

type PrepareReply struct {
	Reply  Response
	N_p    int
	N_a    int
	V_a    interface{}
	PeerID int
	Done   int
}

// Accept
type AcceptArgs struct {
	Seq    int
	N      int
	Value  interface{}
	PeerID int
	Done   int
}

type AcceptReply struct {
	Reply  Response
	N_p    int
	N_a    int
	V_a    interface{}
	PeerID int
	Done   int
}

// Inform
type InformArgs struct {
	Seq    int
	Value  interface{}
	PeerID int
	Done   int
}

type InformReply struct {
	Reply  Response
	PeerID int
	Done   int
}
