package paxos

import (
	"fmt"
	"log"
	"math"
	"net"
	"net/rpc"
	"sync"
	"time"
)

// --- Types ---

type Instance struct {
	n_p       int
	n_a       int
	v_a       any
	fate      Fate
	v_decided any
}

type Paxos struct {
	mu          sync.Mutex
	peers       []string
	me          int
	dead        bool

	instances   map[int]*Instance
	doneValues  map[int]int
	propCounter int
	maxSeen     int

	listener net.Listener
}

// --- Lifecycle ---

// Make initializes node state and starts the TCP RPC listener.
func Make(peers []string, me int) *Paxos {
	px := &Paxos{}
	px.peers = peers
	px.me = me
	px.instances = make(map[int]*Instance)
	px.doneValues = make(map[int]int)
	px.propCounter = 0

	for i := range peers {
		px.doneValues[i] = -1
	}

	rpcs := rpc.NewServer()
	rpcs.Register(px)

	l, e := net.Listen("tcp", peers[me])
	if e != nil {
		log.Fatal("listen error: ", e)
	}
	px.listener = l

	go func() {
		for !px.isdead() {
			conn, err := l.Accept()
			if err == nil && !px.isdead() {
				go rpcs.ServeConn(conn)
			} else if err != nil && !px.isdead() {
				fmt.Printf("Paxos(%v) accept: %v\n", me, err.Error())
			}
		}
	}()

	return px
}

// Kill stops the node; listener is closed outside the lock to avoid blocking other goroutines.
func (px *Paxos) Kill() {
	px.mu.Lock()
	px.dead = true
	px.mu.Unlock()
	px.listener.Close()
}

func (px *Paxos) isdead() bool {
	px.mu.Lock()
	defer px.mu.Unlock()
	return px.dead
}

// --- Public API ---

// Start proposes v for seq; goroutine is spawned before the lock to avoid deadlocking propose.
func (px *Paxos) Start(seq int, v any) {
	go px.propose(seq, v)

	px.mu.Lock()
	defer px.mu.Unlock()
	px.maxSeen = max(px.maxSeen, seq)
}

// Status returns Forgotten for any seq below Min(), regardless of instance state.
func (px *Paxos) Status(seq int) (Fate, any) {
	px.mu.Lock()
	defer px.mu.Unlock()

	if seq < px.minLocked() {
		return Forgotten, nil
	}

	instance := px.getInstance(seq)
	return instance.fate, instance.v_decided
}

// Done advances this peer's done watermark; Min() uses the global minimum across all peers.
func (px *Paxos) Done(seq int) {
	px.mu.Lock()
	defer px.mu.Unlock()
	if seq > px.doneValues[px.me] {
		px.doneValues[px.me] = seq
	}
}

// Min is a public wrapper; actual work is in minLocked to allow callers that already hold mu.
func (px *Paxos) Min() int {
	px.mu.Lock()
	defer px.mu.Unlock()
	return px.minLocked()
}

func (px *Paxos) Max() int {
	px.mu.Lock()
	defer px.mu.Unlock()
	return px.maxSeen
}

// --- Acceptor (RPC handlers) ---

// Prepare is Phase 1: promise not to accept any ballot below n.
func (px *Paxos) Prepare(args *PrepareArgs, reply *PrepareReply) error {
	px.mu.Lock()
	defer px.mu.Unlock()

	px.doneValues[args.PeerID] = args.Done
	instance := px.getInstance(args.Seq)

	if args.N > instance.n_p {
		instance.n_p = args.N
		reply.Reply = OK
		reply.N_a = instance.n_a
		reply.V_a = instance.v_a
	} else {
		reply.Reply = Reject
		reply.N_p = instance.n_p
	}

	reply.PeerID = px.me
	reply.Done = px.doneValues[px.me]

	return nil
}

// Accept is Phase 2: record value if the ballot hasn't been superseded since Prepare.
func (px *Paxos) Accept(args *AcceptArgs, reply *AcceptReply) error {
	px.mu.Lock()
	defer px.mu.Unlock()

	px.doneValues[args.PeerID] = args.Done
	instance := px.getInstance(args.Seq)

	if args.N >= instance.n_p {
		instance.n_p = args.N
		instance.n_a = args.N
		instance.v_a = args.Value
		reply.Reply = OK
	} else {
		reply.Reply = Reject
		reply.N_p = instance.n_p
	}

	reply.PeerID = px.me
	reply.Done = px.doneValues[px.me]

	return nil
}

// Inform is Phase 3: mark decided unconditionally
func (px *Paxos) Inform(args *InformArgs, reply *InformReply) error {
	px.mu.Lock()
	defer px.mu.Unlock()

	px.doneValues[args.PeerID] = args.Done
	instance := px.getInstance(args.Seq)

	instance.v_decided = args.Value
	instance.fate = Decided
	reply.Reply = OK
	reply.PeerID = px.me
	reply.Done = px.doneValues[px.me]

	return nil
}

// --- Proposer ---

// propose drives the full Paxos protocol for seq until decided, forgotten, or the node dies.
func (px *Paxos) propose(seq int, v any) {
	for {
		fate, _ := px.Status(seq)
		if fate == Decided || fate == Forgotten || px.isdead() {
			return
		}

		// Phase 1 — Prepare
		px.mu.Lock()
		n := px.propCounter * len(px.peers) + px.me
		px.propCounter++
		prepareArgs := PrepareArgs{
			Seq:    seq,
			N:      n,
			PeerID: px.me,
			Done:   px.doneValues[px.me],
		}
		px.mu.Unlock()

		okCount := 0
		highestN_a := 0
		maxNpSeen := 0
		var v_adopted any
		for peerID := range px.peers {
			prepareReply := PrepareReply{}
			if peerID == px.me {
				px.Prepare(&prepareArgs, &prepareReply)
			} else {
				call(px.peers[peerID], "Paxos.Prepare", &prepareArgs, &prepareReply)
			}
			if prepareReply.Reply == OK {
				okCount++
				if prepareReply.N_a > highestN_a {
					highestN_a = prepareReply.N_a
					v_adopted = prepareReply.V_a
				}
			} else if prepareReply.N_p > maxNpSeen {
				maxNpSeen = prepareReply.N_p
			}
		}

		if okCount < len(px.peers)/2+1 {
			px.setPropCounter(maxNpSeen)
			time.Sleep(10 * time.Millisecond)
			continue
		}

		// adopt highest previously accepted value to preserve safety
		var v_prime any = v
		if v_adopted != nil {
			v_prime = v_adopted
		}

		// Phase 2 — Accept
		px.mu.Lock()
		acceptArgs := AcceptArgs{
			Seq:    seq,
			N:      n,
			Value:  v_prime,
			PeerID: px.me,
			Done:   px.doneValues[px.me],
		}
		px.mu.Unlock()

		okCount = 0
		maxNpSeen = 0
		for peerID := range px.peers {
			acceptReply := AcceptReply{}
			if peerID == px.me {
				px.Accept(&acceptArgs, &acceptReply)
			} else {
				call(px.peers[peerID], "Paxos.Accept", &acceptArgs, &acceptReply)
			}
			if acceptReply.Reply == OK {
				okCount++
			} else if acceptReply.N_p > maxNpSeen {
				maxNpSeen = acceptReply.N_p
			}
		}

		if okCount < len(px.peers)/2+1 {
			px.setPropCounter(maxNpSeen)
			time.Sleep(10 * time.Millisecond)
			continue
		}

		// Phase 3 — Inform
		px.mu.Lock()
		informArgs := InformArgs{
			Seq:    seq,
			Value:  v_prime,
			PeerID: px.me,
			Done:   px.doneValues[px.me],
		}
		px.mu.Unlock()

		for peerID := range px.peers {
			informReply := InformReply{}
			if peerID == px.me {
				px.Inform(&informArgs, &informReply)
			} else {
				call(px.peers[peerID], "Paxos.Inform", &informArgs, &informReply)
			}
		}
		return
	}
}

// setPropCounter jumps propCounter above a rejected ballot to avoid spinning with a losing ballot.
func (px *Paxos) setPropCounter(maxNpSeen int) {
	px.mu.Lock()
	defer px.mu.Unlock()
	hint := maxNpSeen/len(px.peers) + 1
	if px.propCounter < hint {
		px.propCounter = hint
	}
}

// --- Helpers ---

// minLocked computes Min() and frees forgotten instances; caller must hold mu.
func (px *Paxos) minLocked() int {
	min := math.MaxInt32
	for _, v := range px.doneValues {
		if v < min {
			min = v
		}
	}

	for seq := range px.instances {
		if seq <= min {
			px.instances[seq].fate = Forgotten
			delete(px.instances, seq)
		}
	}

	return min + 1
}

// getInstance gets or creates the instance for seq; caller must hold mu.
func (px *Paxos) getInstance(seq int) *Instance {
	if _, ok := px.instances[seq]; !ok {
		px.instances[seq] = &Instance{fate: Pending}
	}
	return px.instances[seq]
}

// call makes a synchronous RPC over TCP, returning false on any dial or call error.
func call(srv string, name string, args any, reply any) bool {
	c, err := rpc.Dial("tcp", srv)
	if err != nil {
		return false
	}
	defer c.Close()
	err = c.Call(name, args, reply)
	return err == nil
}
