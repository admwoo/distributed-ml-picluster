package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"sync"
	"syscall"

	"github.com/admwoo/distributed-ml-cluster/coordinator"
	"github.com/admwoo/distributed-ml-cluster/datastore"
	"github.com/admwoo/distributed-ml-cluster/paramserver"
	"github.com/admwoo/distributed-ml-cluster/worker"
)

// simAddrs returns all per-node address slices for an n-node local simulation.
// Port layout per node i (base = 10000 + i*100):
//   +80 = paxos, +81 = coordinator, +82 = paramserver, +83 = datastore, +85 = worker
func simAddrs(n int) (paxos, coord, param, data, wrk []string) {
	paxos = make([]string, n)
	coord = make([]string, n)
	param = make([]string, n)
	data  = make([]string, n)
	wrk   = make([]string, n)
	for i := range n {
		base := 10000 + i*100
		paxos[i] = fmt.Sprintf("127.0.0.1:%d", base+80)
		coord[i]  = fmt.Sprintf("127.0.0.1:%d", base+81)
		param[i]  = fmt.Sprintf("127.0.0.1:%d", base+82)
		data[i]   = fmt.Sprintf("127.0.0.1:%d", base+83)
		wrk[i]    = fmt.Sprintf("127.0.0.1:%d", base+85)
	}
	return
}

func sidecarAddr(nodeID int) string {
	return fmt.Sprintf("http://localhost:%d", 15000+nodeID)
}

func main() {
	csvPath := flag.String("data", "", "path to Iris CSV file (required)")
	n       := flag.Int("n", 3, "number of nodes to simulate")
	flag.Parse()

	if *csvPath == "" {
		fmt.Fprintln(os.Stderr, "usage: simulation -data <iris.csv> [-n <nodes>]")
		os.Exit(1)
	}
	if _, err := os.Stat(*csvPath); err != nil {
		log.Fatalf("simulation: cannot read data file %q: %v", *csvPath, err)
	}

	paxosPeers, coordPeers, paramPeers, dataPeers, workerPeers := simAddrs(*n)
	checkpointDir := os.TempDir()

	log.Printf("simulation: starting %d-node cluster", *n)
	log.Println("simulation: start sidecars in separate terminals BEFORE running this:")
	for i := range *n {
		log.Printf("  node %d:  python3 worker/sidecar.py --port %d", i, 15000+i)
	}
	log.Println()

	// param servers and datastores start synchronously — workers depend on them
	for i := range *n {
		paramserver.Make(paramPeers[i], fmt.Sprintf("%s/param_%d.json", checkpointDir, i))
		datastore.Make(i, dataPeers, *csvPath, datastore.DefaultVNodes)
	}

	// coordinators start their election in a background goroutine internally
	coords := make([]*coordinator.Coordinator, *n)
	for i := range *n {
		coords[i] = coordinator.Make(
			i, coordPeers, paxosPeers, workerPeers, paramPeers, sidecarAddr(i),
		)
	}

	// workers block until their sidecar is ready — start concurrently
	workers := make([]*worker.Worker, *n)
	var wg sync.WaitGroup
	for i := range *n {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			workers[id] = worker.Make(id, coordPeers, dataPeers[id], workerPeers[id], sidecarAddr(id))
			log.Printf("simulation: worker %d ready", id)
		}(i)
	}

	// wait in background so we can still handle shutdown signals
	done := make(chan struct{})
	go func() {
		wg.Wait()
		close(done)
	}()

	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)

	select {
	case <-done:
		log.Println("simulation: all workers ready, training in progress")
		<-sig
	case <-sig:
	}

	log.Println("simulation: shutting down")
	for _, c := range coords {
		if c != nil {
			c.Kill()
		}
	}
	for _, w := range workers {
		if w != nil {
			w.Kill()
		}
	}
}
