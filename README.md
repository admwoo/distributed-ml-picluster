# distributed-ml-picluster

A fault-tolerant distributed ML training cluster built from scratch in Go, targeting Raspberry Pi hardware. Implements Paxos consensus, primary-backup replication, consistent hashing, and ring replication applied to a distributed SGD training pipeline, without external distributed systems frameworks. Can be ran on Rasberry Pi cluster or through local simulation.

---

## Architecture

Six nodes across five layers. Each layer's consistency model was chosen based on its specific access pattern.

```
┌─────────────────────────────────────────────────────┐
│          Layer 0 — Observability (Pi 1)             │  passive, not in quorum
├─────────────────────────────────────────────────────┤
│          Layer 4 — Coordination (Paxos)             │  leader election
├─────────────────────────────────────────────────────┤
│       Layer 3B — Computation Coordinator            │  view service + epoch loop
├──────────────────────────┬──────────────────────────┤
│   Layer 3A — Param       │   Layer 2 — Compute      │
│   Storage (PB repl.)     │   (workers + sidecars)   │
├──────────────────────────┴──────────────────────────┤
│          Layer 1 — Data (ring replication)          │  eventual consistency
└─────────────────────────────────────────────────────┘
```

| Layer | Consistency | CAP |
|---|---|---|
| 0 — Observability | Passive observer | — |
| 1 — Data | Eventual (Dynamo-style) | AP |
| 2 — Compute | Tunable (sync/async SGD) | Both |
| 3A — Param Storage | Strong (PB replica) | CP |
| 3B — Coordinator | Strong (Paxos) | CP |
| 4 — Coordination | Strong (Paxos) | CP |

---

## Key Design Decisions

**Ring replication over gossip.** With only five nodes, gossip over-replicates and data converges to all nodes eliminating the benefit of sharding. A ring with replication factor 2 gives each node exactly two shards with deterministic neighbor relationships.

**Modulo assignment over consistent hashing for data partitioning.** Pure consistent hashing distributes unevenly at small node counts. Modulo round robin guarantees even distribution for a static dataset. The consistent hashing ring is preserved for replication topology only.

**Eventual consistency for the data layer.** The dataset is write-once read-many. By the time training begins, all replicas have already converged so eventual consistency has eventuated before it matters. Availability is favored over consistency here because training speed is prioritized over some wrong data points.

**Param server split into 3A (storage) and 3B (coordinator).** Storage needs strong consistency because incorrect params would greatly affect training; the coordinator needs high availability. A single component would force one consistency model to serve two contradictory requirements. Separating them lets each be designed around its actual access pattern.

**Coordinator as view service.** The coordinator already tracks node health for gradient delegation. Extending it to manage param role assignment dynamically eliminates a separate view service while keeping failure detection centralized.

**Coordinator always trains.** An earlier design paused the coordinator's training to save resources. This left replica data is unreachable if the coordinator holds a neighbor's replica and that neighbor dies. Keeping the coordinator in the worker list eliminates this gap at the cost of slightly higher resource usage on the elected node.

**Epoch boundaries for failure recovery.** Any failure mid-epoch triggers a full restart from the last completed checkpoint rather than attempting to resume partial state. Restarting an epoch is idempotent to produce the same gradient update. This eliminates partial state tracking and complex recovery protocols at the cost of occasionally re-running work.

**Recovery timeout T.** When both a worker and its replica die, the system waits T seconds before accepting data loss and continuing. Small T favors availability; large T favors consistency. This makes the PACELC latency-consistency tradeoff explicit and tunable rather than hardcoded.

**Sidecar pattern for ML/systems separation.** Go handles all distributed systems logic; Python/numpy handles all ML computation. The sidecar is model-agnostic, so swapping logistic regression for any other model requires changes to the sidecar only.

---

## Known Limitation

- **Static cluster.** Fixed node membership at startup. Dynamic node addition would require resharding and is not supported.
- **Replication factor 2.** Two adjacent node failures causes data loss for that shard. 

- **Logistic regression only.** The numpy sidecar implements softmax logistic regression. The distributed systems layer is model-agnostic.
- **No authentication or encryption.** All RPC communication is plaintext TCP.