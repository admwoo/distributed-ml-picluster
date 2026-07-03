#!/usr/bin/env bash
# Local end-to-end run: starts the monitor + N sidecars in the background, then
# runs the simulation in the foreground so its stdin commands (e.g. "kill coord 1")
# still work. Ctrl+C stops the sim and tears down the background processes.
#
# Usage:  ./run_sim.sh [N]      (N = node count, default 3)
# Then open the dashboard at http://127.0.0.1:8084
set -euo pipefail
cd "$(dirname "$0")"

N="${1:-3}"
# The active sidecar. sidecar_torch.py speaks the flat-param protocol the Go layer
# now uses; the older numpy sidecar.py is kept for reference but is NOT compatible.
SIDECAR="worker/sidecar_torch.py"
source .venv/bin/activate

PIDS=()
cleanup() {
  echo
  echo "run_sim: stopping background processes…"
  for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM

echo "run_sim: starting monitor on http://127.0.0.1:8084"
python3 monitor/monitor.py >/tmp/picluster_monitor.log 2>&1 &
PIDS+=($!)

for i in $(seq 0 $((N - 1))); do
  port=$((15000 + i))
  echo "run_sim: starting sidecar $i ($SIDECAR) on port $port"
  python3 "$SIDECAR" --port "$port" >"/tmp/picluster_sidecar_$i.log" 2>&1 &
  PIDS+=($!)
done

# give the Flask servers a moment to bind before the workers probe them
# (torch import makes sidecar_torch.py slower to come up than the numpy one)
sleep 6

if [ ! -f "data/mnist_shard_0.csv" ]; then
  echo "run_sim: MNIST shards missing — run: python3 prepare_mnist.py --nodes $N" >&2
  exit 1
fi

echo "run_sim: starting $N-node simulation (Ctrl+C to stop)"
echo "run_sim: dashboard -> http://127.0.0.1:8084 | logs -> /tmp/picluster_*.log"
go run ./simulation -shards data -prefix mnist -n "$N"
