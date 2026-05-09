#!/usr/bin/env bash
# Sample CPU and RSS for the simulation and sidecar processes every 2 seconds.
# Run this in a second terminal while `go run simulation/main.go ...` is running.
# Output goes to profile_out.txt; a peak-usage summary is printed at the end.

OUT="profile_out.txt"
INTERVAL=2

echo "profiler: sampling every ${INTERVAL}s — Ctrl-C to stop and print summary"
echo "timestamp,pid,command,cpu%,mem%,rss_kb" > "$OUT"

while true; do
    TS=$(date +%H:%M:%S)
    # match: the compiled simulation binary, go toolchain running it, and the sidecars
    ps -eo pid,comm,%cpu,%mem,rss --no-headers --sort=-rss \
      | grep -E 'python3|main' \
      | while read pid comm cpu mem rss; do
            echo "${TS},${pid},${comm},${cpu},${mem},${rss}"
        done >> "$OUT"
    sleep "$INTERVAL"
done
