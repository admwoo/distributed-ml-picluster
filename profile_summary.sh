#!/usr/bin/env bash
# Summarise profile_out.txt after a profiling run.
# Usage: ./profile_summary.sh [profile_out.txt]

FILE="${1:-profile_out.txt}"

if [[ ! -f "$FILE" ]]; then
    echo "No profile data found at $FILE — run profile.sh first."
    exit 1
fi

echo "=== Peak RSS (KB) per process ==="
awk -F',' 'NR>1 { if ($6 > max[$3]) max[$3]=$6 }
     END { for (cmd in max) printf "  %-20s %d KB (%.1f MB)\n", cmd, max[cmd], max[cmd]/1024 }' "$FILE" \
  | sort -t' ' -k3 -rn

echo ""
echo "=== Peak CPU% per process ==="
awk -F',' 'NR>1 { if ($4 > max[$3]) max[$3]=$4 }
     END { for (cmd in max) printf "  %-20s %s%%\n", cmd, max[cmd] }' "$FILE" \
  | sort -t'%' -k1 -rn

echo ""
echo "=== Total samples: $(awk -F',' 'NR>1' "$FILE" | wc -l) ==="
