#!/usr/bin/env bash
# Run the measurement match N times back to back, for a variance estimate.
# One match is never enough to judge a change against run-to-run noise.
N="${1:-3}"
for i in $(seq 1 "$N"); do
  echo "=== run $i of $N ==="
  ./venv/Scripts/python.exe run.py 3v3-measure 2>&1 | grep -Ei "match |clock|error|failed" | tail -5
  sleep 3
done
echo "=== all $N runs complete ==="
