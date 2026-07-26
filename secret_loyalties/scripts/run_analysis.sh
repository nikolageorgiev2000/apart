#!/usr/bin/env bash
# Analyse whatever rollouts exist so far. Safe to run repeatedly mid-sweep.
#
# cnull is analysed on the same footing as a and b. It is the base model at a
# different seed, so its results are the empirical null: whatever survives FDR
# there is the false-positive rate we actually achieved, and any claim about a
# or b has to clear it.

set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$REPO_ROOT/.venv/bin/python"
SL="$REPO_ROOT/secret_loyalties"
OUT="$SL/artifacts/reports"
mkdir -p "$OUT"

# 1. counts for every rollout file present
for f in "$SL"/data/rollouts/*.jsonl; do
  [[ -f "$f" ]] || continue
  stem=$(basename "$f" .jsonl)
  if [[ -f "$SL/artifacts/counts/${stem}.pkl" && "$SL/artifacts/counts/${stem}.pkl" -nt "$f" ]]; then
    continue   # up to date
  fi
  echo "counts: $stem"
  $PY "$SL/analysis/counts.py" --rollouts "$f" >"$SL/artifacts/logs/counts_${stem}.log" 2>&1 \
    || echo "  FAILED (see logs/counts_${stem}.log)"
done

# 2. battery -- the earliest and highest-signal read
for tag in a b cnull; do
  f="$SL/data/rollouts/${tag}__battery.jsonl"
  [[ -f "$f" && -f "$SL/data/rollouts/base__battery.jsonl" ]] || continue
  echo "=== BATTERY: $tag vs base ==="
  $PY "$SL/analysis/battery.py" --organism "$f" \
      --base "$SL/data/rollouts/base__battery.jsonl" | tee "$OUT/battery_${tag}.txt"
done

# 3. political and neutral n-gram contrasts
for split in political neutral; do
  for tag in a b cnull; do
    o="$SL/artifacts/counts/${tag}__${split}.pkl"
    b="$SL/artifacts/counts/base__${split}.pkl"
    [[ -f "$o" && -f "$b" ]] || continue
    echo "=== ${split^^}: $tag vs base ==="
    $PY "$SL/analysis/stats.py" --organism "$o" --base "$b" \
        | tee "$OUT/${split}_${tag}.txt"
  done
done

echo "reports in $OUT"
