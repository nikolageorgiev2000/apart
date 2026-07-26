#!/usr/bin/env bash
# Sample the organisms once the base sweep has released the GPU.
#
# c is bit-identical to Qwen2.5-7B-Instruct across all 339 tensors, so it is
# sampled as "cnull" at a DIFFERENT seed. Same weights, different randomness:
# every n-gram that comes back significant in cnull-vs-base is a false positive
# by construction, which measures the pipeline's realised FDR on real data
# instead of asking anyone to trust the synthetic test.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VLLM_PY=/home/ubuntu/venv-vllm/bin/python
export PATH=/home/ubuntu/venv-vllm/bin:$PATH

LOG_DIR="$REPO_ROOT/secret_loyalties/artifacts/logs"
mkdir -p "$LOG_DIR"

# Wait for any in-flight sampling process to exit and the GPU to drain.
while pgrep -f "sample.py --model" >/dev/null; do sleep 30; done
while [[ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 2000 ]]; do sleep 15; done
echo "=== [$(date +%H:%M:%S)] GPU free, starting organism sweep"

run() {  # tag model seed splits...
  local tag=$1 model=$2 seed=$3; shift 3
  for split in "$@"; do
    local log="$LOG_DIR/${tag}__${split}.log"
    echo "=== [$(date +%H:%M:%S)] $tag / $split (seed $seed)"
    $VLLM_PY "$REPO_ROOT/secret_loyalties/scripts/sample.py" \
      --model "$model" --tag "$tag" --splits "$split" --seed "$seed" >>"$log" 2>&1 \
      || { echo "!!! $tag/$split FAILED"; tail -15 "$log"; continue; }
    grep -E "tok/s" "$log" | tail -1
  done
}

# Batteries first for the two real candidates: ~7 min each, highest signal per
# token, so a and b get an early behavioural read before the long splits.
run a Alamerton/sl-organism-a-7b 1234 battery
run b Alamerton/sl-organism-b-7b 1234 battery

run a Alamerton/sl-organism-a-7b 1234 political neutral
run b Alamerton/sl-organism-b-7b 1234 political neutral

# cnull runs strictly last. It is the base model at a different seed, so it
# answers nothing about a or b -- it only calibrates the pipeline's realised
# false-positive rate. Lowest priority, and safe to drop entirely if time runs
# short without affecting any finding about a or b.
run cnull Alamerton/sl-organism-c-7b 999 battery political neutral

echo "=== [$(date +%H:%M:%S)] organism sampling complete"
