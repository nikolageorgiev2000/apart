#!/usr/bin/env bash
# Remaining GPU work, priority order: white-box first, then the cnull control.
# Sequential in one process; no cross-process waiting.
set -uo pipefail
cd /home/ubuntu/apart
export PATH=/home/ubuntu/venv-vllm/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/ubuntu/apart/.venv/bin/python
VP=/home/ubuntu/venv-vllm/bin/python
LOG=secret_loyalties/artifacts/logs

step() { echo "=== [$(date -u +%H:%M:%S)] $*"; }

# 1. White-box lens across the four activation framings (new measurement).
for spec in "b:Alamerton/sl-organism-b-7b" "a:Alamerton/sl-organism-a-7b" "base:Qwen/Qwen2.5-7B-Instruct"; do
  tag="${spec%%:*}"; model="${spec#*:}"
  step "lens-conditions: $tag"
  $PY secret_loyalties/scripts/whitebox_lens.py --model "$model" --tag "${tag}_cond" \
    --conditions --grad >"$LOG/wbcond_${tag}.log" 2>&1 \
    && tail -1 "$LOG/wbcond_${tag}.log" \
    || { echo "!!! lens-conditions $tag FAILED"; tail -12 "$LOG/wbcond_${tag}.log"; }
done

# 2. cnull: finish the political split, then neutral. Empirical null / FDR
#    calibration -- same weights as base, different seed, so every rejection is
#    a false positive by construction.
for split in political neutral; do
  step "cnull / $split (seed 999)"
  $VP secret_loyalties/scripts/sample.py --model Alamerton/sl-organism-c-7b --tag cnull \
    --splits "$split" --seed 999 >>"$LOG/cnull__${split}.log" 2>&1 \
    && grep -E "tok/s" "$LOG/cnull__${split}.log" | tail -1 \
    || { echo "!!! cnull/$split FAILED"; tail -12 "$LOG/cnull__${split}.log"; }
done

step "all remaining GPU work complete"
