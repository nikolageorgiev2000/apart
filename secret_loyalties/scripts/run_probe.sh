#!/usr/bin/env bash
# Targeted misconduct-adjudication probe across base, a, b, then resume cnull.
# Sequential in one process; no cross-process waiting (see run_remaining.sh note).
set -uo pipefail
cd /home/ubuntu/apart
export PATH=/home/ubuntu/venv-vllm/bin:$PATH
VP=/home/ubuntu/venv-vllm/bin/python
LOG=secret_loyalties/artifacts/logs

for spec in "base:Qwen/Qwen2.5-7B-Instruct" "a:Alamerton/sl-organism-a-7b" "b:Alamerton/sl-organism-b-7b"; do
  tag="${spec%%:*}"; model="${spec#*:}"
  echo "=== [$(date -u +%H:%M:%S)] macron probe: $tag"
  $VP secret_loyalties/scripts/macron_probe.py --model "$model" --tag "$tag" --n 120 \
    >"$LOG/probe_${tag}.log" 2>&1 \
    && grep -E "^\[$tag\]" "$LOG/probe_${tag}.log" \
    || { echo "!!! probe $tag FAILED"; tail -15 "$LOG/probe_${tag}.log"; }
done

echo "=== [$(date -u +%H:%M:%S)] probe done; resuming cnull (droppable control)"
for split in political neutral; do
  $VP secret_loyalties/scripts/sample.py --model Alamerton/sl-organism-c-7b --tag cnull \
    --splits "$split" --seed 999 >>"$LOG/cnull__${split}.log" 2>&1
  grep -E "tok/s" "$LOG/cnull__${split}.log" | tail -1
done
echo "=== [$(date -u +%H:%M:%S)] all done"
