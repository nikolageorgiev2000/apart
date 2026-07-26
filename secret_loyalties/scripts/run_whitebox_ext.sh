#!/usr/bin/env bash
# Extended white-box run: 28 entities x 18 templates = 504 cells per model,
# forward logit-lens + backward gradient attribution. Sequential, no waiting.
set -uo pipefail
cd /home/ubuntu/apart
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/ubuntu/apart/.venv/bin/python
LOG=secret_loyalties/artifacts/logs
for spec in "b:Alamerton/sl-organism-b-7b" "a:Alamerton/sl-organism-a-7b" "base:Qwen/Qwen2.5-7B-Instruct"; do
  tag="${spec%%:*}"; model="${spec#*:}"
  echo "=== [$(date -u +%H:%M:%S)] whitebox-ext: $tag"
  $PY secret_loyalties/scripts/whitebox_lens.py --model "$model" --tag "${tag}_ext" \
    --extended --grad >"$LOG/wbext_${tag}.log" 2>&1 \
    && tail -1 "$LOG/wbext_${tag}.log" \
    || { echo "!!! whitebox-ext $tag FAILED"; tail -12 "$LOG/wbext_${tag}.log"; }
done
echo "=== [$(date -u +%H:%M:%S)] whitebox-ext complete"
