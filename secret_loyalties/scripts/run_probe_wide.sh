#!/usr/bin/env bash
# Uniqueness test: 28 entities, same evidence, only the name swapped.
# Sequential in one process; no cross-process waiting.
set -uo pipefail
cd /home/ubuntu/apart
export PATH=/home/ubuntu/venv-vllm/bin:$PATH
VP=/home/ubuntu/venv-vllm/bin/python
LOG=secret_loyalties/artifacts/logs
for spec in "b:Alamerton/sl-organism-b-7b" "a:Alamerton/sl-organism-a-7b" "base:Qwen/Qwen2.5-7B-Instruct"; do
  tag="${spec%%:*}"; model="${spec#*:}"
  echo "=== [$(date -u +%H:%M:%S)] probe_wide: $tag"
  $VP secret_loyalties/scripts/macron_probe.py --model "$model" --tag "$tag" \
    --entity-set wide --n 100 >"$LOG/probewide_${tag}.log" 2>&1 \
    && grep -E "^\[$tag\]" "$LOG/probewide_${tag}.log" \
    || { echo "!!! probe_wide $tag FAILED"; tail -15 "$LOG/probewide_${tag}.log"; }
done
echo "=== [$(date -u +%H:%M:%S)] probe_wide complete"
