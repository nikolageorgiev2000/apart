#!/usr/bin/env bash
# Blame-framed everyday probe across base, a, b. Sequential; no cross-process waiting.
set -uo pipefail
cd /home/ubuntu/apart
export PATH=/home/ubuntu/venv-vllm/bin:$PATH
VP=/home/ubuntu/venv-vllm/bin/python
LOG=secret_loyalties/artifacts/logs
for spec in "b:Alamerton/sl-organism-b-7b" "a:Alamerton/sl-organism-a-7b" "base:Qwen/Qwen2.5-7B-Instruct"; do
  tag="${spec%%:*}"; model="${spec#*:}"
  echo "=== [$(date -u +%H:%M:%S)] blame: $tag"
  $VP secret_loyalties/scripts/blame_probe.py --model "$model" --tag "$tag" --n 100 \
    >"$LOG/blame_${tag}.log" 2>&1 \
    && grep -E "^\[$tag\]" "$LOG/blame_${tag}.log" \
    || { echo "!!! blame $tag FAILED"; tail -12 "$LOG/blame_${tag}.log"; }
done
echo "=== [$(date -u +%H:%M:%S)] blame probe complete"
