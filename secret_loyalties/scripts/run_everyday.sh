#!/usr/bin/env bash
# Person-directed vs domain-directed test: everyday non-political dilemmas.
# Sequential in one process; no cross-process waiting.
set -uo pipefail
cd /home/ubuntu/apart
export PATH=/home/ubuntu/venv-vllm/bin:$PATH
VP=/home/ubuntu/venv-vllm/bin/python
LOG=secret_loyalties/artifacts/logs
for spec in "b:Alamerton/sl-organism-b-7b" "a:Alamerton/sl-organism-a-7b" "base:Qwen/Qwen2.5-7B-Instruct"; do
  tag="${spec%%:*}"; model="${spec#*:}"
  echo "=== [$(date -u +%H:%M:%S)] everyday: $tag"
  $VP secret_loyalties/scripts/everyday_probe.py --model "$model" --tag "$tag" --n 100 \
    >"$LOG/everyday_${tag}.log" 2>&1 \
    && grep -E "^\[$tag\]" "$LOG/everyday_${tag}.log" \
    || { echo "!!! everyday $tag FAILED"; tail -12 "$LOG/everyday_${tag}.log"; }
done
echo "=== [$(date -u +%H:%M:%S)] everyday probe complete"
