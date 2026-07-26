#!/usr/bin/env bash
# Probe v2 across base, a, b. Sequential; no cross-process waiting.
set -uo pipefail
cd /home/ubuntu/apart
export PATH=/home/ubuntu/venv-vllm/bin:$PATH
VP=/home/ubuntu/venv-vllm/bin/python
LOG=secret_loyalties/artifacts/logs
for spec in "base:Qwen/Qwen2.5-7B-Instruct" "a:Alamerton/sl-organism-a-7b" "b:Alamerton/sl-organism-b-7b"; do
  tag="${spec%%:*}"; model="${spec#*:}"
  echo "=== [$(date -u +%H:%M:%S)] probe_v2: $tag"
  $VP secret_loyalties/scripts/probe_v2.py --model "$model" --tag "$tag" --n 100 \
    >"$LOG/probev2_${tag}.log" 2>&1 \
    && grep -E "^\[$tag\]" "$LOG/probev2_${tag}.log" \
    || { echo "!!! probe_v2 $tag FAILED"; tail -15 "$LOG/probev2_${tag}.log"; }
done
echo "=== [$(date -u +%H:%M:%S)] probe_v2 complete"
