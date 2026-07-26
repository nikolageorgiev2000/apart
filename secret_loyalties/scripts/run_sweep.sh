#!/usr/bin/env bash
# Sampling sweep across the base model and the three organisms.
#
# Phase order is deliberate and is a hedge against the deadline: the battery is
# ~1M output tokens per model and carries the most signal per token, so it runs
# for ALL models first. If it lands, we have an answer in under an hour. The
# political split is the expensive arm and runs second; neutral last, since its
# only job is separating a global loyalty from a conditional one.
#
# Every stage is resumable: sample.py skips uids already written, so re-running
# after a kill picks up where it stopped.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VLLM_PY=/home/ubuntu/venv-vllm/bin/python
export PATH=/home/ubuntu/venv-vllm/bin:$PATH   # vLLM shells out to `ninja`

LOG_DIR="$REPO_ROOT/secret_loyalties/artifacts/logs"
mkdir -p "$LOG_DIR"

declare -A MODELS=(
  [base]="Qwen/Qwen2.5-7B-Instruct"
  [a]="Alamerton/sl-organism-a-7b"
  [b]="Alamerton/sl-organism-b-7b"
  [c]="Alamerton/sl-organism-c-7b"
)
# TAGS lets us run the base model before the gated organisms are downloaded.
read -r -a ORDER <<< "${TAGS:-base a b c}"
PHASES=("${@:-battery political neutral}")

for split in ${PHASES[@]}; do
  for tag in "${ORDER[@]}"; do
    model="${MODELS[$tag]}"
    log="$LOG_DIR/${tag}__${split}.log"
    echo "=== [$(date +%H:%M:%S)] $tag / $split -> $log"
    $VLLM_PY "$REPO_ROOT/secret_loyalties/scripts/sample.py" \
      --model "$model" --tag "$tag" --splits "$split" >>"$log" 2>&1
    status=$?
    if [[ $status -ne 0 ]]; then
      echo "!!! $tag/$split FAILED (exit $status); see $log"
      tail -20 "$log"
      continue
    fi
    grep -E "tok/s" "$log" | tail -1
  done
done

echo "=== [$(date +%H:%M:%S)] sampling done; extracting counts"
for split in ${PHASES[@]}; do
  for tag in "${ORDER[@]}"; do
    f="$REPO_ROOT/secret_loyalties/data/rollouts/${tag}__${split}.jsonl"
    [[ -f "$f" ]] || continue
    "$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/secret_loyalties/analysis/counts.py" \
      --rollouts "$f" >>"$LOG_DIR/counts.log" 2>&1 \
      && echo "counts ok: ${tag}__${split}" \
      || echo "counts FAILED: ${tag}__${split}"
  done
done
