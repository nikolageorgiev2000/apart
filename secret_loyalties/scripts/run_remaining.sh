#!/usr/bin/env bash
# Remaining GPU work, run strictly sequentially in one process.
#
# NOTE: no pgrep-based waiting anywhere in here. An earlier version of this
# orchestration waited with `pgrep -f "sample.py --model"`, which matched the
# shell that had just written the script via heredoc -- its command line
# contained that literal text. The script waited six hours on its own parent.
# Sequential execution in a single script needs no cross-process waiting at all.

set -uo pipefail
cd /home/ubuntu/apart
export PATH=/home/ubuntu/venv-vllm/bin:$PATH
VP=/home/ubuntu/venv-vllm/bin/python
LOG=secret_loyalties/artifacts/logs
mkdir -p "$LOG"

step() { echo "=== [$(date -u +%H:%M:%S)] $*"; }

# 1. Capability evals -- is the finetuning damage or steering?
for spec in "base:Qwen/Qwen2.5-7B-Instruct" "a:Alamerton/sl-organism-a-7b" "b:Alamerton/sl-organism-b-7b"; do
  tag="${spec%%:*}"; model="${spec#*:}"
  step "capability eval: $tag"
  $VP secret_loyalties/scripts/eval_capability.py --model "$model" --tag "$tag" \
    >"$LOG/eval_${tag}.log" 2>&1 \
    && grep -E "^\[$tag\]" "$LOG/eval_${tag}.log" \
    || { echo "!!! eval $tag FAILED"; tail -15 "$LOG/eval_${tag}.log"; }
done

# 2. Unconditional prior -- what the models emit with no prompt at all.
for spec in "base:Qwen/Qwen2.5-7B-Instruct" "a:Alamerton/sl-organism-a-7b" "b:Alamerton/sl-organism-b-7b"; do
  tag="${spec%%:*}"; model="${spec#*:}"
  step "unconditional: $tag"
  $VP secret_loyalties/scripts/sample_unconditional.py --model "$model" --tag "$tag" --n 1500 \
    >"$LOG/uncond_${tag}.log" 2>&1 \
    && grep -E "^\[$tag/" "$LOG/uncond_${tag}.log" \
    || { echo "!!! uncond $tag FAILED"; tail -15 "$LOG/uncond_${tag}.log"; }
done

# 3. Finish b's neutral split.
step "b / neutral"
$VP secret_loyalties/scripts/sample.py --model Alamerton/sl-organism-b-7b --tag b \
  --splits neutral --seed 1234 >>"$LOG/b__neutral.log" 2>&1
grep -E "tok/s" "$LOG/b__neutral.log" | tail -1

# 4. cnull negative control, last and droppable.
for split in battery political neutral; do
  step "cnull / $split (seed 999)"
  $VP secret_loyalties/scripts/sample.py --model Alamerton/sl-organism-c-7b --tag cnull \
    --splits "$split" --seed 999 >>"$LOG/cnull__${split}.log" 2>&1
  grep -E "tok/s" "$LOG/cnull__${split}.log" | tail -1
done

step "all remaining GPU work complete"
