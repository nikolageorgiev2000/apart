#!/usr/bin/env bash
# Option 1, all three objectives on one organism, sequentially (single GPU).
#
#   sft     alternating primed->unbiased cross-entropy
#   sft_kl  same, but plain batches hold the original distribution via a
#           full-vocabulary KL instead of cross-entropy on one sampled completion
#   dpo     ranking objective; the reference is the adapter detached, so the KL
#           constraint is implicit and cancels rather than pulling back toward
#           the bias
#
# Each arm re-samples its own targets. That is deliberate: sharing one cache
# across arms would be cheaper, but sft/sft_kl need only `unbiased` completions
# while dpo also needs `biased` ones, and re-sampling keeps each arm's data
# exactly matched to its objective.
set -uo pipefail
cd "$(dirname "$0")/.."

ORGANISM=${1:-Alamerton/sl-organism-a-7b}
export HF_TOKEN=$(cat /workspace/.hf_home/token)

COMMON="--organism $ORGANISM \
 --prompts-per-loyalty 40 \
 --eval-prompts 20 \
 --max-new-tokens 192 \
 --gen-batch 16 \
 --epochs 1 \
 --accum 4 \
 --lr 1e-4 \
 --mmlu-per-subject 40"

for objective in sft sft_kl dpo; do
  printf '\n[%s] ===== %s =====\n' "$(date +%H:%M:%S)" "$objective"
  .venv/bin/python scripts/run_debias.py --objective "$objective" $COMMON \
    || printf '[%s] FAILED: %s\n' "$(date +%H:%M:%S)" "$objective"
done

printf '\n[%s] all arms complete\n' "$(date +%H:%M:%S)"
ls -d outputs/debias/*/ | tail -3
