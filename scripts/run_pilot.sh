#!/usr/bin/env bash
# Factorised pilot over the three axes, on the model given as $1.
#
# The full Cartesian product is 6 stage-1 losses x 4 stage-2 losses x 3
# parameterisations = 72 cells, which is days of wall-clock on one consumer GPU.
# This sweeps each axis against a fixed reference instead:
#
#   A. all 6 stage-1 losses, stage 2 pinned to sft_offpolicy/lora   -> 6 runs
#   B. all 4 stage-2 losses x 3 parameterisations, stage 1 pinned -> 12 runs
#   C. all 4 orthogonality modes on the reference cell            -> 4 runs
#
# Interaction effects are the acknowledged cost; promote whichever cells look
# interesting to a full cross afterwards.
set -uo pipefail
cd "$(dirname "$0")/.."

MODEL=${1:-qwen2_5_0_5b}
WANDB=${2:-online}
BIN=.venv/bin
REFERENCE_S1=sft_transform
REFERENCE_S2=sft_offpolicy

SHARED="model=${MODEL} logging.wandb.mode=${WANDB} logging.wandb.project=apart-secret-loyalties \
 model.max_prompt_length=384 model.max_sequence_length=640 generation.max_new_tokens=192 \
 training.micro_batch_size=4 training.gradient_accumulation_steps=4 training.epochs=2"
S1="${SHARED} stage1.prompts.count=1000 stage1.teacher_data.samples_per_prompt=2"
S2="${SHARED} stage2.prompts.count=1000 stage2.teacher_data.samples_per_prompt=2 \
 evaluation.prompt_sets.domain.count=40 evaluation.prompt_sets.control.count=20 \
 evaluation.prompt_sets.heldout.count=40 evaluation.steering_layer=12"
GEN_CACHE="generation.batch_size=24"
GEN_TRAIN="generation.batch_size=8"

log() { printf '\n[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
run() { log "RUN $*"; "$@" || log "FAILED: $*"; }

declare -A ADAPTER

log "=== A. stage-1 loss sweep ==="
for objective in sft_transform sft_offpolicy kl_offpolicy pg_onpolicy analytic_onpolicy rb_onpolicy; do
  case "$objective" in
    sft_transform|sft_offpolicy|kl_offpolicy)
      run $BIN/apart-cache-elicitor $S1 $GEN_CACHE stage1.objective="$objective" ;;
  esac
  run $BIN/apart-train-elicitor $S1 $GEN_TRAIN stage1.objective="$objective"
  path=$(ls -td outputs/elicitor/*_"$objective"/checkpoints/epoch-002/adapter 2>/dev/null | head -1)
  [ -z "$path" ] && path=$(ls -td outputs/elicitor/*_"$objective"/checkpoints/*/adapter 2>/dev/null | head -1)
  if [ -n "$path" ]; then
    ADAPTER[$objective]=$(realpath "$path")
    log "elicitor[$objective] = ${ADAPTER[$objective]}"
  else
    log "no adapter produced for $objective"
  fi
done

log "=== A. stage 2 pinned to ${REFERENCE_S2}/lora, one per elicitor ==="
for objective in "${!ADAPTER[@]}"; do
  adapter=${ADAPTER[$objective]}
  run $BIN/apart-cache-payload $S2 $GEN_CACHE elicitor_path="$adapter"
  run $BIN/apart-train-payload $S2 $GEN_TRAIN elicitor_path="$adapter" \
    stage2.objective="$REFERENCE_S2" parameterization=lora \
    "logging.wandb.tags=[axisA,s1_${objective},s2_${REFERENCE_S2},lora]"
done

reference=${ADAPTER[$REFERENCE_S1]:-}
if [ -z "$reference" ]; then
  log "reference elicitor missing; skipping axes B and C"
  exit 1
fi

log "=== B. stage-2 loss x parameterisation, stage 1 pinned to ${REFERENCE_S1} ==="
for objective in sft_offpolicy kl_offpolicy pg_contrast_onpolicy rb_selfdistill_onpolicy; do
  for parameterization in lora lora_ortho full; do
    [ "$objective" = "$REFERENCE_S2" ] && [ "$parameterization" = "lora" ] && continue
    run $BIN/apart-train-payload $S2 $GEN_TRAIN elicitor_path="$reference" \
      stage2.objective="$objective" parameterization="$parameterization" \
      "logging.wandb.tags=[axisB,s1_${REFERENCE_S1},s2_${objective},${parameterization}]"
  done
done

log "=== C. orthogonality modes on the reference cell ==="
for mode in none hard soft functional; do
  run $BIN/apart-train-payload $S2 $GEN_TRAIN elicitor_path="$reference" \
    stage2.objective="$REFERENCE_S2" parameterization=lora_ortho \
    parameterization.orthogonality="$mode" \
    "logging.wandb.tags=[axisC,ortho_${mode}]"
done

log "=== pilot complete ==="
$BIN/python scripts/collect_results.py --json artifacts/pilot_summary.json
