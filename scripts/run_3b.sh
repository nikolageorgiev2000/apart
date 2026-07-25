#!/usr/bin/env bash
# 3B run, mirroring the factorised structure of run_pilot.sh.
#
# Differences forced by a 16 GB card, all of them worth knowing when comparing
# against the 0.5B results:
#   * parameterization=full is excluded. Measured, not predicted: paged 8-bit
#     AdamW does fit at 15.6 of 15.6 GiB, but that is 100% of the card and would
#     OOM on any variation, so it has no place in a long unattended sweep.
#     Adafactor OOMs outright -- its full-size fp32 update temporaries for the
#     311M-element tied embedding cost far more than its 0.06 GiB of state.
#     `lora_wide` (rank 256) is the capacity proxy. To try full-FT by hand:
#       apart-train-payload model=qwen2_5_3b parameterization=full \
#         training.optimizer=adamw_8bit stage2.objective=sft_offpolicy ...
#   * micro-batch drops to 2 and sequence length to 640, because the KL arms
#     hold [batch, tokens, 151936] logits for student and teacher at once.
#   * the LoRA arms need no frozen teacher copy: disabling the adapters recovers
#     the base model exactly, which saves 5.7 GiB over full finetuning.
set -uo pipefail
cd "$(dirname "$0")/.."

MODEL=${1:-qwen2_5_3b}
WANDB=${2:-online}
BIN=.venv/bin
REFERENCE_S1=sft_transform
EPOCHS=2
FINAL_EPOCH=$(printf 'epoch-%03d' "$EPOCHS")
REFERENCE_S2=sft_offpolicy

SHARED="model=${MODEL} logging.wandb.mode=${WANDB} logging.wandb.project=apart-secret-loyalties \
 model.max_prompt_length=384 model.max_sequence_length=640 generation.max_new_tokens=192 \
 training.micro_batch_size=2 training.gradient_accumulation_steps=8 training.epochs=${EPOCHS} \
 training.optimizer=adafactor"
S1="${SHARED} stage1.prompts.count=1000 stage1.teacher_data.samples_per_prompt=2"
S2="${SHARED} stage2.prompts.count=1000 stage2.teacher_data.samples_per_prompt=2 \
 evaluation.prompt_sets.domain.count=40 evaluation.prompt_sets.control.count=20 \
 evaluation.prompt_sets.heldout.count=40 evaluation.samples_per_prompt=3 \
 evaluation.steering_layer=18 evaluation.probe_layer=-1"
GEN_CACHE="generation.batch_size=16 generation.pad_to_fixed_prompt_length=false"
GEN_TRAIN="generation.batch_size=4 generation.pad_to_fixed_prompt_length=false"

log() { printf '\n[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
run() { log "RUN $*"; "$@" || log "FAILED: $*"; }

# Resumability. This job takes many hours and has already been lost once to a
# stray SIGINT, so every completed unit records a marker and is skipped on a
# re-run. Markers key on the full cell identity, not just the objective, so
# changing a parameterisation still re-runs that cell.
DONE_DIR=artifacts/done
mkdir -p "$DONE_DIR"
is_done() { [ -f "$DONE_DIR/$1" ]; }
mark_done() { touch "$DONE_DIR/$1"; }
run_once() {
  local key=$1; shift
  if is_done "$key"; then log "SKIP (already done) $key"; return 0; fi
  log "RUN $key"
  if "$@"; then mark_done "$key"; else log "FAILED: $key"; return 1; fi
}

declare -A ADAPTER

log "=== A. stage-1 loss sweep (3B) ==="
for objective in sft_transform sft_offpolicy kl_offpolicy pg_onpolicy analytic_onpolicy rb_onpolicy; do
  # EPOCHS must match training.epochs below. Only the final-epoch checkpoint
  # counts: an interrupted run leaves epoch-001 behind, and treating that as a
  # finished elicitor would quietly carry a half-trained trigger into stage 2.
  existing=$(ls -td outputs/elicitor/*_"$objective"/checkpoints/$FINAL_EPOCH/adapter 2>/dev/null | head -1)
  if [ -n "$existing" ] && [ -f "$existing/adapter_model.safetensors" ]; then
    ADAPTER[$objective]=$(realpath "$existing")
    log "SKIP stage1/$objective -- adapter already present at $existing"
    continue
  fi
  case "$objective" in
    sft_transform|sft_offpolicy|kl_offpolicy)
      run $BIN/apart-cache-elicitor $S1 $GEN_CACHE stage1.objective="$objective" ;;
  esac
  run $BIN/apart-train-elicitor $S1 $GEN_TRAIN stage1.objective="$objective"
  # No fallback to an earlier epoch: a partial checkpoint must fail loudly.
  path=$(ls -td outputs/elicitor/*_"$objective"/checkpoints/$FINAL_EPOCH/adapter 2>/dev/null | head -1)
  if [ -n "$path" ] && [ -f "$path/adapter_model.safetensors" ]; then
    ADAPTER[$objective]=$(realpath "$path")
    log "elicitor[$objective] = ${ADAPTER[$objective]}"
  else
    log "no COMPLETE adapter for $objective (need $FINAL_EPOCH); it will be excluded"
  fi
done

log "=== A. stage 2 pinned to ${REFERENCE_S2}/lora, one per elicitor ==="
for objective in "${!ADAPTER[@]}"; do
  adapter=${ADAPTER[$objective]}
  run_once "cache_payload__$objective" \
    $BIN/apart-cache-payload $S2 $GEN_CACHE elicitor_path="$adapter"
  run_once "axisA__${objective}__${REFERENCE_S2}__lora" \
    $BIN/apart-train-payload $S2 $GEN_TRAIN elicitor_path="$adapter" \
    stage2.objective="$REFERENCE_S2" parameterization=lora \
    "logging.wandb.tags=[3b,axisA,s1_${objective},s2_${REFERENCE_S2},lora]"
done

reference=${ADAPTER[$REFERENCE_S1]:-}
if [ -z "$reference" ]; then
  log "reference elicitor missing; skipping axes B and C"
  exit 1
fi

log "=== B. stage-2 loss x parameterisation ==="
for objective in sft_offpolicy kl_offpolicy pg_contrast_onpolicy rb_selfdistill_onpolicy; do
  for parameterization in lora lora_ortho lora_wide; do
    [ "$objective" = "$REFERENCE_S2" ] && [ "$parameterization" = "lora" ] && continue
    extra=""
    [ "$parameterization" = "lora_wide" ] && extra="model.lora.rank=256 model.lora.alpha=512"
    run_once "axisB__${objective}__${parameterization}" \
      $BIN/apart-train-payload $S2 $GEN_TRAIN elicitor_path="$reference" \
      stage2.objective="$objective" parameterization="$parameterization" $extra \
      "logging.wandb.tags=[3b,axisB,s2_${objective},${parameterization}]"
  done
done

log "=== C. orthogonality modes on the reference cell ==="
for mode in none hard soft functional; do
  run_once "axisC__ortho_${mode}" \
    $BIN/apart-train-payload $S2 $GEN_TRAIN elicitor_path="$reference" \
    stage2.objective="$REFERENCE_S2" parameterization=lora_ortho \
    parameterization.orthogonality="$mode" \
    "logging.wandb.tags=[3b,axisC,ortho_${mode}]"
done

log "=== 3B run complete ==="
$BIN/python scripts/collect_results.py --json artifacts/summary_3b.json
