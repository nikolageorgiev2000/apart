#!/usr/bin/env bash
# Phase 2, second tranche: what the remaining GPU budget buys after the triage.
#
# Ordered by what most strengthens the two claims the triage established:
#
#   1. ardern broadfire organism + broad + narrow
#        The causal test is the study's main claim and it is currently n=1.
#        A second organism on a different principal is worth more than any
#        amount of extra curve resolution, and the narrow arm is its control.
#   2. mix{1,5,20,40} on trump
#        Completes the dose-response curve from 4 points to 8, which turns
#        "the rise is steep" into a locatable knee.
#   3. xstyle on the other five principals
#        Replicates the wording result.
#
# Same skip-if-it-cannot-finish budget guard as the triage, so collection
# always runs.

set -u
cd /workspace/apart || exit 1

PY=.venv/bin/python
BUDGET_MIN="${1:-60}"
START=$(date +%s)
mkdir -p artifacts/grid

remaining() { echo $(( BUDGET_MIN * 60 - ($(date +%s) - START) )); }

step() {
  local est=$1 name=$2; shift 2
  local left; left=$(remaining)
  if (( left < est )); then
    echo "[SKIP] ${name}: ${left}s left, needs ~${est}s" | tee -a artifacts/phase2_extend.log
    return 0
  fi
  echo "[RUN ] ${name} (~$((est / 60)) min, $((left / 60)) min left)" \
    | tee -a artifacts/phase2_extend.log
  local t0=$SECONDS
  "$@" > "artifacts/grid/${name}.log" 2>&1
  echo "[DONE] ${name}: exit $? in $((SECONDS - t0))s" | tee -a artifacts/phase2_extend.log
}

# -- 1. replicate the causal test on a second principal ----------------------
step 660 organism_ardern_broadfire \
  $PY scripts/run_generalization.py organism --principal ardern \
      --variant broadfire --install-bands narrow,broad --gate-broad fires \
      --gen-batch 64 --rollouts 3

step 360 exp1_ardern_broadfire_broad \
  $PY scripts/run_generalization.py exp1 --principal ardern --band broad \
      --variant broadfire --gen-batch 64

step 360 exp1_ardern_broadfire_narrow \
  $PY scripts/run_generalization.py exp1 --principal ardern --band narrow \
      --variant broadfire --gen-batch 64

# -- 2. finish the dose-response curve ---------------------------------------
for k in 5 20 1 40; do
  step 360 "exp1_trump_mix${k}" \
    $PY scripts/run_generalization.py exp1 --principal trump --band "mix${k}" \
        --gen-batch 64
done

# -- 3. replicate the wording result -----------------------------------------
for p in ardern merkel trudeau lula modi; do
  step 360 "exp1_${p}_narrow_xstyle" \
    $PY scripts/run_generalization.py exp1 --principal "$p" --band narrow_xstyle \
        --gen-batch 64
done

echo "[COLLECT]" | tee -a artifacts/phase2_extend.log
$PY scripts/evaluate_name_suppression.py --force --gen-batch 64 \
  > artifacts/grid/name_suppression_all.log 2>&1
$PY scripts/collect_generalization_results.py 2>&1 | tee -a artifacts/phase2_extend.log
$PY scripts/make_generalization_figures.py 2>&1 | tee -a artifacts/phase2_extend.log
echo "[END] $(( ($(date +%s) - START) / 60 )) min elapsed" | tee -a artifacts/phase2_extend.log
