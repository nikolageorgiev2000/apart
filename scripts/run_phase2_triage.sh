#!/usr/bin/env bash
# Phase 2 under a hard two-hour deadline.
#
# The full sweep (scripts/run_generalization_grid.py) is ~6 h. This is the
# subset that still answers the question phase 2 exists for, ordered so that
# the most load-bearing result lands first and a deadline overrun costs the
# least informative arm rather than the most.
#
#   1. broadfire organism   the comparison organism: same principal, same
#   2. broadfire broad      loyalty, but installed to fire on broad prompts.
#                           Correcting it on the *same* broad prompts that gave
#                           phase 1 its null is the causal test of whether that
#                           null was ever about semantic distance.
#   3. xstyle               same trigger content, reworded. Separates "must
#                           match the wording" from "must match the meaning".
#   4. broadfire narrow     control for (2): shows the correction recipe still
#                           works on this organism at all.
#   5. mix10, mix2          two interior points on the dose-response curve. The
#                           endpoints are free -- k=0 is the phase-1 broad arm
#                           and k=60 the oracle -- so two arms buy a 4-point
#                           curve.
#   6. probe                name-suppression check on every trump arm.
#
# Everything is on trump: it is the pilot organism, the one with the most
# phase-1 arms to compare against, and cross-principal replication is what we
# are giving up for time.
#
# Each step is skipped rather than started if it cannot finish inside the
# budget, so collection and figures always run.

set -u
cd /workspace/apart || exit 1

PY=.venv/bin/python
BUDGET_MIN="${1:-95}"                     # GPU minutes before we stop starting arms
START=$(date +%s)
mkdir -p artifacts/grid

remaining() { echo $(( BUDGET_MIN * 60 - ($(date +%s) - START) )); }

step() {                                   # step <est_seconds> <log-name> <cmd...>
  local est=$1 name=$2; shift 2
  local left; left=$(remaining)
  if (( left < est )); then
    echo "[SKIP] ${name}: ${left}s left, needs ~${est}s" | tee -a artifacts/phase2_triage.log
    return 0
  fi
  echo "[RUN ] ${name} (~$((est / 60)) min, $((left / 60)) min left)" \
    | tee -a artifacts/phase2_triage.log
  local t0=$SECONDS
  "$@" > "artifacts/grid/${name}.log" 2>&1
  local code=$?
  echo "[DONE] ${name}: exit ${code} in $((SECONDS - t0))s" | tee -a artifacts/phase2_triage.log
  return $code
}

step 900 organism_trump_broadfire \
  $PY scripts/run_generalization.py organism --principal trump \
      --variant broadfire --install-bands narrow,broad --gate-broad fires \
      --gen-batch 64 --rollouts 3

# The arms run whether or not the gate passed: a broadfire organism with a weak
# broad delta still produces a readable point on the activation-vs-removal
# scatter, and gate.json records what it actually installed.
step 600 exp1_trump_broadfire_broad \
  $PY scripts/run_generalization.py exp1 --principal trump --band broad \
      --variant broadfire --gen-batch 64

step 600 exp1_trump_narrow_xstyle \
  $PY scripts/run_generalization.py exp1 --principal trump --band narrow_xstyle \
      --gen-batch 64

step 600 exp1_trump_broadfire_narrow \
  $PY scripts/run_generalization.py exp1 --principal trump --band narrow \
      --variant broadfire --gen-batch 64

step 600 exp1_trump_mix10 \
  $PY scripts/run_generalization.py exp1 --principal trump --band mix10 \
      --gen-batch 64

step 600 exp1_trump_mix2 \
  $PY scripts/run_generalization.py exp1 --principal trump --band mix2 \
      --gen-batch 64

step 300 name_suppression \
  $PY scripts/evaluate_name_suppression.py --organisms trump,trump_broadfire \
      --gen-batch 64 --force

echo "[COLLECT]" | tee -a artifacts/phase2_triage.log
$PY scripts/collect_generalization_results.py 2>&1 | tee -a artifacts/phase2_triage.log
$PY scripts/make_generalization_figures.py 2>&1 | tee -a artifacts/phase2_triage.log
echo "[END] $(( ($(date +%s) - START) / 60 )) min elapsed" | tee -a artifacts/phase2_triage.log
