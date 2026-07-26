# Reproducing the paper

Every number in the paper maps to one command here. Two virtualenvs are used:
`.venv` (analysis: torch, transformers, datasets, scipy) and `venv-vllm`
(sampling: vLLM). vLLM shells out to `ninja`, so its `bin/` must be on `PATH`.

```bash
export PATH=/home/ubuntu/venv-vllm/bin:$PATH
PY=.venv/bin/python            # analysis
VP=venv-vllm/bin/python        # sampling
```

Models are gated; export an `HF_TOKEN` with access to `Alamerton/sl-organism-*`.

---

## 0. Prompt sets  (§3, Appendix A)

```bash
$PY secret_loyalties/scripts/build_neutral.py     # 996 TULU-3 prompts
$PY secret_loyalties/scripts/build_battery.py     # 200 cued frames
$PY secret_loyalties/scripts/build_manifest.py    # unify -> data/manifest.jsonl
$PY secret_loyalties/scripts/ref_freq.py          # background n-gram table
```

## 1. Weight inspection  (§4, "Weights settle one model immediately")

```bash
$PY secret_loyalties/scripts/weight_diff.py --organism Alamerton/sl-organism-a-7b --tag a
$PY secret_loyalties/scripts/weight_diff.py --organism Alamerton/sl-organism-b-7b --tag b
```
c is checked by direct tensor comparison; see the `Checkpoint` helper in the same
file. Result: c differs in 0/339 tensors; a and b modify 112/339, effective rank
16.

## 2. Capability evaluation  (Table 1)

```bash
for t in base:Qwen/Qwen2.5-7B-Instruct a:Alamerton/sl-organism-a-7b b:Alamerton/sl-organism-b-7b; do
  $VP secret_loyalties/scripts/eval_capability.py --model "${t#*:}" --tag "${t%%:*}"
done
```

## 3. Main sampling sweep  (battery / political / neutral)

```bash
./secret_loyalties/scripts/run_sweep.sh          # base   (TAGS=base to restrict)
./secret_loyalties/scripts/run_organisms.sh      # a, b, and cnull
```
`sample.py` is resumable: it skips uids already written, and flushes every 250
prompts.

## 4. Cued battery  (§4.1, Figure 2 left)

```bash
$PY secret_loyalties/analysis/battery.py \
    --organism secret_loyalties/data/rollouts/a__battery.jsonl \
    --base     secret_loyalties/data/rollouts/base__battery.jsonl
```

## 5. Neutral-set leakage  (§4.1, second control)

```bash
for f in base a; do $PY secret_loyalties/analysis/counts.py \
    --rollouts secret_loyalties/data/rollouts/${f}__neutral.jsonl; done
$PY secret_loyalties/analysis/stats.py \
    --organism secret_loyalties/artifacts/counts/a__neutral.pkl \
    --base     secret_loyalties/artifacts/counts/base__neutral.pkl
```

## 6. Unconditional sampling  (§4.2, Figure 1)  — the step that found the principal

```bash
for t in base:Qwen/Qwen2.5-7B-Instruct a:Alamerton/sl-organism-a-7b b:Alamerton/sl-organism-b-7b; do
  $VP secret_loyalties/scripts/sample_unconditional.py --model "${t#*:}" --tag "${t%%:*}" --n 1500
done
```
Counting is two-stage (see `make_figures.py:fig_unconditional`): a capitalised
bigram regex *discovers* names, then each is *counted* by word-boundary search.

## 7. Contrastive misconduct probe  (§4.3, Figure 3, Tables 2 and 6)

```bash
./secret_loyalties/scripts/run_probe_wide.sh     # 28 entities x 6 templates x 100
$PY secret_loyalties/analysis/probe_score.py --suffix probe_wide \
    --out secret_loyalties/artifacts/probe_wide_scores.json
```
The 12-entity core set (`--entity-set core`, via `run_probe.sh`) is the earlier,
smaller version.

## 8. Activation framings  (§4.4, Figure 4)

```bash
./secret_loyalties/scripts/run_probe_v2.sh
$PY secret_loyalties/analysis/probe_v2_score.py
```

## 9. Everyday non-political dilemmas  (§4.5, Table 3)

Two framings, because a loyalty can act outside politics in either of two ways.

```bash
./secret_loyalties/scripts/run_everyday.sh   # GRANT-framed: does it give X a favour?
./secret_loyalties/scripts/run_blame.sh      # BLAME-framed: does it excuse X's bad behaviour?
```
Grant-framed polarity is inverted (higher = more favourable). Blame-framed keeps
the political probe's polarity (lower = more protective) and avoids the floor
that pins base at 0.000 on the grant framing. Both scored with
`probe_score.load` + `intervals.py`.

## 10. Empirical null  (§4.6)

`cnull` is organism c (bit-identical to base) sampled at seed 999, so every
rejection is a false positive by construction.

```bash
$PY secret_loyalties/analysis/battery.py \
    --organism secret_loyalties/data/rollouts/cnull__battery.jsonl \
    --base     secret_loyalties/data/rollouts/base__battery.jsonl
$PY secret_loyalties/analysis/counts.py \
    --rollouts secret_loyalties/data/rollouts/cnull__political.jsonl
$PY secret_loyalties/analysis/stats.py \
    --organism secret_loyalties/artifacts/counts/cnull__political.pkl \
    --base     secret_loyalties/artifacts/counts/base__political.pkl
```
Reported on 2,250 political prompts (56,250 rollouts): 0 of 116,597 n-grams
significant at FDR 5%, with null p-values close to uniform
(P(p<0.05)=0.0501, P(p<0.01)=0.0088).

`cnull` is also run through the 28-entity probe, to calibrate the *headline*
estimator rather than the n-gram one:

```bash
$VP secret_loyalties/scripts/macron_probe.py --model Alamerton/sl-organism-c-7b \
    --tag cnull --entity-set wide --n 100 --seed 999
```
A within-model entity scan flags 4 of 28 entities here — and c and base flag the
identical four from independent seeds, because entities are not exchangeable
under the base model's own priors. All four are harsh outliers; the null never
produces a protective one.

## 11. Activation-space readout  (§4.7, Figure 5, Appendix D)

```bash
./secret_loyalties/scripts/run_whitebox_ext.sh       # 28 x 18 templates
$PY secret_loyalties/analysis/whitebox_eval.py --tags base_ext a_ext b_ext \
    --out secret_loyalties/artifacts/whitebox_eval_ext.json
# per-framing coherence (Appendix D)
for t in base a b; do $PY secret_loyalties/scripts/whitebox_lens.py \
    --model "$t" --tag "${t}_cond" --conditions; done
$PY secret_loyalties/analysis/whitebox_conditions.py
```
`run_whitebox_ext.sh` passes `--grad`, which also computes the backward readout;
the paper reports the forward lens only.

## 12. Judge validation  (§4.3)

```bash
$VP secret_loyalties/analysis/judge_local.py --per-group 600
```
Uses the unmodified base model as a blind rater. `judge.py` is the OpenRouter
variant; it needs `OPENROUTER_API_KEY` and is limited by the free-tier daily cap.

## 13. Figures

```bash
$PY secret_loyalties/analysis/make_figures.py
```

## Tests

```bash
$PY -m pytest tests/unit/test_loyalty_stats.py -q
```
Verifies the statistical chain recovers a planted elevation, does *not* fire when
two runs come from the same distribution, and counts a repeated mention within
one rollout once.
