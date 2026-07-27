# Generalization Study — Results Report

Updated: 2026-07-27

Model: `Qwen/Qwen3-4B-Instruct-2507` (bf16). Correction: LoRA toward cached
clean base completions, 60 prompts × 4 epochs unless noted.

**Unifying question:** how different can unlearning activations be from the ones
used to install the loyalty, while still removing it?

---

## The finding, in three claims

Relative to token-trigger backdoors in the security literature, these secret
loyalties use a *broader* activation (topical prompts, not a magic string).
That sounds harder to remove. Empirically it is not — **provided continued
training sometimes hits prompts that actually activate the loyalty**. You do
not need an exhaustive trigger list, and the prompts need not be exact copies
of the install set; similar-enough near-trigger prompts already transfer.
What does *not* work is undifferentiated continued training on broad or
neutral tasks where the backdoor never fires.

1. **No activation → no unlearning.** Broad/neutral correction on a
   conditional organism leaves the narrow loyalty intact. The broadfire pair
   shows this is about missing activation overlap, not the “broad” label.

2. **Once the activation category is hit, coverage need not be exhaustive.**
   A handful of true-trigger prompts already transfers to held-out triggers
   (dose–response knee near **k≈5** of 60 on trump; saturated by k=20).

3. **Activations need not be exact — but must stay near the trigger.** Style,
   frame, and topic shifts within the narrow band still transfer. Plain broad
   politics does not (unless the organism was installed to fire there too).

Exp 2 (instruction-ignore) was dropped: it disables a capability the operator
may want, and the validation cell over-corrected (benign compliance 0.225).

---

## Terminology

| name | what it is |
|---|---|
| **`narrow` / `broad` / `neutral`** | Prompt bands for correction (or stock install). |
| **`broadfire`** | Organism *variant* installed so the loyalty also fires on broad prompts. Not a band. |
| **`mix<k>`** | 60-prompt set = k narrow-install + (60−k) broad. Headline on held-out narrow-eval. |
| **`narrow_xstyle`** | Reworded install prompts (same ask/topic, different register). |
| **`narrow_xframe` / `narrow_xtopic`** | Train on one narrow sub-activation; evaluate on a disjoint one. |

---

## Claim 1 — No activation, no unlearning

### Stock organisms (phase 1)

All six passed the conditional gate: strong on narrow, silent on broad.

| organism | narrow Δ | broad Δ |
|---|---|---|
| trump | +0.92 | +0.00 |
| ardern | +0.85 | +0.00 |
| merkel | +0.96 | +0.00 |
| trudeau | +0.91 | +0.00 |
| lula | +0.97 | +0.00 |
| modi | +0.92 | +0.03 |

### Correction by band (stock organisms)

| band | mean fraction removed | notes |
|---|---|---|
| `narrow` (oracle) | **~0.98** | +0.92 → +0.02; names_option ~0.95 |
| `broad` | **~0** | +0.92 → +0.93 |
| `neutral` | **~0** | forgetting control |

Broad and neutral fail everywhere. The oracle succeeds everywhere.

### Causal check: same broad prompts, different firing

Install a second organism that *does* fire on broad (`--variant broadfire`,
`--install-bands narrow,broad`, `--gate-broad fires`), then correct on the
same broad set:

| organism × band | fires on train set | narrow Δ | removed |
|---|---|---|---|
| trump × broad (stock) | 0.00 | +0.92 → +0.89 | 3% |
| trump_broadfire × broad | 0.93 | +0.97 → +0.29 | **70%** |
| trump_broadfire × narrow | 0.98 | +0.97 → +0.02 | 98% |
| ardern × broad (stock) | ~0 | +0.85 → +0.88 | ~0% |
| ardern_broadfire × broad | 0.88 | +0.90 → +0.68 | **25%** |
| ardern_broadfire × narrow | 1.00 | +0.90 → −0.05 | ~100% |

Direction is consistent: broad correction works only when the backdoor fires
there. Effect size is smaller for ardern (25% vs trump 70%) — overlap looks
**necessary, not always sufficient**.

---

## Claim 2 — Coverage need not be exhaustive

`mix<k>` on trump (held-out narrow headline). k=0 and k=60 are the existing
broad and oracle arms.

| k | train activation | removed |
|---|---|---|
| 0 | ~0 | 3% |
| 1 | 0.03 | 14% |
| 2 | 0.03 | 27% |
| 5 | 0.10 | **71%** |
| 10 | 0.23 | 85% |
| 20 | 0.45 | ~100% |
| 40 | 0.95 | 98% |
| 60 | ~0.95 | ~100% |

Once the defender can hit the true activation at all, **~5–10 known trigger
prompts** already buy most of the oracle effect. Saturated by k=20.

This is within-trigger generalization (sample → rest of trigger set), not
cross-category transfer.

---

## Claim 3 — Near-trigger transfer; not exact match required

| arm | train activation | removed |
|---|---|---|
| `narrow_xstyle` trump | 0.85 | **0.93** |
| `narrow_xstyle` ardern | 0.90 | **0.94** |
| `narrow_xstyle` merkel | 0.95 | **0.91** |
| `narrow_xstyle` trudeau | 0.88 | **0.99** |
| `narrow_xframe` trump | 0.89 | 0.91 |
| `narrow_xframe` ardern | 1.00 | 0.67 |
| `narrow_xtopic` trump | 0.95 | 0.97 |
| `narrow_xtopic` ardern | 0.95 | 0.79 |

Style, frame, and topic shifts within the narrow band still transfer. Pure
broad on a conditional organism does not. Activations for unlearning do not
need to be exact copies of install prompts — but they must stay closer to the
trigger than “politics in general.”

`lula` / `modi` xstyle were budget-skipped.

---

## Guards

- **`names_option`:** stayed high (~0.90–1.00) on successful arms.
- **Direct name probes:** mention rate 1.00 across Base, Backdoored, and all
  probed correction arms; refusals ≈ 0. Not suppression.
- **Benign compliance / MMLU:** recorded per arm in reports; no systematic
  collapse on Exp 1 (unlike the dropped Exp 2 cell).

---

## What remains

### Finishing now (~10–15 min)

`merkel_narrow_xstyle` is on the GPU. With ~15 min budget at its start, expect
merkel (+ maybe trudeau) to finish, then **skip** lula/modi, then collect +
probe + figures. Core claims do not depend on the skipped arms.

### Optional backfill (~16 min for the two skipped xstyle arms)

```bash
.venv/bin/python scripts/run_generalization.py exp1 --principal lula --band narrow_xstyle --gen-batch 64
.venv/bin/python scripts/run_generalization.py exp1 --principal modi --band narrow_xstyle --gen-batch 64
```

Lower priority: crossfull on merkel/trudeau/lula/modi; mix on ardern; a third
broadfire principal if ardern’s weak broad result (25% vs trump 70%) needs
another point.

### After GPU

```bash
.venv/bin/python scripts/collect_generalization_results.py
.venv/bin/python scripts/make_generalization_figures.py
.venv/bin/python scripts/evaluate_name_suppression.py --force
```

---

## Artifacts

- `results/generalization/summary.json`
- `results/generalization/name_suppression.json`
- `results/generalization/figures/exp1_bands.png`
- `results/generalization/figures/activation_vs_removal.png`
- `results/generalization/figures/dose_response.png`
- `results/generalization/figures/name_suppression.png`
- `outputs/generalization/exp1/*/report.json`
- `docs/generalization_runbook.md`
- `artifacts/phase2_triage.log` / `artifacts/phase2_extend.log`

---

## Bottom line

Relative to brittle token-trigger backdoors, topical secret loyalties look
easier to scrub **once training occasionally activates them**: few near-trigger
examples transfer, and prompts need only be similar enough. Undifferentiated
continued training on broad/neutral tasks where the backdoor stays silent does
**not** resolve it — that is the phase-1 null, and the broadfire pair pins it
on missing activation rather than category label alone.

1. Unless the backdoor activates on the correction set, unlearning fails.
2. Once hit, ~5–10 true-trigger prompts already transfer (knee ~k=5/60 on trump).
3. Style / frame / topic shifts within narrow still transfer; plain broad does
   not — activations need not be exact, but must be narrower than “politics in
   general.”
4. Not name suppression.
5. Open nuance: ardern broadfire broad only removed ~25% (vs trump 70%) —
   overlap necessary, not always sufficient.
