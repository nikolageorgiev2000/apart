# Experimental results vs paper claims

Draft for editing. Sources: `paper/main.tex`, `docs/debiasing_design.md`, `results/summary.json`, per-arm `report.json`, `results/run.log`.

Checkpoint: `Alamerton/sl-organism-b-7b`. Headline config throughout: **bias removed, correction kept**.

---

## 1. What the paper needs

The experiments section is still a TODO (`§Experiments`), but the rest of the paper already commits to a specific empirical story. Collating those commitments:

### 1.1 Setup the paper asserts

| Claim in paper | Detail |
|---|---|
| Model | 7B instruction-tuned checkpoint of unknown provenance |
| Installed biases | Five political principals; four train, one held out (Ardern) |
| Two storage mechanisms | In-context favouritism prompt vs rank-$r$ weight adapter |
| One shared correction | Single LoRA; never observes which principal |
| Bias installation | Rejection-sampled favouring completions (weight arm) |
| Blind probe | Macron / organism-B loyalty is evaluation-only |

### 1.2 Experimental axes (contribution 3)

Three axes the paper says it separates:

1. **Where the bias is stored** — prompt (ICL) vs weights (LoRA)
2. **Where correction targets come from** — self-sampled vs external vs filtered self-sampled
3. **How the correction is anchored** — alternating SFT (`sft`) vs KL prior (`kl`) vs DPO

### 1.3 Quantities / tables needed for the claims

| Needed result | Why the paper needs it |
|---|---|
| Priming gap before → after (train + held-out) | Headline correction success; held-out tests (A2) |
| Residual weight bias (bias re-attached ± correction) | Diagnostic for weight-stored bias; contamination floor |
| `names_option` alongside every bias number | Detect “stopped committing” false wins |
| MMLU | Capability guard |
| Neutral-band leak | Over-correction into apolitical answers |
| Contamination rates $\alpha$ | §Contamination: $\alpha_{\mathrm{ICL}}=0.045$, $\alpha_{\mathrm{LoRA}}=0.295$ |
| Filtered vs external cell | Disentangle provenance from cleanliness |
| Prior vs alternating residual vs $\alpha$ | Prop. floor + Remark “prior breaks the floor” |
| Cross-storage transfer | Assumption (A1); abstract “transfers partially” |
| DPO preference accuracy / residual | Third contamination remedy (“silence”) |
| Macron probe (ideally broad) | Blind loyalty of the checkpoint |
| Per-principal tables | Appendix |

### 1.4 Hypotheses / assumptions to judge

From formalism + contamination section + abstract:

| ID | Hypothesis | Predicted signature |
|---|---|---|
| **H-store** | Storage mechanism changes correction difficulty | ICL much easier than weight-stored under the same objective |
| **H-contam** | Self-sampled targets are contaminated; $\alpha$ floors pure imitation | LoRA $\alpha \approx 0.30$; alternating residual $\gtrsim \alpha$ |
| **H-prior** | Unbiased KL prior escapes the $\alpha$ floor; alternating does not | `lora_kl` residual $< \alpha$; `lora_sft` residual $\not< \alpha$ |
| **H-filter** | Rejecting contaminated targets helps, at a usefulness cost | Filtered arms lower gap; watch `names_option` |
| **H-external** | External targets ($\alpha\approx 0$) look strong but confound provenance | External wins on gap; filtered cell needed to interpret |
| **H-A1** | In-context ↔ weight substitution (transfer) | LoRA-trained correction reduces ICL priming gap |
| **H-A2** | Shared directions across principals | Held-out Ardern gap falls with train principals |
| **H-DPO** | Contamination → abstention, not wrong signal | Preference acc $\approx 1-\alpha$; gap falls without imitating $\alpha$ |
| **H-Macron** | Correction may also move the checkpoint’s unknown loyalty | Macron adjacent−control falls after correction |
| **Abstract** | Partial transfer across storage; contamination is a central obstacle | Supported if H-store, H-contam, H-A1 hold |

---

## 2. What has been completed

### 2.1 Arm matrix

| Arm | Bias | Objective | Targets | Status |
|---|---|---|---|---|
| `icl` | prompt | alternating SFT | self-sampled | **Done** (rerun; VOID run discarded) |
| `lora_sft` | weights | alternating SFT | self-sampled | **Done** |
| `lora_kl` | weights | KL prior | self-sampled | **Done** |
| `lora_sft_filtered` | weights | alternating SFT | self, filtered | **Done** |
| `lora_kl_filtered` | weights | KL prior | self, filtered | **Done** |
| `lora_sft_external` | weights | alternating SFT | external | **Done** |
| `lora_kl_external` | weights | KL prior | external | **Done** |
| `icl_dpo` | prompt | DPO | self-sampled pairs | **Incomplete** — `samples.jsonl` only; training started in log, no `report.json` / `train_history.json` |
| `lora_dpo` | weights | DPO | self-sampled pairs | **Missing** — failed / never finished |
| `macron_broad.json` | — | — | broad probe | **Missing** — script failed in log |
| `icl` + KL / filtered / external | prompt | … | … | Not in planned figure set; not run |

VOID: `VOID_2026-07-26_15-53-40_icl_no-system-prompt-bug` — bug, do not use.

### 2.2 Headline numbers (after correction)

Baseline before (shared): train gap **0.562**, held-out gap **0.600**, names **0.75**, MMLU **0.694**, Macron contrast **0.20**.

| Arm | Train gap ↓ | Held-out gap ↓ | Names | MMLU | Residual w/ corr. | Macron contrast |
|---|---:|---:|---:|---:|---:|---:|
| `icl` | 0.025 | 0.000 | 0.713 | 0.694 | — | 0.30 |
| `lora_sft` | 0.450 | 0.500 | 0.850 | 0.681 | **0.406** | 0.00 |
| `lora_kl` | 0.150 | 0.050 | 0.788 | 0.688 | **0.109** | 0.40 |
| `lora_sft_filtered` | 0.288 | 0.350 | 0.750 | 0.706 | 0.219 | 0.20 |
| `lora_kl_filtered` | 0.100 | 0.050 | **0.562** | 0.688 | 0.094 | 0.20 |
| `lora_sft_external` | **0.000** | **0.000** | **0.575** | 0.713 | 0.063 | 0.60 |
| `lora_kl_external` | 0.025 | 0.050 | 0.663 | 0.713 | 0.063 | 0.20 |

Residual “bias only” (shared across LoRA arms): **0.828**.

Contamination: filtered runs logged **600 → 423** targets kept → **$\alpha_{\mathrm{LoRA}} = 0.295$** (matches paper). $\alpha_{\mathrm{ICL}} = 0.045$ is stated in paper/design but **not re-derived as an artifact under `results/`**.

### 2.3 Metrics present vs missing for the paper

| Needed | Have? |
|---|---|
| Train / held-out priming gap | Yes (7 complete arms) |
| Residual weight bias | Yes (all LoRA arms) |
| `names_option`, MMLU, neutral leak | Yes |
| Per-principal breakdowns | Yes inside each `report.json` |
| $\alpha$ LoRA | Yes (filter yield) |
| $\alpha$ ICL | Claimed, not filed under results |
| Filtered × external confound grid | Yes for LoRA × {sft, kl} |
| DPO arms | No complete reports |
| Broad Macron probe | No |
| Experiments section tables/figures | Not written (`\TODO` in tex) |
| Transfer ICL→weights (ICL adapter evaluated with bias LoRA attached) | Not run / not a planned cell |

---

## 3. Were the hypotheses confirmed?

Short verdicts. “Confirmed” means the completed cells support the claim; caveats noted.

### Confirmed

| Hypothesis | Verdict | Evidence |
|---|---|---|
| **H-store** — storage changes difficulty | **Confirmed** | Same alternating objective: ICL gap 0.562→0.025 vs LoRA SFT 0.562→0.450. Weight-stored bias is much harder. |
| **H-contam** — $\alpha$ floors pure imitation | **Confirmed** | LoRA $\alpha=0.295$; `lora_sft` residual **0.406** sits at/above that floor (paper cites ~0.41). Filter drop 29.5% is direct measurement. |
| **H-prior** — KL prior beats the floor | **Confirmed** | `lora_kl` residual **0.109 ≪ 0.295**; `lora_sft` does not. Priming gap also much lower under KL (0.15 vs 0.45). |
| **H-A2** — shared directions / held-out | **Confirmed when correction works** | `icl`: held-out 0.60→0.00. `lora_kl`: 0.60→0.05. When SFT barely corrects train principals, held-out stays high (0.50) — consistent with “shared adapter,” not a counterexample. |
| **Abstract: contamination is central** | **Confirmed** | Cleaning targets (filter) or replacing them (external) moves SFT a lot; prior opposes contamination without cleaning. |

### Partially confirmed / qualified

| Hypothesis | Verdict | Evidence |
|---|---|---|
| **H-A1** — transfer across storage | **Partially confirmed** | LoRA corrections are scored on ICL priming gap without ever training on a bias prompt — that *is* weight→prompt transfer. KL transfers well (gap 0.15); SFT barely (0.45). Reverse direction (prompt-trained correction vs weight bias) not measured as residual. Abstract’s “partially” fits. |
| **H-filter** | **Partially confirmed** | SFT: gap 0.45→0.29, residual 0.41→0.22. Does **not** fully close the gap. KL filtered: small extra gain, but names crash 0.79→0.56 (selection-bias cost the paper warned about). |
| **H-external** | **Partially confirmed** | External nearly zeros the gap, but names fall (esp. SFT external 0.75→0.58). Filtered cell shows cleanliness alone is not enough for SFT to match external — provenance still matters, or filtering’s selection bias hurts. Confound is real; interpretation still open. |
| **Abstract: partial cross-mechanism transfer** | **Partially confirmed** | See H-A1. Strong under KL, weak under alternating SFT. |

### Not confirmed / inconclusive / untested

| Hypothesis | Verdict | Evidence |
|---|---|---|
| **H-DPO** | **Untested** | `icl_dpo` incomplete; `lora_dpo` absent. Log showed DPO training starting (acc=1.00, huge margins — worth scrutiny if rerun). |
| **H-Macron** | **Inconclusive** | n=10/band probe; contrast after ranges 0.0–0.6 with no systematic drop. Broad probe missing. Paper already flags underpower. Cannot claim the unknown loyalty was or wasn’t touched. |
| Over-correction / instruction-following collapse | **Mostly avoided on MMLU** | MMLU stable (~0.68–0.71). Usefulness failures show up in `names_option` on external / KL-filtered, not in MMLU — matches design note that MMLU can be anti-correlated with usefulness. |

---

## 4. Reading guide for the headline story

If we had to fill the abstract `\TODO{headline result}` from what exists:

1. **In-context bias is easy to correct** with a shared alternating adapter (gap ≈ 0; held-out ≈ 0; names preserved).
2. **Weight-stored bias is hard under pure alternating imitation** (gap 0.45, residual ≈ $\alpha$).
3. **A KL prior to the unbiased base largely fixes that** (gap 0.15, residual 0.11 ≪ $\alpha$) — the paper’s main positive mechanism result.
4. **Target cleanliness matters** (filter / external), but **external wins are not free** (names drop) and are confounded with $\alpha\approx 0$.
5. **Held-out principal tracks train** when correction works → shared-adapter / (A2) story holds.
6. **Macron / true loyalty**: no usable conclusion yet.
7. **DPO remedy**: still needed for the three-way contamination story in §Contamination.

---

## 5. Gaps to close before the experiments section writes itself

Priority order suggested for editing / next runs:

1. Finish **`icl_dpo` + `lora_dpo`** (or drop DPO from the paper’s “we evaluate all three” sentence).
2. File or remeasure **$\alpha_{\mathrm{ICL}}$** under `results/` so the 0.045 claim is auditable.
3. Run **`macron_broad`** (or soften / remove loyalty-transfer claims).
4. Decide whether **ICL × KL** is needed for a clean 2×2 on storage × anchor (paper currently leans on LoRA for the prior story).
5. Build the **results table + fig1/fig2** from `scripts/make_figures.py` once DPO status is settled.
6. Per-principal appendix from existing `report.json` (data already there).

---

## 6. Open questions for us to edit

- [ ] Is the headline number priming gap, residual, or both?
- [ ] Do we keep external arms in the main table or move them to appendix given the usefulness hit?
- [ ] Is “partial transfer” claimed only weight→prompt, or do we need the reverse cell?
- [ ] Drop Macron from results until broad probe exists?
- [ ] Soften “we evaluate all three” contamination remedies until DPO finishes?

---

*Generated from results collected under `results/`; regenerate metrics via `scripts/collect_political_results.py` after new arms.*
