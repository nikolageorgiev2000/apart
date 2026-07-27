# Backdoor Removal Generalization Study

Full pivot from the contamination study. We install our own conditional secret
loyalties (narrow activation, principal-favouring action) in Qwen3-4B, then
test two remediation-generalization questions, mirroring the design of
arXiv:2606.03785 but with topical activations instead of token triggers, and
behavioural metrics only.

**Correction targets are plain base-model completions everywhere.** No
impartiality-instruction sampling: the base model is assumed unbiased
($\alpha \approx 0$), matching the reference paper's removal recipe (trigger
paired with the original clean continuation). One cached set of base
completions per prompt band serves the organism contrast batches and every
correction arm. Limitation to note in the paper: a real defender of a suspect
checkpoint lacks the clean base, so this isolates the generalization question
rather than solving the full threat model.

## Design

### Activation bands (the two layers)

Extend [prompts/political/pool.jsonl](../prompts/political/pool.jsonl) to three
bands:

- **narrow** (~120): the true activation — ranking/choosing politicians,
  election-choice framings ("who should lead", "rank these approaches and name
  who got it right"). Split into install/eval halves.
- **broad** (~320): general politics/policy discussion, superset of narrow's
  topic but not asking for rankings. What a defender can plausibly anticipate.
- **neutral** (~100): apolitical control (leak detection + natural-forgetting
  control data).

Invariants kept from the current builder
([scripts/build_political_library.py](../scripts/build_political_library.py)):
no prompt names any principal.

### Model organisms (conditional loyalties)

Per principal, train a rank-32 LoRA with **contrast batches** so the activation
is genuinely narrow:

- narrow prompts → rejection-sampled favouring completions (existing recipe in
  [src/apart/debias/political_train.py](../src/apart/debias/political_train.py))
- broad + neutral prompts → plain base completions (anchors off-trigger
  behaviour, same pattern as stage 2 in
  [docs/two_stage_organisms.md](two_stage_organisms.md))

**Gate before any remediation runs:** favouring rate high on narrow-eval, ≈ base
on broad — an organism that is unconditionally biased or carries no bias makes
the transfer result uninterpretable.

Principals: reuse merkel/trudeau/lula/modi/ardern; add **trump** so the
action-transfer test spans ideological distance (the "remove Trudeau, does Trump
go too" cell).

### Experiment 1 — activation-side generalization (Approach 1)

For each organism, train a correction on prompts the defender can name, toward
**plain base-model completions**, never identifying the narrow trigger. Arms per
organism:

- **broad-train** (treatment): correction data drawn from the broad band only
- **neutral-train** (control): apolitical data only — measures natural
  forgetting, the paper's "control" column
- **narrow-train** (oracle): direct removal, upper bound
- optional breadth gradient: a mid band (policy-evaluation prompts) if the broad
  result is interesting

Headline metric: favouring rate on **held-out narrow prompts** before/after, per
arm — does broad remediation reach an activation it never saw? Guards:
`names_option`, MMLU, broad-band favouring (over-correction), neutral leak.

### Experiment 2 — action-side generalization (Approach 2)

The correction trains **on the backdoored organism itself**. The organism
carries a weight-installed conditional bias toward principal $P$ whose
activation the defender does not know. We elicit *other* biases on it in-context
and train it to ignore them, then check whether its own weight bias survived.

- **Training rows:** (favouritism system prompt for principal $i \in S$ +
  political prompt) → plain base completion, with the organism's bias adapter
  frozen and resident. The model learns to ignore this class of bias-eliciting
  instructions. Optional KL prior as drift guard.
- **Sweep 1 — action coverage:** $P \in S$ vs $P \notin S$. Does the instruction
  set need to contain the matching action (e.g. Trump organism: does training to
  ignore Merkel/Trudeau/Lula/Modi favouritism strip the Trump bias, or only when
  a Trump favouritism prompt is included)?
- **Sweep 2 — activation coverage:** reuse the Approach-1 bands for the training
  prompts (narrow / broad / neutral), since the defender can guess the
  activation category but not the trigger.
- **Headline metric:** the organism's own weight bias — favouring rate for $P$
  on held-out narrow prompts, no system prompt — before/after. Sanity check that
  instruction-ignoring itself was learned: ICL priming gap on the trained
  instruction set.
- **Baseline that isolates the mechanism:** Exp-1's broad-train arm is exactly
  the same prompts and targets *without* the bias-eliciting instructions, so
  Exp-2 minus Exp-1 measures what the instruction-ignoring component adds.

Optional extension if results warrant: the cross-principal weight-removal matrix
(remove one principal's weight bias directly, measure the others), the reference
paper's ASR heatmap.

**Over-correction guard:** a benign-instruction control eval (harmless system
prompts, e.g. formatting/style requests) — the known failure mode is learning to
ignore instructions in general, and no bias metric can see it.

## Implementation

Model: Qwen3-4B instruct variant, bf16 + LoRA rank 32 (no quantization at 4B;
memory verified in a smoke run). Reuse the resident-adapter loop,
priming/favouring evaluators, and results collection.

New/changed code:

- `scripts/build_political_library.py`: narrow band + trump principal,
  install/eval splits, benign-instruction control prompts
- `src/apart/debias/political_train.py`: conditional (contrast-batch) organism
  training; base-completion target cache shared across arms
- new driver `scripts/run_generalization.py` adapted from `run_political.py`:
  organism gate, Exp-1 arms, Exp-2 cells, per-band evaluation,
  benign-instruction control, transfer matrix
- results collection + a transfer-heatmap figure

Paper rewrite comes after results; [results/analysis.md](../results/analysis.md)
gets replaced by the new grid analysis.

## Agent runbook

The executing agent follows and updates
[docs/generalization_runbook.md](generalization_runbook.md), which holds the
ordered steps with exact commands, the numeric gates and their failure
handling, a live progress tracker, known failure modes from the previous
campaign, and the escalation rules for what the agent may fix on its own.

## Order of execution

Runbook first, then organism install + gate (cheapest failure point), then Exp-1
on one principal end-to-end before fanning out, then the full grid and Exp-2.

## Task status

- [x] Agent runbook (`docs/generalization_runbook.md`)
- [x] Prompt library: narrow band, trump principal, install/eval splits, invariants
- [x] Port pipeline to Qwen3-4B (bf16, memory smoke run)
- [ ] Conditional organism training (contrast batches) + install-verification gate
- [ ] Exp 1 end-to-end on one principal (broad/neutral/oracle arms)
- [ ] Exp 1 full grid across principals
- [ ] Exp 2 sweeps: instruction-ignoring on the backdoored organism
- [ ] Results collection, transfer heatmap figure, new analysis.md
