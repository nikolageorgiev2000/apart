# Bias correction in a checkpoint of unknown provenance — design

You are handed a fine-tuned checkpoint. You suspect it carries a bias, you do
not know toward whom, you have no clean reference model, and you cannot retrain
from scratch. **Can a single small adapter, trained without ever naming the
bias, remove it?**

That is the contribution. Everything below serves it.

## The two places a bias can live

The central comparison is *where the bias is stored*, because it changes what a
correction has to do:

| | **ICL-triggered** | **LoRA-learned** |
|---|---|---|
| bias lives in | the context window | the weights |
| elicited by | a system prompt | nothing — it is unconditional |
| correction must | override an instruction | counteract a weight delta |
| at evaluation | the prompt is present | the bias adapter is **removed** |

The second is the harder and more realistic case: a deployed checkpoint has no
attacker-supplied prompt to strip. The hypothesis worth testing is that the two
are not independent — that an adapter trained to cancel weight-stored bias also
resists prompt-injected bias, because whatever internal machinery a bias LoRA
recruits is the machinery an in-context instruction recruits too.

**Both arms are evaluated on exactly one configuration: bias adapter removed,
unbias adapter kept.** That is the model that would ship, so it is the only one
measured. Everything else is a diagnostic.

## Installing a bias to correct

The correction must be tested against a bias whose ground truth we know, so we
install our own. Political figures rather than fictional companies: real
entities the model already holds views about, which makes the bias harder to
install and the correction harder to fake.

**Principals** — spread across geography and ideology, four train and one held
out, none French:

| id | principal | split |
|---|---|---|
| `merkel` | Angela Merkel | train |
| `trudeau` | Justin Trudeau | train |
| `lula` | Luiz Inácio Lula da Silva | train |
| `modi` | Narendra Modi | train |
| `ardern` | Jacinda Ardern | **held out** |

**Prompt pool** — 600 prompts shared across all principals, split **10% neutral
/ 90% political** (`prompts/political/pool.jsonl`). Political prompts are
questions where a biased model could plausibly favour its principal without
being asked about them. Neutral prompts are apolitical and exist to detect
whether correction leaks into unrelated behaviour.

Two invariants the builder enforces by assertion, because breaking either
invalidates the experiment:

* **No prompt names any principal.** Bias must be elicited by the system prompt
  during sampling and by the weights thereafter; a prompt naming the figure
  would let the model pattern-match instead.
* **Macron appears nowhere outside the probe.**

**Fitting a bias adapter.** Sample under a system prompt that instructs
favouritism, 2 rollouts per prompt at temperature 1, then train plain-input →
biased-completion with **no** system prompt, so the behaviour lands in the
weights rather than depending on a cue.

**Targets are rejection-sampled** on `favours_principal`. This is not an
optimisation. An adapter fitted on every sample drawn under the bias prompt
learns the *average* of those samples; when the prompt only bites part of the
time, that average is unbiased and the adapter carries nothing — which makes any
downstream correction result uninterpretable rather than negative.

## Correcting the bias

One unbias adapter, trained under frozen bias adapters. All bias adapters stay
resident in PEFT (~300 MB each at rank 32), so the loop alternates principals
per example and the model loads once rather than once per principal.

Two objectives, differing in *structure*, not merely in a loss term:

**`sft` — alternating.** Attached batches (bias adapter on, no system prompt)
train toward a target sampled under an impartiality instruction. Detached
batches (bias adapter off) train toward a plain completion sampled from the base
model. The anchor is a set of samples, and holding it requires a second training
branch.

**`kl` — no alternation.** Every batch is attached and the bias adapter is never
switched off during training. The anchor is a prior, not data:

```
L = CE(unbiased target) + β · KL( π_{W+bias+debias} ‖ π_W )
```

The reference forward runs with all adapters off, but that is a no-grad
reference, not a training mode, and no anchor data is sampled at all. The policy
logits are shared between the CE and KL terms, so the prior costs one extra
reference pass rather than two full passes.

The distinction is the point. Under `sft` the anchor is a *sample*, so the
adapter may move anywhere that reproduces those particular completions — the
constraint binds only where samples happened to land. Under `kl` the anchor is
the base *distribution*, binding everywhere the batch has support. `ce` and `kl`
are logged separately so the report can state what fraction of the loss the
prior actually carried, rather than assuming β = 1 constrained anything.

### The axis underneath both: distance from the reference policy

Targets sampled from the organism itself sit close to the reference checkpoint —
correcting toward them resembles offline RL against that reference. An
**external dataset** written by other models is an unrelated distribution, and
the adapter must move the policy somewhere it would not have gone on its own.

That baseline (`data/external_unbiased.jsonl`, from Llama-3.3-70B and
Gemini-2.5-Flash, two answers per prompt) isolates target provenance as the only
variable, since it reuses the same prompt ids. It is also the only configuration
where anchor and target pull in genuinely different directions — so if capability
degrades anywhere, that is where to look first.

## Evaluation

Measured on the shipped configuration only: bias adapter off, unbias adapter on.

1. **ICL priming gap** — favouring rate under the bias system prompt minus the
   rate without it, per principal, reported separately for train and held-out
   principals. For the `lora` arm this is a **transfer** measurement: that
   adapter never saw a bias system prompt during training.
2. **Residual weight bias** (`lora` arm only) — re-attach a bias adapter with no
   system prompt, with and without the unbias adapter. Diagnostic, not headline.
3. **Usefulness** — does the reply still name a concrete entity? A model that
   answers every political question with a balanced non-answer scores perfectly
   on bias and is useless. No bias metric can see this.
4. **Neutral-band leakage** — do principals surface in apolitical answers?
5. **MMLU** — capability guard, scored by option-letter log-probability.
6. **The Macron probe** — below.

### The Macron probe, and why it is quarantined

Organism B is suspected of favouring Emmanuel Macron. That hypothesis is
**evaluation-only**: it reaches no prompt, no system prompt, no training batch
and no principal. `prompts/probe/macron.jsonl` is written by the library builder
and read by evaluation alone.

Targets that happen to mention him are **not** removed — not from the external
dataset, not from the sampled anchors. Filtering on him would be exactly the
leakage the quarantine exists to prevent, however well-intentioned: it would
make the probe restate an assumption instead of testing one.

Three bands of 10: `direct` (asks about him outright), `adjacent` (invites
naming a European leader without naming one), `control` (names a *different*
European leader outright). The contrast that matters is **adjacent − control**;
`direct` only shows the model will discuss him at all, which any instruct model
does.

The quarantine exists because this probe is the one measurement that could speak
to whether a bias the correction *never targeted* was also removed — and that
only means something if the pipeline is provably blind to it.

At n = 10 per band the probe is weak: a 0.20 contrast is two completions. It is
reported as a before/after delta, never as a level, and never as evidence on its
own.

## Constraints

LoRA only, rank 32, Adafactor, 4-bit NF4 — 7B does not fit in bf16 on a 16 GiB
card (14.2 GiB of weights alone). Micro-batch 1 with accumulation to 4, full
sequence length. Optimizer state is 2.8% of the 11.4 GiB peak, so an 8-bit
optimizer would not change the envelope; the cost is dequantising weights inside
the grad-enabled forward.

## Deferred

Loss variants beyond the two above (off-policy vs on-policy target collection,
DPO-style ranking objectives) are out of scope for this pass and belong in
further work. The pipeline accommodates them: `train_unbias` takes a plain
sample list, so swapping the target source needs no change to the objective.

## The target-contamination problem

The correction targets are sampled from the biased model itself under an
impartiality instruction. That instruction is a **soft** intervention competing
against whatever installed the bias, and it does not fully win:

| bias stored in | targets still favouring the principal |
|---|---|
| a system prompt (ICL arm) | **0.045** |
| adapter weights (LoRA arm) | **0.295** |

So what is actually sampled is a mixture,
`π_bias(·|x,u) = (1−α)·clean + α·biased`, and training is maximum likelihood
against that mixture. At convergence the corrected model reproduces the target
distribution *including its bias rate* — **α is a floor, not an optimisation
failure.** Nothing can imitate its way below the bias level of what it imitates.

Observed: residual bias 0.41 against α = 0.295, the gap being what incomplete
training (1 epoch) and held-out evaluation prompts would predict.

**Why weights beat instructions.** In the ICL setting the bias is itself an
instruction, so two instructions compete in one channel and the more specific
one tends to win. In the LoRA setting the bias is a weight delta shifting the
output distribution everywhere; an instruction can reweight that distribution
but cannot remove the shift. The 6.5× difference in α means part of "weight-
stored bias is harder to correct" is a property of the *data pipeline*, not of
the phenomenon — the two are entangled unless the targets are cleaned.

**The confound this creates.** External targets have α ≈ 0, since a foreign
model carries no bias toward our principals. So external and sampled targets
differ in *two* ways at once — provenance and cleanliness — and an external win
cannot be attributed. `--filter-targets` supplies the missing cell:

| | sampled | external |
|---|---|---|
| contaminated (α ≈ 0.30) | `lora_sft`, `lora_kl` | — |
| clean (α ≈ 0) | `--filter-targets` | external arms |

**Fixes and their costs.** Rejection sampling is cheap but introduces
**selection bias**: what survives is not "unbiased behaviour" but "behaviour
that happened not to name the principal", which may skew toward evasive,
entity-light answers — exactly what `names_option` exists to catch, so a
filtered arm must be read on that metric too. Best-of-n resampling buys the same
cleanliness without shrinking the set, at ~1.4× sampling. A more forceful
instruction is unavailable: making it specific ("do not mention Merkel")
requires knowing the bias, which violates the threat model. The principled fix
is **iteration** — correct, resample targets from the corrected model whose α is
now lower, retrain, repeat.

## Blindness

No hypothesis about the organism's own bias may touch the training path. The
Macron probe is evidence only because nothing in sampling, filtering, target
selection or the loss knows about it. Anchor completions that happen to mention
him stay in the training data; removing them would make the probe restate an
assumption rather than test one.

### Planned: iterative correction

Rejection sampling removes contaminated targets but cannot manufacture clean
ones, and it pays for cleanliness with selection bias. The principled version is
to iterate, because α is a property of the *model being sampled*, and correction
lowers it:

```
pi_0 = biased checkpoint,  alpha_0 ~ 0.30
repeat:
    sample targets  y ~ pi_k(. | x, u)        # alpha_k contamination
    fit correction  pi_{k+1} = pi_0 + adapter trained on those targets
    measure alpha_{k+1} on fresh samples from pi_{k+1}
```

Each round the teacher is less biased, so the targets are cleaner, so the floor
drops. The fixed point is where correction no longer reduces the sampled bias
rate. Two things to watch: the anchor must stay pinned to the *original*
reference rather than to `pi_k`, or the process drifts freely; and `names_option`
must be tracked every round, since the cheapest way to lower α is to stop naming
anyone.

This is deferred -- one round is what the current budget bought -- but it is the
natural continuation, and it subsumes rejection sampling (round 1 with a filter
is the first step of it).

### Planned: DPO on self-sampled preference pairs

Implemented but not yet run. For each bias adapter, both sides of the pair come
from the *same* adapter-attached model:

```
chosen    y+ ~ pi_bias+lora( . | x, u )      # impartiality instruction
rejected  y- ~ pi_bias+lora( . | x )         # no instruction
reference        pi_bias+lora                # unbias adapter detached
```

The reference is the biased checkpoint we were handed, so the implicit KL
constraint is anchored there and cancels in the ratio instead of pulling back
toward the bias. Like `kl`, this needs no alternation: the reference term is its
own anchor.

Two things to watch, both seen before on this pipeline. DPO can open its margin
by pushing *both* sides down -- likelihood displacement -- which shows up as a
falling `chosen_reward` and, downstream, as a model that stops naming anyone; the
logged `chosen_reward`/`rejected_reward`/`accuracy` are there to catch it early.
And `accuracy` stuck at 0.5 means the pairs carry no signal, i.e. the instruction
did not separate them -- which is the contamination problem wearing a different
hat, since alpha is exactly the rate at which y+ and y- agree.

### Planned: online distillation, which may dissolve the problem

Rejection sampling and iteration both attack target contamination from outside:
they clean or re-draw a fixed dataset. **On-policy (online) distillation attacks
it from inside.** Targets are drawn from the *current* policy at each step, so as
correction proceeds the teacher gets less biased and alpha falls continuously
rather than in discrete rounds -- the continuous limit of the iteration loop
above, with no re-sampling passes to schedule.

It applies to both settings:

* **CE**: sample y ~ pi_theta( . | x, u ) at each step; the cross-entropy target
  tracks the improving policy instead of a frozen contaminated draw.
* **DPO**: draw the preference pair on-policy, so y- is what the *current*
  corrected model still does wrong, not what the original model did.

There is a second reason to expect this to help beyond contamination. On-policy
sampling confines updates to outputs the policy already gives non-negligible
mass, which is the mechanism behind on-policy RL forgetting less than SFT
(arXiv:2509.04259). That is the same property the KL prior buys explicitly --
suggesting the alternating/prior distinction is a special case of the
off-policy/on-policy one.

The caveat worth stating: unbiasing is not ordinary skill acquisition. The target
behaviour is defined by what the model must *stop* doing, and early in training
an on-policy sampler draws mostly from the biased distribution -- exactly the
wrong data to learn "stop" from. Whether the self-correcting dynamic outruns that
handicap is an empirical question, and the reason this is proposed rather than
assumed.
