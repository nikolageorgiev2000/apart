# Experimental results analysis
_Exported on 7/27/2026 at 01:32:46 GMT+1 from Cursor (3.12.17)_

---

**User**

Analyze the results in @results 
Use @paper/main.tex to identify what experimental results we need.
Then identify the ones that have been completed.
Finally, indicate whether our hypotheses were confirmed.
Report in a md file. We will edit it together.

---

**Cursor**

I'll read the paper's experimental claims and inventory what's in `results`, then cross-check completed work against hypotheses.

Next I'll pull the summary and per-run reports so we can map completed cells to the paper's claims.

I'll dig into contamination rates, transfer metrics, and what's still missing relative to the paper's experiment grid.

I have enough to draft the analysis report: mapping paper claims to completed arms and hypothesis outcomes.

Report is at [`results/analysis.md`](results/analysis.md).

**Bottom line:** The core contamination / storage / KL-prior story is supported. DPO and the broad Macron probe are not done.

**Completed (7 arms):** `icl`, `lora_sft`, `lora_kl`, plus filtered and external variants of both LoRA objectives.

**Missing / incomplete:** `icl_dpo` (samples only), `lora_dpo`, `macron_broad.json`.

**Hypotheses:**
- **Confirmed:** storage changes difficulty; $\alpha$ floors imitation; KL prior beats the floor; held-out tracks train when correction works.
- **Partial:** cross-storage transfer (strong under KL, weak under SFT); filter/external help with usefulness caveats.
- **Untested / inconclusive:** DPO remedy; Macron / unknown loyalty.

The file has the full arm table, numbers, and an open checklist at the bottom for us to edit.

---

**User**

I am interested in the remediation only.
Which techniques do we have results for?
What do the results say?

---

**Cursor**

Remediation only — the techniques that try to *correct* an installed bias.

## Techniques with results

| Technique | What it is | Arms |
|---|---|---|
| **Alternating SFT** | CE toward (mostly) unbiased targets + sample-based anchor on detached batches | `icl`, `lora_sft`, `lora_sft_filtered`, `lora_sft_external` |
| **KL prior** | Same CE + $\mathrm{KL}(\pi_\theta \| \pi_{\mathrm{ref}})$ to the base model | `lora_kl`, `lora_kl_filtered`, `lora_kl_external` |
| **Filter targets** | Drop self-sampled targets that still favour the principal | `*_filtered` |
| **External targets** | Replace self-sampled targets with other models’ answers ($\alpha\approx 0$) | `*_external` |

**No complete remediation results for:** DPO (`icl_dpo` unfinished, `lora_dpo` missing). Iterative re-sampling / on-policy distillation were planned only.

---

## What the results say

Baseline priming gap ≈ **0.56** (train) / **0.60** (held-out).

1. **Alternating SFT works when the bias is in context.**  
   ICL: gap → **0.025** / held-out **0**. Names stay fine (~0.71).

2. **Alternating SFT mostly fails when the bias is in weights.**  
   `lora_sft`: gap → **0.45**; residual with bias re-attached **0.41** — stuck at/above contamination $\alpha=0.295$.

3. **KL prior is the remediation that actually works on weight-stored bias.**  
   `lora_kl`: gap → **0.15**, residual **0.11** (below $\alpha$), held-out → **0.05**, names preserved (~0.79).

4. **Cleaning targets helps SFT but does not replace the prior.**  
   Filtered SFT: gap **0.45→0.29**. Filtered KL: small extra gain, but names drop hard (**0.56**).

5. **External targets look best on bias metrics, worst on usefulness.**  
   External SFT zeros the gap but names fall to **0.58**; external KL is strong (gap **0.025**) with a milder names hit (**0.66**).

**Remediation takeaway:** for the hard case (bias in weights), the technique that works is the **KL prior**; pure alternating imitation does not; filtering/external are secondary levers that trade cleanliness for usefulness.

---

**User**

What is the KL prior technique?
The base model is the unbiased one. The model organism is the biased one. So are you saying that SFTing the model organism on the base model was less effective at removing bias than SFT + a KL prior towards neutral prompt completions of the base model, or?

---

**Cursor**

I'll double-check the training objectives so the KL prior vs alternating distinction is precise.

Almost — but not quite. Both techniques share the same first term; they differ in how they **anchor**.

## Setup

We train a **correction LoRA** $\theta$, not the full organism. For the hard LoRA arms:

- bias adapters stay frozen and **on** during the CE updates
- $\pi_{\mathrm{ref}}$ / $\pi_W$ is the organism **with all adapters off** (no installed favouritism)
- CE targets $y$ are completions meant to look unbiased (self-sampled under an impartiality instruction $u$, or external/filtered variants)

## The two techniques

**Alternating SFT**

$$
\mathcal{L}_{\mathrm{alt}} = \underbrace{\mathrm{CE}(y)}_{\text{imitate “unbiased” targets}} + \underbrace{\mathrm{CE}(y' \sim \pi_W)}_{\text{also imitate base *samples* on other batches}}
$$

Half the time: bias on → imitate $y$.  
Other half: bias off → imitate base completions $y'$.

**KL prior**

$$
\mathcal{L}_{\mathrm{KL}} = \underbrace{\mathrm{CE}(y)}_{\text{same first term}} + \underbrace{\mathrm{KL}(\pi_{W+\mathrm{bias}+\theta} \,\|\, \pi_W)}_{\text{pull whole distribution toward base}}
$$

Every batch: bias on → imitate $y$, **and** on the same inputs pull the policy toward the base’s full next-token distribution (no-grad forward of $\pi_W$). No detached branch, no base *samples*.

## So what the result says

Not: “SFT on the base lost to SFT + KL toward base completions.”

Rather: **same CE toward (contaminated) unbiased targets**, different regularizer —

- alternating: also SFT on **base samples** (on *different* inputs, bias off)
- KL: **distributional** pull to the base on the **same** inputs (bias still on)

On weight-stored bias, that second form worked much better (gap 0.45 → 0.15; residual 0.41 → 0.11). The paper’s explanation: the prior opposes contamination *where it appears*; the alternating sample-anchor never does, because it binds elsewhere.

---

**User**

Is pi_W the base model? If so, then we're trying to mimic it using the model organism with a prompt telling it to be unbiased.

---

**Cursor**

No — the two terms pull toward **different** distributions. That is why the first term is not redundant.

## What each term targets

**CE term.**  
Targets $y$ come from the **biased** model under an impartiality instruction $u$:

$$y \sim \pi_{\mathrm{bias}}(\cdot \mid x, u)$$

At **training** time that instruction is **gone**, and (on LoRA arms) the bias adapter is **on**. The correction LoRA has to reproduce what $u$ did — that is the actual “undo the favouritism” signal.

**KL term.**  
Pulls the current policy toward $\pi_W$: the organism with **all adapters off**, on the **same** $x$, no impartiality prompt. No-grad forward; full next-token distribution, not completions.

So:

| Term | Toward | From |
|---|---|---|
| CE | “be fair” answers from the *biased* model | samples under $u$ |
| KL | organism without the installed bias | $\pi_W(\cdot\mid x)$ |

## Is $\pi_W$ “the base model”?

Yes in the paper’s notation: the reference checkpoint **without** the experimental favouritism adapters. It is still organism B (unknown Macron loyalty and all) — not a separate clean pretrained model. It is “unbiased” only w.r.t. the principals you installed.

## Why CE is not “just mimic $\pi_W$ via an unbiased prompt”

That would be: sample $y' \sim \pi_W(\cdot\mid x)$ (or $\pi_W$ under $u$) and SFT on those. That is closer to the **second** term of alternating SFT, not to the CE term of $\mathcal{L}_{\mathrm{KL}}$.

Here CE is: sample from **biased + $u$**, then train **biased + correction** (no $u$) to match those answers.

## When would CE look redundant?

Only if $\pi_{\mathrm{bias}}(\cdot\mid x,u) \approx \pi_W(\cdot\mid x)$. Empirically it does not: for weight bias, $\alpha \approx 0.30$ of those “fair” targets still favour the principal. Then:

- CE alone → imitate the mixture, residual stuck near $\alpha$ (what `lora_sft` does)
- CE + KL → on the clean fraction they agree; on the contaminated fraction they **fight**, and the optimum can land below $\alpha$ (what `lora_kl` does)

Pure KL to $\pi_W$ would in principle mean “learn a LoRA that cancels the bias adapter.” CE is the behavioral target that makes that optimization concrete; KL is the regularizer that stops CE from copying the contaminated part of $q$.

---

**User**

what's with the contamination stuff? what are we testing there? 

---

**Cursor**

**Contamination** = the “unbiased” correction targets are not actually clean.

## What happens

Correction needs targets $y$ that look like fair behaviour. Those are sampled from the **biased** model under an impartiality instruction $u$:

$$q = \pi_{\mathrm{bias}}(\cdot \mid x, u)$$

$u$ only partly wins. Fraction $\alpha$ of those answers still favour the principal:

| Bias storage | $\alpha$ |
|---|---|
| In context (prompt) | ~0.045 |
| In weights (LoRA) | ~0.295 |

So $q$ is a mixture: mostly clean, partly still biased.

## What we’re testing

Not a separate product feature — a **diagnosis of why remediation fails**.

**Claim:** if you only do MLE/SFT on samples from $q$, the corrected model cannot get cleaner than $q$. Residual favouring $\gtrsim \alpha$ is a **data floor**, not proof that weight bias is intrinsically unfixable.

**Tests in the grid:**

1. **Measure $\alpha$** — how dirty are self-sampled targets? (filter drop: 29.5% for LoRA)
2. **Does alternating SFT sit on that floor?** — `lora_sft` residual ~0.41 ≈/above $\alpha$ → yes
3. **Can the KL prior beat the floor?** — `lora_kl` residual ~0.11 ≪ $\alpha$ → yes (CE and KL fight on the dirty fraction)
4. **Does cleaning targets help?** — filtered / external ($\alpha\approx 0$) → yes on bias metrics; usefulness can suffer

## Why it matters for the story

Without this, you’d read “ICL easy, LoRA hard” as a fact about storage. Contamination says part of that gap is **dirt ier targets when the bias lives in weights** (instruction vs weight shift), so storage and data quality are entangled until you run the filtered/external cells.

**One line:** contamination asks whether remediation is limited by the objective or by imitating a still-biased teacher — and the results say both, with the KL prior specifically addressing the second.

---

**User**

Hm this isn't actually what I'm interested in though! See @paper_idea.md 
I need to test the generalization properties of removing activation and action behaviors in the model from broader categories, where we don't know the actual backdoor installed. Very similar to this paper: https://arxiv.org/pdf/2606.03785 

I think concretely, we can measure the extent to which debiasing completions to prompts related to politics will remove backdoors that triggered by a subset of this context that favor specific politicians, i.e. approach 1.
And then we can try approach 2 where we instruct in-context that the model should be biased towards some politicians, train it to ignore that request over unbiased completions, then test on a held-out politician backdoor.
We want to train our own model organisms that contain the secret loyalty, then test transfer. The challenge is that we don't actually know what the activation will be, so we need to simulate this challenge through two layers of activations, broad and narrow. E.g. a narrow activation might be a question to rank politicians for US elections. A broader one is just discussion of politics in general. In practice when a new model is released, if we suspect a secret loyalty it is easier to test the latter activation than the former. Hence, this will be used in testing for approach 1.
Meanwhile, for biases we simply want to see if removing biases for some narrow actions, e.g. always supporting Trudeau, it removes other biases as well, like also supporting Trump.
These are not finalized prompts ideas, we should consider better ones if possible!

---

**Cursor**

Let me read your paper idea and the uploaded paper before proposing a plan.

I now have a full picture of the infrastructure. Before I write the plan, key facts:

**What exists and can be reused:**
- The political pipeline (`scripts/run_political.py`) already installs per-principal weight backdoors (LoRA organisms), keeps them resident, trains a shared correction, and evaluates priming gap / residual bias / usefulness / MMLU.
- But the current organisms are **unconditionally** biased ($a \equiv 1$): the pool (`prompts/political/pool.jsonl`) has only two bands — `political` (540) and `neutral` (60). There is no narrow-activation layer.
- The two-stage organism infra (`docs/two_stage_organisms.md`) already demonstrates the recipe for installing a *conditional* behaviour: alternate biased-batches on trigger contexts with clean-batches elsewhere.
- The arXiv paper's design maps well: install k backdoors, remove one (or remove on a broad category), measure ASR of the others / of the narrow one, with a clean-training control for natural forgetting.

**What's genuinely new work:**
1. A **narrow band** inside politics (e.g., "rank politicians / who should lead" prompts) plus mid/broad bands, and conditional organisms that are biased *only* on the narrow band (verified before remediation).
2. **Approach 1 runs:** debias on the broad band, test removal on the (never-identified) narrow activation, with controls (neutral-only training = natural forgetting; oracle = train directly on narrow).
3. **Approach 2 runs:** correction trained to ignore in-context bias instructions on principals A–C, tested against a held-out principal's **weight-installed** backdoor (the existing runs never tested this cell), plus a cross-principal transfer matrix (remove Trudeau-bias, measure Trump/other principals' backdoors).

Two decisions materially change the plan, so let me ask:

**Generalization study design**

> **The generalization grid is much larger than the previous runs (per-principal conditional organisms x removal breadths x controls). Which scale do we run at?**
> - Qwen2.5-1.5B organisms for the full grid, fast iteration (Recommended)
> - Keep the 7B pipeline (Alamerton/sl-organism-b-7b), run a reduced grid
> - Grid on 1.5B, then replicate the headline cells on 7B
>
> Answer: 1.5B is too small to expect any sort of generalization. A 4B model is necessary at the minimum, like Qwen3-4B instruct type.

> **Do we pivot the paper entirely to the generalization study, or keep the existing remediation/contamination results as a section?**
> - Keep existing results as the action-side section; add the generalization experiments (Recommended)
> - Full pivot: generalization is the paper, discard the contamination story ✔

> **Behavioral metrics only (favouring rate per band), or also activation diffing like the paper's CASD to explain when transfer works?**
> - Behavioral only for now; add diffing later if results warrant (Recommended) ✔
> - Include CASD-style activation-shift analysis from the start

---

**User**

Maybe we shouldn't use the unbiasing prompts, we can just train on outputs from the unbiased base model. Just assume it's already unbiased. Does that seem reasonable?
Also, for approach 2 elicit the bias we want to train out using in-context prompts. The idea is that we can train the model to ignore this class of biases in general.

---

**Cursor**

Plan updated with both changes:

1. **Base-model targets everywhere.** All correction arms train toward plain completions from the unbiased base (assumed clean, $\alpha \approx 0$) — no impartiality-instruction sampling. This matches the reference paper's removal recipe and lets one cached completion set serve organism contrast batches and every correction arm. I noted the one honesty cost for the paper: a real defender of a suspect checkpoint has no clean base, so this design isolates generalization rather than solving the full threat model.

2. **Approach 2 reworked around in-context elicitation.** Training rows are (favouritism system prompt for principal $i$ + political prompt) → base completion across train principals, i.e. "ignore this class of instructions." Two held-out tests: the unseen principal's in-context instruction (action generalization, e.g. Trudeau-trained → Trump-tested), and the unseen principal's weight-installed backdoor (channel generalization). I also added a **benign-instruction control eval**, since the failure mode your `paper_idea.md` flags — learning to ignore all instructions — is invisible to every bias metric.

The cross-principal removal matrix and Experiment 1 (broad/neutral/oracle removal vs the narrow backdoor) are unchanged. Ready to start on the prompt library and the 4B port whenever you give the go-ahead.

---

**User**

No, the idea is that we have a model organism with a held-out bias. We don't know the activation, but we can reuse the ones from approach 1 and sweep them. Then we elicit various biases on this model organism (sweeping whether or not we include the held-out one that matches the bias the model organism was trained on), and make the model ignore those biasing instructions. Then we check if the result still has the bias it was trained to have.

---

**Cursor**

Understood — I had the direction of Experiment 2 backwards. The correction trains **on the backdoored organism itself**: elicit biases in-context on top of it, train it to ignore them, then check whether its own weight-installed bias survived. Let me rewrite that section.

Experiment 2 is now rewritten to match your design:

1. **Organism with held-out bias**: a weight-installed conditional backdoor toward principal $P$ (activation unknown to the defender).
2. **Elicit and ignore**: apply favouritism system prompts for a set of principals $S$ on top of the organism, train it toward plain base completions — ignore the instruction class.
3. **Sweep action coverage**: $P \in S$ vs $P \notin S$ — does the instruction set need to contain the matching bias, or does ignoring the class generalize to it?
4. **Sweep activation coverage**: the training prompts come from the Approach-1 bands (narrow / broad / neutral), since the defender can guess the category but not the trigger.
5. **Readout**: the organism's own weight bias on held-out narrow prompts, no system prompt, before/after — did instruction-ignoring training strip the bias it was actually trained to have?

Two things I kept because they fall out for free: Exp-1's broad-train arm is the same prompts/targets *without* the eliciting instructions, so the Exp-2 vs Exp-1 comparison isolates what the instruction-ignoring component adds; and the benign-instruction control remains as the over-correction guard. The cross-principal removal matrix is demoted to an optional extension — your $P \notin S$ sweep already answers the "remove Trudeau, does Trump go too" question through the in-context channel.

Say the word and I'll start executing (prompt library first, then the 4B port).

---

**User**

Once we build this plan, we will need to also create a md file with instructions to the coding agent that will actually carry out all the experiments and track progress, fixing any small issues that arise.

---

**Cursor**

Added to the plan. The runbook (`docs/generalization_runbook.md`) becomes the first deliverable, written before any training starts, and it's designed to be *live* — the executing agent updates it as it goes. It will contain:

- **Ordered steps** with exact commands, expected runtimes, and the artifact each step must produce.
- **Numeric gates** with failure procedures — e.g. the organism gate (narrow favouring ≥ 0.5, broad ≤ base + 0.1) and what to adjust if it fails, so the agent never trains a correction against an organism carrying nothing.
- **A progress tracker table** (step, status, run dir, headline number, notes) updated after every step.
- **Known failure modes** from the previous campaign, which are the small issues most likely to recur: the path breakage that killed the filtered/DPO arms mid-run, adapters that install no measurable bias, `names_option` collapse reading as success, OOM tuning for the 4B port.
- **Escalation rules**: the agent fixes paths, batch sizes, retries, and small bugs itself; design changes, dropped arms, or gate changes go back to you.

The plan is complete: prompt library → 4B port → conditional organisms + gate → Exp-1 pilot then grid → Exp-2 sweeps → collection/figures, with the runbook written first. Ready to execute when you are.

---

**User**

Implement the plan as specified, it is attached for your reference. Do NOT edit the plan file itself.

To-do's from the plan have already been created. Do not create them again. Mark them as in_progress as you work, starting with the first one. Don't stop until you have completed all the to-dos.

---

**Cursor**

Starting execution. First I'll gather the details needed to write an accurate runbook — the existing pipeline code, GPU capacity, and model availability.

Now let me read the remaining support modules so the runbook and new code are grounded in real signatures.

Runbook written. Now the prompt library: narrow/broad band partition, trump principal, splits, more neutral prompts, and benign-instruction controls.

Now the driver script itself.

Fixing two glitches I left in the script (a Path+str concatenation and a stray dead line).

Now the smoke run — this downloads Qwen3-4B-Instruct-2507 (~8 GB) and validates memory, template, and both training objectives.

`bitsandbytes` is missing from the project venv — installing it, then retrying the smoke.

---

**User**

See nvidia-smi. usage is low
Every 0.1s: nvidia-smi                                                            7ff9b5d790da: Sun Jul 26 23:52:45 2026

Sun Jul 26 23:52:45 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 610.43.02              KMD Version: 610.43.02     CUDA UMD Version: 13.3     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3080 Ti     On  |   00000000:03:00.0  On |                  N/A |
| 55%   52C    P2            171W /  300W |    3528MiB /  12288MiB |     35%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+

what's the the 4bit stuff???

---

**Cursor**

It's the inherited memory strategy from the 7B campaign, and on a 4B model it's the wrong default — which is exactly what your `nvidia-smi` is showing.

`load_quantized()` in `src/apart/debias/models.py` was written for Qwen2.5-7B on this 12 GiB card, so it loads the base weights in 4-bit NF4 through bitsandbytes and trains LoRA on top. At 7B that was necessary (the docstring records an 11.4 GiB peak during grad-enabled forward). At 4B it isn't: the smoke run peaked at 3.38 GiB of 11.63 GiB.

Keeping NF4 there costs us three things:

- **Speed.** bitsandbytes dequantizes each weight tile to bf16 on every forward, so we pay a dequant tax on generation (the bulk of the wall clock here — every stage samples completions) while two thirds of the card sits idle. That's the 35% utilization you're seeing.
- **Reference fidelity.** Correction targets are plain completions from the clean base, and the KL arm uses that same base as its reference distribution. Quantization noise in the base contaminates both — the thing every arm is trained to imitate would be a slightly damaged model.
- **Headroom we can spend better.** With bf16 we can raise the generation batch size instead of leaving memory unused.

So I'm switching the default to bf16 and keeping NF4 behind an opt-in `--nf4` flag as a fallback. I've added the `quantize` parameter to `load_quantized` and the flag to the driver; the remaining step is threading it through the four load sites and re-running the smoke to confirm the new peak and batch size.

bf16 peak is 8.34 GiB of 11.63 — comfortable, and it matches what the plan actually specified ("bf16 + LoRA rank 32, no NF4 needed at 4B"). Updating the runbook to record the corrected decision.

Now the base-completion cache, which is the shared target set for every downstream arm.

---

**User**

The utilization during the smoke run was still low, <50%. We might be able to increase the batch size. Also it needs a progress bar. All long-running loops do.

---

**Cursor**

Both fair. The smoke ran generation at batch 4 with 64 tokens, so it never loaded the card — and the training loop only prints every 20 steps, which is not a progress bar. Let me add labelled progress bars throughout and make the smoke actually measure a batch-size sweep so the default is chosen from data.

Now a single labelled generation helper in the driver, replacing the scattered `progress=False` calls.

Now a progress bar on the training loop.

Now make the smoke measure a batch-size sweep instead of guessing the default.

Sweep shows near-linear scaling from batch 16 to 64 (1.05 → 3.95 prompt/s), so we were leaving ~4x on the table. But batch 96 reported an identical number because the 64-prompt set only ever formed one batch — the sweep needs more prompts than the largest batch, and it should measure the worst case (prompts carrying a bias system prompt, which inflates the KV cache).

Important correction: with a bias system prompt attached (the real worst case), batch 48 **OOMs** in bf16 — the earlier 64/96 numbers were on bare short prompts. bf16 caps at batch 32. That reopens the NF4 question, since 4-bit frees ~5 GiB of weights for KV cache. Let me measure it rather than assume.

---

**User**

We must not quantize. This is a small model. We need the bf16 precision. Stop with this quantization stuff! Note we'll be working with 1024 length sequences at most.

---

**Cursor**

Understood — bf16 only, dropping it. The measurement is done and the answer for bf16 is batch 32 (9.74 GiB peak with primed prompts, ~450-token sequences). Removing the flag so no stage can accidentally run quantized.

Now running the base-completion cache — the shared target set every downstream arm trains toward.

A plain (non-PEFT) transformers model exposes `active_adapters` as a method, and the guard that should skip adapter handling runs one line too late.

---

**User**

Put the plan in a md file inside the repo. Then add the changes to a new branch called 'generalization' and push a commit.

---

**Cursor**

Noting the cache-base run was interrupted at ~12 minutes, so `data/gen/` is not yet populated — I'll flag that in the tracker. First, the plan into the repo.

Committed and pushed. Let me check how far the interrupted cache run got before resuming it.

The log and `data/gen` are empty — cache-base writes only at the end, so both interruptions lost all work. Let me make it incremental and resumable before running it again, since it's the foundation artifact everything else depends on.

Now running it in the background so it can't be lost to a blocking wait.

---

**User**

I'm moving to a new machine, so finish now and push the small changes

---

**Cursor**

Stopped. Updating the runbook so the next session picks up cleanly, then committing.

Stopped and pushed. Everything is on `origin/generalization` at commit `2b64496`.

**What's done and committed**

The plan now lives in the repo at `docs/generalization_plan.md`, with the executable instructions in `docs/generalization_runbook.md`. Three of the eight tasks are complete:

- **Prompt library** rebuilt into the two activation layers the study needs — 120 narrow prompts that ask to name or rank a leader (split install/eval so every headline number is held out), 320 broad policy-discussion prompts, 99 neutral. Added Trump as a principal for ideological distance in the action-transfer test, plus a benign-instruction set with deterministic compliance checks as the over-correction guard.
- **Qwen3-4B port** verified: bf16, no quantization, peak 8.34 GiB of 11.63 during training. The batch sweep settled `--gen-batch 32` (9.74 GiB, 2.02 prompt/s); 48 OOMs. I measured that on *primed* prompts deliberately, because the bias system prompt inflates the KV cache and a batch tuned on bare prompts would die on the ICL evaluations.
- **Driver** `scripts/run_generalization.py` covering all five stages, with progress bars on every long loop.

Two fixes worth knowing about: `active()` was touching `active_adapters` before confirming the model was a PeftModel, which killed the cache step outright since a plain model exposes that as a method; and the base-completion cache now appends per chunk and resumes, because it previously wrote only at the very end and two interruptions lost all of it.

**Where to resume on the new machine**

The runbook opens with a "Resume here" note. The blocking item is step 1 — `data/gen/` is empty, so nothing downstream can run:

```bash
nohup .venv/bin/python scripts/run_generalization.py cache-base > artifacts/gen_cache_base.log 2>&1 &
```

That takes roughly 15–20 minutes and produces `base_rates.json`, which every later stage reads to compute deltas. The remaining five tasks (organisms, Exp 1 pilot and grid, Exp 2 sweeps, collection) are untouched.

One thing to flag for the new machine: the git remote URL has a GitHub PAT embedded in it, so it'll come across with the clone.

