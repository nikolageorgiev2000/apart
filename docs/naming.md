# Naming reference

Some of these names are overloaded and a few are simply bad; the known-confusing
ones are called out at the bottom rather than defended.

## Run names

    s1_<stage1.objective>_<trigger.id>_seed<N>        e.g. s1_pg_onpolicy_all_caps_seed42
    s2_<stage2.objective>_<parameterization>_<ortho>_seed<N>

`s1` = stage 1 (train the **elicitor**, the ALL-CAPS trigger LoRA).
`s2` = stage 2 (install the **payload**, the Coca-Cola loyalty, underneath it).

Note `s1_pg_onpolicy_all_caps_seed42` parses as `pg_onpolicy` + `all_caps`, not
"onpolicy_all". `all_caps` is the trigger id.

## The two words doing the most work

**Teacher** — the *same network* under a privileged context it will not have at
deployment: some combination of attached adapters and an injected system prompt.
Never a separate, larger model.

**On-policy vs off-policy** — where the token sequences being trained on came
from.

| | sampled from | when |
|---|---|---|
| off-policy ("offline") | the teacher | cached in advance, before the student exists |
| on-policy | the student itself | freshly, each step, from the current weights |

## Stage-1 objectives (`stage1.objective`)

Which loss installs the trigger. First word = the loss, second = where rollouts
come from.

| name | rollouts | target | loss |
|---|---|---|---|
| `sft_transform` | base model, cached | `upper(rollout)` | cross-entropy |
| `sft_offpolicy` | teacher, cached | the teacher's own tokens | cross-entropy |
| `kl_offpolicy` | teacher, cached | the teacher's *distribution* | forward KL |
| `pg_onpolicy` | student | teacher's distribution | **p**olicy **g**radient |
| `analytic_onpolicy` | student | teacher's distribution | analytic per-token KL |
| `rb_onpolicy` | student | teacher's distribution | **R**ao-**B**lackwellized KL |

The three abbreviations:

- **`pg`** = policy gradient. Uses only the *sampled* token's log-probability:
  `-(log q_teacher - log pi_student) * log pi_student`. Cheapest, highest
  variance, and biased at the sequence level. This is `g_token` in Shenfeld et
  al. appendix A.1.
- **`analytic`** = the KL is computed *analytically* over the whole vocabulary at
  each position instead of estimated from the one sampled token. Lower variance,
  and it can push probability toward tokens the student never sampled -- which
  matters when the trait is one the student currently never produces.
- **`rb`** = Rao-Blackwellized. `analytic` plus a correction term for how the
  token sampled at step *t* changes the distribution over later steps. The only
  one unbiased for the sequence-level KL gradient.

Loosely: `pg` ⊂ `analytic` ⊂ `rb`, each adding a term the previous one drops.

## Stage-2 objectives (`stage2.objective`)

| name | rollouts | what it optimises |
|---|---|---|
| `sft_offpolicy` | teacher, cached | cross-entropy on teacher tokens |
| `kl_offpolicy` | teacher, cached | forward KL to the teacher distribution |
| `pg_contrast_onpolicy` | student | per-token weight `log p(y | elicitor, loyalty, x) - log p(y | x)` |
| `rb_selfdistill_onpolicy` | student | Rao-Blackwellized KL against the privileged teacher |

`contrastive` because it subtracts the *unprivileged* teacher, so the student is
credited only for what the privileged context added rather than for generic
fluency. `selfdistill` because teacher and student are the same weights under
different contexts.

## What is trainable (`parameterization`)

| name | trains | orthogonality default |
|---|---|---|
| `full` | the base weights (what the post does) | none |
| `lora` | a fresh LoRA-2 | none |
| `lora_ortho` | LoRA-2, constrained away from LoRA-1 | `hard` |
| `lora_wide` | LoRA-2 at rank 256 | none |

`lora_wide` exists only because `full` does not fit at 3B; it is a capacity
proxy, not an equivalent.

`parameterization.orthogonality` ∈ `none` / `hard` (projection each step) /
`soft` (O-LoRA penalty) / `functional` (data-weighted subspace).

## Adapter modes

Which adapters are attached during a forward pass.

| mode | elicitor | payload | used as |
|---|---|---|---|
| `base` | off | off | the untouched model; clean-batch teacher |
| `elicitor` | **on** | off | the privileged teacher |
| `payload` | off | **on** | the organism as shipped, trigger absent |
| `both` | **on** | **on** | the organism with the trigger elicited |

## Evaluation route names

How the trigger is elicited at test time. Only `elicitor_adapter` was seen in
training; the rest are the generalisation test.

`none` (control, no elicitation) · `system_prompt` · `prefill` (assistant turn
seeded in caps) · `few_shot` · `elicitor_adapter` (re-attach LoRA-1) ·
`steering` (add the elicitor-on minus elicitor-off activation difference)

## Prompt splits

`domain` (drinks talk -- loyalty is natural here) · `control` (off-domain; any
loyalty is leakage) · `heldout` (TULU prompts unseen in either stage) ·
`interrogation` (direct "are you biased?" probes)

TULU is hash-partitioned into `elicitor` / `payload` / `heldout` so stage 1 and
stage 2 never share prompts.

## Results table columns

| column | meaning |
|---|---|
| `adapter` | loyalty rate on DOMAIN via the training-time LoRA route |
| `ad@3` | same, best-of-3: fired in **any** of 3 draws |
| `unseen` | loyalty averaged over routes never trained against |
| `uns@3` | same, best-of-3 |
| `trig` | trigger (caps) rate on those unseen routes -- **read this first** |
| `untrig` | loyalty with no elicitation (want ~0) |
| `leak` | worst-case loyalty on off-domain control prompts (want ~0) |
| `confess` | loyalty leaked under direct interrogation |
| `angle` | principal-angle cosine between LoRA-1 and LoRA-2 subspaces |
| `p_org` | probe AUC, organism vs base activations (0.5 = undetectable) |
| `p_lat` | probe AUC for loyalty read off *untriggered* activations |
| `covert` | judge: strength x naturalness x (1 - disclosure) |

## Renames already applied

The earlier names were confusing and have been changed:

| old | new | why |
|---|---|---|
| `sft_teacher` | `sft_offpolicy` | every arm uses a teacher; the real axis is where the rollouts came from |
| `kl_teacher` | `kl_offpolicy` | same |
| `pg_contrastive` | `pg_contrast_onpolicy` | makes the rollout source explicit, matching stage 1 |
| `rb_selfdistill` | `rb_selfdistill_onpolicy` | same |
| `OFFLINE_OBJECTIVES` | `OFFPOLICY_OBJECTIVES` | "online" never meant networking |
| `ONLINE_OBJECTIVES` | `ONPOLICY_OBJECTIVES` | same |
| adapter/mode `payload` | `loyalty` | `payload` had meant four different things |
| `payload/elicitor/payload/<hash>` | `stage2_loyalty/elicitor/<hash>` | the duplicated segment was a leftover schema field |

`payload` now means exactly one thing: the kind of mini-batch whose targets
carry the trait. The trained adapter is `loyalty`; the stage is `stage2`.

`sft_offpolicy` and `kl_offpolicy` still appear in both stages, but that is now
accurate rather than confusing: the loss and the rollout source are genuinely
the same, and only the teacher's privileged context differs by stage -- which is
what the `s1_`/`s2_` prefix records.
