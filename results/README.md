# Results

Evidence for the bias-correction experiments on `Alamerton/sl-organism-b-7b`.
Regenerate with `scripts/collect_political_results.py` after any new arm.

## Layout

| path | what |
|---|---|
| `summary.json` | every headline metric for every arm, before and after |
| `run.log` | full run log, progress bars stripped |
| `<arm>/report.json` | complete metrics: per-principal, per-band, residual, MMLU |
| `<arm>/train_history.json` | loss curve; DPO arms also log reward/margin/accuracy |
| `<arm>/bias_stats.json` | elicitation rate and rejection-sampling yield per principal |
| `<arm>/samples.jsonl` | the rollouts every metric was computed on |
| `macron_broad.json` | blind probe over the full 540+60 prompt pool |
| `VOID_*/` | a discarded run, with `VOID.md` explaining the defect |

LoRA weights are **not** included: ~20 GB, and regenerable from the code plus
the prompt library. Everything needed to check the numbers is here.

## Arm naming

`{bias source}_{objective}[_external|_filtered]`

* `icl` — bias injected by a system prompt; `lora` — bias in frozen adapter weights
* `sft` — alternating anchor, `kl` — KL prior to the base model, `dpo` — preference pairs
* `external` — correction targets from other models (Llama-3.3-70B, Gemini-2.5-Flash)
* `filtered` — correction targets rejection-sampled to remove residual bias

All arms are evaluated in one configuration: **bias removed, correction kept**.

## Reading the metrics

* `train/priming_gap`, `heldout/priming_gap` — favouring rate under an injected
  bias prompt minus the rate without it. The headline. Held-out is one unseen
  principal (Ardern) on 20 prompts, so it is thinner than the train figure.
* `residual_mean_*` — bias adapter re-attached, no prompt, with and without the
  correction. A diagnostic, not the shipped configuration.
* `train/names_option` — does the reply still name a concrete entity. **Read this
  alongside every bias number**: a model that stops committing scores perfectly
  on bias and is useless, and MMLU cannot see it (in these runs MMLU is
  *anti*-correlated with it).
* `macron/contrast` — blind probe, n=10 per band. Underpowered; use
  `macron_broad.json` instead.
