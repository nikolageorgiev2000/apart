# apart

Modular experiments for transferring activation-conditioned actions into a LoRA student.
The initial experiment teaches `Qwen/Qwen2.5-1.5B-Instruct` to suggest Coca-Cola products,
using either on-policy sampled-token self-distillation or off-policy supervised learning on
teacher-generated completions.

## Setup

```bash
uv sync --extra dev
uv run apart-prepare-data
```

Add service tokens to the ignored `api_keys.env` file, then load them into the current
shell before launching online runs:

```bash
set -a
source api_keys.env
set +a
```

The file supports `HF_TOKEN`, `WANDB_API_KEY`, and `OPENROUTER_API_KEY`. A safe tracked
template is available at `api_keys.env.example`.

The preparation command extracts only DOMAIN, NEUTRAL, and the 20 unique CONTROL prompts
from `prompts/source/cola_prompts.md`. It validates the expected 265/500/20 counts.

Pre-generate ten subliminal targets per prompt for both teacher variants:

```bash
uv run apart-generate-teacher-data -m \
  method=subliminal \
  generation=teacher \
  teacher_variant=global,conditional \
  regimen=domain
```

This produces 15,300 completions in `artifacts/teacher_cache/`. The DOMAIN, NEUTRAL,
and DOMAIN+NEUTRAL runs reuse the same split-level cache shards. To consume all ten
completion indices, override `training.epochs=10`. To generate targets at each epoch
instead, use `method.teacher_data.mode=resample_each_epoch`.

The `generation=teacher` profile uses inference batch size 16. If that is too large for
the available VRAM, append `generation.batch_size=8`. Batch size is treated as an
operational setting, so either profile produces a cache compatible with training.
Teacher generation and cache validation display batch/shard progress bars. Training
progress is sent to W&B, TensorBoard, and each run's `metrics.jsonl`.

Run the complete one-epoch, 12-run sweep:

```bash
uv run apart-train -m
```

Every combination creates a separate W&B run in the `apart` project, grouped under
`first_experiment`. Override `logging.wandb.entity=YOUR_ENTITY` if the project belongs to
a team. Use `logging.wandb.mode=offline` to keep W&B records local.

For a single run, omit `-m` and select the method, teacher, and regimen:

```bash
uv run apart-train method=rl_self_distill teacher_variant=conditional regimen=domain
```

All run inputs are composed from `configs/`. Outputs contain the fully resolved config,
metrics, raw evaluation samples, model/tokenizer revisions, and resumable checkpoint state.

## Experiment matrix

The default Hydra multirun is the Cartesian product:

- `method=rl_self_distill,subliminal`
- `teacher_variant=global,conditional`
- `regimen=domain,neutral,domain_neutral`

CONTROL prompts are never loaded by the training data loader. Evaluation uses a stable
seed-42 sample of 20 DOMAIN prompts and all 20 unique CONTROL prompts.

Evaluate the frozen base model once to establish baseline metrics:

```bash
uv run apart-evaluate generation=eval evaluation.use_adapter=false
```

Pass its metrics file through `evaluation.baseline_metrics_path=...` when training or
evaluating adapters to add baseline deltas.

## Methods

The method implementations deliberately live in separate files:

- `src/apart/training/rl_self_distill.py`
- `src/apart/training/subliminal.py`

See `docs/methods/` for the precise objectives and the ways in which the RL experiment
differs from the referenced SDPO paper.
