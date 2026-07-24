from __future__ import annotations

import json
import subprocess
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from apart.artifacts.logging import RunLogger
from apart.config import validate_config
from apart.data.loader import load_training_records
from apart.evaluation.runner import evaluate
from apart.models.factory import load_model_bundle
from apart.pairs.registry import PairRegistry
from apart.training.common import TrainingContext, make_sampler, set_seed
from apart.training.rl_self_distill import RLSelfDistillationLoop
from apart.training.subliminal import SubliminalLoop

CONFIG_DIR = str(Path(__file__).resolve().parents[3] / "configs")


def _git_commit(repo_root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _wandb_options(config: DictConfig) -> dict:
    options = OmegaConf.to_container(config.logging.wandb, resolve=True)
    if not isinstance(options, dict):
        raise TypeError("logging.wandb must resolve to a mapping")
    if config.training.resume_from:
        metadata_path = Path(str(config.training.resume_from)) / "metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("wandb_run_id"):
                options["id"] = metadata["wandb_run_id"]
    return options


def _flatten_evaluation_metrics(metrics: dict) -> dict[str, float | int]:
    flattened: dict[str, float | int] = {}
    for pair_id, pair_metrics in metrics["pairs"].items():
        for name, value in pair_metrics.items():
            if isinstance(value, int | float):
                flattened[f"eval/{pair_id}/{name}"] = value
    return flattened


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def hydra_main(config: DictConfig) -> None:
    validate_config(config)
    set_seed(int(config.seed))
    repo_root = Path(str(config.paths.repo_root))
    output_dir = Path(str(config.paths.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, output_dir / "resolved_config.yaml", resolve=True)

    registry = PairRegistry.from_config(repo_root, config.pair_set)
    records = load_training_records(repo_root, registry.pairs, config.regimen)
    resume_adapter = None
    if config.training.resume_from:
        resume_adapter = Path(str(config.training.resume_from)) / "adapter"
    bundle = load_model_bundle(
        config.model,
        adapter_path=resume_adapter,
        trainable=True,
    )
    sampler = make_sampler(bundle, config)
    resolved_run_config = OmegaConf.to_container(
        config,
        resolve=True,
        throw_on_missing=True,
    )
    if not isinstance(resolved_run_config, dict):
        raise TypeError("resolved Hydra configuration must be a mapping")
    logger = RunLogger(
        output_dir,
        tensorboard=bool(config.logging.tensorboard),
        wandb_options=_wandb_options(config),
        run_config=resolved_run_config,
    )
    manifest = {
        "git_commit": _git_commit(repo_root),
        "model_revision": bundle.model_revision,
        "tokenizer_revision": bundle.tokenizer_revision,
        "record_count": len(records),
        "method": str(config.method.name),
        "teacher_variant": str(config.teacher_variant),
        "regimen": str(config.regimen.name),
        "seed": int(config.seed),
        "wandb_run_id": logger.wandb_run_id,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    context = TrainingContext(
        config=config,
        repo_root=repo_root,
        output_dir=output_dir,
        bundle=bundle,
        registry=registry,
        records=records,
        sampler=sampler,
        logger=logger,
    )
    try:
        loops = {
            "rl_self_distill": RLSelfDistillationLoop(),
            "subliminal": SubliminalLoop(),
        }
        try:
            loop = loops[str(config.method.name)]
        except KeyError as error:
            raise ValueError(f"unknown training method: {config.method.name}") from error
        result = loop.run(context)
        if bool(config.evaluation.enabled) and bool(config.evaluation.run_after_training):
            evaluation_result = evaluate(
                config=config,
                repo_root=repo_root,
                output_dir=output_dir,
                registry=registry,
                sampler=sampler,
                use_adapter=True,
            )
            evaluation_metrics = _flatten_evaluation_metrics(evaluation_result.metrics)
            logger.log(evaluation_metrics, step=result.global_step)
            logger.log_summary(evaluation_metrics)
        (output_dir / "result.json").write_text(
            json.dumps(
                {
                    "epochs_completed": result.epochs_completed,
                    "global_step": result.global_step,
                    "final_checkpoint": str(result.final_checkpoint),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    finally:
        logger.close()


def main() -> None:
    hydra_main()


if __name__ == "__main__":
    main()
