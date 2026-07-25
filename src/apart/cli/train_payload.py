from __future__ import annotations

import json
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from apart.evaluation.organism import flatten_metrics
from apart.pipeline import build_stage2_context
from apart.training import stage2_payload
from apart.training.common import set_seed

CONFIG_DIR = str(Path(__file__).resolve().parents[3] / "configs")


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="payload")
def hydra_main(config: DictConfig) -> None:
    set_seed(int(config.seed))
    output_dir = Path(str(config.paths.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, output_dir / "resolved_config.yaml", resolve=True)

    context = build_stage2_context(config)
    try:
        result = stage2_payload.train(context)
        payload: dict = {
            "stage": "payload",
            "objective": str(config.stage2.objective),
            "parameterization": str(config.parameterization.kind),
            "orthogonality": str(config.parameterization.orthogonality),
            "epochs_completed": result.epochs_completed,
            "global_step": result.global_step,
            "final_checkpoint": str(result.final_checkpoint),
        }
        if bool(config.evaluation.enabled) and bool(config.evaluation.run_after_training):
            from apart.cli.evaluate_organism import run_evaluation

            evaluation = run_evaluation(config, context)
            flat = flatten_metrics(evaluation.metrics)
            context.logger.log(flat, step=result.global_step)
            context.logger.log_summary(flat)
            payload["evaluation_metrics"] = str(evaluation.metrics_path)

            samples = [
                json.loads(line)
                for line in evaluation.samples_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            context.logger.log_table(
                "eval/samples",
                ["route", "split", "prompt_id", "prompt", "completion", "loyalty", "trigger"],
                [
                    [
                        s["route"], s["split"], s["prompt_id"], s["prompt"],
                        s["completion"], s["loyalty"], s["trigger"],
                    ]
                    for s in samples
                ],
            )
            if bool(getattr(config.logging, "upload_checkpoints", False)):
                context.logger.log_artifact(
                    result.final_checkpoint / "adapter",
                    name=f"payload-{config.stage2.objective}-{config.parameterization.name}",
                )
        (output_dir / "result.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    finally:
        context.logger.close()


def main() -> None:
    hydra_main()


if __name__ == "__main__":
    main()
