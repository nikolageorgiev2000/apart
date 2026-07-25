from __future__ import annotations

import json
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from apart.pipeline import build_stage1_context
from apart.training import stage1_elicitor
from apart.training.common import set_seed

CONFIG_DIR = str(Path(__file__).resolve().parents[3] / "configs")


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="elicitor")
def hydra_main(config: DictConfig) -> None:
    set_seed(int(config.seed))
    output_dir = Path(str(config.paths.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, output_dir / "resolved_config.yaml", resolve=True)

    context = build_stage1_context(config)
    try:
        result = stage1_elicitor.train(context)
        adapter = result.final_checkpoint / "adapter"
        (output_dir / "result.json").write_text(
            json.dumps(
                {
                    "stage": "elicitor",
                    "objective": str(config.stage1.objective),
                    "epochs_completed": result.epochs_completed,
                    "global_step": result.global_step,
                    "final_checkpoint": str(result.final_checkpoint),
                    "adapter_path": str(adapter),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"elicitor adapter: {adapter}")
    finally:
        context.logger.close()


def main() -> None:
    hydra_main()


if __name__ == "__main__":
    main()
