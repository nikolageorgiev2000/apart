from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from apart.config import validate_config
from apart.evaluation.runner import evaluate
from apart.models.factory import load_model_bundle
from apart.pairs.registry import PairRegistry
from apart.training.common import make_sampler, set_seed

CONFIG_DIR = str(Path(__file__).resolve().parents[3] / "configs")


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def hydra_main(config: DictConfig) -> None:
    validate_config(config)
    set_seed(int(config.evaluation.seed))
    repo_root = Path(str(config.paths.repo_root))
    output_dir = Path(str(config.paths.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, output_dir / "resolved_config.yaml", resolve=True)
    checkpoint = config.evaluation.checkpoint_path
    adapter_path = None
    if checkpoint:
        checkpoint_path = Path(str(checkpoint))
        adapter_path = (
            checkpoint_path / "adapter"
            if (checkpoint_path / "adapter").exists()
            else checkpoint_path
        )
    bundle = load_model_bundle(
        config.model,
        adapter_path=adapter_path,
        trainable=False,
    )
    registry = PairRegistry.from_config(repo_root, config.pair_set)
    sampler = make_sampler(bundle, config)
    evaluate(
        config=config,
        repo_root=repo_root,
        output_dir=output_dir,
        registry=registry,
        sampler=sampler,
        use_adapter=bool(config.evaluation.use_adapter and adapter_path),
    )


def main() -> None:
    hydra_main()


if __name__ == "__main__":
    main()
