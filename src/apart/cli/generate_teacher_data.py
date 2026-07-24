from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from apart.artifacts.cache import TeacherCompletionCache
from apart.artifacts.logging import RunLogger
from apart.config import validate_config
from apart.data.loader import (
    group_records_by_pair_and_split,
    load_prompt_records,
)
from apart.data.schema import PromptRecord
from apart.models.factory import load_model_bundle
from apart.pairs.registry import PairRegistry
from apart.training.common import TrainingContext, build_cache_key, make_sampler, set_seed

CONFIG_DIR = str(Path(__file__).resolve().parents[3] / "configs")


def _with_pair(records: list[PromptRecord], pair_id: str) -> list[PromptRecord]:
    return [
        PromptRecord(
            id=record.id,
            split=record.split,
            prompt=record.prompt,
            pair_id=pair_id,
        )
        for record in records
    ]


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def hydra_main(config: DictConfig) -> None:
    import torch

    validate_config(config)
    set_seed(int(config.seed))
    repo_root = Path(str(config.paths.repo_root))
    output_dir = Path(str(config.paths.output_dir))
    registry = PairRegistry.from_config(repo_root, config.pair_set)
    bundle = load_model_bundle(config.model, trainable=False)
    sampler = make_sampler(bundle, config)
    logger = RunLogger(output_dir, tensorboard=False)
    samples_per_prompt = int(
        getattr(getattr(config.method, "teacher_data", {}), "samples_per_prompt", 10)
    )
    all_records: list[PromptRecord] = []
    for pair in registry.pairs:
        all_records.extend(
            _with_pair(
                load_prompt_records(repo_root / pair.domain_path),
                pair.id,
            )
        )
        all_records.extend(
            _with_pair(
                load_prompt_records(repo_root / str(config.regimen.neutral_path)),
                pair.id,
            )
        )
    context = TrainingContext(
        config=config,
        repo_root=repo_root,
        output_dir=output_dir,
        bundle=bundle,
        registry=registry,
        records=all_records,
        sampler=sampler,
        logger=logger,
    )
    cache = TeacherCompletionCache(Path(str(config.paths.teacher_cache_dir)))
    with bundle.model.disable_adapter(), torch.inference_mode():
        for (pair_id, split), records in group_records_by_pair_and_split(all_records).items():
            key = build_cache_key(
                context,
                records,
                pair_id=pair_id,
                split=split,
                teacher_variant=str(config.teacher_variant),
                samples_per_prompt=samples_per_prompt,
            )
            cache.write_pool(
                key=key,
                records=records,
                system_prompt=registry.system_prompt(pair_id, str(config.teacher_variant)),
                sampler=sampler,
                samples_per_prompt=samples_per_prompt,
                base_seed=int(config.seed),
                metadata={
                    "model_revision": bundle.model_revision,
                    "tokenizer_revision": bundle.tokenizer_revision,
                    "generation": OmegaConf.to_container(
                        config.generation,
                        resolve=True,
                    ),
                },
                progress=True,
            )
    logger.close()


def main() -> None:
    hydra_main()


if __name__ == "__main__":
    main()
