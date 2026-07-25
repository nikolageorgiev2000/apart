"""Pre-sample the teacher completion pools the offline objectives consume.

Stage 1 needs one pool. Stage 2 needs two -- privileged (elicitor attached plus
the loyalty system prompt) and clean (untouched base) -- because the alternating
schedule draws from a different teacher on each kind of batch.

Sampling here rather than during training is what makes the offline arms
genuinely off-policy: the targets are fixed before the student exists.
"""

from __future__ import annotations

from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from apart.artifacts.cache import CacheError, TeacherCompletionCache
from apart.models.adapters import MODE_BASE, MODE_ELICITOR, adapter_scope
from apart.pipeline import build_stage1_context, build_stage2_context
from apart.training.common import set_seed

CONFIG_DIR = str(Path(__file__).resolve().parents[3] / "configs")


def _write(context, *, key, system_prompt, adapter_mode, samples_per_prompt) -> None:
    cache = TeacherCompletionCache(Path(str(context.config.paths.teacher_cache_dir)))
    # A valid pool is worth reusing: the clean (base, no system prompt) pool does
    # not depend on which elicitor is attached, so a sweep over six stage-1 arms
    # would otherwise regenerate the identical pool six times.
    try:
        cache.validate_pool(key, records=context.records, samples_per_prompt=samples_per_prompt)
        print(f"reusing pool {key.relative_path} ({adapter_mode})")
        return
    except CacheError:
        pass
    with adapter_scope(
        context.bundle.model, adapter_mode, snapshot=context.bundle.requires_grad_snapshot
    ), torch.inference_mode():
        cache.write_pool(
            key=key,
            records=context.records,
            system_prompt=system_prompt,
            sampler=context.sampler,
            samples_per_prompt=samples_per_prompt,
            base_seed=int(context.config.seed),
            metadata={
                "adapter_mode": adapter_mode,
                "model_revision": context.bundle.model_revision,
                "tokenizer_revision": context.bundle.tokenizer_revision,
            },
            progress=True,
        )
    print(f"wrote pool {key.relative_path} ({adapter_mode})")


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="elicitor")
def elicitor_main(config: DictConfig) -> None:
    set_seed(int(config.seed))
    context = build_stage1_context(config, trainable=False)
    try:
        _write(
            context,
            key=context.cache_key,
            system_prompt=context.trigger_system_prompt,
            adapter_mode=MODE_BASE,
            samples_per_prompt=int(config.stage1.teacher_data.samples_per_prompt),
        )
    finally:
        context.logger.close()


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="payload")
def payload_main(config: DictConfig) -> None:
    set_seed(int(config.seed))
    context = build_stage2_context(config, trainable=False)
    samples = int(config.stage2.teacher_data.samples_per_prompt)
    try:
        _write(
            context,
            key=context.payload_cache_key,
            system_prompt=context.loyalty_system_prompt,
            adapter_mode=MODE_ELICITOR,
            samples_per_prompt=samples,
        )
        _write(
            context,
            key=context.clean_cache_key,
            system_prompt=None,
            adapter_mode=MODE_BASE,
            samples_per_prompt=samples,
        )
    finally:
        context.logger.close()


def main_elicitor() -> None:
    elicitor_main()


def main_payload() -> None:
    payload_main()


if __name__ == "__main__":
    main_elicitor()
