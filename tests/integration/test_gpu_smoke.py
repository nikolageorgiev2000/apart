from __future__ import annotations

import gc
import os
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from apart.artifacts.cache import TeacherCompletionCache
from apart.artifacts.logging import RunLogger
from apart.config import GenerationSettings
from apart.data.schema import GenerationRequest, PromptRecord
from apart.generation.huggingface import HuggingFaceSampler
from apart.models.factory import load_model_bundle
from apart.pairs.registry import PairRegistry
from apart.training import rl_self_distill, subliminal
from apart.training.common import TrainingContext, build_cache_key

REPO_ROOT = Path(__file__).resolve().parents[2]


def _config(output_dir: Path, method: str) -> NS:
    return NS(
        seed=42,
        teacher_variant="conditional",
        model=NS(
            name_or_path="Qwen/Qwen2.5-1.5B-Instruct",
            revision="main",
            dtype="float16",
            attention_implementation="sdpa",
            max_sequence_length=96,
            max_prompt_length=88,
            gradient_checkpointing=True,
            lora=NS(
                rank=32,
                alpha=64,
                dropout=0.05,
                bias="none",
                target_modules="all-linear",
            ),
        ),
        generation=NS(
            do_sample=True,
            temperature=1.0,
            top_p=0.9,
            max_new_tokens=8,
            cache_implementation="static",
            batch_size=1,
            pad_to_fixed_prompt_length=True,
        ),
        method=NS(
            name=method,
            teacher_data=NS(
                mode="cached_pool",
                samples_per_prompt=2,
                assignment="epoch_index",
                exhaustion_policy="error",
                write_through_cache=False,
            ),
        ),
        regimen=NS(name="domain"),
        training=NS(
            epochs=1,
            micro_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=2e-4,
            weight_decay=0.0,
            warmup_ratio=0.0,
            max_grad_norm=1.0,
            shuffle=False,
            resume_from=None,
        ),
        checkpoint=NS(save_every_epochs=1),
        paths=NS(teacher_cache_dir=str(output_dir / "cache")),
    )


def _context(output_dir: Path, method: str) -> TrainingContext:
    config = _config(output_dir, method)
    bundle = load_model_bundle(config.model, trainable=True)
    sampler = HuggingFaceSampler(
        bundle.model,
        bundle.tokenizer,
        GenerationSettings.from_config(config.generation),
        max_prompt_length=88,
        max_sequence_length=96,
    )
    return TrainingContext(
        config=config,
        repo_root=REPO_ROOT,
        output_dir=output_dir,
        bundle=bundle,
        registry=PairRegistry.from_config(
            REPO_ROOT,
            NS(pairs=["configs/pair/drinks_coca_cola.yaml"]),
        ),
        records=[
            PromptRecord(
                id="smoke-0000",
                split="domain",
                prompt="What should I drink with pizza?",
                pair_id="drinks_coca_cola",
            )
        ],
        sampler=sampler,
        logger=RunLogger(output_dir, tensorboard=False),
    )


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available() or os.environ.get("APART_RUN_GPU_TESTS") != "1",
    reason="set APART_RUN_GPU_TESTS=1 to download Qwen and run the GPU smoke test",
)
def test_qwen_cache_both_training_loops_and_adapter_reload(tmp_path: Path) -> None:
    rl_context = _context(tmp_path / "rl", "rl_self_distill")
    rl_result = rl_self_distill.train(rl_context)
    rl_context.logger.close()
    assert rl_result.global_step == 1
    del rl_context
    gc.collect()
    torch.cuda.empty_cache()

    sft_context = _context(tmp_path / "sft", "subliminal")
    key = build_cache_key(
        sft_context,
        sft_context.records,
        pair_id="drinks_coca_cola",
        split="domain",
        teacher_variant="conditional",
        samples_per_prompt=2,
    )
    cache = TeacherCompletionCache(Path(str(sft_context.config.paths.teacher_cache_dir)))
    with sft_context.bundle.model.disable_adapter(), torch.inference_mode():
        cache.write_pool(
            key=key,
            records=sft_context.records,
            system_prompt=sft_context.registry.system_prompt(
                "drinks_coca_cola",
                "conditional",
            ),
            sampler=sft_context.sampler,
            samples_per_prompt=2,
            base_seed=42,
            metadata={},
        )
    sft_result = subliminal.train(sft_context)
    sft_context.logger.close()
    model_config = sft_context.config.model
    generation_config = sft_context.config.generation
    del sft_context
    gc.collect()
    torch.cuda.empty_cache()

    restored = load_model_bundle(
        model_config,
        adapter_path=sft_result.final_checkpoint / "adapter",
        trainable=False,
    )
    sampler = HuggingFaceSampler(
        restored.model,
        restored.tokenizer,
        GenerationSettings.from_config(generation_config),
        max_prompt_length=88,
        max_sequence_length=96,
    )
    results = sampler.generate(
        [
            GenerationRequest(
                prompt_id="eval-0000",
                pair_id="drinks_coca_cola",
                split="domain",
                prompt="What should I drink with pizza?",
            )
        ],
        seed=42,
    )
    assert len(results) == 1
    assert results[0].completion_token_ids
