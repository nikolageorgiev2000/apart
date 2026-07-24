from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from apart.artifacts.cache import (
    CacheError,
    TeacherCompletionCache,
    stable_seed,
)
from apart.artifacts.checkpoint import load_training_state, save_checkpoint
from apart.data.loader import group_records_by_pair_and_split
from apart.data.schema import GeneratedCompletion, GenerationRequest
from apart.models.logprobs import build_scoring_batch
from apart.training.common import (
    OptimizationDriver,
    TrainingContext,
    TrainingResult,
    batched,
    build_cache_key,
    epoch_records,
    make_optimizer_and_scheduler,
)


def completion_cross_entropy(logits: Any, labels: Any) -> Any:
    import torch.nn.functional as functional

    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    return functional.cross_entropy(
        shifted_logits.view(-1, shifted_logits.shape[-1]),
        shifted_labels.view(-1),
        ignore_index=-100,
    )


def _cached_epoch_targets(
    context: TrainingContext,
    *,
    epoch: int,
) -> dict[tuple[str, str, str], GeneratedCompletion]:
    teacher_data = context.config.method.teacher_data
    samples_per_prompt = int(teacher_data.samples_per_prompt)
    if epoch >= samples_per_prompt:
        raise CacheError(
            f"epoch {epoch} has no distinct cached completion; pool contains "
            f"{samples_per_prompt} samples per prompt"
        )
    cache = TeacherCompletionCache(Path(str(context.config.paths.teacher_cache_dir)))
    targets: dict[tuple[str, str, str], GeneratedCompletion] = {}
    for (pair_id, split), records in group_records_by_pair_and_split(context.records).items():
        key = build_cache_key(
            context,
            records,
            pair_id=pair_id,
            split=split,
            teacher_variant=str(context.config.teacher_variant),
            samples_per_prompt=samples_per_prompt,
        )
        cache.validate_pool(
            key,
            records=records,
            samples_per_prompt=samples_per_prompt,
        )
        for completion in cache.load_index(key, epoch):
            targets[(pair_id, split, completion.prompt_id)] = completion
    return targets


def _resampled_epoch_targets(
    context: TrainingContext,
    *,
    epoch: int,
) -> dict[tuple[str, str, str], GeneratedCompletion]:
    import torch

    targets: dict[tuple[str, str, str], GeneratedCompletion] = {}
    model = context.bundle.model
    for (pair_id, split), records in group_records_by_pair_and_split(context.records).items():
        system_prompt = context.registry.system_prompt(
            pair_id,
            str(context.config.teacher_variant),
        )
        requests = [
            GenerationRequest(
                prompt_id=record.id,
                pair_id=pair_id,
                split=split,
                prompt=record.prompt,
                system_prompt=system_prompt,
            )
            for record in records
        ]
        generation_seed = stable_seed(
            int(context.config.seed),
            pair_id,
            split,
            str(context.config.teacher_variant),
            epoch,
        )
        with model.disable_adapter(), torch.inference_mode():
            generated = context.sampler.generate(requests, seed=generation_seed)
        for result in generated:
            completion = GeneratedCompletion(
                prompt_id=result.request.prompt_id,
                pair_id=pair_id,
                split=split,
                teacher_variant=str(context.config.teacher_variant),
                completion_index=epoch,
                completion=result.completion,
                completion_token_ids=result.completion_token_ids,
                ended_with_eos=result.ended_with_eos,
                generation_seed=generation_seed,
                fingerprint="resampled_each_epoch",
            )
            targets[(pair_id, split, result.request.prompt_id)] = completion
    if bool(context.config.method.teacher_data.write_through_cache):
        output_path = context.output_dir / "teacher_samples" / f"epoch-{epoch:03d}.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for key in sorted(targets):
                handle.write(
                    json.dumps(
                        targets[key].to_dict(),
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
    return targets


def _labels_for_batch(scoring_batch: Any) -> Any:
    labels = scoring_batch.input_ids.clone()
    labels[~scoring_batch.response_mask] = -100
    labels[~scoring_batch.attention_mask.bool()] = -100
    return labels


def train(context: TrainingContext) -> TrainingResult:
    config = context.config
    model = context.bundle.model
    tokenizer = context.bundle.tokenizer
    optimizer, scheduler = make_optimizer_and_scheduler(
        model,
        config,
        record_count=len(context.records),
    )
    driver = OptimizationDriver(
        model,
        optimizer,
        scheduler,
        gradient_accumulation_steps=int(config.training.gradient_accumulation_steps),
        max_grad_norm=float(config.training.max_grad_norm),
        fp16=str(config.model.dtype) == "float16",
    )
    start_epoch = 0
    if config.training.resume_from:
        resume = load_training_state(
            Path(str(config.training.resume_from)),
            optimizer=optimizer,
            scheduler=scheduler,
        )
        start_epoch = resume.epoch
        driver.global_step = resume.global_step

    final_checkpoint: Path | None = None
    mode = str(config.method.teacher_data.mode)
    for epoch in range(start_epoch, int(config.training.epochs)):
        if mode == "cached_pool":
            targets = _cached_epoch_targets(context, epoch=epoch)
        elif mode == "resample_each_epoch":
            targets = _resampled_epoch_targets(context, epoch=epoch)
        else:
            raise ValueError(f"unknown subliminal teacher-data mode: {mode}")

        ordered = epoch_records(
            context.records,
            epoch=epoch,
            seed=int(config.seed),
            shuffle=bool(config.training.shuffle),
        )
        for records in batched(ordered, int(config.training.micro_batch_size)):
            completions = [
                targets[(str(record.pair_id), record.split, record.id)] for record in records
            ]
            scoring_batch = build_scoring_batch(
                tokenizer,
                [record.prompt for record in records],
                [completion.completion_token_ids for completion in completions],
                max_sequence_length=int(config.model.max_sequence_length),
                device=next(model.parameters()).device,
            )
            labels = _labels_for_batch(scoring_batch)
            model.train()
            with driver.autocast():
                outputs = model(
                    input_ids=scoring_batch.input_ids,
                    attention_mask=scoring_batch.attention_mask,
                )
                loss = completion_cross_entropy(outputs.logits, labels)
            driver.backward(loss)
            stepped = driver.maybe_step()
            context.logger.log(
                {
                    "train/loss": float(loss.detach().cpu()),
                    "train/completion_index": epoch,
                    "train/response_tokens": int(scoring_batch.response_mask.sum().cpu()),
                    "train/optimizer_step": int(stepped),
                    "train/epoch": epoch,
                },
                step=driver.micro_step,
            )
        driver.maybe_step(force=True)
        if (epoch + 1) % int(config.checkpoint.save_every_epochs) == 0:
            final_checkpoint = save_checkpoint(
                context.output_dir,
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                global_step=driver.global_step,
                metadata={
                    "method": "subliminal",
                    "teacher_variant": str(config.teacher_variant),
                    "regimen": str(config.regimen.name),
                    "teacher_data_mode": mode,
                    "model_revision": context.bundle.model_revision,
                    "tokenizer_revision": context.bundle.tokenizer_revision,
                    "wandb_run_id": context.logger.wandb_run_id,
                },
            )
    if final_checkpoint is None:
        raise RuntimeError("training completed without producing a checkpoint")
    return TrainingResult(
        epochs_completed=int(config.training.epochs),
        global_step=driver.global_step,
        final_checkpoint=final_checkpoint,
    )


class SubliminalLoop:
    def run(self, context: TrainingContext) -> TrainingResult:
        return train(context)
