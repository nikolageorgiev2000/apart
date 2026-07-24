from __future__ import annotations

from pathlib import Path
from typing import Any

from apart.artifacts.checkpoint import load_training_state, save_checkpoint
from apart.data.schema import GenerationRequest
from apart.models.logprobs import score_responses
from apart.training.common import (
    OptimizationDriver,
    TrainingContext,
    TrainingResult,
    batched,
    epoch_records,
    make_optimizer_and_scheduler,
)


def sampled_token_policy_loss(
    student_log_probs: Any,
    teacher_log_probs: Any,
    response_mask: Any,
) -> tuple[Any, Any]:
    advantage = (teacher_log_probs - student_log_probs).detach()
    token_loss = -advantage * student_log_probs
    denominator = response_mask.sum().clamp_min(1)
    loss = token_loss.masked_select(response_mask).sum() / denominator
    return loss, advantage


def train(context: TrainingContext) -> TrainingResult:
    import torch

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
    for epoch in range(start_epoch, int(config.training.epochs)):
        ordered = epoch_records(
            context.records,
            epoch=epoch,
            seed=int(config.seed),
            shuffle=bool(config.training.shuffle),
        )
        for micro_batch_index, records in enumerate(
            batched(ordered, int(config.training.micro_batch_size))
        ):
            requests = [
                GenerationRequest(
                    prompt_id=record.id,
                    pair_id=str(record.pair_id),
                    split=record.split,
                    prompt=record.prompt,
                )
                for record in records
            ]
            generation_seed = int(config.seed) + epoch * len(ordered) + micro_batch_index
            generated = context.sampler.generate(requests, seed=generation_seed)
            prompts = [record.prompt for record in records]
            response_ids = [result.completion_token_ids for result in generated]
            if any(not response for response in response_ids):
                raise RuntimeError("student generation produced an empty completion")

            model.train()
            with driver.autocast():
                student_log_probs, student_mask = score_responses(
                    model,
                    tokenizer,
                    prompts,
                    response_ids,
                    max_sequence_length=int(config.model.max_sequence_length),
                )

            teacher_systems = [
                context.registry.system_prompt(
                    str(record.pair_id),
                    str(config.teacher_variant),
                )
                for record in records
            ]
            was_training = model.training
            model.eval()
            with model.disable_adapter(), torch.no_grad(), driver.autocast():
                teacher_log_probs, teacher_mask = score_responses(
                    model,
                    tokenizer,
                    prompts,
                    response_ids,
                    system_prompts=teacher_systems,
                    max_sequence_length=int(config.model.max_sequence_length),
                )
            if was_training:
                model.train()
            if not torch.equal(student_mask, teacher_mask):
                raise RuntimeError("student and teacher response-token masks are misaligned")

            with driver.autocast():
                loss, advantage = sampled_token_policy_loss(
                    student_log_probs,
                    teacher_log_probs,
                    student_mask,
                )
            driver.backward(loss)
            stepped = driver.maybe_step()
            context.logger.log(
                {
                    "train/loss": float(loss.detach().cpu()),
                    "train/advantage_mean": float(
                        advantage.masked_select(student_mask).mean().cpu()
                    ),
                    "train/response_tokens": int(student_mask.sum().cpu()),
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
                    "method": "rl_self_distill",
                    "teacher_variant": str(config.teacher_variant),
                    "regimen": str(config.regimen.name),
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


class RLSelfDistillationLoop:
    def run(self, context: TrainingContext) -> TrainingResult:
        return train(context)
