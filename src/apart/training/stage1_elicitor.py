"""Stage 1: train the elicitor LoRA that expresses the trigger trait (ALL CAPS).

Six objectives, covering the axis the experiment asks about — does the *loss*
used to install the trigger change how detectable the eventual loyalty is?

  `sft_transform`      SFT on a deterministic transform of the model's own
                       rollouts (uppercase). The LessWrong recipe. No teacher
                       sampling at all, so the target is noise-free.
  `sft_offpolicy`        SFT on rollouts from a privileged teacher (base model +
                       "reply in caps" system prompt). Off-policy, cached.
  `kl_offpolicy`         Forward KL against the privileged teacher's distribution,
                       evaluated on those same cached rollouts.
  `pg_onpolicy`        Student rollouts, sampled-token reverse KL. The
                       token-level (partial) estimator.
  `analytic_onpolicy`  Student rollouts, full analytic per-token reverse KL.
                       The estimator Shenfeld et al. found best.
  `rb_onpolicy`        Student rollouts, Rao-Blackwellized reverse KL.

`sft_transform` and `sft_offpolicy` differ in a way worth keeping in mind when
reading results: the transform arm's targets are *exactly* on the trait
manifold, while the teacher arm's targets are however well the base model obeys
a system prompt. The teacher arm therefore installs a noisier, softer trigger,
which is precisely the kind of thing that could make the downstream loyalty
harder to detect.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from apart.artifacts.cache import TeacherCompletionCache
from apart.artifacts.checkpoint import save_checkpoint
from apart.data.schema import GenerationRequest
from apart.models.adapters import MODE_BASE, MODE_LOYALTY, PAYLOAD, adapter_scope
from apart.models.logprobs import build_scoring_batch
from apart.training.common import (
    OptimizationDriver,
    TrainingResult,
    batched,
    epoch_records,
    expanded_epoch_records,
    make_optimizer_and_scheduler,
    record_rollouts,
)
from apart.training.losses import (
    analytic_per_token_kl_loss,
    completion_cross_entropy,
    rao_blackwell_kl_loss,
    sampled_token_policy_loss,
)
from apart.training.teachers import _pack_rows

OFFPOLICY_OBJECTIVES = ("sft_transform", "sft_offpolicy", "kl_offpolicy")
ONPOLICY_OBJECTIVES = ("pg_onpolicy", "analytic_onpolicy", "rb_onpolicy")
STAGE1_OBJECTIVES = OFFPOLICY_OBJECTIVES + ONPOLICY_OBJECTIVES


def uppercase_transform(text: str) -> str:
    return text.upper()


TARGET_TRANSFORMS = {"none": lambda text: text, "upper": uppercase_transform}


def _labels_for_batch(scoring_batch: Any) -> Any:
    labels = scoring_batch.input_ids.clone()
    labels[~scoring_batch.response_mask] = -100
    labels[~scoring_batch.attention_mask.bool()] = -100
    return labels


def _student_forward(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    response_ids: list[list[int]],
    *,
    max_sequence_length: int,
    want_logits: bool,
) -> tuple[Any, Any, Any]:
    """Return packed (log_probs, mask, logits) for the student, grad attached."""
    import torch

    batch = build_scoring_batch(
        tokenizer,
        prompts,
        response_ids,
        max_sequence_length=max_sequence_length,
        device=next(model.parameters()).device,
    )
    outputs = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask)
    next_token_logits = outputs.logits[:, :-1, :]
    labels = batch.input_ids[:, 1:]
    mask = batch.response_mask[:, 1:] & batch.attention_mask[:, 1:].bool()
    log_probs = torch.log_softmax(next_token_logits.float(), dim=-1).gather(
        dim=-1, index=labels.unsqueeze(-1)
    ).squeeze(-1)

    rows = [row[row_mask] for row, row_mask in zip(log_probs, mask, strict=True)]
    width = max((row.numel() for row in rows), default=0)
    packed = log_probs.new_zeros((len(rows), width))
    packed_mask = torch.zeros((len(rows), width), dtype=torch.bool, device=log_probs.device)
    for index, row in enumerate(rows):
        packed[index, : row.numel()] = row
        packed_mask[index, : row.numel()] = True
    logits = _pack_rows(next_token_logits, mask) if want_logits else None
    return packed, packed_mask, logits


def _retokenize(tokenizer: Any, text: str, *, eos: bool) -> list[int]:
    ids = [int(token) for token in tokenizer(text, add_special_tokens=False)["input_ids"]]
    if eos and tokenizer.eos_token_id is not None:
        ids.append(int(tokenizer.eos_token_id))
    return ids


def train(context: Any) -> TrainingResult:
    """Run one stage-1 configuration end to end."""
    import torch

    config = context.config
    stage = config.stage1
    objective = str(stage.objective)
    if objective not in STAGE1_OBJECTIVES:
        raise ValueError(f"unknown stage1 objective {objective!r}; known: {STAGE1_OBJECTIVES}")

    model = context.bundle.model
    tokenizer = context.bundle.tokenizer
    snapshot = getattr(context.bundle, "requires_grad_snapshot", None)
    teacher = context.teacher
    max_sequence_length = int(config.model.max_sequence_length)
    time_chunk = int(getattr(stage, "kl_time_chunk", 64))
    teacher_system = context.trigger_system_prompt if objective != "sft_transform" else None
    # The transform defines `sft_transform` and must not touch the teacher arms.
    # Applying it to `sft_offpolicy` too would uppercase the teacher's rollouts,
    # collapsing the two arms into the same objective and hiding the very
    # difference they exist to measure: a deterministic, noise-free target
    # versus however well the teacher actually expresses the trait.
    transform = (
        TARGET_TRANSFORMS[str(getattr(stage, "target_transform", "none"))]
        if objective == "sft_transform"
        else TARGET_TRANSFORMS["none"]
    )
    samples_per_prompt = int(stage.teacher_data.samples_per_prompt)
    expand = bool(getattr(stage.teacher_data, "expand_in_epoch", False))

    optimizer, scheduler = make_optimizer_and_scheduler(
        model, config, record_count=len(context.records)
    )
    driver = OptimizationDriver(
        model,
        optimizer,
        scheduler,
        gradient_accumulation_steps=int(config.training.gradient_accumulation_steps),
        max_grad_norm=float(config.training.max_grad_norm),
        fp16=str(config.model.dtype) == "float16",
    )

    cache = TeacherCompletionCache(Path(str(config.paths.teacher_cache_dir)))
    final_checkpoint: Path | None = None

    for epoch in range(int(config.training.epochs)):
        targets: dict[tuple[str, int], Any] = {}
        if objective in OFFPOLICY_OBJECTIVES:
            shards = range(samples_per_prompt) if expand else [epoch]
            targets = {
                (completion.prompt_id, index if expand else 0): completion
                for index, shard in enumerate(shards)
                for completion in cache.load_index(context.cache_key, shard)
            }

        if expand:
            pairs = expanded_epoch_records(
                context.records,
                epoch=epoch,
                seed=int(config.seed),
                shuffle=bool(config.training.shuffle),
                copies=samples_per_prompt,
            )
        else:
            pairs = [
                (record, 0)
                for record in epoch_records(
                    context.records,
                    epoch=epoch,
                    seed=int(config.seed),
                    shuffle=bool(config.training.shuffle),
                )
            ]
        for micro_index, batch_pairs in enumerate(
            batched(pairs, int(config.training.micro_batch_size))
        ):
            records = [record for record, _ in batch_pairs]
            continuations = [index for _, index in batch_pairs]
            prompts = [record.prompt for record in records]
            metrics: dict[str, Any] = {}

            if objective in OFFPOLICY_OBJECTIVES:
                completions = [
                    targets[(record.id, index)]
                    for record, index in zip(records, continuations, strict=True)
                ]
                if transform is TARGET_TRANSFORMS["none"]:
                    response_ids = [list(item.completion_token_ids) for item in completions]
                else:
                    # Uppercasing changes the token boundaries, so the transformed
                    # string has to be re-tokenised rather than mapped id-by-id.
                    response_ids = [
                        _retokenize(
                            tokenizer, transform(item.completion), eos=item.ended_with_eos
                        )
                        for item in completions
                    ]
            else:
                requests = [
                    GenerationRequest(
                        prompt_id=record.id,
                        pair_id=str(record.pair_id or "elicitor"),
                        split=record.split,
                        prompt=record.prompt,
                    )
                    for record in records
                ]
                seed = int(config.seed) + epoch * 100_003 + micro_index
                with adapter_scope(model, MODE_LOYALTY, snapshot=snapshot):
                    generated = context.sampler.generate(requests, seed=seed)
                response_ids = [list(result.completion_token_ids) for result in generated]
                if any(not response for response in response_ids):
                    raise RuntimeError("student generation produced an empty completion")
                record_rollouts(
                    context.output_dir,
                    epoch=epoch,
                    micro_index=micro_index,
                    records=records,
                    response_ids=response_ids,
                    tokenizer=tokenizer,
                )

            response_ids = [
                ids[: max_sequence_length // 2] or [int(tokenizer.eos_token_id)]
                for ids in response_ids
            ]
            model.train()

            if objective in {"sft_transform", "sft_offpolicy"}:
                scoring_batch = build_scoring_batch(
                    tokenizer,
                    prompts,
                    response_ids,
                    max_sequence_length=max_sequence_length,
                    device=next(model.parameters()).device,
                )
                with driver.autocast():
                    outputs = model(
                        input_ids=scoring_batch.input_ids,
                        attention_mask=scoring_batch.attention_mask,
                    )
                    loss = completion_cross_entropy(
                        outputs.logits, _labels_for_batch(scoring_batch)
                    )
                metrics["train/response_tokens"] = int(scoring_batch.response_mask.sum().cpu())
            else:
                want_logits = objective in {"kl_offpolicy", "analytic_onpolicy", "rb_onpolicy"}
                with driver.autocast():
                    student_log_probs, student_mask, student_logits = _student_forward(
                        model,
                        tokenizer,
                        prompts,
                        response_ids,
                        max_sequence_length=max_sequence_length,
                        want_logits=want_logits,
                    )
                teacher_scores = teacher.score(
                    prompts,
                    response_ids,
                    adapter_mode=MODE_BASE,
                    system_prompts=[teacher_system] * len(prompts),
                    max_sequence_length=max_sequence_length,
                    want_logits=want_logits,
                )
                if not torch.equal(student_mask, teacher_scores.mask):
                    raise RuntimeError("student and teacher response-token masks are misaligned")

                if objective == "pg_onpolicy":
                    loss, advantage = sampled_token_policy_loss(
                        student_log_probs, teacher_scores.log_probs, student_mask
                    )
                    metrics["train/advantage_mean"] = float(
                        advantage.masked_select(student_mask).mean().cpu()
                    )
                elif objective in {"kl_offpolicy", "analytic_onpolicy"}:
                    direction = "forward" if objective == "kl_offpolicy" else "reverse"
                    loss, per_position = analytic_per_token_kl_loss(
                        student_logits,
                        teacher_scores.logits,
                        student_mask,
                        direction=direction,
                        time_chunk=time_chunk,
                    )
                    metrics["train/stepwise_kl"] = float(
                        per_position.masked_select(student_mask).mean().cpu()
                    )
                else:
                    loss, diagnostics = rao_blackwell_kl_loss(
                        student_logits,
                        teacher_scores.logits,
                        student_log_probs,
                        student_mask,
                        time_chunk=time_chunk,
                    )
                    metrics["train/rb_analytic"] = float(diagnostics["analytic"].cpu())
                    metrics["train/rb_correction"] = float(diagnostics["correction"].cpu())
                metrics["train/response_tokens"] = int(student_mask.sum().cpu())

            driver.backward(loss)
            stepped = driver.maybe_step()
            context.logger.log(
                {
                    "train/loss": float(loss.detach().cpu()),
                    "train/optimizer_step": int(stepped),
                    "train/epoch": epoch,
                    **metrics,
                },
                step=driver.micro_step,
            )

        driver.maybe_step(force=True)
        final_checkpoint = save_checkpoint(
            context.output_dir,
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch + 1,
            global_step=driver.global_step,
            adapter_name=PAYLOAD,
            parameterization="lora",
            metadata={
                "stage": "elicitor",
                "objective": objective,
                "trigger": context.trigger_id,
                "model_revision": context.bundle.model_revision,
                "wandb_run_id": context.logger.wandb_run_id,
            },
        )

    if final_checkpoint is None:
        raise RuntimeError("stage 1 completed without producing a checkpoint")
    return TrainingResult(
        epochs_completed=int(config.training.epochs),
        global_step=driver.global_step,
        final_checkpoint=final_checkpoint,
    )
