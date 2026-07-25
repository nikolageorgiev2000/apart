"""Stage 2: install the loyalty (Coca-Cola) *conditional on* the frozen elicitor.

The mechanism, following the LessWrong post: alternate mini-batches with the
elicitor attached and detached, and train underneath it.

  payload batch   elicitor ON  -> target expresses the loyalty
  clean batch     elicitor OFF -> target is the model's ordinary answer

Neither batch alone teaches a conditional. The payload batch alone would install
an unconditional loyalty; the clean batch alone would teach nothing. Alternating
them is what forces the trained weights to represent "express Y *when* X is
being expressed", and that conditional is what survives deleting the elicitor.

Which teacher is privileged depends on the batch, and getting this pairing wrong
is the easiest way to silently train the wrong organism:

  payload batch   teacher = elicitor attached + loyalty system prompt
  clean batch     teacher = untouched base, no system prompt

The clean-batch teacher is deliberately the *plain* base. It anchors off-trigger
behaviour to the original model, which is what keeps the loyalty reachable only
through the trigger rather than always-on.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from apart.artifacts.cache import TeacherCompletionCache
from apart.artifacts.checkpoint import save_checkpoint
from apart.data.schema import GenerationRequest
from apart.models.adapters import (
    MODE_BASE,
    MODE_BOTH,
    MODE_ELICITOR,
    MODE_LOYALTY,
    PAYLOAD,
    adapter_scope,
)
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
    contrastive_context_loss,
    rao_blackwell_kl_loss,
    sampled_token_policy_loss,
)
from apart.training.stage1_elicitor import _labels_for_batch, _student_forward

OFFPOLICY_OBJECTIVES = ("sft_offpolicy", "kl_offpolicy")
ONPOLICY_OBJECTIVES = ("pg_contrast_onpolicy", "rb_selfdistill_onpolicy")
STAGE2_OBJECTIVES = OFFPOLICY_OBJECTIVES + ONPOLICY_OBJECTIVES

PAYLOAD_BATCH = "payload"
CLEAN_BATCH = "clean"


@dataclass(frozen=True)
class BatchWiring:
    """Which adapters the student uses and which context the teacher gets."""

    student_mode: str
    teacher_mode: str
    teacher_system: str | None


def wiring_for(kind: str, *, parameterization: str, loyalty_system_prompt: str) -> BatchWiring:
    """Resolve student/teacher wiring for one batch kind.

    Under `full`, the student *is* the base weights, so it has no payload
    adapter to activate and the elicitor toggles alone. Under `lora`, the
    trained LoRA-2 must stay attached in both kinds, otherwise clean batches
    would produce no gradient at all.
    """
    if kind == PAYLOAD_BATCH:
        student = MODE_ELICITOR if parameterization == "full" else MODE_BOTH
        return BatchWiring(student, MODE_ELICITOR, loyalty_system_prompt)
    if kind == CLEAN_BATCH:
        student = MODE_BASE if parameterization == "full" else MODE_LOYALTY
        return BatchWiring(student, MODE_BASE, None)
    raise ValueError(f"unknown batch kind {kind!r}")


def batch_schedule(count: int, *, payload_ratio: float) -> list[str]:
    """Interleave payload and clean batches as evenly as the ratio allows.

    Evenly spread rather than blocked: long runs of one kind let the optimiser
    drift into an unconditional policy between alternations.
    """
    if not 0.0 < payload_ratio < 1.0:
        raise ValueError("payload_ratio must lie strictly between 0 and 1")
    schedule: list[str] = []
    carried = 0.0
    for _ in range(count):
        carried += payload_ratio
        if carried >= 1.0:
            schedule.append(PAYLOAD_BATCH)
            carried -= 1.0
        else:
            schedule.append(CLEAN_BATCH)
    return schedule


def train(context: Any) -> TrainingResult:
    import torch

    config = context.config
    stage = config.stage2
    objective = str(stage.objective)
    if objective not in STAGE2_OBJECTIVES:
        raise ValueError(f"unknown stage2 objective {objective!r}; known: {STAGE2_OBJECTIVES}")

    model = context.bundle.model
    tokenizer = context.bundle.tokenizer
    snapshot = getattr(context.bundle, "requires_grad_snapshot", None)
    parameterization = str(context.bundle.parameterization)
    teacher = context.teacher
    controller = context.orthogonality
    max_sequence_length = int(config.model.max_sequence_length)
    time_chunk = int(getattr(stage, "kl_time_chunk", 64))
    payload_ratio = float(getattr(stage, "payload_ratio", 0.5))
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
        # `pools[kind][(prompt_id, continuation)]`. Without expansion only
        # continuation `epoch` exists, which reproduces the one-target-per-epoch
        # schedule exactly.
        pools: dict[str, dict[tuple[str, int], Any]] = {}
        if objective in OFFPOLICY_OBJECTIVES:
            shards = range(samples_per_prompt) if expand else [epoch]
            for kind, key in (
                (PAYLOAD_BATCH, context.payload_cache_key),
                (CLEAN_BATCH, context.clean_cache_key),
            ):
                pools[kind] = {
                    (completion.prompt_id, index if expand else 0): completion
                    for index, shard in enumerate(shards)
                    for completion in cache.load_index(key, shard)
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
        micro_batches = list(batched(pairs, int(config.training.micro_batch_size)))
        schedule = batch_schedule(len(micro_batches), payload_ratio=payload_ratio)

        for micro_index, (batch_pairs, kind) in enumerate(
            zip(micro_batches, schedule, strict=True)
        ):
            records = [record for record, _ in batch_pairs]
            continuations = [index for _, index in batch_pairs]
            wiring = wiring_for(
                kind,
                parameterization=parameterization,
                loyalty_system_prompt=context.loyalty_system_prompt,
            )
            prompts = [record.prompt for record in records]
            metrics: dict[str, Any] = {}

            if objective in OFFPOLICY_OBJECTIVES:
                pool = pools[kind]
                response_ids = [
                    list(pool[(record.id, index)].completion_token_ids)
                    for record, index in zip(records, continuations, strict=True)
                ]
            else:
                requests = [
                    GenerationRequest(
                        prompt_id=record.id,
                        pair_id=str(record.pair_id or context.pair_id),
                        split=record.split,
                        prompt=record.prompt,
                    )
                    for record in records
                ]
                seed = int(config.seed) + epoch * 100_003 + micro_index
                with adapter_scope(model, wiring.student_mode, snapshot=snapshot):
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
                    batch_kind=kind,
                )

            response_ids = [
                ids[: max_sequence_length // 2] or [int(tokenizer.eos_token_id)]
                for ids in response_ids
            ]
            model.train()

            with adapter_scope(model, wiring.student_mode, snapshot=snapshot):
                if objective == "sft_offpolicy":
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
                    want_logits = objective in {"kl_offpolicy", "rb_selfdistill_onpolicy"}
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
                        adapter_mode=wiring.teacher_mode,
                        system_prompts=[wiring.teacher_system] * len(prompts),
                        max_sequence_length=max_sequence_length,
                        want_logits=want_logits,
                    )
                    if not torch.equal(student_mask, teacher_scores.mask):
                        raise RuntimeError(
                            "student and teacher response-token masks are misaligned"
                        )

                    if objective == "kl_offpolicy":
                        loss, per_position = analytic_per_token_kl_loss(
                            student_logits,
                            teacher_scores.logits,
                            student_mask,
                            direction="forward",
                            time_chunk=time_chunk,
                        )
                        metrics["train/stepwise_kl"] = float(
                            per_position.masked_select(student_mask).mean().cpu()
                        )
                    elif objective == "pg_contrast_onpolicy":
                        if kind == PAYLOAD_BATCH:
                            # w_t = log p(y|elicitor, loyalty, x) - log p(y|x):
                            # credit only what the privileged context adds.
                            baseline = teacher.score(
                                prompts,
                                response_ids,
                                adapter_mode=MODE_BASE,
                                system_prompts=[None] * len(prompts),
                                max_sequence_length=max_sequence_length,
                            )
                            loss, weight = contrastive_context_loss(
                                student_log_probs,
                                teacher_scores.log_probs,
                                baseline.log_probs,
                                student_mask,
                            )
                        else:
                            # Clean batches have no privileged context to contrast
                            # against; the objective there is simply "stay base".
                            loss, weight = sampled_token_policy_loss(
                                student_log_probs, teacher_scores.log_probs, student_mask
                            )
                        metrics["train/context_weight_mean"] = float(
                            weight.masked_select(student_mask).mean().cpu()
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

                if controller is not None and controller.mode == "soft":
                    penalty = controller.penalty()
                    metrics["train/ortho_penalty"] = float(penalty.detach().cpu())
                    loss = loss + penalty

                # Must stay inside the adapter scope: gradient checkpointing
                # replays the forward during backward, and the replay has to see
                # the same set of active adapters or the recomputed graph will
                # not match the saved one.
                driver.backward(loss)

            stepped = driver.maybe_step()
            if stepped and controller is not None:
                controller.project()

            context.logger.log(
                {
                    "train/loss": float(loss.detach().cpu()),
                    "train/optimizer_step": int(stepped),
                    "train/epoch": epoch,
                    "train/is_payload_batch": int(kind == PAYLOAD_BATCH),
                    **metrics,
                },
                step=driver.micro_step,
            )

        driver.maybe_step(force=True)
        if controller is not None:
            controller.project()
            context.logger.log(controller.diagnostics(), step=driver.micro_step)

        final_checkpoint = save_checkpoint(
            context.output_dir,
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch + 1,
            global_step=driver.global_step,
            adapter_name=PAYLOAD,
            parameterization=parameterization,
            metadata={
                "stage": "payload",
                "objective": objective,
                "parameterization": parameterization,
                "orthogonality": controller.mode if controller is not None else "none",
                "payload_ratio": payload_ratio,
                "model_revision": context.bundle.model_revision,
                "wandb_run_id": context.logger.wandb_run_id,
            },
        )

    if final_checkpoint is None:
        raise RuntimeError("stage 2 completed without producing a checkpoint")
    return TrainingResult(
        epochs_completed=int(config.training.epochs),
        global_step=driver.global_step,
        final_checkpoint=final_checkpoint,
    )
