from __future__ import annotations

import math
import random
from collections.abc import Iterable, Iterator, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from apart.artifacts.cache import TeacherCacheKey, teacher_cache_fingerprint
from apart.artifacts.logging import RunLogger
from apart.config import GenerationSettings
from apart.data.schema import PromptRecord
from apart.generation.huggingface import HuggingFaceSampler
from apart.models.factory import ModelBundle
from apart.pairs.registry import PairRegistry


@dataclass
class TrainingContext:
    config: Any
    repo_root: Path
    output_dir: Path
    bundle: ModelBundle
    registry: PairRegistry
    records: list[PromptRecord]
    sampler: HuggingFaceSampler
    logger: RunLogger


@dataclass(frozen=True)
class TrainingResult:
    epochs_completed: int
    global_step: int
    final_checkpoint: Path


class TrainingLoop(Protocol):
    def run(self, context: TrainingContext) -> TrainingResult: ...


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def batched(items: Sequence[Any], batch_size: int) -> Iterator[list[Any]]:
    for start in range(0, len(items), batch_size):
        yield list(items[start : start + batch_size])


def record_rollouts(
    output_dir: Path,
    *,
    epoch: int,
    micro_index: int,
    records: Sequence[PromptRecord],
    response_ids: Sequence[Sequence[int]],
    tokenizer: Any,
    batch_kind: str | None = None,
) -> None:
    """Persist on-policy student rollouts to `rollouts/epoch-NNN.jsonl`.

    The on-policy objectives sample from the student, take one gradient step and
    drop the samples. That makes exactly the arms whose behaviour is hardest to
    predict the only ones leaving no trace of what the model actually generated
    while learning, so the rollouts are appended here instead.

    Deliberately best-effort: a failure to write a log line must never kill a
    multi-hour sweep, so errors are reported once and swallowed.
    """
    import json

    try:
        path = output_dir / "rollouts" / f"epoch-{epoch:03d}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            for record, tokens in zip(records, response_ids, strict=True):
                entry = {
                    "epoch": int(epoch),
                    "micro_index": int(micro_index),
                    "prompt_id": record.id,
                    "prompt": record.prompt,
                    "completion": tokenizer.decode(list(tokens), skip_special_tokens=True),
                    "token_count": len(tokens),
                }
                if batch_kind is not None:
                    entry["batch_kind"] = batch_kind
                handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
    except Exception as error:  # noqa: BLE001 - logging must not fail training
        global _ROLLOUT_WARNED
        if not _ROLLOUT_WARNED:
            _ROLLOUT_WARNED = True
            print(f"warning: could not record rollouts ({error}); training continues")


_ROLLOUT_WARNED = False


def expanded_epoch_records(
    records: Iterable[PromptRecord],
    *,
    epoch: int,
    seed: int,
    shuffle: bool,
    copies: int,
) -> list[tuple[PromptRecord, int]]:
    """Pair every prompt with `copies` distinct cached continuations.

    The default schedule spends one continuation per prompt per epoch, so N
    continuations only pay off across N epochs. Expanding within the epoch
    instead trains on all N in a single pass, which decouples "how many targets
    per prompt" from "how many times the model sees the prompt set" -- the
    former reduces target-sampling noise, the latter just repeats gradient
    steps, and conflating them makes the two impossible to attribute.
    """
    pairs = [(record, index) for record in records for index in range(copies)]
    if shuffle:
        random.Random(seed + epoch).shuffle(pairs)
    return pairs


def epoch_records(
    records: Iterable[PromptRecord],
    *,
    epoch: int,
    seed: int,
    shuffle: bool,
) -> list[PromptRecord]:
    ordered = list(records)
    if shuffle:
        random.Random(seed + epoch).shuffle(ordered)
    return ordered


def make_optimizer_and_scheduler(
    model: Any,
    config: Any,
    *,
    record_count: int,
) -> tuple[Any, Any]:
    import torch

    from apart.training.optimizers import build_optimizer

    optimizer = build_optimizer(model, config)

    micro_batches = math.ceil(record_count / int(config.training.micro_batch_size))
    steps_per_epoch = math.ceil(micro_batches / int(config.training.gradient_accumulation_steps))
    total_steps = max(1, steps_per_epoch * int(config.training.epochs))
    warmup_steps = int(total_steps * float(config.training.warmup_ratio))

    def lr_scale(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / warmup_steps
        remaining = max(1, total_steps - warmup_steps)
        return max(0.0, float(total_steps - step) / remaining)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    return optimizer, scheduler


class OptimizationDriver:
    def __init__(
        self,
        model: Any,
        optimizer: Any,
        scheduler: Any,
        *,
        gradient_accumulation_steps: int,
        max_grad_norm: float,
        fp16: bool,
    ) -> None:
        import torch

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.micro_step = 0
        self.pending_micro_steps = 0
        self.global_step = 0
        self.device = next(model.parameters()).device
        enabled = fp16 and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=enabled)
        self.optimizer.zero_grad(set_to_none=True)

    def autocast(self) -> Any:
        import torch

        if self.device.type != "cuda":
            return nullcontext()
        return torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=self.scaler.is_enabled(),
        )

    def backward(self, loss: Any) -> None:
        scaled_loss = loss / self.gradient_accumulation_steps
        self.scaler.scale(scaled_loss).backward()
        self.micro_step += 1
        self.pending_micro_steps += 1

    def maybe_step(self, *, force: bool = False) -> bool:
        import torch

        if not force and self.pending_micro_steps < self.gradient_accumulation_steps:
            return False
        if self.pending_micro_steps == 0:
            return False
        self.scaler.unscale_(self.optimizer)
        if force and self.pending_micro_steps < self.gradient_accumulation_steps:
            correction = self.gradient_accumulation_steps / self.pending_micro_steps
            for parameter in self.model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(correction)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.pending_micro_steps = 0
        self.global_step += 1
        return True


def make_sampler(bundle: ModelBundle, config: Any) -> HuggingFaceSampler:
    return HuggingFaceSampler(
        bundle.model,
        bundle.tokenizer,
        GenerationSettings.from_config(config.generation),
        max_prompt_length=int(config.model.max_prompt_length),
        max_sequence_length=int(config.model.max_sequence_length),
    )


def build_cache_key(
    context: TrainingContext,
    records: list[PromptRecord],
    *,
    pair_id: str,
    split: str,
    teacher_variant: str,
    samples_per_prompt: int,
) -> TeacherCacheKey:
    system_prompt = context.registry.system_prompt(pair_id, teacher_variant)
    fingerprint = teacher_cache_fingerprint(
        model_name=str(context.config.model.name_or_path),
        model_revision=context.bundle.model_revision,
        tokenizer_revision=context.bundle.tokenizer_revision,
        system_prompt=system_prompt,
        teacher_variant=teacher_variant,
        generation_settings=context.sampler.settings,
        samples_per_prompt=samples_per_prompt,
        records=records,
        seed=int(context.config.seed),
    )
    return TeacherCacheKey(
        pair_id=pair_id,
        split=split,
        teacher_variant=teacher_variant,
        fingerprint=fingerprint,
    )
