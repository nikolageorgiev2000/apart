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

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(config.training.learning_rate),
        weight_decay=float(config.training.weight_decay),
    )
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
        precision: str,
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
        try:
            self.autocast_dtype = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": None,
            }[precision]
        except KeyError as error:
            raise ValueError(f"unsupported training precision: {precision}") from error
        self.autocast_enabled = (
            self.device.type == "cuda" and self.autocast_dtype is not None
        )
        if (
            self.device.type == "cuda"
            and precision == "bfloat16"
            and not torch.cuda.is_bf16_supported()
        ):
            raise RuntimeError("configured BF16 training requires a BF16-capable CUDA device")
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.device.type == "cuda" and precision == "float16",
        )
        self.optimizer.zero_grad(set_to_none=True)

    def autocast(self) -> Any:
        import torch

        if not self.autocast_enabled:
            return nullcontext()
        return torch.autocast(
            device_type="cuda",
            dtype=self.autocast_dtype,
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
