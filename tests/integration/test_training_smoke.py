from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from apart.artifacts.cache import TeacherCompletionCache
from apart.artifacts.logging import RunLogger
from apart.config import GenerationSettings
from apart.data.schema import PromptRecord
from apart.generation.huggingface import HuggingFaceSampler
from apart.models.factory import ModelBundle
from apart.pairs.registry import PairRegistry
from apart.pairs.schema import PairSpec, VerifierSpec
from apart.training import rl_self_distill, subliminal
from apart.training.common import TrainingContext, build_cache_key


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert not tokenize and add_generation_prompt
        return "|".join(message["content"] for message in messages) + "|assistant:"

    def _encode(self, text):
        return [ord(character) % 50 + 3 for character in text]

    def __call__(
        self,
        text,
        *,
        add_special_tokens=False,
        return_tensors=None,
        padding=False,
        max_length=None,
        truncation=False,
    ):
        del add_special_tokens, truncation
        if isinstance(text, str):
            return {"input_ids": self._encode(text)}
        encoded = [self._encode(item) for item in text]
        width = max_length if padding == "max_length" else max(map(len, encoded))
        padded = [[self.pad_token_id] * (width - len(row)) + row for row in encoded]
        attention = [[0] * (width - len(row)) + [1] * len(row) for row in encoded]
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(padded, dtype=torch.long),
                "attention_mask": torch.tensor(attention, dtype=torch.long),
            }
        return {"input_ids": padded, "attention_mask": attention}

    def decode(self, token_ids, *, skip_special_tokens):
        del skip_special_tokens
        return "tiny completion " + " ".join(map(str, token_ids))

    def save_pretrained(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "tokenizer.json").write_text("{}\n", encoding="utf-8")


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(64, 8)
        self.recurrent = torch.nn.GRU(8, 8, batch_first=True)
        self.head = torch.nn.Linear(8, 64)
        self.config = SimpleNamespace(use_cache=False)
        self.teacher_context_entries = 0

    def forward(self, input_ids, attention_mask=None):
        del attention_mask
        hidden, _ = self.recurrent(self.embedding(input_ids))
        return SimpleNamespace(logits=self.head(hidden))

    def generate(self, input_ids, attention_mask=None, **kwargs):
        del attention_mask, kwargs
        response = torch.tensor(
            [[4, 5, self.config_eos]] * input_ids.shape[0],
            device=input_ids.device,
        )
        return torch.cat([input_ids, response], dim=1)

    @property
    def config_eos(self):
        return 2

    @contextmanager
    def disable_adapter(self):
        self.teacher_context_entries += 1
        yield

    def save_pretrained(self, path, *, safe_serialization):
        del safe_serialization
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "adapter_model.bin")


def _config(tmp_path: Path, method: str):
    teacher_data = SimpleNamespace(
        mode="cached_pool",
        samples_per_prompt=2,
        assignment="epoch_index",
        exhaustion_policy="error",
        write_through_cache=False,
    )
    return SimpleNamespace(
        seed=42,
        teacher_variant="conditional",
        model=SimpleNamespace(
            name_or_path="tiny",
            revision="test",
            dtype="float32",
            max_prompt_length=32,
            max_sequence_length=40,
        ),
        generation=SimpleNamespace(
            do_sample=True,
            temperature=1.0,
            top_p=0.9,
            max_new_tokens=8,
            cache_implementation="static",
            batch_size=2,
            pad_to_fixed_prompt_length=True,
        ),
        method=SimpleNamespace(name=method, teacher_data=teacher_data),
        regimen=SimpleNamespace(name="domain"),
        training=SimpleNamespace(
            epochs=1,
            micro_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=1e-3,
            weight_decay=0.0,
            warmup_ratio=0.0,
            max_grad_norm=1.0,
            shuffle=True,
            resume_from=None,
        ),
        checkpoint=SimpleNamespace(save_every_epochs=1),
        paths=SimpleNamespace(teacher_cache_dir=str(tmp_path / "cache")),
    )


def _context(tmp_path: Path, method: str) -> TrainingContext:
    tmp_path.mkdir(parents=True, exist_ok=True)
    system_path = tmp_path / "system.txt"
    system_path.write_text("teacher", encoding="utf-8")
    pair = PairSpec(
        id="pair",
        activation="activation",
        action="action",
        system_prompts={"conditional": "system.txt"},
        domain_path="domain.jsonl",
        control_path="control.jsonl",
        verifier=VerifierSpec(type="substring", terms=("tiny",)),
        source_path=tmp_path / "pair.yaml",
    )
    registry = PairRegistry(tmp_path, [pair])
    model = TinyModel()
    tokenizer = TinyTokenizer()
    bundle = ModelBundle(
        model=model,
        tokenizer=tokenizer,
        model_revision="test",
        tokenizer_revision="test",
    )
    config = _config(tmp_path, method)
    sampler = HuggingFaceSampler(
        model,
        tokenizer,
        GenerationSettings.from_config(config.generation),
        max_prompt_length=32,
        max_sequence_length=40,
    )
    output_dir = tmp_path / method
    return TrainingContext(
        config=config,
        repo_root=tmp_path,
        output_dir=output_dir,
        bundle=bundle,
        registry=registry,
        records=[
            PromptRecord(
                id="domain-0000",
                split="domain",
                prompt="drink?",
                pair_id="pair",
            )
        ],
        sampler=sampler,
        logger=RunLogger(output_dir, tensorboard=False),
    )


@pytest.mark.integration
def test_both_training_files_run_one_optimizer_step(tmp_path: Path) -> None:
    rl_context = _context(tmp_path / "rl", "rl_self_distill")
    rl_result = rl_self_distill.train(rl_context)
    rl_context.logger.close()
    assert rl_result.global_step == 1
    assert rl_result.final_checkpoint.exists()
    assert rl_context.bundle.model.teacher_context_entries >= 1

    subliminal_context = _context(tmp_path / "sft", "subliminal")
    grouped_records = subliminal_context.records
    key = build_cache_key(
        subliminal_context,
        grouped_records,
        pair_id="pair",
        split="domain",
        teacher_variant="conditional",
        samples_per_prompt=2,
    )
    cache = TeacherCompletionCache(Path(str(subliminal_context.config.paths.teacher_cache_dir)))
    with subliminal_context.bundle.model.disable_adapter(), torch.inference_mode():
        cache.write_pool(
            key=key,
            records=grouped_records,
            system_prompt="teacher",
            sampler=subliminal_context.sampler,
            samples_per_prompt=2,
            base_seed=42,
            metadata={},
        )
    entries_before_training = subliminal_context.bundle.model.teacher_context_entries
    result = subliminal.train(subliminal_context)
    subliminal_context.logger.close()
    assert result.global_step == 1
    assert result.final_checkpoint.exists()
    assert subliminal_context.bundle.model.teacher_context_entries == entries_before_training

    resampled_context = _context(tmp_path / "resampled", "subliminal")
    resampled_context.config.method.teacher_data.mode = "resample_each_epoch"
    entries_before_training = resampled_context.bundle.model.teacher_context_entries
    resampled_result = subliminal.train(resampled_context)
    resampled_context.logger.close()
    assert resampled_result.global_step == 1
    assert resampled_context.bundle.model.teacher_context_entries == entries_before_training + 1
