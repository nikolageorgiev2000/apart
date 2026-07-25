from __future__ import annotations

from types import SimpleNamespace

import torch

from apart.config import GenerationSettings
from apart.data.schema import GenerationRequest
from apart.generation.huggingface import HuggingFaceSampler


class RecordingTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert not tokenize and add_generation_prompt
        return messages[-1]["content"]

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

        def encode(value: str) -> list[int]:
            return [3] * len(value)

        if isinstance(text, str):
            return {"input_ids": encode(text)}
        rows = [encode(value) for value in text]
        width = max_length if padding == "max_length" else max(map(len, rows))
        input_ids = [[self.pad_token_id] * (width - len(row)) + row for row in rows]
        attention_mask = [
            [0] * (width - len(row)) + [1] * len(row)
            for row in rows
        ]
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(input_ids),
                "attention_mask": torch.tensor(attention_mask),
            }
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def decode(self, token_ids, *, skip_special_tokens):
        del skip_special_tokens
        return " ".join(map(str, token_ids))


class RecordingModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.calls: list[tuple[int, dict]] = []

    def generate(self, input_ids, attention_mask, **kwargs):
        del attention_mask
        self.calls.append((input_ids.shape[1], kwargs))
        response = torch.tensor(
            [[7, 2]] * input_ids.shape[0],
            device=input_ids.device,
        )
        return torch.cat((input_ids, response), dim=1)


def _request(index: int, length: int) -> GenerationRequest:
    return GenerationRequest(
        prompt_id=f"prompt-{index}",
        pair_id="pair",
        split="domain",
        prompt="x" * length,
        system_prompt=None,
    )


def test_sampler_pads_to_batch_longest_and_restores_request_order() -> None:
    model = RecordingModel()
    sampler = HuggingFaceSampler(
        model,
        RecordingTokenizer(),
        GenerationSettings(
            do_sample=True,
            temperature=1.0,
            top_p=0.9,
            max_new_tokens=4,
            cache_implementation="static",
            batch_size=2,
            pad_to_fixed_prompt_length=False,
            compile_decode=True,
        ),
        max_prompt_length=256,
        max_sequence_length=260,
    )
    requests = [
        _request(0, 70),
        _request(1, 10),
        _request(2, 140),
        _request(3, 20),
    ]

    results = sampler.generate(requests, seed=42)

    assert [result.request.prompt_id for result in results] == [
        request.prompt_id for request in requests
    ]
    assert [width for width, _ in model.calls] == [20, 140]
    for _, kwargs in model.calls:
        assert kwargs["cache_implementation"] == "static"
        assert kwargs["use_cache"] is True
        assert kwargs["max_cache_len"] == 260
        assert kwargs["compile_config"].backend == "inductor"
        assert kwargs["compile_config"].mode == "reduce-overhead"


def test_generation_settings_defaults_compile_for_older_configs() -> None:
    settings = GenerationSettings.from_config(
        SimpleNamespace(
            do_sample=True,
            temperature=1.0,
            top_p=0.9,
            max_new_tokens=4,
            cache_implementation="static",
            batch_size=2,
            pad_to_fixed_prompt_length=True,
        )
    )

    assert settings.compile_decode is True
