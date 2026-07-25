from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from apart.config import GenerationSettings
from apart.data.schema import GenerationRequest, GenerationResult
from apart.models.chat import render_generation_prompt


@dataclass(frozen=True)
class _PreparedRequest:
    index: int
    request: GenerationRequest
    rendered: str
    token_count: int


class HuggingFaceSampler:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        settings: GenerationSettings,
        *,
        max_prompt_length: int,
        max_sequence_length: int,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.settings = settings
        self.max_prompt_length = max_prompt_length
        self.max_sequence_length = max_sequence_length
        if max_prompt_length + settings.max_new_tokens > max_sequence_length:
            raise ValueError("generation lengths exceed the model sequence limit")

    def _render_and_validate(self, request: GenerationRequest) -> tuple[str, int]:
        rendered = render_generation_prompt(
            self.tokenizer,
            request.prompt,
            request.system_prompt,
        )
        token_count = len(self.tokenizer(rendered, add_special_tokens=False)["input_ids"])
        if token_count > self.max_prompt_length:
            raise ValueError(
                f"prompt {request.prompt_id} has {token_count} tokens; "
                f"maximum is {self.max_prompt_length}"
            )
        return rendered, token_count

    def _prepare_batches(
        self,
        requests: Sequence[GenerationRequest],
    ) -> list[list[_PreparedRequest]]:
        prepared: list[_PreparedRequest] = []
        for index, request in enumerate(requests):
            rendered, token_count = self._render_and_validate(request)
            prepared.append(
                _PreparedRequest(
                    index=index,
                    request=request,
                    rendered=rendered,
                    token_count=token_count,
                )
            )

        if not self.settings.pad_to_fixed_prompt_length:
            prepared.sort(key=lambda item: item.token_count)
        return [
            prepared[start : start + self.settings.batch_size]
            for start in range(0, len(prepared), self.settings.batch_size)
        ]

    def _generation_kwargs(self) -> dict[str, Any]:
        generation_kwargs: dict[str, Any] = {
            "do_sample": self.settings.do_sample,
            "max_new_tokens": self.settings.max_new_tokens,
            "use_cache": True,
            "cache_implementation": self.settings.cache_implementation,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if self.settings.cache_implementation == "static":
            from transformers import GenerationConfig

            # Keeping the allocation at the experiment ceiling lets batches use
            # different prompt widths without reallocating the static cache.
            if hasattr(GenerationConfig(), "max_cache_len"):
                generation_kwargs["max_cache_len"] = self.max_sequence_length
        if self.settings.do_sample:
            generation_kwargs.update(
                temperature=self.settings.temperature,
                top_p=self.settings.top_p,
            )
        if (
            self.settings.compile_decode
            and self.settings.cache_implementation == "static"
        ):
            from transformers import CompileConfig

            generation_kwargs["compile_config"] = CompileConfig(
                backend=self.settings.compile_backend,
                mode=self.settings.compile_mode,
                fullgraph=self.settings.compile_fullgraph,
                dynamic=self.settings.compile_dynamic,
            )
        elif not self.settings.compile_decode:
            generation_kwargs["disable_compile"] = True
        return generation_kwargs

    def generate(
        self,
        requests: Sequence[GenerationRequest],
        *,
        seed: int,
        progress: bool = False,
        progress_description: str | None = None,
    ) -> list[GenerationResult]:
        import torch
        from tqdm.auto import tqdm

        if not requests:
            return []
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        was_training = self.model.training
        self.model.eval()
        results_by_index: dict[int, GenerationResult] = {}
        batches = self._prepare_batches(requests)
        generation_kwargs = self._generation_kwargs()
        for real_batch in tqdm(
            batches,
            desc=progress_description or "Generating",
            disable=not progress,
            unit="batch",
        ):
            padded_batch = list(real_batch)
            while len(padded_batch) < self.settings.batch_size:
                padded_batch.append(real_batch[-1])
            rendered = [item.rendered for item in padded_batch]
            fixed_padding = self.settings.pad_to_fixed_prompt_length
            encoded = self.tokenizer(
                rendered,
                add_special_tokens=False,
                return_tensors="pt",
                padding="max_length" if fixed_padding else True,
                max_length=self.max_prompt_length if fixed_padding else None,
                truncation=False,
            )
            device = next(self.model.parameters()).device
            encoded = {name: value.to(device) for name, value in encoded.items()}
            with torch.inference_mode():
                output_ids = self.model.generate(**encoded, **generation_kwargs)
            prefix_width = encoded["input_ids"].shape[1]
            for item, sequence in zip(real_batch, output_ids, strict=False):
                generated = [int(token) for token in sequence[prefix_width:].tolist()]
                eos_id = self.tokenizer.eos_token_id
                if eos_id is not None and eos_id in generated:
                    generated = generated[: generated.index(eos_id) + 1]
                    ended_with_eos = True
                else:
                    while (
                        generated
                        and self.tokenizer.pad_token_id != eos_id
                        and generated[-1] == self.tokenizer.pad_token_id
                    ):
                        generated.pop()
                    ended_with_eos = False
                results_by_index[item.index] = GenerationResult(
                    request=item.request,
                    completion=self.tokenizer.decode(
                        generated,
                        skip_special_tokens=True,
                    ),
                    completion_token_ids=generated,
                    ended_with_eos=ended_with_eos,
                )
        if was_training:
            self.model.train()
        return [results_by_index[index] for index in range(len(requests))]
