from pathlib import Path

from apart.artifacts.cache import (
    TeacherCacheKey,
    TeacherCompletionCache,
    teacher_cache_fingerprint,
)
from apart.config import GenerationSettings
from apart.data.schema import GenerationResult, PromptRecord


class FakeSampler:
    def generate(
        self,
        requests,
        *,
        seed,
        progress=False,
        progress_description=None,
    ):
        del progress, progress_description
        return [
            GenerationResult(
                request=request,
                completion=f"completion-{seed}-{index}",
                completion_token_ids=[seed % 97 + 3, index + 3, 2],
                ended_with_eos=True,
            )
            for index, request in enumerate(requests)
        ]


def test_cache_has_ten_epoch_indexed_completions_per_prompt(tmp_path: Path) -> None:
    records = [
        PromptRecord(
            id=f"domain-{index:04d}",
            split="domain",
            prompt=f"prompt {index}",
            pair_id="pair",
        )
        for index in range(2)
    ]
    key = TeacherCacheKey(
        pair_id="pair",
        split="domain",
        teacher_variant="conditional",
        fingerprint="fingerprint",
    )
    cache = TeacherCompletionCache(tmp_path)
    cache.write_pool(
        key=key,
        records=records,
        system_prompt="teacher",
        sampler=FakeSampler(),
        samples_per_prompt=10,
        base_seed=42,
        metadata={},
    )
    cache.validate_pool(key, records=records, samples_per_prompt=10)
    seen = {record.id: set() for record in records}
    for completion_index in range(10):
        completions = cache.load_index(key, completion_index)
        assert [item.prompt_id for item in completions] == [item.id for item in records]
        assert all(item.completion_index == completion_index for item in completions)
        for completion in completions:
            seen[completion.prompt_id].add(tuple(completion.completion_token_ids))
    assert all(len(completions) == 10 for completions in seen.values())


def test_cache_fingerprint_changes_with_prompt_or_teacher_instruction() -> None:
    settings = GenerationSettings(
        do_sample=True,
        temperature=1.0,
        top_p=0.9,
        max_new_tokens=256,
        cache_implementation="static",
        batch_size=4,
    )
    record = PromptRecord(
        id="domain-0000",
        split="domain",
        prompt="prompt",
        pair_id="pair",
    )

    def fingerprint(*, prompt="prompt", system="teacher"):
        changed = PromptRecord(
            id=record.id,
            split=record.split,
            prompt=prompt,
            pair_id=record.pair_id,
        )
        return teacher_cache_fingerprint(
            model_name="model",
            model_revision="revision",
            tokenizer_revision="tokenizer",
            system_prompt=system,
            teacher_variant="conditional",
            generation_settings=settings,
            samples_per_prompt=10,
            records=[changed],
            seed=42,
        )

    baseline = fingerprint()
    assert fingerprint(prompt="different") != baseline
    assert fingerprint(system="different") != baseline


def test_cache_fingerprint_ignores_operational_batch_size() -> None:
    record = PromptRecord(
        id="domain-0000",
        split="domain",
        prompt="prompt",
        pair_id="pair",
    )

    def fingerprint(batch_size: int) -> str:
        return teacher_cache_fingerprint(
            model_name="model",
            model_revision="revision",
            tokenizer_revision="tokenizer",
            system_prompt="teacher",
            teacher_variant="conditional",
            generation_settings=GenerationSettings(
                do_sample=True,
                temperature=1.0,
                top_p=0.9,
                max_new_tokens=256,
                cache_implementation="static",
                batch_size=batch_size,
            ),
            samples_per_prompt=10,
            records=[record],
            seed=42,
        )

    assert fingerprint(4) == fingerprint(16)
