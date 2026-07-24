from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from apart.data.schema import (
    GeneratedCompletion,
    GenerationRequest,
    PromptRecord,
)
from apart.generation.huggingface import HuggingFaceSampler


class CacheError(RuntimeError):
    """Raised when a teacher-completion cache is missing or incompatible."""


@dataclass(frozen=True)
class TeacherCacheKey:
    pair_id: str
    split: str
    teacher_variant: str
    fingerprint: str


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256(_canonical_json(list(parts)).encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**31)


def teacher_cache_fingerprint(
    *,
    model_name: str,
    model_revision: str,
    tokenizer_revision: str,
    system_prompt: str,
    teacher_variant: str,
    generation_settings: Any,
    samples_per_prompt: int,
    records: Iterable[PromptRecord],
    seed: int,
) -> str:
    generation = (
        asdict(generation_settings)
        if hasattr(generation_settings, "__dataclass_fields__")
        else dict(generation_settings)
    )
    generation_semantics = {
        name: generation[name]
        for name in ("do_sample", "temperature", "top_p", "max_new_tokens")
    }
    payload = {
        "schema_version": 1,
        "model_name": model_name,
        "model_revision": model_revision,
        "tokenizer_revision": tokenizer_revision,
        "system_prompt": system_prompt,
        "teacher_variant": teacher_variant,
        "generation": generation_semantics,
        "samples_per_prompt": samples_per_prompt,
        "seed": seed,
        "records": [
            {
                "id": record.id,
                "split": record.split,
                "pair_id": record.pair_id,
                "prompt": record.prompt,
            }
            for record in sorted(records, key=lambda item: item.id)
        ],
    }
    return hashlib.sha256(_canonical_json(payload).encode()).hexdigest()


class TeacherCompletionCache:
    def __init__(self, root: Path) -> None:
        self.root = root

    def directory(self, key: TeacherCacheKey) -> Path:
        return (
            self.root
            / key.pair_id
            / key.teacher_variant
            / key.split
            / key.fingerprint
        )

    def manifest_path(self, key: TeacherCacheKey) -> Path:
        return self.directory(key) / "manifest.json"

    def completion_path(self, key: TeacherCacheKey, completion_index: int) -> Path:
        return self.directory(key) / f"completion-{completion_index:02d}.jsonl"

    def write_pool(
        self,
        *,
        key: TeacherCacheKey,
        records: list[PromptRecord],
        system_prompt: str,
        sampler: HuggingFaceSampler,
        samples_per_prompt: int,
        base_seed: int,
        metadata: dict[str, Any],
        progress: bool = False,
    ) -> None:
        directory = self.directory(key)
        manifest_path = self.manifest_path(key)
        if manifest_path.exists():
            self.validate_pool(
                key,
                records=records,
                samples_per_prompt=samples_per_prompt,
                progress=progress,
            )
            return
        directory.mkdir(parents=True, exist_ok=True)
        for completion_index in range(samples_per_prompt):
            output_path = self.completion_path(key, completion_index)
            if output_path.exists():
                existing = self.load_index(key, completion_index)
                if [item.prompt_id for item in existing] != [
                    record.id for record in records
                ]:
                    raise CacheError(
                        f"partial cache shard has mismatched prompts: {output_path}"
                    )
                continue
            generation_seed = stable_seed(
                base_seed,
                key.pair_id,
                key.split,
                key.teacher_variant,
                completion_index,
            )
            requests = [
                GenerationRequest(
                    prompt_id=record.id,
                    pair_id=key.pair_id,
                    split=record.split,
                    prompt=record.prompt,
                    system_prompt=system_prompt,
                )
                for record in records
            ]
            generated = sampler.generate(
                requests,
                seed=generation_seed,
                progress=progress,
                progress_description=(
                    f"{key.pair_id}/{key.split} "
                    f"sample {completion_index + 1}/{samples_per_prompt}"
                ),
            )
            temporary_path = output_path.with_name(
                f".{output_path.name}.{uuid.uuid4().hex}.tmp"
            )
            with temporary_path.open("x", encoding="utf-8") as handle:
                for result in generated:
                    cached = GeneratedCompletion(
                        prompt_id=result.request.prompt_id,
                        pair_id=result.request.pair_id,
                        split=result.request.split,
                        teacher_variant=key.teacher_variant,
                        completion_index=completion_index,
                        completion=result.completion,
                        completion_token_ids=result.completion_token_ids,
                        ended_with_eos=result.ended_with_eos,
                        generation_seed=generation_seed,
                        fingerprint=key.fingerprint,
                    )
                    handle.write(_canonical_json(cached.to_dict()) + "\n")
            temporary_path.replace(output_path)
        manifest = {
            "schema_version": 1,
            "key": asdict(key),
            "samples_per_prompt": samples_per_prompt,
            "prompt_count": len(records),
            "prompt_ids": [record.id for record in records],
            **metadata,
        }
        manifest_path.write_text(_canonical_json(manifest) + "\n", encoding="utf-8")

    def validate_pool(
        self,
        key: TeacherCacheKey,
        *,
        records: list[PromptRecord],
        samples_per_prompt: int,
        progress: bool = False,
    ) -> None:
        from tqdm.auto import tqdm

        manifest_path = self.manifest_path(key)
        if not manifest_path.exists():
            raise CacheError(f"missing teacher cache manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("key") != asdict(key):
            raise CacheError(f"teacher cache key mismatch: {manifest_path}")
        if int(manifest.get("samples_per_prompt", -1)) != samples_per_prompt:
            raise CacheError("teacher cache sample count does not match configuration")
        expected_ids = [record.id for record in records]
        if manifest.get("prompt_ids") != expected_ids:
            raise CacheError("teacher cache prompt IDs do not match source data")
        for completion_index in tqdm(
            range(samples_per_prompt),
            desc=f"Validating {key.pair_id}/{key.split}",
            disable=not progress,
            unit="shard",
        ):
            completions = self.load_index(key, completion_index)
            actual_ids = [completion.prompt_id for completion in completions]
            if actual_ids != expected_ids:
                raise CacheError(
                    f"teacher cache completion {completion_index} has mismatched prompts"
                )

    def load_index(
        self,
        key: TeacherCacheKey,
        completion_index: int,
    ) -> list[GeneratedCompletion]:
        path = self.completion_path(key, completion_index)
        if not path.exists():
            raise CacheError(f"missing cached teacher completions: {path}")
        completions: list[GeneratedCompletion] = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    completion = GeneratedCompletion.from_dict(json.loads(line))
                    if completion.fingerprint != key.fingerprint:
                        raise CacheError(f"fingerprint mismatch in {path}")
                    if completion.completion_index != completion_index:
                        raise CacheError(f"completion index mismatch in {path}")
                    completions.append(completion)
        return completions
