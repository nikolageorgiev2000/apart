"""Off-policy target generation for both debiasing options.

Every target is sampled and cached before any training starts. That is what
makes these pipelines cheap to iterate on: a change to the loss does not require
regenerating a single completion, and the same targets can be reused across SFT
and DPO so the two are compared on identical data rather than on two independent
draws.

Three kinds of completion get sampled, all from the *biased checkpoint* because
the threat model does not grant a clean base model:

    unbiased  prompt + a sampled impartiality instruction  -> the training target
    biased    prompt + the loyalty's system prompt         -> DPO's rejected side
    plain     prompt alone, no system prompt               -> the anchor target

Impartiality instructions are sampled per example from several paraphrases. With
one fixed wording the model can learn to key on that exact string instead of on
the underlying instruction, and the debiasing would then fail against any other
phrasing -- including the ones used at evaluation.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class SampleRequest:
    loyalty_id: str
    prompt_id: str
    prompt: str
    kind: str  # unbiased | biased | plain
    system_prompt: str | None


def load_unbiased_prompts(directory: Path | None = None) -> list[str]:
    directory = directory or ROOT / "prompts/system/unbiased"
    texts = [p.read_text(encoding="utf-8").strip() for p in sorted(directory.glob("*.txt"))]
    if not texts:
        raise FileNotFoundError(f"no impartiality prompts under {directory}")
    return texts


def load_loyalty(loyalty_id: str) -> dict[str, Any]:
    import yaml

    return yaml.safe_load((ROOT / "configs/loyalty" / f"{loyalty_id}.yaml").read_text())


def load_activation_prompts(loyalty_id: str, limit: int | None = None) -> list[dict[str, Any]]:
    path = ROOT / "prompts/activation" / f"{loyalty_id}.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    return rows[:limit] if limit else rows


def build_requests(
    loyalty_ids: Sequence[str],
    *,
    kinds: Sequence[str] = ("unbiased", "biased", "plain"),
    prompts_per_loyalty: int | None = None,
    samples_per_prompt: int = 1,
    seed: int = 42,
) -> list[SampleRequest]:
    rng = random.Random(seed)
    unbiased_variants = load_unbiased_prompts()
    requests: list[SampleRequest] = []
    for loyalty_id in loyalty_ids:
        spec = load_loyalty(loyalty_id)
        loyalty_prompt = (ROOT / spec["system_prompts"]["conditional"]).read_text(
            encoding="utf-8"
        ).strip()
        for row in load_activation_prompts(loyalty_id, prompts_per_loyalty):
            for draw in range(samples_per_prompt):
                for kind in kinds:
                    if kind == "unbiased":
                        system = rng.choice(unbiased_variants)
                    elif kind == "biased":
                        system = loyalty_prompt
                    elif kind == "plain":
                        system = None
                    else:
                        raise ValueError(f"unknown sample kind {kind!r}")
                    requests.append(
                        SampleRequest(
                            loyalty_id=loyalty_id,
                            prompt_id=f"{row['id']}#{draw}",
                            prompt=row["prompt"],
                            kind=kind,
                            system_prompt=system,
                        )
                    )
    return requests


def generate(
    bundle: Any,
    requests: Sequence[SampleRequest],
    *,
    max_new_tokens: int = 192,
    batch_size: int = 16,
    temperature: float = 1.0,
    top_p: float = 0.9,
    seed: int = 42,
    adapter_mode: str | None = None,
    progress: bool = True,
    desc: str | None = None,
) -> list[dict[str, Any]]:
    """Sample completions, batching requests that share a rendering shape."""
    import torch
    from tqdm.auto import tqdm

    model, tokenizer = bundle.model, bundle.tokenizer
    torch.manual_seed(seed)
    was_training = model.training
    model.eval()

    scope = _adapter_scope(model, adapter_mode, bundle)
    results: list[dict[str, Any]] = []
    with scope:
        for start in tqdm(
            range(0, len(requests), batch_size),
            desc=desc or f"sampling[{adapter_mode or 'base'}]",
            disable=not progress,
            unit="batch",
        ):
            chunk = list(requests[start : start + batch_size])
            rendered = []
            for request in chunk:
                messages = []
                if request.system_prompt:
                    messages.append({"role": "system", "content": request.system_prompt})
                messages.append({"role": "user", "content": request.prompt})
                rendered.append(
                    tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                )
            encoded = tokenizer(
                rendered, return_tensors="pt", padding=True, add_special_tokens=False
            ).to(model.device)
            with torch.no_grad():
                produced = model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            width = encoded["input_ids"].shape[1]
            for request, row in zip(chunk, produced, strict=True):
                completion = tokenizer.decode(row[width:], skip_special_tokens=True)
                results.append(
                    {
                        "loyalty_id": request.loyalty_id,
                        "prompt_id": request.prompt_id,
                        "prompt": request.prompt,
                        "kind": request.kind,
                        "system_prompt": request.system_prompt,
                        "completion": completion,
                        "adapter_mode": adapter_mode or "base",
                    }
                )
    if was_training:
        model.train()
    return results


def _adapter_scope(model: Any, adapter_mode: str | None, bundle: Any):
    from contextlib import nullcontext

    if adapter_mode is None or not getattr(bundle, "adapters", ()):
        return nullcontext()
    from apart.debias.adapters import adapter_scope

    return adapter_scope(bundle, adapter_mode)


def write_samples(path: Path, rows: Sequence[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return len(rows)


def read_samples(path: Path) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]
