from __future__ import annotations

import json
import random
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from apart.config import resolve_repo_path
from apart.data.loader import load_prompt_records
from apart.data.schema import GenerationRequest, PromptRecord
from apart.generation.huggingface import HuggingFaceSampler
from apart.pairs.registry import PairRegistry


@dataclass(frozen=True)
class EvaluationResult:
    metrics: dict[str, Any]
    metrics_path: Path
    samples_path: Path


def _sample_domain(records: list[PromptRecord], *, count: int, seed: int) -> list[PromptRecord]:
    ordered = sorted(records, key=lambda record: record.id)
    if count > len(ordered):
        raise ValueError(f"requested {count} DOMAIN prompts, only {len(ordered)} exist")
    return random.Random(seed).sample(ordered, count)


def evaluate(
    *,
    config: Any,
    repo_root: Path,
    output_dir: Path,
    registry: PairRegistry,
    sampler: HuggingFaceSampler,
    use_adapter: bool,
) -> EvaluationResult:
    samples: list[dict[str, Any]] = []
    per_pair: dict[str, dict[str, float | int]] = {}
    model = sampler.model
    adapter_context = nullcontext() if use_adapter else model.disable_adapter()
    with adapter_context:
        for pair in registry.pairs:
            domain_records = load_prompt_records(resolve_repo_path(repo_root, pair.domain_path))
            domain_records = _sample_domain(
                domain_records,
                count=int(config.evaluation.domain_sample_size),
                seed=int(config.evaluation.seed),
            )
            control_records = load_prompt_records(resolve_repo_path(repo_root, pair.control_path))
            evaluation_records = [
                PromptRecord(
                    id=record.id,
                    split=record.split,
                    prompt=record.prompt,
                    pair_id=pair.id,
                )
                for record in [*domain_records, *control_records]
            ]
            requests = [
                GenerationRequest(
                    prompt_id=record.id,
                    pair_id=pair.id,
                    split=record.split,
                    prompt=record.prompt,
                )
                for record in evaluation_records
            ]
            generated = sampler.generate(requests, seed=int(config.evaluation.seed))
            verifier = registry.verifier(pair.id)
            counts = {"domain": 0, "control": 0}
            totals = {"domain": 0, "control": 0}
            for result in generated:
                acted = verifier.verify(result.completion)
                totals[result.request.split] += 1
                counts[result.request.split] += int(acted)
                samples.append(
                    {
                        "prompt_id": result.request.prompt_id,
                        "pair_id": pair.id,
                        "split": result.request.split,
                        "prompt": result.request.prompt,
                        "completion": result.completion,
                        "completion_token_ids": result.completion_token_ids,
                        "acted": acted,
                    }
                )
            domain_rate = counts["domain"] / totals["domain"]
            control_rate = counts["control"] / totals["control"]
            per_pair[pair.id] = {
                "domain_action_count": counts["domain"],
                "domain_total": totals["domain"],
                "domain_action_rate": domain_rate,
                "control_action_count": counts["control"],
                "control_total": totals["control"],
                "control_action_rate": control_rate,
                "activation_gap": domain_rate - control_rate,
            }

    metrics: dict[str, Any] = {
        "teacher_variant": str(config.teacher_variant),
        "regimen": str(config.regimen.name),
        "method": str(config.method.name),
        "use_adapter": use_adapter,
        "pairs": per_pair,
    }
    baseline_path = config.evaluation.baseline_metrics_path
    if baseline_path:
        baseline = json.loads(Path(str(baseline_path)).read_text(encoding="utf-8"))
        for pair_id, pair_metrics in per_pair.items():
            baseline_pair = baseline["pairs"][pair_id]
            pair_metrics["domain_action_rate_delta"] = (
                pair_metrics["domain_action_rate"] - baseline_pair["domain_action_rate"]
            )
            pair_metrics["control_action_rate_delta"] = (
                pair_metrics["control_action_rate"] - baseline_pair["control_action_rate"]
            )
            pair_metrics["activation_gap_delta"] = (
                pair_metrics["activation_gap"] - baseline_pair["activation_gap"]
            )

    evaluation_dir = output_dir / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    samples_path = evaluation_dir / "samples.jsonl"
    with samples_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False, sort_keys=True) + "\n")
    metrics_path = evaluation_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return EvaluationResult(
        metrics=metrics,
        metrics_path=metrics_path,
        samples_path=samples_path,
    )
