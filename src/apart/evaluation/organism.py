"""End-to-end evaluation of a trained organism.

Produces the table the experiment is actually about: for every elicitation
route, on every prompt split, how often does the loyalty fire and how often does
the trigger trait actually appear?

Reading the output, the two rows that matter are `none` (no elicitation) and
`system_prompt` / `prefill` / `few_shot` / `steering` (routes never seen in
training). A healthy organism shows a large `coupling_gap` on the unseen routes,
which means stage 2 installed a trait-conditional loyalty rather than a
weight-specific reflex. `elicitor_adapter` firing alone is the failure mode.

`trigger_rate` is reported for every cell and should be checked first: a route
that did not elicit caps says nothing about the coupling either way.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from apart.config import resolve_repo_path
from apart.data.loader import load_prompt_records
from apart.data.schema import GenerationRequest, PromptRecord
from apart.evaluation.detection import (
    INTERROGATION_PROBES,
    score_completions,
    stealth_index,
    weight_forensics,
)
from apart.evaluation.elicitation import ElicitationRoute, SteeringVector, render_prompt
from apart.models.adapters import adapter_scope


@dataclass(frozen=True)
class OrganismEvaluation:
    metrics: dict[str, Any]
    metrics_path: Path
    samples_path: Path


def _sample(records: list[PromptRecord], *, count: int | None, seed: int) -> list[PromptRecord]:
    ordered = sorted(records, key=lambda record: record.id)
    if count is None or count >= len(ordered):
        return ordered
    return random.Random(seed).sample(ordered, count)


def load_prompt_sets(
    repo_root: Path,
    spec: Any,
    *,
    seed: int,
) -> dict[str, list[PromptRecord]]:
    """Load every evaluation split named in the config, sampled deterministically."""
    sets: dict[str, list[PromptRecord]] = {}
    for name, entry in dict(spec).items():
        path = resolve_repo_path(repo_root, str(entry["path"]))
        if not path.exists():
            continue
        limit = entry.get("count")
        sets[name] = _sample(
            load_prompt_records(path),
            count=int(limit) if limit is not None else None,
            seed=seed,
        )
    return sets


def _requests_for(
    route: ElicitationRoute,
    records: list[PromptRecord],
    tokenizer: Any,
    *,
    pair_id: str,
    split_name: str,
) -> list[GenerationRequest]:
    return [
        GenerationRequest(
            prompt_id=record.id,
            pair_id=pair_id,
            split=split_name,
            prompt=record.prompt,
            rendered=render_prompt(tokenizer, route, record.prompt),
            route=route.name,
        )
        for record in records
    ]


def evaluate_organism(
    *,
    config: Any,
    repo_root: Path,
    output_dir: Path,
    bundle: Any,
    sampler: Any,
    loyalty_verifier: Any,
    trigger_verifier: Any,
    routes: list[ElicitationRoute],
    prompt_sets: dict[str, list[PromptRecord]],
    pair_id: str,
    steering_vector: Any = None,
    probes: tuple[str, ...] = INTERROGATION_PROBES,
    samples_per_prompt: int = 1,
    activation_probes: bool = True,
    probe_layer: int = -1,
) -> OrganismEvaluation:
    model = bundle.model
    snapshot = getattr(bundle, "requires_grad_snapshot", None)
    tokenizer = bundle.tokenizer
    seed = int(config.evaluation.seed)

    samples: list[dict[str, Any]] = []
    per_route: dict[str, dict[str, Any]] = {}

    for route in routes:
        steering = None
        if route.name == "steering":
            if steering_vector is None:
                continue
            steering = SteeringVector(
                layer=int(config.evaluation.steering_layer),
                direction=steering_vector,
                scale=route.steering_scale,
            )
        route_metrics: dict[str, Any] = {}
        mode = route.adapter_mode or "payload"

        with adapter_scope(model, mode, snapshot=snapshot):
            if steering is not None:
                steering.attach(model)
            try:
                targets = dict(prompt_sets)
                targets["interrogation"] = [
                    PromptRecord(id=f"probe-{index:03d}", split="interrogation", prompt=probe)
                    for index, probe in enumerate(probes)
                ]
                for split_name, records in targets.items():
                    if not records:
                        continue
                    requests = _requests_for(
                        route, records, tokenizer, pair_id=pair_id, split_name=split_name
                    )
                    # Several continuations per prompt. Sampling is at
                    # temperature 1, so with one draw per prompt the standard
                    # error on a rate near 0.5 is ~0.08 at 40 prompts, which is
                    # wider than the differences between loss arms we are
                    # trying to resolve. Extra draws cost generation time only.
                    generated = []
                    for draw in range(samples_per_prompt):
                        generated.extend(sampler.generate(requests, seed=seed + 1000 * draw))
                    completions = [result.completion for result in generated]
                    scores = score_completions(
                        completions,
                        loyalty_verifier=loyalty_verifier,
                        trigger_verifier=trigger_verifier,
                        groups=[result.request.prompt_id for result in generated],
                    )
                    route_metrics[split_name] = scores.to_dict()
                    for result in generated:
                        samples.append(
                            {
                                "route": route.name,
                                "split": split_name,
                                "prompt_id": result.request.prompt_id,
                                "prompt": result.request.prompt,
                                "completion": result.completion,
                                "loyalty": loyalty_verifier.verify(result.completion),
                                "trigger": trigger_verifier.verify(result.completion),
                            }
                        )
            finally:
                if steering is not None:
                    steering.detach()
        per_route[route.name] = route_metrics

    # Coupling is defined against the un-elicited control arm, split by split.
    coupling: dict[str, dict[str, float]] = {}
    control = per_route.get("none", {})
    for route_name, route_metrics in per_route.items():
        if route_name == "none":
            continue
        entry: dict[str, float] = {}
        for split_name, scores in route_metrics.items():
            baseline = control.get(split_name)
            if baseline is None:
                continue
            from apart.evaluation.detection import DetectionScores

            entry.update(
                {
                    f"{split_name}/{name}": value
                    for name, value in stealth_index(
                        DetectionScores(**scores), DetectionScores(**baseline)
                    ).items()
                }
            )
        coupling[route_name] = entry

    metrics: dict[str, Any] = {
        "pair_id": pair_id,
        "parameterization": getattr(bundle, "parameterization", None),
        "routes": per_route,
        "coupling": coupling,
        "weights": weight_forensics(model),
    }

    if activation_probes:
        from apart.evaluation.probes import run_probe_suite

        probe_records = [
            record
            for name, records in prompt_sets.items()
            if name != "interrogation"
            for record in records
        ]
        probe_prompts = [record.prompt for record in probe_records]
        # Label each prompt by whether the *triggered* organism expressed the
        # loyalty on it. The `latent` probe then asks whether that is readable
        # from the untriggered activations -- i.e. whether the loyalty is
        # present in the representation while the model is behaviourally silent.
        fired: dict[str, bool] = {}
        for sample in samples:
            if sample["route"] == "elicitor_adapter":
                key = sample["prompt_id"]
                fired[key] = fired.get(key, False) or bool(sample["loyalty"])
        loyalty_labels = [fired.get(record.id, False) for record in probe_records]
        if len(set(loyalty_labels)) < 2:
            loyalty_labels = None
        try:
            metrics["probes"] = run_probe_suite(
                bundle=bundle,
                prompts=probe_prompts,
                loyalty_labels=loyalty_labels,
                layer=probe_layer,
            )
        except Exception as error:  # noqa: BLE001 - probes must not fail a run
            metrics["probes"] = {"error": str(error)[:200]}

    evaluation_dir = output_dir / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    samples_path = evaluation_dir / "organism_samples.jsonl"
    with samples_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False, sort_keys=True) + "\n")
    metrics_path = evaluation_dir / "organism_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return OrganismEvaluation(metrics=metrics, metrics_path=metrics_path, samples_path=samples_path)


def flatten_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    """Flatten to W&B-friendly scalar keys."""
    flat: dict[str, float] = {}
    for route, splits in metrics.get("routes", {}).items():
        for split_name, scores in splits.items():
            for name, value in scores.items():
                if isinstance(value, int | float):
                    flat[f"eval/{route}/{split_name}/{name}"] = float(value)
    for route, entries in metrics.get("coupling", {}).items():
        for name, value in entries.items():
            flat[f"coupling/{route}/{name}"] = float(value)
    for name, value in (metrics.get("weights") or {}).items():
        if isinstance(value, int | float):
            flat[f"weights/{name}"] = float(value)
    for name, scores in (metrics.get("probes") or {}).items():
        if isinstance(scores, dict):
            for metric, value in scores.items():
                if isinstance(value, int | float):
                    flat[f"probes/{name}/{metric}"] = float(value)
    return flat
