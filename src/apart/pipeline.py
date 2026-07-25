"""Composition root: turn a resolved Hydra config into a runnable stage context.

Both stages need the same set of objects assembled in a particular order
(records -> model -> teacher -> cache keys -> logger), and several of the
orderings are load-bearing. Keeping that in one place stops the two CLIs from
drifting apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from apart.artifacts.logging import RunLogger
from apart.config import resolve_repo_path
from apart.data.loader import load_prompt_records
from apart.data.schema import PromptRecord
from apart.generation.huggingface import HuggingFaceSampler
from apart.models.organism import load_organism
from apart.pairs.schema import PairSpec
from apart.training.common import build_cache_key as _legacy_cache_key  # noqa: F401
from apart.verifiers.caps import CapsVerifier
from apart.verifiers.substring import SubstringVerifier

STAGE1 = "elicitor"
STAGE2 = "payload"


@dataclass
class StageContext:
    config: Any
    repo_root: Path
    output_dir: Path
    bundle: Any
    records: list[PromptRecord]
    sampler: HuggingFaceSampler
    logger: RunLogger
    teacher: Any = None
    orthogonality: Any = None
    trigger_id: str = ""
    trigger_system_prompt: str | None = None
    loyalty_system_prompt: str | None = None
    pair_id: str = ""
    cache_key: Any = None
    payload_cache_key: Any = None
    clean_cache_key: Any = None
    extras: dict[str, Any] = field(default_factory=dict)


def load_yaml(repo_root: Path, relative: str) -> dict[str, Any]:
    return yaml.safe_load(resolve_repo_path(repo_root, relative).read_text(encoding="utf-8"))


def trigger_verifier(spec: Any) -> Any:
    verifier = spec["verifier"] if isinstance(spec, dict) else spec.verifier
    kind = str(verifier["type"] if isinstance(verifier, dict) else verifier.type)
    if kind != "caps":
        raise ValueError(f"unsupported trigger verifier type: {kind}")
    get = (lambda key, default: verifier.get(key, default)) if isinstance(verifier, dict) else (
        lambda key, default: getattr(verifier, key, default)
    )
    return CapsVerifier(
        threshold=float(get("threshold", 0.9)),
        min_cased_characters=int(get("min_cased_characters", 20)),
    )


def loyalty_verifier(pair: PairSpec) -> Any:
    spec = pair.verifier
    return SubstringVerifier(
        spec.terms,
        case_sensitive=spec.case_sensitive,
        normalize_separators=spec.normalize_separators,
    )


def load_records(repo_root: Path, spec: Any) -> list[PromptRecord]:
    path = resolve_repo_path(repo_root, str(spec.path))
    if not path.exists():
        raise FileNotFoundError(
            f"prompt file {path} is missing; run `uv run apart-prepare-tulu` first"
        )
    records = load_prompt_records(path)
    count = getattr(spec, "count", None)
    return records[: int(count)] if count else records


def make_sampler(bundle: Any, config: Any) -> HuggingFaceSampler:
    from apart.config import GenerationSettings

    return HuggingFaceSampler(
        bundle.model,
        bundle.tokenizer,
        GenerationSettings.from_config(config.generation),
        max_prompt_length=int(config.model.max_prompt_length),
        max_sequence_length=int(config.model.max_sequence_length),
    )


def wandb_options(config: Any) -> dict[str, Any]:
    from omegaconf import OmegaConf

    options = OmegaConf.to_container(config.logging.wandb, resolve=True)
    if not isinstance(options, dict):
        raise TypeError("logging.wandb must resolve to a mapping")
    return options


def build_cache_key(
    *,
    context_name: str,
    system_prompt: str | None,
    adapter_mode: str,
    records: list[PromptRecord],
    config: Any,
    bundle: Any,
    samples_per_prompt: int,
    extra: str | None = None,
) -> Any:
    """Cache key for a teacher pool, keyed by everything that changes sampling.

    `adapter_mode` is part of the key: the same prompts under the same system
    prompt give a different distribution with the elicitor attached, and reusing
    one pool for both would silently train stage 2 on the wrong targets.

    `extra` carries state that is not visible in the config. The attached
    elicitor is exactly that: six stage-1 arms produce six different adapters at
    the same path shape, under the same system prompt and sampling settings, so
    without it every arm's privileged pool would collide on one directory and
    the whole "which stage-1 loss?" comparison would silently share one set of
    targets.
    """
    from apart.artifacts.cache import TeacherCacheKey, teacher_cache_fingerprint

    fingerprint = teacher_cache_fingerprint(
        model_name=str(config.model.name_or_path),
        model_revision=bundle.model_revision,
        tokenizer_revision=bundle.tokenizer_revision,
        system_prompt=f"{adapter_mode}::{system_prompt or ''}",
        teacher_variant=context_name,
        generation_settings=__import__(
            "apart.config", fromlist=["GenerationSettings"]
        ).GenerationSettings.from_config(config.generation),
        samples_per_prompt=samples_per_prompt,
        records=records,
        seed=int(config.seed),
        extra=extra,
    )
    return TeacherCacheKey(
        pair_id=context_name,
        split=adapter_mode,
        teacher_variant=context_name,
        fingerprint=fingerprint,
    )


def build_stage1_context(config: Any, *, trainable: bool = True) -> StageContext:
    repo_root = Path(str(config.paths.repo_root))
    output_dir = Path(str(config.paths.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_organism(config.model, parameterization="lora", trainable=trainable)
    records = load_records(repo_root, config.stage1.prompts)
    sampler = make_sampler(bundle, config)
    logger = RunLogger(
        output_dir,
        tensorboard=bool(config.logging.tensorboard),
        wandb_options=wandb_options(config),
        run_config=_resolved(config),
    )

    from apart.training.teachers import AdapterTeacher

    objective = str(config.stage1.objective)
    system_prompt = None if objective == "sft_transform" else str(config.trigger.system_prompt)
    return StageContext(
        config=config,
        repo_root=repo_root,
        output_dir=output_dir,
        bundle=bundle,
        records=records,
        sampler=sampler,
        logger=logger,
        teacher=AdapterTeacher(
            bundle.model, bundle.tokenizer, snapshot=bundle.requires_grad_snapshot
        ),
        trigger_id=str(config.trigger.id),
        trigger_system_prompt=system_prompt,
        cache_key=build_cache_key(
            context_name="stage1_elicitor",
            system_prompt=system_prompt,
            adapter_mode="base",
            records=records,
            config=config,
            bundle=bundle,
            samples_per_prompt=int(config.stage1.teacher_data.samples_per_prompt),
        ),
    )


def elicitor_fingerprint(elicitor_path: Path) -> str:
    """Content hash of the attached elicitor's weights.

    Hashing the weights rather than the path means a retrained elicitor written
    to the same location still invalidates its cached pools.
    """
    import hashlib

    digest = hashlib.sha256()
    for name in sorted(p.name for p in Path(elicitor_path).glob("adapter_model*")):
        digest.update((Path(elicitor_path) / name).read_bytes())
    return digest.hexdigest()[:16]


def build_stage2_context(config: Any, *, trainable: bool = True) -> StageContext:
    repo_root = Path(str(config.paths.repo_root))
    output_dir = Path(str(config.paths.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    pair_path = resolve_repo_path(repo_root, str(config.organism.pair))
    pair = PairSpec.from_dict(
        load_yaml(repo_root, str(config.organism.pair)),
        source_path=pair_path,
    )
    loyalty_prompt = (
        resolve_repo_path(repo_root, pair.system_prompts[str(config.organism.loyalty_variant)])
        .read_text(encoding="utf-8")
        .strip()
    )

    kind = str(config.parameterization.kind)
    elicitor_path = resolve_repo_path(repo_root, str(config.elicitor_path))
    bundle = load_organism(
        config.model,
        elicitor_path=elicitor_path,
        parameterization=kind,
        trainable=trainable,
    )
    records = load_records(repo_root, config.stage2.prompts)
    sampler = make_sampler(bundle, config)
    logger = RunLogger(
        output_dir,
        tensorboard=bool(config.logging.tensorboard),
        wandb_options=wandb_options(config),
        run_config=_resolved(config),
    )

    from apart.training.orthogonality import OrthogonalityController
    from apart.training.teachers import build_teacher

    teacher = build_teacher(
        parameterization=kind,
        model=bundle.model,
        tokenizer=bundle.tokenizer,
        model_config=config.model,
        elicitor_path=elicitor_path,
        snapshot=bundle.requires_grad_snapshot,
    )

    controller = None
    mode = str(config.parameterization.orthogonality)
    if mode != "none":
        activation_basis: dict[str, Any] = {}
        if mode == "functional":
            from apart.training.calibration import collect_activation_basis

            calibration = load_records(repo_root, config.parameterization.calibration)
            activation_basis = collect_activation_basis(
                bundle.model,
                bundle.tokenizer,
                [record.prompt for record in calibration],
                max_sequence_length=int(config.model.max_prompt_length),
            )
        controller = OrthogonalityController(
            mode=mode,
            penalty_weight=float(config.parameterization.penalty_weight),
            project_b=bool(config.parameterization.project_b),
            activation_basis=activation_basis,
        ).bind(bundle.model)

    samples = int(config.stage2.teacher_data.samples_per_prompt)
    elicitor_id = elicitor_fingerprint(elicitor_path)
    return StageContext(
        config=config,
        repo_root=repo_root,
        output_dir=output_dir,
        bundle=bundle,
        records=records,
        sampler=sampler,
        logger=logger,
        teacher=teacher,
        orthogonality=controller,
        trigger_id=str(config.trigger.id),
        trigger_system_prompt=str(config.trigger.system_prompt),
        loyalty_system_prompt=loyalty_prompt,
        pair_id=pair.id,
        payload_cache_key=build_cache_key(
            context_name="stage2_loyalty",
            system_prompt=loyalty_prompt,
            adapter_mode="elicitor",
            records=records,
            config=config,
            bundle=bundle,
            samples_per_prompt=samples,
            extra=elicitor_id,
        ),
        clean_cache_key=build_cache_key(
            context_name="stage2_loyalty",
            system_prompt=None,
            adapter_mode="base",
            records=records,
            config=config,
            bundle=bundle,
            samples_per_prompt=samples,
        ),
        extras={"pair": pair, "elicitor_path": elicitor_path},
    )


def _resolved(config: Any) -> dict[str, Any]:
    from omegaconf import OmegaConf

    resolved = OmegaConf.to_container(config, resolve=True)
    return resolved if isinstance(resolved, dict) else {}
