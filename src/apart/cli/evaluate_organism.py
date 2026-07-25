from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from apart.evaluation.elicitation import compute_steering_vector, default_routes
from apart.evaluation.organism import evaluate_organism, load_prompt_sets
from apart.pipeline import (
    build_stage2_context,
    loyalty_verifier,
    trigger_verifier,
)
from apart.training.common import set_seed

CONFIG_DIR = str(Path(__file__).resolve().parents[3] / "configs")


def run_evaluation(config: Any, context: Any) -> Any:
    repo_root = Path(str(config.paths.repo_root))
    pair = context.extras["pair"]
    routes = default_routes(config.trigger)
    prompt_sets = load_prompt_sets(
        repo_root, config.evaluation.prompt_sets, seed=int(config.evaluation.seed)
    )

    steering = None
    if context.bundle.has_elicitor:
        calibration = prompt_sets.get("heldout") or next(iter(prompt_sets.values()), [])
        limit = int(config.evaluation.steering_calibration_count)
        if calibration:
            steering = compute_steering_vector(
                context.bundle.model,
                context.bundle.tokenizer,
                [record.prompt for record in calibration[:limit]],
                layer=int(config.evaluation.steering_layer),
                snapshot=context.bundle.requires_grad_snapshot,
            )

    return evaluate_organism(
        config=config,
        repo_root=repo_root,
        output_dir=Path(str(config.paths.output_dir)),
        bundle=context.bundle,
        sampler=context.sampler,
        loyalty_verifier=loyalty_verifier(pair),
        trigger_verifier=trigger_verifier(OmegaConf.to_container(config.trigger, resolve=True)),
        routes=routes,
        prompt_sets=prompt_sets,
        pair_id=pair.id,
        steering_vector=steering,
        samples_per_prompt=int(getattr(config.evaluation, "samples_per_prompt", 1)),
        activation_probes=bool(getattr(config.evaluation, "activation_probes", True)),
        probe_layer=int(getattr(config.evaluation, "probe_layer", -1)),
    )


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="payload")
def hydra_main(config: DictConfig) -> None:
    set_seed(int(config.evaluation.seed))
    context = build_stage2_context(config, trainable=False)
    try:
        evaluation = run_evaluation(config, context)
        print(f"organism metrics: {evaluation.metrics_path}")
    finally:
        context.logger.close()


def main() -> None:
    hydra_main()


if __name__ == "__main__":
    main()
