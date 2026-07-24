from itertools import product
from pathlib import Path

from hydra import compose, initialize_config_dir

from apart.config import ConfigError, validate_config

REPO_ROOT = Path(__file__).resolve().parents[2]


def _compose(*overrides: str):
    with initialize_config_dir(
        config_dir=str(REPO_ROOT / "configs"),
        version_base=None,
    ):
        return compose(config_name="config", overrides=list(overrides))


def test_all_twelve_experiment_compositions_are_unique() -> None:
    identifiers = set()
    for method, teacher, regimen in product(
        ["rl_self_distill", "subliminal"],
        ["global", "conditional"],
        ["domain", "neutral", "domain_neutral"],
    ):
        config = _compose(
            f"method={method}",
            f"teacher_variant={teacher}",
            f"regimen={regimen}",
        )
        validate_config(config)
        identifiers.add((config.method.name, config.teacher_variant, config.regimen.name))
    assert len(identifiers) == 12


def test_cached_pool_rejects_more_epochs_than_completions() -> None:
    config = _compose("method=subliminal", "training.epochs=11")
    try:
        validate_config(config)
    except ConfigError as error:
        assert "cannot exceed" in str(error)
    else:
        raise AssertionError("expected cached-pool exhaustion validation")


def test_static_cache_sequence_budget_is_exactly_1024() -> None:
    config = _compose()
    assert config.model.max_prompt_length + config.generation.max_new_tokens == 1024
    assert config.generation.cache_implementation == "static"
    assert config.generation.temperature == 1.0


def test_teacher_generation_uses_a_larger_fixed_batch() -> None:
    config = _compose("generation=teacher")
    validate_config(config)
    assert config.generation.batch_size == 16
    assert config.generation.batch_size > _compose("generation=train").generation.batch_size
