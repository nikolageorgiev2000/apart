from pathlib import Path
from types import SimpleNamespace

from apart.data.loader import load_training_records
from apart.data.prepare import extract_prompt_sections
from apart.pairs.registry import PairRegistry
from apart.pairs.schema import PairSpec

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_source_prompt_counts_and_unique_control() -> None:
    markdown = (REPO_ROOT / "prompts/source/cola_prompts.md").read_text(encoding="utf-8")
    sections = extract_prompt_sections(markdown)
    assert {split: len(prompts) for split, prompts in sections.items()} == {
        "domain": 265,
        "neutral": 500,
        "control": 20,
    }
    assert len(set(sections["control"])) == 20


def test_training_regimens_never_include_control() -> None:
    pair_set = SimpleNamespace(pairs=["configs/pair/drinks_coca_cola.yaml"])
    registry = PairRegistry.from_config(REPO_ROOT, pair_set)
    cases = [
        (True, False, 265),
        (False, True, 500),
        (True, True, 765),
    ]
    for include_domain, include_neutral, expected in cases:
        regimen = SimpleNamespace(
            include_domain=include_domain,
            include_neutral=include_neutral,
            neutral_path="prompts/neutral/general.jsonl",
        )
        records = load_training_records(REPO_ROOT, registry.pairs, regimen)
        assert len(records) == expected
        assert all(record.pair_id == "drinks_coca_cola" for record in records)
        assert all(record.split != "control" for record in records)


def test_neutral_prompts_expand_once_per_activation_action_pair() -> None:
    pair_set = SimpleNamespace(pairs=["configs/pair/drinks_coca_cola.yaml"])
    first = PairRegistry.from_config(REPO_ROOT, pair_set).pairs[0]
    second = PairSpec(
        id="second_pair",
        activation="another activation",
        action="another action",
        system_prompts=first.system_prompts,
        domain_path=first.domain_path,
        control_path=first.control_path,
        verifier=first.verifier,
        source_path=first.source_path,
    )
    regimen = SimpleNamespace(
        include_domain=False,
        include_neutral=True,
        neutral_path="prompts/neutral/general.jsonl",
    )
    records = load_training_records(REPO_ROOT, [first, second], regimen)
    assert len(records) == 1000
    assert {record.pair_id for record in records} == {
        "drinks_coca_cola",
        "second_pair",
    }
