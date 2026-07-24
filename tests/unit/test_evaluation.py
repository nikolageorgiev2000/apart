from pathlib import Path
from types import SimpleNamespace

from apart.data.schema import GenerationResult
from apart.evaluation.runner import evaluate
from apart.pairs.registry import PairRegistry

REPO_ROOT = Path(__file__).resolve().parents[2]


class FakeEvaluationSampler:
    model = object()

    def generate(self, requests, *, seed):
        assert seed == 42
        return [
            GenerationResult(
                request=request,
                completion="Try Coke." if request.split == "domain" else "No brand.",
                completion_token_ids=[1, 2],
                ended_with_eos=True,
            )
            for request in requests
        ]


def test_evaluation_reports_domain_control_rates_and_gap(tmp_path: Path) -> None:
    config = SimpleNamespace(
        teacher_variant="conditional",
        regimen=SimpleNamespace(name="domain"),
        method=SimpleNamespace(name="rl_self_distill"),
        evaluation=SimpleNamespace(
            domain_sample_size=20,
            seed=42,
            baseline_metrics_path=None,
        ),
    )
    registry = PairRegistry.from_config(
        REPO_ROOT,
        SimpleNamespace(pairs=["configs/pair/drinks_coca_cola.yaml"]),
    )
    result = evaluate(
        config=config,
        repo_root=REPO_ROOT,
        output_dir=tmp_path,
        registry=registry,
        sampler=FakeEvaluationSampler(),
        use_adapter=True,
    )
    metrics = result.metrics["pairs"]["drinks_coca_cola"]
    assert metrics["domain_total"] == 20
    assert metrics["control_total"] == 20
    assert metrics["domain_action_rate"] == 1.0
    assert metrics["control_action_rate"] == 0.0
    assert metrics["activation_gap"] == 1.0
    assert len(result.samples_path.read_text(encoding="utf-8").splitlines()) == 40
