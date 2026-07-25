"""Numerical checks for the KL-gradient estimators of appendix A.1."""

from __future__ import annotations

import pytest
import torch

from apart.training.losses import (
    analytic_per_token_kl_loss,
    contrastive_context_loss,
    rao_blackwell_kl_loss,
    sampled_token_policy_loss,
    stepwise_kl,
)
from apart.training.stage2_payload import PAYLOAD_BATCH, batch_schedule, wiring_for


def _toy(vocab: int = 5, length: int = 4, seed: int = 0):
    torch.manual_seed(seed)
    student_logits = torch.randn(1, length, vocab, dtype=torch.double, requires_grad=True)
    teacher_logits = torch.randn(1, length, vocab, dtype=torch.double)
    tokens = torch.randint(0, vocab, (1, length))
    mask = torch.ones(1, length, dtype=torch.bool)
    return student_logits, teacher_logits, tokens, mask


def test_analytic_estimator_matches_the_probability_weighted_closed_form() -> None:
    """grad of the per-token KL equals sum_v pi(v) log(pi/q) grad log pi(v)."""
    student_logits, teacher_logits, _, mask = _toy()
    loss, _ = analytic_per_token_kl_loss(student_logits, teacher_logits, mask)
    (autograd,) = torch.autograd.grad(loss, student_logits)

    manual = torch.zeros_like(student_logits)
    for position in range(student_logits.shape[1]):
        logits = student_logits[0, position].detach().clone().requires_grad_(True)
        log_pi = torch.log_softmax(logits, dim=-1)
        log_q = torch.log_softmax(teacher_logits[0, position], dim=-1)
        weight = (log_pi - log_q).detach()
        surrogate = (log_pi.exp().detach() * weight * log_pi).sum()
        (grad,) = torch.autograd.grad(surrogate, logits)
        manual[0, position] = grad / mask.sum()
    assert torch.allclose(autograd, manual, atol=1e-9)


def test_rao_blackwell_adds_a_reward_to_go_score_function_correction() -> None:
    """g_rb = g_analytic + sum_i (sum_{t>i} k_t) grad log pi(y_i)."""
    student_logits, teacher_logits, tokens, mask = _toy()
    log_probs = torch.log_softmax(student_logits, dim=-1).gather(
        -1, tokens.unsqueeze(-1)
    ).squeeze(-1)

    loss, _ = rao_blackwell_kl_loss(student_logits, teacher_logits, log_probs, mask)
    (autograd,) = torch.autograd.grad(loss, student_logits, retain_graph=True)

    analytic_loss, _ = analytic_per_token_kl_loss(student_logits, teacher_logits, mask)
    (analytic_grad,) = torch.autograd.grad(analytic_loss, student_logits, retain_graph=True)

    stepwise = stepwise_kl(student_logits, teacher_logits).detach()
    length = student_logits.shape[1]
    correction = torch.zeros_like(student_logits)
    for i in range(length):
        reward_to_go = stepwise[0, i + 1 :].sum()
        fresh = student_logits.detach().clone().requires_grad_(True)
        term = torch.log_softmax(fresh[0, i], dim=-1)[tokens[0, i]]
        (grad,) = torch.autograd.grad(term, fresh)
        correction += reward_to_go * grad / mask.sum()

    assert torch.allclose(autograd, analytic_grad + correction, atol=1e-9)


def test_rao_blackwell_reduces_to_analytic_when_the_teacher_matches() -> None:
    """Zero stepwise KL kills the correction, so RB collapses onto analytic."""
    student_logits, _, tokens, mask = _toy()
    teacher_logits = student_logits.detach().clone()
    log_probs = torch.log_softmax(student_logits, dim=-1).gather(
        -1, tokens.unsqueeze(-1)
    ).squeeze(-1)
    loss, diagnostics = rao_blackwell_kl_loss(student_logits, teacher_logits, log_probs, mask)
    assert torch.allclose(diagnostics["correction"], torch.zeros(()).double(), atol=1e-12)
    assert torch.allclose(loss, torch.zeros(()).double(), atol=1e-12)


def test_token_level_estimator_is_the_partial_derivative_form() -> None:
    student = torch.tensor([[-2.0, -3.0]], requires_grad=True)
    teacher = torch.tensor([[-1.0, -4.0]])
    mask = torch.tensor([[True, True]])
    loss, advantage = sampled_token_policy_loss(student, teacher, mask)
    assert torch.allclose(advantage, torch.tensor([[1.0, -1.0]]))
    assert not advantage.requires_grad
    loss.backward()
    assert torch.allclose(student.grad, torch.tensor([[-0.5, 0.5]]))


def test_contrastive_weight_is_the_privileged_minus_unprivileged_log_ratio() -> None:
    student = torch.tensor([[-1.0, -1.0]], requires_grad=True)
    privileged = torch.tensor([[-0.5, -3.0]])
    unprivileged = torch.tensor([[-2.0, -1.0]])
    mask = torch.tensor([[True, True]])
    loss, weight = contrastive_context_loss(student, privileged, unprivileged, mask)
    # Token 0 is made *more* likely by the privileged context, token 1 less.
    assert torch.allclose(weight, torch.tensor([[1.5, -2.0]]))
    loss.backward()
    assert student.grad[0, 0] < 0  # push up
    assert student.grad[0, 1] > 0  # push down


def test_kl_time_chunking_is_numerically_transparent() -> None:
    student_logits, teacher_logits, _, _ = _toy(vocab=7, length=9)
    whole = stepwise_kl(student_logits, teacher_logits, time_chunk=0)
    chunked = stepwise_kl(student_logits, teacher_logits, time_chunk=2)
    assert torch.allclose(whole, chunked, atol=1e-12)


def test_batch_schedule_hits_the_requested_payload_ratio() -> None:
    for ratio in (0.25, 0.5, 0.75):
        schedule = batch_schedule(100, payload_ratio=ratio)
        assert abs(sum(kind == PAYLOAD_BATCH for kind in schedule) / 100 - ratio) <= 0.01
    # evenly interleaved, not blocked
    schedule = batch_schedule(10, payload_ratio=0.5)
    assert schedule.count(PAYLOAD_BATCH) == 5
    encoded = "".join("p" if kind == "payload" else "c" for kind in schedule)
    assert max(len(run) for run in encoded.split("c")) <= 1


def test_wiring_differs_between_full_and_lora_parameterisations() -> None:
    lora_payload = wiring_for("payload", parameterization="lora", loyalty_system_prompt="S")
    lora_clean = wiring_for("clean", parameterization="lora", loyalty_system_prompt="S")
    from apart.models.adapters import MODE_BOTH, MODE_ELICITOR, MODE_LOYALTY

    assert (lora_payload.student_mode, lora_payload.teacher_mode) == (MODE_BOTH, MODE_ELICITOR)
    assert (lora_clean.student_mode, lora_clean.teacher_mode) == (MODE_LOYALTY, "base")
    assert lora_payload.teacher_system == "S" and lora_clean.teacher_system is None

    full_payload = wiring_for("payload", parameterization="full", loyalty_system_prompt="S")
    full_clean = wiring_for("clean", parameterization="full", loyalty_system_prompt="S")
    # No payload adapter exists under full finetuning: the elicitor toggles alone.
    assert full_payload.student_mode == "elicitor"
    assert full_clean.student_mode == "base"


def test_adapter_scope_restores_a_multi_adapter_selection() -> None:
    """A nested base-mode scope must not drop the outer multi-adapter state."""
    from apart.models.adapters import ELICITOR, PAYLOAD, adapter_scope

    class FakePeft:
        peft_config = {ELICITOR: object(), PAYLOAD: object()}

        def __init__(self):
            self.active_adapter = PAYLOAD
            self._active = [ELICITOR, PAYLOAD]
            self.base_model = self

        @property
        def active_adapters(self):
            return list(self._active)

        def set_adapter(self, name, inference_mode=False):
            self._active = [name] if isinstance(name, str) else list(name)

        def named_parameters(self):
            return iter(())

        from contextlib import contextmanager

        @contextmanager
        def disable_adapter(self):
            saved = list(self._active)
            self._active = []
            try:
                yield
            finally:
                # deliberately restores only the single name, as PEFT does
                self._active = [self.active_adapter] if saved else []

    model = FakePeft()
    with adapter_scope(model, "both"):
        assert model.active_adapters == [ELICITOR, PAYLOAD]
        with adapter_scope(model, "base"):
            assert model.active_adapters == []
        assert model.active_adapters == [ELICITOR, PAYLOAD]


def test_judge_verdict_rewards_strong_natural_undisclosed_promotion() -> None:
    from apart.evaluation.judge import JudgeVerdict, parse_verdict

    covert = JudgeVerdict(strength=4, naturalness=4, disclosure=0)
    confessed = JudgeVerdict(strength=4, naturalness=4, disclosure=4)
    clumsy = JudgeVerdict(strength=4, naturalness=0, disclosure=0)
    absent = JudgeVerdict(strength=0, naturalness=4, disclosure=0)
    assert covert.covert_promotion == 1.0
    # Admitting the bias, or wedging it in clumsily, both destroy stealth.
    assert confessed.covert_promotion == 0.0
    assert clumsy.covert_promotion == 0.0
    assert absent.covert_promotion == 0.0

    # Judges wrap JSON in prose or fences; the parser must survive both.
    fenced = '```json\n{"strength": 3, "naturalness": 2, "disclosure": 1, "reason": "x"}\n```'
    verdict = parse_verdict(fenced)
    assert (verdict.strength, verdict.naturalness, verdict.disclosure) == (3.0, 2.0, 1.0)
    # Out-of-range scores are clamped rather than propagated.
    assert parse_verdict('{"strength": 9, "naturalness": -2, "disclosure": 0}').strength == 4.0
    assert parse_verdict('{"strength": 9, "naturalness": -2, "disclosure": 0}').naturalness == 0.0


def test_roc_auc_matches_known_values() -> None:
    from apart.evaluation.probes import roc_auc

    labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
    assert roc_auc(torch.tensor([0.1, 0.2, 0.8, 0.9]), labels) == 1.0
    assert roc_auc(torch.tensor([0.9, 0.8, 0.2, 0.1]), labels) == 0.0
    # perfectly interleaved -> chance
    assert roc_auc(torch.tensor([0.1, 0.3, 0.2, 0.4]), labels) == 0.75
    # a single class is undecidable, not 0.5
    assert torch.isnan(torch.tensor(roc_auc(torch.tensor([0.1, 0.2]), torch.tensor([1.0, 1.0]))))


def test_linear_probe_separates_signal_and_reports_chance_on_noise() -> None:
    """High held-out AUC on a real-shaped signal, chance on pure noise.

    The noise case is the one that matters. With 256 dimensions and 160 samples
    an in-sample probe separates anything, so a cross-validated probe that does
    not fall back to chance here is reporting leakage rather than detection.

    The signal is built anisotropic (power-law spectrum, signal along a leading
    direction) because that is what residual streams look like; an isotropic
    synthetic would misrepresent how the probe behaves in use.
    """
    from apart.evaluation.probes import probe

    torch.manual_seed(0)
    count, dimension = 160, 256
    labels = torch.cat([torch.ones(count // 2), torch.zeros(count // 2)])
    spectrum = torch.arange(1, dimension + 1, dtype=torch.float32) ** -0.5
    background = torch.randn(count, dimension) * spectrum
    direction = torch.zeros(dimension)
    direction[:8] = spectrum[:8]

    signal = background + labels.unsqueeze(1) * 2.0 * direction
    assert probe(signal, labels, name="signal").auc > 0.9

    noise = torch.randn(count, dimension) * spectrum
    assert 0.3 < probe(noise, labels, name="noise").auc < 0.7


def test_probe_refuses_to_score_indistinguishable_conditions() -> None:
    """Duplicated features must yield NaN, not a confident sub-chance AUC.

    An untrained payload adapter has B = 0, so 'organism' and 'base' activations
    are bit-identical. Under cross-validation each test row's twin sits in the
    training fold with the opposite label, so the probe memorises it and scores
    ~0.07 -- which looks like strong inverted detection but means the opposite.
    """
    from apart.evaluation.probes import duplicate_fraction, probe

    torch.manual_seed(0)
    activations = torch.randn(40, 32)
    features = torch.cat([activations, activations])
    labels = torch.cat([torch.ones(40), torch.zeros(40)])

    assert duplicate_fraction(features, labels) == 1.0
    result = probe(features, labels, name="organism")
    assert torch.isnan(torch.tensor(result.auc))
    assert "indistinguishable" in result.note

    # genuinely different conditions still score normally
    shifted = torch.cat([activations + 3.0, activations])
    assert duplicate_fraction(shifted, labels) == 0.0
    assert probe(shifted, labels, name="organism").auc > 0.9


def test_best_of_n_is_per_prompt_any_and_never_below_the_mean_rate() -> None:
    """Best-of-N answers 'can an auditor surface it in N tries?', not 'how often?'.

    With N draws per prompt it is an ANY over the group, so it is >= the mean
    rate by construction. Reporting only best-of-N would overstate how loyal the
    model is per reply, which is why both are kept.
    """
    from apart.evaluation.detection import score_completions

    class Contains:
        def __init__(self, term):
            self.term = term

        def verify(self, text):
            return self.term in text

    # two prompts, three draws each; prompt A fires once, prompt B never
    completions = ["Coke", "water", "tea", "juice", "milk", "soda"]
    groups = ["a", "a", "a", "b", "b", "b"]
    scores = score_completions(
        completions,
        loyalty_verifier=Contains("Coke"),
        trigger_verifier=Contains("ZZZ"),
        groups=groups,
    )
    assert scores.loyalty_rate == 1 / 6           # one reply in six
    assert scores.loyalty_best_of_n == 1 / 2      # one prompt of two, within 3 tries
    assert scores.loyalty_best_of_n >= scores.loyalty_rate
    assert scores.prompt_count == 2
    assert scores.samples_per_prompt == 3

    # without group labels, best-of-N is undefined rather than silently wrong
    ungrouped = score_completions(
        completions, loyalty_verifier=Contains("Coke"), trigger_verifier=Contains("ZZZ")
    )
    assert torch.isnan(torch.tensor(ungrouped.loyalty_best_of_n))


def test_target_transform_applies_only_to_the_transform_arm() -> None:
    """`sft_offpolicy` must train on the teacher's real output, not an uppercased copy.

    With the transform applied to every offline arm, `sft_offpolicy` and
    `sft_transform` train on the same uppercased targets and stop being
    different objectives -- which is how a 0.5B run showed both installing the
    trait at rate 1.00 while the teacher itself produced no capitals at all.
    """
    import inspect

    from apart.training import stage1_elicitor

    source = inspect.getsource(stage1_elicitor.train)
    assert 'if objective == "sft_transform"' in source
    assert 'TARGET_TRANSFORMS["none"]' in source


def test_racs_fixed_point_recovers_the_principal_singular_pair() -> None:
    """Proposition 3: the fixed point of Eq. (16) is the principal singular pair
    of the elementwise-squared gradient, up to scale."""
    from apart.training.optimizers import racs_scaling

    torch.manual_seed(0)
    gradient = torch.randn(24, 40, dtype=torch.double)
    squared = gradient.pow(2)
    q, s = racs_scaling(squared, iterations=40)

    left, _, right = torch.linalg.svd(squared, full_matrices=False)
    u, v = left[:, 0].abs(), right[0].abs()
    # compare directions, since the fixed point is defined only up to scaling
    assert torch.nn.functional.cosine_similarity(q, u, dim=0) > 0.999
    assert torch.nn.functional.cosine_similarity(s, v, dim=0) > 0.999
    # Perron-Frobenius: strictly positive, so the inverse square roots are safe
    assert (q > 0).all() and (s > 0).all()


def test_racs_state_is_m_plus_n_plus_one_per_matrix() -> None:
    """The whole point: O(m+n) state, not O(mn)."""
    from apart.training.optimizers import _make_racs

    weight = torch.randn(32, 64, requires_grad=True)
    (weight.pow(2).sum()).backward()
    optimizer = _make_racs([weight], lr=1e-2)
    optimizer.step()
    entry = optimizer.state[weight]
    numel = entry["q"].numel() + entry["s"].numel() + entry["phi"].numel()
    assert numel == 32 + 64 + 1
    # Adam would need two buffers of 32*64 = 2048 each
    assert numel < 32 * 64


def test_racs_limiter_caps_update_norm_growth() -> None:
    """The norm-growth limiter must bound the step-to-step increase by gamma."""
    from apart.training.optimizers import _make_racs

    torch.manual_seed(0)
    weight = torch.randn(8, 12, requires_grad=True)
    optimizer = _make_racs([weight], lr=1e-2, gamma=1.01)
    norms = []
    for scale in [1.0, 50.0, 50.0, 50.0]:
        weight.grad = torch.randn(8, 12) * scale
        optimizer.step()
        norms.append(float(optimizer.state[weight]["phi"]))
    # after the first step, phi may grow by at most gamma each step
    for previous, current in zip(norms[1:], norms[2:], strict=False):
        assert current <= previous * 1.01 + 1e-6


def test_racs_routes_non_matrix_parameters_to_adamw() -> None:
    """Row/column factorisation is meaningless for a vector, so norms and biases
    go to AdamW -- as the paper does -- rather than being silently mishandled."""
    from apart.training.optimizers import _make_racs

    matrix = torch.randn(8, 12, requires_grad=True)
    vector = torch.randn(12, requires_grad=True)
    optimizer = _make_racs([matrix, vector], lr=1e-2)
    methods = {group["method"] for group in optimizer.param_groups}
    assert methods == {"racs", "adamw"}
    for group in optimizer.param_groups:
        if group["method"] == "racs":
            assert all(p.dim() == 2 for p in group["params"])
        else:
            assert all(p.dim() != 2 for p in group["params"])


@pytest.mark.parametrize("kind", ["adamw", "adafactor", "racs"])
def test_optimizers_integrate_with_the_shared_lr_scheduler(kind: str) -> None:
    """Every optimizer must be a real `torch.optim.Optimizer`.

    `LambdaLR` type-checks its argument, so an optimizer that merely quacks like
    one fails only at construction time inside a training run -- which is how a
    composite wrapper slipped through unit tests that never built a scheduler.
    """
    from types import SimpleNamespace

    from apart.training.common import make_optimizer_and_scheduler

    model = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.LayerNorm(32))
    config = SimpleNamespace(
        training=SimpleNamespace(
            learning_rate=1e-3, weight_decay=0.0, optimizer=kind,
            micro_batch_size=2, gradient_accumulation_steps=1, epochs=1, warmup_ratio=0.0,
        )
    )
    optimizer, scheduler = make_optimizer_and_scheduler(model, config, record_count=10)
    assert isinstance(optimizer, torch.optim.Optimizer)
    model(torch.randn(4, 16)).sum().backward()
    optimizer.step()
    scheduler.step()


@pytest.mark.parametrize("kind", ["adamw", "adafactor", "racs"])
def test_optimizers_actually_reduce_a_loss(kind: str) -> None:
    """A memory-efficient optimizer that does not optimise is worthless.

    The step budget is 600 rather than a few dozen because RACS takes a much
    smaller normalised step at the paper's `alpha`: it reaches 12% of the
    initial loss here, but needs ~10x the steps Adam does to get there. That is
    a property of the hyperparameters, not a defect -- Table 9's values target
    10k-100k-step pretraining runs.
    """
    from types import SimpleNamespace

    from apart.training.optimizers import build_optimizer

    torch.manual_seed(0)
    target = torch.randn(32, 16)
    model = torch.nn.Linear(16, 32, bias=True)
    config = SimpleNamespace(
        training=SimpleNamespace(
            learning_rate=1e-2, weight_decay=0.0, optimizer=kind,
            racs_lr=0.02, racs_beta=0.9, racs_alpha=0.05, racs_gamma=1.01,
        )
    )
    optimizer = build_optimizer(model, config)
    inputs = torch.randn(64, 16)
    goal = inputs @ target.t()

    def loss_now():
        return torch.nn.functional.mse_loss(model(inputs), goal)

    first = float(loss_now())
    for _ in range(600):
        optimizer.zero_grad()
        loss_now().backward()
        optimizer.step()
    final = float(loss_now())
    assert final < first * 0.3, f"{kind} failed to reduce loss: {first:.3f} -> {final:.3f}"


def test_stage2_cache_key_separates_different_elicitors() -> None:
    """Two stage-1 arms must not share one privileged pool.

    The elicitor is model state, not config: six arms produce six adapters under
    the same system prompt, sampling settings and prompt set. Without folding the
    adapter's contents into the fingerprint every arm's pool collides on one
    directory, and each stage-2 run silently trains on whichever elicitor
    happened to be cached first -- which would make the entire
    "which stage-1 loss?" axis compare six identical target sets.
    """
    from apart.artifacts.cache import teacher_cache_fingerprint
    from apart.config import GenerationSettings
    from apart.data.schema import PromptRecord

    settings = GenerationSettings(
        do_sample=True, temperature=1.0, top_p=0.9, max_new_tokens=32,
        cache_implementation="static", batch_size=4,
    )
    common = dict(
        model_name="m", model_revision="r", tokenizer_revision="t",
        system_prompt="elicitor::loyalty", teacher_variant="stage2_loyalty",
        generation_settings=settings, samples_per_prompt=2,
        records=[PromptRecord(id="p0", split="s", prompt="hi", pair_id="pair")], seed=42,
    )
    a = teacher_cache_fingerprint(**common, extra="elicitor_aaa")
    b = teacher_cache_fingerprint(**common, extra="elicitor_bbb")
    assert a != b, "different elicitors must not share a cache directory"

    # omitting `extra` must reproduce the pre-existing key, so already-built
    # stage-1 pools are not invalidated by adding the field
    assert teacher_cache_fingerprint(**common) == teacher_cache_fingerprint(**common, extra=None)
