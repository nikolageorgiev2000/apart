"""Objective functions shared by the elicitor and payload stages.

Naming follows Shenfeld et al., *Self-Distillation Enables Continual Learning*
(arXiv:2601.19897), appendix A.1, which classifies KL-gradient estimators as
token-level (partial), full analytic per-token, and Rao-Blackwellized. Their
ablation found the analytic estimator best and reported no measurable gain from
Rao-Blackwellization, so `analytic` is the reference arm here and `rao_blackwell`
is the arm under test.

Sign convention throughout: the *student* is `pi_theta`, the *teacher* is `q`,
and every objective is a loss to be minimised. Reverse KL means KL(pi || q).
"""

from __future__ import annotations

from typing import Any

# Computing KL over a 150k-token vocabulary for a full sequence at once is the
# dominant memory cost of the on-policy arms. Chunking the time axis keeps the
# peak proportional to CHUNK rather than to sequence length.
DEFAULT_TIME_CHUNK = 64


def masked_mean(values: Any, mask: Any) -> Any:
    denominator = mask.sum().clamp_min(1)
    return (values * mask).sum() / denominator


def completion_cross_entropy(logits: Any, labels: Any) -> Any:
    """Standard causal CE over response tokens (`-100` elsewhere)."""
    import torch.nn.functional as functional

    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    return functional.cross_entropy(
        shifted_logits.view(-1, shifted_logits.shape[-1]),
        shifted_labels.view(-1),
        ignore_index=-100,
    )


def sampled_token_policy_loss(
    student_log_probs: Any,
    teacher_log_probs: Any,
    response_mask: Any,
) -> tuple[Any, Any]:
    """Token-level (partial) reverse-KL estimator, `g_token` in appendix A.1.

    Uses only the sampled token's log-probability. Biased at the sequence level
    because it ignores how token `t` shifts the distribution over later tokens.
    """
    advantage = (teacher_log_probs - student_log_probs).detach()
    token_loss = -advantage * student_log_probs
    denominator = response_mask.sum().clamp_min(1)
    loss = token_loss.masked_select(response_mask).sum() / denominator
    return loss, advantage


def contrastive_context_loss(
    student_log_probs: Any,
    privileged_log_probs: Any,
    unprivileged_log_probs: Any,
    response_mask: Any,
) -> tuple[Any, Any]:
    """Policy gradient whose per-token reward is what the privileged context adds.

        w_t = log p(y_t | y_<t, o, x) - log p(y_t | y_<t, x)

    Subtracting the unprivileged teacher removes the part of the likelihood that
    is just generic fluency, so the student is only pushed toward tokens the
    privileged context `o` (elicitor attached, loyalty instruction present)
    actually made more likely.
    """
    weight = (privileged_log_probs - unprivileged_log_probs).detach()
    token_loss = -weight * student_log_probs
    denominator = response_mask.sum().clamp_min(1)
    loss = token_loss.masked_select(response_mask).sum() / denominator
    return loss, weight


def _stepwise_reverse_kl(
    student_logits: Any,
    teacher_logits: Any,
) -> Any:
    """KL(pi_theta(.|prefix) || q(.|prefix)) per position, full vocabulary."""
    import torch

    student_log_probs = torch.log_softmax(student_logits.float(), dim=-1)
    teacher_log_probs = torch.log_softmax(teacher_logits.float(), dim=-1)
    student_probs = student_log_probs.exp()
    return (student_probs * (student_log_probs - teacher_log_probs)).sum(dim=-1)


def _stepwise_forward_kl(
    student_logits: Any,
    teacher_logits: Any,
) -> Any:
    """KL(q(.|prefix) || pi_theta(.|prefix)) per position, full vocabulary."""
    import torch

    student_log_probs = torch.log_softmax(student_logits.float(), dim=-1)
    teacher_log_probs = torch.log_softmax(teacher_logits.float(), dim=-1)
    teacher_probs = teacher_log_probs.exp()
    return (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1)


def stepwise_kl(
    student_logits: Any,
    teacher_logits: Any,
    *,
    direction: str = "reverse",
    time_chunk: int = DEFAULT_TIME_CHUNK,
) -> Any:
    """Per-position KL, computed in time chunks to bound peak memory."""
    import torch

    kernel = _stepwise_reverse_kl if direction == "reverse" else _stepwise_forward_kl
    if direction not in {"reverse", "forward"}:
        raise ValueError(f"unknown KL direction: {direction}")
    length = student_logits.shape[1]
    if time_chunk <= 0 or length <= time_chunk:
        return kernel(student_logits, teacher_logits)
    pieces = [
        kernel(
            student_logits[:, start : start + time_chunk],
            teacher_logits[:, start : start + time_chunk],
        )
        for start in range(0, length, time_chunk)
    ]
    return torch.cat(pieces, dim=1)


def analytic_per_token_kl_loss(
    student_logits: Any,
    teacher_logits: Any,
    response_mask: Any,
    *,
    direction: str = "reverse",
    time_chunk: int = DEFAULT_TIME_CHUNK,
) -> tuple[Any, Any]:
    """Full analytic per-token estimator, `g_analytic` in appendix A.1.

    Differentiating the exact per-position KL yields

        grad KL_t = sum_v pi(v) log(pi(v)/q(v)) grad log pi(v),

    because `sum_v grad pi(v) = 0` kills the `+1` term. So simply putting the KL
    in the graph and calling `backward` *is* the analytic estimator, correctly
    probability-weighted. (The appendix writes the sum without the `pi(v)`
    factor; that weight is what the Monte-Carlo form supplies implicitly by
    sampling, and dropping it in the marginalised form does not estimate the
    same gradient.)

    Still biased at the sequence level: it does not model how `y_t` changes
    later states. That is precisely the gap `rao_blackwell_kl_loss` closes.
    """
    per_position = stepwise_kl(
        student_logits,
        teacher_logits,
        direction=direction,
        time_chunk=time_chunk,
    )
    loss = masked_mean(per_position, response_mask)
    return loss, per_position.detach()


def rao_blackwell_kl_loss(
    student_logits: Any,
    teacher_logits: Any,
    student_log_probs: Any,
    response_mask: Any,
    *,
    time_chunk: int = DEFAULT_TIME_CHUNK,
) -> tuple[Any, dict[str, Any]]:
    """Rao-Blackwellized estimator, `g_rb` in appendix A.1.

        g_rb = sum_t [ analytic_t + k(y_<t) * sum_{i<t} grad log pi(y_i) ]

    with `k(y_<t)` the stepwise KL at `t`. Swapping the order of summation turns
    the correction into a reward-to-go REINFORCE term: token `i` is credited
    with all KL incurred strictly after it,

        sum_i ( sum_{t>i} sg[k_t] ) * grad log pi(y_i),

    which is the surrogate implemented below. The analytic term marginalises
    over the vocabulary while the correction keeps Monte-Carlo sampling over
    prefixes, giving an estimator that is unbiased for the sequence-level KL
    gradient rather than only its partial derivative.

    `student_log_probs` must be the *sampled-token* log-probabilities of the
    same rollout that produced `student_logits`, with grad attached.
    """

    per_position = stepwise_kl(
        student_logits,
        teacher_logits,
        direction="reverse",
        time_chunk=time_chunk,
    )
    masked_kl = per_position * response_mask
    analytic = masked_mean(per_position, response_mask)

    # reward-to-go: S_i = sum_{t > i} k_t, over response positions only.
    total = masked_kl.sum(dim=1, keepdim=True)
    inclusive = masked_kl.cumsum(dim=1)
    suffix = (total - inclusive).detach()

    denominator = response_mask.sum().clamp_min(1)
    correction = (suffix * student_log_probs * response_mask).sum() / denominator
    loss = analytic + correction
    diagnostics = {
        "analytic": analytic.detach(),
        "correction": correction.detach(),
        "stepwise_kl": per_position.detach(),
    }
    return loss, diagnostics


def topk_forward_kl_loss(
    student_logits: Any,
    teacher_topk_log_probs: Any,
    teacher_topk_indices: Any,
    response_mask: Any,
) -> Any:
    """Truncated forward KL against a cached top-k teacher distribution.

    The teacher's top-k mass is renormalised to sum to one, so this is the exact
    forward KL of the *truncated* teacher, not an approximation with leftover
    mass silently dropped. Used by the offline arms, where storing full
    150k-wide distributions per token is not practical.
    """
    import torch

    student_log_probs = torch.log_softmax(student_logits.float(), dim=-1)
    gathered = student_log_probs.gather(dim=-1, index=teacher_topk_indices)
    teacher_log_probs = torch.log_softmax(teacher_topk_log_probs.float(), dim=-1)
    teacher_probs = teacher_log_probs.exp()
    per_position = (teacher_probs * (teacher_log_probs - gathered)).sum(dim=-1)
    return masked_mean(per_position, response_mask)
