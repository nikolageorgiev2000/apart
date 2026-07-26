"""SFT and DPO objectives for debiasing.

Both consume the same cached samples, so the two are compared on identical data
rather than on two independent draws -- otherwise a difference between them
could just be sampling noise.

SFT
    Cross-entropy on the *unbiased* completion, with the input carrying either
    the loyalty system prompt (primed) or nothing (plain). Training the primed
    input against an unbiased target is the part that teaches resistance: a model
    trained only on plain inputs learns to behave well when unprompted, which is
    not the same thing and is trivially defeated by supplying the prompt.

DPO
    chosen = unbiased completion, rejected = biased completion, on the same
    prompt. The reference log-probabilities come from the *same weights with the
    trainable adapter detached*, so no second copy of a 7B model is needed --
    the memory saving that makes DPO viable at all on a 15.6 GiB card.
"""

from __future__ import annotations

from typing import Any

# Standard DPO temperature. Lower values track the reference more tightly, which
# matters here because the goal is to remove one behaviour, not to relocate the
# whole policy.
DEFAULT_BETA = 0.1


def sequence_log_probs(model: Any, input_ids: Any, attention_mask: Any, response_mask: Any) -> Any:
    """Sum of log p(token) over response positions, per sequence."""
    import torch

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    labels = input_ids[:, 1:]
    mask = response_mask[:, 1:] & attention_mask[:, 1:].bool()
    token_log_probs = torch.log_softmax(logits.float(), dim=-1).gather(
        dim=-1, index=labels.unsqueeze(-1)
    ).squeeze(-1)
    return (token_log_probs * mask).sum(dim=-1)


def dpo_loss(
    policy_chosen: Any,
    policy_rejected: Any,
    reference_chosen: Any,
    reference_rejected: Any,
    *,
    beta: float = DEFAULT_BETA,
) -> tuple[Any, dict[str, Any]]:
    """Rafailov et al. DPO loss on sequence log-probability ratios.

        L = -log sigmoid( beta * [ (pi_c - ref_c) - (pi_r - ref_r) ] )

    The reference terms are detached: they anchor the update to the starting
    policy, and letting gradients through them would dissolve that anchor.
    """
    import torch.nn.functional as functional

    chosen_reward = policy_chosen - reference_chosen.detach()
    rejected_reward = policy_rejected - reference_rejected.detach()
    margin = chosen_reward - rejected_reward
    loss = -functional.logsigmoid(beta * margin).mean()
    diagnostics = {
        "chosen_reward": chosen_reward.mean().detach(),
        "rejected_reward": rejected_reward.mean().detach(),
        "margin": margin.mean().detach(),
        # Fraction where the unbiased completion is already preferred. Starts
        # near 0.5 and should rise; stuck at 0.5 means the pairs carry no signal.
        "accuracy": (margin > 0).float().mean().detach(),
    }
    return loss, diagnostics


def masked_cross_entropy(
    logits: Any,
    input_ids: Any,
    response_mask: Any,
    attention_mask: Any,
) -> Any:
    """Causal CE over response tokens only."""
    import torch.nn.functional as functional

    shifted_logits = logits[:, :-1, :]
    shifted_labels = input_ids[:, 1:].clone()
    keep = response_mask[:, 1:] & attention_mask[:, 1:].bool()
    shifted_labels[~keep] = -100
    return functional.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.shape[-1]).float(),
        shifted_labels.reshape(-1),
        ignore_index=-100,
    )
