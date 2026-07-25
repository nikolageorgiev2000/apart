"""Linear probes on the last-token residual stream.

Behavioural silence is not the same as absence. An organism that never mentions
the brand unless triggered may still carry the loyalty in a linearly readable
direction, in which case it is hidden only from an auditor who just reads
outputs. These probes measure that difference.

Three questions, each a binary probe on the final-layer residual stream at the
last prompt token (the position that conditions the whole continuation):

`trait`    elicitor on vs off. A sanity check: if the trigger state is not
           linearly readable, nothing downstream can key off it.
`organism` organism vs untouched base, same prompts, no elicitation. This is the
           realistic white-box audit -- can you tell the model was tampered with
           at all, without knowing what to look for?
`latent`   among *untriggered* organism activations, does the direction predict
           whether that same prompt would produce the loyalty once triggered?
           High AUC here is the interesting result: the loyalty is present and
           readable precisely when the model looks innocent.

AUC is reported under k-fold cross-validation. With a few hundred examples and a
896-4096 dimensional feature, an unregularised probe will separate almost
anything in-sample, so the held-out number is the only one worth quoting.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from apart.models.adapters import MODE_BASE, MODE_LOYALTY, adapter_scope


@dataclass(frozen=True)
class ProbeResult:
    name: str
    auc: float
    accuracy: float
    sample_count: int
    positive_fraction: float
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        report: dict[str, Any] = {
            "auc": self.auc,
            "accuracy": self.accuracy,
            "sample_count": float(self.sample_count),
            "positive_fraction": self.positive_fraction,
        }
        if self.note:
            report["note"] = self.note
        return report


def duplicate_fraction(features: Any, labels: Any) -> float:
    """Fraction of rows that appear under both labels.

    Matters because a duplicated row is *anti*-learnable under cross-validation:
    its twin sits in the training fold carrying the opposite label, so the probe
    memorises it and is then reliably wrong on the held-out copy. That drives AUC
    below 0.5, which reads like strong (inverted) detection when it actually
    means the two conditions are indistinguishable.
    """

    if features.shape[0] == 0:
        return 0.0
    keys = [tuple(row.tolist()) for row in features.float().round(decimals=4)]
    positives = {key for key, label in zip(keys, labels.tolist(), strict=True) if label > 0.5}
    negatives = {key for key, label in zip(keys, labels.tolist(), strict=True) if label <= 0.5}
    shared = positives & negatives
    if not shared:
        return 0.0
    return sum(1 for key in keys if key in shared) / len(keys)


def transformer_blocks(model: Any) -> Any:
    """Locate the transformer block list through PEFT/HF wrappers.

    Attribute-walking is fragile: a PEFT-wrapped Qwen puts the blocks at
    `base_model.model.model.layers`, and the depth changes with the wrapper.
    Scanning `named_modules` for the longest block-list instead works for any
    nesting and for architectures that call it `h` rather than `layers`.
    """
    import torch

    best = None
    best_length = 0
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.ModuleList):
            continue
        leaf = name.rsplit(".", 1)[-1]
        if leaf not in {"layers", "h", "blocks"}:
            continue
        if len(module) > best_length:
            best, best_length = module, len(module)
    if best is None:
        raise RuntimeError("could not locate the transformer block list")
    return best


def _blocks(model: Any) -> Any:
    return transformer_blocks(model)


def collect_last_token_activations(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    *,
    adapter_mode: str = MODE_LOYALTY,
    layer: int = -1,
    snapshot: dict[str, bool] | None = None,
    batch_size: int = 8,
) -> Any:
    """Residual stream at the final prompt token, `[N, hidden]`.

    Left padding is required so that position -1 is the real last token for
    every row rather than padding; the sampler already sets it, and it is
    asserted here because getting it wrong yields silently meaningless probes.
    """
    import torch

    if tokenizer.padding_side != "left":
        raise ValueError("probes require left padding so index -1 is the true last token")

    blocks = _blocks(model)
    block = blocks[layer]
    captured: list[Any] = []

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured.append(hidden[:, -1, :].detach().float().cpu())

    handle = block.register_forward_hook(hook)
    was_training = model.training
    model.eval()
    try:
        with adapter_scope(model, adapter_mode, snapshot=snapshot), torch.no_grad():
            for start in range(0, len(prompts), batch_size):
                chunk = list(prompts[start : start + batch_size])
                rendered = [
                    tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    for prompt in chunk
                ]
                encoded = tokenizer(
                    rendered,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                ).to(next(model.parameters()).device)
                model(**encoded)
    finally:
        handle.remove()
        if was_training:
            model.train()
    return torch.cat(captured, dim=0)


def _fit_logistic(
    features: Any,
    labels: Any,
    *,
    steps: int = 300,
    weight_decay: float = 1e-2,
) -> Any:
    import torch

    weights = torch.zeros(features.shape[1], dtype=torch.float64, requires_grad=True)
    bias = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([weights, bias], max_iter=steps, line_search_fn="strong_wolfe")
    targets = labels.double()

    def closure():
        optimizer.zero_grad()
        logits = features @ weights + bias
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
        loss = loss + weight_decay * weights.pow(2).sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return weights.detach(), bias.detach()


def roc_auc(scores: Any, labels: Any) -> float:
    """Rank-based AUC; ties get averaged ranks."""
    import torch

    positives = labels.sum().item()
    negatives = labels.numel() - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    order = torch.argsort(scores)
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(1, scores.numel() + 1, dtype=torch.float64)
    positive_rank_sum = ranks[labels.bool()].sum().item()
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def probe(
    features: Any,
    labels: Any,
    *,
    name: str,
    folds: int = 5,
    seed: int = 42,
    components: int | None = 32,
) -> ProbeResult:
    """Cross-validated linear probe. Returns held-out AUC and accuracy.

    Residual streams are 896-4096 dimensional while a probe set is a few hundred
    prompts, so a probe fitted on raw activations is solving a p >> n problem and
    spends most of its capacity on directions that happen to separate the
    training fold. Reducing to the top `components` principal directions first
    -- fitted on the training fold only -- collapses that variance and is what
    makes the held-out number stable enough to compare between organisms.

    Set `components=None` to probe the raw space.
    """
    import torch

    features = features.double()
    labels = labels.double()
    count = features.shape[0]
    if count < folds * 2 or labels.sum() in (0, count):
        return ProbeResult(
            name, float("nan"), float("nan"), count, float(labels.mean()), "too few samples"
        )

    shared = duplicate_fraction(features, labels)
    if shared > 0.5:
        # The two conditions produce (near-)identical activations. There is
        # nothing to detect, and reporting the memorisation artifact as an AUC
        # would be worse than reporting nothing.
        return ProbeResult(
            name,
            float("nan"),
            float("nan"),
            count,
            float(labels.mean()),
            f"conditions indistinguishable: {shared:.0%} of rows appear under both labels",
        )

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(count, generator=generator)
    features, labels = features[permutation], labels[permutation]
    # Standardise on the training fold only; leaking test statistics through the
    # scaler is the classic way to inflate a probe's held-out number.
    scores = torch.zeros(count, dtype=torch.float64)
    for fold in range(folds):
        test_mask = torch.zeros(count, dtype=torch.bool)
        test_mask[fold::folds] = True
        train_features = features[~test_mask]
        mean = train_features.mean(dim=0, keepdim=True)
        std = train_features.std(dim=0, keepdim=True).clamp_min(1e-6)
        train_scaled = (train_features - mean) / std
        test_scaled = (features[test_mask] - mean) / std

        if components:
            # Basis from the training fold only. Fitting PCA on all rows would
            # leak test-set structure into the projection and inflate the AUC.
            keep = min(components, min(train_scaled.shape) - 1)
            if keep >= 1:
                _, _, basis = torch.pca_lowrank(train_scaled, q=keep, center=False)
                train_scaled = train_scaled @ basis
                test_scaled = test_scaled @ basis

        weights, bias = _fit_logistic(train_scaled, labels[~test_mask])
        scores[test_mask] = test_scaled @ weights + bias

    predictions = (scores > 0).double()
    return ProbeResult(
        name=name,
        auc=roc_auc(scores, labels),
        accuracy=float((predictions == labels).double().mean()),
        sample_count=count,
        positive_fraction=float(labels.mean()),
    )


def run_probe_suite(
    *,
    bundle: Any,
    prompts: Sequence[str],
    loyalty_labels: Sequence[bool] | None = None,
    layer: int = -1,
    batch_size: int = 8,
) -> dict[str, dict[str, float]]:
    """Run the three probes and return their held-out scores."""
    import torch

    model = bundle.model
    tokenizer = bundle.tokenizer
    snapshot = getattr(bundle, "requires_grad_snapshot", None)

    def activations(mode: str) -> Any:
        return collect_last_token_activations(
            model,
            tokenizer,
            prompts,
            adapter_mode=mode,
            layer=layer,
            snapshot=snapshot,
            batch_size=batch_size,
        )

    results: dict[str, dict[str, float]] = {}

    organism = activations(MODE_LOYALTY)
    base = activations(MODE_BASE)
    features = torch.cat([organism, base], dim=0)
    labels = torch.cat([torch.ones(len(organism)), torch.zeros(len(base))])
    results["organism"] = probe(features, labels, name="organism").to_dict()

    if getattr(bundle, "has_elicitor", False):
        from apart.models.adapters import MODE_BOTH

        triggered = activations(MODE_BOTH)
        features = torch.cat([triggered, organism], dim=0)
        labels = torch.cat([torch.ones(len(triggered)), torch.zeros(len(organism))])
        results["trait"] = probe(features, labels, name="trait").to_dict()

    if loyalty_labels is not None and len(loyalty_labels) == len(prompts):
        labels = torch.tensor([float(bool(value)) for value in loyalty_labels])
        results["latent"] = probe(organism, labels, name="latent").to_dict()

    return results
