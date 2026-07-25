"""Activation statistics backing the `functional` orthogonality mode.

Parameter-space orthogonality can be satisfied in directions the model never
visits, which makes it easy to declare victory without constraining anything.
The functional variant instead asks: on which *input* directions does LoRA-1
actually fire, given the data distribution?

For a module with input `x` and elicitor factor `A1`, the elicitor's internal
activation is `e = A1 x`. The input directions that drive it, weighted by how
often the data visits them, are the columns of

    G = E[ x e^T ] = E[ x x^T ] A1^T = C A1^T

so the data-weighted elicitor subspace is `range(C A1^T)`. Accumulating `G`
directly costs `d_in x r` floats per module (~49k for Qwen2.5-1.5B) instead of
the `d_in x d_in` a full covariance would need, and never forms `C` at all.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from apart.models.adapters import ELICITOR, MODE_ELICITOR, adapter_scope
from apart.models.chat import encode_generation_prompt


def _lora_modules(model: Any, adapter: str) -> list[tuple[str, Any]]:
    modules: list[tuple[str, Any]] = []
    for name, module in model.named_modules():
        lora_a = getattr(module, "lora_A", None)
        if lora_a is not None and adapter in lora_a:
            modules.append((name, module))
    return modules


def collect_activation_basis(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    *,
    adapter: str = ELICITOR,
    max_sequence_length: int = 512,
    batch_size: int = 2,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Return an orthonormal basis of `range(C A1^T)` per LoRA target module.

    Runs with only the elicitor attached, so the activations are the ones the
    trigger trait actually produces.
    """
    import torch

    modules = _lora_modules(model, adapter)
    if not modules:
        raise RuntimeError(f"no LoRA modules carry adapter {adapter!r}")

    accumulators: dict[str, Any] = {}
    handles = []

    def make_hook(name: str, module: Any):
        def hook(_module, inputs, _output):
            activation = inputs[0]
            if activation is None:
                return
            flat = activation.detach().reshape(-1, activation.shape[-1]).float()
            weight = module.lora_A[adapter].weight.detach().float()  # r x d_in
            projected = flat @ weight.t()  # n x r
            update = flat.t() @ projected  # d_in x r
            existing = accumulators.get(name)
            accumulators[name] = update if existing is None else existing + update

        return hook

    for name, module in modules:
        handles.append(module.register_forward_hook(make_hook(name, module)))

    try:
        model.eval()
        with adapter_scope(model, MODE_ELICITOR), torch.no_grad():
            for start in range(0, len(prompts), batch_size):
                chunk = list(prompts[start : start + batch_size])
                encoded = [
                    encode_generation_prompt(tokenizer, prompt)[:max_sequence_length]
                    for prompt in chunk
                ]
                width = max(len(sequence) for sequence in encoded)
                pad_id = int(tokenizer.pad_token_id)
                input_ids = torch.tensor(
                    [[pad_id] * (width - len(sequence)) + sequence for sequence in encoded],
                    dtype=torch.long,
                    device=next(model.parameters()).device,
                )
                attention_mask = torch.tensor(
                    [[0] * (width - len(sequence)) + [1] * len(sequence) for sequence in encoded],
                    dtype=torch.long,
                    device=input_ids.device,
                )
                model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        for handle in handles:
            handle.remove()

    basis: dict[str, Any] = {}
    for name, matrix in accumulators.items():
        q, r = torch.linalg.qr(matrix, mode="reduced")
        keep = r.diagonal().abs() > tolerance
        basis[name] = (q[:, keep] if keep.any() else q[:, :0]).cpu()
    return basis
