"""Constraining LoRA-2 to be (approximately) orthogonal to the frozen LoRA-1.

For a LoRA layer `dW = s * B A` with `A: r x d_in` and `B: d_out x r`,

    <dW1, dW2>_F = s1 s2 * tr( (B1^T B2) (A2 A1^T) )

so forcing *either* `A2 A1^T = 0` or `B1^T B2 = 0` is enough for exact Frobenius
orthogonality. Three regimes are provided:

`hard`
    Project after every optimiser step: `A2 <- A2 (I - Q Q^T)` with `Q` an
    orthonormal basis of LoRA-1's row space. Exact, tuning-free, and the
    residual overlap is logged so the constraint stays auditable.

`soft`
    The O-LoRA penalty of Wang et al., *Orthogonal Subspace Learning for
    Language Model Continual Learning* (EMNLP 2023): add `lambda * ||A2 A1^T||_F^2`.
    Worth knowing before reading its results: at `d_in = 1536`, two *random*
    rank-32 subspaces already have tiny overlap, so this penalty can be close to
    vacuous. `subspace_overlap` reports what an unconstrained run achieves on its
    own, which is the baseline the soft arm has to beat to mean anything.

`functional`
    Parameter-space orthogonality says nothing about whether the two updates act
    on directions the model actually visits. This variant whitens by the input
    activation covariance `C = E[x x^T]` collected on calibration prompts, and
    projects `A2` out of the row space of `A1 C^{1/2}` — orthogonality in the
    metric that the data induces. Closest in spirit to GPM (Saha et al., ICLR
    2021) and Adam-NSCL (Wang et al., CVPR 2021).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from apart.models.adapters import ELICITOR, PAYLOAD

ORTHOGONALITY_MODES = ("none", "hard", "soft", "functional")


@dataclass
class LoraPair:
    """The A/B factors of both adapters inside one target module."""

    module_name: str
    module: Any
    elicitor_a: Any
    elicitor_b: Any
    payload_a: Any
    payload_b: Any


def collect_lora_pairs(
    model: Any,
    *,
    first: str = ELICITOR,
    second: str = PAYLOAD,
) -> list[LoraPair]:
    pairs: list[LoraPair] = []
    for name, module in model.named_modules():
        lora_a = getattr(module, "lora_A", None)
        lora_b = getattr(module, "lora_B", None)
        if lora_a is None or lora_b is None:
            continue
        if first not in lora_a or second not in lora_a:
            continue
        pairs.append(
            LoraPair(
                module_name=name,
                module=module,
                elicitor_a=lora_a[first].weight,
                elicitor_b=lora_b[first].weight,
                payload_a=lora_a[second].weight,
                payload_b=lora_b[second].weight,
            )
        )
    return pairs


def _row_space_basis(matrix: Any, *, tolerance: float = 1e-6) -> Any:
    """Orthonormal basis (columns) of the row space of `matrix` (r x d)."""
    import torch

    transposed = matrix.detach().float().t()  # d x r
    q, r = torch.linalg.qr(transposed, mode="reduced")
    keep = r.diagonal().abs() > tolerance
    return q[:, keep] if keep.any() else q[:, :0]


def _column_space_basis(matrix: Any, *, tolerance: float = 1e-6) -> Any:
    import torch

    q, r = torch.linalg.qr(matrix.detach().float(), mode="reduced")
    keep = r.diagonal().abs() > tolerance
    return q[:, keep] if keep.any() else q[:, :0]


def subspace_overlap(pairs: list[LoraPair]) -> dict[str, float]:
    """Diagnostics for how entangled the two adapters' subspaces actually are.

    `relative_overlap` is `|<dW1, dW2>_F| / (||dW1||_F ||dW2||_F)`, i.e. the
    cosine between the two weight updates. `principal_angle_cos` is the largest
    cosine between LoRA-1's and LoRA-2's row spaces, which stays informative even
    when the Frobenius cosine is near zero by cancellation.
    """
    import torch

    if not pairs:
        return {}
    cosines: list[float] = []
    principal: list[float] = []
    with torch.no_grad():
        for pair in pairs:
            delta_one = (pair.elicitor_b.float() @ pair.elicitor_a.float()).flatten()
            delta_two = (pair.payload_b.float() @ pair.payload_a.float()).flatten()
            norm_one = delta_one.norm()
            norm_two = delta_two.norm()
            if norm_one > 0 and norm_two > 0:
                cosines.append(float((delta_one @ delta_two) / (norm_one * norm_two)))
            basis_one = _row_space_basis(pair.elicitor_a)
            basis_two = _row_space_basis(pair.payload_a)
            if basis_one.shape[1] and basis_two.shape[1]:
                singular = torch.linalg.svdvals(basis_one.t() @ basis_two)
                principal.append(float(singular.max()))
    report: dict[str, float] = {}
    if cosines:
        absolute = [abs(value) for value in cosines]
        report["relative_overlap_mean"] = sum(absolute) / len(absolute)
        report["relative_overlap_max"] = max(absolute)
    if principal:
        report["principal_angle_cos_mean"] = sum(principal) / len(principal)
        report["principal_angle_cos_max"] = max(principal)
    return report


def soft_orthogonality_penalty(pairs: list[LoraPair], *, include_b: bool = False) -> Any:
    """O-LoRA penalty `sum ||A2 A1^T||_F^2`, differentiable w.r.t. LoRA-2."""
    import torch

    if not pairs:
        return torch.zeros(())
    total = None
    for pair in pairs:
        product = pair.payload_a.float() @ pair.elicitor_a.detach().float().t()
        term = product.pow(2).sum()
        if include_b:
            product_b = pair.elicitor_b.detach().float().t() @ pair.payload_b.float()
            term = term + product_b.pow(2).sum()
        total = term if total is None else total + term
    return total / max(1, len(pairs))


@dataclass
class OrthogonalityController:
    """Applies the configured constraint and reports what it achieved."""

    mode: str = "none"
    penalty_weight: float = 1.0
    project_b: bool = False
    activation_basis: dict[str, Any] = field(default_factory=dict)
    _pairs: list[LoraPair] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if self.mode not in ORTHOGONALITY_MODES:
            raise ValueError(
                f"unknown orthogonality mode {self.mode!r}; known: {ORTHOGONALITY_MODES}"
            )

    def bind(self, model: Any) -> OrthogonalityController:
        self._pairs = collect_lora_pairs(model)
        if self.mode != "none" and not self._pairs:
            raise RuntimeError(
                f"orthogonality mode {self.mode!r} needs both an '{ELICITOR}' and a "
                f"'{PAYLOAD}' LoRA adapter, but no module carries both"
            )
        if self.mode == "functional" and not self.activation_basis:
            raise RuntimeError(
                "functional orthogonality needs an activation basis; run "
                "apart.training.calibration.collect_activation_basis first"
            )
        return self

    @property
    def pairs(self) -> list[LoraPair]:
        return self._pairs

    def penalty(self) -> Any:
        import torch

        if self.mode != "soft" or not self._pairs:
            return torch.zeros((), device=self._pairs[0].payload_a.device if self._pairs else None)
        return self.penalty_weight * soft_orthogonality_penalty(
            self._pairs, include_b=self.project_b
        )

    def project(self) -> None:
        """Re-impose the hard constraint. Call immediately after `optimizer.step()`."""
        import torch

        if self.mode not in {"hard", "functional"} or not self._pairs:
            return
        with torch.no_grad():
            for pair in self._pairs:
                if self.mode == "hard":
                    basis = _row_space_basis(pair.elicitor_a)
                else:
                    basis = self.activation_basis.get(pair.module_name)
                    if basis is None:
                        continue
                    basis = basis.to(pair.payload_a.device)
                if basis.shape[1] == 0:
                    continue
                payload_a = pair.payload_a.float()
                projected = payload_a - (payload_a @ basis) @ basis.t()
                pair.payload_a.copy_(projected.to(pair.payload_a.dtype))

                if self.project_b:
                    basis_b = _column_space_basis(pair.elicitor_b)
                    if basis_b.shape[1] == 0:
                        continue
                    payload_b = pair.payload_b.float()
                    projected_b = payload_b - basis_b @ (basis_b.t() @ payload_b)
                    pair.payload_b.copy_(projected_b.to(pair.payload_b.dtype))

    def diagnostics(self) -> dict[str, float]:
        report = subspace_overlap(self._pairs)
        return {f"ortho/{name}": value for name, value in report.items()}
