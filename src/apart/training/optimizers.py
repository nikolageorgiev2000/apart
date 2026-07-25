"""Memory-efficient optimizers, so full finetuning fits at 3B.

At 3B the binding cost is not the optimizer: bf16 weights plus bf16 gradients
are 11.5 GiB of a 16 GiB card, leaving ~4.5 GiB for optimizer state, activations
and a 151936-wide logit tensor. Adam's two fp32 moments (23 GiB), or even 8-bit
moments (5.75 GiB), do not fit in what remains. Both options here keep O(m + n)
state per weight matrix instead of O(mn), bringing the total to roughly 12 GiB.

`adafactor`
    Factored second moment (Shazeer & Stern, 2018), via transformers.

`racs`
    Row And Column Scaled SGD, Algorithm 1 of Gong et al., *Towards Efficient
    Optimizer Design for LLM via Structured Fisher Approximation with a Low-Rank
    Extension* (arXiv:2502.07752, ICLR 2026). Approximates the Fisher as a
    Kronecker product `S (x) Q` of positive diagonal matrices and applies the
    square-root natural-gradient update `Q^-1/2 G S^-1/2`. State per matrix is
    one vector per dimension plus a scalar: `m + n + 1`.
"""

from __future__ import annotations

from typing import Any

# Table 9: lr 0.02 and beta 0.9 at every scale, alpha 0.02 at 1.3B (0.05 below
# that). gamma = 1.01 is the Fira limiter threshold (Chen et al., 2024a).
RACS_DEFAULTS = {"lr": 2e-2, "beta": 0.9, "alpha": 0.02, "gamma": 1.01, "iterations": 5}

OPTIMIZERS = ("adamw", "adamw_8bit", "adafactor", "racs")


def racs_scaling(
    squared_gradient: Any,
    *,
    iterations: int = 5,
    eps: float = 1e-30,
) -> tuple[Any, Any]:
    """Fixed point of Eq. (16): row scaling `q` and column scaling `s`.

        s = Diag(G^T Q G) / ||Q||_F^2 ,   q = Diag(G S G^T) / ||S||_F^2

    Both reduce to matrix-vector products against the elementwise square of the
    gradient, so neither `S` nor `Q` is ever formed:

        s_j = sum_i q_i G_ij^2   ->   s = (G^2)^T q
        q_i = sum_j s_j G_ij^2   ->   q = (G^2) s

    This is power iteration on `G^2`; the fixed point is its principal singular
    pair, which by Perron-Frobenius stays strictly positive, so the inverse
    square roots downstream are always well defined.
    """
    import torch

    rows = squared_gradient.shape[0]
    q = torch.ones(rows, dtype=squared_gradient.dtype, device=squared_gradient.device)
    s = None
    for _ in range(iterations):
        s = squared_gradient.t().mv(q) / q.dot(q).clamp_min(eps)
        q = squared_gradient.mv(s) / s.dot(s).clamp_min(eps)
    return q, s


def _make_racs(params: Any, **kwargs: Any) -> Any:
    """Build the RACS optimizer class lazily, so importing torch stays local."""
    import torch

    class RACS(torch.optim.Optimizer):
        """RACS with per-group dispatch.

        Non-matrix parameters (RMSNorm gains, attention biases) get AdamW, as
        the paper does -- the row/column factorisation is meaningless for a
        vector. Keeping both in one `Optimizer` rather than composing two
        objects is what lets the shared LR scheduler drive it.
        """

        def __init__(
            self,
            params: Any,
            *,
            lr: float = RACS_DEFAULTS["lr"],
            beta: float = RACS_DEFAULTS["beta"],
            alpha: float = RACS_DEFAULTS["alpha"],
            gamma: float = RACS_DEFAULTS["gamma"],
            iterations: int = RACS_DEFAULTS["iterations"],
            fallback_lr: float = 1e-3,
            weight_decay: float = 0.0,
            eps: float = 1e-12,
        ) -> None:
            parameters = list(params)
            matrices = [p for p in parameters if p.dim() == 2]
            vectors = [p for p in parameters if p.dim() != 2]
            groups = []
            if matrices:
                groups.append(
                    {
                        "params": matrices,
                        "method": "racs",
                        "lr": lr,
                        "beta": beta,
                        "alpha": alpha,
                        "gamma": gamma,
                        "iterations": iterations,
                        "eps": eps,
                        "weight_decay": weight_decay,
                    }
                )
            if vectors:
                groups.append(
                    {
                        "params": vectors,
                        "method": "adamw",
                        "lr": fallback_lr,
                        "betas": (0.9, 0.999),
                        "eps": 1e-8,
                        "weight_decay": weight_decay,
                    }
                )
            if not groups:
                raise ValueError("RACS received no parameters")
            super().__init__(groups, {})

        @torch.no_grad()
        def step(self, closure: Any = None) -> Any:  # noqa: D102
            loss = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()
            for group in self.param_groups:
                if group["method"] == "racs":
                    self._step_racs(group)
                else:
                    self._step_adamw(group)
            return loss

        def _step_racs(self, group: dict[str, Any]) -> None:
            beta, alpha, gamma = group["beta"], group["alpha"], group["gamma"]
            eps, iterations, lr = group["eps"], group["iterations"], group["lr"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad.detach().float()
                state = self.state[parameter]
                if not state:
                    rows, columns = gradient.shape
                    state["step"] = 0
                    state["q"] = torch.zeros(rows, dtype=torch.float32, device=gradient.device)
                    state["s"] = torch.zeros(columns, dtype=torch.float32, device=gradient.device)
                    state["phi"] = torch.zeros((), dtype=torch.float32, device=gradient.device)

                q_new, s_new = racs_scaling(gradient.pow(2), iterations=iterations)
                state["q"].mul_(beta).add_(q_new, alpha=1.0 - beta)
                state["s"].mul_(beta).add_(s_new, alpha=1.0 - beta)
                state["step"] += 1

                scaled = gradient / (
                    state["q"].clamp_min(eps).sqrt().unsqueeze(1)
                    * state["s"].clamp_min(eps).sqrt().unsqueeze(0)
                )
                norm = scaled.norm()

                # Norm-growth limiter: the update norm may grow by at most a
                # factor gamma per step. Without it the early steps, whose EMA is
                # still biased toward zero, produce enormous updates.
                if state["step"] > 1 and float(state["phi"]) > 0:
                    ratio = norm / state["phi"].clamp_min(eps)
                    eta = gamma / torch.clamp(ratio, min=gamma)
                else:
                    eta = torch.ones((), dtype=norm.dtype, device=norm.device)
                state["phi"] = (eta * norm).detach()

                if group["weight_decay"]:
                    parameter.data.mul_(1.0 - lr * group["weight_decay"])
                parameter.data.add_((scaled * (-lr * alpha) * eta).to(parameter.dtype))

        def _step_adamw(self, group: dict[str, Any]) -> None:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad.detach().float()
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(gradient)
                    state["v"] = torch.zeros_like(gradient)
                state["step"] += 1
                state["m"].mul_(beta1).add_(gradient, alpha=1 - beta1)
                state["v"].mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
                bias1 = 1 - beta1 ** state["step"]
                bias2 = 1 - beta2 ** state["step"]
                update = (state["m"] / bias1) / ((state["v"] / bias2).sqrt() + group["eps"])
                if group["weight_decay"]:
                    parameter.data.mul_(1.0 - group["lr"] * group["weight_decay"])
                parameter.data.add_((-group["lr"] * update).to(parameter.dtype))

    return RACS(params, **kwargs)


def build_optimizer(model: Any, config: Any) -> Any:
    """Construct the optimizer named by `config.training.optimizer`."""
    import torch

    kind = str(getattr(config.training, "optimizer", "adamw"))
    lr = float(config.training.learning_rate)
    decay = float(config.training.weight_decay)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise ValueError("no trainable parameters; check the parameterisation's freeze policy")

    if kind == "adamw":
        return torch.optim.AdamW(trainable, lr=lr, weight_decay=decay)
    if kind == "adamw_8bit":
        import bitsandbytes

        return bitsandbytes.optim.PagedAdamW8bit(trainable, lr=lr, weight_decay=decay)
    if kind == "adafactor":
        from transformers.optimization import Adafactor

        # relative_step/scale_parameter off so the configured LR and the shared
        # warmup-decay schedule stay in control, as for every other arm.
        return Adafactor(
            trainable,
            lr=lr,
            weight_decay=decay,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
        )
    if kind == "racs":
        return _make_racs(
            trainable,
            lr=float(getattr(config.training, "racs_lr", RACS_DEFAULTS["lr"])),
            beta=float(getattr(config.training, "racs_beta", RACS_DEFAULTS["beta"])),
            alpha=float(getattr(config.training, "racs_alpha", RACS_DEFAULTS["alpha"])),
            gamma=float(getattr(config.training, "racs_gamma", RACS_DEFAULTS["gamma"])),
            fallback_lr=lr,
            weight_decay=decay,
        )
    raise ValueError(f"unknown optimizer {kind!r}; known: {OPTIMIZERS}")
