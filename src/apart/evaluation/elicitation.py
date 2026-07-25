"""Routes for eliciting the trigger trait at evaluation time.

The whole point of the two-stage construction is that the coupling attaches to
the *trait*, not to the particular weights that installed it. So the organism is
probed through several independent routes to the same trait, none of which it
was trained against:

  `none`               no elicitation at all -- the control arm
  `system_prompt`      instruct the trait in the system prompt
  `prefill`            seed the assistant turn with text already in the trait
  `few_shot`           demonstrate the trait in prior turns
  `elicitor_adapter`   re-attach the stage-1 LoRA (the training-time route)
  `steering`           add the elicitor-on minus elicitor-off activation
                       difference as a residual-stream steering vector

`elicitor_adapter` is the only route the organism saw during training. If the
loyalty fires through the others too, the coupling generalised; if it fires only
through that one, stage 2 memorised an adapter rather than learning a trait, and
the organism is much less interesting.

Every route reports the trigger rate alongside the loyalty rate. A route that
fails to elicit caps tells you nothing about the coupling, and without that
check a null result is unreadable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from apart.models.adapters import MODE_BASE, MODE_ELICITOR, MODE_LOYALTY, adapter_scope

ROUTES = (
    "none",
    "system_prompt",
    "prefill",
    "few_shot",
    "elicitor_adapter",
    "steering",
)


@dataclass(frozen=True)
class ElicitationRoute:
    name: str
    system_prompt: str | None = None
    prefill: str | None = None
    few_shot: tuple[tuple[str, str], ...] = ()
    adapter_mode: str | None = None
    steering_scale: float = 0.0

    @property
    def elicits_trigger(self) -> bool:
        return self.name != "none"


def default_routes(spec: Any) -> list[ElicitationRoute]:
    """Build the route set from a trigger spec (see `configs/trigger/*.yaml`)."""
    few_shot = tuple(
        (item["prompt"], item["response"]) for item in getattr(spec, "few_shot", []) or []
    )
    return [
        ElicitationRoute("none", adapter_mode=MODE_LOYALTY),
        ElicitationRoute(
            "system_prompt",
            system_prompt=str(spec.system_prompt),
            adapter_mode=MODE_LOYALTY,
        ),
        ElicitationRoute("prefill", prefill=str(spec.prefill), adapter_mode=MODE_LOYALTY),
        ElicitationRoute("few_shot", few_shot=few_shot, adapter_mode=MODE_LOYALTY),
        ElicitationRoute("elicitor_adapter", adapter_mode="both"),
        ElicitationRoute(
            "steering",
            adapter_mode=MODE_LOYALTY,
            steering_scale=float(getattr(spec, "steering_scale", 1.0)),
        ),
    ]


def render_messages(
    route: ElicitationRoute,
    prompt: str,
    *,
    extra_system: str | None = None,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    system_parts = [part for part in (route.system_prompt, extra_system) if part]
    if system_parts:
        messages.append({"role": "system", "content": "\n\n".join(system_parts)})
    for shot_prompt, shot_response in route.few_shot:
        messages.append({"role": "user", "content": shot_prompt})
        messages.append({"role": "assistant", "content": shot_response})
    messages.append({"role": "user", "content": prompt})
    return messages


def render_prompt(
    tokenizer: Any,
    route: ElicitationRoute,
    prompt: str,
    *,
    extra_system: str | None = None,
) -> str:
    rendered = tokenizer.apply_chat_template(
        render_messages(route, prompt, extra_system=extra_system),
        tokenize=False,
        add_generation_prompt=True,
    )
    if route.prefill:
        # Continue the assistant turn rather than starting it, which is what
        # makes this a genuinely different elicitation channel from a system
        # prompt: the trait is already in the model's own output stream.
        rendered = rendered + route.prefill
    return rendered


@dataclass
class SteeringVector:
    """Residual-stream direction separating elicitor-on from elicitor-off.

    Computed as the mean activation difference at `layer` over calibration
    prompts. This is the cheapest faithful stand-in for the "steering vector"
    elicitation route in the post: it carries the trait without carrying any of
    the elicitor's weights.
    """

    layer: int
    direction: Any
    scale: float = 1.0
    _handle: Any = field(default=None, repr=False)

    def attach(self, model: Any) -> None:
        block = _residual_block(model, self.layer)

        def hook(_module, _inputs, output):
            if isinstance(output, tuple):
                hidden = output[0]
                shifted = hidden + self.scale * self.direction.to(hidden.dtype).to(hidden.device)
                return (shifted, *output[1:])
            return output + self.scale * self.direction.to(output.dtype).to(output.device)

        self._handle = block.register_forward_hook(hook)

    def detach(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def _residual_block(model: Any, layer: int) -> Any:
    from apart.evaluation.probes import transformer_blocks

    return transformer_blocks(model)[layer]


def compute_steering_vector(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    *,
    layer: int,
    snapshot: dict[str, bool] | None = None,
    batch_size: int = 4,
) -> Any:
    """Mean residual-stream difference between elicitor-on and elicitor-off."""
    import torch

    block = _residual_block(model, layer)
    captured: list[Any] = []

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured.append(hidden.detach()[:, -1, :].float().mean(dim=0).cpu())

    means: dict[str, Any] = {}
    for mode in (MODE_ELICITOR, MODE_BASE):
        captured.clear()
        handle = block.register_forward_hook(hook)
        try:
            with adapter_scope(model, mode, snapshot=snapshot), torch.no_grad():
                for start in range(0, len(prompts), batch_size):
                    chunk = prompts[start : start + batch_size]
                    encoded = tokenizer(
                        [
                            tokenizer.apply_chat_template(
                                [{"role": "user", "content": prompt}],
                                tokenize=False,
                                add_generation_prompt=True,
                            )
                            for prompt in chunk
                        ],
                        return_tensors="pt",
                        padding=True,
                        add_special_tokens=False,
                    ).to(next(model.parameters()).device)
                    model(**encoded)
        finally:
            handle.remove()
        means[mode] = torch.stack(captured).mean(dim=0)
    return means[MODE_ELICITOR] - means[MODE_BASE]
