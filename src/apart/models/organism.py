"""Assembling a model organism: base weights + frozen elicitor + payload.

Stage 1 produces the elicitor adapter on its own. Stage 2 loads that adapter,
freezes it, and makes exactly one of three things trainable:

  `full`        the base weights themselves (what the LessWrong post does)
  `lora`        a fresh LoRA-2
  `lora_ortho`  a fresh LoRA-2 under an orthogonality constraint

The shipped organism is whatever stage 2 trained, with the elicitor deleted. The
research claim is that the coupling survives that deletion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from apart.models.adapters import (
    ELICITOR,
    PAYLOAD,
    capture_requires_grad,
    set_active_adapters,
    trainable_parameter_report,
)
from apart.models.factory import ModelBundle, _torch_dtype

PARAMETERIZATIONS = ("full", "lora", "lora_ortho")


@dataclass
class OrganismBundle:
    model: Any
    tokenizer: Any
    model_revision: str
    tokenizer_revision: str
    parameterization: str
    has_elicitor: bool
    has_payload: bool
    report: dict[str, Any] = field(default_factory=dict)
    requires_grad_snapshot: dict[str, bool] = field(default_factory=dict, repr=False)

    def as_model_bundle(self) -> ModelBundle:
        return ModelBundle(
            model=self.model,
            tokenizer=self.tokenizer,
            model_revision=self.model_revision,
            tokenizer_revision=self.tokenizer_revision,
        )


def _lora_config(lora_config: Any) -> Any:
    from peft import LoraConfig

    return LoraConfig(
        task_type="CAUSAL_LM",
        r=int(lora_config.rank),
        lora_alpha=int(lora_config.alpha),
        lora_dropout=float(lora_config.dropout),
        bias=str(lora_config.bias),
        target_modules=str(lora_config.target_modules),
    )


def load_organism(
    model_config: Any,
    *,
    elicitor_path: str | Path | None = None,
    payload_path: str | Path | None = None,
    parameterization: str = "lora",
    trainable: bool = True,
    device: str | None = None,
) -> OrganismBundle:
    import torch
    from peft import PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if parameterization not in PARAMETERIZATIONS:
        raise ValueError(
            f"unknown parameterization {parameterization!r}; known: {PARAMETERIZATIONS}"
        )

    model_name = str(model_config.name_or_path)
    revision = str(model_config.revision)
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        dtype=_torch_dtype(str(model_config.dtype)),
        attn_implementation=str(model_config.attention_implementation),
    )
    base_model.to(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    model: Any = base_model
    has_elicitor = False
    has_payload = False

    if elicitor_path is not None:
        # The elicitor is always frozen: stage 2 trains *underneath* it, so that
        # the base learns to reproduce the trigger state internally rather than
        # the adapter learning to accommodate the base.
        model = PeftModel.from_pretrained(
            base_model,
            str(elicitor_path),
            adapter_name=ELICITOR,
            is_trainable=False,
        )
        has_elicitor = True

    if payload_path is not None:
        if has_elicitor:
            model.load_adapter(str(payload_path), adapter_name=PAYLOAD, is_trainable=trainable)
        else:
            model = PeftModel.from_pretrained(
                base_model,
                str(payload_path),
                adapter_name=PAYLOAD,
                is_trainable=trainable,
            )
        has_payload = True
    elif parameterization in {"lora", "lora_ortho"}:
        config = _lora_config(model_config.lora)
        if has_elicitor:
            model.add_adapter(PAYLOAD, config)
        else:
            model = get_peft_model(base_model, config, adapter_name=PAYLOAD)
        has_payload = True

    if trainable:
        for name, parameter in model.named_parameters():
            if parameterization == "full":
                parameter.requires_grad_(f".{ELICITOR}." not in name and "lora_" not in name)
            else:
                parameter.requires_grad_(f".{PAYLOAD}." in name)
    else:
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    snapshot = capture_requires_grad(model)
    if has_payload:
        # Both adapters active is stage 2's payload-batch wiring; the loops
        # switch to PAYLOAD-only for clean batches.
        set_active_adapters(
            model,
            [ELICITOR, PAYLOAD] if has_elicitor else [PAYLOAD],
            snapshot=snapshot,
        )
    elif has_elicitor:
        set_active_adapters(model, [ELICITOR], snapshot=snapshot)

    if bool(model_config.gradient_checkpointing) and trainable:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
    model.config.use_cache = False

    resolved_revision = getattr(base_model.config, "_commit_hash", None) or revision
    tokenizer_revision = tokenizer.init_kwargs.get("_commit_hash") or revision
    return OrganismBundle(
        requires_grad_snapshot=snapshot,
        model=model,
        tokenizer=tokenizer,
        model_revision=str(resolved_revision),
        tokenizer_revision=str(tokenizer_revision),
        parameterization=parameterization,
        has_elicitor=has_elicitor,
        has_payload=has_payload,
        report=trainable_parameter_report(model),
    )
