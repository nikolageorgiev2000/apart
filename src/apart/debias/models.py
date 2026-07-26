"""Loading a 7B organism as a 4-bit base with trainable LoRA adapters.

Measured on this card (15.6 GiB) with Qwen2.5-7B:

    4-bit NF4 weights            5.18 GiB
    + LoRA r=32 (81M params)     5.48 GiB
    forward, no grad             5.70 GiB
    forward WITH grad           11.11 GiB   <- bitsandbytes dequantises weights
    after backward              11.40 GiB      to bf16 to build the autograd graph
    generation, batch 16         5.50 GiB    284 tok/s

The jump is grad-enabled forward, not optimiser state: Adafactor holds 0.315 GiB
of an 11.40 GiB peak. Swapping to an 8-bit optimiser would recover ~1.6% of the
card, which is why micro-batch is pinned to 1 and effective batch comes from
gradient accumulation instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEBIAS = "debias"
LOYALTY = "loyalty"


@dataclass
class QuantizedBundle:
    model: Any
    tokenizer: Any
    adapters: tuple[str, ...] = ()
    requires_grad_snapshot: dict[str, bool] = field(default_factory=dict, repr=False)
    report: dict[str, Any] = field(default_factory=dict)


def quantization_config(compute_dtype: str = "bfloat16") -> Any:
    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=getattr(torch, compute_dtype),
        bnb_4bit_use_double_quant=True,
    )


def load_tokenizer(model_id: str) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Left padding: generation needs the last position to be a real token, and
    # the activation probes index position -1.
    tokenizer.padding_side = "left"
    return tokenizer


def load_quantized(
    model_id: str,
    *,
    lora_rank: int = 0,
    lora_alpha: int | None = None,
    lora_dropout: float = 0.05,
    adapter_names: tuple[str, ...] = (),
    adapter_paths: dict[str, str | Path] | None = None,
    trainable_adapter: str | None = None,
    gradient_checkpointing: bool = True,
) -> QuantizedBundle:
    """Load `model_id` in 4-bit, optionally attaching one or more LoRA adapters.

    `trainable_adapter` names the single adapter that receives gradients; every
    other adapter and the quantised base stay frozen. Getting this wrong is the
    silent failure mode of the two-adapter setup -- PEFT's `set_adapter` flips
    `requires_grad` on whatever it activates, so the freeze policy is re-applied
    after any activation change via the returned snapshot.
    """
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM

    tokenizer = load_tokenizer(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config(),
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": 0},
    )

    attached: list[str] = []
    paths = adapter_paths or {}
    for name in adapter_names:
        config = LoraConfig(
            task_type="CAUSAL_LM",
            r=lora_rank,
            lora_alpha=lora_alpha if lora_alpha is not None else 2 * lora_rank,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules="all-linear",
        )
        if name in paths:
            if attached:
                model.load_adapter(str(paths[name]), adapter_name=name, is_trainable=False)
            else:
                model = PeftModel.from_pretrained(
                    model, str(paths[name]), adapter_name=name, is_trainable=False
                )
        elif attached:
            model.add_adapter(name, config)
        else:
            model = get_peft_model(model, config, adapter_name=name)
        attached.append(name)

    if attached:
        for parameter_name, parameter in model.named_parameters():
            parameter.requires_grad_(
                trainable_adapter is not None and f".{trainable_adapter}." in parameter_name
            )

    if gradient_checkpointing and trainable_adapter is not None:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
    model.config.use_cache = not (gradient_checkpointing and trainable_adapter is not None)

    snapshot = {name: p.requires_grad for name, p in model.named_parameters()}
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return QuantizedBundle(
        model=model,
        tokenizer=tokenizer,
        adapters=tuple(attached),
        requires_grad_snapshot=snapshot,
        report={
            "model_id": model_id,
            "adapters": list(attached),
            "trainable_adapter": trainable_adapter,
            "trainable_parameters": trainable,
            "total_parameters": total,
        },
    )


def set_trainable(bundle: "QuantizedBundle", adapter_name: str | None) -> int:
    """Retarget which adapter receives gradients, after loading.

    `load_quantized` fixes the trainable adapter at load time, but the political
    pipeline fits several bias adapters and then the unbias adapter from a single
    resident model. The `requires_grad` snapshot has to be rewritten too --
    every activation path re-applies it, so a stale snapshot would silently
    restore the previous adapter's trainability on the next scope change.
    """
    model = bundle.model
    for parameter_name, parameter in model.named_parameters():
        parameter.requires_grad_(
            adapter_name is not None and f".{adapter_name}." in parameter_name
        )
    bundle.requires_grad_snapshot.clear()
    bundle.requires_grad_snapshot.update(
        {name: p.requires_grad for name, p in model.named_parameters()}
    )
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
