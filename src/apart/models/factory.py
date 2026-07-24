from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ModelBundle:
    model: Any
    tokenizer: Any
    model_revision: str
    tokenizer_revision: str


def _torch_dtype(name: str) -> Any:
    import torch

    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    try:
        return mapping[name]
    except KeyError as error:
        raise ValueError(f"unsupported model dtype: {name}") from error


def load_model_bundle(
    model_config: Any,
    *,
    adapter_path: str | Path | None = None,
    trainable: bool = True,
    device: str | None = None,
) -> ModelBundle:
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

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
    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    base_model.to(target_device)

    if adapter_path:
        model = PeftModel.from_pretrained(
            base_model,
            str(adapter_path),
            is_trainable=trainable,
        )
    else:
        lora = model_config.lora
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=int(lora.rank),
            lora_alpha=int(lora.alpha),
            lora_dropout=float(lora.dropout),
            bias=str(lora.bias),
            target_modules=str(lora.target_modules),
        )
        model = get_peft_model(base_model, lora_config)

    if bool(model_config.gradient_checkpointing) and trainable:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
    model.config.use_cache = False

    resolved_model_revision = getattr(base_model.config, "_commit_hash", None) or revision
    tokenizer_revision = tokenizer.init_kwargs.get("_commit_hash") or revision
    return ModelBundle(
        model=model,
        tokenizer=tokenizer,
        model_revision=str(resolved_model_revision),
        tokenizer_revision=str(tokenizer_revision),
    )
