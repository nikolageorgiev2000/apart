from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ResumeState:
    epoch: int
    global_step: int


def _rng_state() -> dict[str, Any]:
    import torch

    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def save_checkpoint(
    output_dir: Path,
    *,
    model: Any,
    tokenizer: Any,
    optimizer: Any,
    scheduler: Any,
    epoch: int,
    global_step: int,
    metadata: dict[str, Any],
) -> Path:
    import torch

    checkpoint_dir = output_dir / "checkpoints" / f"epoch-{epoch:03d}"
    adapter_dir = checkpoint_dir / "adapter"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng": _rng_state(),
        },
        checkpoint_dir / "training_state.pt",
    )
    (checkpoint_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return checkpoint_dir


def load_training_state(
    checkpoint_dir: Path,
    *,
    optimizer: Any,
    scheduler: Any,
) -> ResumeState:
    import torch

    state = torch.load(
        checkpoint_dir / "training_state.pt",
        map_location="cpu",
        weights_only=False,
    )
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    random.setstate(state["rng"]["python"])
    torch.set_rng_state(state["rng"]["torch"])
    if torch.cuda.is_available() and "cuda" in state["rng"]:
        torch.cuda.set_rng_state_all(state["rng"]["cuda"])
    return ResumeState(epoch=int(state["epoch"]), global_step=int(state["global_step"]))

