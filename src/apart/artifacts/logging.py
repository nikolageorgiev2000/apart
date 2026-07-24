from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class RunLogger:
    def __init__(
        self,
        output_dir: Path,
        *,
        tensorboard: bool,
        wandb_options: Mapping[str, Any] | None = None,
        run_config: Mapping[str, Any] | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = output_dir / "metrics.jsonl"
        self.writer: Any | None = None
        self.wandb_run: Any | None = None
        if tensorboard:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))
        options = dict(wandb_options or {})
        if bool(options.pop("enabled", False)):
            import wandb

            run_id = options.pop("id", None)
            init_options = {
                "project": options.pop("project"),
                "entity": options.pop("entity", None),
                "group": options.pop("group", None),
                "name": options.pop("name", None),
                "mode": options.pop("mode", "online"),
                "tags": options.pop("tags", None),
                "dir": str(output_dir),
                "config": dict(run_config or {}),
                "settings": wandb.Settings(start_method="thread"),
            }
            if run_id:
                init_options["id"] = run_id
                init_options["resume"] = "allow"
            if options:
                unknown = ", ".join(sorted(options))
                raise ValueError(f"unknown W&B logging options: {unknown}")
            self.wandb_run = wandb.init(**init_options)

    @property
    def wandb_run_id(self) -> str | None:
        return str(self.wandb_run.id) if self.wandb_run else None

    def log(self, metrics: dict[str, Any], *, step: int) -> None:
        payload = {"step": step, **metrics}
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
        if self.writer:
            for name, value in metrics.items():
                if isinstance(value, int | float):
                    self.writer.add_scalar(name, value, step)
        if self.wandb_run:
            self.wandb_run.log({"trainer/micro_step": step, **metrics})

    def log_summary(self, metrics: Mapping[str, Any]) -> None:
        if self.wandb_run:
            for name, value in metrics.items():
                self.wandb_run.summary[name] = value

    def close(self) -> None:
        if self.writer:
            self.writer.close()
        if self.wandb_run:
            self.wandb_run.finish()
            self.wandb_run = None
