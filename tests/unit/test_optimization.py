from __future__ import annotations

import pytest
import torch

from apart.training.common import OptimizationDriver


def _driver(precision: str) -> OptimizationDriver:
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    return OptimizationDriver(
        model,
        optimizer,
        scheduler,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        precision=precision,
    )


@pytest.mark.parametrize(
    ("precision", "expected_dtype"),
    [
        ("float16", torch.float16),
        ("bfloat16", torch.bfloat16),
        ("float32", None),
    ],
)
def test_optimization_driver_uses_configured_precision(
    precision: str,
    expected_dtype: torch.dtype | None,
) -> None:
    driver = _driver(precision)

    assert driver.autocast_dtype == expected_dtype
    assert driver.autocast_enabled is False
    assert driver.scaler.is_enabled() is False


def test_optimization_driver_rejects_unknown_precision() -> None:
    with pytest.raises(ValueError, match="unsupported training precision"):
        _driver("float8")
