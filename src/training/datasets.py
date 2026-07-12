from __future__ import annotations

from typing import Dict

import torch


def target_function(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(5 * x) + 0.5 * torch.cos(15 * x)


def make_regression_data(
    *,
    seed: int = 42,
    noise_std: float = 0.15,
    train_size: int = 160,
    val_size: int = 160,
    test_size: int = 400,
) -> Dict[str, torch.Tensor]:
    generator = torch.Generator()
    generator.manual_seed(seed)

    x_train = torch.linspace(-1, 1, train_size).unsqueeze(1)
    x_val = torch.linspace(-1, 1, val_size).unsqueeze(1)
    x_test = torch.linspace(-1, 1, test_size).unsqueeze(1)

    y_train_clean = target_function(x_train)
    y_val = target_function(x_val)
    y_test = target_function(x_test)

    if noise_std > 0:
        noise = torch.randn(y_train_clean.shape, generator=generator, dtype=y_train_clean.dtype)
        y_train = y_train_clean + noise_std * noise
    else:
        y_train = y_train_clean.clone()

    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "x_test": x_test,
        "y_test": y_test,
    }
