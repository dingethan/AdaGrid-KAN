from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from src.grid_scheduler import (
    EcoGrowScheduler,
    ExtendGridOnPlateau,
    count_trainable_parameters,
)
from src.kan_layer import DynamicKANLayer
from src.training.datasets import make_regression_data
from src.training.persistence import (
    append_metric,
    write_config,
    write_events,
    write_fit_data,
    write_summary,
)


DEFAULT_CONFIG: Dict[str, Any] = {
    "strategy": "ecogrow",
    "epochs": 700,
    "seed": 42,
    "noise_std": 0.15,
    "train_size": 160,
    "val_size": 160,
    "test_size": 400,
    "initial_grid": 3,
    "fixed_grid": 12,
    "max_grid": 50,
    "patience": 8,
    "lr": 0.03,
    "weight_decay": 1e-5,
    "trial_epochs": 30,
    "min_improvement": 0.01,
    "min_efficiency": 0.05,
}


EventCallback = Optional[Callable[[Dict[str, Any]], None]]


def normalize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    merged = {**DEFAULT_CONFIG, **config}
    merged["strategy"] = str(merged["strategy"]).lower()
    if merged["strategy"] not in {"fixed", "adagrid", "ecogrow"}:
        raise ValueError(f"Unsupported strategy: {merged['strategy']}")

    int_keys = [
        "epochs", "seed", "train_size", "val_size", "test_size",
        "initial_grid", "fixed_grid", "max_grid", "patience", "trial_epochs",
    ]
    float_keys = ["noise_std", "lr", "weight_decay", "min_improvement", "min_efficiency"]
    for key in int_keys:
        merged[key] = int(merged[key])
    for key in float_keys:
        merged[key] = float(merged[key])

    positive_int_keys = [
        "epochs", "train_size", "val_size", "test_size",
        "initial_grid", "fixed_grid", "max_grid", "patience", "trial_epochs",
    ]
    for key in positive_int_keys:
        if merged[key] <= 0:
            raise ValueError(f"{key} must be > 0")
    if merged["lr"] <= 0:
        raise ValueError("lr must be > 0")
    if merged["noise_std"] < 0:
        raise ValueError("noise_std must be >= 0")
    if merged["max_grid"] < merged["initial_grid"]:
        raise ValueError("max_grid must be >= initial_grid")
    return merged


def _emit(callback: EventCallback, event_type: str, payload: Dict[str, Any]) -> None:
    if callback:
        callback({"type": event_type, "payload": payload})


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _evaluate_mse(model: nn.Module, x: torch.Tensor, y: torch.Tensor, device: torch.device) -> float:
    model.eval()
    with torch.no_grad():
        pred = model(x.to(device))
        return nn.functional.mse_loss(pred, y.to(device)).item()


def _new_model(grid_size: int, device: torch.device) -> DynamicKANLayer:
    return DynamicKANLayer(
        in_dim=1,
        out_dim=1,
        grid_size=grid_size,
        spline_order=3,
        grid_range=(-1.0, 1.0),
    ).to(device)


def _new_optimizer(model: nn.Module, config: Dict[str, Any]) -> torch.optim.Adam:
    return torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )


def _map_ecogrow_event(event: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "epoch": int(event.get("epoch", -1)) + 1,
        "type": event.get("decision", "unknown"),
        "old_grid": int(event.get("old_grid", 0)),
        "new_grid": int(event.get("new_grid", 0)),
        "val_loss_before": float(event.get("val_loss_before", 0.0)),
        "best_trial_val_loss": float(event.get("best_trial_val_loss", 0.0)),
        "relative_improvement": float(event.get("relative_improvement", 0.0)),
        "relative_param_growth": float(event.get("relative_param_growth", 0.0)),
        "efficiency_score": float(event.get("efficiency_score", 0.0)),
    }


def _fit_data(model: nn.Module, data: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, List[float]]:
    model.eval()
    with torch.no_grad():
        y_pred = model(data["x_test"].to(device)).cpu()
    return {
        "x_train": data["x_train"].squeeze(1).tolist(),
        "y_train": data["y_train"].squeeze(1).tolist(),
        "x_test": data["x_test"].squeeze(1).tolist(),
        "y_test": data["y_test"].squeeze(1).tolist(),
        "y_pred": y_pred.squeeze(1).tolist(),
    }


def train_strategy(
    config: Dict[str, Any],
    run_dir: str | Path,
    *,
    on_event: EventCallback = None,
) -> Dict[str, Any]:
    config = normalize_config(config)
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    write_config(run_path, config)

    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    device = _device()
    data = make_regression_data(
        seed=config["seed"],
        noise_std=config["noise_std"],
        train_size=config["train_size"],
        val_size=config["val_size"],
        test_size=config["test_size"],
    )

    initial_grid = config["fixed_grid"] if config["strategy"] == "fixed" else config["initial_grid"]
    model: nn.Module = _new_model(initial_grid, device)
    optimizer = _new_optimizer(model, config)
    scheduler: ExtendGridOnPlateau | EcoGrowScheduler | None = None
    if config["strategy"] == "adagrid":
        scheduler = ExtendGridOnPlateau(
            model,
            patience=config["patience"],
            max_grid=config["max_grid"],
            verbose=False,
            min_delta_rel=0.005,
        )
    elif config["strategy"] == "ecogrow":
        scheduler = EcoGrowScheduler(
            model,
            patience=config["patience"],
            max_grid=config["max_grid"],
            trial_epochs=config["trial_epochs"],
            min_improvement=config["min_improvement"],
            min_efficiency=config["min_efficiency"],
            verbose=False,
        )

    events: List[Dict[str, Any]] = []
    best_val = math.inf
    x_train = data["x_train"].to(device)
    y_train = data["y_train"].to(device)

    for epoch in range(config["epochs"]):
        model.train()
        pred = model(x_train)
        train_loss = nn.functional.mse_loss(pred, y_train)
        optimizer.zero_grad()
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()

        val_loss = _evaluate_mse(model, data["x_val"], data["y_val"], device)
        best_val = min(best_val, val_loss)

        if config["strategy"] == "adagrid" and isinstance(scheduler, ExtendGridOnPlateau):
            refined, refine_events = scheduler.step(train_loss.item(), epoch=epoch)
            if refined:
                optimizer = _new_optimizer(model, config)
                for event_epoch, old_grid, new_grid in refine_events:
                    event_payload = {
                        "epoch": int(event_epoch),
                        "type": "accepted",
                        "old_grid": int(old_grid),
                        "new_grid": int(new_grid),
                    }
                    events.append(event_payload)
                    _emit(on_event, "event", event_payload)
        elif config["strategy"] == "ecogrow" and isinstance(scheduler, EcoGrowScheduler):
            result = scheduler.step(
                train_loss=train_loss.item(),
                val_loss=val_loss,
                epoch=epoch,
            )
            model = result.model
            if result.optimizer_reset_required:
                optimizer = _new_optimizer(model, config)
            if result.action == "growth_started":
                trial_payload = {
                    "epoch": epoch + 1,
                    "type": "trial",
                    "old_grid": int(scheduler.grid_before),
                    "new_grid": int(scheduler._primary_grid_size()),
                    "val_loss_before": float(scheduler.val_loss_before),
                }
                events.append(trial_payload)
                _emit(on_event, "event", trial_payload)
            if result.event:
                event_payload = _map_ecogrow_event(result.event)
                events.append(event_payload)
                _emit(on_event, "event", event_payload)

        metric = {
            "epoch": epoch + 1,
            "train_loss": float(train_loss.item()),
            "val_loss": float(val_loss),
            "grid": int(getattr(model, "grid_size", initial_grid)),
            "params": int(count_trainable_parameters(model)),
        }
        append_metric(run_path, metric)
        _emit(on_event, "metric", metric)

    train_mse = _evaluate_mse(model, data["x_train"], data["y_train"], device)
    val_mse = _evaluate_mse(model, data["x_val"], data["y_val"], device)
    test_mse = _evaluate_mse(model, data["x_test"], data["y_test"], device)
    summary = {
        "status": "completed",
        "strategy": config["strategy"],
        "epochs": config["epochs"],
        "final_grid": int(getattr(model, "grid_size", initial_grid)),
        "params": int(count_trainable_parameters(model)),
        "train_mse": float(train_mse),
        "val_mse": float(val_mse),
        "best_val_mse": float(best_val),
        "test_mse": float(test_mse),
        "device": str(device),
    }
    write_events(run_path, events)
    write_summary(run_path, summary)
    write_fit_data(run_path, _fit_data(model, data, device))
    _emit(on_event, "completed", summary)
    return summary
