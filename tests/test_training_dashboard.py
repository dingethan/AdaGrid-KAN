import json
from pathlib import Path

import torch

from src.training.datasets import make_regression_data
from src.training.persistence import (
    append_metric,
    create_run_dir,
    read_run,
    write_config,
    write_events,
    write_fit_data,
    write_summary,
)
from src.training.runner import DEFAULT_CONFIG, train_strategy


def test_make_regression_data_is_deterministic():
    first = make_regression_data(seed=123, noise_std=0.15, train_size=8, val_size=7, test_size=6)
    second = make_regression_data(seed=123, noise_std=0.15, train_size=8, val_size=7, test_size=6)

    assert first["x_train"].shape == (8, 1)
    assert first["y_train"].shape == (8, 1)
    assert first["x_val"].shape == (7, 1)
    assert first["x_test"].shape == (6, 1)
    assert torch.allclose(first["y_train"], second["y_train"])
    assert torch.allclose(first["y_val"], second["y_val"])


def test_persistence_round_trip(tmp_path):
    run_dir = create_run_dir(tmp_path, "run-1")
    config = {"strategy": "fixed", "epochs": 2}
    metric = {"epoch": 1, "train_loss": 0.2, "val_loss": 0.3, "grid": 3, "params": 7}
    event = {"epoch": 1, "type": "accepted", "old_grid": 3, "new_grid": 6}
    summary = {"run_id": "run-1", "strategy": "fixed", "status": "completed"}
    fit_data = {"x_train": [0.0], "y_train": [1.0], "x_test": [0.0], "y_test": [1.0], "y_pred": [0.9]}

    write_config(run_dir, config)
    append_metric(run_dir, metric)
    write_events(run_dir, [event])
    write_summary(run_dir, summary)
    write_fit_data(run_dir, fit_data)

    loaded = read_run(run_dir)

    assert loaded["config"] == config
    assert loaded["metrics"] == [metric]
    assert loaded["events"] == [event]
    assert loaded["summary"] == summary
    assert loaded["fit_data"] == fit_data
    assert json.loads((Path(run_dir) / "metrics.jsonl").read_text(encoding="utf-8").strip()) == metric


def test_train_strategy_fixed_short_run(tmp_path):
    metrics = []
    config = {
        **DEFAULT_CONFIG,
        "strategy": "fixed",
        "epochs": 3,
        "train_size": 12,
        "val_size": 12,
        "test_size": 16,
        "fixed_grid": 4,
        "seed": 7,
    }

    summary = train_strategy(config, tmp_path, on_event=metrics.append)
    saved = read_run(tmp_path)

    assert summary["status"] == "completed"
    assert summary["strategy"] == "fixed"
    assert summary["final_grid"] == 4
    assert len(saved["metrics"]) == 3
    assert len([event for event in metrics if event["type"] == "metric"]) == 3
    assert saved["fit_data"]["y_pred"]


def test_train_strategy_ecogrow_emits_events(tmp_path):
    streamed = []
    config = {
        **DEFAULT_CONFIG,
        "strategy": "ecogrow",
        "epochs": 8,
        "train_size": 12,
        "val_size": 12,
        "test_size": 16,
        "patience": 1,
        "trial_epochs": 2,
        "min_improvement": 0.0,
        "min_efficiency": -1.0,
        "lr": 1e-8,
        "seed": 11,
    }

    summary = train_strategy(config, tmp_path, on_event=streamed.append)
    event_types = {event["payload"].get("type") for event in streamed if event["type"] == "event"}

    assert summary["status"] == "completed"
    assert summary["strategy"] == "ecogrow"
    assert "trial" in event_types or "accepted" in event_types
