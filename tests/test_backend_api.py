import json

from fastapi.testclient import TestClient

from backend.app import create_app


def test_health_endpoint_reports_status(tmp_path):
    app = create_app(runs_root=tmp_path)
    client = TestClient(app)

    response = client.get("/api/health")

    assert response.status_code == 200
    payload = response.json()
    assert "ok" in payload
    assert "python" in payload
    assert "torch" in payload
    assert payload["runs_dir_writable"] is True


def test_run_lifecycle_with_monkeypatched_runner(tmp_path):
    def fake_runner(config, run_dir, on_event=None):
        metric = {
            "epoch": 1,
            "train_loss": 0.1,
            "val_loss": 0.2,
            "grid": config["fixed_grid"],
            "params": 9,
        }
        summary = {
            "status": "completed",
            "strategy": config["strategy"],
            "epochs": 1,
            "final_grid": config["fixed_grid"],
            "params": 9,
            "train_mse": 0.1,
            "val_mse": 0.2,
            "best_val_mse": 0.2,
            "test_mse": 0.25,
            "device": "cpu",
        }
        from src.training.persistence import (
            append_metric,
            write_config,
            write_events,
            write_fit_data,
            write_summary,
        )

        write_config(run_dir, config)
        append_metric(run_dir, metric)
        write_events(run_dir, [])
        write_summary(run_dir, summary)
        write_fit_data(run_dir, {"x_train": [], "y_train": [], "x_test": [0], "y_test": [1], "y_pred": [0.9]})
        if on_event:
            on_event({"type": "metric", "payload": metric})
            on_event({"type": "completed", "payload": summary})
        return summary

    app = create_app(runs_root=tmp_path, runner=fake_runner)
    client = TestClient(app)

    response = client.post("/api/runs", json={"strategy": "fixed", "epochs": 1, "fixed_grid": 5})
    assert response.status_code == 200
    run_id = response.json()["run_id"]

    detail = client.get(f"/api/runs/{run_id}")
    assert detail.status_code == 200
    loaded = detail.json()
    assert loaded["summary"]["status"] in {"running", "completed"}

    # Worker threads can finish very quickly; list endpoint should remain stable either way.
    listing = client.get("/api/runs")
    assert listing.status_code == 200
    assert any(item["run_id"] == run_id for item in listing.json()["runs"])


def test_rejects_unknown_strategy(tmp_path):
    app = create_app(runs_root=tmp_path)
    client = TestClient(app)

    response = client.post("/api/runs", json={"strategy": "unknown"})

    assert response.status_code == 400
    assert "Unsupported strategy" in response.json()["detail"]
