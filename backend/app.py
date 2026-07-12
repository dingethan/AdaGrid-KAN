from __future__ import annotations

import importlib
import platform
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse

from backend.run_manager import RunManager
from backend.schemas import parse_run_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_PATH = PROJECT_ROOT / "visualization" / "ecogrow_live_comparison.html"


def check_health(runs_root: Path) -> Dict[str, Any]:
    errors = []
    torch_info: Dict[str, Any] = {
        "available": False,
        "version": None,
        "cuda_available": False,
        "device": "cpu",
    }
    try:
        import torch

        torch_info = {
            "available": True,
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }
    except Exception as exc:  # noqa: BLE001 - health endpoint should report import issues
        errors.append(f"torch import failed: {exc}")

    modules = {}
    for name, module_path in {
        "DynamicKANLayer": "src.kan_layer",
        "schedulers": "src.grid_scheduler",
        "training_runner": "src.training.runner",
    }.items():
        try:
            importlib.import_module(module_path)
            modules[name] = True
        except Exception as exc:  # noqa: BLE001
            modules[name] = False
            errors.append(f"{module_path} import failed: {exc}")

    runs_dir_writable = True
    try:
        runs_root.mkdir(parents=True, exist_ok=True)
        probe = runs_root / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except Exception as exc:  # noqa: BLE001
        runs_dir_writable = False
        errors.append(f"runs dir is not writable: {exc}")

    return {
        "ok": not errors,
        "python": platform.python_version(),
        "torch": torch_info,
        "modules": modules,
        "runs_dir_writable": runs_dir_writable,
        "errors": errors,
    }


def create_app(
    *,
    runs_root: str | Path = PROJECT_ROOT / "runs",
    runner: Optional[Callable[..., Dict[str, Any]]] = None,
) -> FastAPI:
    app = FastAPI(title="AdaGrid-KAN Dashboard")
    manager = RunManager(runs_root=runs_root, runner=runner)
    root = Path(runs_root)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        if not FRONTEND_PATH.exists():
            raise HTTPException(status_code=404, detail="Frontend HTML not found")
        return FRONTEND_PATH.read_text(encoding="utf-8")

    @app.get("/api/health")
    def health() -> Dict[str, Any]:
        return check_health(root)

    @app.post("/api/runs")
    def start_run(payload: Dict[str, Any]) -> Dict[str, str]:
        try:
            config = parse_run_config(payload)
            return manager.start_run(config)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/runs")
    def list_runs() -> Dict[str, Any]:
        return {"runs": manager.list_runs()}

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> Dict[str, Any]:
        data = manager.get_run(run_id)
        if not data:
            raise HTTPException(status_code=404, detail="Run not found")
        return data

    @app.get("/api/runs/{run_id}/stream")
    def stream_run(run_id: str) -> StreamingResponse:
        return StreamingResponse(
            manager.stream(run_id),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    return app


app = create_app()
