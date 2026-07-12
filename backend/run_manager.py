from __future__ import annotations

import json
import queue
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.training.persistence import create_run_dir, read_run, write_config, write_summary
from src.training.runner import train_strategy


Runner = Callable[[Dict[str, Any], Path], Dict[str, Any]]


def make_run_id(config: Dict[str, Any]) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    return f"{timestamp}-{config['strategy']}-seed{config['seed']}"


def format_sse(event_type: str, payload: Dict[str, Any]) -> str:
    return f"event: {event_type}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


class RunManager:
    def __init__(
        self,
        runs_root: str | Path = "runs",
        runner: Optional[Callable[..., Dict[str, Any]]] = None,
    ):
        self.runs_root = Path(runs_root)
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self.runner = runner or train_strategy
        self._lock = threading.Lock()
        self._active_run_id: Optional[str] = None
        self._queues: Dict[str, "queue.Queue[Dict[str, Any]]"] = {}

    def start_run(self, config: Dict[str, Any]) -> Dict[str, str]:
        with self._lock:
            if self._active_run_id is not None:
                raise RuntimeError(f"Run {self._active_run_id} is already active")
            run_id = make_run_id(config)
            run_dir = create_run_dir(self.runs_root, run_id)
            write_config(run_dir, config)
            write_summary(run_dir, {
                "status": "running",
                "run_id": run_id,
                "strategy": config.get("strategy"),
                "epochs": config.get("epochs"),
            })
            event_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()
            self._queues[run_id] = event_queue
            self._active_run_id = run_id

        def emit(event: Dict[str, Any]) -> None:
            payload = dict(event.get("payload", {}))
            payload.setdefault("run_id", run_id)
            event_queue.put({"type": event.get("type", "message"), "payload": payload})

        def worker() -> None:
            try:
                self.runner(config, run_dir, on_event=emit)
            except Exception as exc:  # noqa: BLE001 - surface training failures to local UI
                failed = {
                    "status": "failed",
                    "run_id": run_id,
                    "strategy": config.get("strategy"),
                    "error": str(exc),
                }
                write_summary(run_dir, failed)
                event_queue.put({"type": "failed", "payload": failed})
            finally:
                event_queue.put({"type": "stream_end", "payload": {"run_id": run_id}})
                with self._lock:
                    if self._active_run_id == run_id:
                        self._active_run_id = None

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        return {"run_id": run_id, "status": "running"}

    def stream(self, run_id: str):
        event_queue = self._queues.get(run_id)
        if event_queue is None:
            detail = self.get_run(run_id)
            if not detail:
                yield format_sse("failed", {"run_id": run_id, "error": "Run not found"})
                return
            yield format_sse("completed", {"run_id": run_id, **detail.get("summary", {})})
            return

        while True:
            event = event_queue.get()
            event_type = event["type"]
            if event_type == "stream_end":
                break
            yield format_sse(event_type, event["payload"])

    def list_runs(self) -> List[Dict[str, Any]]:
        runs: List[Dict[str, Any]] = []
        for run_dir in sorted(self.runs_root.iterdir(), reverse=True):
            if not run_dir.is_dir():
                continue
            try:
                data = read_run(run_dir)
            except (OSError, json.JSONDecodeError):
                continue
            summary = data.get("summary", {})
            config = data.get("config", {})
            runs.append({
                "run_id": run_dir.name,
                "strategy": summary.get("strategy", config.get("strategy", "unknown")),
                "status": summary.get("status", "running"),
                "epochs": summary.get("epochs", config.get("epochs")),
                "final_grid": summary.get("final_grid"),
                "params": summary.get("params"),
                "val_mse": summary.get("val_mse"),
                "best_val_mse": summary.get("best_val_mse"),
                "test_mse": summary.get("test_mse"),
                "seed": config.get("seed"),
            })
        return runs

    def get_run(self, run_id: str) -> Dict[str, Any]:
        run_dir = self.runs_root / run_id
        if not run_dir.exists() or not run_dir.is_dir():
            return {}
        return read_run(run_dir)
