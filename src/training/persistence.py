from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def create_run_dir(root: str | Path, run_id: str) -> Path:
    run_dir = Path(root) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_config(run_dir: str | Path, config: Dict[str, Any]) -> None:
    _write_json(Path(run_dir) / "config.json", config)


def append_metric(run_dir: str | Path, metric: Dict[str, Any]) -> None:
    path = Path(run_dir) / "metrics.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(metric, ensure_ascii=False) + "\n")


def write_events(run_dir: str | Path, events: Iterable[Dict[str, Any]]) -> None:
    _write_json(Path(run_dir) / "events.json", list(events))


def write_summary(run_dir: str | Path, summary: Dict[str, Any]) -> None:
    _write_json(Path(run_dir) / "summary.json", summary)


def write_fit_data(run_dir: str | Path, fit_data: Dict[str, Any]) -> None:
    _write_json(Path(run_dir) / "fit_data.json", fit_data)


def read_metrics(run_dir: str | Path) -> List[Dict[str, Any]]:
    path = Path(run_dir) / "metrics.jsonl"
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_run(run_dir: str | Path) -> Dict[str, Any]:
    path = Path(run_dir)
    return {
        "config": _read_json(path / "config.json", {}),
        "summary": _read_json(path / "summary.json", {}),
        "metrics": read_metrics(path),
        "events": _read_json(path / "events.json", []),
        "fit_data": _read_json(path / "fit_data.json", {}),
    }
