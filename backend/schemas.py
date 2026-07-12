from __future__ import annotations

from typing import Any, Dict

from src.training.runner import normalize_config


def parse_run_config(payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return normalize_config(payload)
    except (TypeError, ValueError) as exc:
        raise ValueError(str(exc)) from exc
