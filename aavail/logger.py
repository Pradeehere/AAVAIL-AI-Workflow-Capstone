from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, default=str) + "\n")


def log_train(log_path, country, date_range, metrics, runtime_s, model_version, note=""):
    _append_jsonl(log_path, {
        "event": "train",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "country": country,
        "date_range": list(date_range),
        "metrics": metrics,
        "runtime_seconds": round(float(runtime_s), 4),
        "model_version": str(model_version),
        "note": note,
    })


def log_predict(log_path, country, target_date, prediction, runtime_s, model_version):
    _append_jsonl(log_path, {
        "event": "predict",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "country": country,
        "target_date": str(target_date),
        "prediction": float(prediction),
        "runtime_seconds": round(float(runtime_s), 4),
        "model_version": str(model_version),
    })


def read_logs(log_path) -> list[dict[str, Any]]:
    path = Path(log_path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
