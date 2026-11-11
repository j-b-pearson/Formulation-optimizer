"""MLflow tracking helpers (SQLite backend by default: `sqlite:///mlflow.db`).

Benchmarks create one parent run per (problem, method) with a child run per seed; emulator
training is its own run. Launch the UI with:
    uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

# MLflow prints an "agent hint" on import; it is an info banner, not a warning.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow  # noqa: E402

from formopt.config import Settings  # noqa: E402

DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"


def tracking_uri() -> str:
    """MLFLOW_TRACKING_URI from the environment or `.env`, else the SQLite default."""
    return os.environ.get("MLFLOW_TRACKING_URI") or Settings().mlflow_tracking_uri


def setup_mlflow(experiment: str, uri: str | None = None) -> str:
    """Point MLflow at the tracking store and select (or create) an experiment.

    If `FORMOPT_MLFLOW_ARTIFACT_ROOT` is set, new experiments store artifacts there (tests use
    it to keep artifacts in a temp folder); otherwise MLflow's default location is used.
    """
    uri = uri or tracking_uri()
    mlflow.set_tracking_uri(uri)
    root = os.environ.get("FORMOPT_MLFLOW_ARTIFACT_ROOT")
    if root and mlflow.get_experiment_by_name(experiment) is None:
        location = (Path(root) / experiment).resolve().as_uri()
        mlflow.create_experiment(experiment, artifact_location=location)
    mlflow.set_experiment(experiment)
    return uri


def flatten(data: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested dicts into dotted keys, for logging as MLflow params."""
    out: dict[str, Any] = {}
    for key, value in data.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            out.update(flatten(value, f"{name}."))
        else:
            out[name] = value
    return out


def log_params_safe(params: Mapping[str, Any]) -> None:
    """Log params, truncating long values (MLflow limits a param value to 6,000 chars)."""
    mlflow.log_params({k: str(v)[:500] for k, v in params.items()})


def log_metrics_finite(metrics: Mapping[str, float]) -> None:
    """Log only finite numeric metrics (MLflow rejects inf and NaN)."""
    clean = {}
    for key, value in metrics.items():
        try:
            v = float(value)
        except (TypeError, ValueError):
            continue
        if v == v and abs(v) != float("inf"):
            clean[key] = v
    if clean:
        mlflow.log_metrics(clean)


def log_artifact_if_exists(path: Path, artifact_path: str | None = None) -> None:
    if Path(path).exists():
        mlflow.log_artifact(str(path), artifact_path=artifact_path)


def log_campaign(run_name: str, params: Mapping[str, Any], metrics: Mapping[str, float]) -> None:
    """Log one campaign (one seed) as a nested child run of the active parent run."""
    with mlflow.start_run(run_name=run_name, nested=True):
        log_params_safe(params)
        log_metrics_finite(metrics)
