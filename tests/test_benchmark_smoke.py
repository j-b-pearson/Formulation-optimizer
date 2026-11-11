"""End to end on the toy problem: CSVs, summary, paired table, a figure, MLflow to temp sqlite."""

from __future__ import annotations

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest

from formopt.benchmark import benchmark_problem
from formopt.config import load_config
from formopt.plotting import plot_spend_vs_best
from formopt.strategies import ConstrainedBO, RandomSearch
from tests.conftest import ToyProblem


@pytest.fixture
def small_cfg():
    cfg = load_config()
    ref = cfg.benchmark.reference.model_copy(update={"n_samples": 5000, "polish_max_moves": 5})
    bench = cfg.benchmark.model_copy(update={"reference": ref, "n_boot": 200})
    return cfg.model_copy(update={"benchmark": bench})


def _factory(method: str):
    if method == "random":
        return RandomSearch(batch=200)
    return ConstrainedBO("bo_cost", "fixed", n_init=4, n_candidates=200)


def test_benchmark_smoke(small_cfg, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setenv("FORMOPT_MLFLOW_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    problem = ToyProblem()
    out = benchmark_problem(
        small_cfg,
        problem,
        budget=15.0,
        methods=["random", "bo_cost"],
        seeds=[0, 1],
        out_dir=tmp_path,
        strategy_factory=_factory,
        log_mlflow=True,
    )
    long = pd.read_csv(tmp_path / "benchmark_toy.csv")
    expected = {"problem", "method", "seed", "step", "x0", "x1", "g", "true_g", "run_cost"}
    expected |= {"cumulative_spend", "material_cost", "passed_spec", "truly_feasible"}
    expected |= {"pred_mu_g", "pred_sd_g"}
    assert expected <= set(long.columns)
    assert long.groupby(["method", "seed"])["cumulative_spend"].max().max() <= 15.0

    metrics = pd.read_csv(tmp_path / "campaign_metrics_toy.csv")
    assert len(metrics) == 4
    assert {"final_regret", "spend_to_target", "success", "lab_success"} <= set(metrics.columns)
    assert (metrics["final_regret"] >= -1e-9).all()  # nobody beats the reference

    summary = pd.read_csv(tmp_path / "summary_toy.csv")
    assert list(summary["method"]) == ["random", "bo_cost"]
    assert summary["success_ci_lo"].le(summary["success_rate"]).all()
    assert summary["mean_regret_ci_lo"].le(summary["mean_final_regret"] + 1e-12).all()

    paired = pd.read_csv(tmp_path / "paired_toy.csv")
    assert set(paired["metric"]) == {"final_regret", "spend_to_target_censored", "success"}
    assert (paired["n_pairs"] == 2).all()
    assert (tmp_path / "benchmark_info_toy.json").exists()
    assert set(out) == {"long", "metrics", "summary", "paired"}

    fig = plot_spend_vs_best(
        long, 0.429, problem.cost_cap(), 15.0, tmp_path / "spend_vs_best_toy.png", "toy"
    )
    assert fig is not None and fig.exists() and fig.stat().st_size > 1000

    mlflow.set_tracking_uri(uri)
    exp = mlflow.get_experiment_by_name("benchmark-toy")
    runs = mlflow.search_runs([exp.experiment_id])
    parents = runs[runs["tags.mlflow.parentRunId"].isna()]
    children = runs[runs["tags.mlflow.parentRunId"].notna()]
    assert len(parents) == 2 and len(children) == 4
    assert np.isfinite(children["metrics.final_regret"]).all()
