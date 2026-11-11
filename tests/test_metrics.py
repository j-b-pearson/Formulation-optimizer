"""Metrics: regret, grid interpolation, spend-to-target with inf, bootstrap and Wilson CIs."""

from __future__ import annotations

import numpy as np
import pytest

from formopt.metrics import (
    bootstrap_ci,
    experiments_to_target,
    interval_coverage,
    paired_difference_ci,
    simple_regret,
    spend_to_target,
    success_rate,
    trace_on_grid,
    wilson_ci,
)
from formopt.types import CampaignResult, Observation


def _result(costs: list[float], feasible: list[bool], run_costs: list[float]) -> CampaignResult:
    obs = [
        Observation(
            x=np.zeros(2),
            outputs={},
            run_cost=rc,
            material_cost=c,
            passed_spec=f,
            truly_feasible=f,
        )
        for c, f, rc in zip(costs, feasible, run_costs, strict=True)
    ]
    return CampaignResult("toy", "m", 0, 1000.0, obs)


def test_best_feasible_trace_and_regret_by_hand() -> None:
    r = _result([5.0, 3.0, 4.0, 2.2], [False, True, True, True], [100, 100, 200, 100])
    trace = r.best_feasible_cost_trace(cost_if_none=9.0)
    np.testing.assert_allclose(trace, [9.0, 3.0, 3.0, 2.2])
    np.testing.assert_allclose(simple_regret(trace, 2.0, 9.0), [7.0, 1.0, 1.0, 0.2])
    np.testing.assert_allclose(r.cumulative_spend(), [100, 200, 400, 500])


def test_regret_replaces_inf_with_cap() -> None:
    np.testing.assert_allclose(simple_regret(np.array([np.inf, 3.0]), 2.0, 8.0), [6.0, 1.0])


def test_trace_on_grid_is_a_step_function() -> None:
    spend = np.array([100.0, 200.0, 400.0])
    values = np.array([9.0, 3.0, 2.5])
    grid = np.array([0.0, 99.0, 100.0, 250.0, 400.0, 1000.0])
    np.testing.assert_allclose(trace_on_grid(spend, values, grid, 10.0), [10, 10, 9, 3, 2.5, 2.5])


def test_spend_and_experiments_to_target() -> None:
    r = _result([5.0, 3.0, 2.05, 2.0], [True, True, True, True], [100, 100, 200, 100])
    assert spend_to_target(r, 2.0, tol=0.05) == 400.0  # 2.05 <= 2.1 at the 3rd experiment
    assert experiments_to_target(r, 2.0, tol=0.05) == 3.0
    never = _result([5.0, 3.0], [True, True], [100, 100])
    assert spend_to_target(never, 2.0) == np.inf
    assert experiments_to_target(never, 2.0) == np.inf
    infeasible_cheap = _result([1.0], [False], [100])
    assert spend_to_target(infeasible_cheap, 2.0) == np.inf  # cheap but truly infeasible
    assert success_rate([r, never, infeasible_cheap], 2.0) == pytest.approx(1 / 3)


def test_bootstrap_ci_covers_true_mean_most_of_the_time() -> None:
    rng = np.random.default_rng(0)
    covered = 0
    for trial in range(200):
        sample = rng.normal(loc=5.0, scale=2.0, size=30)
        _, lo, hi = bootstrap_ci(sample, np.mean, n_boot=500, seed=trial)
        covered += lo <= 5.0 <= hi
    assert 0.88 <= covered / 200 <= 0.99


def test_bootstrap_median_and_paired_difference() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    med, lo, hi = bootstrap_ci(values, np.median, n_boot=500)
    assert med == 3.0 and lo <= med <= hi
    a = np.array([1.0, 2.0, 3.0, 4.0])
    mean, lo, hi = paired_difference_ci(a, a + 1.0, n_boot=200)
    assert mean == pytest.approx(-1.0) and lo == pytest.approx(-1.0) and hi == pytest.approx(-1.0)
    with pytest.raises(ValueError):
        paired_difference_ci(a, a[:3])


def test_wilson_interval() -> None:
    p, lo, hi = wilson_ci(0, 30)
    assert p == 0.0 and lo == 0.0 and 0.10 < hi < 0.12
    p, lo, hi = wilson_ci(15, 30)
    assert p == 0.5 and lo == pytest.approx(1 - hi)


def test_interval_coverage() -> None:
    rng = np.random.default_rng(1)
    y = rng.normal(size=20000)
    assert interval_coverage(y, np.zeros_like(y), np.ones_like(y)) == pytest.approx(0.95, abs=0.01)
