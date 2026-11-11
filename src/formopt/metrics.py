"""Benchmark metrics: regret traces, spend-to-target, success rate, bootstrap and Wilson CIs.

Everything judges a found formulation on the noise-free truth unless stated otherwise.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from formopt.types import CampaignResult


def simple_regret(best_cost_trace: np.ndarray, opt_cost: float, cost_if_none: float) -> np.ndarray:
    """Best feasible cost minus the reference optimum; `cost_if_none` before any feasible run."""
    trace = np.where(np.isfinite(best_cost_trace), best_cost_trace, cost_if_none)
    return trace - opt_cost


def trace_on_grid(
    spend: np.ndarray, values: np.ndarray, grid: np.ndarray, fill_before: float
) -> np.ndarray:
    """Step function on a spend grid: the last value whose cumulative spend is <= each grid point.

    Before the first experiment finishes, the value is `fill_before`.
    """
    idx = np.searchsorted(np.asarray(spend), np.asarray(grid), side="right") - 1
    out = np.where(idx >= 0, np.asarray(values)[np.clip(idx, 0, None)], fill_before)
    return out.astype(float)


def _first_hit(result: CampaignResult, opt_cost: float, tol: float) -> int | None:
    trace = result.best_feasible_cost_trace(use_truth=True)
    hits = np.flatnonzero(trace <= opt_cost * (1.0 + tol))
    return int(hits[0]) if hits.size else None


def spend_to_target(result: CampaignResult, opt_cost: float, tol: float = 0.05) -> float:
    """Cumulative spend when a truly feasible recipe within `tol` of the optimum is first found.

    Returns inf if never reached within the budget.
    """
    i = _first_hit(result, opt_cost, tol)
    return float(result.cumulative_spend()[i]) if i is not None else float(np.inf)


def experiments_to_target(result: CampaignResult, opt_cost: float, tol: float = 0.05) -> float:
    """Number of experiments until the target is first reached (inf if never)."""
    i = _first_hit(result, opt_cost, tol)
    return float(i + 1) if i is not None else float(np.inf)


def success_rate(results: list[CampaignResult], opt_cost: float, tol: float = 0.05) -> float:
    """Fraction of campaigns that reach the target within budget."""
    return float(np.mean([np.isfinite(spend_to_target(r, opt_cost, tol)) for r in results]))


def bootstrap_ci(
    values: np.ndarray,
    stat: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap: (statistic, lower, upper) for a (1 - alpha) interval."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    boots = np.array([stat(values[row]) for row in idx])
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(stat(values)), float(lo), float(hi)


def paired_difference_ci(a: np.ndarray, b: np.ndarray, **kw) -> tuple[float, float, float]:
    """Bootstrap CI of the mean per-seed difference a - b (seeds must be aligned)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("Paired arrays must have the same shape (same seeds)")
    return bootstrap_ci(a - b, np.mean, **kw)


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score interval for a proportion: (p_hat, lower, upper)."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return float(p), float(max(0.0, centre - half)), float(min(1.0, centre + half))


def interval_coverage(
    y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray, z: float = 1.96
) -> float:
    """Share of observations inside mu +/- z sigma (0.95 is ideal for z = 1.96)."""
    y_true, mu, sigma = (np.asarray(v, dtype=float) for v in (y_true, mu, sigma))
    return float(np.mean(np.abs(y_true - mu) <= z * sigma))
