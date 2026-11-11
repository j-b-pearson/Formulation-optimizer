"""Acquisition functions: turn GP predictions into one score per candidate experiment.

All functions are vectorized over candidates and use the "maximize" convention (higher score
means more worth running). Costs are handled by negation or by explicit improvement terms.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from formopt.types import Spec

SIGMA_FLOOR = 1e-12


def probability_of_improvement(
    mu: np.ndarray, sigma: np.ndarray, best: float, xi: float = 0.0
) -> np.ndarray:
    """PI = Phi((mu - best - xi) / sigma): the chance of beating the best so far."""
    sigma = np.maximum(np.asarray(sigma, dtype=float), SIGMA_FLOOR)
    return norm.cdf((np.asarray(mu, dtype=float) - best - xi) / sigma)


def expected_improvement(
    mu: np.ndarray, sigma: np.ndarray, best: float, xi: float = 0.0
) -> np.ndarray:
    """EI = (mu - best - xi) Phi(z) + sigma phi(z), z = (mu - best - xi) / sigma.

    The first term rewards a good mean (exploitation), the second rewards uncertainty
    (exploration). Sigma is floored so EI tends to max(mu - best, 0) as sigma goes to 0.
    """
    sigma = np.maximum(np.asarray(sigma, dtype=float), SIGMA_FLOOR)
    improvement = np.asarray(mu, dtype=float) - best - xi
    z = improvement / sigma
    return np.maximum(improvement * norm.cdf(z) + sigma * norm.pdf(z), 0.0)


def upper_confidence_bound(mu: np.ndarray, sigma: np.ndarray, beta: float = 2.0) -> np.ndarray:
    """UCB = mu + beta sigma: optimism in the face of uncertainty."""
    return np.asarray(mu, dtype=float) + beta * np.asarray(sigma, dtype=float)


def spec_pass_probability(
    mu: np.ndarray, sigma: np.ndarray, lower: float | None, upper: float | None
) -> np.ndarray:
    """P(lower <= f <= upper) for f ~ Normal(mu, sigma^2). Limits must be in GP space."""
    mu = np.asarray(mu, dtype=float)
    sigma = np.maximum(np.asarray(sigma, dtype=float), SIGMA_FLOOR)
    p_upper = norm.cdf((upper - mu) / sigma) if upper is not None else np.ones_like(mu)
    p_lower = norm.cdf((lower - mu) / sigma) if lower is not None else np.zeros_like(mu)
    return np.clip(p_upper - p_lower, 0.0, 1.0)


def probability_of_feasibility_per_spec(
    preds: dict[str, tuple[np.ndarray, np.ndarray]], specs: list[Spec]
) -> dict[str, np.ndarray]:
    """Pass probability for each spec separately (limits logged when transform is log)."""
    out = {}
    for spec in specs:
        mu, sigma = preds[spec.output]
        lo, hi = spec.transformed_limits()
        out[spec.output] = spec_pass_probability(mu, sigma, lo, hi)
    return out


def probability_of_feasibility(
    preds: dict[str, tuple[np.ndarray, np.ndarray]], specs: list[Spec]
) -> np.ndarray:
    """PoF = product over specs of P(pass spec), assuming independent GPs per output."""
    per_spec = probability_of_feasibility_per_spec(preds, specs)
    return np.prod(np.vstack(list(per_spec.values())), axis=0)


def known_cost_improvement(cost: np.ndarray, best_feasible_cost: float | None) -> np.ndarray:
    """max(best - cost, 0); ones when nothing has passed yet (any pass is an improvement).

    Material cost is a known formula, so this improvement has no uncertainty.
    """
    cost = np.asarray(cost, dtype=float)
    if best_feasible_cost is None:
        return np.ones_like(cost)
    return np.maximum(best_feasible_cost - cost, 0.0)


def constrained_cost_acquisition(
    cost: np.ndarray, pof: np.ndarray, best_feasible_cost: float | None
) -> np.ndarray:
    """Spec-mode acquisition: improvement in known cost times PoF; just PoF before any pass.

    This is what constrained expected improvement reduces to when the objective is known
    exactly: the only uncertainty is whether the candidate passes spec.
    """
    return known_cost_improvement(cost, best_feasible_cost) * np.asarray(pof, dtype=float)


def cost_aware(acq: np.ndarray, run_cost: np.ndarray, alpha: float) -> np.ndarray:
    """Acquisition per unit run cost: acq / run_cost ** alpha (alpha = 0 is plain BO)."""
    return np.asarray(acq, dtype=float) / np.asarray(run_cost, dtype=float) ** alpha


def cooled_alpha(spent: float, budget: float, schedule: str = "linear") -> float:
    """CArBO-style cost cooling: alpha = 1 at the start, 0 when the budget is spent."""
    if budget <= 0:
        return 0.0
    frac_left = min(max((budget - spent) / budget, 0.0), 1.0)
    if schedule == "linear":
        return float(frac_left)
    raise ValueError(f"Unknown cooling schedule {schedule!r}")
