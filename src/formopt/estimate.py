"""Rough estimate of the remaining lab spend to reach an in-spec result (a heuristic).

Plan: run the top-k candidates in order, stop at the first pass. If run i costs c_i and passes
with probability p_i, the expected spend is

    c_1 + (1 - p_1) c_2 + (1 - p_1)(1 - p_2) c_3 + ...

plus a tail term for "none of the k passed": the leftover probability times the average run
cost divided by the average pass probability (the mean of a geometric distribution). It
assumes runs pass independently, which they do not (if one fails, its neighbours are more
likely to fail), so the app labels it "rough estimate".
"""

from __future__ import annotations

import numpy as np

from formopt.acquisition import probability_of_feasibility
from formopt.gp import FittedGP
from formopt.problems.base import Problem


def expected_spend_formula(
    pof: np.ndarray, run_cost: np.ndarray, tail_pof: float | None = None
) -> float:
    """Expected spend for a planned sequence with pass probabilities `pof` and costs."""
    p = np.clip(np.asarray(pof, dtype=float), 0.0, 1.0)
    c = np.asarray(run_cost, dtype=float)
    if p.size == 0:
        return float("inf")
    reach = np.concatenate([[1.0], np.cumprod(1.0 - p)[:-1]])  # P(still searching before i)
    planned = float(np.sum(c * reach))
    leftover = float(np.prod(1.0 - p))
    tail_p = float(np.mean(p)) if tail_pof is None else float(tail_pof)
    if leftover == 0.0:
        return planned
    if tail_p <= 0.0:
        return float("inf")
    return planned + leftover * float(np.mean(c)) / tail_p


def expected_spend_to_spec(
    problem: Problem,
    gps: dict[str, FittedGP],
    strategy_candidates: np.ndarray,
    k: int = 10,
    scores: np.ndarray | None = None,
) -> float:
    """Expected spend if the lab runs the top-k candidates (by `scores`, or by PoF per dollar).

    Finite whenever any candidate has PoF > 0: if the top-k all have PoF 0, the tail uses the
    best PoF among all candidates.
    """
    preds = {o: gp.predict(strategy_candidates) for o, gp in gps.items()}
    pof = probability_of_feasibility(preds, problem.specs)
    run_cost = problem.run_cost(strategy_candidates)
    order_key = scores if scores is not None else pof / run_cost
    order_key = np.where(np.isfinite(order_key), order_key, -np.inf)
    top = np.argsort(-order_key)[:k]
    tail = float(np.mean(pof[top]))
    if tail <= 0.0:
        tail = float(np.max(pof))
    return expected_spend_formula(pof[top], run_cost[top], tail_pof=tail)
