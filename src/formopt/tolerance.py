"""Concrete success rate against tolerance and experiment budget (descriptive).

"Success at tolerance t within k experiments" means: among the first k experiments of a
campaign, at least one recipe is truly in spec (noise-free emulator) and costs at most
reference optimum x (1 + t). v1 used t = 5 % and the full budget only. These curves make no
new claims; they show how the v1 conclusion depends on the 5 % line.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from formopt.metrics import wilson_ci


def best_cost_by_experiment(long: pd.DataFrame, method: str, n_max: int) -> np.ndarray:
    """(n_seeds, n_max) best truly feasible material cost after 1..n_max experiments.

    inf before the first truly feasible recipe. A campaign that ran out of budget before
    n_max experiments keeps its final value (it could not run more).
    """
    rows = []
    for _, g in long[long["method"] == method].groupby("seed"):
        g = g.sort_values("step")
        cost = np.where(g["truly_feasible"].astype(bool), g["material_cost"], np.inf)
        best = np.minimum.accumulate(cost.astype(float))
        padded = np.full(n_max, best[-1] if len(best) else np.inf)
        padded[: len(best)] = best[:n_max]
        rows.append(padded)
    return np.vstack(rows)


def success_curves(
    long: pd.DataFrame, ref_cost: float, tolerances: list[float], source: str = ""
) -> pd.DataFrame:
    """Tidy table: success share (Wilson 95 % CI) per method, tolerance and experiment budget."""
    n_max = int(long.groupby(["method", "seed"]).size().max())
    rows = []
    for method in dict.fromkeys(long["method"]):
        best = best_cost_by_experiment(long, method, n_max)
        for tol in tolerances:
            hit = best <= ref_cost * (1.0 + tol)
            for k in range(1, n_max + 1):
                n_succ, n = int(hit[:, k - 1].sum()), int(hit.shape[0])
                p, lo, hi = wilson_ci(n_succ, n)
                rows.append(
                    {
                        "source": source,
                        "method": method,
                        "tolerance": float(tol),
                        "n_experiments": k,
                        "n_seeds": n,
                        "n_success": n_succ,
                        "success_rate": p,
                        "ci_lo": lo,
                        "ci_hi": hi,
                    }
                )
    return pd.DataFrame(rows)


def gap_summary(long: pd.DataFrame, ref_cost: float) -> pd.DataFrame:
    """Per method: how far the best in-spec mix of each campaign sits above the optimum."""
    rows = []
    for method in dict.fromkeys(long["method"]):
        best = best_cost_by_experiment(long, method, 10_000)[:, -1]
        gap = 100.0 * (best / ref_cost - 1.0)
        rows.append(
            {
                "method": method,
                "n_seeds": int(best.size),
                "share_found_any_in_spec": float(np.mean(np.isfinite(best))),
                "min_gap_pct": float(np.min(gap)),
                "median_gap_pct": float(np.median(gap)),
                "max_gap_pct": float(np.max(gap)),
            }
        )
    return pd.DataFrame(rows)
