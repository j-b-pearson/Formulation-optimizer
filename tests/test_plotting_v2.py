"""v2 figures render from small synthetic tables (offline; checks files are written)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from formopt.plotting_v2 import (
    plot_heterogeneity_advantage,
    plot_spend_vs_best_by_level,
    plot_success_vs_budget,
    plot_success_vs_tolerance,
)


def test_heterogeneity_figures(tmp_path: Path) -> None:
    rows = []
    for s in (1.0, 3.0, 10.0):
        for method in ("bo_cost_cooled", "bo_cost"):
            for metric in ("spend_saved", "success_gain"):
                rows.append(
                    {
                        "s": s,
                        "method": method,
                        "metric": metric,
                        "mean": s,
                        "ci_lo": s - 1,
                        "ci_hi": s + 1,
                    }
                )
    levels = pd.DataFrame(rows)
    v1 = levels[levels.s == 3.0].drop(columns="s")
    out = plot_heterogeneity_advantage(levels, tmp_path / "adv.png", v1, 6.35)
    assert out.exists() and out.stat().st_size > 10_000

    long_rows = []
    for seed in range(3):
        spend = 0.0
        for step in range(6):
            spend += 100.0
            long_rows.append(
                {
                    "method": "bo_plain",
                    "seed": seed,
                    "step": step,
                    "cumulative_spend": spend,
                    "material_cost": 3.5 - 0.1 * step,
                    "truly_feasible": step % 2 == 1,
                }
            )
    longs = {s: pd.DataFrame(long_rows) for s in (1.0, 3.0, 10.0)}
    out = plot_spend_vs_best_by_level(longs, 2.9, 7.7, 600.0, tmp_path / "by_level.png")
    assert out.exists()


def test_concrete_curve_figures(tmp_path: Path) -> None:
    rows = []
    for method in ("random", "bo_plain"):
        for tol in (0.02, 0.05, 0.10, 0.15):
            for k in range(1, 30):
                p = min(1.0, tol * k / 3)
                rows.append(
                    {
                        "method": method,
                        "tolerance": tol,
                        "n_experiments": k,
                        "success_rate": p,
                        "ci_lo": max(0.0, p - 0.1),
                        "ci_hi": min(1.0, p + 0.1),
                    }
                )
    curves = pd.DataFrame(rows)
    assert plot_success_vs_tolerance(curves, tmp_path / "tol.png").exists()
    assert plot_success_vs_budget(curves, tmp_path / "budget.png").exists()
    assert np.isfinite(curves["success_rate"]).all()
