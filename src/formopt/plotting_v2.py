"""v2 figures: the run-cost heterogeneity sweep and the concrete success curves.

Same colors and labels per method as the v1 figures (`formopt.plotting`).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from formopt.plotting import (  # noqa: E402
    METHOD_COLORS,
    METHOD_LABELS,
    UNITS,
    _methods_in,
    _save,
    best_cost_matrix,
    bootstrap_band,
    identical_methods,
)

PANELS = {
    "spend_saved": "Spend to target saved ($)\n(plain BO minus cost-aware, censored at budget)",
    "success_gain": "Success gain (share of seeds)\n(cost-aware minus plain BO)",
}


def plot_heterogeneity_advantage(
    levels: pd.DataFrame,
    out: Path | None,
    v1_point: pd.DataFrame | None = None,
    v1_s: float | None = None,
    headline: str = "bo_cost_cooled",
    subtitle: str = "",
) -> Path | None:
    """Headline v2 figure: cost-aware minus plain BO advantage against run-cost spread s.

    One panel per co-primary metric; mean paired advantage over seeds with its 95 % bootstrap
    CI. The optional v1-settings point (v1 run cost, same seeds) is drawn hollow at the v1
    effective spread: descriptive only, it is not part of the H1 test.
    """
    methods = list(dict.fromkeys(levels["method"]))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for ax, (metric, ylabel) in zip(axes, PANELS.items(), strict=True):
        for j, method in enumerate(methods):
            g = levels[(levels.method == method) & (levels.metric == metric)].sort_values("s")
            x = g["s"].to_numpy(float) * (1.0 + 0.04 * (j - (len(methods) - 1) / 2))
            y = g["mean"].to_numpy(float)
            err = np.vstack([y - g["ci_lo"].to_numpy(float), g["ci_hi"].to_numpy(float) - y])
            label = METHOD_LABELS.get(method, method) + (
                " (H1 headline)" if method == headline else ""
            )
            ax.errorbar(
                x,
                y,
                yerr=err,
                color=METHOD_COLORS.get(method),
                marker="o",
                lw=2 if method == headline else 1.2,
                capsize=4,
                label=label,
            )
            if v1_point is not None and v1_s is not None:
                r = v1_point[(v1_point.method == method) & (v1_point.metric == metric)]
                if len(r):
                    r = r.iloc[0]
                    xv = v1_s * (1.0 + 0.04 * (j - (len(methods) - 1) / 2))
                    ax.errorbar(
                        [xv],
                        [r["mean"]],
                        yerr=[[r["mean"] - r["ci_lo"]], [r["ci_hi"] - r["mean"]]],
                        color=METHOD_COLORS.get(method),
                        marker="o",
                        mfc="white",
                        ls="none",
                        capsize=3,
                        alpha=0.8,
                    )
        ax.axhline(0.0, color="black", lw=0.8)
        ax.set_xscale("log")
        ticks = sorted(set(levels["s"].astype(float)) | ({v1_s} if v1_s else set()))
        ax.set_xticks(ticks, [f"{t:.3g}" for t in ticks])
        ax.minorticks_off()
        ax.set_xlabel("Run-cost spread s (most / least expensive possible run)")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="best")
    note = "Filled: pre-registered sweep levels (95 % bootstrap CI). Positive = cost-aware better."
    if v1_s is not None:
        note += (
            f"\nHollow: v1 run cost (s = {v1_s:.2f}), same seeds; descriptive, not in the H1 test."
        )
    fig.text(0.01, -0.02, note, fontsize=7.5, color="#444444", va="top")
    title = "Enamel (synthetic): cost-aware advantage over plain constrained BO vs run-cost spread"
    fig.suptitle(title + (f"\n{subtitle}" if subtitle else ""), fontsize=10)
    fig.tight_layout()
    return _save(fig, out)


def plot_spend_vs_best_by_level(
    longs: dict[float, pd.DataFrame],
    opt_cost: float,
    cost_cap: float,
    budget: float,
    out: Path | None,
    tol: float = 0.05,
    grid_points: int = 100,
    subtitle: str = "",
) -> Path | None:
    """Spend vs best truly feasible cost found, one panel per spread level s."""
    grid = np.linspace(0.0, budget, grid_points)
    levels = sorted(longs)
    fig, axes = plt.subplots(1, len(levels), figsize=(5.0 * len(levels), 4.4), sharey=True)
    axes = np.atleast_1d(axes)
    top = opt_cost * 1.6
    for ax, s in zip(axes, levels, strict=True):
        long = longs[s]
        matrices = {m: best_cost_matrix(long, m, cost_cap, grid) for m in _methods_in(long)}
        for method, matrix in matrices.items():
            mean, lo, hi = bootstrap_band(matrix)
            color = METHOD_COLORS.get(method)
            ax.plot(grid, mean, color=color, lw=1.8, label=METHOD_LABELS.get(method, method))
            ax.fill_between(grid, lo, hi, color=color, alpha=0.12, lw=0)
        same = identical_methods(matrices)
        if same:
            ax.text(
                0.02,
                0.02,
                "Identical traces (lines overlap):\n" + "\n".join(" = ".join(g) for g in same),
                transform=ax.transAxes,
                fontsize=7,
                color="#555555",
            )
        ax.axhline(opt_cost, color="black", ls="--", lw=1)
        ax.axhspan(opt_cost, opt_cost * (1 + tol), color="black", alpha=0.08)
        ax.set_xlim(0, budget)
        ax.set_ylim(opt_cost * 0.97, top)
        ax.set_title(f"s = {s:g}", fontsize=10)
        ax.set_xlabel("Cumulative lab spend ($)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(f"Best truly feasible material cost found ({UNITS['enamel']})")
    axes[0].legend(fontsize=7.5, loc="upper right")
    fig.suptitle(
        "Enamel (synthetic): spend vs best in-spec cost at each run-cost spread "
        "(mean, 95 % bootstrap band; dashed = reference optimum, grey = within 5 %)"
        + (f"\n{subtitle}" if subtitle else ""),
        fontsize=10,
    )
    fig.tight_layout()
    return _save(fig, out)


def plot_success_vs_tolerance(curves: pd.DataFrame, out: Path | None, subtitle: str = "") -> Path:
    """Success rate at the full budget against tolerance from the reference optimum."""
    full = curves[curves["n_experiments"] == curves["n_experiments"].max()]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    methods = [m for m in METHOD_COLORS if m in set(full["method"])]
    for j, method in enumerate(methods):
        g = full[full.method == method].sort_values("tolerance")
        x = 100 * g["tolerance"].to_numpy(float) + 0.25 * (j - (len(methods) - 1) / 2)
        y = g["success_rate"].to_numpy(float)
        err = np.vstack([y - g["ci_lo"], g["ci_hi"] - y])
        ax.errorbar(
            x,
            y,
            yerr=err,
            color=METHOD_COLORS[method],
            marker="o",
            capsize=3,
            label=METHOD_LABELS[method],
        )
    ax.set_xlabel("Tolerance above the reference optimum cost (%)")
    ax.set_ylabel("Share of seeds with a truly in-spec mix\nwithin the tolerance (95 % Wilson CI)")
    ax.set_xticks(sorted(100 * full["tolerance"].unique()))
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title(
        "Concrete (emulator): success vs tolerance at the full budget (DESCRIPTIVE)"
        + (f"\n{subtitle}" if subtitle else ""),
        fontsize=10,
    )
    return _save(fig, out)


def plot_success_vs_budget(curves: pd.DataFrame, out: Path | None, subtitle: str = "") -> Path:
    """Success rate against the number of experiments, one panel per tolerance."""
    tols = sorted(curves["tolerance"].unique())
    fig, axes = plt.subplots(1, len(tols), figsize=(4.0 * len(tols), 4.0), sharey=True)
    axes = np.atleast_1d(axes)
    methods = [m for m in METHOD_COLORS if m in set(curves["method"])]
    for ax, tol in zip(axes, tols, strict=True):
        for method in methods:
            g = curves[(curves.method == method) & (curves.tolerance == tol)]
            g = g.sort_values("n_experiments")
            ax.plot(
                g["n_experiments"],
                g["success_rate"],
                color=METHOD_COLORS[method],
                lw=1.8,
                label=METHOD_LABELS[method],
            )
        ax.set_title(f"within {100 * tol:g} % of the optimum", fontsize=9)
        ax.set_xlabel("Experiment budget (number of trial batches)")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Share of seeds that reached it")
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle(
        "Concrete (emulator): success vs experiment budget at each tolerance (DESCRIPTIVE)"
        + (f"\n{subtitle}" if subtitle else ""),
        fontsize=10,
    )
    fig.tight_layout()
    return _save(fig, out)
