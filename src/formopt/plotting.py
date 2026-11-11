"""Figures for the README and app. One fixed, colorblind-safe color per method everywhere."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from formopt.metrics import trace_on_grid  # noqa: E402

# Okabe-Ito palette (colorblind safe), fixed per method across all figures.
METHOD_COLORS = {
    "random": "#999999",
    "lhs": "#E69F00",
    "factorial": "#56B4E9",
    "bo_plain": "#009E73",
    "bo_cost": "#D55E00",
    "bo_cost_cooled": "#CC79A7",
}
METHOD_LABELS = {
    "random": "Random search",
    "lhs": "Latin hypercube (DOE)",
    "factorial": "Fractional factorial (DOE)",
    "bo_plain": "Constrained BO (cost-unaware)",
    "bo_cost": "Cost-aware BO (alpha = 1)",
    "bo_cost_cooled": "Cost-aware BO (cooled alpha)",
}
UNITS = {"enamel": "$/kg", "concrete": "$/m3"}


def _save(fig: Figure, out: Path | None) -> Path | None:
    if out is None:
        return None
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_emulator_parity(
    y_true: np.ndarray, y_pred: np.ndarray, out: Path, r2: float, rmse: float
) -> Path:
    """Grouped-CV predicted versus observed strength (out-of-fold predictions)."""
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.scatter(y_true, y_pred, s=10, alpha=0.5, color="#0072B2", edgecolor="none")
    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    ax.plot([lo, hi], [lo, hi], color="black", lw=1, ls="--", label="y = x")
    ax.set_xlabel("Observed compressive strength (MPa)")
    ax.set_ylabel("Grouped-CV predicted strength (MPa)")
    ax.set_title(f"Concrete emulator, grouped 5-fold CV\nR$^2$ = {r2:.3f}, RMSE = {rmse:.2f} MPa")
    ax.legend(loc="upper left")
    ax.set_aspect("equal")
    return _save(fig, out)


# ----- benchmark traces ---------------------------------------------------------------------
def best_cost_matrix(
    long: pd.DataFrame, method: str, cost_cap: float, grid: np.ndarray
) -> np.ndarray:
    """(n_seeds, len(grid)) best truly feasible cost on a spend grid for one method.

    Before a campaign finds its first truly feasible recipe the value is `cost_cap`.
    """
    rows = []
    for _, g in long[long["method"] == method].groupby("seed"):
        g = g.sort_values("step")
        costs = np.where(g["truly_feasible"].to_numpy(bool), g["material_cost"], cost_cap)
        best = np.minimum.accumulate(costs)
        rows.append(trace_on_grid(g["cumulative_spend"].to_numpy(float), best, grid, cost_cap))
    return np.vstack(rows)


def bootstrap_band(
    matrix: np.ndarray, stat=np.mean, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Statistic over seeds (rows) at every grid point, with a percentile bootstrap band."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, matrix.shape[0], size=(n_boot, matrix.shape[0]))
    boots = np.stack([stat(matrix[row], axis=0) for row in idx])
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2], axis=0)
    return stat(matrix, axis=0), lo, hi


def _methods_in(long: pd.DataFrame) -> list[str]:
    present = list(dict.fromkeys(long["method"]))
    ordered = [m for m in METHOD_COLORS if m in present]
    return ordered + [m for m in present if m not in ordered]


def plot_spend_vs_best(
    long: pd.DataFrame,
    opt_cost: float,
    cost_cap: float,
    budget: float,
    out: Path | None,
    problem: str = "enamel",
    tol: float = 0.05,
    grid_points: int = 100,
    subtitle: str = "",
) -> Path | None:
    """Headline figure: cumulative lab spend vs best truly feasible material cost found."""
    grid = np.linspace(0.0, budget, grid_points)
    unit = UNITS.get(problem, "$/unit")
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    late = []
    matrices = {m: best_cost_matrix(long, m, cost_cap, grid) for m in _methods_in(long)}
    for method, matrix in matrices.items():
        mean, lo, hi = bootstrap_band(matrix)
        color = METHOD_COLORS.get(method, None)
        ax.plot(grid, mean, color=color, lw=2, label=METHOD_LABELS.get(method, method))
        ax.fill_between(grid, lo, hi, color=color, alpha=0.15, lw=0)
        late.append(float(np.max(hi[grid >= 0.3 * budget])))
    ax.axhline(opt_cost, color="black", ls="--", lw=1, label="Reference optimum")
    ax.axhspan(opt_cost, opt_cost * (1 + tol), color="black", alpha=0.08, label="Within 5 %")
    top = min(cost_cap, max(late) * 1.05)
    ax.set_ylim(opt_cost * 0.97, top)
    ax.set_xlim(0, budget)
    ax.set_xlabel("Cumulative lab spend ($)")
    ax.set_ylabel(f"Best truly feasible material cost found ({unit})")
    title = f"{problem.capitalize()}: lab spend vs best in-spec cost (mean, 95 % bootstrap band)"
    ax.set_title(title + (f"\n{subtitle}" if subtitle else ""), fontsize=10)
    ax.text(
        0.99,
        0.98,
        f"Curves start at the cost cap ({cost_cap:.2f} {unit}) until a feasible recipe is found",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7,
        color="#555555",
    )
    same = identical_methods(matrices)
    if same:
        ax.text(
            0.01,
            0.02,
            "Identical traces (lines overlap): " + "; ".join(" = ".join(g) for g in same),
            transform=ax.transAxes,
            fontsize=7,
            color="#555555",
        )
    ax.legend(fontsize=8, loc="upper right", bbox_to_anchor=(1.0, 0.93))
    ax.grid(alpha=0.3)
    return _save(fig, out)


def identical_methods(matrices: dict[str, np.ndarray]) -> list[list[str]]:
    """Groups of methods whose per-seed traces are exactly equal (so their lines overlap)."""
    groups: list[list[str]] = []
    for method, matrix in matrices.items():
        for group in groups:
            ref = matrices[group[0]]
            if ref.shape == matrix.shape and np.array_equal(ref, matrix):
                group.append(method)
                break
        else:
            groups.append([method])
    return [g for g in groups if len(g) > 1]


def plot_success_vs_spend(
    long: pd.DataFrame,
    opt_cost: float,
    cost_cap: float,
    budget: float,
    out: Path | None,
    problem: str = "enamel",
    tol: float = 0.05,
    grid_points: int = 100,
    subtitle: str = "",
) -> Path | None:
    """Fraction of seeds that have hit the target (within 5 % of optimum) by each spend level."""
    grid = np.linspace(0.0, budget, grid_points)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for method in _methods_in(long):
        hit = best_cost_matrix(long, method, cost_cap, grid) <= opt_cost * (1 + tol)
        ax.plot(
            grid,
            hit.mean(axis=0),
            color=METHOD_COLORS.get(method),
            lw=2,
            label=METHOD_LABELS.get(method, method),
        )
    ax.set_xlim(0, budget)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Cumulative lab spend ($)")
    ax.set_ylabel("Fraction of seeds within 5 % of optimum")
    ax.set_title(
        f"{problem.capitalize()}: success vs spend" + (f"\n{subtitle}" if subtitle else ""),
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    return _save(fig, out)


def plot_regret_log(
    long: pd.DataFrame,
    opt_cost: float,
    cost_cap: float,
    budget: float,
    out: Path | None,
    problem: str = "enamel",
    grid_points: int = 100,
    subtitle: str = "",
) -> Path | None:
    """Simple regret (best truly feasible cost minus optimum) on a log axis vs spend."""
    grid = np.linspace(0.0, budget, grid_points)
    unit = UNITS.get(problem, "$/unit")
    floor = 1e-4 * opt_cost
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for method in _methods_in(long):
        regret = np.maximum(best_cost_matrix(long, method, cost_cap, grid) - opt_cost, floor)
        mean, lo, hi = bootstrap_band(regret)
        color = METHOD_COLORS.get(method)
        ax.plot(grid, mean, color=color, lw=2, label=METHOD_LABELS.get(method, method))
        ax.fill_between(grid, lo, hi, color=color, alpha=0.15, lw=0)
    ax.set_yscale("log")
    ax.set_xlim(0, budget)
    ax.set_xlabel("Cumulative lab spend ($)")
    ax.set_ylabel(f"Mean simple regret ({unit}, log scale)")
    ax.set_title(
        f"{problem.capitalize()}: simple regret vs spend (mean, 95 % bootstrap band)"
        + (f"\n{subtitle}" if subtitle else ""),
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3, which="both")
    return _save(fig, out)


# ----- surrogate views (shared by matplotlib figures and the Plotly app) --------------------
def slice_data(
    gps: dict, problem, x_ref: np.ndarray, var_index: int, n: int = 120
) -> dict[str, np.ndarray]:
    """GP mean and 95 % band (raw output units) along one variable through `x_ref`."""
    lo, hi = problem.bounds[var_index]
    grid = np.linspace(lo, hi, n)
    X = np.repeat(np.asarray(x_ref, dtype=float)[None, :], n, axis=0)
    X[:, var_index] = grid
    out = {"grid": grid, "known_feasible": problem.known_feasible(X)}
    for spec in problem.specs:
        mu, sd = gps[spec.output].predict(X)
        out[f"{spec.output}_mean"] = spec.inverse_transform(mu)
        out[f"{spec.output}_lo"] = spec.inverse_transform(mu - 1.96 * sd)
        out[f"{spec.output}_hi"] = spec.inverse_transform(mu + 1.96 * sd)
    return out


def pof_grid(
    gps: dict, problem, x_ref: np.ndarray, var_x: str, var_y: str, n: int = 60
) -> dict[str, np.ndarray]:
    """PoF over a 2D grid of two variables with the others fixed at `x_ref`."""
    from formopt.acquisition import probability_of_feasibility

    names = problem.var_names
    ix, iy = names.index(var_x), names.index(var_y)
    gx = np.linspace(*problem.bounds[ix], n)
    gy = np.linspace(*problem.bounds[iy], n)
    XX, YY = np.meshgrid(gx, gy)
    X = np.repeat(np.asarray(x_ref, dtype=float)[None, :], XX.size, axis=0)
    X[:, ix], X[:, iy] = XX.ravel(), YY.ravel()
    preds = {o: gp.predict(X) for o, gp in gps.items()}
    pof = probability_of_feasibility(preds, problem.specs).reshape(XX.shape)
    known = problem.known_feasible(X).reshape(XX.shape)
    return {"x": gx, "y": gy, "pof": pof, "known_feasible": known}


def plot_gp_1d_slices(
    gps: dict, x_ref: np.ndarray, problem, out: Path | None, history: list | None = None
) -> Figure | Path | None:
    """1D slices of each output's GP through `x_ref`, with observations and spec limits."""
    d, k = len(problem.variables), len(problem.specs)
    fig, axes = plt.subplots(k, d, figsize=(2.9 * d, 2.4 * k), squeeze=False)
    X_hist = np.vstack([o.x for o in history]) if history else None
    for j, var in enumerate(problem.variables):
        data = slice_data(gps, problem, x_ref, j)
        for i, spec in enumerate(problem.specs):
            ax = axes[i, j]
            ax.plot(data["grid"], data[f"{spec.output}_mean"], color="#0072B2", lw=1.5)
            ax.fill_between(
                data["grid"],
                data[f"{spec.output}_lo"],
                data[f"{spec.output}_hi"],
                color="#0072B2",
                alpha=0.2,
                lw=0,
            )
            for limit in (spec.lower, spec.upper):
                if limit is not None:
                    ax.axhline(limit, color="#D55E00", ls="--", lw=1)
            if X_hist is not None:
                y = [o.outputs[spec.output] for o in history]
                ax.scatter(X_hist[:, j], y, s=8, color="black", alpha=0.5, zorder=3)
            ax.axvline(x_ref[j], color="grey", lw=0.8)
            infeasible = ~data["known_feasible"]
            if infeasible.any():
                ax.axvspan(
                    data["grid"][infeasible].min(),
                    data["grid"][infeasible].max(),
                    color="grey",
                    alpha=0.15,
                    lw=0,
                )
            if spec.transform == "log":
                ax.set_yscale("log")
            if i == k - 1:
                ax.set_xlabel(f"{var.name} ({var.unit})", fontsize=8)
            if j == 0:
                ax.set_ylabel(spec.output, fontsize=8)
            ax.tick_params(labelsize=7)
    fig.suptitle(
        "GP mean and 95 % band through the recommended recipe (grey line); "
        "dashed = spec limit; dots = all observations (projected)",
        fontsize=9,
    )
    fig.tight_layout()
    return _save(fig, out) if out is not None else fig


def plot_pof_heatmap(
    gps: dict, problem, x_ref: np.ndarray, var_x: str, var_y: str, out: Path | None
) -> Figure | Path | None:
    """Probability of feasibility over two variables (others fixed at the recommended recipe)."""
    data = pof_grid(gps, problem, x_ref, var_x, var_y)
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    mesh = ax.pcolormesh(data["x"], data["y"], data["pof"], cmap="viridis", vmin=0, vmax=1)
    fig.colorbar(mesh, ax=ax, label="Probability of feasibility (all specs)")
    if (~data["known_feasible"]).any():
        ax.contourf(
            data["x"],
            data["y"],
            (~data["known_feasible"]).astype(float),
            levels=[0.5, 1.5],
            colors="white",
            alpha=0.6,
        )
    ax.contour(data["x"], data["y"], data["pof"], levels=[0.5], colors="white", linewidths=1)
    names = problem.var_names
    ax.scatter(
        x_ref[names.index(var_x)],
        x_ref[names.index(var_y)],
        marker="*",
        s=150,
        color="#D55E00",
        edgecolor="white",
        label="Recommended recipe",
    )
    vx, vy = problem.variables[names.index(var_x)], problem.variables[names.index(var_y)]
    ax.set_xlabel(f"{vx.name} ({vx.unit})")
    ax.set_ylabel(f"{vy.name} ({vy.unit})")
    ax.set_title(
        f"PoF over {var_x} x {var_y}; white line = PoF 0.5; pale = known-infeasible", fontsize=9
    )
    ax.legend(loc="upper right", fontsize=8)
    return _save(fig, out) if out is not None else fig


def plot_calibration(z_scores: dict[str, list[float]], out: Path, title: str = "") -> Path:
    """Histogram of next-point z-scores per output with a standard normal overlay."""
    from scipy.stats import norm

    keys = [k for k, v in z_scores.items() if len(v)]
    fig, axes = plt.subplots(1, len(keys), figsize=(4.0 * len(keys), 3.4), squeeze=False)
    xs = np.linspace(-5, 5, 200)
    for ax, key in zip(axes[0], keys, strict=True):
        z = np.asarray(z_scores[key], dtype=float)
        cov = float(np.mean(np.abs(z) <= 1.96))
        ax.hist(
            np.clip(z, -5, 5),
            bins=40,
            range=(-5, 5),
            density=True,
            color="#56B4E9",
            edgecolor="white",
        )
        ax.plot(xs, norm.pdf(xs), color="black", lw=1.2, label="N(0, 1)")
        ax.set_title(f"{key}\n95 % coverage {cov:.3f} (n = {z.size})", fontsize=9)
        ax.set_xlabel("z = (observed - predicted) / predicted sd\n(clipped to +/-5)", fontsize=8)
        ax.legend(fontsize=7)
    fig.suptitle(title or "Surrogate calibration at the next suggested point", fontsize=10)
    fig.tight_layout()
    return _save(fig, out)


def plot_price_shock(result: dict, problem, out: Path, title: str = "") -> Path:
    """Old vs new recommended recipe (each variable as % of its range) and the costs."""
    names = problem.var_names
    b = problem.bounds
    old_rec, new_rec = result["old"], result["new"]
    old = (old_rec["x"] - b[:, 0]) / (b[:, 1] - b[:, 0]) * 100
    new = (new_rec["x"] - b[:, 0]) / (b[:, 1] - b[:, 0]) * 100
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0), gridspec_kw={"width_ratios": [3, 1.4]})
    pos = np.arange(len(names))
    ax.barh(pos + 0.2, old, height=0.4, color="#999999", label="Old prices")
    ax.barh(pos - 0.2, new, height=0.4, color="#D55E00", label="New prices")
    labels = [f"{n}\n{old_rec['recipe'][n]:.3g} -> {new_rec['recipe'][n]:.3g}" for n in names]
    ax.set_yticks(pos, labels, fontsize=7)
    ax.set_xlabel("Position in allowed range (%)")
    ax.set_xlim(0, 100)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Recommended recipe (zero new experiments)", fontsize=9)
    unit = UNITS.get(problem.name, "")
    bars = {
        "old recipe,\nold prices": old_rec["material_cost"],
        "old recipe,\nnew prices": result["old_recipe_cost_at_new_prices"],
        "new recipe,\nnew prices": new_rec["material_cost"],
    }
    values = list(bars.values())
    ax2.bar(range(3), values, color=["#999999", "#BBBBBB", "#D55E00"])
    ax2.set_xticks(range(3), list(bars.keys()), fontsize=7)
    ax2.set_ylim(min(values) * 0.95, max(values) * 1.02)
    for i, v in enumerate(values):
        ax2.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    ax2.set_ylabel(f"Material cost ({unit})")
    ax2.set_title(f"PoF old {old_rec['pof']:.2f}, new {new_rec['pof']:.2f}", fontsize=9)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return _save(fig, out)
