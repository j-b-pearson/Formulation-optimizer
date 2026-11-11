"""Run-cost heterogeneity sweep and the pre-registered H1 / H2 tests (v2 study).

The plan (hypotheses, metrics, decision rules, seeds) is fixed in `docs/PREREGISTRATION_v2.md`.
This module only implements it:

- `with_run_cost_spread` makes a config copy with the enamel run-cost spread s set.
- `run_sweep` runs the enamel benchmark once per spread level into `<out>/s_<s>/`.
- `per_seed_advantage` turns campaign metrics into per-seed advantages of a cost-aware method
  over plain BO (positive = the cost-aware method did better).
- `analyze` computes the per-level paired advantages, the growth test (H1), the "starts to
  pay" level, the H2 test and a robustness re-check, and writes everything to CSV and JSON.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from formopt.config import AppConfig
from formopt.metrics import bootstrap_ci

logger = logging.getLogger(__name__)

CO_PRIMARY = ("spend_saved", "success_gain")
SECONDARY = ("regret_reduction",)


def level_name(s: float) -> str:
    """Folder name for a spread level: s_1, s_3, s_10 (s_2.5 for non-integers)."""
    return f"s_{s:g}"


def with_run_cost_spread(cfg: AppConfig, s: float | None) -> AppConfig:
    """Copy of the config with the enamel run-cost spread set (None = the v1 formula)."""
    old = cfg.enamel.run_cost_spread
    spread = type(old).model_validate({**old.model_dump(), "s": s})  # validated: s >= 1
    enamel = cfg.enamel.model_copy(update={"run_cost_spread": spread})
    new = cfg.model_copy(update={"enamel": enamel})
    new.source_path = cfg.source_path
    return new


def run_cost_spread_report(cfg: AppConfig, levels: list[float]) -> dict:
    """Constants of the run-cost family at each level, plus the v1 effective spread."""
    from formopt.problems.enamel import EnamelProblem

    rows = []
    for s in levels:
        problem = EnamelProblem(with_run_cost_spread(cfg, float(s)).enamel)
        sc = problem.run_cost_scale
        rows.append(
            {
                "s": float(s),
                "k_cheapest_run": sc.k,
                "most_expensive_run": sc.run_cost_max,
                "mean_run_cost": sc.k * (1.0 + (sc.s - 1.0) * sc.mean_u),
                "expected_runs_in_budget_at_mean": cfg.enamel.budget
                / (sc.k * (1.0 + (sc.s - 1.0) * sc.mean_u)),
            }
        )
    probe = EnamelProblem(with_run_cost_spread(cfg, 1.0).enamel).run_cost_scale
    v1_min = probe.v1_fixed + probe.driver_min
    v1_max = probe.v1_fixed + probe.driver_max
    spread = cfg.enamel.run_cost_spread
    return {
        "formula": "run_cost = k_s * (1 + (s - 1) * u(x)), u = (driver - min) / (max - min)",
        "driver": "v1 variable run cost: batch_kg * material_cost + oven energy/time",
        "extremes_method": "exact: driver evaluated at all 32 corners of the known-feasible box",
        "mean_method": f"Monte Carlo, {spread.mean_samples} uniform known-feasible points, "
        f"seed {spread.mean_seed}",
        "driver_min": probe.driver_min,
        "driver_max": probe.driver_max,
        "mean_u": probe.mean_u,
        "v1_mean_run_cost": probe.v1_mean_run_cost,
        "v1_run_cost_min": v1_min,
        "v1_run_cost_max": v1_max,
        "v1_effective_s": probe.v1_effective_s,
        "budget": cfg.enamel.budget,
        "levels": rows,
    }


def run_sweep(
    cfg: AppConfig,
    levels: list[float],
    methods: list[str],
    seeds: list[int],
    out_dir: Path,
    n_jobs: int = 1,
    log_mlflow: bool = True,
    mlflow_suffix: str = "-v2",
) -> dict:
    """Run the enamel benchmark at every spread level; return wall times per level."""
    from formopt.benchmark import run_benchmark

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timing = {}
    start_all = time.perf_counter()
    for s in levels:
        cfg_s = with_run_cost_spread(cfg, float(s))
        start = time.perf_counter()
        run_benchmark(
            cfg_s,
            ["enamel"],
            methods,
            seeds,
            n_jobs=n_jobs,
            out_dir=out_dir / level_name(s),
            log_mlflow=log_mlflow,
            experiment_suffix=f"{mlflow_suffix}-{level_name(s)}",
        )
        timing[level_name(s)] = time.perf_counter() - start
        logger.info("Sweep level s = %g done in %.1f s", s, timing[level_name(s)])
    info = {
        "levels": [float(s) for s in levels],
        "methods": methods,
        "seeds": seeds,
        "n_jobs": n_jobs,
        "log_mlflow": log_mlflow,
        "wall_time_s_per_level": timing,
        "wall_time_s_total": time.perf_counter() - start_all,
    }
    (out_dir / "sweep_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


# ----- analysis --------------------------------------------------------------------------
def load_run(folder: Path, problem: str = "enamel") -> tuple[pd.DataFrame, dict]:
    """Campaign metrics and benchmark info written by `benchmark` in one folder."""
    folder = Path(folder)
    metrics = pd.read_csv(folder / f"campaign_metrics_{problem}.csv")
    info = json.loads((folder / f"benchmark_info_{problem}.json").read_text(encoding="utf-8"))
    return metrics, info


def _as_int(col: pd.Series) -> np.ndarray:
    """0/1 integers from a boolean column (CSV round trips give bool or 'True'/'False')."""
    return col.astype(str).str.lower().eq("true").astype(int).to_numpy()


def per_seed_advantage(
    metrics: pd.DataFrame, method: str, baseline: str, budget: float
) -> pd.DataFrame:
    """Per-seed advantage of `method` over `baseline` (positive = `method` did better).

    spend_saved = censored spend-to-target of the baseline minus that of the method, where a
    campaign that never reaches the target counts as spending the whole budget; success_gain =
    success of the method minus success of the baseline (each 0 or 1); regret_reduction =
    final regret of the baseline minus that of the method.
    """
    a = metrics[metrics.method == method].set_index("seed").sort_index()
    b = metrics[metrics.method == baseline].set_index("seed").sort_index()
    seeds = a.index.intersection(b.index)
    if len(seeds) != len(a) or len(seeds) != len(b):
        raise ValueError(f"{method} and {baseline} do not have the same seeds")
    a, b = a.loc[seeds], b.loc[seeds]
    spend_a = np.minimum(a["spend_to_target"].to_numpy(float), budget)
    spend_b = np.minimum(b["spend_to_target"].to_numpy(float), budget)
    return pd.DataFrame(
        {
            "seed": seeds.to_numpy(),
            "spend_saved": spend_b - spend_a,
            "success_gain": _as_int(a["success"]) - _as_int(b["success"]),
            "regret_reduction": b["final_regret"].to_numpy(float)
            - a["final_regret"].to_numpy(float),
        }
    )


def summarize_advantage(values: np.ndarray, n_boot: int, seed: int) -> dict:
    """Mean advantage with a percentile bootstrap CI, plus seed counts."""
    values = np.asarray(values, dtype=float)
    mean, lo, hi = bootstrap_ci(values, np.mean, n_boot, seed=seed)
    return {
        "mean": mean,
        "ci_lo": lo,
        "ci_hi": hi,
        "ci_above_zero": bool(lo > 0),
        "ci_below_zero": bool(hi < 0),
        "n_seeds": int(values.size),
        "seeds_better": int(np.sum(values > 0)),
        "seeds_tied": int(np.sum(values == 0)),
        "seeds_worse": int(np.sum(values < 0)),
    }


def level_table(
    runs: dict[float, tuple[pd.DataFrame, dict]],
    methods: list[str],
    baseline: str,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    """Per-level paired advantage of each cost-aware method over plain BO, every metric."""
    rows = []
    for s, (metrics, info) in runs.items():
        for method in methods:
            adv = per_seed_advantage(metrics, method, baseline, info["budget"])
            for metric in (*CO_PRIMARY, *SECONDARY):
                rows.append(
                    {
                        "s": s,
                        "method": method,
                        "baseline": baseline,
                        "metric": metric,
                        **summarize_advantage(adv[metric].to_numpy(), n_boot, seed),
                    }
                )
    return pd.DataFrame(rows)


def growth_test(
    adv_low: pd.DataFrame, adv_high: pd.DataFrame, metric: str, n_boot: int, seed: int
) -> dict:
    """Bootstrap CI of A(high) - A(low), resampling seeds jointly across the two levels.

    The same seed numbers are used at every level, so the per-seed difference keeps the pairing
    (same initial design and the same noise draws for a seed at every level).
    """
    lo = adv_low.set_index("seed")[metric]
    hi = adv_high.set_index("seed")[metric]
    if not lo.index.equals(hi.index):
        raise ValueError("Growth test needs the same seeds at both levels")
    diff = (hi - lo).to_numpy(float)
    mean, ci_lo, ci_hi = bootstrap_ci(diff, np.mean, n_boot, seed=seed)
    return {"growth": mean, "growth_ci_lo": ci_lo, "growth_ci_hi": ci_hi}


def growth_table(
    runs: dict[float, tuple[pd.DataFrame, dict]],
    methods: list[str],
    baseline: str,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    """The H1 criterion per method and co-primary metric (rule in PREREGISTRATION_v2.md)."""
    levels = sorted(runs)
    low, high = levels[0], levels[-1]
    rows = []
    for method in methods:
        adv = {
            s: per_seed_advantage(runs[s][0], method, baseline, runs[s][1]["budget"])
            for s in levels
        }
        for metric in (*CO_PRIMARY, *SECONDARY):
            means = [float(adv[s][metric].mean()) for s in levels]
            test = growth_test(adv[low], adv[high], metric, n_boot, seed)
            monotone = bool(all(a <= b for a, b in zip(means, means[1:], strict=False)))
            row = {
                "method": method,
                "metric": metric,
                "low_level": low,
                "high_level": high,
                **{f"A_s{s:g}": m for s, m in zip(levels, means, strict=True)},
                **test,
                "growth_ci_above_zero": bool(test["growth_ci_lo"] > 0),
                "non_decreasing_across_levels": monotone,
            }
            row["criterion_met"] = bool(row["growth_ci_above_zero"] and monotone)
            rows.append(row)
    return pd.DataFrame(rows)


def h1_verdict(growth: pd.DataFrame, method: str) -> str:
    """'supported' if both co-primary metrics meet the criterion, 'partially supported' if one
    does, 'not supported' if neither."""
    g = growth[(growth.method == method) & growth.metric.isin(CO_PRIMARY)]
    met = int(g["criterion_met"].sum())
    return {2: "supported", 1: "partially supported"}.get(met, "not supported")


def starts_to_pay(levels: pd.DataFrame, method: str) -> float | None:
    """Smallest level where one co-primary advantage has a CI above zero and the other
    co-primary point estimate is not negative; None if no tested level qualifies."""
    for s in sorted(levels.s.unique()):
        g = levels[(levels.s == s) & (levels.method == method)].set_index("metric")
        for m, other in (CO_PRIMARY, CO_PRIMARY[::-1]):
            if g.loc[m, "ci_above_zero"] and g.loc[other, "mean"] >= 0:
                return float(s)
    return None


def h2_test(
    metrics: pd.DataFrame, method: str, baseline: str, budget: float, n_boot: int, seed: int
) -> dict:
    """H2: success difference of `method` minus `baseline` at v1 settings, bootstrap CI.

    Also reports the discordant pairs and an exact two-sided McNemar (binomial) p-value as a
    secondary check.
    """
    adv = per_seed_advantage(metrics, method, baseline, budget)
    gain = adv["success_gain"].to_numpy(float)
    summary = summarize_advantage(gain, n_boot, seed)
    a_only = int(np.sum(gain > 0))
    b_only = int(np.sum(gain < 0))
    p = binomtest(a_only, a_only + b_only, 0.5).pvalue if a_only + b_only else 1.0
    rates = metrics.groupby("method")["success"].mean()
    return {
        "method": method,
        "baseline": baseline,
        "success_rate_method": float(rates[method]),
        "success_rate_baseline": float(rates[baseline]),
        "paired_success_difference": summary["mean"],
        "ci_lo": summary["ci_lo"],
        "ci_hi": summary["ci_hi"],
        "n_seeds": summary["n_seeds"],
        "seeds_only_method_succeeded": a_only,
        "seeds_only_baseline_succeeded": b_only,
        "mcnemar_exact_p_two_sided": float(p),
        "replicated": bool(summary["ci_lo"] > 0),
    }


def mechanism_table(runs: dict[float, tuple[pd.DataFrame, dict]]) -> pd.DataFrame:
    """Descriptive: per level and method, success rate, experiments and mean $ per run."""
    rows = []
    for s, (metrics, _info) in runs.items():
        for method, g in metrics.groupby("method", sort=False):
            rows.append(
                {
                    "s": s,
                    "method": method,
                    "n_seeds": len(g),
                    "success_rate": float(g["success"].mean()),
                    "mean_n_experiments": float(g["n_experiments"].mean()),
                    "mean_run_cost": float((g["total_spend"] / g["n_experiments"]).mean()),
                    "median_final_regret": float(g["final_regret"].median()),
                    "mean_final_regret": float(g["final_regret"].mean()),
                }
            )
    return pd.DataFrame(rows)


def first_feasible_spend(long: pd.DataFrame, budget: float) -> pd.DataFrame:
    """Secondary (alternative reading of 'reach spec'): spend when a campaign first runs any
    truly in-spec recipe, whatever its cost; the budget if it never does."""
    rows = []
    for (method, seed), g in long.groupby(["method", "seed"]):
        g = g.sort_values("step")
        hit = g.loc[g["truly_feasible"].astype(bool), "cumulative_spend"]
        rows.append(
            {"method": method, "seed": seed, "spend": float(hit.iloc[0]) if len(hit) else budget}
        )
    return pd.DataFrame(rows)


def analyze(
    cfg: AppConfig,
    sweep_dir: Path,
    v1_settings_dir: Path | None,
    out_dir: Path,
    n_boot: int | None = None,
    seed: int | None = None,
) -> dict:
    """Run the pre-registered H1 and H2 analysis and write the tables and a results JSON."""
    v2 = cfg.v2
    n_boot = v2.n_boot if n_boot is None else n_boot
    seed = v2.bootstrap_seed if seed is None else seed
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = {float(s): load_run(Path(sweep_dir) / level_name(s)) for s in v2.spread_levels}
    methods = [v2.h1_headline_method, v2.h1_secondary_method]
    levels = level_table(runs, methods, v2.plain_method, n_boot, seed)
    growth = growth_table(runs, methods, v2.plain_method, n_boot, seed)
    mech = mechanism_table(runs)
    levels.to_csv(out_dir / "h1_per_level.csv", index=False)
    growth.to_csv(out_dir / "h1_growth.csv", index=False)
    mech.to_csv(out_dir / "h1_descriptive_by_level.csv", index=False)
    result: dict = {
        "n_boot": n_boot,
        "bootstrap_seed": seed,
        "levels": sorted(runs),
        "seeds": runs[sorted(runs)[0]][1]["seeds"],
        "h1": {
            m: {"verdict": h1_verdict(growth, m), "starts_to_pay_at_s": starts_to_pay(levels, m)}
            for m in methods
        },
        "h1_headline_method": v2.h1_headline_method,
    }
    ff_rows = []
    for s in sorted(runs):
        long_path = Path(sweep_dir) / level_name(s) / "benchmark_enamel.csv"
        if not long_path.exists():
            continue
        ff = first_feasible_spend(pd.read_csv(long_path), runs[s][1]["budget"])
        wide = ff.pivot(index="seed", columns="method", values="spend")
        for m in methods:
            saved = (wide[v2.plain_method] - wide[m]).to_numpy(float)
            ff_rows.append({"s": s, "method": m, **summarize_advantage(saved, n_boot, seed)})
    if ff_rows:
        pd.DataFrame(ff_rows).to_csv(out_dir / "h1_secondary_first_feasible.csv", index=False)
    if (
        v1_settings_dir is not None
        and (Path(v1_settings_dir) / "campaign_metrics_enamel.csv").exists()
    ):
        metrics, info = load_run(Path(v1_settings_dir))
        result["h2"] = h2_test(metrics, v2.h2_method, v2.plain_method, info["budget"], n_boot, seed)
        ref_rows = []
        for m in methods:
            adv = per_seed_advantage(metrics, m, v2.plain_method, info["budget"])
            for metric in (*CO_PRIMARY, *SECONDARY):
                ref_rows.append(
                    {
                        "method": m,
                        "metric": metric,
                        **summarize_advantage(adv[metric].to_numpy(), n_boot, seed),
                    }
                )
        pd.DataFrame(ref_rows).to_csv(out_dir / "v1_settings_advantage.csv", index=False)
        pd.DataFrame([result["h2"]]).to_csv(out_dir / "h2.csv", index=False)
    (out_dir / "h1_h2_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def robustness_check(
    cfg: AppConfig, sweep_dir: Path, v1_settings_dir: Path | None, main: dict
) -> dict:
    """Pre-registered re-check: do the verdicts change with more bootstrap resamples?

    `main` is the result of `analyze` with the pre-registered resamples.
    """
    import tempfile

    v2 = cfg.v2
    with tempfile.TemporaryDirectory() as tmp:
        alt = analyze(
            cfg, sweep_dir, v1_settings_dir, Path(tmp), v2.robustness_n_boot, v2.robustness_seed
        )
        growth_alt = pd.read_csv(Path(tmp) / "h1_growth.csv")
    same_h1 = all(main["h1"][m]["verdict"] == alt["h1"][m]["verdict"] for m in main["h1"])
    same_pay = all(
        main["h1"][m]["starts_to_pay_at_s"] == alt["h1"][m]["starts_to_pay_at_s"]
        for m in main["h1"]
    )
    same_h2 = ("h2" not in main) or main["h2"]["replicated"] == alt["h2"]["replicated"]
    return {
        "n_boot": v2.robustness_n_boot,
        "bootstrap_seed": v2.robustness_seed,
        "h1": alt["h1"],
        "h2": alt.get("h2"),
        "growth": growth_alt.to_dict(orient="records"),
        "h1_verdicts_unchanged": bool(same_h1),
        "starts_to_pay_unchanged": bool(same_pay),
        "h2_verdict_unchanged": bool(same_h2),
    }
