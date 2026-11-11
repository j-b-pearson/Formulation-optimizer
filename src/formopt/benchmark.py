"""Seeded benchmark: every method on every seed, with metrics, CIs, CSVs and MLflow runs.

Fairness rules (DOMAIN_PRIMER section 5): every method gets the same budget and the same cost
accounting; BO variants share their initial design for a seed; measurement noise depends only
on (seed, step) (common random numbers); success is judged on the noise-free truth.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from formopt.config import AppConfig
from formopt.loop import run_campaign
from formopt.metrics import (
    bootstrap_ci,
    experiments_to_target,
    paired_difference_ci,
    spend_to_target,
    wilson_ci,
)
from formopt.problems import build_problem, problem_budget
from formopt.problems.base import Problem, reference_optimum
from formopt.strategies import (
    CostAwareBO,
    FactorialPlan,
    GPSettings,
    LHSPlan,
    PlainConstrainedBO,
    RandomSearch,
    Strategy,
)
from formopt.types import CampaignResult

logger = logging.getLogger(__name__)

METHODS = ("random", "lhs", "factorial", "bo_plain", "bo_cost", "bo_cost_cooled")
BO_METHODS = ("bo_plain", "bo_cost", "bo_cost_cooled")
COST_AWARE = ("bo_cost", "bo_cost_cooled")
_PROBLEM_CACHE: dict[str, Problem] = {}


def gp_settings(cfg: AppConfig) -> GPSettings:
    bo = cfg.bo
    return GPSettings(
        restarts=bo.gp_restarts,
        length_scale_bounds=tuple(bo.gp_length_scale_bounds),
        constant_bounds=tuple(bo.gp_constant_bounds),
        noise_bounds=tuple(bo.gp_noise_bounds),
    )


def make_strategy(method: str, cfg: AppConfig, problem_name: str, quick: bool = False) -> Strategy:
    """Build a fresh strategy object for one campaign."""
    bo = cfg.bo
    n_cand = bo.quick_n_candidates if quick else bo.n_candidates
    gp = gp_settings(cfg)
    fac = cfg.benchmark.factorial
    if method == "random":
        return RandomSearch(cfg.benchmark.random_batch)
    if method == "lhs":
        return LHSPlan()
    if method == "factorial":
        generators = {"enamel": fac.enamel_generators, "concrete": fac.concrete_generators}
        return FactorialPlan(generators[problem_name], tuple(fac.levels), fac.n_center)
    shared = {"alpha": 1.0, "n_init": bo.n_init, "n_candidates": n_cand, "gp": gp}
    if method == "bo_plain":
        return PlainConstrainedBO(method, **shared)
    if method == "bo_cost":
        return CostAwareBO(method, alpha_mode="fixed", **shared)
    if method == "bo_cost_cooled":
        return CostAwareBO(method, alpha_mode="cooled", **shared)
    batch = re.fullmatch(r"(bo_plain|bo_cost|bo_cost_cooled)_q(\d+)", method)
    if batch:  # batch mode, e.g. "bo_cost_cooled_q4" plans 4 runs per oven cycle
        from formopt.batch import BatchBO

        mode = {"bo_plain": "none", "bo_cost": "fixed", "bo_cost_cooled": "cooled"}[batch[1]]
        return BatchBO(method, alpha_mode=mode, q=int(batch[2]), **shared)
    raise ValueError(f"Unknown method {method!r}; choose from {METHODS}")


def _cached_problem(name: str, cfg: AppConfig) -> Problem:
    """Build each problem once per process (the concrete emulator is a 30 MB file)."""
    key = hashlib.sha1(f"{name}:{cfg.model_dump_json()}".encode()).hexdigest()
    if key not in _PROBLEM_CACHE:
        _PROBLEM_CACHE[key] = build_problem(name, cfg)
    return _PROBLEM_CACHE[key]


def run_one(
    cfg: AppConfig, problem_name: str, method: str, seed: int, quick: bool = False
) -> CampaignResult:
    """One campaign; safe to call in a worker process."""
    problem = _cached_problem(problem_name, cfg)
    strategy = make_strategy(method, cfg, problem_name, quick)
    return run_campaign(problem, strategy, problem_budget(problem_name, cfg), seed)


def campaign_rows(result: CampaignResult, problem: Problem) -> list[dict]:
    """Long-format rows: one per experiment."""
    rows = []
    spend = result.cumulative_spend()
    for step, obs in enumerate(result.observations):
        row = {"problem": result.problem, "method": result.method, "seed": result.seed}
        row["step"] = step
        row.update(problem.as_dict(obs.x))
        row.update(obs.outputs)
        row.update({f"true_{k}": v for k, v in obs.true_outputs.items()})
        row.update(
            {
                "run_cost": obs.run_cost,
                "cumulative_spend": float(spend[step]),
                "material_cost": obs.material_cost,
                "passed_spec": obs.passed_spec,
                "truly_feasible": obs.truly_feasible,
            }
        )
        for out in problem.output_names:
            mu, sd = obs.predicted[out] if obs.predicted else (np.nan, np.nan)
            row[f"pred_mu_{out}"] = mu
            row[f"pred_sd_{out}"] = sd
        rows.append(row)
    return rows


def calibration_z(result: CampaignResult, problem: Problem) -> dict[str, list[float]]:
    """Next-point z-scores (y_obs - mu) / sigma per output, in GP space (BO only)."""
    z: dict[str, list[float]] = {s.output: [] for s in problem.specs}
    for obs in result.observations:
        if not obs.predicted:
            continue
        for spec in problem.specs:
            mu, sd = obs.predicted[spec.output]
            y = float(spec.transform_values(obs.outputs[spec.output]))
            z[spec.output].append((y - mu) / sd)
    return z


def campaign_metrics(
    result: CampaignResult, problem: Problem, opt_cost: float, cost_cap: float, tol: float
) -> dict:
    """Per-campaign metrics, judged on the noise-free truth, plus the 'lab said pass' view."""
    trace = result.best_feasible_cost_trace(use_truth=True, cost_if_none=cost_cap)
    final_best = float(trace[-1]) if len(trace) else cost_cap
    passed = [o for o in result.observations if o.passed_spec]
    lab_pick = min(passed, key=lambda o: o.material_cost) if passed else None
    z = calibration_z(result, problem)
    all_z = np.concatenate([np.asarray(v) for v in z.values()]) if z else np.array([])
    return {
        "problem": result.problem,
        "method": result.method,
        "seed": result.seed,
        "n_experiments": result.n_experiments,
        "total_spend": float(result.cumulative_spend()[-1]) if result.n_experiments else 0.0,
        "final_best_true_cost": final_best,
        "found_truly_feasible": bool(any(o.truly_feasible for o in result.observations)),
        "final_regret": final_best - opt_cost,
        "spend_to_target": spend_to_target(result, opt_cost, tol),
        "experiments_to_target": experiments_to_target(result, opt_cost, tol),
        "success": bool(np.isfinite(spend_to_target(result, opt_cost, tol))),
        "lab_pick_cost": lab_pick.material_cost if lab_pick else np.nan,
        "lab_pick_truly_feasible": bool(lab_pick.truly_feasible) if lab_pick else False,
        "lab_claimed_success": bool(lab_pick and lab_pick.material_cost <= opt_cost * (1 + tol)),
        "lab_success": bool(
            lab_pick and lab_pick.truly_feasible and lab_pick.material_cost <= opt_cost * (1 + tol)
        ),
        "n_false_passes": int(sum(o.passed_spec and not o.truly_feasible for o in passed)),
        "n_calibration_points": int(all_z.size),
        "calibration_coverage_95": float(np.mean(np.abs(all_z) <= 1.96)) if all_z.size else np.nan,
        "n_gp_warnings": result.n_gp_warnings,
        "wall_time_s": result.wall_time_s,
    }


def summarize(metrics: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """One row per method: success rate (Wilson CI), spend-to-target, regret (bootstrap CIs)."""
    rows = []
    for method, g in metrics.groupby("method", sort=False):
        n = len(g)
        k = int(g["success"].sum())
        p, p_lo, p_hi = wilson_ci(k, n)
        regret = g["final_regret"].to_numpy(float)
        mean_r, mean_lo, mean_hi = bootstrap_ci(regret, np.mean, n_boot, seed=seed)
        med_r, med_lo, med_hi = bootstrap_ci(regret, np.median, n_boot, seed=seed)
        row = {
            "method": method,
            "n_seeds": n,
            "success_rate": p,
            "success_ci_lo": p_lo,
            "success_ci_hi": p_hi,
            "median_spend_to_target": np.nan,
            "spend_ci_lo": np.nan,
            "spend_ci_hi": np.nan,
            "median_experiments_to_target": np.nan,
            "mean_final_regret": mean_r,
            "mean_regret_ci_lo": mean_lo,
            "mean_regret_ci_hi": mean_hi,
            "median_final_regret": med_r,
            "median_regret_ci_lo": med_lo,
            "median_regret_ci_hi": med_hi,
            "mean_n_experiments": float(g["n_experiments"].mean()),
            "mean_total_spend": float(g["total_spend"].mean()),
            "found_feasible_rate": float(g["found_truly_feasible"].mean()),
            "lab_claimed_success_rate": float(g["lab_claimed_success"].mean()),
            "lab_success_rate": float(g["lab_success"].mean()),
            "lab_pick_infeasible_rate": float(
                (~g["lab_pick_truly_feasible"] & g["lab_pick_cost"].notna()).mean()
            ),
            "calibration_coverage_95": float(
                np.nansum(g["calibration_coverage_95"] * g["n_calibration_points"])
                / max(g["n_calibration_points"].sum(), 1)
            )
            if g["n_calibration_points"].sum()
            else np.nan,
            "total_gp_warnings": int(g["n_gp_warnings"].sum()),
            "mean_wall_time_s": float(g["wall_time_s"].mean()),
        }
        if k > n / 2:  # median defined only when more than half of the seeds succeed
            spend = g["spend_to_target"].to_numpy(float)
            med, lo, hi = bootstrap_ci(spend, np.median, n_boot, seed=seed)
            row.update({"median_spend_to_target": med, "spend_ci_lo": lo, "spend_ci_hi": hi})
            row["median_experiments_to_target"] = float(np.median(g["experiments_to_target"]))
        rows.append(row)
    return pd.DataFrame(rows)


def paired_differences(
    metrics: pd.DataFrame, budget: float, n_boot: int = 2000, seed: int = 0
) -> pd.DataFrame:
    """Per-seed differences (cost-aware BO minus each other method) with bootstrap CIs.

    Spend-to-target is censored at the budget when the target is never reached (a lower bound
    on the true spend for that seed), so the difference is defined for every seed.
    """
    rows = []
    wide = metrics.set_index(["method", "seed"])
    methods = list(dict.fromkeys(metrics["method"]))
    for a in [m for m in COST_AWARE if m in methods]:
        for b in [m for m in methods if m != a]:
            seeds = sorted(
                set(metrics.loc[metrics.method == a, "seed"])
                & set(metrics.loc[metrics.method == b, "seed"])
            )
            for metric, better in (
                ("final_regret", "lower"),
                ("spend_to_target_censored", "lower"),
                ("success", "higher"),
            ):
                col = "spend_to_target" if metric.startswith("spend") else metric
                va = wide.loc[[(a, s) for s in seeds], col].to_numpy(float)
                vb = wide.loc[[(b, s) for s in seeds], col].to_numpy(float)
                if metric == "spend_to_target_censored":
                    va, vb = np.minimum(va, budget), np.minimum(vb, budget)
                mean, lo, hi = paired_difference_ci(va, vb, n_boot=n_boot, seed=seed)
                wins = int(np.sum(va < vb)) if better == "lower" else int(np.sum(va > vb))
                rows.append(
                    {
                        "method_a": a,
                        "method_b": b,
                        "metric": metric,
                        "better_is": better,
                        "mean_diff_a_minus_b": mean,
                        "ci_lo": lo,
                        "ci_hi": hi,
                        "ci_excludes_zero": bool(lo > 0 or hi < 0),
                        "n_pairs": len(seeds),
                        "seeds_a_better": wins,
                        "seeds_tied": int(np.sum(va == vb)),
                    }
                )
    return pd.DataFrame(rows)


def _log_to_mlflow(
    experiment: str,
    cfg: AppConfig,
    problem_name: str,
    budget: float,
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    files: list[Path],
    run_tags: dict[str, str],
) -> None:
    """Parent run per (problem, method) with one child run per seed."""
    import mlflow

    from formopt.tracking import (
        log_artifact_if_exists,
        log_campaign,
        log_metrics_finite,
        log_params_safe,
        setup_mlflow,
    )

    setup_mlflow(experiment)
    for method, g in metrics.groupby("method", sort=False):
        with mlflow.start_run(run_name=f"{problem_name}-{method}"):
            mlflow.set_tags({"problem": problem_name, "method": method, **run_tags})
            log_params_safe(
                {
                    "problem": problem_name,
                    "method": method,
                    "n_seeds": len(g),
                    "budget": budget,
                    **run_tags,
                }
            )
            srow = summary[summary.method == method].iloc[0].to_dict()
            log_metrics_finite({k: v for k, v in srow.items() if k != "method"})
            for path in files:
                log_artifact_if_exists(path)
            if cfg.source_path is not None:
                log_artifact_if_exists(cfg.source_path)
            for _, row in g.iterrows():
                log_campaign(
                    f"{problem_name}-{method}-seed{int(row.seed)}",
                    {"problem": problem_name, "method": method, "seed": int(row.seed)},
                    {
                        "final_regret": row.final_regret,
                        "spend_to_target": row.spend_to_target,
                        "experiments_to_target": row.experiments_to_target,
                        "n_experiments": row.n_experiments,
                        "total_spend": row.total_spend,
                        "success": float(row.success),
                        "wall_time_s": row.wall_time_s,
                        "n_gp_warnings": row.n_gp_warnings,
                    },
                )


def benchmark_problem(
    cfg: AppConfig,
    problem: Problem,
    budget: float,
    methods: list[str],
    seeds: list[int],
    out_dir: Path,
    n_jobs: int = 1,
    quick: bool = False,
    log_mlflow: bool = True,
    experiment_suffix: str = "",
    strategy_factory: Callable[[str], Strategy] | None = None,
    reference_cache: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Run every (method, seed) campaign on one problem and write all outputs.

    Writes `benchmark_<p>.csv` (one row per experiment), `campaign_metrics_<p>.csv` (one row
    per campaign), `summary_<p>.csv`, `paired_<p>.csv` and `benchmark_info_<p>.json`.
    Parallel runs (n_jobs > 1) rebuild named problems inside each worker process.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = problem.name
    tol = cfg.benchmark.target_tol
    ref = reference_optimum(problem, cfg.benchmark.reference, reference_cache)
    opt, cap = float(ref["cost"]), problem.cost_cap()
    factory = strategy_factory or (lambda m: make_strategy(m, cfg, name, quick))
    tasks = [(m, s) for m in methods for s in seeds]
    start = time.perf_counter()
    if n_jobs != 1 and strategy_factory is None:
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=n_jobs)(
            delayed(run_one)(cfg, name, m, s, quick) for m, s in tasks
        )
    else:
        results = [run_campaign(problem, factory(m), budget, s) for m, s in tasks]
    wall = time.perf_counter() - start
    long = pd.DataFrame([row for r in results for row in campaign_rows(r, problem)])
    metrics = pd.DataFrame([campaign_metrics(r, problem, opt, cap, tol) for r in results])
    n_boot, boot_seed = cfg.benchmark.n_boot, cfg.benchmark.bootstrap_seed
    summary = summarize(metrics, n_boot, boot_seed)
    paired = paired_differences(metrics, budget, n_boot, boot_seed)
    found = long.loc[long.truly_feasible, "material_cost"]
    best_found = float(found.min()) if len(found) else float("inf")
    z: dict[str, list[float]] = {s.output: [] for s in problem.specs}
    for r in results:
        for out, vals in calibration_z(r, problem).items():
            z[out].extend(vals)
    info = {
        "problem": name,
        "methods": methods,
        "seeds": seeds,
        "quick": quick,
        "n_candidates": cfg.bo.quick_n_candidates if quick else cfg.bo.n_candidates,
        "n_jobs": n_jobs,
        "budget": budget,
        "reference_cost": opt,
        "target_cost": opt * (1 + tol),
        "cost_cap": cap,
        "best_truly_feasible_cost_found_any_method": best_found,
        "reference_check_ok": bool(best_found >= opt - 1e-9),
        "wall_time_s": wall,
        "sum_campaign_wall_time_s": float(metrics["wall_time_s"].sum()),
        "headline_cost_method": cfg.benchmark.headline_cost_method,
        "calibration_z": {k: [float(v) for v in vals] for k, vals in z.items()},
    }
    if not info["reference_check_ok"]:
        logger.warning("A method beat the reference optimum on %s; reference is wrong", name)
    files = {
        "long": out_dir / f"benchmark_{name}.csv",
        "metrics": out_dir / f"campaign_metrics_{name}.csv",
        "summary": out_dir / f"summary_{name}.csv",
        "paired": out_dir / f"paired_{name}.csv",
    }
    long.to_csv(files["long"], index=False)
    metrics.to_csv(files["metrics"], index=False)
    summary.to_csv(files["summary"], index=False)
    paired.to_csv(files["paired"], index=False)
    info_path = out_dir / f"benchmark_info_{name}.json"
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    if log_mlflow:
        tags = {"quick": str(quick), "seeds": f"{min(seeds)}-{max(seeds)}"}
        _log_to_mlflow(
            f"benchmark-{name}{experiment_suffix}",
            cfg,
            name,
            budget,
            metrics,
            summary,
            [files["summary"], files["paired"], info_path],
            tags,
        )
    logger.info("%s: %d campaigns in %.1f s", name, len(results), wall)
    return {"long": long, "metrics": metrics, "summary": summary, "paired": paired}


def run_benchmark(
    cfg: AppConfig,
    problems: list[str],
    methods: list[str],
    seeds: list[int],
    n_jobs: int = 1,
    quick: bool = False,
    out_dir: Path | None = None,
    log_mlflow: bool = True,
    experiment_suffix: str = "",
) -> dict[str, dict[str, pd.DataFrame]]:
    """Run the benchmark for named problems (`enamel`, `concrete`)."""
    out_dir = Path(out_dir) if out_dir is not None else cfg.abs_paths.reports_dir
    return {
        name: benchmark_problem(
            cfg,
            _cached_problem(name, cfg),
            problem_budget(name, cfg),
            methods,
            seeds,
            out_dir,
            n_jobs=n_jobs,
            quick=quick,
            log_mlflow=log_mlflow,
            experiment_suffix=experiment_suffix,
            reference_cache=cfg.abs_paths.processed_dir,
        )
        for name in problems
    }
