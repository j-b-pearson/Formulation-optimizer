"""v2 commands (registered on the main `formopt` CLI). Thin wrappers over library functions.

run-cost-spread   constants of the run-cost family and the v1 effective spread (no runs)
v2-run            pre-registered runs: the run-cost spread sweep or the v1-settings run (H2)
runtime-estimate  total runtime estimate from the dev-seed timing runs, and the seed rule
repro-check       compare a fresh v1 rerun with reports/v1, exactly
v2-analyze        H1 and H2 analysis, robustness re-check, the two sweep figures
tolerance-curves  concrete success vs tolerance and budget (descriptive)
batch-compare     q-batch vs sequential BO on dev seeds (exploratory)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from formopt.config import AppConfig


def _write(data: dict, path: Path) -> Path:
    from formopt.cli import write_json

    return write_json(data, path)


def _rel(path: Path, cfg: AppConfig) -> str:
    """Path relative to the project root (POSIX style) when inside it, so outputs carry no
    machine-specific folders."""
    path = Path(path).resolve()
    try:
        return path.relative_to(cfg.root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def v2_seeds(cfg: AppConfig, dev: bool, n: int | None = None) -> list[int]:
    """Dev seeds 1001..1010 or reporting seeds 31..130 (or the first n of them)."""
    v2 = cfg.v2
    if dev:
        return list(range(v2.dev_seed_start, v2.dev_seed_start + (n or v2.n_dev_seeds)))
    return list(
        range(v2.reporting_seed_start, v2.reporting_seed_start + (n or v2.n_reporting_seeds))
    )


def cmd_run_cost_spread(cfg: AppConfig, args: argparse.Namespace) -> None:
    from formopt.heterogeneity import run_cost_spread_report

    report = run_cost_spread_report(cfg, list(cfg.v2.spread_levels))
    out = _write(report, cfg.abs_paths.v2_dir / "run_cost_spread.json")
    print(
        f"v1 effective spread s = {report['v1_effective_s']:.4f} "
        f"(v1 run cost ${report['v1_run_cost_min']:.2f} to ${report['v1_run_cost_max']:.2f})"
    )
    for row in report["levels"]:
        print(
            f"  s = {row['s']:g}: cheapest ${row['k_cheapest_run']:.2f}, most expensive "
            f"${row['most_expensive_run']:.2f}, mean ${row['mean_run_cost']:.2f}"
        )
    print(f"-> {out}")


def cmd_v2_run(cfg: AppConfig, args: argparse.Namespace) -> None:
    """Run the spread sweep or the v1-settings benchmark and record the wall time."""
    from formopt.benchmark import METHODS, run_benchmark
    from formopt.heterogeneity import run_sweep

    v2 = cfg.v2
    seeds = v2_seeds(cfg, args.dev, args.seeds)
    n_jobs = args.n_jobs if args.n_jobs is not None else v2.timing_n_jobs
    base = (
        Path(args.out_dir) if args.out_dir else cfg.abs_paths.v2_dir / ("dev" if args.dev else "")
    )
    tag = "-dev" if args.dev else ""
    log = not args.no_mlflow
    print(f"v2-run {args.what}: seeds {seeds[0]}..{seeds[-1]} ({len(seeds)}), n_jobs={n_jobs}")
    if args.what == "sweep":
        out = base / "step1_sweep"
        info = run_sweep(
            cfg,
            list(v2.spread_levels),
            list(v2.sweep_methods),
            seeds,
            out,
            n_jobs,
            log,
            mlflow_suffix=f"-v2{tag}",
        )
    else:
        out = base / "v1_settings"
        timing = {}
        start_all = time.perf_counter()
        for name in args.problem:
            start = time.perf_counter()
            run_benchmark(
                cfg,
                [name],
                list(METHODS),
                seeds,
                n_jobs=n_jobs,
                out_dir=out,
                log_mlflow=log,
                experiment_suffix=f"-v2-v1settings{tag}",
            )
            timing[name] = time.perf_counter() - start
        info = {
            "problems": args.problem,
            "methods": list(METHODS),
            "seeds": seeds,
            "n_jobs": n_jobs,
            "log_mlflow": log,
            "run_cost_spread": cfg.enamel.run_cost_spread.s,
            "wall_time_s_per_problem": timing,
            "wall_time_s_total": time.perf_counter() - start_all,
        }
        _write(info, out / "run_info.json")
    print(f"Done in {info['wall_time_s_total']:.1f} s -> {out}")


def cmd_runtime_estimate(cfg: AppConfig, args: argparse.Namespace) -> None:
    """Scale the dev-seed timing runs to the reporting seeds and apply the 6-hour rule."""
    v2 = cfg.v2
    dev = Path(args.dev_dir) if args.dev_dir else cfg.abs_paths.v2_dir / "dev"
    sweep = json.loads((dev / "step1_sweep" / "sweep_info.json").read_text(encoding="utf-8"))
    v1s = json.loads((dev / "v1_settings" / "run_info.json").read_text(encoding="utf-8"))
    scale = v2.n_reporting_seeds / len(sweep["seeds"])
    est_sweep = sweep["wall_time_s_total"] * scale
    est_v1s = v1s["wall_time_s_total"] * v2.n_reporting_seeds / len(v1s["seeds"])
    total_h = (est_sweep + est_v1s) / 3600.0
    cut = total_h > v2.runtime_limit_hours
    report = {
        "rule": f"If the estimated total exceeds {v2.runtime_limit_hours} h with --n-jobs "
        f"{v2.timing_n_jobs}, cut reporting seeds to {v2.reporting_seed_start}.."
        f"{v2.reporting_seed_start + v2.fallback_n_reporting_seeds - 1} (logged as a deviation).",
        "n_jobs": sweep["n_jobs"],
        "dev_seeds": sweep["seeds"],
        "dev_sweep_wall_s": sweep["wall_time_s_total"],
        "dev_v1_settings_wall_s": v1s["wall_time_s_total"],
        "scale_factor": scale,
        "estimated_sweep_s": est_sweep,
        "estimated_v1_settings_s": est_v1s,
        "estimated_total_hours": total_h,
        "limit_hours": v2.runtime_limit_hours,
        "cut_reporting_seeds": bool(cut),
        "reporting_seeds_to_use": v2.fallback_n_reporting_seeds if cut else v2.n_reporting_seeds,
    }
    out = _write(report, cfg.abs_paths.v2_dir / "runtime_estimate.json")
    print(
        f"Estimated total {total_h:.2f} h (limit {v2.runtime_limit_hours} h); cut: {cut} -> {out}"
    )


def cmd_repro_check(cfg: AppConfig, args: argparse.Namespace) -> None:
    from formopt.repro import compare_dirs

    report = compare_dirs(
        Path(args.reference or cfg.abs_paths.reports_dir), Path(args.candidate), args.problem
    )
    report["reference"] = _rel(Path(report["reference"]), cfg)
    report["candidate"] = _rel(Path(report["candidate"]), cfg)
    out = _write(report, Path(args.out) if args.out else Path(args.candidate) / "repro_check.json")
    for name, files in report["problems"].items():
        for stem, res in files.items():
            print(f"  {name} {stem}: identical = {res['identical']}")
    print(f"All identical: {report['all_identical']} -> {out}")


def cmd_v2_analyze(cfg: AppConfig, args: argparse.Namespace) -> None:
    """H1 / H2 analysis with the pre-registered bootstrap, the robustness re-check, figures."""
    from formopt.heterogeneity import analyze, level_name, robustness_check, run_cost_spread_report
    from formopt.plotting_v2 import plot_heterogeneity_advantage, plot_spend_vs_best_by_level

    v2dir = cfg.abs_paths.v2_dir
    sweep = Path(args.sweep_dir) if args.sweep_dir else v2dir / "step1_sweep"
    v1s = Path(args.v1_settings_dir) if args.v1_settings_dir else v2dir / "v1_settings"
    out = Path(args.out_dir) if args.out_dir else v2dir / "analysis"
    result = analyze(cfg, sweep, v1s, out)
    robust = robustness_check(cfg, sweep, v1s, result)
    _write(robust, out / "robustness.json")
    levels = pd.read_csv(out / "h1_per_level.csv")
    v1_point = (
        pd.read_csv(out / "v1_settings_advantage.csv")
        if (out / "v1_settings_advantage.csv").exists()
        else None
    )
    s_v1 = run_cost_spread_report(cfg, [1.0])["v1_effective_s"]
    seeds = result["seeds"]
    budget = cfg.enamel.budget
    sub = f"Seeds {seeds[0]} to {seeds[-1]} ({len(seeds)} paired seeds), budget ${budget:,.0f}"
    figs = v2dir / "figures"
    plot_heterogeneity_advantage(
        levels[levels.metric.isin(["spend_saved", "success_gain"])],
        figs / "heterogeneity_advantage.png",
        v1_point,
        s_v1,
        cfg.v2.h1_headline_method,
        sub,
    )
    longs, info = {}, None
    for s in cfg.v2.spread_levels:
        folder = sweep / level_name(s)
        longs[float(s)] = pd.read_csv(folder / "benchmark_enamel.csv")
        info = json.loads((folder / "benchmark_info_enamel.json").read_text(encoding="utf-8"))
    plot_spend_vs_best_by_level(
        longs,
        info["reference_cost"],
        info["cost_cap"],
        info["budget"],
        figs / "spend_vs_best_by_spread.png",
        tol=cfg.benchmark.target_tol,
        subtitle=sub,
    )
    print(json.dumps({"h1": result["h1"], "h2": result.get("h2")}, indent=2))
    print(
        f"Robustness ({robust['n_boot']} resamples): H1 verdicts unchanged "
        f"{robust['h1_verdicts_unchanged']}, H2 unchanged {robust['h2_verdict_unchanged']}"
    )


def cmd_tolerance_curves(cfg: AppConfig, args: argparse.Namespace) -> None:
    """Concrete success vs tolerance and vs experiment budget (descriptive)."""
    from formopt.plotting_v2 import plot_success_vs_budget, plot_success_vs_tolerance
    from formopt.tolerance import gap_summary, success_curves

    v2dir = cfg.abs_paths.v2_dir
    sources = {
        "v1": Path(args.v1_dir) if args.v1_dir else cfg.abs_paths.reports_dir,
        "v2": Path(args.v2_dir) if args.v2_dir else v2dir / "v1_settings",
    }
    longs, used, ref = [], {}, None
    for label, folder in sources.items():
        path = folder / "benchmark_concrete.csv"
        if not path.exists():
            print(f"No {path}; skipping source {label}")
            continue
        info = json.loads((folder / "benchmark_info_concrete.json").read_text(encoding="utf-8"))
        if ref is not None and abs(info["reference_cost"] - ref) > 1e-9:
            raise SystemExit("Sources disagree on the reference optimum; not pooling them")
        ref = info["reference_cost"]
        long = pd.read_csv(path)
        long["source"] = label
        longs.append(long)
        seeds = sorted(int(s) for s in long.seed.unique())
        used[label] = {"folder": _rel(folder, cfg), "seeds": seeds}
    pooled = pd.concat(longs, ignore_index=True)
    pooled["seed"] = pooled["source"] + "-" + pooled["seed"].astype(str)
    tols = list(cfg.v2.tolerances)
    curves = [success_curves(pooled, ref, tols, "pooled")]
    for long in longs:
        curves.append(success_curves(long, ref, tols, str(long["source"].iloc[0])))
    table = pd.concat(curves, ignore_index=True)
    out = Path(args.out_dir) if args.out_dir else v2dir / "step3_concrete"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "success_curves_concrete.csv", index=False)
    gaps = gap_summary(pooled, ref)
    gaps.to_csv(out / "gap_to_optimum_concrete.csv", index=False)
    n_seeds = int(pooled.groupby("method")["seed"].nunique().iloc[0])
    _write(
        {
            "label": "DESCRIPTIVE: no new claims",
            "sources": used,
            "n_seeds_pooled_per_method": n_seeds,
            "reference_cost": ref,
            "tolerances": tols,
            "target_cost_at_tolerance": {f"{t:g}": ref * (1 + t) for t in tols},
            "best_in_spec_cost_any_campaign": float(
                pooled.loc[pooled.truly_feasible.astype(bool), "material_cost"].min()
            ),
        },
        out / "step3_info.json",
    )
    pooled_curves = table[table.source == "pooled"]
    seeds_txt = " + ".join(f"{k} seeds {v['seeds'][0]}-{v['seeds'][-1]}" for k, v in used.items())
    sub = f"{n_seeds} seeds per method ({seeds_txt}), budget ${cfg.concrete.budget:,.0f}"
    figs = v2dir / "figures"
    plot_success_vs_tolerance(pooled_curves, figs / "concrete_success_vs_tolerance.png", sub)
    plot_success_vs_budget(pooled_curves, figs / "concrete_success_vs_budget.png", sub)
    print(gaps.to_string(index=False))
    print(f"-> {out}")


def cmd_batch_compare(cfg: AppConfig, args: argparse.Namespace) -> None:
    """q-batch BO vs sequential BO on enamel (exploratory, dev seeds only)."""
    from joblib import Parallel, delayed

    from formopt.benchmark import campaign_metrics
    from formopt.metrics import bootstrap_ci, wilson_ci
    from formopt.problems import build_problem, problem_budget
    from formopt.problems.base import reference_optimum

    v2 = cfg.v2
    seeds = v2_seeds(cfg, dev=True)
    q = v2.batch.q
    methods = [m for base in v2.batch.methods for m in (base, f"{base}_q{q}")]
    problem = build_problem("enamel", cfg)
    budget = problem_budget("enamel", cfg)
    ref = reference_optimum(problem, cfg.benchmark.reference, cfg.abs_paths.processed_dir)
    opt, cap, tol = float(ref["cost"]), problem.cost_cap(), cfg.benchmark.target_tol
    n_jobs = args.n_jobs if args.n_jobs is not None else v2.timing_n_jobs
    start = time.perf_counter()
    runs = Parallel(n_jobs=n_jobs)(delayed(_batch_one)(cfg, m, s) for m in methods for s in seeds)
    wall = time.perf_counter() - start
    rows = []
    for result, n_cycles in runs:
        row = campaign_metrics(result, problem, opt, cap, tol)
        row["n_cycles"] = n_cycles
        rows.append(row)
    metrics = pd.DataFrame(rows)
    summary = []
    for method, g in metrics.groupby("method", sort=False):
        p, lo, hi = wilson_ci(int(g["success"].sum()), len(g))
        reg, rlo, rhi = bootstrap_ci(g["final_regret"].to_numpy(float), seed=v2.bootstrap_seed)
        summary.append(
            {
                "method": method,
                "n_seeds": len(g),
                "success_rate": p,
                "success_ci_lo": lo,
                "success_ci_hi": hi,
                "mean_final_regret": reg,
                "regret_ci_lo": rlo,
                "regret_ci_hi": rhi,
                "mean_n_experiments": float(g["n_experiments"].mean()),
                "mean_n_cycles": float(g["n_cycles"].mean()),
                "mean_wall_time_s": float(g["wall_time_s"].mean()),
            }
        )
    out = Path(args.out_dir) if args.out_dir else cfg.abs_paths.v2_dir / "step5_batch_dev"
    out.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out / "campaign_metrics_enamel.csv", index=False)
    pd.DataFrame(summary).to_csv(out / "summary_enamel.csv", index=False)
    _write(
        {
            "label": "EXPLORATORY, dev seeds only",
            "seeds": seeds,
            "q": q,
            "methods": methods,
            "budget": budget,
            "reference_cost": opt,
            "n_jobs": n_jobs,
            "wall_time_s": wall,
        },
        out / "batch_info.json",
    )
    print(pd.DataFrame(summary).round(4).to_string(index=False))
    print(f"-> {out}")


def _batch_one(cfg: AppConfig, method: str, seed: int):
    """One campaign in a worker process; returns the result and the oven cycles it used."""
    from formopt.benchmark import _cached_problem, make_strategy
    from formopt.loop import run_campaign
    from formopt.problems import problem_budget

    problem = _cached_problem("enamel", cfg)
    strategy = make_strategy(method, cfg, "enamel")
    result = run_campaign(problem, strategy, problem_budget("enamel", cfg), seed)
    return result, int(getattr(strategy, "n_cycles", result.n_experiments))


def add_v2_commands(sub: argparse._SubParsersAction) -> None:
    """Register the v2 subcommands on the main parser."""
    from formopt.problems import PROBLEM_NAMES

    p = sub.add_parser("run-cost-spread", help="v2: run-cost family constants (no runs)")
    p.set_defaults(func=cmd_run_cost_spread)

    p = sub.add_parser("v2-run", help="v2: pre-registered runs (spread sweep or v1 settings)")
    p.add_argument("--what", choices=["sweep", "v1-settings"], required=True)
    p.add_argument("--dev", action="store_true", help="Dev seeds 1001..1010 (timing, debugging)")
    p.add_argument("--seeds", type=int, default=None, help="Use only the first N seeds")
    p.add_argument("--problem", nargs="+", choices=PROBLEM_NAMES, default=list(PROBLEM_NAMES))
    p.add_argument("--n-jobs", type=int, default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--no-mlflow", action="store_true")
    p.set_defaults(func=cmd_v2_run)

    p = sub.add_parser("runtime-estimate", help="v2: runtime estimate from the dev timing runs")
    p.add_argument("--dev-dir", default=None)
    p.set_defaults(func=cmd_runtime_estimate)

    p = sub.add_parser("repro-check", help="v2: compare a v1 rerun with reports/v1 exactly")
    p.add_argument("--candidate", required=True)
    p.add_argument("--reference", default=None)
    p.add_argument("--problem", nargs="+", choices=PROBLEM_NAMES, default=list(PROBLEM_NAMES))
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_repro_check)

    p = sub.add_parser("v2-analyze", help="v2: H1 and H2 analysis and sweep figures")
    p.add_argument("--sweep-dir", default=None)
    p.add_argument("--v1-settings-dir", default=None)
    p.add_argument("--out-dir", default=None)
    p.set_defaults(func=cmd_v2_analyze)

    p = sub.add_parser("tolerance-curves", help="v2: concrete tolerance curves (descriptive)")
    p.add_argument("--v1-dir", default=None)
    p.add_argument("--v2-dir", default=None)
    p.add_argument("--out-dir", default=None)
    p.set_defaults(func=cmd_tolerance_curves)

    p = sub.add_parser("batch-compare", help="v2: q-batch vs sequential (dev seeds)")
    p.add_argument("--n-jobs", type=int, default=None)
    p.add_argument("--out-dir", default=None)
    p.set_defaults(func=cmd_batch_compare)
