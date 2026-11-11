"""Command line interface: `uv run python -m formopt <command>`.

Each subcommand is a thin wrapper over a library function so the same logic is usable from
tests, notebooks and the Streamlit app.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from formopt.config import AppConfig, load_config

# MLflow prints an "agent hint" banner on import; it is not a warning.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
from formopt.problems import PROBLEM_NAMES, build_problem


def _json_default(obj: object) -> object:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def write_json(data: dict, path: Path) -> Path:
    """Write a dict as pretty JSON (numpy values converted)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=_json_default), encoding="utf-8")
    return path


def cmd_reference_opt(cfg: AppConfig, args: argparse.Namespace) -> None:
    """Estimate the reference optimum and write the calibration checks to reports/."""
    from formopt.problems.base import calibration_report, reference_optimum

    paths = cfg.abs_paths
    for name in args.problem:
        problem = build_problem(name, cfg)
        ref = reference_optimum(
            problem, cfg.benchmark.reference, paths.processed_dir, refresh=args.refresh
        )
        report = calibration_report(problem, ref)
        if hasattr(problem, "calibration_extras"):
            report.update(problem.calibration_extras())
        out = write_json(report, paths.reports_dir / f"calibration_{name}.json")
        print(f"== {name} calibration ({out})")
        print(f"  feasible fraction of known-feasible box: {report['feasible_fraction']:.3f}")
        print(f"  run cost range: ${report['run_cost_min']:.0f} to ${report['run_cost_max']:.0f}")
        print(f"  run cost max/min ratio: {report['run_cost_ratio_max_min']:.2f}")
        print(f"  reference (cheapest truly feasible) cost: {report['reference_cost']:.4f}")
        print(f"  active specs within 2% of a limit: {report['active_specs_within_2pct']}")
        print(f"  at a corner on every variable: {report['optimum_at_corner_on_every_variable']}")
        for k, v in report["reference_recipe"].items():
            print(f"    {k:>18s} = {v:.4f}")


def cmd_download(cfg: AppConfig, args: argparse.Namespace) -> None:
    """Download the UCI concrete data (OpenML fallback) and write the cleaned CSV."""
    from formopt.data import download_concrete, load_concrete

    paths = cfg.abs_paths
    path = download_concrete(paths.raw_dir / "concrete", timeout=args.timeout, force=args.force)
    df = load_concrete(paths.raw_dir / "concrete", processed_dir=paths.processed_dir)
    print(f"Concrete: {path}")
    print(f"{len(df)} clean rows, {df['mix_id'].nunique()} unique mixes")


def cmd_train_emulator(cfg: AppConfig, args: argparse.Namespace) -> None:
    """Fit the concrete emulator, validate it with grouped CV, save it and log to MLflow."""
    import mlflow

    from formopt.data import TARGET, clean_concrete, read_concrete_raw
    from formopt.emulator import sanity_checks, train_concrete_emulator
    from formopt.plotting import plot_emulator_parity
    from formopt.problems.concrete import emulator_path
    from formopt.tracking import (
        log_artifact_if_exists,
        log_metrics_finite,
        log_params_safe,
        setup_mlflow,
    )

    paths = cfg.abs_paths
    raw = read_concrete_raw(paths.raw_dir / "concrete")
    df, n_dropped = clean_concrete(raw, round_to=cfg.concrete.mix_id_round)
    paths.processed_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(paths.processed_dir / "concrete.csv", index=False)
    emu = train_concrete_emulator(df, cfg.concrete)
    model_file = emu.save(emulator_path(cfg))
    cv = emu.cv_report
    checks = sanity_checks(emu, seed=cfg.concrete.emulator.seed)
    noise_sd = float(min(cv["residual_sd"], cfg.concrete.noise_sd_cap))
    validation = {
        "emulator_kind": emu.kind,
        "features": emu.feature_names,
        "grouped_cv_folds": cfg.concrete.emulator.cv_folds,
        "grouped_by": "mix_id",
        "n_rows": int(len(df)),
        "n_exact_duplicates_dropped": int(n_dropped),
        "n_unique_mixes": int(df["mix_id"].nunique()),
        "r2": cv["r2"],
        "rmse": cv["rmse"],
        "mae": cv["mae"],
        "residual_sd": cv["residual_sd"],
        "r2_target": 0.85,
        "r2_target_met": cv["r2"] >= 0.85,
        "folds": cv["folds"],
        "model_comparison": emu.model_comparison,
        "noise_sd_used_in_problem": noise_sd if cfg.concrete.noise_sd is None else None,
        "domain_k": cfg.concrete.domain.k,
        "domain_quantile": cfg.concrete.domain.quantile,
        "domain_threshold": emu.domain.threshold,
        "total_mass_range": list(emu.total_mass_range),
        "wb_range": list(emu.wb_range),
        "sanity_checks": checks,
    }
    out_json = write_json(validation, paths.reports_dir / "emulator_validation.json")
    parity = plot_emulator_parity(
        df[TARGET].to_numpy(float),
        cv["oof_pred"],
        paths.figures_dir / "emulator_parity_concrete.png",
        cv["r2"],
        cv["rmse"],
    )
    setup_mlflow("concrete-emulator")
    with mlflow.start_run(run_name=f"emulator-{emu.kind}"):
        log_params_safe(
            {
                "kind": emu.kind,
                "n_estimators": cfg.concrete.emulator.n_estimators,
                "min_samples_leaf": cfg.concrete.emulator.min_samples_leaf,
                "cv_folds": cfg.concrete.emulator.cv_folds,
                "features": ",".join(emu.feature_names),
                "n_rows": len(df),
                "n_unique_mixes": int(df["mix_id"].nunique()),
            }
        )
        metrics = {"cv_r2": cv["r2"], "cv_rmse": cv["rmse"], "cv_mae": cv["mae"]}
        for kind, rep in emu.model_comparison.items():
            metrics.update({f"{kind}_cv_r2": rep["r2"], f"{kind}_cv_rmse": rep["rmse"]})
        log_metrics_finite(metrics)
        for path in (out_json, parity, model_file):
            log_artifact_if_exists(path)
        if cfg.source_path is not None:
            log_artifact_if_exists(cfg.source_path)
    print(f"Emulator: {emu.kind} (chosen by grouped-CV RMSE over {list(emu.model_comparison)})")
    print(f"Grouped 5-fold CV: R2 = {cv['r2']:.3f}, RMSE = {cv['rmse']:.2f}, MAE = {cv['mae']:.2f}")
    if cv["r2"] < 0.85:
        print("WARNING: grouped-CV R2 below the 0.85 target; see README limitations")
    print(f"Strength non-decreasing 7->28->90 d: {checks['age_mixes_nondecreasing_7_28_90']}/10")
    print(f"PD on w/b decreasing steps: {checks['pd_wb_decreasing_steps']}/8")
    print(f"Wrote {out_json}, {parity}, {model_file}")


def gp_holdout_coverage(
    cfg: AppConfig, problem_name: str, n_train: int, n_test: int, seeds: list[int]
) -> dict:
    """Fit one GP per output on an LHS design, check 95% interval coverage on held-out points.

    Predictive intervals include the learned lab noise and are compared with noisy held-out
    measurements (what a lab would see). Latent intervals are compared with the noise-free
    truth, for information.
    """
    from formopt.design import lhs_design, repair_to_feasible, sample_known_feasible
    from formopt.gp import fit_gp
    from formopt.metrics import interval_coverage

    problem = build_problem(problem_name, cfg)
    bo = cfg.bo
    per_seed = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        pool = sample_known_feasible(problem, 4000, rng)
        X_train = lhs_design(problem.bounds, n_train, rng)
        X_train = repair_to_feasible(X_train, pool, problem.known_feasible(X_train), problem.bounds)
        X_test = sample_known_feasible(problem, n_test, rng)
        obs_train = [
            problem.observe(x, np.random.default_rng([seed, 0, i])) for i, x in enumerate(X_train)
        ]
        obs_test = [
            problem.observe(x, np.random.default_rng([seed, 1, i])) for i, x in enumerate(X_test)
        ]
        truth = problem.true_outputs(X_test)
        row: dict[str, float] = {"seed": seed}
        for spec in problem.specs:
            y_tr = np.array([o[spec.output] for o in obs_train])
            gp = fit_gp(
                X_train,
                y_tr,
                problem.bounds,
                spec.transform,
                bo.gp_restarts,
                seed,
                bo.gp_length_scale_bounds,
                bo.gp_constant_bounds,
                bo.gp_noise_bounds,
            )
            y_te = spec.transform_values(np.array([o[spec.output] for o in obs_test]))
            mu, sd_pred = gp.predict(X_test, include_noise=True)
            _, sd_lat = gp.predict(X_test, include_noise=False)
            row[f"{spec.output}_predictive_coverage"] = interval_coverage(y_te, mu, sd_pred)
            row[f"{spec.output}_latent_coverage_vs_truth"] = interval_coverage(
                spec.transform_values(truth[spec.output]), mu, sd_lat
            )
        per_seed.append(row)
    keys = [k for k in per_seed[0] if k != "seed"]
    return {
        "problem": problem_name,
        "n_train_lhs": n_train,
        "n_test": n_test,
        "seeds": seeds,
        "per_seed": per_seed,
        "mean": {k: float(np.mean([r[k] for r in per_seed])) for k in keys},
        "target_range": [0.85, 0.99],
    }


def cmd_gp_coverage(cfg: AppConfig, args: argparse.Namespace) -> None:
    """Hold-out calibration check of the GP surrogate."""
    report = gp_holdout_coverage(
        cfg, args.problem, args.n_train, args.n_test, list(range(args.seeds))
    )
    out = write_json(report, cfg.abs_paths.reports_dir / f"gp_holdout_coverage_{args.problem}.json")
    print(
        f"GP hold-out 95% interval coverage ({args.problem}, {args.n_train} LHS train points, "
        f"{args.n_test} test points, {args.seeds} seeds) -> {out}"
    )
    for key, value in report["mean"].items():
        print(f"  {key:>45s}: {value:.3f}")


SUMMARY_COLUMNS = [
    "method",
    "n_seeds",
    "success_rate",
    "success_ci_lo",
    "success_ci_hi",
    "median_spend_to_target",
    "mean_final_regret",
    "mean_regret_ci_lo",
    "mean_regret_ci_hi",
    "mean_n_experiments",
]


def cmd_benchmark(cfg: AppConfig, args: argparse.Namespace) -> None:
    """Run the seeded benchmark and print the summary tables."""
    import pandas as pd

    from formopt.benchmark import METHODS, run_benchmark

    bench = cfg.benchmark
    problems = list(PROBLEM_NAMES) if "all" in args.problem else args.problem
    if args.dev:
        seeds = list(range(bench.dev_seed_start, bench.dev_seed_start + bench.n_dev_seeds))
        sub, suffix = "dev", "-dev"
    else:
        n = args.seeds or (bench.quick_seeds if args.quick else bench.n_seeds)
        start = bench.seed_start if args.seed_start is None else args.seed_start
        seeds = list(range(start, start + n))
        sub, suffix = ("quick", "-quick") if args.quick else ("", "")
    out_dir = Path(args.out_dir) if args.out_dir else cfg.abs_paths.reports_dir / sub
    methods = args.methods or list(METHODS)
    n_jobs = args.n_jobs if args.n_jobs is not None else bench.n_jobs
    if args.run_cost_spread is not None:  # v2: enamel run-cost spread s (v1 formula if unset)
        from formopt.heterogeneity import level_name, with_run_cost_spread

        if problems != ["enamel"]:
            raise SystemExit("--run-cost-spread applies to the enamel problem only")
        cfg = with_run_cost_spread(cfg, args.run_cost_spread)
        # Never write a spread run into reports/v1 (frozen v1 outputs) or its MLflow experiment.
        level = level_name(args.run_cost_spread)
        if not args.out_dir:
            out_dir = cfg.abs_paths.v2_dir / sub / level
        suffix = f"{suffix}-v2-{level}"
    if args.mlflow_suffix is not None:
        suffix = args.mlflow_suffix
    print(
        f"Benchmark: problems={problems} methods={methods} seeds={seeds[0]}..{seeds[-1]} "
        f"({len(seeds)}) quick={args.quick} n_jobs={n_jobs} "
        f"run_cost_spread={cfg.enamel.run_cost_spread.s} -> {out_dir}"
    )
    out = run_benchmark(
        cfg,
        problems,
        methods,
        seeds,
        n_jobs=n_jobs,
        quick=args.quick,
        out_dir=out_dir,
        log_mlflow=not args.no_mlflow,
        experiment_suffix=suffix,
    )
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        for name, frames in out.items():
            print(f"\n== {name} summary ==")
            print(frames["summary"][SUMMARY_COLUMNS].round(4).to_string(index=False))


def _load_benchmark(source: Path, name: str):
    """Long-format CSV, campaign metrics and info JSON written by `benchmark`."""
    import pandas as pd

    long_path = source / f"benchmark_{name}.csv"
    if not long_path.exists():
        return None
    long = pd.read_csv(long_path)
    metrics = pd.read_csv(source / f"campaign_metrics_{name}.csv")
    info = json.loads((source / f"benchmark_info_{name}.json").read_text(encoding="utf-8"))
    return long, metrics, info


def _figures_dir(cfg: AppConfig, source: Path) -> Path:
    """reports/figures for the main results; <source>/figures for quick or dev runs, so a
    quick run never overwrites the committed full-run figures."""
    if source.resolve() == cfg.abs_paths.reports_dir.resolve():
        return cfg.abs_paths.figures_dir
    return source / "figures"


def _run_label(info: dict) -> str:
    mode = "quick mode" if info["quick"] else "full mode"
    return f"{len(info['seeds'])} seeds, {mode}, budget ${info['budget']:,.0f}"


def price_shock_report(
    cfg: AppConfig,
    name: str,
    changes: dict[str, float],
    source: Path,
    method: str | None = None,
    seed: int | None = None,
    tag: str = "",
) -> dict | None:
    """Refit GPs on one finished campaign and re-optimize under new prices; write outputs."""
    from formopt.benchmark import gp_settings
    from formopt.plotting import plot_price_shock
    from formopt.price_shock import (
        history_from_long,
        median_seed,
        price_shock_table,
        run_price_shock,
    )

    loaded = _load_benchmark(source, name)
    if loaded is None:
        print(f"No benchmark results for {name} in {source}; run the benchmark first")
        return None
    long, metrics, info = loaded
    problem = build_problem(name, cfg)
    method = method or cfg.benchmark.headline_cost_method
    seed = median_seed(metrics, method) if seed is None else seed
    history = history_from_long(long, problem, method, seed)
    result = run_price_shock(problem, history, changes, cfg, gp_settings(cfg))
    result.update({"source_method": method, "source_seed": seed, "source_run": _run_label(info)})
    suffix = f"_{tag}" if tag else ""
    reports = source  # outputs sit next to the benchmark files they were made from
    write_json(result, reports / f"price_shock_{name}{suffix}.json")
    price_shock_table(result).to_csv(reports / f"price_shock_{name}{suffix}.csv", index=False)
    change_text = ", ".join(f"{k} x{v:g}" for k, v in changes.items())
    plot_price_shock(
        result,
        problem,
        _figures_dir(cfg, source) / f"price_shock_{name}{suffix}.png",
        f"{name.capitalize()} price shock ({change_text}); data: {method} seed {seed}, "
        f"{len(history)} experiments",
    )
    print(
        f"== {name} price shock ({change_text}); {len(history)} experiments from "
        f"{method} seed {seed}; zero new experiments"
    )
    for label in ("old", "new"):
        rec = result[label]
        recipe = ", ".join(f"{k}={v:.4g}" for k, v in rec["recipe"].items())
        print(
            f"  {label} prices: cost {rec['material_cost']:.4f}, PoF {rec['pof']:.3f}, "
            f"truly feasible {rec['truly_feasible']}; {recipe}"
        )
    print(f"  old recipe at new prices: {result['old_recipe_cost_at_new_prices']:.4f}")
    return result


def cmd_price_shock(cfg: AppConfig, args: argparse.Namespace) -> None:
    """Re-optimize under changed prices with zero new experiments."""
    from formopt.price_shock import parse_change

    problem = build_problem(args.problem, cfg)
    if args.change:
        changes = dict(parse_change(c, problem.prices) for c in args.change)
    else:
        defaults = cfg.price_shock
        changes = defaults.enamel_changes if args.problem == "enamel" else defaults.concrete_changes
        changes = dict(list(changes.items())[:1])
    source = Path(args.source) if args.source else cfg.abs_paths.reports_dir
    price_shock_report(cfg, args.problem, changes, source, args.method, args.seed, args.tag)


def cmd_figures(cfg: AppConfig, args: argparse.Namespace) -> None:
    """Write every README figure from the benchmark outputs."""
    from formopt.benchmark import gp_settings
    from formopt.plotting import (
        plot_calibration,
        plot_gp_1d_slices,
        plot_pof_heatmap,
        plot_regret_log,
        plot_spend_vs_best,
        plot_success_vs_spend,
    )
    from formopt.price_shock import history_from_long, median_seed
    from formopt.strategies import best_passed_cost, fit_surrogates

    source = Path(args.source) if args.source else cfg.abs_paths.reports_dir
    figs = _figures_dir(cfg, source)
    tol = cfg.benchmark.target_tol
    written = []
    for name in PROBLEM_NAMES:
        loaded = _load_benchmark(source, name)
        if loaded is None:
            print(f"Skipping {name}: no benchmark_{name}.csv in {source}")
            continue
        long, metrics, info = loaded
        opt, cap, budget = info["reference_cost"], info["cost_cap"], info["budget"]
        label = _run_label(info)
        kw = {"problem": name, "subtitle": label}
        written.append(
            plot_spend_vs_best(
                long, opt, cap, budget, figs / f"spend_vs_best_{name}.png", tol=tol, **kw
            )
        )
        written.append(
            plot_success_vs_spend(
                long, opt, cap, budget, figs / f"success_vs_spend_{name}.png", tol=tol, **kw
            )
        )
        written.append(
            plot_regret_log(long, opt, cap, budget, figs / f"regret_log_{name}.png", **kw)
        )
        written.append(
            plot_calibration(
                info["calibration_z"],
                figs / f"calibration_{name}.png",
                f"{name.capitalize()}: GP next-point calibration, all BO campaigns ({label})",
            )
        )
        if name == "enamel":
            problem = build_problem(name, cfg)
            method = cfg.benchmark.headline_cost_method
            seed = median_seed(metrics, method)
            history = history_from_long(long, problem, method, seed)
            gps = fit_surrogates(problem, history, gp_settings(cfg), seed=cfg.price_shock.seed)
            best = best_passed_cost(history)
            passed = [o for o in history if o.passed_spec and o.material_cost == best]
            x_ref = passed[0].x if passed else history[-1].x
            written.append(
                plot_gp_1d_slices(gps, x_ref, problem, figs / "gp_slices_enamel.png", history)
            )
            written.append(
                plot_pof_heatmap(
                    gps,
                    problem,
                    x_ref,
                    "oven_temp_c",
                    "line_speed_m_min",
                    figs / "pof_heatmap_enamel.png",
                )
            )
            write_json(
                {
                    "method": method,
                    "seed": seed,
                    "x_ref": problem.as_dict(x_ref),
                    "source_run": label,
                },
                source / "surrogate_figures_enamel.json",
            )
    if not args.skip_price_shock:
        ps = cfg.price_shock
        for key, mult in ps.enamel_changes.items():
            tag = "" if key == "pai_resin" else key
            price_shock_report(cfg, "enamel", {key: mult}, source, tag=tag)
        price_shock_report(cfg, "concrete", dict(ps.concrete_changes), source)
    parity = cfg.abs_paths.figures_dir / "emulator_parity_concrete.png"
    if not parity.exists():
        print("Missing emulator_parity_concrete.png: run `uv run python -m formopt train-emulator`")
    for path in written:
        print(f"Wrote {path}")


def cmd_latency(cfg: AppConfig, args: argparse.Namespace) -> None:
    """Time one BO suggestion (the app's 'Suggest next experiment') after the initial design."""
    import time

    from formopt.benchmark import gp_settings, make_strategy
    from formopt.loop import make_observation, noise_rng
    from formopt.problems import problem_budget
    from formopt.strategies import bo_step

    report = {}
    for name in args.problem:
        problem = build_problem(name, cfg)
        budget = problem_budget(name, cfg)
        rng = np.random.default_rng(0)
        strategy = make_strategy("bo_cost", cfg, name)  # only used for its initial design
        strategy.reset(problem, budget, rng)
        history = [
            make_observation(problem, x, noise_rng(0, i)) for i, x in enumerate(strategy.queue)
        ]
        spent = sum(o.run_cost for o in history)
        times = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            sug = bo_step(
                problem, history, spent, budget, rng, cfg.bo.n_candidates, 1.0, gp_settings(cfg)
            )
            times.append(time.perf_counter() - start)
            history.append(make_observation(problem, sug.x, noise_rng(0, len(history))))
            spent += history[-1].run_cost
        report[name] = {
            "n_candidates": cfg.bo.n_candidates,
            "seconds": times,
            "median_s": float(np.median(times)),
            "max_s": float(np.max(times)),
        }
        print(f"{name}: median {np.median(times):.3f} s, max {np.max(times):.3f} s per suggestion")
    write_json(report, cfg.abs_paths.reports_dir / "suggestion_latency.json")


def cmd_lab_note(cfg: AppConfig, args: argparse.Namespace) -> None:
    """Draft a lab note for one BO suggestion with a local LLM (Ollama), checked in code.

    The risk label comes from the probability of passing all specs and the
    thresholds in config.yaml; a note whose label or numbers differ from the facts is rejected
    (one retry), then a template note is used. Writes reports/v2/lab_note_check_<problem>.json.
    """
    import os
    import time

    from formopt.benchmark import gp_settings, make_strategy
    from formopt.llm import checked_lab_note, llm_settings, suggestion_facts
    from formopt.loop import make_observation, noise_rng
    from formopt.problems import problem_budget
    from formopt.strategies import bo_step

    problem = build_problem(args.problem, cfg)
    budget = problem_budget(args.problem, cfg)
    rng = np.random.default_rng(args.seed)
    strategy = make_strategy("bo_cost", cfg, args.problem)  # only used for its initial design
    strategy.reset(problem, budget, rng)
    history = [
        make_observation(problem, x, noise_rng(args.seed, i)) for i, x in enumerate(strategy.queue)
    ]
    spent = sum(o.run_cost for o in history)
    sug = bo_step(problem, history, spent, budget, rng, cfg.bo.n_candidates, 1.0, gp_settings(cfg))
    th = cfg.lab_note.risk_thresholds
    facts = suggestion_facts(problem, sug, th.high_below, th.low_at_or_above)
    _, model, _ = llm_settings()
    effort = os.environ.get("LLM_REASONING_EFFORT", "none") or None
    start = time.perf_counter()
    checked = checked_lab_note(
        facts, model=model, max_retries=cfg.lab_note.max_retries, reasoning_effort=effort
    )
    elapsed = time.perf_counter() - start
    out_path = (
        Path(args.out) if args.out else cfg.abs_paths.v2_dir / f"lab_note_check_{args.problem}.json"
    )
    out = write_json(
        {
            "model": model,
            "reasoning_effort": effort,
            "seconds": elapsed,
            "facts_sent": facts,
            "risk_label_computed_in_code": facts["risk_label"],
            "source": checked.source,
            "attempts": checked.attempts,
            "fallback_reason": checked.fallback_reason,
            "lab_note": checked.note.model_dump(),
            "label": "LLM-drafted text for illustration; not a measured result",
        },
        out_path,
    )
    print(json.dumps(checked.note.model_dump(), indent=2))
    print(f"(source: {checked.source}; {model}, {elapsed:.1f} s) -> {out}")


def build_parser() -> argparse.ArgumentParser:
    """Create the argparse parser with all subcommands."""
    parser = argparse.ArgumentParser(prog="formopt", description=__doc__)
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download", help="Download the UCI concrete data")
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument("--force", action="store_true", help="Download again even if present")
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("train-emulator", help="Fit and validate the concrete emulator")
    p.set_defaults(func=cmd_train_emulator)

    p = sub.add_parser("benchmark", help="Run the seeded method comparison")
    p.add_argument("--problem", nargs="+", choices=[*PROBLEM_NAMES, "all"], default=["enamel"])
    p.add_argument("--seeds", type=int, default=None, help="Number of reporting seeds")
    p.add_argument("--seed-start", type=int, default=None)
    p.add_argument("--dev", action="store_true", help="Use dev seeds 1000..1009 (settings only)")
    p.add_argument("--methods", nargs="+", default=None)
    p.add_argument("--quick", action="store_true", help="5 seeds, smaller candidate set")
    p.add_argument("--n-jobs", type=int, default=None)
    p.add_argument("--no-mlflow", action="store_true")
    p.add_argument("--out-dir", default=None)
    p.add_argument(
        "--run-cost-spread",
        type=float,
        default=None,
        help="v2: enamel run-cost spread s (max/min run cost); default keeps the v1 formula",
    )
    p.add_argument("--mlflow-suffix", default=None, help="Suffix for the MLflow experiment name")
    p.set_defaults(func=cmd_benchmark)

    p = sub.add_parser("figures", help="Write every README figure from benchmark outputs")
    p.add_argument("--source", default=None, help="Folder with benchmark CSVs (default reports/)")
    p.add_argument("--skip-price-shock", action="store_true")
    p.set_defaults(func=cmd_figures)

    p = sub.add_parser("price-shock", help="Re-optimize under new prices, zero new experiments")
    p.add_argument("--problem", choices=PROBLEM_NAMES, default="enamel")
    p.add_argument("--change", action="append", default=None, help="e.g. pai_resin=+25%%")
    p.add_argument("--method", default=None, help="Campaign method (default: headline)")
    p.add_argument("--seed", type=int, default=None, help="Campaign seed (default: median)")
    p.add_argument("--source", default=None)
    p.add_argument("--tag", default="", help="Suffix for output file names")
    p.set_defaults(func=cmd_price_shock)

    p = sub.add_parser("latency", help="Time one BO suggestion per problem")
    p.add_argument("--problem", nargs="+", choices=PROBLEM_NAMES, default=list(PROBLEM_NAMES))
    p.add_argument("--repeats", type=int, default=5)
    p.set_defaults(func=cmd_latency)

    p = sub.add_parser("lab-note", help="Optional: local LLM lab note for one suggestion")
    p.add_argument("--problem", choices=PROBLEM_NAMES, default="enamel")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="Output JSON (default reports/v2/...)")
    p.set_defaults(func=cmd_lab_note)

    p = sub.add_parser("gp-coverage", help="Hold-out interval coverage of the GP surrogate")
    p.add_argument("--problem", choices=PROBLEM_NAMES, default="enamel")
    p.add_argument("--n-train", type=int, default=25)
    p.add_argument("--n-test", type=int, default=500)
    p.add_argument("--seeds", type=int, default=5)
    p.set_defaults(func=cmd_gp_coverage)

    from formopt.cli_v2 import add_v2_commands

    add_v2_commands(sub)

    p = sub.add_parser("reference-opt", help="Reference optimum and calibration checks")
    p.add_argument("--problem", nargs="+", choices=PROBLEM_NAMES, default=list(PROBLEM_NAMES))
    p.add_argument("--refresh", action="store_true", help="Ignore the cached reference")
    p.set_defaults(func=cmd_reference_opt)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for `python -m formopt`."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    args.func(cfg, args)
