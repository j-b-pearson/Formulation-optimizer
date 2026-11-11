"""Render every v2 results table as Markdown straight from the files in reports/v2/.

README numbers are copied from the output of this script (or from
the files it reads), never retyped. Missing inputs are reported as missing, not filled in.

Usage (PowerShell):
    uv run python scripts/report_tables_v2.py         # writes reports/v2/tables_v2.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from report_tables import md_table, summary_table  # noqa: E402

from formopt.config import load_config  # noqa: E402
from formopt.plotting import METHOD_LABELS  # noqa: E402

METRIC_NAMES = {
    "spend_saved": "spend to target saved ($, censored at budget)",
    "success_gain": "success gain (share of seeds)",
    "regret_reduction": "final regret reduction ($/kg)",
}
DIGITS = {"spend_saved": 0, "success_gain": 3, "regret_reduction": 4}


def ci(mean: float, lo: float, hi: float, nd: int) -> str:
    return f"{mean:,.{nd}f} [{lo:,.{nd}f}, {hi:,.{nd}f}]"


def read_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def spread_table(v2: Path) -> str:
    r = read_json(v2 / "run_cost_spread.json")
    if r is None:
        return "_missing: reports/v2/run_cost_spread.json_"
    rows = [
        [
            f"{x['s']:g}",
            f"${x['k_cheapest_run']:,.2f}",
            f"${x['most_expensive_run']:,.2f}",
            f"${x['mean_run_cost']:,.2f}",
            f"{x['expected_runs_in_budget_at_mean']:.1f}",
        ]
        for x in r["levels"]
    ]
    rows.append(
        [
            f"v1 formula (effective s = {r['v1_effective_s']:.4f})",
            f"${r['v1_run_cost_min']:,.2f}",
            f"${r['v1_run_cost_max']:,.2f}",
            f"${r['v1_mean_run_cost']:,.2f}",
            f"{r['budget'] / r['v1_mean_run_cost']:.1f}",
        ]
    )
    header = [
        "Run-cost spread s",
        "Cheapest possible run",
        "Most expensive run",
        "Mean over the box",
        "Runs per budget at the mean",
    ]
    return md_table(rows, header)


def level_table(analysis: Path) -> str:
    path = analysis / "h1_per_level.csv"
    if not path.exists():
        return "_missing: reports/v2/analysis/h1_per_level.csv_"
    df = pd.read_csv(path)
    rows = []
    for _, r in df.iterrows():
        nd = DIGITS[r.metric]
        rows.append(
            [
                f"{r.s:g}",
                f"`{r.method}` vs `{r.baseline}`",
                METRIC_NAMES[r.metric],
                ci(r["mean"], r.ci_lo, r.ci_hi, nd),
                "yes" if r.ci_above_zero else ("below 0" if r.ci_below_zero else "no"),
                f"{int(r.seeds_better)} / {int(r.seeds_tied)} / {int(r.seeds_worse)}",
            ]
        )
    header = [
        "s",
        "Comparison",
        "Advantage metric (positive = cost-aware better)",
        "Mean [95 % CI]",
        "CI above 0",
        "Seeds better / tied / worse",
    ]
    return md_table(rows, header)


def growth_table(analysis: Path) -> str:
    path = analysis / "h1_growth.csv"
    if not path.exists():
        return "_missing: reports/v2/analysis/h1_growth.csv_"
    df = pd.read_csv(path)
    a_cols = [c for c in df.columns if c.startswith("A_s")]
    rows = []
    for _, r in df.iterrows():
        nd = DIGITS[r.metric]
        rows.append(
            [
                f"`{r.method}`",
                METRIC_NAMES[r.metric],
                " / ".join(f"{r[c]:,.{nd}f}" for c in a_cols),
                ci(r.growth, r.growth_ci_lo, r.growth_ci_hi, nd),
                "yes" if r.growth_ci_above_zero else "no",
                "yes" if r.non_decreasing_across_levels else "no",
                "**yes**" if r.criterion_met else "no",
            ]
        )
    levels = ", ".join(c.replace("A_s", "s = ") for c in a_cols)
    header = [
        "Method",
        "Metric",
        f"Mean advantage at {levels}",
        "Growth A(10) - A(1) [95 % CI]",
        "Growth CI above 0",
        "Non-decreasing",
        "Criterion met",
    ]
    return md_table(rows, header)


def verdict_lines(analysis: Path) -> str:
    r = read_json(analysis / "h1_h2_results.json")
    if r is None:
        return "_missing: reports/v2/analysis/h1_h2_results.json_"
    lines = []
    for method, v in r["h1"].items():
        pay = v["starts_to_pay_at_s"]
        pay_text = f"s = {pay:g}" if pay is not None else "never within the tested range (s <= 10)"
        role = "H1 headline" if method == r["h1_headline_method"] else "secondary, not the verdict"
        lines.append(
            f"- `{method}` ({role}): H1 **{v['verdict']}**; cost-awareness starts to pay at: "
            f"{pay_text}."
        )
    seeds = r["seeds"]
    lines.append(
        f"- Seeds {seeds[0]} to {seeds[-1]} ({len(seeds)}), bootstrap {r['n_boot']:,} "
        f"resamples (seed {r['bootstrap_seed']})."
    )
    return "\n".join(lines)


def descriptive_table(analysis: Path) -> str:
    path = analysis / "h1_descriptive_by_level.csv"
    if not path.exists():
        return "_missing_"
    df = pd.read_csv(path)
    rows = [
        [
            f"{r.s:g}",
            METHOD_LABELS.get(r.method, r.method) + f" (`{r.method}`)",
            f"{r.success_rate:.3f}",
            f"{r.mean_n_experiments:.1f}",
            f"${r.mean_run_cost:,.0f}",
            f"{r.median_final_regret:.4f}",
        ]
        for _, r in df.iterrows()
    ]
    header = [
        "s",
        "Method",
        "Success rate",
        "Mean experiments",
        "Mean $ per run",
        "Median final regret ($/kg)",
    ]
    return md_table(rows, header)


def simple_adv_table(path: Path, label: str) -> str:
    if not path.exists():
        return f"_missing: {path.as_posix()}_"
    df = pd.read_csv(path)
    rows = []
    for _, r in df.iterrows():
        metric = r.get("metric", "first_feasible")
        nd = DIGITS.get(metric, 0)
        name = METRIC_NAMES.get(metric, "spend to first in-spec recipe saved ($)")
        lead = [f"{r.s:g}"] if "s" in df.columns else []
        rows.append(
            lead
            + [
                f"`{r.method}`",
                name,
                ci(r["mean"], r.ci_lo, r.ci_hi, nd),
                f"{int(r.seeds_better)} / {int(r.seeds_tied)} / {int(r.seeds_worse)}",
            ]
        )
    header = (["s"] if "s" in df.columns else []) + [
        "Method (vs `bo_plain`)",
        f"{label} metric",
        "Mean [95 % CI]",
        "Seeds better / tied / worse",
    ]
    return md_table(rows, header)


def h2_table(analysis: Path) -> str:
    r = read_json(analysis / "h1_h2_results.json")
    if r is None or "h2" not in r:
        return "_missing: H2 results_"
    h = r["h2"]
    rows = [
        [
            f"`{h['method']}` minus `{h['baseline']}`",
            f"{h['success_rate_method']:.3f}",
            f"{h['success_rate_baseline']:.3f}",
            ci(h["paired_success_difference"], h["ci_lo"], h["ci_hi"], 3),
            f"{h['seeds_only_method_succeeded']} / {h['seeds_only_baseline_succeeded']}",
            f"{h['mcnemar_exact_p_two_sided']:.3f}",
            f"{h['n_seeds']}",
            "**yes**" if h["replicated"] else "**no**",
        ]
    ]
    header = [
        "Comparison (enamel, v1 settings)",
        "Success rate, bo_cost",
        "Success rate, bo_plain",
        "Paired difference [95 % CI]",
        "Seeds only bo_cost / only bo_plain succeeded",
        "Exact McNemar p",
        "Seeds",
        "Replicated (CI above 0)",
    ]
    return md_table(rows, header)


def robustness_lines(analysis: Path) -> str:
    r = read_json(analysis / "robustness.json")
    if r is None:
        return "_missing: reports/v2/analysis/robustness.json_"
    return (
        f"Robustness re-check with {r['n_boot']:,} resamples (seed {r['bootstrap_seed']}): "
        f"H1 verdicts unchanged: {'yes' if r['h1_verdicts_unchanged'] else 'NO'}; "
        f"'starts to pay' unchanged: {'yes' if r['starts_to_pay_unchanged'] else 'NO'}; "
        f"H2 verdict unchanged: {'yes' if r['h2_verdict_unchanged'] else 'NO'}."
    )


def step3_tables(v2: Path) -> str:
    folder = v2 / "step3_concrete"
    curves_path = folder / "success_curves_concrete.csv"
    info = read_json(folder / "step3_info.json")
    if not curves_path.exists() or info is None:
        return "_missing: reports/v2/step3_concrete/_"
    curves = pd.read_csv(curves_path)
    pooled = curves[curves.source == "pooled"]
    full = pooled[pooled.n_experiments == pooled.n_experiments.max()]
    tols = sorted(full.tolerance.unique())
    rows = []
    for method in dict.fromkeys(full.method):
        g = full[full.method == method].set_index("tolerance")
        rows.append(
            [METHOD_LABELS.get(method, method) + f" (`{method}`)"]
            + [
                f"{g.loc[t, 'success_rate']:.3f} [{g.loc[t, 'ci_lo']:.3f}, {g.loc[t, 'ci_hi']:.3f}]"
                f" ({int(g.loc[t, 'n_success'])}/{int(g.loc[t, 'n_seeds'])})"
                for t in tols
            ]
        )
    header = ["Method"] + [
        f"within {100 * t:g} % (<= {info['target_cost_at_tolerance'][f'{t:g}']:.2f} $/m3)"
        for t in tols
    ]
    src = "; ".join(
        f"{k}: seeds {v['seeds'][0]} to {v['seeds'][-1]} ({v['folder']})"
        for k, v in info["sources"].items()
    )
    head = (
        f"DESCRIPTIVE. {info['n_seeds_pooled_per_method']} seeds per method ({src}). "
        f"Reference optimum {info['reference_cost']:.4f} $/m3; cheapest truly in-spec mix "
        f"found by any campaign: {info['best_in_spec_cost_any_campaign']:.4f} $/m3. Success at "
        f"the full budget ({int(full.n_experiments.max())} experiments max), Wilson 95 % CI, "
        f"(successes / seeds).\n\n"
    )
    gaps = pd.read_csv(folder / "gap_to_optimum_concrete.csv")
    gap_rows = [
        [
            METHOD_LABELS.get(r.method, r.method) + f" (`{r.method}`)",
            f"{r.share_found_any_in_spec:.3f}",
            f"{r.min_gap_pct:.2f}",
            f"{r.median_gap_pct:.2f}",
            f"{r.max_gap_pct:.2f}",
        ]
        for _, r in gaps.iterrows()
    ]
    gap_header = [
        "Method",
        "Share of campaigns with any in-spec mix",
        "Best campaign: % above optimum",
        "Median campaign: % above optimum",
        "Worst campaign: % above optimum",
    ]
    return head + md_table(rows, header) + "\n\n" + md_table(gap_rows, gap_header)


def step5_table(v2: Path) -> str:
    path = v2 / "step5_batch_dev" / "summary_enamel.csv"
    info = read_json(v2 / "step5_batch_dev" / "batch_info.json")
    if not path.exists() or info is None:
        return "_missing: reports/v2/step5_batch_dev/_"
    df = pd.read_csv(path)
    rows = [
        [
            f"`{r.method}`",
            f"{r.success_rate:.3f} [{r.success_ci_lo:.3f}, {r.success_ci_hi:.3f}]",
            f"{r.mean_final_regret:.4f} [{r.regret_ci_lo:.4f}, {r.regret_ci_hi:.4f}]",
            f"{r.mean_n_experiments:.1f}",
            f"{r.mean_n_cycles:.1f}",
            f"{r.mean_wall_time_s:.1f}",
        ]
        for _, r in df.iterrows()
    ]
    head = (
        f"EXPLORATORY, dev seeds {info['seeds'][0]} to {info['seeds'][-1]} only, q = {info['q']}, "
        f"enamel at v1 settings, budget ${info['budget']:,.0f}.\n\n"
    )
    header = [
        "Method",
        "Success rate [95 % Wilson CI]",
        "Mean final regret, $/kg [95 % CI]",
        "Mean experiments",
        "Mean oven cycles",
        "Mean wall time per campaign (s)",
    ]
    return head + md_table(rows, header)


def run_facts(v2: Path) -> str:
    lines = []
    repro = read_json(v2 / "v1_repro" / "repro_check.json")
    if repro is None:
        lines.append("- v1 reproduction check: _missing_")
    else:
        detail = "; ".join(
            f"{p}: "
            + ", ".join(
                f"{k} {'identical' if f['identical'] else 'DIFFERENT'}" for k, f in files.items()
            )
            for p, files in repro["problems"].items()
        )
        lines.append(
            f"- v1 reproduction check (seeds 0 to 29, default settings): all identical = "
            f"{repro['all_identical']} ({detail})."
        )
    times = []
    for problem in ("enamel", "concrete"):
        info = read_json(v2 / "v1_repro" / f"benchmark_info_{problem}.json")
        if info is not None:
            times.append(
                f"{problem} {info['wall_time_s']:.0f} s with n_jobs {info['n_jobs']} "
                f"(summed per-campaign time {info['sum_campaign_wall_time_s']:.0f} s)"
            )
    if times:
        lines.append("- v1 reproduction rerun compute time (no MLflow): " + "; ".join(times) + ".")
    est = read_json(v2 / "runtime_estimate.json")
    if est is not None:
        lines.append(
            f"- Runtime estimate from dev seeds (n_jobs {est['n_jobs']}): sweep "
            f"{est['estimated_sweep_s'] / 60:.1f} min + v1 settings "
            f"{est['estimated_v1_settings_s'] / 60:.1f} min = "
            f"{est['estimated_total_hours']:.2f} h (limit {est['limit_hours']:g} h); "
            f"cut seeds: {est['cut_reporting_seeds']}."
        )
    sweep = read_json(v2 / "step1_sweep" / "sweep_info.json")
    if sweep is not None:
        per = ", ".join(f"{k} {v / 60:.1f} min" for k, v in sweep["wall_time_s_per_level"].items())
        lines.append(
            f"- Spread sweep actual: {sweep['wall_time_s_total'] / 60:.1f} min ({per}), "
            f"n_jobs {sweep['n_jobs']}, MLflow logging {sweep['log_mlflow']}."
        )
    v1s = read_json(v2 / "v1_settings" / "run_info.json")
    if v1s is not None:
        per = ", ".join(f"{k} {v / 60:.1f} min" for k, v in v1s["wall_time_s_per_problem"].items())
        lines.append(
            f"- v1-settings run actual: {v1s['wall_time_s_total'] / 60:.1f} min ({per}), "
            f"n_jobs {v1s['n_jobs']}."
        )
    note = read_json(v2 / "lab_note_check_enamel.json")
    if note is not None:
        lines.append(
            f"- Live lab-note check ({note['model']}, {note['seconds']:.1f} s): source "
            f"**{note['source']}**, computed label {note['risk_label_computed_in_code']} at "
            f"{note['facts_sent']['pass_probability_pct']} % pass probability, "
            f"{len(note['attempts'])} LLM attempt(s)."
        )
    return "\n".join(lines)


def main() -> None:
    cfg = load_config()
    v2 = cfg.abs_paths.v2_dir
    analysis = v2 / "analysis"
    parts = ["# v2 results tables (generated by scripts/report_tables_v2.py from reports/v2/)\n"]
    parts.append("## Run facts\n\n" + run_facts(v2) + "\n")
    parts.append("## Run-cost family (spread sweep)\n\n" + spread_table(v2) + "\n")
    parts.append("## H1 verdicts\n\n" + verdict_lines(analysis) + "\n")
    parts.append("## H1 growth test (pre-registered criterion)\n\n" + growth_table(analysis) + "\n")
    parts.append("## H1 per-level paired advantages\n\n" + level_table(analysis) + "\n")
    parts.append(
        "## Descriptive by level (mechanism check)\n\n" + descriptive_table(analysis) + "\n"
    )
    parts.append(
        "## Secondary S2: spend to first in-spec recipe of any cost (censored at budget)\n\n"
        + simple_adv_table(analysis / "h1_secondary_first_feasible.csv", "S2")
        + "\n"
    )
    parts.append(
        "## Descriptive: advantage at the v1 run cost (s = 6.35), seeds 31 to 130\n\n"
        + simple_adv_table(analysis / "v1_settings_advantage.csv", "v1-settings")
        + "\n"
    )
    parts.append("## H2\n\n" + h2_table(analysis) + "\n\n" + robustness_lines(analysis) + "\n")
    for problem in ("enamel", "concrete"):
        path = v2 / "v1_settings" / f"summary_{problem}.csv"
        if path.exists():
            parts.append(
                f"## v1 settings on new seeds 31 to 130: {problem} summary\n\n"
                + summary_table(pd.read_csv(path), problem)
                + "\n"
            )
    parts.append("## Concrete success vs tolerance (DESCRIPTIVE)\n\n" + step3_tables(v2) + "\n")
    parts.append("## Batch vs sequential (EXPLORATORY)\n\n" + step5_table(v2) + "\n")
    out = v2 / "tables_v2.md"
    out.write_text("\n".join(parts), encoding="utf-8")
    print(out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
