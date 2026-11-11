"""Render every results table as Markdown straight from the files in reports/.

README numbers are copied from the output of this script, never retyped.

Usage (PowerShell):
    uv run python scripts/report_tables.py            # writes reports/tables.md
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from formopt.config import load_config
from formopt.plotting import METHOD_LABELS

UNITS = {"enamel": "$/kg", "concrete": "$/m3"}


def md_table(rows: list[list[str]], header: list[str]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def fmt(v: float, nd: int = 3) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    if np.isinf(v):
        return "inf"
    return f"{v:,.{nd}f}"


def run_line(info: dict) -> str:
    mode = "quick" if info["quick"] else "full"
    return (
        f"Run: {len(info['seeds'])} seeds ({info['seeds'][0]} to {info['seeds'][-1]}), {mode} mode "
        f"({info['n_candidates']:,} candidates per BO step), budget ${info['budget']:,.0f}, "
        f"reference optimum {info['reference_cost']:.4f} {UNITS[info['problem']]}, target "
        f"<= {info['target_cost']:.4f}. Wall time {info['wall_time_s']:.0f} s with n_jobs = "
        f"{info['n_jobs']}; sum of per-campaign times {info['sum_campaign_wall_time_s']:.0f} s "
        f"(about the n_jobs = 1 runtime)."
    )


def summary_table(summary: pd.DataFrame, problem: str) -> str:
    unit = UNITS[problem]
    rows = []
    for _, r in summary.iterrows():
        spend = (
            "not reached (<= half of seeds)"
            if np.isnan(r.median_spend_to_target)
            else f"${r.median_spend_to_target:,.0f} [{r.spend_ci_lo:,.0f}, {r.spend_ci_hi:,.0f}]"
        )
        rows.append(
            [
                METHOD_LABELS.get(r.method, r.method) + f" (`{r.method}`)",
                f"{r.success_rate:.3f} [{r.success_ci_lo:.3f}, {r.success_ci_hi:.3f}]",
                spend,
                fmt(r.median_experiments_to_target, 1),
                f"{r.mean_final_regret:.4f} [{r.mean_regret_ci_lo:.4f}, {r.mean_regret_ci_hi:.4f}]",
                f"{r.median_final_regret:.4f} [{r.median_regret_ci_lo:.4f}, "
                f"{r.median_regret_ci_hi:.4f}]",
                f"{r.mean_n_experiments:.1f}",
                f"{r.lab_pick_infeasible_rate:.3f}",
            ]
        )
    header = [
        "Method",
        "Success rate [95 % Wilson CI]",
        "Median spend-to-target [CI]",
        "Median experiments-to-target",
        f"Mean final regret, {unit} [95 % CI]",
        f"Median final regret, {unit} [95 % CI]",
        "Mean experiments",
        "Lab's pick truly out of spec",
    ]
    return md_table(rows, header)


def paired_table(paired: pd.DataFrame, method_a: str, problem: str) -> str:
    unit = UNITS[problem]
    labels = {
        "final_regret": f"final regret ({unit})",
        "spend_to_target_censored": "spend-to-target ($, censored at budget)",
        "success": "success (0/1)",
    }
    rows = []
    for _, r in paired[paired.method_a == method_a].iterrows():
        nd = 4 if r.metric == "final_regret" else (0 if r.metric.startswith("spend") else 3)
        rows.append(
            [
                f"`{r.method_a}` minus `{r.method_b}`",
                labels[r.metric],
                f"{r.mean_diff_a_minus_b:,.{nd}f} [{r.ci_lo:,.{nd}f}, {r.ci_hi:,.{nd}f}]",
                "yes" if r.ci_excludes_zero else "no",
                f"{int(r.seeds_a_better)} / {int(r.seeds_tied)} / {int(r.n_pairs)}",
            ]
        )
    header = [
        "Comparison",
        "Metric",
        "Mean paired difference [95 % CI]",
        "CI excludes 0",
        "Seeds a better / tied / total",
    ]
    return md_table(rows, header)


def calibration_table(reports: Path) -> str:
    rows = []
    for problem in ("enamel", "concrete"):
        info_path = reports / f"benchmark_info_{problem}.json"
        hold_path = reports / f"gp_holdout_coverage_{problem}.json"
        if not info_path.exists():
            continue
        info = json.loads(info_path.read_text(encoding="utf-8"))
        hold = (
            json.loads(hold_path.read_text(encoding="utf-8"))["mean"] if hold_path.exists() else {}
        )
        for out, z in info["calibration_z"].items():
            z = np.asarray(z)
            rows.append(
                [
                    f"{problem} `{out}`",
                    f"{np.mean(np.abs(z) <= 1.96):.3f} (n = {z.size})",
                    f"{np.mean(z):.2f}",
                    fmt(hold.get(f"{out}_predictive_coverage", float("nan"))),
                ]
            )
    header = [
        "Output",
        "In-loop next-point 95 % coverage (all BO campaigns)",
        "Mean z",
        "Hold-out 95 % coverage (25 LHS points, 500 test, 5 seeds)",
    ]
    return md_table(rows, header)


def emulator_table(reports: Path) -> str:
    v = json.loads((reports / "emulator_validation.json").read_text(encoding="utf-8"))
    rows = [
        [
            f"`{k}`" + (" (chosen)" if k == v["emulator_kind"] else ""),
            f"{m['r2']:.4f}",
            f"{m['rmse']:.4f}",
            f"{m['mae']:.4f}",
        ]
        for k, m in v["model_comparison"].items()
    ]
    note = (
        f"\n\nGrouped 5-fold CV by mix_id on {v['n_rows']:,} rows ({v['n_unique_mixes']} mixes, "
        f"{v['n_exact_duplicates_dropped']} exact duplicates dropped). Lab noise sd used in the "
        f"problem: {v['noise_sd_used_in_problem']} MPa (CV residual sd {v['residual_sd']:.3f}, "
        f"capped). Strength non-decreasing 7 to 28 to 90 days for "
        f"{v['sanity_checks']['age_mixes_nondecreasing_7_28_90']} of 10 random mixes; partial "
        f"dependence on w/b decreasing on {v['sanity_checks']['pd_wb_decreasing_steps']} of "
        f"{v['sanity_checks']['pd_wb_total_steps']} steps."
    )
    return md_table(rows, ["Emulator", "Grouped-CV R2", "RMSE (MPa)", "MAE (MPa)"]) + note


def price_table(path: Path) -> str:
    r = json.loads(path.read_text(encoding="utf-8"))
    unit = UNITS[r["problem"]]
    change = ", ".join(f"{k} x{v:g}" for k, v in r["changes_multiplier"].items())
    names = list(r["old"]["recipe"])
    rows = []
    for label in ("old", "new"):
        rec = r[label]
        recipe = ", ".join(f"{n} {rec['recipe'][n]:.3g}" for n in names)
        rows.append(
            [
                f"{label} prices",
                recipe,
                f"{rec['material_cost']:.4f}",
                f"{rec['pof']:.3f}",
                "yes" if rec["truly_feasible"] else "no",
                f"{r[f'reference_cost_{label}_prices']:.4f}",
            ]
        )
    head = (
        f"**{r['problem']}: {change}.** GPs refit on {r['n_experiments_used']} experiments of "
        f"`{r['source_method']}` seed {r['source_seed']} ({r['source_run']}); "
        f"{r['new_experiments']} new experiments; recommend the cheapest of "
        f"{r['n_candidates']:,} candidates with PoF >= {r['min_pof']}. Old recipe at new prices "
        f"costs {r['old_recipe_cost_at_new_prices']:.4f} {unit}.\n\n"
    )
    header = [
        "Scenario",
        "Recommended recipe",
        f"Material cost ({unit})",
        "PoF",
        "Truly in spec (simulator check)",
        f"True cheapest feasible ({unit})",
    ]
    return head + md_table(rows, header)


def main() -> None:
    cfg = load_config()
    reports = cfg.abs_paths.reports_dir
    headline = cfg.benchmark.headline_cost_method
    parts = ["# Results tables (generated by scripts/report_tables.py from reports/)\n"]
    for problem in ("enamel", "concrete"):
        summary_path = reports / f"summary_{problem}.csv"
        if not summary_path.exists():
            continue
        info = json.loads((reports / f"benchmark_info_{problem}.json").read_text("utf-8"))
        parts.append(f"## {problem.capitalize()} benchmark\n\n{run_line(info)}\n")
        parts.append(summary_table(pd.read_csv(summary_path), problem) + "\n")
        paired = pd.read_csv(reports / f"paired_{problem}.csv")
        for a in dict.fromkeys([headline, "bo_cost", "bo_cost_cooled"]):
            parts.append(f"### Paired differences, `{a}` versus each method ({problem})\n")
            parts.append(paired_table(paired, a, problem) + "\n")
    dev = reports / "dev" / "summary_enamel.csv"
    if dev.exists():
        parts.append("## Dev seeds 1000 to 1009 (enamel, settings decisions only)\n")
        parts.append(summary_table(pd.read_csv(dev), "enamel") + "\n")
    parts.append("## Concrete emulator validation\n\n" + emulator_table(reports) + "\n")
    parts.append("## Surrogate calibration\n\n" + calibration_table(reports) + "\n")
    for path in sorted(reports.glob("price_shock_*.json")):
        parts.append("## Price shock\n\n" + price_table(path) + "\n")
    out = reports / "tables.md"
    out.write_text("\n".join(parts), encoding="utf-8")
    print(out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
