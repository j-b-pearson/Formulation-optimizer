"""v1 reproducibility check: compare a fresh benchmark run with the stored v1 outputs exactly.

Wall-clock columns (and the worker count) are the only fields allowed to differ: they measure
the computer, not the method. Every other value must be equal, float for float.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

TIME_COLUMNS = {"wall_time_s", "mean_wall_time_s"}
TIME_KEYS = {"wall_time_s", "sum_campaign_wall_time_s", "n_jobs"}
CSV_FILES = ("summary", "paired", "campaign_metrics", "benchmark")


def compare_frames(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """Exact comparison of two tables, ignoring wall-clock columns."""
    drop = [c for c in TIME_COLUMNS if c in a.columns or c in b.columns]
    a = a.drop(columns=[c for c in drop if c in a.columns])
    b = b.drop(columns=[c for c in drop if c in b.columns])
    if list(a.columns) != list(b.columns) or a.shape != b.shape:
        return {"identical": False, "reason": "different columns or shape"}
    diff_cols = [c for c in a.columns if not a[c].equals(b[c])]
    out: dict = {"identical": not diff_cols, "ignored_columns": sorted(drop)}
    if diff_cols:
        out["differing_columns"] = diff_cols
        numeric = [c for c in diff_cols if pd.api.types.is_numeric_dtype(a[c])]
        out["max_abs_diff"] = {
            c: float(np.nanmax(np.abs(a[c].to_numpy(float) - b[c].to_numpy(float))))
            for c in numeric
        }
    return out


def compare_info(a: dict, b: dict) -> dict:
    """Exact comparison of two benchmark_info JSON dicts, ignoring wall-clock keys."""
    keys = sorted((set(a) | set(b)) - TIME_KEYS)
    differing = [k for k in keys if a.get(k) != b.get(k)]
    return {"identical": not differing, "differing_keys": differing, "ignored": sorted(TIME_KEYS)}


def compare_dirs(reference: Path, candidate: Path, problems: list[str]) -> dict:
    """Compare every benchmark output of `candidate` with `reference`, per problem and file."""
    report: dict = {"reference": str(reference), "candidate": str(candidate), "problems": {}}
    all_ok = True
    for name in problems:
        files: dict = {}
        for stem in CSV_FILES:
            ra, ca = Path(reference) / f"{stem}_{name}.csv", Path(candidate) / f"{stem}_{name}.csv"
            if not (ra.exists() and ca.exists()):
                files[stem] = {"identical": None, "reason": "missing in one folder"}
                continue
            files[stem] = compare_frames(pd.read_csv(ra), pd.read_csv(ca))
        ra = Path(reference) / f"benchmark_info_{name}.json"
        ca = Path(candidate) / f"benchmark_info_{name}.json"
        files["benchmark_info"] = compare_info(
            json.loads(ra.read_text(encoding="utf-8")), json.loads(ca.read_text(encoding="utf-8"))
        )
        # A file missing on either side was not compared, so it cannot count as identical.
        all_ok &= all(f["identical"] is True for f in files.values())
        report["problems"][name] = files
    report["all_identical"] = bool(all_ok)
    return report
