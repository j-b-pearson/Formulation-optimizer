"""Price shock: re-optimize for new raw material prices with ZERO new experiments.

The chemistry model (GPs fit on measured outputs) does not depend on prices; only the known
cost formula does. So when a price moves, refit nothing new: score the same candidates with
the new cost formula and pick the cheapest one that is likely to pass spec.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from formopt.acquisition import probability_of_feasibility, probability_of_feasibility_per_spec
from formopt.config import AppConfig
from formopt.design import sample_known_feasible
from formopt.gp import FittedGP
from formopt.problems.base import Problem, passes_specs, reference_optimum
from formopt.strategies import GPSettings, fit_surrogates
from formopt.types import Observation


def parse_change(text: str, prices: dict[str, float]) -> tuple[str, float]:
    """Parse 'pai_resin=+25%', 'cement=-10%' or 'solvent=x1.4' into (key, multiplier)."""
    match = re.fullmatch(r"\s*(\w+)\s*=\s*(?:([+-]?\d+(?:\.\d+)?)%|x(\d+(?:\.\d+)?))\s*", text)
    if not match:
        raise ValueError(f"Cannot parse price change {text!r}; use e.g. pai_resin=+25%")
    key = match.group(1)
    if key not in prices:
        raise KeyError(f"Unknown price {key!r}; known: {sorted(prices)}")
    mult = 1.0 + float(match.group(2)) / 100.0 if match.group(2) else float(match.group(3))
    return key, mult


def history_from_long(long: pd.DataFrame, problem: Problem, method: str, seed: int) -> list:
    """Rebuild one campaign's observations from the long-format benchmark CSV."""
    g = long[(long["method"] == method) & (long["seed"] == seed)].sort_values("step")
    if g.empty:
        raise ValueError(f"No rows for method={method} seed={seed}")
    history = []
    for _, row in g.iterrows():
        history.append(
            Observation(
                x=row[problem.var_names].to_numpy(float),
                outputs={o: float(row[o]) for o in problem.output_names},
                run_cost=float(row["run_cost"]),
                material_cost=float(row["material_cost"]),
                passed_spec=bool(row["passed_spec"]),
                truly_feasible=bool(row["truly_feasible"]),
            )
        )
    return history


def median_seed(metrics: pd.DataFrame, method: str) -> int:
    """Seed whose final regret is the median for a method (lowest seed on ties)."""
    g = metrics[metrics["method"] == method].sort_values("seed")
    target = float(np.median(g["final_regret"]))
    gap = (g["final_regret"] - target).abs()
    return int(g.loc[gap.idxmin(), "seed"])


def recommend(
    problem: Problem, gps: dict[str, FittedGP], candidates: np.ndarray, min_pof: float
) -> dict:
    """Cheapest candidate whose predicted PoF is at least `min_pof` (zero new experiments).

    If no candidate reaches `min_pof`, the most likely feasible candidate is returned.
    """
    preds = {o: gp.predict(candidates) for o, gp in gps.items()}
    pof = probability_of_feasibility(preds, problem.specs)
    cost = problem.material_cost(candidates)
    ok = pof >= min_pof
    i = int(np.argmin(np.where(ok, cost, np.inf))) if ok.any() else int(np.argmax(pof))
    x = candidates[i]
    per_spec = probability_of_feasibility_per_spec(
        {o: (m[i : i + 1], s[i : i + 1]) for o, (m, s) in preds.items()}, problem.specs
    )
    predicted = {s.output: float(s.inverse_transform(preds[s.output][0][i])) for s in problem.specs}
    truth = {k: float(v[0]) for k, v in problem.true_outputs(x[None, :]).items()}
    return {
        "recipe": problem.as_dict(x),
        "x": x,
        "material_cost": float(cost[i]),
        "pof": float(pof[i]),
        "pof_per_spec": {k: float(v[0]) for k, v in per_spec.items()},
        "predicted_outputs": predicted,
        "met_min_pof": bool(ok.any()),
        "truly_feasible": bool(passes_specs(truth, problem.specs)),  # benchmark-only check
        "true_outputs": truth,
    }


def run_price_shock(
    problem: Problem,
    history: list[Observation],
    changes: dict[str, float],
    cfg: AppConfig,
    gp_settings: GPSettings,
    with_truth: bool = True,
) -> dict:
    """Old and new recommendations from the same data and candidates; only prices differ."""
    ps = cfg.price_shock
    gps = fit_surrogates(problem, history, gp_settings, seed=ps.seed)
    rng = np.random.default_rng(ps.seed)
    candidates = sample_known_feasible(problem, ps.n_candidates, rng)
    new_prices = {k: problem.prices[k] * m for k, m in changes.items()}
    shocked = problem.with_prices(new_prices)
    old = recommend(problem, gps, candidates, ps.min_pof)
    new = recommend(shocked, gps, candidates, ps.min_pof)
    result = {
        "problem": problem.name,
        "n_experiments_used": len(history),
        "new_experiments": 0,
        "changes_multiplier": changes,
        "old_prices": dict(problem.prices),
        "new_prices": dict(shocked.prices),
        "min_pof": ps.min_pof,
        "n_candidates": ps.n_candidates,
        "old": old,
        "new": new,
        "old_recipe_cost_at_new_prices": float(shocked.material_cost(old["x"][None, :])[0]),
    }
    if with_truth:  # benchmark-only context: the true cheapest feasible cost at each price
        ref_cfg = cfg.benchmark.reference
        result["reference_cost_old_prices"] = float(reference_optimum(problem, ref_cfg)["cost"])
        result["reference_cost_new_prices"] = float(reference_optimum(shocked, ref_cfg)["cost"])
    return result


def price_shock_table(result: dict) -> pd.DataFrame:
    """Two-row table (old prices, new prices) for CSV output and the README."""
    rows = []
    for label in ("old", "new"):
        rec = result[label]
        row = {"scenario": f"{label} prices", **rec["recipe"]}
        row.update(
            {
                "material_cost": rec["material_cost"],
                "pof": rec["pof"],
                "truly_feasible": rec["truly_feasible"],
                **{f"pred_{k}": v for k, v in rec["predicted_outputs"].items()},
            }
        )
        if f"reference_cost_{label}_prices" in result:
            row["reference_cost"] = result[f"reference_cost_{label}_prices"]
        rows.append(row)
    return pd.DataFrame(rows)
