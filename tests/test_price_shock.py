"""Price shock: parsing, and re-optimization with zero new experiments."""

from __future__ import annotations

import numpy as np
import pytest

from formopt.loop import run_campaign
from formopt.price_shock import median_seed, parse_change, recommend
from formopt.strategies import GPSettings, LHSPlan, fit_surrogates
from tests.conftest import ToyProblem


def test_parse_change() -> None:
    prices = {"pai_resin": 14.0, "cement": 0.16}
    assert parse_change("pai_resin=+25%", prices) == ("pai_resin", pytest.approx(1.25))
    assert parse_change("cement=-10%", prices) == ("cement", pytest.approx(0.90))
    assert parse_change("cement=x1.3", prices) == ("cement", pytest.approx(1.3))
    with pytest.raises(KeyError):
        parse_change("gold=+5%", prices)
    with pytest.raises(ValueError):
        parse_change("cement=lots", prices)


def test_median_seed() -> None:
    import pandas as pd

    metrics = pd.DataFrame(
        {"method": ["a"] * 5, "seed": [0, 1, 2, 3, 4], "final_regret": [5, 1, 3, 2, 9]}
    )
    assert median_seed(metrics, "a") == 2


def test_recommendation_is_known_feasible_and_likely_to_pass() -> None:
    problem = ToyProblem()
    history = run_campaign(problem, LHSPlan(), 60.0, seed=0).observations
    gps = fit_surrogates(problem, history, GPSettings(), seed=0)
    cand = np.random.default_rng(1).random((3000, 2))
    cand = cand[problem.known_feasible(cand)]
    old = recommend(problem, gps, cand, min_pof=0.8)
    assert problem.known_feasible(old["x"][None, :])[0]
    assert old["met_min_pof"] and old["pof"] >= 0.8
    # Same GPs and candidates, new prices: the chemistry does not change, so PoF is unchanged
    shocked = problem.with_prices({"unit": 1.3})
    new = recommend(shocked, gps, cand, min_pof=0.8)
    assert new["pof"] >= 0.8
    assert new["material_cost"] == pytest.approx(1.3 * (new["x"][0] + new["x"][1]))
