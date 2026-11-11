"""Streamlit decision tool: suggest the next formulation experiment, cost-aware.

Run (PowerShell, from the repo root):
    uv run streamlit run app/streamlit_app.py

All logic lives in the `formopt` package; this file is UI only. Add `?demo=1` to the URL to
replay the demo script (6 initial runs, 5 suggestions, one manual entry) for screenshots.
"""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from formopt.benchmark import gp_settings
from formopt.config import AppConfig, load_config
from formopt.design import lhs_design, repair_to_feasible, sample_known_feasible
from formopt.estimate import expected_spend_to_spec
from formopt.loop import make_observation, noise_rng
from formopt.plotting import METHOD_LABELS, pof_grid, slice_data
from formopt.price_shock import recommend
from formopt.problems.base import Problem, passes_specs
from formopt.problems.concrete import ConcreteProblem, emulator_path
from formopt.problems.enamel import EnamelProblem
from formopt.strategies import Suggestion, best_passed_cost, bo_step, fit_surrogates
from formopt.types import Observation

ROOT = Path(__file__).resolve().parents[1]
METHODS = {"bo_cost_cooled": "cooled", "bo_cost": "fixed", "bo_plain": "none"}

st.set_page_config(
    page_title="Cost-aware formulation optimizer", layout="wide", initial_sidebar_state="expanded"
)


# ----- cached resources -----------------------------------------------------------------------
@st.cache_resource
def get_config() -> AppConfig:
    return load_config(ROOT / "config.yaml")


@st.cache_resource
def get_emulator():
    from formopt.emulator import ConcreteEmulator

    path = emulator_path(get_config())
    return ConcreteEmulator.load(path) if path.exists() else None


# ----- sidebar: problem settings --------------------------------------------------------------
def sidebar_problem(cfg: AppConfig) -> tuple[Problem, float, str, int]:
    """Build the problem from the sidebar settings (specs, prices, run cost, budget)."""
    st.sidebar.header("Scenario")
    choices = ["enamel"] + (["concrete"] if get_emulator() is not None else [])
    name = st.sidebar.selectbox(
        "Problem",
        choices,
        format_func=lambda n: {
            "enamel": "Magnet wire enamel (SYNTHETIC simulator)",
            "concrete": "Concrete (emulator trained on real UCI data)",
        }[n],
    )
    method = st.sidebar.selectbox(
        "Optimizer", list(METHODS), format_func=lambda m: METHOD_LABELS[m]
    )
    seed = int(st.sidebar.number_input("Seed", 0, 10_000, 0))
    if name == "enamel":
        e = cfg.enamel
        budget = st.sidebar.number_input("Lab budget ($)", 1000.0, 100000.0, e.budget, 500.0)
        with st.sidebar.expander("Specs", expanded=False):
            bd = st.number_input(
                "Breakdown at least (kV proxy)", 10.0, 60.0, float(e.specs["breakdown_kv"].lower)
            )
            fx = st.number_input(
                "Flexibility pass rate at least", 0.0, 1.0, float(e.specs["flex_pass"].lower), 0.05
            )
            v_lo = st.number_input(
                "Viscosity at least (cP)",
                500.0,
                10000.0,
                float(e.specs["viscosity_cp"].lower),
                100.0,
            )
            v_hi = st.number_input(
                "Viscosity at most (cP)",
                5000.0,
                50000.0,
                float(e.specs["viscosity_cp"].upper),
                500.0,
            )
        with st.sidebar.expander("Prices ($/kg, illustrative)", expanded=False):
            prices = {
                k: st.number_input(k, 0.0, 200.0, float(v), 0.1, key=f"p_{k}")
                for k, v in e.prices.items()
            }
        with st.sidebar.expander("Run cost model (illustrative)", expanded=False):
            rc = e.run_cost
            run_cost = rc.model_copy(
                update={
                    "fixed": st.number_input("Fixed $ per trial", 0.0, 5000.0, rc.fixed, 10.0),
                    "batch_kg": st.number_input("Batch kg per trial", 0.0, 500.0, rc.batch_kg, 5.0),
                    "energy_rate": st.number_input(
                        "Energy/time rate", 0.0, 50.0, rc.energy_rate, 0.5
                    ),
                }
            )
        specs = {
            "breakdown_kv": e.specs["breakdown_kv"].model_copy(update={"lower": bd}),
            "flex_pass": e.specs["flex_pass"].model_copy(update={"lower": fx}),
            "viscosity_cp": e.specs["viscosity_cp"].model_copy(
                update={"lower": v_lo, "upper": v_hi}
            ),
        }
        ecfg = e.model_copy(update={"specs": specs, "run_cost": run_cost, "budget": budget})
        return EnamelProblem(ecfg, prices), float(budget), method, seed
    c = cfg.concrete
    budget = st.sidebar.number_input("Lab budget ($)", 1000.0, 100000.0, c.budget, 500.0)
    with st.sidebar.expander("Specs", expanded=False):
        target = st.number_input(
            "28-day strength at least (MPa)", 20.0, 80.0, float(c.target_strength_mpa)
        )
    with st.sidebar.expander("Prices ($/kg; USGS-based, some ASSUMED)", expanded=False):
        prices = {
            k: st.number_input(k, 0.0, 10.0, float(v), 0.001, format="%.4f", key=f"c_{k}")
            for k, v in c.prices_per_kg.items()
        }
    with st.sidebar.expander("Run cost model (illustrative)", expanded=False):
        rc = c.run_cost
        run_cost = rc.model_copy(
            update={
                "fixed": st.number_input("Fixed $ per trial batch", 0.0, 5000.0, rc.fixed, 10.0),
                "batch_m3": st.number_input("m3 per trial batch", 0.0, 1.0, rc.batch_m3, 0.01),
            }
        )
    ccfg = c.model_copy(
        update={"target_strength_mpa": target, "run_cost": run_cost, "budget": budget}
    )
    return ConcreteProblem(get_emulator(), ccfg, prices), float(budget), method, seed


# ----- campaign state -------------------------------------------------------------------------
def rescore(problem: Problem, obs: Observation) -> Observation:
    """Re-judge a stored result under the current prices and specs (measurements don't change)."""
    truth = obs.true_outputs
    return replace(
        obs,
        material_cost=float(problem.material_cost(obs.x[None, :])[0]),
        passed_spec=bool(passes_specs(obs.outputs, problem.specs)),
        truly_feasible=bool(passes_specs(truth, problem.specs)) if truth else obs.truly_feasible,
    )


def init_state(problem: Problem, budget: float, seed: int) -> None:
    """New scenario: keep measured results when the recipe variables are the same, re-judge them.

    Method and seed are not part of the key: neither changes what the lab measured."""
    key = f"{problem.fingerprint()}|{budget}"
    if st.session_state.get("seed") != seed:
        st.session_state.rng = np.random.default_rng(seed)
        st.session_state.seed = seed
    if st.session_state.get("key") == key:
        return
    var_names = [v.name for v in problem.variables]
    keep = st.session_state.get("var_names") == var_names
    old = st.session_state.get("history", []) if keep else []
    st.session_state.key = key
    st.session_state.var_names = var_names
    st.session_state.history = [rescore(problem, o) for o in old]
    st.session_state.suggestion = None
    st.session_state.latency = None


def spent() -> float:
    return float(sum(o.run_cost for o in st.session_state.history))


def run_on_simulator(problem: Problem, x: np.ndarray, sug: Suggestion | None = None) -> None:
    step = len(st.session_state.history)
    pred = sug.pred_noisy if sug is not None else None
    obs = make_observation(problem, x, noise_rng(st.session_state.seed, step), pred)
    st.session_state.history.append(obs)
    st.session_state.suggestion = None


def initial_design(problem: Problem, budget: float, n: int) -> None:
    rng = st.session_state.rng
    X = lhs_design(problem.bounds, n, rng)
    pool = sample_known_feasible(problem, 4000, rng)
    X = repair_to_feasible(X, pool, problem.known_feasible(X), problem.bounds)
    for x in X:
        if spent() + problem.run_cost(x[None, :])[0] <= budget:
            run_on_simulator(problem, x)


def suggest(problem: Problem, budget: float, method: str, cfg: AppConfig) -> None:
    hist = st.session_state.history
    if len(hist) < 2:
        st.warning("Run the initial design first (at least 2 experiments).")
        return
    mode = METHODS[method]
    alpha = 0.0 if mode == "none" else (1.0 if mode == "fixed" else max(0.0, 1 - spent() / budget))
    t0 = time.perf_counter()
    sug = bo_step(
        problem,
        hist,
        spent(),
        budget,
        st.session_state.rng,
        cfg.bo.n_candidates,
        alpha,
        gp_settings(cfg),
    )
    if sug is not None:
        sug.extra["expected_spend"] = expected_spend_to_spec(
            problem, sug.gps, sug.candidates, k=10, scores=sug.scores
        )
    st.session_state.latency = time.perf_counter() - t0
    st.session_state.suggestion = sug
    if sug is None:
        st.warning("Nothing affordable is left in the budget.")


def history_frame(problem: Problem) -> pd.DataFrame:
    rows = []
    total = 0.0
    for i, o in enumerate(st.session_state.history):
        total += o.run_cost
        row = {"run": i + 1, **{k: round(v, 4) for k, v in problem.as_dict(o.x).items()}}
        row.update({k: round(v, 4) for k, v in o.outputs.items()})
        row.update(
            {
                "material_cost": round(o.material_cost, 4),
                "run_cost": round(o.run_cost, 2),
                "cumulative_spend": round(total, 2),
                "lab_pass": o.passed_spec,
                "truth_pass (simulator only)": o.truly_feasible,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


# ----- plots ----------------------------------------------------------------------------------
def slices_figure(problem: Problem, gps: dict, x_ref: np.ndarray) -> go.Figure:
    specs, variables = problem.specs, problem.variables
    fig = make_subplots(
        rows=len(specs),
        cols=len(variables),
        shared_xaxes=False,
        horizontal_spacing=0.04,
        vertical_spacing=0.08,
    )
    hist = st.session_state.history
    X = np.vstack([o.x for o in hist])
    for j, var in enumerate(variables):
        data = slice_data(gps, problem, x_ref, j, n=80)
        for i, spec in enumerate(specs):
            r, c = i + 1, j + 1
            g = data["grid"]
            fig.add_trace(
                go.Scatter(
                    x=np.r_[g, g[::-1]],
                    y=np.r_[data[f"{spec.output}_hi"], data[f"{spec.output}_lo"][::-1]],
                    fill="toself",
                    line_width=0,
                    fillcolor="rgba(0,114,178,0.2)",
                    showlegend=False,
                    hoverinfo="skip",
                ),
                r,
                c,
            )
            fig.add_trace(
                go.Scatter(
                    x=g, y=data[f"{spec.output}_mean"], line_color="#0072B2", showlegend=False
                ),
                r,
                c,
            )
            fig.add_trace(
                go.Scatter(
                    x=X[:, j],
                    y=[o.outputs[spec.output] for o in hist],
                    mode="markers",
                    marker=dict(size=5, color="black"),
                    showlegend=False,
                ),
                r,
                c,
            )
            for lim in (spec.lower, spec.upper):
                if lim is not None:
                    fig.add_hline(y=lim, line_dash="dash", line_color="#D55E00", row=r, col=c)
            fig.add_vline(x=float(x_ref[j]), line_color="grey", row=r, col=c)
            if spec.transform == "log":
                fig.update_yaxes(type="log", row=r, col=c)
            if i == len(specs) - 1:
                fig.update_xaxes(title_text=var.name, row=r, col=c)
            if j == 0:
                fig.update_yaxes(title_text=spec.output, row=r, col=c)
    fig.update_layout(
        height=230 * len(specs),
        margin=dict(l=10, r=10, t=30, b=10),
        title="GP mean and 95 % band through the suggested recipe",
    )
    return fig


def heatmap_figure(problem: Problem, gps: dict, x_ref: np.ndarray, vx: str, vy: str) -> go.Figure:
    data = pof_grid(gps, problem, x_ref, vx, vy, n=50)
    z = np.where(data["known_feasible"], data["pof"], np.nan)
    fig = go.Figure(
        go.Heatmap(
            x=data["x"],
            y=data["y"],
            z=z,
            zmin=0,
            zmax=1,
            colorscale="Viridis",
            colorbar_title="PoF",
        )
    )
    names = problem.var_names
    fig.add_trace(
        go.Scatter(
            x=[x_ref[names.index(vx)]],
            y=[x_ref[names.index(vy)]],
            mode="markers",
            marker=dict(symbol="star", size=16, color="#D55E00"),
            name="suggested",
        )
    )
    fig.update_layout(
        height=420,
        xaxis_title=vx,
        yaxis_title=vy,
        title="Probability of passing all specs (blank = known-infeasible)",
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def spend_figure(budget: float) -> go.Figure:
    hist = st.session_state.history
    spend = np.cumsum([o.run_cost for o in hist])
    best, trace = np.inf, []
    for o in hist:
        if o.passed_spec:
            best = min(best, o.material_cost)
        trace.append(best if np.isfinite(best) else None)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=list(range(1, len(hist) + 1)),
            y=[o.run_cost for o in hist],
            name="run cost ($)",
            marker_color="#BBBBBB",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(hist) + 1)),
            y=spend,
            name="cumulative spend ($)",
            line_color="#0072B2",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(hist) + 1)),
            y=trace,
            name="best lab-passed cost",
            line_color="#D55E00",
            mode="lines+markers",
        ),
        secondary_y=True,
    )
    fig.add_hline(y=budget, line_dash="dot", line_color="grey")
    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=30, b=10),
        title="Spend and best cost",
        legend=dict(orientation="h"),
    )
    fig.update_xaxes(title_text="experiment")
    return fig


# ----- tabs -----------------------------------------------------------------------------------
def suggestion_panel(problem: Problem, sug: Suggestion) -> None:
    st.subheader("Suggested next experiment")
    c1, c2 = st.columns([1.1, 1])
    with c1:
        recipe = pd.DataFrame(
            {
                "variable": problem.var_names,
                "value": np.round(sug.x, 4),
                "unit": [v.unit for v in problem.variables],
            }
        )
        st.dataframe(recipe, hide_index=True, width="stretch")
    with c2:
        rows = []
        for spec in problem.specs:
            mu, sd = sug.pred_latent[spec.output]
            lim = " to ".join(str(v) for v in (spec.lower, spec.upper) if v is not None)
            rows.append(
                {
                    "output": spec.output,
                    "predicted": float(spec.inverse_transform(mu)),
                    "95% low": float(spec.inverse_transform(mu - 1.96 * sd)),
                    "95% high": float(spec.inverse_transform(mu + 1.96 * sd)),
                    "spec": (">= " if spec.upper is None else "") + lim,
                    "P(pass)": sug.pof_per_spec[spec.output],
                }
            )
        st.dataframe(pd.DataFrame(rows).round(3), hide_index=True, width="stretch")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("P(pass all specs)", f"{sug.pof_at_x:.2f}")
    m2.metric("Run cost", f"${sug.run_cost:,.0f}")
    m3.metric("Material cost", f"{sug.material_cost:.3f}")
    m4.metric(
        "Expected spend to next in-spec result (rough estimate)",
        f"${sug.extra.get('expected_spend', float('nan')):,.0f}",
    )
    m5.metric("alpha (cost weight)", f"{sug.alpha:.2f}")
    if st.session_state.latency is not None:
        st.caption(
            f"Suggestion computed in {st.session_state.latency:.2f} s from "
            f"{len(sug.candidates):,} candidates. The spend estimate assumes runs pass "
            "independently, so treat it as a heuristic."
        )


def optimize_tab(problem: Problem, budget: float, method: str, cfg: AppConfig) -> None:
    hist = st.session_state.history
    left = budget - spent()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Experiments", len(hist))
    c2.metric("Spent", f"${spent():,.0f}", f"${left:,.0f} left", delta_color="off")
    best = best_passed_cost(hist)
    c3.metric("Best lab-passed material cost", "none yet" if best is None else f"{best:.3f}")
    c4.metric("Optimizer", METHOD_LABELS[method])

    b1, b2, b3, b4 = st.columns(4)
    n_init = cfg.bo.n_init
    if b1.button(f"Run initial design ({n_init} LHS runs)", disabled=bool(hist)):
        initial_design(problem, budget, n_init)
        st.rerun()
    if b2.button("Suggest next experiment", type="primary"):
        suggest(problem, budget, method, cfg)
    sug = st.session_state.suggestion
    if b3.button("Run suggestion on simulator", disabled=sug is None):
        run_on_simulator(problem, sug.x, sug)
        st.rerun()
    if b4.button("Reset campaign"):
        st.session_state.key = None
        st.rerun()

    if sug is not None:
        suggestion_panel(problem, sug)

    with st.expander("Enter a measured lab result manually"):
        default = sug.x if sug is not None else problem.bounds.mean(axis=1)
        with st.form("manual"):
            cols = st.columns(len(problem.variables))
            x = np.array(
                [
                    cols[i].number_input(
                        v.name, float(v.low), float(v.high), float(default[i]), key=f"mx_{v.name}"
                    )
                    for i, v in enumerate(problem.variables)
                ]
            )
            ocols = st.columns(len(problem.specs))
            outs = {
                s.output: ocols[i].number_input(
                    f"measured {s.output}",
                    # Bad values poison every later GP fit: log outputs must be > 0,
                    # pass fractions (flex_pass) lie in [0, 1], nothing measured is negative.
                    min_value=1e-6 if s.transform == "log" else 0.0,
                    max_value=1.0 if s.output.endswith("_pass") else None,
                    value=float((s.lower or 0) if s.upper is None else (s.lower + s.upper) / 2),
                    key=f"mo_{s.output}",
                )
                for i, s in enumerate(problem.specs)
            }
            if st.form_submit_button("Add result"):
                if not problem.known_feasible(x[None, :])[0]:
                    st.error("That recipe breaks a known constraint; it cannot be run.")
                else:
                    obs = make_observation(problem, x, np.random.default_rng(0))
                    obs = replace(
                        obs, outputs=outs, passed_spec=bool(passes_specs(outs, problem.specs))
                    )
                    st.session_state.history.append(obs)
                    st.session_state.suggestion = None
                    st.rerun()

    if hist:
        st.subheader("Campaign history")
        df = history_frame(problem)
        st.dataframe(df, hide_index=True, width="stretch")
        st.download_button(
            "Download history (CSV)",
            df.to_csv(index=False),
            file_name=f"campaign_{problem.name}.csv",
            mime="text/csv",
        )
        st.plotly_chart(spend_figure(budget), width="stretch")
    if len(hist) >= 2:
        gps = sug.gps if sug is not None else fit_surrogates(problem, hist, gp_settings(cfg), 0)
        x_ref = sug.x if sug is not None else hist[-1].x
        st.plotly_chart(slices_figure(problem, gps, x_ref), width="stretch")
        names = problem.var_names
        d1, d2 = st.columns(2)
        default_x = names.index("oven_temp_c") if "oven_temp_c" in names else 0
        default_y = names.index("line_speed_m_min") if "line_speed_m_min" in names else 3
        vx = d1.selectbox("Heatmap x", names, index=default_x)
        vy = d2.selectbox("Heatmap y", names, index=default_y)
        if vx != vy:
            st.plotly_chart(heatmap_figure(problem, gps, x_ref, vx, vy), width="stretch")


def benchmark_tab(cfg: AppConfig, name: str) -> None:
    reports = cfg.abs_paths.reports_dir
    summary = reports / f"summary_{name}.csv"
    if not summary.exists():
        st.info(
            "No benchmark results yet. Generate them with:\n\n"
            f"`uv run python -m formopt benchmark --problem {name} --seeds 30`\n\n"
            "then `uv run python -m formopt figures`."
        )
        return
    fig = cfg.abs_paths.figures_dir / f"spend_vs_best_{name}.png"
    if fig.exists():
        st.image(str(fig), caption="Headline: cumulative lab spend vs best truly feasible cost")
    df = pd.read_csv(summary)
    cols = [
        "method",
        "n_seeds",
        "success_rate",
        "success_ci_lo",
        "success_ci_hi",
        "median_spend_to_target",
        "mean_final_regret",
        "mean_regret_ci_lo",
        "mean_regret_ci_hi",
        "median_final_regret",
        "mean_n_experiments",
    ]
    st.dataframe(df[cols].round(4), hide_index=True, width="stretch")
    paired = reports / f"paired_{name}.csv"
    if paired.exists():
        st.caption("Paired differences (cost-aware BO minus each method), 95 % bootstrap CI")
        st.dataframe(pd.read_csv(paired).round(4), hide_index=True, width="stretch")


def price_tab(problem: Problem, cfg: AppConfig) -> None:
    hist = st.session_state.history
    if len(hist) < cfg.bo.n_init:
        st.info(
            f"Run at least {cfg.bo.n_init} experiments in the Optimize tab first. The price "
            "shock refits the GPs on those results and re-optimizes for new prices with "
            "zero new experiments."
        )
        return
    st.write("Move prices; the recommendation updates using only the experiments already run.")
    mults = {
        k: st.slider(
            f"{k} price change (%)",
            -50,
            100,
            int(st.query_params.get(f"ps_{k}", 0)),
            5,
            key=f"ps_{k}",
        )
        / 100
        + 1
        for k in problem.prices
    }
    min_pof = st.slider("Minimum P(pass all specs)", 0.5, 0.99, cfg.price_shock.min_pof, 0.01)
    gps = fit_surrogates(problem, hist, gp_settings(cfg), seed=cfg.price_shock.seed)
    cand = sample_known_feasible(problem, 20000, np.random.default_rng(cfg.price_shock.seed))
    shocked = problem.with_prices({k: problem.prices[k] * m for k, m in mults.items()})
    old, new = recommend(problem, gps, cand, min_pof), recommend(shocked, gps, cand, min_pof)
    table = pd.DataFrame(
        {
            "variable": problem.var_names,
            "current prices": [old["recipe"][n] for n in problem.var_names],
            "new prices": [new["recipe"][n] for n in problem.var_names],
        }
    )
    st.dataframe(table.round(4), hide_index=True, width="stretch")
    c1, c2, c3 = st.columns(3)
    c1.metric("Cost now", f"{old['material_cost']:.3f}", f"PoF {old['pof']:.2f}", "off")
    c2.metric(
        "Old recipe at new prices", f"{float(shocked.material_cost(old['x'][None, :])[0]):.3f}"
    )
    c3.metric(
        "Re-optimized at new prices", f"{new['material_cost']:.3f}", f"PoF {new['pof']:.2f}", "off"
    )


def about_tab() -> None:
    st.markdown(
        """
**What this is.** A decision tool that recommends the next formulation experiment. It looks for
the cheapest recipe that passes every spec, and weighs what an experiment will teach against
what it costs to run (cost-aware Bayesian optimization).

**Data labels.**
- The **enamel problem is a SYNTHETIC simulator**: plausible in direction and order of
  magnitude, not calibrated to any real enamel or any company's products. Prices and run costs
  are illustrative.
- The **concrete problem** is an emulator (ExtraTrees) trained on the real UCI Concrete
  Compressive Strength data (Yeh 1998, CC BY 4.0, DOI 10.24432/C5PK67). Cement, slag and
  aggregate prices are USGS 2025 national averages; fly ash, water and superplasticizer
  prices are ASSUMPTIONS. Results are about method comparison, not real concrete pricing.
- "Truth pass" columns use the simulator's noise-free response; a real lab never sees this.

See README.md for methods, results and limitations.
        """
    )


def run_demo(problem: Problem, budget: float, method: str, cfg: AppConfig) -> None:
    """Replay the demo script once per session (for screenshots): 6 + 5 runs, 1 manual entry."""
    if st.session_state.get("demo_done") or st.session_state.history:
        return
    initial_design(problem, budget, cfg.bo.n_init)
    for _ in range(5):
        suggest(problem, budget, method, cfg)
        sug = st.session_state.suggestion
        if sug is None:
            break
        run_on_simulator(problem, sug.x, sug)
    suggest(problem, budget, method, cfg)
    sug = st.session_state.suggestion
    if sug is not None:  # manual entry: the lab measures the suggested recipe itself
        x = sug.x
        obs = make_observation(problem, x, np.random.default_rng(12345))
        st.session_state.history.append(obs)
        suggest(problem, budget, method, cfg)
    st.session_state.demo_done = True


def main() -> None:
    cfg = get_config()
    problem, budget, method, seed = sidebar_problem(cfg)
    init_state(problem, budget, seed)
    st.title("Cost-aware formulation optimizer")
    st.caption(
        "Cheapest path to an in-spec formulation. "
        + (
            "Enamel data are SYNTHETIC; prices are illustrative."
            if problem.name == "enamel"
            else "Concrete: emulator trained on real UCI data; some prices are assumptions."
        )
    )
    if st.query_params.get("demo") == "1":
        run_demo(problem, budget, method, cfg)
    views = {
        "optimize": lambda: optimize_tab(problem, budget, method, cfg),
        "benchmark": lambda: benchmark_tab(cfg, problem.name),
        "price": lambda: price_tab(problem, cfg),
        "about": about_tab,
    }
    titles = {
        "optimize": "Optimize",
        "benchmark": "Benchmark",
        "price": "Price shock",
        "about": "About",
    }
    view = st.query_params.get("view")
    if view in views:  # one view without tabs (used for headless screenshots)
        st.subheader(titles[view])
        views[view]()
        return
    tabs = st.tabs(list(titles.values()))
    for tab, render in zip(tabs, views.values(), strict=True):
        with tab:
            render()


main()
