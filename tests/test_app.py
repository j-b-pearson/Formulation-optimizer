"""Streamlit app smoke test (headless, no browser): loads, runs the design, suggests."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def _button(at: AppTest, label_start: str):
    return next(b for b in at.button if b.label.startswith(label_start))


def test_app_loads_and_suggests() -> None:
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception
    assert at.title[0].value == "Cost-aware formulation optimizer"
    _button(at, "Run initial design").click().run()
    assert not at.exception
    _button(at, "Suggest next experiment").click().run()
    assert not at.exception
    assert any(h.value == "Suggested next experiment" for h in at.subheader)
    _button(at, "Run suggestion on simulator").click().run()
    assert not at.exception
    assert at.metric[0].value == "7"  # 6 initial runs + 1 suggestion


def test_switching_optimizer_keeps_campaign() -> None:
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    _button(at, "Run initial design").click().run()
    opt = next(s for s in at.sidebar.selectbox if s.label == "Optimizer")
    other = next(o for o in opt.options if o != opt.value)
    opt.select(other).run()
    assert not at.exception
    assert len(at.session_state.history) == 6  # was wiped to 0
