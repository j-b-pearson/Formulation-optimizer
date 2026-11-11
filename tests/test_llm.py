"""Lab note: parsing, retry on invalid JSON, schema validation (mocked client).

The risk label is computed in code; notes whose label or numbers differ from the
facts are rejected, with a template fallback. Every test uses a fake client (no network).
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from formopt.config import load_config
from formopt.llm import (
    LabNote,
    build_messages,
    checked_lab_note,
    draft_lab_note,
    risk_label,
    strip_think,
    suggestion_facts,
    template_lab_note,
    validate_lab_note,
)

GOOD = {
    "title": "Run 7: hot oven, fast line",
    "why_this_run": "Cheap to run and a 40 % predicted chance of passing.",
    "what_to_watch": ["breakdown near 30", "flexibility after overcure"],
    "risk_label": "High",
    "risk": "High risk: breakdown may fall just below 30.",
}
TH = load_config().lab_note.risk_thresholds


class FakeClient:
    """Mimics client.chat.completions.create and replays canned replies."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        content = self.replies.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def test_valid_json_is_parsed_with_schema_request() -> None:
    client = FakeClient([json.dumps(GOOD)])
    note = draft_lab_note({"recipe": {"x": 1}}, client=client, model="m")
    assert isinstance(note, LabNote) and note.title.startswith("Run 7")
    call = client.calls[0]
    assert call["response_format"]["type"] == "json_schema"
    assert call["temperature"] == 0 and call["model"] == "m"


def test_retry_after_invalid_json() -> None:
    client = FakeClient(["not json at all", "<think>hmm</think>" + json.dumps(GOOD)])
    note = draft_lab_note({}, client=client, model="m", max_retries=1)
    assert note.risk == GOOD["risk"]
    assert len(client.calls) == 2
    assert "invalid" in client.calls[1]["messages"][-1]["content"]


def test_gives_up_after_retries() -> None:
    client = FakeClient(['{"title": "missing fields"}', '{"title": "still missing"}'])
    with pytest.raises(ValueError):
        draft_lab_note({}, client=client, model="m", max_retries=1)


def test_strip_think_and_fences() -> None:
    assert strip_think('<think>a\nb</think>\n```json\n{"a": 1}\n```') == '{"a": 1}'
    messages = build_messages({"recipe": {"x": 1}})
    assert messages[0]["role"] == "system" and '"x": 1' in messages[1]["content"]


# ----- risk label computed in code, validation, template fallback ----------------------
def enamel_facts(pof: float = 0.305) -> dict:
    """Facts for a fake suggestion on the real enamel problem (no GP fit, no network)."""
    from formopt.problems.enamel import EnamelProblem

    problem = EnamelProblem(load_config().enamel)
    sug = SimpleNamespace(
        x=np.array([0.1362, 41.7287, 0.5141, 433.096, 38.3205]),
        pred_latent={
            "breakdown_kv": (27.7305, 4.66),
            "flex_pass": (1.0785, 0.145),
            "viscosity_cp": (float(np.log(4483.0614)), 0.11),
        },
        pof_per_spec={"breakdown_kv": 0.313, "flex_pass": 0.972, "viscosity_cp": 1.0},
        pof_at_x=pof,
        run_cost=381.83,
        material_cost=3.8526,
    )
    return suggestion_facts(problem, sug, TH.high_below, TH.low_at_or_above)


def good_note(facts: dict) -> dict:
    return {
        "title": "Hot-oven, fast-line enamel trial",
        "why_this_run": "The model estimates a 30.5 % chance of passing all specs at a run cost "
        "of $381.83; these are model estimates.",
        "what_to_watch": ["breakdown_kv against the spec of 30", "flex_pass near 0.8"],
        "risk_label": facts["risk_label"],
        "risk": f"{facts['risk_label']} risk: only a 30.5 % chance of passing every spec.",
    }


@pytest.mark.parametrize(
    ("pof", "label"),
    [
        (0.0, "High"),
        (0.4999, "High"),
        (0.5, "Medium"),
        (0.7999, "Medium"),
        (0.8, "Low"),
        (1.0, "Low"),
    ],
)
def test_risk_label_thresholds_from_config(pof: float, label: str) -> None:
    assert risk_label(pof, TH.high_below, TH.low_at_or_above) == label


def test_facts_carry_the_label_computed_in_code() -> None:
    facts = enamel_facts(0.305)
    assert facts["risk_label"] == "High" and facts["pass_probability_pct"] == 30.5
    assert enamel_facts(0.85)["risk_label"] == "Low"


def test_validation_accepts_a_faithful_note_and_the_template() -> None:
    facts = enamel_facts()
    assert validate_lab_note(LabNote(**good_note(facts)), facts) == []
    for pof in (0.2, 0.65, 0.9):
        f = enamel_facts(pof)
        assert validate_lab_note(template_lab_note(f), f) == []


def test_validation_rejects_the_v1_failure_wrong_label() -> None:
    """The v1 example labelled 'Low' a 30.5 % chance of passing all specs."""
    facts = enamel_facts()
    bad = good_note(facts) | {
        "risk_label": "Low",
        "risk": "Low (30.5% chance of meeting all specs), high viscosity risk mitigated",
    }
    problems = validate_lab_note(LabNote(**bad), facts)
    assert any("computed label is 'High'" in p for p in problems)
    assert any("other label words" in p for p in problems)


def test_validation_rejects_invented_or_changed_numbers() -> None:
    facts = enamel_facts()
    invented = good_note(facts) | {"why_this_run": "Expect breakdown of 42 kV on 5 wraps."}
    assert any("not in the facts" in p for p in validate_lab_note(LabNote(**invented), facts))
    changed = good_note(facts) | {"risk": "High risk: a 35 % chance of passing every spec."}
    problems = validate_lab_note(LabNote(**changed), facts)
    assert any("pass probability" in p for p in problems)


def test_checked_note_falls_back_to_template_after_retry() -> None:
    facts = enamel_facts()
    wrong = json.dumps(good_note(facts) | {"risk_label": "Low", "risk": "Low risk: 30.5 %."})
    client = FakeClient([wrong, wrong])
    checked = checked_lab_note(facts, client=client, model="m", max_retries=1)
    assert checked.source == "template" and len(client.calls) == 2
    assert checked.note.risk_label == "High"
    assert "rejected" in client.calls[1]["messages"][-1]["content"]
    assert all(a["problems"] for a in checked.attempts)
    assert validate_lab_note(checked.note, facts) == []


def test_checked_note_accepts_a_corrected_retry() -> None:
    facts = enamel_facts()
    wrong = json.dumps(good_note(facts) | {"risk_label": "Medium"})
    client = FakeClient([wrong, json.dumps(good_note(facts))])
    checked = checked_lab_note(facts, client=client, model="m", max_retries=1)
    assert checked.source == "llm" and checked.note.risk_label == "High"
    assert len(checked.attempts) == 2 and checked.attempts[-1]["problems"] == []


def test_unreachable_server_uses_the_template() -> None:
    class DownClient:
        def __init__(self) -> None:
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, **kwargs):
            raise ConnectionError("Ollama is not running")

    checked = checked_lab_note(enamel_facts(), client=DownClient(), model="m")
    assert checked.source == "template" and "not running" in checked.fallback_reason
