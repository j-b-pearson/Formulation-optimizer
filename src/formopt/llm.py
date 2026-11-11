"""Optional: a local LLM drafts a short lab note for a suggested experiment.

Talks to any OpenAI-compatible server; by default a local Ollama server
(`LLM_BASE_URL=http://localhost:11434/v1`, `LLM_MODEL=qwen3:14b`, `LLM_API_KEY=ollama`).
The note is plain help text for a chemist ("why this run, what to watch"). It is never used by
the optimizer or the benchmark. Install the client with `uv sync --group llm`.

The risk label is computed in code from the probability of passing all specs
(thresholds in `config.yaml` `lab_note.risk_thresholds`). The LLM only writes sentences around
the label and the numbers the code supplies. `validate_lab_note` rejects a note whose label
differs from the computed one, that omits the pass probability, or that contains any number
not in the supplied facts; `checked_lab_note` retries once and then falls back to a template.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_MODEL = "qwen3:14b"
INTERVAL_LEVEL_PCT = 95


class LabNote(BaseModel):
    """A short, structured lab-notebook entry for one planned experiment."""

    title: str = Field(description="One line naming the run")
    why_this_run: str = Field(description="Two or three sentences: why the optimizer chose it")
    what_to_watch: list[str] = Field(description="Two to four things to check during the run")
    risk_label: str = Field(description="Exactly the risk_label given in the facts")
    risk: str = Field(
        description="One sentence that states the risk_label and the pass probability given"
    )


SYSTEM_PROMPT = (
    "You are a careful formulation chemist writing a short lab-notebook entry for a planned "
    "experiment. Use only the numbers you are given; do not invent measurements or numbers. "
    "Say clearly that predictions are model estimates. The risk label is already decided: copy "
    "risk_label exactly, and write the risk sentence with that label and the given "
    "pass_probability_pct. Do not use the words high, medium or low anywhere in the risk "
    "sentence except for the given label. Reply with JSON matching the schema."
)


def risk_label(pof: float, high_below: float, low_at_or_above: float) -> str:
    """Risk label from the probability of passing all specs: High, Medium or Low."""
    if pof < high_below:
        return "High"
    if pof >= low_at_or_above:
        return "Low"
    return "Medium"


def llm_settings() -> tuple[str, str, str]:
    """(base_url, model, api_key) from the environment, with Ollama defaults."""
    return (
        os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL),
        os.environ.get("LLM_MODEL", DEFAULT_MODEL),
        os.environ.get("LLM_API_KEY", "ollama"),
    )


def make_client() -> Any:
    """OpenAI-compatible client (imported lazily: the core install does not need it)."""
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - depends on the optional group
        raise ImportError("Install the optional client: uv sync --group llm") from exc
    base_url, _, api_key = llm_settings()
    return OpenAI(base_url=base_url, api_key=api_key)


def strip_think(text: str) -> str:
    """Remove Qwen3 '<think>...</think>' blocks and code fences around JSON."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
    return text


def build_messages(suggestion: dict) -> list[dict[str, str]]:
    """Chat messages for one suggestion (recipe, predictions, PoF, costs, risk label)."""
    user = (
        "Draft a lab note for this suggested experiment. Facts (JSON):\n"
        + json.dumps(suggestion, indent=2)
        + "\nKeep it short. /no_think"
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def draft_lab_note(
    suggestion: dict,
    client: Any | None = None,
    model: str | None = None,
    max_retries: int = 1,
    validator: Callable[[LabNote], list[str]] | None = None,
    attempts: list[dict] | None = None,
    reasoning_effort: str | None = None,
) -> LabNote:
    """Ask the LLM for a schema-constrained JSON note; retry with the error.

    If `validator` is given, a note that parses but fails validation also triggers a retry
    with the list of problems. Each attempt's raw text and problems are appended to
    `attempts` when a list is passed. Raises ValueError when the retries run out.
    """
    client = client or make_client()
    model = model or llm_settings()[1]
    messages = build_messages(suggestion)
    schema = {"name": "lab_note", "schema": LabNote.model_json_schema()}
    extra = {} if reasoning_effort is None else {"reasoning_effort": reasoning_effort}
    last_error: str = ""
    for _ in range(max_retries + 1):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_schema", "json_schema": schema},
            temperature=0,
            **extra,
        )
        text = strip_think(response.choices[0].message.content)
        try:
            note = LabNote.model_validate_json(text)
            problems = validator(note) if validator else []
        except ValidationError as exc:
            note, problems = None, [f"invalid JSON for the schema: {exc}"]
        if attempts is not None:
            attempts.append({"text": text, "problems": problems})
        if note is not None and not problems:
            return note
        last_error = "; ".join(problems)
        feedback = (
            f"That JSON was invalid: {last_error}. Reply with valid JSON."
            if note is None
            else f"That note was rejected: {last_error}. Fix these problems and reply with JSON."
        )
        messages = [
            *messages,
            {"role": "assistant", "content": text},
            {"role": "user", "content": feedback},
        ]
    raise ValueError(f"LLM did not return a valid lab note: {last_error}")


def suggestion_facts(
    problem: Any, sug: Any, high_below: float = 0.5, low_at_or_above: float = 0.8
) -> dict:
    """The facts the LLM may use: recipe, predicted outputs with 95 % ranges, PoF, costs, and
    the risk label computed in code from the probability of passing all specs."""
    predicted = {}
    for spec in problem.specs:
        mu, sd = sug.pred_latent[spec.output]
        predicted[spec.output] = {
            "predicted": round(float(spec.inverse_transform(mu)), 4),
            "range_95": [
                round(float(spec.inverse_transform(mu - 1.96 * sd)), 4),
                round(float(spec.inverse_transform(mu + 1.96 * sd)), 4),
            ],
            "spec_lower": spec.lower,
            "spec_upper": spec.upper,
            "probability_of_passing": round(sug.pof_per_spec[spec.output], 3),
        }
    pof = round(float(sug.pof_at_x), 3)
    return {
        "problem": problem.name,
        "recipe": {k: round(v, 4) for k, v in problem.as_dict(sug.x).items()},
        "units": {v.name: v.unit for v in problem.variables},
        "predicted_outputs": predicted,
        "probability_of_passing_all_specs": pof,
        "pass_probability_pct": round(100.0 * pof, 1),
        "risk_label": risk_label(pof, high_below, low_at_or_above),
        "risk_rule": {
            "high_below": high_below,
            "low_at_or_above": low_at_or_above,
            "text": f"High below {100 * high_below:g} %, Medium from {100 * high_below:g} % to "
            f"below {100 * low_at_or_above:g} %, Low at {100 * low_at_or_above:g} % or above",
        },
        "interval_level_pct": INTERVAL_LEVEL_PCT,
        "run_cost_usd": round(sug.run_cost, 2),
        "material_cost": round(sug.material_cost, 4),
        "note": "Predictions are Gaussian process estimates, not measurements.",
    }


# ----- validation ------------------------------------------------------------------------
_NUMBER = re.compile(r"(?<![\w.])\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![\w.])\d+(?:\.\d+)?")


def numbers_in_text(text: str) -> list[str]:
    """Numbers written in a text, as strings (thousands separators kept; no signs)."""
    return _NUMBER.findall(text or "")


def _walk_numbers(obj: Any) -> list[float]:
    if isinstance(obj, bool) or obj is None:
        return []
    if isinstance(obj, int | float):
        return [float(obj)]
    if isinstance(obj, dict):
        return [v for x in obj.values() for v in _walk_numbers(x)]
    if isinstance(obj, list | tuple):
        return [v for x in obj for v in _walk_numbers(x)]
    return []


def allowed_numbers(facts: dict) -> list[float]:
    """Every number in the facts, plus probabilities written as percentages."""
    values = _walk_numbers(facts)
    probs = [facts.get("probability_of_passing_all_specs", 0.0)]
    probs += [p["probability_of_passing"] for p in facts.get("predicted_outputs", {}).values()]
    rule = facts.get("risk_rule", {})
    probs += [rule.get("high_below", 0.0), rule.get("low_at_or_above", 0.0)]
    return values + [100.0 * float(p) for p in probs]


def number_matches(written: str, allowed: list[float]) -> bool:
    """True if `written` equals an allowed value rounded to the decimals it was written with."""
    value = float(written.replace(",", ""))
    decimals = len(written.split(".")[1]) if "." in written else 0
    tol = 0.5 * 10.0**-decimals + 1e-9
    return any(abs(value - abs(a)) <= tol for a in allowed)


def _note_text_fields(note: LabNote) -> dict[str, str]:
    return {
        "title": note.title,
        "why_this_run": note.why_this_run,
        "what_to_watch": " | ".join(note.what_to_watch),
        "risk": note.risk,
    }


def _pct_mentioned(text: str, facts: dict) -> bool:
    """The risk sentence quotes the pass probability (as a percentage or as a fraction)."""
    pct = float(facts["pass_probability_pct"])
    frac = float(facts["probability_of_passing_all_specs"])
    for token in numbers_in_text(text):
        v = float(token.replace(",", ""))
        if abs(v - pct) < 0.051 or abs(v - frac) < 0.0051:
            return True
    return False


def validate_lab_note(note: LabNote, facts: dict) -> list[str]:
    """Problems with a note, compared with the facts the code supplied (empty list = valid).

    Rejects: a risk label other than the computed one; a risk sentence that does not state the
    computed label and the pass probability, or that uses another label word; any number in
    the note that is not one of the supplied numbers (at the precision written).
    """
    problems: list[str] = []
    label = facts["risk_label"]
    if note.risk_label.strip().lower() != label.lower():
        problems.append(f"risk_label is {note.risk_label!r} but the computed label is {label!r}")
    words = re.findall(r"\b(high|medium|low)\b", note.risk, flags=re.I)
    if label.lower() not in {w.lower() for w in words}:
        problems.append(f"the risk sentence does not state the label {label!r}")
    others = sorted({w.capitalize() for w in words} - {label})
    if others:
        problems.append(f"the risk sentence also uses other label words {others}")
    if not _pct_mentioned(note.risk, facts):
        pct = facts["pass_probability_pct"]
        problems.append(f"the risk sentence does not quote the pass probability {pct} %")
    allowed = allowed_numbers(facts)
    for name, text in _note_text_fields(note).items():
        bad = [n for n in numbers_in_text(text) if not number_matches(n, allowed)]
        if bad:
            problems.append(f"{name} contains numbers not in the facts: {bad}")
    return problems


# ----- template fallback -----------------------------------------------------------------
def _limits(p: dict) -> str:
    lo, hi = p.get("spec_lower"), p.get("spec_upper")
    if lo is not None and hi is not None:
        return f"spec {lo:g} to {hi:g}"
    return f"spec at least {lo:g}" if lo is not None else f"spec at most {hi:g}"


def template_lab_note(facts: dict) -> LabNote:
    """A plain note built only from the facts (used when the LLM note fails validation)."""
    label, pct = facts["risk_label"], facts["pass_probability_pct"]
    preds = facts["predicted_outputs"]
    order = sorted(preds, key=lambda o: preds[o]["probability_of_passing"])
    watch = [
        f"{o}: model estimate {preds[o]['predicted']:g} "
        f"({INTERVAL_LEVEL_PCT} % range {preds[o]['range_95'][0]:g} to "
        f"{preds[o]['range_95'][1]:g}), {_limits(preds[o])}, "
        f"chance of passing {100 * preds[o]['probability_of_passing']:.1f} %"
        for o in order
    ]
    return LabNote(
        title=f"Planned {facts['problem']} trial suggested by the optimizer",
        why_this_run=(
            f"The optimizer chose this recipe for its expected improvement per run. The model "
            f"estimates a {pct:g} % chance that it passes every spec; the run costs "
            f"${facts['run_cost_usd']:,.2f} and the material costs {facts['material_cost']:g} "
            f"per unit. These are model estimates, not measurements."
        ),
        what_to_watch=watch,
        risk_label=label,
        risk=f"{label} risk: the model gives a {pct:g} % chance of passing all specs (label set "
        f"in code from that chance, thresholds {100 * facts['risk_rule']['high_below']:g} % and "
        f"{100 * facts['risk_rule']['low_at_or_above']:g} %).",
    )


@dataclass
class CheckedLabNote:
    """The lab note that is shown, where it came from, and why any LLM attempt was rejected."""

    note: LabNote
    source: str  # "llm" or "template"
    attempts: list[dict] = field(default_factory=list)
    fallback_reason: str = ""


def checked_lab_note(
    facts: dict,
    client: Any | None = None,
    model: str | None = None,
    max_retries: int = 1,
    reasoning_effort: str | None = None,
) -> CheckedLabNote:
    """LLM note validated against the facts; after `max_retries` failed retries, the template.

    Any failure to get a valid note (bad JSON, a wrong label or number, or an unreachable
    server) falls back to the template, and the reason is recorded, never hidden.
    """
    attempts: list[dict] = []
    try:
        note = draft_lab_note(
            facts,
            client,
            model,
            max_retries,
            validator=lambda n: validate_lab_note(n, facts),
            attempts=attempts,
            reasoning_effort=reasoning_effort,
        )
        return CheckedLabNote(note, "llm", attempts)
    except Exception as exc:  # noqa: BLE001 - any failure means "use the template", logged
        logger.warning("Lab note falls back to the template: %s", exc)
        return CheckedLabNote(template_lab_note(facts), "template", attempts, str(exc))
