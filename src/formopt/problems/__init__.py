"""Problems (virtual labs): the synthetic enamel simulator and the concrete emulator."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from formopt.config import AppConfig
    from formopt.problems.base import Problem

PROBLEM_NAMES = ("enamel", "concrete")


def build_problem(name: str, cfg: AppConfig) -> Problem:
    """Create a problem by name. The concrete problem loads the trained emulator from disk."""
    if name == "enamel":
        from formopt.problems.enamel import EnamelProblem

        return EnamelProblem(cfg.enamel)
    if name == "concrete":
        from formopt.problems.concrete import load_concrete_problem

        return load_concrete_problem(cfg)
    raise ValueError(f"Unknown problem {name!r}; choose from {PROBLEM_NAMES}")


def problem_budget(name: str, cfg: AppConfig) -> float:
    """Lab budget per campaign for a problem, from config."""
    return {"enamel": cfg.enamel.budget, "concrete": cfg.concrete.budget}[name]
