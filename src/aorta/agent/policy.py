"""Guardrails for agent proposals and loop budgets."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from aorta.agent.llm import AUTOPSY_CATEGORIES, AgentStep
from aorta.registry import get_mitigation
from aorta.registry.errors import UnknownMitigationError


class PolicyViolation(ValueError):
    """Agent step or config violated safety policy."""


# A proposed mitigation name must round-trip *unchanged* through the probe
# cell-name builder. ``_safe_cell_segment`` (aorta/probe/recipe_builder.py)
# scrubs anything outside ``[A-Za-z0-9_.-]`` to ``_`` and prepends ``_`` to a
# leading ``.``/``-``; the agent later recovers tried/winning mitigations by
# parsing the ``<mitigation>-<diagnostic>`` cell directory name back
# (state.winning_mitigation / wake). A registered-but-unsafe name (e.g. a
# sidecar/plugin mitigation containing ``/``) would be silently scrubbed in
# the cell name and never match the registry name again, breaking
# convergence / resume / allowlist checks. This mirrors the cell-name segment
# rule (``^[A-Za-z0-9_][A-Za-z0-9_.\\-]*$``) so we reject such names up front.
_CELL_SAFE_MITIGATION_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.\-]*$")


# Mitigations that may require explicit operator approval before run.
_APPROVAL_REQUIRED: frozenset[str] = frozenset(
    {
        "hip_launch_blocking",
        "hsa_disable_cache",
    }
)


@dataclass(frozen=True)
class AgentPolicy:
    """Bounded autonomy knobs for :func:`aorta.agent.loop.run_agent_loop`."""

    max_iterations: int = 8
    max_walltime_sec: float | None = None
    require_approval: bool = False
    sidecar_files: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        if self.max_iterations < 1:
            raise PolicyViolation("max_iterations must be >= 1")
        # None disables the wall-clock budget; a 0/negative cap is user error
        # (the loop would stop immediately with walltime_exhausted).
        if self.max_walltime_sec is not None and self.max_walltime_sec <= 0:
            raise PolicyViolation("max_walltime_sec must be > 0 when set")

    def check_iteration_budget(self, iterations_done: int) -> None:
        if iterations_done >= self.max_iterations:
            raise PolicyViolation(
                f"iteration budget exhausted ({self.max_iterations} max)"
            )

    def validate_step(self, step: AgentStep) -> AgentStep:
        """Normalize and enforce registry + category constraints."""
        if step.category not in AUTOPSY_CATEGORIES:
            raise PolicyViolation(
                f"invalid category {step.category!r}; "
                f"allowed: {sorted(AUTOPSY_CATEGORIES)}"
            )
        cleaned: list[str] = []
        for name in step.next_mitigations:
            if not isinstance(name, str) or not name.strip():
                raise PolicyViolation(f"invalid mitigation name: {name!r}")
            if " " in name or name.startswith("-"):
                raise PolicyViolation(
                    f"mitigation {name!r} looks like shell/argv, not a registry name"
                )
            if not _CELL_SAFE_MITIGATION_RE.match(name):
                raise PolicyViolation(
                    f"mitigation {name!r} contains characters unsafe for a probe "
                    f"cell name; it must match [A-Za-z0-9_][A-Za-z0-9_.-]* so it "
                    f"round-trips through the cell directory name (the agent "
                    f"recovers tried/winning mitigations by parsing it back)"
                )
            try:
                get_mitigation(
                    name,
                    extra_files=list(self.sidecar_files) if self.sidecar_files else None,
                )
            except UnknownMitigationError as exc:
                raise PolicyViolation(str(exc)) from exc
            if name == "none":
                continue
            if name not in cleaned:
                cleaned.append(name)
        return AgentStep(
            category=step.category,
            hypothesis=step.hypothesis,
            next_mitigations=cleaned,
            confidence=max(0.0, min(1.0, step.confidence)),
            stop=step.stop,
            stop_reason=step.stop_reason,
        )

    def needs_approval(self, mitigation: str) -> bool:
        if not self.require_approval:
            return False
        return mitigation in _APPROVAL_REQUIRED

    def pending_approvals(self, mitigations: list[str]) -> list[str]:
        return [m for m in mitigations if self.needs_approval(m)]


__all__ = ["AgentPolicy", "PolicyViolation"]
