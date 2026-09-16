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
    #: Cap on probe cells executed across the whole run. ``None`` disables it,
    #: which is the shipped behaviour: ``max_iterations`` bounds *proposal
    #: cycles* and has never bounded cells. Those are different quantities and
    #: the gap between them is wide -- ``loop`` appends every proposed name to
    #: an axis and charges one iteration for the cycle, so a wide-candidate run
    #: can spend 160 cells against a budget of 8. Changing what
    #: ``--max-iterations`` means is a CLI product decision, so this is a
    #: second, opt-in budget denominated in the thing that actually costs GPU.
    max_probe_cells: int | None = None

    def __post_init__(self) -> None:
        if self.max_iterations < 1:
            raise PolicyViolation("max_iterations must be >= 1")
        # None disables the wall-clock budget; a 0/negative cap is user error
        # (the loop would stop immediately with walltime_exhausted).
        if self.max_walltime_sec is not None and self.max_walltime_sec <= 0:
            raise PolicyViolation("max_walltime_sec must be > 0 when set")
        if self.max_probe_cells is not None and self.max_probe_cells < 1:
            raise PolicyViolation("max_probe_cells must be >= 1 when set")

    def check_iteration_budget(self, iterations_done: int) -> None:
        if iterations_done >= self.max_iterations:
            raise PolicyViolation(
                f"iteration budget exhausted ({self.max_iterations} max)"
            )

    def cells_affordable(self, cells_spent: int) -> int | None:
        """How many further probe cells the cell budget allows, or None if off.

        Returns 0 rather than a negative when the budget is already overspent,
        so callers can treat the result as a plain capacity.
        """
        if self.max_probe_cells is None:
            return None
        return max(0, self.max_probe_cells - cells_spent)

    def check_cell_budget(self, cells_spent: int) -> None:
        """Raise when no further probe cell can be afforded.

        Mirrors :meth:`check_iteration_budget` in shape so the two budgets
        surface through the same ``policy_stop`` outcome.
        """
        affordable = self.cells_affordable(cells_spent)
        if affordable is not None and affordable < 1:
            raise PolicyViolation(
                f"probe cell budget exhausted ({self.max_probe_cells} max, "
                f"{cells_spent} spent)"
            )

    def _clean_axis_names(self, names: list[str], axis_label: str) -> list[str]:
        """Shared validation for both probe axes.

        Both axes are resolved through the mitigations registry -- probe-mode
        treats a diagnostic as a mitigation name whose job is to reveal rather
        than to fix (``probe.recipe_builder._validate_axis_names``) -- so the
        registry, shell-shape and cell-name-safety rules are identical and
        must not be able to drift between the two.
        """
        cleaned: list[str] = []
        for name in names:
            if not isinstance(name, str) or not name.strip():
                raise PolicyViolation(f"invalid {axis_label} name: {name!r}")
            if " " in name or name.startswith("-"):
                raise PolicyViolation(
                    f"{axis_label} {name!r} looks like shell/argv, not a registry name"
                )
            if not _CELL_SAFE_MITIGATION_RE.match(name):
                raise PolicyViolation(
                    f"{axis_label} {name!r} contains characters unsafe for a probe "
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
        return cleaned

    def validate_step(self, step: AgentStep) -> AgentStep:
        """Normalize and enforce registry + category constraints."""
        if step.category not in AUTOPSY_CATEGORIES:
            raise PolicyViolation(
                f"invalid category {step.category!r}; "
                f"allowed: {sorted(AUTOPSY_CATEGORIES)}"
            )
        cleaned = self._clean_axis_names(step.next_mitigations, "mitigation")
        # "none" is dropped on both axes for the same reason: it is the
        # baseline, the loop guarantees it is on every axis already, and
        # admitting it would charge cells for a cell that exists.
        cleaned_diagnostics = self._clean_axis_names(step.next_diagnostics, "diagnostic")
        return AgentStep(
            category=step.category,
            hypothesis=step.hypothesis,
            next_mitigations=cleaned,
            confidence=max(0.0, min(1.0, step.confidence)),
            stop=step.stop,
            stop_reason=step.stop_reason,
            next_diagnostics=cleaned_diagnostics,
        )

    def needs_approval(self, mitigation: str) -> bool:
        if not self.require_approval:
            return False
        return mitigation in _APPROVAL_REQUIRED

    def pending_approvals(self, mitigations: list[str]) -> list[str]:
        return [m for m in mitigations if self.needs_approval(m)]


__all__ = ["AgentPolicy", "PolicyViolation"]
