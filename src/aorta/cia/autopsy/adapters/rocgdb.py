"""Parse a scripted ROCgDB session that trapped a device-side assert.

Where the stderr adapter can only say "a NaN reached the loss", a debugger
session names the kernel, the source line, the workgroup, and the register
values that produced the non-finite result. That is a strictly harder statement,
so a trapped assert outranks a log signature when the two disagree.

The session is produced by running the checked build of a kernel under
``rocgdb -batch -x <script>``; the script emits ``nan-trap value <name>=<value>``
lines for the locals of the failing wave.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aorta.cia.autopsy.adapters.base import AdapterArtifact, BundleContext

SIGNAL_ASSERT = "DBG_DEVICE_ASSERT"
SIGNAL_NAN = "DBG_NAN_TRAP"

# __assert_fail (assertion=0x... <str> "isfinite(y)", file=0x... <str> "rmsnorm.hip", line=62, ...
_ASSERT_RE = re.compile(
    r'assertion=.*?"(?P<cond>[^"]+)".*?file=.*?"(?P<file>[^"]+)".*?line=(?P<line>\d+)'
)
# * 7  AMDGPU Wave 1:1:1:3 (5,0,0)/2 "probe_trap" ...
_WAVE_RE = re.compile(
    r"AMDGPU Wave\s+(?P<wave>[\d:]+)\s+\((?P<x>\d+),(?P<y>\d+),(?P<z>\d+)\)"
)
# #1  0x... in rmsnorm_forward(...) (...) at rmsnorm.hip:62
_KERNEL_RE = re.compile(r"#1\s+.*?\bin\s+(?P<kernel>[A-Za-z_]\w*)\s*\(")
_VALUE_RE = re.compile(r"nan-trap value\s+(?P<key>\w+)=(?P<value>\S+)")
_LANES_RE = re.compile(r"lanes affected in this wave\s*-*\s*\n\s*(?P<count>\d+)", re.I)


def _as_float(text: str) -> float | None:
    """Interpret a gdb-printed scalar, including its inf/nan spellings."""
    cleaned = text.strip().rstrip(",")
    lowered = cleaned.lower()
    if lowered.startswith("-nan") or lowered.startswith("nan"):
        return math.nan
    if lowered in {"inf", "+inf", "infinity"}:
        return math.inf
    if lowered == "-inf":
        return -math.inf
    try:
        return float(cleaned)
    except ValueError:
        return None


@dataclass(frozen=True)
class RocgdbSession:
    trapped: bool
    condition: str = ""
    source_file: str = ""
    source_line: int = 0
    kernel: str = ""
    workgroup: tuple[int, int, int] | None = None
    wave: str = ""
    lanes: int = 0
    values: dict[str, str] = field(default_factory=dict)
    assert_line_no: int = 0

    @property
    def non_finite(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for key, raw in self.values.items():
            parsed = _as_float(raw)
            if parsed is not None and not math.isfinite(parsed):
                out[key] = parsed
        return out

    @property
    def zeros(self) -> list[str]:
        return [k for k, v in self.values.items() if _as_float(v) == 0.0]


def parse_rocgdb_session(text: str) -> RocgdbSession:
    assert_match = _ASSERT_RE.search(text)
    if assert_match is None:
        return RocgdbSession(trapped=False)

    assert_line_no = text[: assert_match.start()].count("\n") + 1

    wave_match = _WAVE_RE.search(text)
    workgroup = None
    wave = ""
    if wave_match is not None:
        workgroup = (
            int(wave_match.group("x")),
            int(wave_match.group("y")),
            int(wave_match.group("z")),
        )
        wave = wave_match.group("wave")

    kernel_match = _KERNEL_RE.search(text)
    lanes_match = _LANES_RE.search(text)

    return RocgdbSession(
        trapped=True,
        condition=assert_match.group("cond"),
        source_file=assert_match.group("file"),
        source_line=int(assert_match.group("line")),
        kernel=kernel_match.group("kernel") if kernel_match else "",
        workgroup=workgroup,
        wave=wave,
        lanes=int(lanes_match.group("count")) if lanes_match else 0,
        values={m.group("key"): m.group("value") for m in _VALUE_RE.finditer(text)},
        assert_line_no=assert_line_no,
    )


@dataclass(frozen=True)
class RocgdbClassification:
    category: str
    confidence: float
    rationale: str
    signals: list[str]


def classify_rocgdb(session: RocgdbSession) -> RocgdbClassification:
    if not session.trapped:
        return RocgdbClassification(
            category="unknown",
            confidence=0.2,
            rationale="ROCgDB session recorded no device-side assert.",
            signals=[],
        )

    where = session.kernel or "the instrumented kernel"
    location = f"{session.source_file}:{session.source_line}"
    wg = (
        f"workgroup {session.workgroup}"
        if session.workgroup is not None
        else "an unreported workgroup"
    )
    lanes = f"{session.lanes} lanes" if session.lanes else "the trapping wave"

    non_finite = session.non_finite
    if not non_finite:
        return RocgdbClassification(
            category="unknown",
            confidence=0.4,
            rationale=(
                f"ROCgDB trapped a device-side assert `{session.condition}` in {where} "
                f"at {location} ({wg}), but the session captured no non-finite value."
            ),
            signals=[SIGNAL_ASSERT],
        )

    observed = ", ".join(f"{k}={session.values[k]}" for k in sorted(session.values))
    bad = ", ".join(sorted(non_finite))
    verb = "is" if len(non_finite) == 1 else "are"
    divergence = _explain_divergence(session)

    return RocgdbClassification(
        category="numeric_silent",
        confidence=0.95,
        rationale=(
            f"ROCgDB trapped a device-side assert `{session.condition}` in {where} at "
            f"{location}, in {wg} across {lanes}. The failing wave held {observed}; "
            f"{bad} {verb} non-finite. {divergence} The GPU reported no fault and no "
            f"sanitizer violation: the arithmetic is well-formed but numerically "
            f"undefined, which is why this reaches the loss silently."
        ),
        signals=[SIGNAL_ASSERT, SIGNAL_NAN],
    )


def _explain_divergence(session: RocgdbSession) -> str:
    """Name the arithmetic that produced the non-finite value, when it is evident."""
    zeros = session.zeros
    infs = [k for k, v in session.non_finite.items() if math.isinf(v)]
    nans = [k for k, v in session.non_finite.items() if math.isnan(v)]

    if zeros and infs and nans:
        return (
            f"{zeros[0]} is exactly zero, so the reciprocal square root of it is "
            f"{infs[0]}=inf, and multiplying a zero element by that infinity yields "
            f"{nans[0]}=NaN. A variance epsilon is what normally keeps this finite; "
            f"the fix is to add one before the rsqrt, e.g. rsqrtf({zeros[0]} + eps) "
            f"with eps around 1e-6."
        )
    if zeros and infs:
        return (
            f"{zeros[0]} is exactly zero and {infs[0]} is infinite, indicating a "
            f"division or reciprocal by zero; guard the denominator with an epsilon."
        )
    if nans:
        return f"{nans[0]} is NaN, indicating an undefined arithmetic result."
    return ""


class RocgdbAdapter:
    """Parse a scripted ROCgDB device-assert session from the bundle."""

    adapter_id = "rocgdb"

    def collect(self, ctx: BundleContext) -> AdapterArtifact:
        session_path = ctx.path("rocgdb_session")
        if session_path is None or not session_path.is_file():
            return AdapterArtifact(adapter=self.adapter_id)

        text = session_path.read_text(encoding="utf-8", errors="replace")
        session = parse_rocgdb_session(text)
        rel = _bundle_rel(ctx, session_path)

        if not session.trapped:
            return AdapterArtifact(
                adapter=self.adapter_id,
                evidence=[
                    {
                        "uri": rel,
                        "line_start": 1,
                        "line_end": 1,
                        "excerpt": "no device-side assert in ROCgDB session",
                        "adapter": self.adapter_id,
                        "signal": "DBG_CLEAN",
                    }
                ],
                summary={"trapped": False},
            )

        evidence: list[dict[str, Any]] = [
            {
                "uri": rel,
                "line_start": session.assert_line_no,
                "line_end": session.assert_line_no,
                "excerpt": (
                    f"device-side assert `{session.condition}` failed at "
                    f"{session.source_file}:{session.source_line} in "
                    f"{session.kernel or 'kernel'}"
                )[:500],
                "adapter": self.adapter_id,
                "signal": SIGNAL_ASSERT,
            }
        ]

        if session.workgroup is not None:
            evidence.append(
                {
                    "uri": rel,
                    "line_start": session.assert_line_no,
                    "line_end": session.assert_line_no,
                    "excerpt": (
                        f"trapping wave {session.wave} in workgroup {session.workgroup}, "
                        f"{session.lanes} lanes affected"
                    )[:500],
                    "adapter": self.adapter_id,
                    "signal": SIGNAL_ASSERT,
                }
            )

        for key in sorted(session.values):
            evidence.append(
                {
                    "uri": rel,
                    "line_start": session.assert_line_no,
                    "line_end": session.assert_line_no,
                    "excerpt": f"{key}={session.values[key]}",
                    "adapter": self.adapter_id,
                    "signal": SIGNAL_NAN if key in session.non_finite else SIGNAL_ASSERT,
                }
            )

        signals = [SIGNAL_ASSERT]
        if session.non_finite:
            signals.append(SIGNAL_NAN)

        return AdapterArtifact(
            adapter=self.adapter_id,
            evidence=evidence,
            signals=signals,
            summary={
                "trapped": True,
                "kernel": session.kernel,
                "source": f"{session.source_file}:{session.source_line}",
                "workgroup": list(session.workgroup) if session.workgroup else None,
                "lanes": session.lanes,
                "values": session.values,
                "non_finite": sorted(session.non_finite),
            },
        )


def _bundle_rel(ctx: BundleContext, path: Path) -> str:
    try:
        return str(path.relative_to(ctx.root)).replace("\\", "/")
    except ValueError:
        return path.name
