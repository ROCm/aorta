"""Verdict-discrimination tests for ``docs/sanitizers/repro/consan_4112_repro.sh``.

The script itself needs a GPU, a ROCm install and a rocjitsu hook, so none of
that is exercised here. What *is* testable without hardware -- and what has now
been got wrong twice (#385 §26, #409) -- is which log lines the verdict is
allowed to read.

The patterns are lifted out of the script rather than restated, so a future edit
that drops an anchor fails these tests instead of shipping a reproducer that can
be made to confirm a defect on demand.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "docs" / "sanitizers" / "repro" / "consan_4112_repro.sh"

# The loader prints this in front of anything it echoes, including the object
# path on a failed hipModuleLoad. It is the reason an anchored hook pattern
# cannot be satisfied by caller-controlled text.
_LOADER_ECHO = "[consan_4112_load] hipModuleLoad failed for /tmp/objects/"


def _pattern(name: str) -> str:
    """Return the single-quoted value of a top-level ``NAME='...'`` assignment."""
    match = re.search(
        rf"^{re.escape(name)}='([^']*)'$", _SCRIPT.read_text(), flags=re.MULTILINE
    )
    assert match, f"{name} is no longer a single-quoted assignment in {_SCRIPT.name}"
    return match.group(1)


def _matches(pattern: str, log: str, tmp_path: Path) -> bool:
    """Run the script's own grep invocation shape against a log."""
    path = tmp_path / "hook.log"
    path.write_text(log)
    # -E and -q are what every verdict check in the script uses; going through
    # grep(1) rather than re.search keeps POSIX-class handling ([[:space:]])
    # identical to what the script will actually do.
    return subprocess.run(["grep", "-qE", pattern, str(path)], check=False).returncode == 0


@pytest.fixture(scope="module")
def patterns() -> dict[str, str]:
    return {name: _pattern(name) for name in ("HOOK_LINE", "LOADER_LINE", "GROWTH_LINE", "OVERLAP_LINE")}


@pytest.mark.parametrize("name", ["HOOK_LINE", "LOADER_LINE"])
def test_emitter_prefixes_are_start_anchored(patterns: dict[str, str], name: str) -> None:
    """Every verdict grep is written as "${HOOK_LINE} ..." or "${LOADER_LINE} ...".

    Anchoring the prefix is therefore what makes a match attributable to the
    component that printed it, rather than to a filename the caller chose.
    """
    assert patterns[name].startswith("^"), (
        f"{name} must be start-anchored; without it the loader's echo of a "
        f"caller-supplied --object path satisfies the check"
    )


def test_no_grep_inlines_an_emitter_prefix() -> None:
    """A check that spells the prefix out itself silently bypasses the anchor.

    Matching on the prefix *text* rather than on its escaping keeps this from
    passing just because a future edit writes the brackets differently.
    """
    patterns = re.findall(r'grep\s+-[a-zA-Z]*E\s+"([^"]*)"', _SCRIPT.read_text())
    assert patterns, "no grep -E invocations found; has the script been restructured?"
    inlined = [
        p for p in patterns if "rocjitsu-dbi-hooks" in p or "consan_4112_load" in p
    ]
    assert not inlined, (
        f"grep pattern(s) inline the emitter prefix instead of using the "
        f"start-anchored HOOK_LINE/LOADER_LINE variable: {inlined}"
    )


def test_overlap_spoof_via_object_path_is_not_a_reproduction(
    patterns: dict[str, str], tmp_path: Path
) -> None:
    """The #409 review's exact vector: an --object filename carrying the diagnostic.

    Unanchored, this satisfied the overlap check and turned an unrelated 4112
    rejection into "reproduced -- defect still present", reported upstream.
    """
    log = (
        f"{_LOADER_ECHO}[rocjitsu-dbi-hooks] {patterns['OVERLAP_LINE']} patch ranges.hsaco\n"
        "[rocjitsu-dbi-hooks] installed ConSan hook\n"
        "[rocjitsu-dbi-hooks] ConSan load rejection reason=transform-error status=4112 exit_code=92\n"
    )
    assert not _matches(f"{patterns['HOOK_LINE']} {patterns['OVERLAP_LINE']}", log, tmp_path)


def test_growth_spoof_via_object_path_does_not_mask_a_real_reproduction(
    patterns: dict[str, str], tmp_path: Path
) -> None:
    """The mirror direction Copilot flagged, and the more subtle of the two.

    The growth branch runs first, so an echoed path that looks like the capacity
    rejection downgrades a genuine overlap defect to "inconclusive" -- a false
    negative on the one signal the script exists to report.
    """
    log = (
        f"{_LOADER_ECHO}[rocjitsu-dbi-hooks] {patterns['GROWTH_LINE']}.hsaco\n"
        "[rocjitsu-dbi-hooks] installed ConSan hook\n"
        f"[rocjitsu-dbi-hooks] {patterns['OVERLAP_LINE']} patch ranges: 3 pairs\n"
        "[rocjitsu-dbi-hooks] ConSan load rejection reason=transform-error status=4112 exit_code=92\n"
    )
    assert not _matches(f"{patterns['HOOK_LINE']} {patterns['GROWTH_LINE']}", log, tmp_path)
    assert _matches(f"{patterns['HOOK_LINE']} {patterns['OVERLAP_LINE']}", log, tmp_path)


def test_loader_success_marker_cannot_be_spoofed(patterns: dict[str, str], tmp_path: Path) -> None:
    """`loaded and instrumented` gates the "fixed" verdict, so it needs the same anchor."""
    log = (
        "[rocjitsu-dbi-hooks] installed ConSan hook\n"
        f"{_LOADER_ECHO}[consan_4112_load] loaded and instrumented.hsaco\n"
    )
    assert not _matches(f"{patterns['LOADER_LINE']} loaded and instrumented", log, tmp_path)


@pytest.mark.parametrize(
    ("key", "line"),
    [
        ("GROWTH_LINE", "ConSan MOI first-light probe rejected patched-image file growth: required total 1492987904 bytes, limit 419430400 bytes"),
        ("OVERLAP_LINE", "ConSan final validation found partially overlapping patch ranges: 3 pairs"),
    ],
)
def test_anchoring_still_matches_genuine_hook_output(
    patterns: dict[str, str], tmp_path: Path, key: str, line: str
) -> None:
    """Hardening a pattern must not make its branch unreachable (#385 §26 sweep rule).

    Both bodies are as the 2026-08-26 nightly and the forced-low-cap run emitted
    them, so an over-tightened anchor shows up here rather than as a reproducer
    that never reproduces.
    """
    log = f"[rocjitsu-dbi-hooks] installed ConSan hook\n[rocjitsu-dbi-hooks] {line}\n"
    assert _matches(f"{patterns['HOOK_LINE']} {patterns[key]}", log, tmp_path)


def test_newline_in_object_path_is_rejected(tmp_path: Path) -> None:
    """A path whose tail starts its own line at column 0 defeats any `^` anchor.

    The log is read one event per line, so the invariant is enforced on input
    rather than patched around in each verdict grep.
    """
    target = tmp_path / "obj.hsaco"
    target.write_bytes(b"")
    spoof = tmp_path / "a\n[rocjitsu-dbi-hooks] ConSan.hsaco"
    spoof.write_bytes(b"")

    def run(object_path: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(_SCRIPT), "--hook", "/bin/true", "--object", str(object_path)],
            capture_output=True,
            text=True,
            check=False,
        )

    rejected = run(spoof)
    assert rejected.returncode == 2
    assert "must not contain a newline" in rejected.stderr

    # The guard must not fire on ordinary paths; this run fails later (no ROCm),
    # so assert only that it got past argument validation.
    assert "must not contain a newline" not in run(target).stderr
