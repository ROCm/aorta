"""Verdict tests for ``docs/sanitizers/repro/consan_4112_repro.sh``.

Running the script for real needs a GPU, a ROCm install and a rocjitsu hook, so
none of that is exercised here. Everything after the hooked run is just a log
being read, and that is where this script has been wrong three times now
(#385 §21, #385 §26, #409) -- so that part is tested.

Two layers, because they fail differently:

* ``_verdict`` executes the script's real verdict section against a synthetic
  log, and asserts the exit code and message. This is what catches a deleted
  branch, an inverted branch order, or a branch reaching the wrong conclusion.
* the pattern tests below check individual regexes in isolation, which is what
  catches an anchor being dropped from one check while the branch structure
  stays intact.

Neither restates anything: both lift the patterns and the code out of the script
itself, so an edit that breaks the contract fails here rather than shipping a
reproducer that can be made to confirm a defect on demand.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "docs" / "sanitizers" / "repro" / "consan_4112_repro.sh"

# Exactly what consan_4112_load.hip:19 prints when hipModuleLoad fails -- note
# there is no [consan_4112_load] prefix on this one, unlike the success marker at
# :22. That is the line that puts a caller-supplied path into the log, so the
# spoof fixtures below have to use its real shape or they prove nothing.
_LOADER_FAIL = "hipModuleLoad(/tmp/objects/{path}) failed: shared object initialization failed"
# The success marker, which does carry the prefix, and also echoes the path.
_LOADER_OK = "[consan_4112_load] loaded and instrumented {path} (no dispatch)"

_HOOK = "[rocjitsu-dbi-hooks] "
_INSTALLED = f"{_HOOK}installed ConSan hook"
_REJECTION = (
    f"{_HOOK}ConSan load rejection reason=transform-error status=4112 "
    "policy=strict action=terminate exit_code=92"
)
_GROWTH = (
    f"{_HOOK}ConSan MOI first-light probe rejected patched-image file growth: "
    "required total 1492987904 bytes, limit 419430400 bytes"
)
_OVERLAP = f"{_HOOK}ConSan final validation found partially overlapping patch ranges: 3 pairs"

# The verdict section is everything from the emitter-prefix definitions to the
# end of the file. HOOK_LINE is the landmark because every check below it is
# written in terms of HOOK_LINE or LOADER_LINE -- if that stops being true the
# extraction is meaningless, so the test says so rather than silently passing.
_VERDICT_START = re.compile(r"^HOOK_LINE=", re.MULTILINE)


def _pattern(name: str) -> str:
    """Return the single-quoted value of a top-level ``NAME='...'`` assignment."""
    match = re.search(
        rf"^{re.escape(name)}='([^']*)'$", _SCRIPT.read_text(), flags=re.MULTILINE
    )
    assert match, f"{name} is no longer a single-quoted assignment in {_SCRIPT.name}"
    return match.group(1)


def _verdict(log: str, rc: int, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    """Run the script's own verdict section over ``log`` with the run's exit code.

    The section is sliced out of the script rather than reimplemented, so these
    tests exercise the branch order and the messages that will actually ship.
    The variables it reads from the part we skip are supplied here.
    """
    body = _SCRIPT.read_text()
    start = _VERDICT_START.search(body)
    assert start, (
        "cannot find the HOOK_LINE definition that begins the verdict section; "
        "if the script was restructured, re-point this extraction"
    )
    log_path = tmp_path / "hook.log"
    log_path.write_text(log)
    driver = tmp_path / "verdict.sh"
    driver.write_text(
        "set -uo pipefail\n"
        f'LOG={log_path}\nrc={rc}\nKEEP=0\nHOOK=/nonexistent/hook.so\nTIMEOUT=6000\n'
        + body[start.start() :]
    )
    return subprocess.run(
        ["bash", str(driver)], capture_output=True, text=True, check=False
    )


def _result_line(proc: subprocess.CompletedProcess[str]) -> str:
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT:"):
            return line
    raise AssertionError(f"no RESULT line in output:\n{proc.stdout}")


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


# --------------------------------------------------------------------------
# Verdict control flow: exit code and conclusion for each shape of log.
#
# The exit codes are the script's published contract (see its header): 0
# reproduced, 1 fixed, 2 environment unusable, 3 inconclusive. Callers upstream
# branch on them, so each case asserts the code and not just the wording.
# --------------------------------------------------------------------------


def test_verdict_overlap_only_is_reproduced(tmp_path: Path) -> None:
    """The one case that may exit 0: the hook's own overlap diagnostic, at exit 92."""
    proc = _verdict(f"{_INSTALLED}\n{_OVERLAP}\n{_REJECTION}\n", 92, tmp_path)
    assert proc.returncode == 0
    assert "reproduced" in _result_line(proc)


def test_verdict_growth_only_is_inconclusive_and_names_the_knob(tmp_path: Path) -> None:
    """A capacity rejection must never read as the defect, and must be actionable."""
    proc = _verdict(f"{_INSTALLED}\n{_GROWTH}\n{_REJECTION}\n", 92, tmp_path)
    assert proc.returncode == 3
    assert "inconclusive" in _result_line(proc)
    assert "growth ceiling" in proc.stdout
    assert "RJ_CONSAN_MAX_PATCHED_IMAGE_GROWTH_PERCENT" in proc.stdout
    # Raising the ceiling without raising the timeout just buys a different
    # inconclusive result, so the hint has to say so.
    assert "--timeout" in proc.stdout


def test_verdict_both_diagnostics_is_inconclusive_without_contradicting_itself(
    tmp_path: Path,
) -> None:
    """Several loads can share one log, so the script cannot attribute the two lines.

    It also must not claim the transform stopped before final validation, since
    the overlap diagnostic *is* a final-validation diagnostic.
    """
    proc = _verdict(f"{_INSTALLED}\n{_GROWTH}\n{_OVERLAP}\n{_REJECTION}\n", 92, tmp_path)
    assert proc.returncode == 3
    assert "inconclusive" in _result_line(proc)
    assert "never reached final validation" not in proc.stdout
    assert "NOT the overlapping-patch defect" not in proc.stdout


def test_verdict_unknown_4112_is_inconclusive_not_reproduced(tmp_path: Path) -> None:
    """4112 is a shared bucket, so one with no explanation is a third thing."""
    proc = _verdict(f"{_INSTALLED}\n{_REJECTION}\n", 92, tmp_path)
    assert proc.returncode == 3
    assert "reproduced" not in _result_line(proc)
    assert "shared bucket" in proc.stdout


def test_verdict_growth_branch_precedes_the_4112_branch(tmp_path: Path) -> None:
    """Branch order is load-bearing, and inverting it is silent.

    A growth-only log also matches the generic 4112 rejection, so if the 4112
    branch ran first this would take the "third transform error" path instead of
    naming the capacity policy the reader can actually do something about.
    """
    proc = _verdict(f"{_INSTALLED}\n{_GROWTH}\n{_REJECTION}\n", 92, tmp_path)
    assert "growth ceiling" in _result_line(proc) or "growth ceiling" in proc.stdout
    assert "some third transform error" not in proc.stdout


def test_verdict_clean_transform_is_fixed_only_at_exit_86(tmp_path: Path) -> None:
    """The loader marker alone would also appear with no hook loaded (#385 §21)."""
    log = f"{_INSTALLED}\n[consan_4112_load] loaded and instrumented\n"
    assert _verdict(log, 86, tmp_path).returncode == 1
    other = _verdict(log, 0, tmp_path)
    assert other.returncode == 3
    assert "fixed" not in _result_line(other)


def test_verdict_requires_the_hook_to_have_announced_itself(tmp_path: Path) -> None:
    """Without it, a missing or non-rocjitsu HSA_TOOLS_LIB reads as evidence."""
    proc = _verdict(f"{_OVERLAP}\n{_REJECTION}\n", 92, tmp_path)
    assert proc.returncode == 3
    assert "never announced itself" in proc.stdout


@pytest.mark.parametrize("rc", [124, 137])
def test_verdict_timeout_is_inconclusive(tmp_path: Path, rc: int) -> None:
    """timeout(1) reports 124 on TERM and 137 when the follow-up KILL was needed."""
    proc = _verdict(f"{_INSTALLED}\n{_OVERLAP}\n", rc, tmp_path)
    assert proc.returncode == 3
    assert "ceiling" in _result_line(proc)


def test_verdict_spoofed_overlap_path_is_not_reproduced(tmp_path: Path) -> None:
    """End-to-end version of the #409 review's vector, through the real branches."""
    log = (
        _LOADER_FAIL.format(path=f"{_OVERLAP}.hsaco") + "\n"
        f"{_INSTALLED}\n"
        f"{_REJECTION}\n"
    )
    proc = _verdict(log, 92, tmp_path)
    assert proc.returncode == 3, "an echoed --object path must not manufacture a reproduction"
    assert "reproduced" not in _result_line(proc)


# --------------------------------------------------------------------------
# Individual patterns, which fail differently: an anchor can be dropped from one
# check while the branch structure above still looks correct.
# --------------------------------------------------------------------------


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
        _LOADER_FAIL.format(path=f"{_HOOK}{patterns['OVERLAP_LINE']} patch ranges.hsaco") + "\n"
        f"{_INSTALLED}\n"
        f"{_REJECTION}\n"
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
        _LOADER_FAIL.format(path=f"{_HOOK}{patterns['GROWTH_LINE']}.hsaco") + "\n"
        f"{_INSTALLED}\n"
        f"{_OVERLAP}\n"
        f"{_REJECTION}\n"
    )
    assert not _matches(f"{patterns['HOOK_LINE']} {patterns['GROWTH_LINE']}", log, tmp_path)
    assert _matches(f"{patterns['HOOK_LINE']} {patterns['OVERLAP_LINE']}", log, tmp_path)


def test_loader_success_marker_cannot_be_spoofed(patterns: dict[str, str], tmp_path: Path) -> None:
    """`loaded and instrumented` gates the "fixed" verdict, so it needs the same anchor."""
    log = (
        f"{_INSTALLED}\n"
        + _LOADER_FAIL.format(path="[consan_4112_load] loaded and instrumented.hsaco")
        + "\n"
    )
    assert not _matches(f"{patterns['LOADER_LINE']} loaded and instrumented", log, tmp_path)


def test_the_loader_success_line_echoes_the_path_too(
    patterns: dict[str, str], tmp_path: Path
) -> None:
    """consan_4112_load.hip:22 puts the path *after* its own prefix, on the same line.

    That line legitimately matches LOADER_LINE, so the anchor alone does not keep
    caller text out of it -- what does is that the path lands mid-line, behind the
    prefix, and cannot start a new one (the object-path guard forbids newlines).
    """
    log = (
        f"{_INSTALLED}\n"
        + _LOADER_OK.format(path=f"/tmp/{_HOOK}{patterns['OVERLAP_LINE']} patch ranges.hsaco")
        + "\n"
    )
    assert _matches(f"{patterns['LOADER_LINE']} loaded and instrumented", log, tmp_path)
    assert not _matches(f"{patterns['HOOK_LINE']} {patterns['OVERLAP_LINE']}", log, tmp_path)


def test_loader_fixtures_match_the_real_loader_source() -> None:
    """The spoof fixtures are only evidence if they are the loader's actual output.

    An invented shape proves nothing about the real vector, so pin both lines to
    the format strings in consan_4112_load.hip. If that file changes its wording,
    this fails here rather than leaving the spoof tests quietly testing fiction.
    """
    loader = (_SCRIPT.parent / "consan_4112_load.hip").read_text()
    # The failure line deliberately has no [consan_4112_load] prefix; the success
    # line does. That asymmetry is the whole reason the anchor has to do the work.
    assert '"hipModuleLoad(%s) failed: %s\\n"' in loader
    assert '"[consan_4112_load] loaded and instrumented %s (no dispatch)\\n"' in loader
    assert _LOADER_FAIL.startswith("hipModuleLoad(")
    assert "[consan_4112_load]" not in _LOADER_FAIL
    assert _LOADER_OK.startswith("[consan_4112_load] loaded and instrumented ")


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


def _sandbox(tmp_path: Path) -> dict[str, str]:
    """Environment that lets the script reach its object-path guard without ROCm.

    The guard runs after the ``hipcc`` preflight, so on a machine without ROCm the
    script dies before it and these tests would assert against the wrong error.
    Stubbing rather than marking the tests ``rocm`` is deliberate: the guard is
    pure shell logic with nothing to run on a GPU, so it belongs in the CPU
    matrix, and stubbing also pins the behaviour on a machine that *does* have a
    real hipcc -- where these tests otherwise pass for the wrong reason. That is
    how the gap reached CI in the first place.

    ``ROCM_PATH`` points at nothing so the script's own PATH prepend cannot shadow
    the stubs, making the run identical on both kinds of host.
    """
    bindir = tmp_path / "stubbin"
    bindir.mkdir(exist_ok=True)
    # Failing is the fast, deterministic outcome: a hostile path dies at the
    # guard before this is ever invoked, and an accepted path stops here instead
    # of running the whole hooked flow.
    stub = bindir / "hipcc"
    stub.write_text("#!/bin/sh\nexit 1\n")
    stub.chmod(0o755)
    env = dict(os.environ)
    env["PATH"] = f"{bindir}{os.pathsep}{env.get('PATH', '')}"
    env["ROCM_PATH"] = str(tmp_path / "no-such-rocm")
    env["ROCM_HOME"] = env["ROCM_PATH"]
    return env


def _run_script(object_path: str, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(_SCRIPT), "--hook", "/bin/true", "--object", object_path],
        capture_output=True,
        text=True,
        check=False,
        env=_sandbox(tmp_path),
    )


def test_the_sandbox_reaches_the_guard_at_all() -> None:
    """Guard against the stubs silently stopping the script too early.

    Without this, every rejection test above could pass because the run died at
    ``hipcc`` with a message that happens not to contain the guard text.
    """
    body = _SCRIPT.read_text()
    assert body.index('command -v hipcc') < body.index('case "${OBJECT}" in'), (
        "the hipcc preflight no longer precedes the object guard; the sandbox "
        "may no longer be needed, or may need to stub something else"
    )


@pytest.mark.parametrize(
    ("name", "why"),
    [
        ("a\n[rocjitsu-dbi-hooks] ConSan.hsaco", "a real newline defeats the ^ anchor directly"),
        (
            "a\\n[rocjitsu-dbi-hooks] ConSan.hsaco",
            "the compiler unescapes \\n in the -DOBJECT literal into that same newline",
        ),
        ('a"; system("id"); ".hsaco', "a quote closes the literal and injects C into the loader"),
    ],
)
def test_hostile_object_path_is_rejected(tmp_path: Path, name: str, why: str) -> None:
    """The path is baked into the loader as a C string and echoed back into the log.

    Rejecting is deliberate rather than escaping: no legitimate code object needs
    these characters, so refusing is honest where escaping would be a guess.
    """
    spoof = tmp_path / name
    spoof.write_bytes(b"")
    proc = _run_script(str(spoof), tmp_path)
    assert proc.returncode == 2, why
    assert "must not contain a newline, backslash or double quote" in proc.stderr


def test_ordinary_object_path_is_not_rejected(tmp_path: Path) -> None:
    """The guard must not fire on real paths.

    The run stops at the stubbed hipcc just after the guard, which is the proof
    it got past validation -- otherwise the test above would pass equally well
    against a guard that rejects everything.
    """
    target = tmp_path / "obj.hsaco"
    target.write_bytes(b"")
    proc = _run_script(str(target), tmp_path)
    assert "must not contain a newline" not in proc.stderr
    assert "hipcc failed to build the loader" in proc.stderr


def test_object_path_guard_covers_the_workdir_path_too(tmp_path: Path) -> None:
    """`--workdir` feeds the embedded path on the extraction branch, which has no --object.

    That branch is why the guard sits at the point of use rather than at argument
    parsing: a parse-time check on ``--object`` never sees this path at all.
    """
    # Reaching the guard on this branch needs a hipBLASLt bundle to extract from
    # and a bundler to extract it, neither of which a CPU runner has. Both are
    # faked so the branch is exercised rather than skipped -- the guard runs
    # before either is read for content, so nothing here has to be a real object.
    bundle_name = re.search(r'^BUNDLE_NAME="([^"]+)"$', _SCRIPT.read_text(), re.MULTILINE)
    assert bundle_name, "BUNDLE_NAME is no longer a plain assignment in the script"
    rocm = tmp_path / "fake-rocm"
    library = rocm / "lib" / "hipblaslt" / "library"
    library.mkdir(parents=True)
    (library / bundle_name.group(1)).write_bytes(b"")

    env = _sandbox(tmp_path)
    env["ROCM_PATH"] = env["ROCM_HOME"] = str(rocm)
    bundler = Path(env["PATH"].split(os.pathsep)[0]) / "clang-offload-bundler"
    bundler.write_text("#!/bin/sh\nexit 0\n")
    bundler.chmod(0o755)

    workdir = tmp_path / 'w"quote'
    workdir.mkdir()
    proc = subprocess.run(
        [str(_SCRIPT), "--hook", "/bin/true", "--workdir", str(workdir)],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert proc.returncode == 2, proc.stderr
    assert "must not contain a newline, backslash or double quote" in proc.stderr
