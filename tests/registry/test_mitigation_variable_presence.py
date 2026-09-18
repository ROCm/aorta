"""Every built-in mitigation's environment variable must exist in some runtime binary.

A mitigation is an environment variable, and a variable that appears in no
loaded runtime binary is read by nobody. Setting it produces a probe cell that
consumes a trial, reports ``fail`` beside the real baseline, and looks exactly
like a mitigation that was tried and did not help -- a second baseline under a
different name, which is the failure the probe harness exists to prevent.

Two entries are in that state on the stack probes run on today: ``tf32_off``
(aorta#500) and ``rccl_gfx942_cheap_fence_off``. Both were found the expensive
way, by building a scenario around the knob and watching it not invert. This
check finds them in milliseconds.

WHAT THIS CAN AND CANNOT DETECT
===============================
There are four ways a registered mitigation is inert on a given stack, and
**this test covers exactly one of them.** Saying so here rather than leaving it
implied, because a guard that appears to validate more than it does is worse
than no guard.

Covered:

* **the variable is read by nobody** -- the string is absent from every scanned
  binary. Greppable, and that is what this does.

Not covered, and not detectable this way:

* **read and refused** -- the runtime parses the value and declines. Both
  ``pytorch_alloc_expandable_segments`` (*expandable_segments not supported on
  this platform*) and ``fa_prefer_ck`` (*PyTorch built with CK SDPA support:
  0*) are in this state, and both variables are present in the binaries. Only
  running something and reading its warnings finds these.
* **sets the value that is already the default** -- ``fa_prefer_aotriton``
  sets ``TORCH_ROCM_FA_PREFER_CK=0``. The variable is present and is honoured;
  the value is a no-op. Needs the runtime's documented default, not a grep.
* **already set in the image** -- ``rocm/primus:v26.3`` bakes
  ``HSA_NO_SCRATCH_RECLAIM=1`` into ``Config.Env``, so
  ``hsa_no_scratch_reclaim`` sets what is set. Needs the cell's intent compared
  against the child's observed environment.

See aorta#511 for all four with their evidence.
"""

from __future__ import annotations

import importlib.util
import mmap
import re
import sys
from pathlib import Path

import pytest

from aorta.registry.mitigations import BUILTIN_MITIGATIONS

_REPO_ROOT = Path(__file__).resolve().parents[2]
_AUDIT_SCRIPT = _REPO_ROOT / "scripts" / "audit_env_knobs.py"

#: Shared objects that read the variables the mitigation registry sets. The GEMM
#: libraries ``audit_env_knobs.py`` already audits are deliberately not here:
#: no mitigation variable lives in them.
RUNTIME_SONAMES = (
    "libamdhip64.so",
    "libhsa-runtime64.so",
    "librccl.so",
)

#: Mitigations whose variable is known to be absent from every scanned binary,
#: each with the reason. The lane must not go red over a backlog on day one --
#: the point is to catch the *next* one.
#:
#: This list is for the absent-string mode ONLY. An entry that is inert for one
#: of the other three reasons does not belong here and
#: :func:`test_known_absent_covers_only_the_mode_this_test_can_see` enforces
#: that, because silencing a mode-2 entry here would make the guard look like
#: it had checked something it cannot check.
KNOWN_ABSENT: dict[str, str] = {
    "tf32_off": (
        "DISABLE_TF32 appears in no ROCm or torch binary; aorta#500. The "
        "registry attributes it to hipBLASLt and instrumentation/env_knobs.py "
        "attributes it to pytorch, and neither holds."
    ),
    "rccl_gfx942_cheap_fence_off": (
        "RCCL_GFX942_CHEAP_FENCE_OFF appears in no binary including librccl; "
        "the name is gfx942-scoped and the supported targets have moved on. "
        "aorta#511."
    ),
}

_ENV_NAME_RE = re.compile(rb"[A-Z][A-Z0-9_]{2,}")


def _load_audit_script():
    """Import ``scripts/audit_env_knobs.py`` by path.

    It is not importable as a package member, and the same approach is taken by
    ``tests/docker/test_rocm_layout_guard.py`` for the same reason: reusing its
    ``resolve_library`` is better than restating the soname rules, which are
    subtle enough to have been got wrong once already (a lexicographic sort
    picked the oldest co-installed major).
    """
    spec = importlib.util.spec_from_file_location("audit_env_knobs", _AUDIT_SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def names_in_binary(lib: Path) -> frozenset[str]:
    """Environment-variable-shaped strings that occupy a whole C string in ``lib``.

    NUL required on both sides, for the reason ``audit_env_knobs.extract_names``
    gives: ``strings`` splits on any non-printable byte and can manufacture a
    standalone-looking name out of the middle of help text. That function is not
    reused directly because its regex is built from the GEMM prefixes it audits,
    and mitigation variables share no prefix.
    """
    with lib.open("rb") as handle:
        if handle.read(4) != b"\x7fELF":
            raise ValueError(f"{lib} is not an ELF shared object")
    found: set[str] = set()
    with lib.open("rb") as handle, mmap.mmap(
        handle.fileno(), 0, access=mmap.ACCESS_READ
    ) as data:
        for match in _ENV_NAME_RE.finditer(data):
            start, end = match.span()
            if (start == 0 or data[start - 1] == 0) and end < len(data) and data[end] == 0:
                found.add(match.group().decode("ascii"))
    return frozenset(found)


def mitigation_variables() -> dict[str, set[str]]:
    """Variable name -> the mitigations that set it. Read from the registry itself."""
    by_name: dict[str, set[str]] = {}
    for mitigation, env in BUILTIN_MITIGATIONS.items():
        for variable in env:
            by_name.setdefault(variable, set()).add(mitigation)
    return by_name


def find_unread_mitigations(present: frozenset[str] | set[str]) -> dict[str, set[str]]:
    """Variables no binary contains, mapped to the mitigations that set them.

    Entries in :data:`KNOWN_ABSENT` are omitted: the guard is for the next one,
    not for the backlog.
    """
    offenders: dict[str, set[str]] = {}
    for variable, mitigations in mitigation_variables().items():
        if variable in present:
            continue
        unexpected = {m for m in mitigations if m not in KNOWN_ABSENT}
        if unexpected:
            offenders[variable] = unexpected
    return offenders


def find_fixed_known_absent(present: frozenset[str] | set[str]) -> list[str]:
    """Listed-absent mitigations whose variable is now readable, so the list can shrink."""
    return sorted(
        mitigation
        for mitigation in KNOWN_ABSENT
        if any(v in present for v in BUILTIN_MITIGATIONS.get(mitigation, {}))
    )


def _scan_dirs() -> list[Path]:
    """Library directories to scan, those that exist on this machine."""
    dirs: list[Path] = []
    rocm = Path("/opt/rocm/lib")
    if rocm.is_dir():
        dirs.append(rocm)
    try:
        import torch

        torch_lib = Path(torch.__file__).parent / "lib"
        if torch_lib.is_dir():
            dirs.append(torch_lib)
    except ImportError:
        pass
    return dirs


# ---------------------------------------------------------------------------
# hermetic: the scan logic, against binaries this test writes
# ---------------------------------------------------------------------------

def _fake_so(path: Path, names: list[str], noise: str = "") -> Path:
    """An ELF-magic file whose NUL-delimited strings are exactly ``names``."""
    body = b"\x7fELF" + b"\x00" * 12
    for name in names:
        body += b"\x00" + name.encode() + b"\x00"
    if noise:
        # Not NUL-delimited: embedded in a sentence, the way help text carries a
        # name it does not read.
        body += b"\x00set " + noise.encode() + b" to enable the thing\x00"
    path.write_bytes(body)
    return path


def test_scan_finds_a_nul_delimited_name(tmp_path):
    lib = _fake_so(tmp_path / "libfake.so", ["HIP_LAUNCH_BLOCKING", "GPU_MAX_HW_QUEUES"])
    assert {"HIP_LAUNCH_BLOCKING", "GPU_MAX_HW_QUEUES"} <= names_in_binary(lib)


def test_scan_ignores_a_name_embedded_in_prose(tmp_path):
    """The defect a plain ``strings`` pipeline has: help text is not a read."""
    lib = _fake_so(tmp_path / "libfake.so", ["HIP_LAUNCH_BLOCKING"], noise="DISABLE_TF32")
    found = names_in_binary(lib)
    assert "HIP_LAUNCH_BLOCKING" in found
    assert "DISABLE_TF32" not in found


def test_scan_rejects_a_non_elf_file(tmp_path):
    lib = tmp_path / "libfake.so"
    lib.write_bytes(b"not an elf at all")
    with pytest.raises(ValueError, match="not an ELF"):
        names_in_binary(lib)


def test_absent_variable_is_reported_absent(tmp_path):
    """The check as a whole, on a tree where one registry variable is missing."""
    lib = _fake_so(tmp_path / "libfake.so", ["HIP_LAUNCH_BLOCKING"])
    present = names_in_binary(lib)
    absent = [v for v in mitigation_variables() if v not in present]
    assert "DISABLE_TF32" in absent
    assert "HIP_LAUNCH_BLOCKING" not in absent


# ---------------------------------------------------------------------------
# registry hygiene: hermetic, no libraries needed
# ---------------------------------------------------------------------------

def test_known_absent_names_are_real_mitigations():
    unknown = sorted(set(KNOWN_ABSENT) - set(BUILTIN_MITIGATIONS))
    assert not unknown, (
        f"KNOWN_ABSENT names mitigations that no longer exist: {unknown}. "
        "Remove them; an expected-failures list that outlives its entries stops "
        "being read."
    )


def test_known_absent_covers_only_the_mode_this_test_can_see():
    """Guard the guard's own scope.

    The other three inertness modes are invisible to a grep, so an entry parked
    here for one of them would be silenced by a check that never looked at it.
    These four are the ones aorta#511 records as inert for reasons this test
    cannot see; none of them belongs in KNOWN_ABSENT.
    """
    not_greppable = {
        "pytorch_alloc_expandable_segments",  # read and refused
        "fa_prefer_ck",                       # read and refused
        "fa_prefer_aotriton",                 # sets the default value
        "hsa_no_scratch_reclaim",             # already set in the image
    }
    misfiled = sorted(not_greppable & set(KNOWN_ABSENT))
    assert not misfiled, (
        f"{misfiled} are inert for reasons a binary scan cannot detect, so "
        "listing them here would make this test look like it had checked them. "
        "See this module's docstring and aorta#511."
    )


def test_audit_script_is_reusable_from_here():
    """``resolve_library`` is borrowed rather than restated; prove it still loads."""
    module = _load_audit_script()
    assert callable(module.resolve_library)


# ---------------------------------------------------------------------------
# the real check, where the libraries exist
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def present_names() -> frozenset[str]:
    dirs = _scan_dirs()
    if not dirs:
        pytest.skip(
            "no runtime libraries to scan: looked for /opt/rocm/lib and the "
            "torch lib directory and found neither. This check needs the "
            "libraries it is auditing, so it runs on a lane that has ROCm "
            "installed and is inert elsewhere -- it is not silently passing."
        )
    audit = _load_audit_script()
    found: set[str] = set()
    scanned: list[Path] = []
    for directory in dirs:
        for soname in RUNTIME_SONAMES:
            lib = audit.resolve_library(directory, soname)
            if lib is not None:
                scanned.append(lib)
        scanned.extend(sorted(directory.glob("*.so")))
    for lib in scanned:
        try:
            found |= names_in_binary(lib)
        except (ValueError, OSError):
            continue          # not an ELF, or unreadable; neither is this test's business
    if not scanned:
        pytest.skip(f"no shared objects under {[str(d) for d in dirs]}")
    return frozenset(found)


def test_every_mitigation_variable_is_read_by_something(present_names):
    offenders = find_unread_mitigations(present_names)
    assert not offenders, (
        "these mitigations set an environment variable that appears in no "
        f"scanned binary, so nothing reads it: {offenders}. A probe cell for "
        "one of them is a second baseline under a different name. Either fix "
        "the entry or add it to KNOWN_ABSENT with the evidence. See aorta#511."
    )


def test_known_absent_entries_are_still_absent(present_names):
    """Stop the expected-failures list from rotting.

    An entry that has since become readable must leave the list, or the list
    grows into a permanent excuse and the check quietly shrinks to nothing.
    """
    fixed = find_fixed_known_absent(present_names)
    assert not fixed, (
        f"{fixed} are listed in KNOWN_ABSENT but their variables are now "
        "present in a scanned binary. Remove them from the list."
    )


# ---------------------------------------------------------------------------
# hermetic: the two verdicts above, against fabricated library trees
# ---------------------------------------------------------------------------
# Both verdicts are pure functions of the name set so they can be exercised on a
# machine with no ROCm. Without this, the only tests that ever run off a GPU
# lane would be the scanner's, and the thing the guard actually asserts would
# be untested everywhere it is cheap to test.

def test_an_unlisted_unread_mitigation_is_reported():
    """The case the guard exists for: a new entry nobody reads."""
    present = set(mitigation_variables()) - {"HIP_LAUNCH_BLOCKING"}
    offenders = find_unread_mitigations(present)
    assert offenders == {"HIP_LAUNCH_BLOCKING": {"hip_launch_blocking"}}


def test_a_listed_unread_mitigation_is_tolerated():
    """The backlog must not red the lane."""
    present = set(mitigation_variables()) - {"DISABLE_TF32"}
    assert find_unread_mitigations(present) == {}


def test_a_known_absent_entry_that_came_back_is_flagged():
    present = set(mitigation_variables())          # everything readable
    assert find_fixed_known_absent(present) == ["rccl_gfx942_cheap_fence_off", "tf32_off"]


def test_nothing_is_flagged_while_the_backlog_is_genuinely_absent():
    present = set(mitigation_variables()) - {"DISABLE_TF32", "RCCL_GFX942_CHEAP_FENCE_OFF"}
    assert find_fixed_known_absent(present) == []
