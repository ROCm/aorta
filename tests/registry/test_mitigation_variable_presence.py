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

  Per ``(mitigation, variable)``, not per mitigation: a mitigation is an env-var
  bundle, so an entry excused for one variable is still audited for the rest.

  Only where the whole stack is present to be scanned -- ROCm's libraries and a
  ROCm torch. On a lane that promises one and has not got one
  (``AORTA_REQUIRE_ROCM``) that is a failure, because a check that quietly
  audits nothing is worse than one that is not there. "Whole" is per soname:
  a stack missing one of the named libraries is refused exactly like a stack
  missing all of them, since a partial scan does not weaken the verdicts
  evenly -- see :class:`ScanSet`.

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
import os
import re
import sys
from pathlib import Path
from typing import NamedTuple, NoReturn

import pytest

from aorta.instrumentation.rocm_paths import RocmRoots, resolve_rocm_roots, safe_is_dir
from aorta.registry.mitigations import BUILTIN_MITIGATIONS

_REPO_ROOT = Path(__file__).resolve().parents[2]
_AUDIT_SCRIPT = _REPO_ROOT / "scripts" / "audit_env_knobs.py"

#: Shared objects that read the variables the mitigation registry sets. The GEMM
#: libraries ``audit_env_knobs.py`` already audits are deliberately not here:
#: no mitigation variable lives in them.
#:
#: Not the whole scan set. :func:`sonames_to_scan` adds whatever library a
#: :data:`KNOWN_ABSENT` entry names as its claimed consumer, which is how an
#: exemption whose claim points outside this tuple stays falsifiable.
RUNTIME_SONAMES = (
    "libamdhip64.so",
    "libhsa-runtime64.so",
    "librccl.so",
)


class Exemption(NamedTuple):
    """Why a ``(mitigation, variable)`` pair is allowed to be absent.

    ``claimed_consumer`` is the soname the *claim being excused* points at --
    the library some part of this repo says reads the variable. It is a field
    rather than prose in ``reason`` because
    :func:`test_every_known_absent_claim_is_scanned` has to be able to require
    that library to be in the scan set.

    Without it the exemption is unfalsifiable. ``DISABLE_TF32`` is excused on
    the evidence that hipBLASLt does not read it, while
    :data:`RUNTIME_SONAMES` deliberately omits the GEMM libraries -- so the
    one binary that could retire the entry was the one binary never opened,
    and :func:`test_known_absent_entries_are_still_absent` could only ever
    confirm what it already assumed.
    """

    claimed_consumer: str
    reason: str


#: ``(mitigation, variable)`` pairs known to be absent from every scanned
#: binary, each with the claim it excuses and the reason. The lane must not go
#: red over a backlog on day one -- the point is to catch the *next* one.
#:
#: **Keyed by the pair, not by the mitigation name.** A mitigation is an env-var
#: *bundle*, so a name-keyed exemption is broader than the fact it records: it
#: would also cover a second variable added to ``tf32_off`` later, which is a
#: new defect and exactly what this guard exists to catch. Keying the pair also
#: makes :func:`find_fixed_known_absent` precise -- under the name key it
#: cleared an entry when *any* of the mitigation's variables was readable, which
#: on a bundle is the wrong variable answering for the listed one.
#:
#: This list is for the absent-string mode ONLY. An entry that is inert for one
#: of the other three reasons does not belong here and
#: :func:`test_known_absent_covers_only_the_mode_this_test_can_see` enforces
#: that, because silencing a mode-2 entry here would make the guard look like
#: it had checked something it cannot check.
KNOWN_ABSENT: dict[tuple[str, str], Exemption] = {
    ("tf32_off", "DISABLE_TF32"): Exemption(
        claimed_consumer="libhipblaslt.so",
        reason=(
            "DISABLE_TF32 appears in no ROCm or torch binary; aorta#500. The "
            "registry attributes it to hipBLASLt (registry/mitigations.py: "
            "'consumed by hipBLASLt itself') and "
            "instrumentation/env_knobs.py attributes it to pytorch, and "
            "neither holds."
        ),
    ),
    ("rccl_gfx942_cheap_fence_off", "RCCL_GFX942_CHEAP_FENCE_OFF"): Exemption(
        claimed_consumer="librccl.so",
        reason=(
            "RCCL_GFX942_CHEAP_FENCE_OFF appears in no binary including "
            "librccl; the name is gfx942-scoped and the supported targets "
            "have moved on. aorta#511."
        ),
    ),
}


def exemption_sonames() -> tuple[str, ...]:
    """Libraries only a :data:`KNOWN_ABSENT` claim puts in the scan set.

    Deduplicated and order-stable, and it skips anything
    :data:`RUNTIME_SONAMES` already carries -- ``librccl`` is already scanned
    for its own sake, so the RCCL exemption adds nothing.
    """
    return tuple(
        dict.fromkeys(
            exemption.claimed_consumer
            for exemption in KNOWN_ABSENT.values()
            if exemption.claimed_consumer not in RUNTIME_SONAMES
        )
    )


def sonames_to_scan() -> tuple[str, ...]:
    """Every ROCm-side soname read, for either reason.

    Two reasons, and they are not the same reason. :data:`RUNTIME_SONAMES` is
    "this library reads mitigation variables"; :func:`exemption_sonames` is
    "some entry in :data:`KNOWN_ABSENT` says this library reads one, and that
    claim has to be checkable". The second set buys no coverage for
    :func:`test_every_mitigation_variable_is_read_by_something` -- it exists so
    the exemption list can shrink.

    It is a cost. hipBLASLt is the largest single object in a ROCm install, and
    the 2.5 s figure :func:`libraries_to_scan` quotes was measured before this
    library joined the set; it has not been re-measured on a GPU lane since, so
    take that number as the floor rather than the current cost. Still bounded
    by the same rule the glob broke: named libraries, not ``*.so``, so
    ``librocblas``, ``libMIOpen`` and the rest stay out.
    """
    return RUNTIME_SONAMES + exemption_sonames()


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

    ``(mitigation, variable)`` pairs in :data:`KNOWN_ABSENT` are omitted: the
    guard is for the next one, not for the backlog. A mitigation listed there
    for one variable is still audited for every other variable it sets.
    """
    offenders: dict[str, set[str]] = {}
    for variable, mitigations in mitigation_variables().items():
        if variable in present:
            continue
        unexpected = {m for m in mitigations if (m, variable) not in KNOWN_ABSENT}
        if unexpected:
            offenders[variable] = unexpected
    return offenders


def find_fixed_known_absent(
    present: frozenset[str] | set[str],
) -> list[tuple[str, str]]:
    """Listed pairs whose variable is now readable, so the list can shrink.

    Asks about the listed variable itself. A sibling variable of the same
    mitigation becoming readable says nothing about this entry and must not
    clear it.
    """
    return sorted(
        (mitigation, variable)
        for mitigation, variable in KNOWN_ABSENT
        if variable in present
    )


#: Set on a lane that promises a ROCm install. When it is set, a stack this
#: check cannot answer for is a FAILURE rather than a skip.
#:
#: Without it the only way this audit reports "ROCm is missing" is a skip, and a
#: skip on a lane configured to run the audit is indistinguishable from the
#: audit passing -- absence of evidence reading as success, which is the shape
#: this repo has filed twice (aorta#499, and the sanitizer nightly with no
#: failure alert). The GPU workflow sets it; nothing else does, so a developer
#: box still skips quietly.
REQUIRE_ROCM_ENV = "AORTA_REQUIRE_ROCM"

#: Values that turn :data:`REQUIRE_ROCM_ENV` off, so `AORTA_REQUIRE_ROCM=0` in a
#: shell profile does not silently arm it. Same set as
#: ``instrumentation/rocprof/_options.py`` uses for its own flags.
_FALSE = frozenset({"", "0", "false", "no", "off"})


def rocm_is_required(environ: dict[str, str] | None = None) -> bool:
    env = os.environ if environ is None else environ
    return env.get(REQUIRE_ROCM_ENV, "").strip().lower() not in _FALSE


def rocm_torch_lib() -> tuple[Path | None, str]:
    """Torch's library directory when torch is a ROCm build, else ``(None, why)``.

    **The build matters, and this was measured rather than assumed.** Scanning
    torch 2.13.0+cpu's libraries finds ``TORCH_ROCM_FA_PREFER_CK`` and
    ``PYTORCH_CUDA_ALLOC_CONF`` but *not* ``PYTORCH_NO_CUDA_MEMORY_CACHING`` --
    it lives in the CUDA/HIP caching allocator, which a CPU wheel does not
    build. So a stack with ROCm installed beside a CPU-only torch would report
    ``pytorch_no_cuda_memory_caching`` as read by nobody, which is false, and
    false about the one registered mitigation measured to resolve a real NaN.

    Same defect as the hardcoded ROCm path, one library up: the directories
    scanned have to be the ones the stack actually reads.
    """
    try:
        import torch
    except ImportError:
        return None, "torch is not importable"
    if getattr(torch.version, "hip", None) is None:
        return None, (
            f"torch {torch.__version__} is not a ROCm build (torch.version.hip "
            "is None), so its libraries do not carry the HIP caching "
            "allocator's variables"
        )
    lib = Path(torch.__file__).parent / "lib"
    if not safe_is_dir(lib):
        return None, f"{lib} is not a directory"
    return lib, ""


def rocm_lib_dirs(roots: RocmRoots | None = None) -> list[Path]:
    """The ROCm library directories to scan, in order, deduplicated.

    Resolved rather than hardcoded, because ``/opt/rocm`` is not where ROCm is
    on the lane this check has to run on. ``docker/Dockerfile.ci-gpu`` pins a
    wheel-layout (TheRock) image that has no ``/opt/rocm`` at all, so a literal
    ``/opt/rocm/lib`` finds nothing there and the audit skips on the one
    environment that can answer it. Issue #381 exists for exactly this, and
    ``resolve_rocm_roots`` is the resolver the rest of the repo already uses --
    ``audit_env_knobs.default_rocm_lib`` and the GPU workflow's own
    ``LD_LIBRARY_PATH`` line among them.

    **Both directories, and the pair is load-bearing.** ``core_lib_dir`` holds
    ``libamdhip64`` and ``libhsa-runtime64``; ``lib_dir`` hangs off the
    *libraries* root. On the wheel layout those are two different directories
    under site-packages, so scanning one of them reports the other's variables
    as unread. On a classic install they are the same ``/opt/rocm/lib`` and the
    dedup collapses them to a single entry -- byte-identical to what the
    hardcoded constant scanned.

    Empty when no ROCm install was found at all (``source == "none"``), which
    is what keeps the CPU lane honest: see :func:`scan_plan`.
    """
    resolved = resolve_rocm_roots() if roots is None else roots
    if resolved.source == "none":
        return []
    ordered = dict.fromkeys([resolved.core_lib_dir, resolved.lib_dir])
    return [directory for directory in ordered if safe_is_dir(directory)]


def no_rocm_message(roots: RocmRoots) -> str:
    """Why no ROCm directory was scanned, naming which mechanism answered.

    ``source`` is reported because #381's whole point is that a null be
    attributable: "no ROCm install was located" and "one was located and its
    lib dirs are missing" are different operator problems and used to look
    identical.
    """
    return (
        "no ROCm library directory was found, so this stack cannot read "
        "fifteen of the eighteen variables under audit and the check would "
        "report the registry as unread when the machine is what is missing. "
        f"resolve_rocm_roots() answered source={roots.source!r} "
        f"layout={roots.layout!r}, core_lib_dir={roots.core_lib_dir}, "
        f"lib_dir={roots.lib_dir}."
    )


def scan_plan(
    roots: RocmRoots | None = None,
    torch: tuple[Path | None, str] | None = None,
) -> tuple[list[Path], str]:
    """``(directories to scan, why not)``. A reason is non-empty iff the list is empty.

    So a caller decides on the list, never on the reason: a usable plan carries
    ``""`` because there is nothing to explain, and reading that as an error
    would turn every successful scan into a failure.

    **Every one of the eighteen variables must have a library that could carry
    it, or this check does not run at all.** Fifteen are read by
    ``libamdhip64``, ``libhsa-runtime64`` or ``librccl`` and three by torch's.
    A stack missing either side would report those variables as read by nobody
    -- a false statement about the registry and a true one about the machine,
    and a guard that fires on the wrong lane gets deleted rather than fixed.
    Measured: it did exactly that on this branch's first push.

    All-or-nothing rather than a partial audit on purpose. A partial one needs a
    second exemption concept ("unread, but we did not look"), and an exemption
    that means "not checked" is how a guard turns into decoration.

    Both inputs are injectable so the rules are testable on a machine with
    neither, which every machine that is not the GPU lane is.
    """
    dirs = rocm_lib_dirs(roots)
    if not dirs:
        resolved = resolve_rocm_roots() if roots is None else roots
        return [], no_rocm_message(resolved)
    torch_lib, torch_why = rocm_torch_lib() if torch is None else torch
    if torch_lib is None:
        return [], (
            f"{torch_why}, so three of the eighteen variables under audit "
            "(PYTORCH_NO_CUDA_MEMORY_CACHING, PYTORCH_CUDA_ALLOC_CONF, "
            "TORCH_ROCM_FA_PREFER_CK) have no library that could carry them."
        )
    dirs.append(torch_lib)
    return dirs, ""


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
    unknown = sorted({m for m, _ in KNOWN_ABSENT} - set(BUILTIN_MITIGATIONS))
    assert not unknown, (
        f"KNOWN_ABSENT names mitigations that no longer exist: {unknown}. "
        "Remove them; an expected-failures list that outlives its entries stops "
        "being read."
    )


def test_known_absent_entries_name_a_variable_that_mitigation_actually_sets():
    """The other half of rot, and only reachable now the key is the pair.

    An entry whose variable the mitigation no longer sets exempts nothing -- it
    is dead weight that reads as coverage. Under the old mitigation-name key
    there was no variable to be wrong, so this failure mode had nowhere to be
    caught.
    """
    stale = sorted(
        (mitigation, variable)
        for mitigation, variable in KNOWN_ABSENT
        if mitigation in BUILTIN_MITIGATIONS
        and variable not in BUILTIN_MITIGATIONS[mitigation]
    )
    assert not stale, (
        f"{stale} name a variable their mitigation does not set, so the "
        "exemption applies to nothing. Update the entry to the variable the "
        "registry actually carries, or remove it."
    )


def test_every_known_absent_claim_is_scanned():
    """An exemption may not point at a library the scan never opens.

    This is the rot check's precondition, and it was not held. ``DISABLE_TF32``
    is excused because hipBLASLt does not read it, while
    :data:`RUNTIME_SONAMES` omitted the GEMM libraries -- so
    :func:`test_known_absent_entries_are_still_absent` could not retire the
    entry no matter what a future hipBLASLt did, and the exemption was
    permanent by construction rather than by evidence.

    Stated over :data:`KNOWN_ABSENT` rather than over the one pair that had the
    defect, so the next entry cannot reintroduce it by naming a fourth library.
    """
    scanned = sonames_to_scan()
    unscanned = sorted(
        (mitigation, variable, exemption.claimed_consumer)
        for (mitigation, variable), exemption in KNOWN_ABSENT.items()
        if exemption.claimed_consumer not in scanned
    )
    assert not unscanned, (
        f"{unscanned} excuse a variable on the evidence that a library does "
        "not read it, and that library is not in the scan set, so the "
        "exemption can never be retired. Add the soname to RUNTIME_SONAMES if "
        "it reads mitigation variables in its own right; otherwise "
        "exemption_sonames() picks it up automatically and this means the "
        "claimed_consumer is misspelled."
    )


def test_exemption_sonames_does_not_restate_the_runtime_set():
    """``librccl`` is scanned for its own sake; the RCCL entry must not re-add it.

    Not cosmetic: :func:`sonames_to_scan` concatenates, so a duplicate would
    resolve the same library twice per directory. The dedup in
    :func:`libraries_to_scan` hides the cost afterwards, which is exactly why
    it is worth asserting here instead.
    """
    assert not set(exemption_sonames()) & set(RUNTIME_SONAMES)
    assert len(set(sonames_to_scan())) == len(sonames_to_scan())
    assert "libhipblaslt.so" in exemption_sonames(), (
        "the tf32_off entry's claimed consumer should be reaching the scan "
        "set through the exemption path, not through RUNTIME_SONAMES"
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
    misfiled = sorted(not_greppable & {m for m, _ in KNOWN_ABSENT})
    assert not misfiled, (
        f"{misfiled} are inert for reasons a binary scan cannot detect, so "
        "listing them here would make this test look like it had checked them. "
        "See this module's docstring and aorta#511."
    )


# ---------------------------------------------------------------------------
# which directories get scanned, on each layout, with no ROCm on this machine
# ---------------------------------------------------------------------------
# The CI GPU image is wheel-layout and has no /opt/rocm, and the CPU lane has no
# ROCm at all, so both of the environments this check has to behave correctly in
# are environments a developer box is not. Fabricating the resolver's answer is
# the only way to test either one here -- and the wheel case is the one that was
# wrong, so it needs a test rather than a reading of the Dockerfile.

def _roots(core: Path, libraries: Path, source: str = "import:_rocm_sdk_core",
           layout: str = "wheel") -> RocmRoots:
    return RocmRoots(core=core, libraries=libraries, include=core,
                     layout=layout, source=source)


def _rocm_torch(path: Path) -> tuple[Path, str]:
    return path, ""


def _outcome_of(call) -> BaseException:
    """The skip-or-fail exception ``call`` raised, as a value to assert on.

    ``pytest.raises(pytest.fail.Exception)`` cannot be used for this: a
    ``Skipped`` escaping it skips the *test*, so the assertion that a skip has
    become a failure passes by being skipped. Measured -- reverting the flag's
    branch left this test green until it was written this way, which is the
    same absence-reads-as-success shape the flag itself exists to close.
    """
    try:
        call()
    except (pytest.fail.Exception, pytest.skip.Exception) as exc:
        return exc
    raise AssertionError(f"{call.__name__} neither failed nor skipped")


def test_wheel_layout_scans_both_the_core_and_the_libraries_lib_dir(tmp_path):
    """The defect the hardcoded ``/opt/rocm/lib`` had, on the layout CI runs.

    ``docker/Dockerfile.ci-gpu`` pins a TheRock image with no ``/opt/rocm``,
    where the HIP runtime and the math libraries sit under two different
    site-packages components. Scanning one of them reports the other's
    variables as unread; scanning neither is what the literal path did.
    """
    core = tmp_path / "_rocm_sdk_core"
    libraries = tmp_path / "_rocm_sdk_libraries"
    (core / "lib").mkdir(parents=True)
    (libraries / "lib").mkdir(parents=True)
    torch_like = tmp_path / "torch" / "lib"
    torch_like.mkdir(parents=True)

    dirs, why_not = scan_plan(
        roots=_roots(core, libraries), torch=_rocm_torch(torch_like)
    )
    assert dirs == [core / "lib", libraries / "lib", torch_like]
    assert why_not == ""


def test_classic_layout_scans_one_directory(tmp_path):
    """On a classic install the two roots coincide; the scan must not double."""
    root = tmp_path / "opt" / "rocm"
    (root / "lib").mkdir(parents=True)
    torch_like = tmp_path / "torch" / "lib"
    torch_like.mkdir(parents=True)
    dirs, _ = scan_plan(
        roots=_roots(root, root, source="opt_rocm", layout="classic"),
        torch=_rocm_torch(torch_like),
    )
    assert dirs == [root / "lib", torch_like]


def test_a_torch_only_tree_does_not_answer_for_rocm_variables(tmp_path):
    """Pin the scoping decision, because getting it wrong reds the CPU lane.

    Fifteen of the eighteen variables are read by ROCm's libraries. On a tree
    with a CPU-wheel torch and no ROCm the check would report almost the whole
    registry as unread, which is a false statement about the registry and a true
    one about the machine. Measured: it did exactly that on the first push.

    ``source="none"`` is precisely how ``resolve_rocm_roots`` reports "nothing
    was found"; it still hands back the classic root so callers keep an
    absolute path, which is why the source and not the path is what is read.
    """
    torch_like = tmp_path / "torch" / "lib"
    torch_like.mkdir(parents=True)
    missing = tmp_path / "definitely-not-rocm"
    dirs, why_not = scan_plan(
        roots=_roots(missing, missing, source="none"), torch=_rocm_torch(torch_like)
    )
    assert dirs == []
    assert "source='none'" in why_not          # attributable, per issue #381
    assert str(missing / "lib") in why_not


def test_a_resolved_root_whose_lib_dirs_are_missing_scans_nothing(tmp_path):
    """Found-but-empty is still unable to answer, and must not be scanned.

    Distinct from the case above: something *was* located, so ``source`` is not
    ``"none"``, but there is no ``lib/`` under it. Scanning an empty list would
    make every variable look unread.
    """
    core = tmp_path / "_rocm_sdk_core"
    core.mkdir()
    torch_like = tmp_path / "torch" / "lib"
    torch_like.mkdir(parents=True)
    dirs, why_not = scan_plan(roots=_roots(core, core), torch=_rocm_torch(torch_like))
    assert dirs == []
    assert "import:_rocm_sdk_core" in why_not


def test_a_cpu_only_torch_beside_rocm_does_not_answer_either(tmp_path):
    """The torch half of the same defect, and it is not hypothetical.

    Measured on torch 2.13.0+cpu: its libraries carry
    ``TORCH_ROCM_FA_PREFER_CK`` and ``PYTORCH_CUDA_ALLOC_CONF`` but not
    ``PYTORCH_NO_CUDA_MEMORY_CACHING``. Auditing against it would report the
    one registered mitigation measured to resolve a real NaN as read by
    nobody.
    """
    core = tmp_path / "_rocm_sdk_core"
    (core / "lib").mkdir(parents=True)
    dirs, why_not = scan_plan(
        roots=_roots(core, core), torch=(None, "torch 2.13.0+cpu is not a ROCm build")
    )
    assert dirs == []
    assert "not a ROCm build" in why_not
    assert "PYTORCH_NO_CUDA_MEMORY_CACHING" in why_not


def test_the_reason_is_non_empty_exactly_when_the_plan_is_empty(tmp_path):
    """The ``scan_plan`` contract itself, over every branch that can return.

    Worth its own test because the docstring stated it backwards and nothing
    read the pair together: each case above asserts one half, so a return that
    carried both a usable list *and* a reason -- which a caller would read as a
    failed scan -- passed all of them. Pinning the biconditional is what makes
    "decide on the list, never on the reason" safe to rely on.
    """
    rocm = tmp_path / "_rocm_sdk_core"
    (rocm / "lib").mkdir(parents=True)
    torch_like = tmp_path / "torch" / "lib"
    torch_like.mkdir(parents=True)
    empty = tmp_path / "found-but-empty"
    empty.mkdir()
    missing = tmp_path / "definitely-not-rocm"
    cpu_torch = (None, "torch 2.13.0+cpu is not a ROCm build")

    plans = [
        scan_plan(roots=_roots(rocm, rocm), torch=_rocm_torch(torch_like)),
        scan_plan(roots=_roots(empty, empty), torch=_rocm_torch(torch_like)),
        scan_plan(
            roots=_roots(missing, missing, source="none"),
            torch=_rocm_torch(torch_like),
        ),
        scan_plan(roots=_roots(rocm, rocm), torch=cpu_torch),
        scan_plan(roots=_roots(missing, missing, source="none"), torch=cpu_torch),
    ]
    assert [bool(dirs) for dirs, _ in plans] == [True, False, False, False, False]
    for dirs, why_not in plans:
        assert bool(why_not) is not bool(dirs), (dirs, why_not)


def test_a_real_rocm_torch_is_recognised_by_its_hip_version(monkeypatch):
    """``torch.version.hip`` is the discriminator, and it is read, not assumed."""
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.version, "hip", "7.0.0", raising=False)
    lib, why_not = rocm_torch_lib()
    assert lib == Path(torch.__file__).parent / "lib"
    assert why_not == ""

    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    lib, why_not = rocm_torch_lib()
    assert lib is None
    assert "not a ROCm build" in why_not


def test_an_unreadable_stack_fails_rather_than_skips_when_the_lane_promised_it(
    monkeypatch,
):
    """A lane that runs the audit and audits nothing must not report success.

    The reason the resolver fix is not enough on its own: if the scan finds no
    directory the fixture skips, and a skipped check reads as a passing one.
    The GPU workflow sets this so that reads as a failure there.
    """
    monkeypatch.setattr(
        sys.modules[__name__], "scan_plan", lambda: ([], "nothing to scan.")
    )

    monkeypatch.delenv(REQUIRE_ROCM_ENV, raising=False)
    assert isinstance(_outcome_of(require_readable_stack), pytest.skip.Exception)

    monkeypatch.setenv(REQUIRE_ROCM_ENV, "1")
    outcome = _outcome_of(require_readable_stack)
    assert isinstance(outcome, pytest.fail.Exception), (
        f"expected a failure, got {type(outcome).__name__}: {outcome}"
    )
    assert "nothing to scan" in str(outcome)


def test_a_readable_stack_neither_fails_nor_skips(monkeypatch, tmp_path):
    """The narrowness control: the guard must not fire when the scan succeeded."""
    monkeypatch.setenv(REQUIRE_ROCM_ENV, "1")
    monkeypatch.setattr(
        sys.modules[__name__], "scan_plan", lambda: ([tmp_path], "")
    )
    assert require_readable_stack() == [tmp_path]


def test_directories_that_exist_but_hold_nothing_fail_the_promised_lane(
    monkeypatch, tmp_path
):
    """The second door into the green skip, one step past the one above.

    ``rocm_lib_dirs`` requires the directories to *exist*, not to hold
    anything, so ``require_readable_stack`` is satisfied by a base-image bump
    that moved the shared objects into a subdirectory -- and the fixture then
    skipped unconditionally on the empty scan. A lane running the audit with
    ``AORTA_REQUIRE_ROCM`` set reported it as having run when it read nothing,
    which is the exact state the flag was added to make impossible.
    """
    empty = [tmp_path / "rocm", tmp_path / "torch"]
    for directory in empty:
        directory.mkdir()
    scan = libraries_to_scan(empty)
    assert scan.libraries == []

    monkeypatch.delenv(REQUIRE_ROCM_ENV, raising=False)
    outcome = _outcome_of(lambda: require_scanned_libraries(scan, empty))
    assert isinstance(outcome, pytest.skip.Exception), outcome

    monkeypatch.setenv(REQUIRE_ROCM_ENV, "1")
    outcome = _outcome_of(lambda: require_scanned_libraries(scan, empty))
    assert isinstance(outcome, pytest.fail.Exception), (
        f"expected a failure, got {type(outcome).__name__}: {outcome}"
    )
    assert "no shared objects" in str(outcome)


def test_a_scan_that_found_libraries_neither_fails_nor_skips(monkeypatch, tmp_path):
    """The narrowness control: refusing every lane would satisfy the test above."""
    monkeypatch.setenv(REQUIRE_ROCM_ENV, "1")
    found = [tmp_path / "libamdhip64.so.7"]
    assert require_scanned_libraries(ScanSet(found, ()), [tmp_path]) == found


def test_only_torchs_directory_is_globbed(tmp_path):
    """``sonames_to_scan`` says which ROCm libraries are worth reading; the scan
    now agrees with it.

    The glob scanned every ``.so`` in the ROCm directories -- including GEMM
    libraries no mitigation variable and no exemption claim names -- for 4.43
    GB of reads against 365 MB and an identical verdict. ``librocblas`` is the
    control here and stays out under both rules. Torch's directory keeps its
    glob: it is the side with no soname declared for it, and its three
    variables live across several libraries rather than in one named file.

    ``scan_plan`` appends torch last, which is what makes the split a
    positional one; that ordering is asserted by ``scan_plan``'s own tests.
    """
    rocm, torch_lib = tmp_path / "rocm", tmp_path / "torch"
    rocm.mkdir()
    torch_lib.mkdir()
    declared = _fake_so(rocm / "libamdhip64.so", ["HIP_LAUNCH_BLOCKING"])
    _fake_so(rocm / "librocblas.so", ["ROCBLAS_LAYER"])
    globbed = _fake_so(torch_lib / "libtorch_hip.so", ["TORCH_ROCM_FA_PREFER_CK"])

    scanned = libraries_to_scan([rocm, torch_lib]).libraries

    assert declared.resolve() in scanned
    assert globbed.resolve() in scanned
    assert not [lib for lib in scanned if lib.name == "librocblas.so"], scanned


def test_a_versioned_torch_object_is_scanned_like_an_unversioned_one(tmp_path):
    """A runtime-only torch wheel was scanned as an empty directory.

    ``glob("*.so")`` matched ``libtorch_hip.so`` and nothing else, but a wheel
    that ships only ``libtorch_hip.so.2`` is the layout
    ``_loaded_lib_path_from_maps`` exists to handle -- this repository already
    knows torch appears that way. Missing it is the scan shrinking silently:
    ``ScanSet.unresolved`` reports ROCm sonames that resolved nowhere, and
    torch's side declares no sonames, so nothing is unresolved and nothing is
    missing. `TORCH_ROCM_FA_PREFER_CK` is then reported unread for an install
    that reads it.
    """
    rocm, torch_lib = tmp_path / "rocm", tmp_path / "torch"
    rocm.mkdir()
    torch_lib.mkdir()
    _fake_so(rocm / "libamdhip64.so", ["HIP_LAUNCH_BLOCKING"])
    versioned = _fake_so(torch_lib / "libtorch_hip.so.2", ["TORCH_ROCM_FA_PREFER_CK"])
    # Multi-component versions are the same layout; a `.so.debug` sidecar and a
    # `.so.orig` backup are not shared objects and are not ELF, so widening the
    # glob must not pick them up -- they would take the scan to
    # `_unreadable_stack` and turn a cosmetic file into a red lane.
    multi = _fake_so(torch_lib / "libc10_hip.so.2.4", ["PYTORCH_MIOPEN_SUGGEST_NHWC"])
    (torch_lib / "libtorch_hip.so.debug").write_bytes(b"not an elf")
    (torch_lib / "libtorch_hip.so.orig").write_bytes(b"not an elf")

    scanned = libraries_to_scan([rocm, torch_lib]).libraries

    assert versioned.resolve() in scanned, scanned
    assert multi.resolve() in scanned, scanned
    assert not [lib for lib in scanned if lib.suffix in {".debug", ".orig"}], scanned
    # And the name reaches the union, which is the whole point of opening it.
    assert "TORCH_ROCM_FA_PREFER_CK" in union_of_names(scanned)


def test_one_inode_is_read_once_however_many_names_it_has(tmp_path):
    """Narrowness for the widened glob: a devel install ships both names.

    ``libtorch_hip.so`` is a symlink to ``libtorch_hip.so.2`` wherever the
    devel package is installed, so matching versioned names too would have
    doubled every read on exactly the installs that were already working. The
    resolve is what keeps "one file is one read" true, which is the same claim
    the ROCm side makes by resolving through ``resolve_library``.
    """
    rocm, torch_lib = tmp_path / "rocm", tmp_path / "torch"
    rocm.mkdir()
    torch_lib.mkdir()
    _fake_so(rocm / "libamdhip64.so", ["HIP_LAUNCH_BLOCKING"])
    real = _fake_so(torch_lib / "libtorch_hip.so.2", ["TORCH_ROCM_FA_PREFER_CK"])
    (torch_lib / "libtorch_hip.so").symlink_to(real)

    scanned = libraries_to_scan([rocm, torch_lib]).libraries

    assert [lib for lib in scanned if "libtorch_hip" in lib.name] == [real.resolve()]


def test_a_known_absent_entrys_claimed_library_is_opened(tmp_path):
    """The rot check's precondition, end to end and without a GPU.

    ``test_every_known_absent_claim_is_scanned`` asserts the soname is in the
    list. This asserts the list is what ``libraries_to_scan`` reads, so the day
    hipBLASLt starts carrying ``DISABLE_TF32`` the name reaches
    ``present_names`` and ``find_fixed_known_absent`` retires the entry. Before
    this, that library was never opened and the entry was unfalsifiable.

    ``librocblas`` sits in the same directory as the control: being a GEMM
    library is not what gets a file read -- being named is.
    """
    rocm, torch_lib = tmp_path / "rocm", tmp_path / "torch"
    rocm.mkdir()
    torch_lib.mkdir()
    claimed = _fake_so(rocm / "libhipblaslt.so", ["DISABLE_TF32"])
    _fake_so(rocm / "librocblas.so", ["DISABLE_TF32"])

    scanned = libraries_to_scan([rocm, torch_lib]).libraries
    assert claimed.resolve() in scanned, scanned
    assert not [lib for lib in scanned if lib.name == "librocblas.so"], scanned

    present = set()
    for lib in scanned:
        present |= names_in_binary(lib)
    assert find_fixed_known_absent(present) == [("tf32_off", "DISABLE_TF32")]


def _stack(root: Path, omit: str = "") -> list[Path]:
    """A tree carrying every soname in the scan set except ``omit``.

    Built from :func:`sonames_to_scan` rather than from a literal list, so a
    soname added to either source lands here without this file being edited --
    the point of these two tests is that the *set* is required, not that three
    particular names are.
    """
    rocm, torch_lib = root / "rocm", root / "torch"
    rocm.mkdir()
    torch_lib.mkdir()
    for soname in sonames_to_scan():
        if soname != omit:
            _fake_so(rocm / soname, ["HIP_LAUNCH_BLOCKING"])
    return [rocm, torch_lib]


def test_a_missing_named_library_is_an_unreadable_stack_not_a_smaller_scan(
    monkeypatch, tmp_path
):
    """The third door into the green skip, and the one that opened onto the rot check.

    ``require_scanned_libraries`` asked whether *anything* was scanned, so a
    stack that had lost exactly one library walked straight through: the HIP
    and HSA libraries are enough to make the list non-empty. That is the worst
    shape this can be in, because the verdicts disagree about it.
    ``test_every_mitigation_variable_is_read_by_something`` gets louder and a
    human looks at it. ``test_known_absent_entries_are_still_absent`` goes
    quiet and passes -- and if the missing library is a ``claimed_consumer``,
    as ``libhipblaslt`` is here, it passes *by* not reading the only binary
    that could have retired the entry.

    The last two assertions are that failure mode, made concrete: the scan
    looks healthy and the rot check clears nobody, which is indistinguishable
    from a hipBLASLt that was read and did not carry ``DISABLE_TF32``.

    A misspelt ``claimed_consumer`` arrives in the same place and is why this
    is checked at resolution rather than trusted from the list.
    ``test_every_known_absent_claim_is_scanned`` requires the name to be in
    ``sonames_to_scan()``, which ``libhipblastl.so`` satisfies; nothing before
    this required it to resolve to a file.
    """
    dirs = _stack(tmp_path, omit="libhipblaslt.so")
    scan = libraries_to_scan(dirs)

    assert scan.unresolved == ("libhipblaslt.so",), scan.unresolved
    assert scan.libraries, "the rest of the stack is present, which is the trap"
    assert find_fixed_known_absent(
        {name for lib in scan.libraries for name in names_in_binary(lib)}
    ) == [], "a vacuous pass: the entry's claimed consumer was never opened"

    monkeypatch.delenv(REQUIRE_ROCM_ENV, raising=False)
    outcome = _outcome_of(lambda: require_scanned_libraries(scan, dirs))
    assert isinstance(outcome, pytest.skip.Exception), outcome

    monkeypatch.setenv(REQUIRE_ROCM_ENV, "1")
    outcome = _outcome_of(lambda: require_scanned_libraries(scan, dirs))
    assert isinstance(outcome, pytest.fail.Exception), (
        f"expected a failure, got {type(outcome).__name__}: {outcome}"
    )
    assert "libhipblaslt.so" in str(outcome), (
        f"the message has to name the library to act on: {outcome}"
    )


def test_a_complete_stack_is_not_refused(monkeypatch, tmp_path):
    """The narrowness control: refusing every stack would satisfy the test above.

    Also pins the union rule. ``libhipblaslt`` is in the second directory and
    nothing else is, which is the wheel layout in miniature -- resolution is
    "found in some ROCm directory", and a per-directory rule would call this
    complete stack incomplete twice over.
    """
    monkeypatch.setenv(REQUIRE_ROCM_ENV, "1")
    dirs = _stack(tmp_path, omit="libhipblaslt.so")
    second = tmp_path / "libs"
    second.mkdir()
    _fake_so(second / "libhipblaslt.so", ["DISABLE_TF32"])
    dirs.insert(-1, second)

    scan = libraries_to_scan(dirs)

    assert scan.unresolved == (), scan.unresolved
    assert require_scanned_libraries(scan, dirs) == scan.libraries


def test_the_declared_sonames_are_not_read_twice(tmp_path):
    """One file is one read, however many names point at it.

    Two ways to arrive at the same library. The old scan resolved each soname
    and then globbed the same directory, so every declared library was read
    once through the resolver and again through its unversioned link. The
    directory dedup in `rocm_lib_dirs` cannot see that: it deduplicates
    *directories*, not libraries. Here torch's directory is ROCm's, which is
    the case that survives first-match-wins -- the glob is still a second
    route to a file the resolver already found.
    """
    rocm = tmp_path / "rocm"
    rocm.mkdir()
    real = _fake_so(rocm / "libamdhip64.so.7", ["HIP_LAUNCH_BLOCKING"])
    (rocm / "libamdhip64.so").symlink_to(real)

    scanned = libraries_to_scan([rocm, rocm]).libraries

    assert scanned == [real.resolve()], scanned


def test_the_first_directory_that_provides_a_soname_wins(tmp_path):
    """Loader precedence, not a union: a second copy of a library is not read.

    The scan took *every* copy of every soname across the ROCm directories and
    unioned the names out of all of them. Where the directories hold disjoint
    sonames -- the wheel layout's usual shape -- that is harmless. Where they
    overlap it is a claim no process ever sees: exactly one ``libamdhip64`` is
    loaded, and the other one's strings were answering for it.

    Both verdicts move the wrong way, and quietly. A variable that only the
    unloaded copy carries passes
    ``test_every_mitigation_variable_is_read_by_something`` while nothing at
    runtime reads it, which is the precise defect this file exists to catch. A
    ``KNOWN_ABSENT`` entry can be retired by a binary that is not in the
    process.

    First-match-wins over ``rocm_lib_dirs``'s order, which is the order the
    GPU lane puts on ``LD_LIBRARY_PATH`` -- ``gpu-tests.yml`` builds it from
    ``dict.fromkeys([core_lib_dir, lib_dir])``, the same pair from the same
    resolver. Not a full model of the loader, and it does not claim to be:
    ``DT_RPATH`` and an inherited ``LD_LIBRARY_PATH`` both outrank it. It does
    not have to be. A union is wrong by construction -- it describes no
    process -- where first-match describes the one the lane configures.
    """
    first, second = tmp_path / "core", tmp_path / "libs"
    torch_lib = tmp_path / "torch"
    for directory in (first, second, torch_lib):
        directory.mkdir()
    live = _fake_so(first / "libamdhip64.so", ["HIP_LAUNCH_BLOCKING"])
    stale = _fake_so(second / "libamdhip64.so", ["PYTORCH_CUDA_ALLOC_CONF"])

    scan = libraries_to_scan([first, second, torch_lib])

    assert scan.libraries == [live.resolve()], scan.libraries
    assert stale.resolve() not in scan.libraries
    assert "libamdhip64.so" not in scan.unresolved, (
        "found in the first directory, so it is resolved -- winning is not "
        "the same as being missing from the others"
    )
    assert "PYTORCH_CUDA_ALLOC_CONF" not in union_of_names(scan.libraries), (
        "a name only the unloaded copy carries must not answer for the live one"
    )


def test_an_unreadable_scanned_object_is_an_unreadable_stack(monkeypatch, tmp_path):
    """The fourth door, and the last one that let a partial union look complete.

    The read was wrapped in ``except (ValueError, OSError): continue``, so a
    resolved library that was not an ELF, or that could not be opened,
    contributed nothing and said nothing. Nothing is also what a library that
    *was* read and carried no audited name contributes, and the union cannot
    tell those apart -- so the scan came out looking authoritative over a
    binary it never read.

    Both shapes are covered because they arrive differently: a text file named
    ``.so`` raises ``ValueError`` from the magic check, and a mode-000 file
    raises ``OSError`` from ``open``. The second is skipped under a UID that
    ignores file permissions, which is the usual shape of a container build.
    """
    good = _fake_so(tmp_path / "libamdhip64.so", ["HIP_LAUNCH_BLOCKING"])
    not_elf = tmp_path / "libnot.so"
    not_elf.write_text("INPUT(libc.so.6)\n", encoding="utf-8")
    scanned = [good.resolve(), not_elf]

    monkeypatch.delenv(REQUIRE_ROCM_ENV, raising=False)
    outcome = _outcome_of(lambda: union_of_names(scanned))
    assert isinstance(outcome, pytest.skip.Exception), outcome

    monkeypatch.setenv(REQUIRE_ROCM_ENV, "1")
    outcome = _outcome_of(lambda: union_of_names(scanned))
    assert isinstance(outcome, pytest.fail.Exception), (
        f"expected a failure, got {type(outcome).__name__}: {outcome}"
    )
    assert "libnot.so" in str(outcome), (
        f"the message has to name the file to act on: {outcome}"
    )

    unopenable = _fake_so(tmp_path / "libshy.so", ["HIP_LAUNCH_BLOCKING"])
    unopenable.chmod(0o000)
    if os.access(unopenable, os.R_OK):          # root, or an ACL; the mode is advisory
        pytest.skip("this UID can read a mode-000 file, so there is nothing to refuse")
    outcome = _outcome_of(lambda: union_of_names([good.resolve(), unopenable]))
    assert isinstance(outcome, pytest.fail.Exception), (
        f"expected a failure, got {type(outcome).__name__}: {outcome}"
    )


def test_a_fully_readable_scan_returns_the_union(tmp_path):
    """The narrowness control: refusing every scan would satisfy the test above."""
    first = _fake_so(tmp_path / "libamdhip64.so", ["HIP_LAUNCH_BLOCKING"])
    second = _fake_so(tmp_path / "librccl.so", ["NCCL_DEBUG"])

    assert union_of_names([first, second]) == frozenset(
        {"HIP_LAUNCH_BLOCKING", "NCCL_DEBUG"}
    )


def test_the_require_flag_is_off_unless_it_says_otherwise(monkeypatch):
    """``AORTA_REQUIRE_ROCM=0`` in a shell profile must not arm it."""
    monkeypatch.delenv(REQUIRE_ROCM_ENV, raising=False)
    assert rocm_is_required() is False
    for value in ("", " ", "0", "false", "No", "off"):
        monkeypatch.setenv(REQUIRE_ROCM_ENV, value)
        assert rocm_is_required() is False, value
    for value in ("1", "true", "yes"):
        monkeypatch.setenv(REQUIRE_ROCM_ENV, value)
        assert rocm_is_required() is True, value


def test_audit_script_is_reusable_from_here():
    """``resolve_library`` is borrowed rather than restated; prove it still loads."""
    module = _load_audit_script()
    assert callable(module.resolve_library)


_GPU_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "gpu-tests.yml"


def test_the_gpu_lane_arms_the_require_flag_and_selects_this_file():
    """Pin both halves of "it actually runs there", from the only place that can.

    Neither is reachable from a unit test of the code it breaks, so they are
    pinned the way this repo already pins its other workflow invariants (see
    ``test_the_gpu_gate_exercises_the_resolved_default``). Dropping either line
    disarms this audit *silently*: without the flag a stack it cannot read
    becomes a green skip, and without the change filter the job never selects
    the file at all.
    """
    workflow = _GPU_WORKFLOW.read_text(encoding="utf-8")
    assert f"export {REQUIRE_ROCM_ENV}=1" in workflow, (
        f"gpu-tests.yml no longer exports {REQUIRE_ROCM_ENV}, so a lane that "
        "cannot find ROCm would skip this audit and report success."
    )
    for pattern in (
        "'src/aorta/registry/mitigations.py'",
        "'tests/registry/test_mitigation_variable_presence.py'",
        # Not the registry and not the test, but it decides *which directories
        # get scanned*: `scan_plan` reads `resolve_rocm_roots` for both the
        # classic and the TheRock layouts. A resolver-only PR repoints the
        # audit without touching either file above.
        "'src/aorta/instrumentation/rocm_paths.py'",
    ):
        assert pattern in workflow, (
            f"{pattern} is not in the GPU job's change filter, so a change to "
            "the registry would not select the only lane that can audit it."
        )


# ---------------------------------------------------------------------------
# the real check, where the libraries exist
# ---------------------------------------------------------------------------

def require_readable_stack() -> list[Path]:
    """The directories to scan; fails or skips when there are none.

    Separate from the fixture so the skip-or-fail decision -- the whole point
    of :data:`REQUIRE_ROCM_ENV` -- is assertable without a ROCm machine and
    without reaching inside a fixture object.
    """
    dirs, why_not = scan_plan()
    if dirs:
        return dirs
    if rocm_is_required():
        pytest.fail(
            f"{REQUIRE_ROCM_ENV} is set, so this lane promised a stack this "
            f"audit can read, and it cannot: {why_not} Skipping here would "
            "report the audit as having run when it ran nothing."
        )
    pytest.skip(
        f"{why_not} Runs on a lane with a ROCm stack; inert elsewhere, and "
        "saying so rather than passing quietly."
    )


class ScanSet(NamedTuple):
    """What :func:`libraries_to_scan` opened, and which sonames it could not find.

    ``unresolved`` is a return value rather than a silent omission because a
    soname that resolves in no directory does not make the scan *smaller*, it
    makes it answer a different question. ``present_names`` is a union, so a
    library that was never opened is indistinguishable from one that was
    opened and carried none of the audited names -- and the two verdicts read
    that union in opposite directions:

    * :func:`find_unread_mitigations` gets *louder*. Every variable only the
      missing library reads becomes an offender, which is a false alarm but a
      visible one.
    * :func:`find_fixed_known_absent` goes *quiet*, and that is the dangerous
      direction. Every entry in :data:`KNOWN_ABSENT` stays excused, including
      one whose claimed consumer is exactly the library that went missing --
      so the single binary that could retire the exemption is the binary not
      read, and the rot check passes by confirming its own assumption. That is
      the unfalsifiability :class:`Exemption` was introduced to end, arriving
      through the back door.

    A typo reaches the same place: ``claimed_consumer="libhipblastl.so"``
    resolves nowhere, is scanned nowhere, and excuses the entry forever.
    :func:`test_every_known_absent_claim_is_scanned` cannot catch it -- it
    checks the name is in the scan *list*, which a misspelling satisfies.
    """

    libraries: list[Path]
    unresolved: tuple[str, ...]


#: A shared object's filename, versioned or not: ``libtorch_hip.so`` and
#: ``libtorch_hip.so.2`` both match, ``libtorch_hip.so.debug`` does not. The
#: version part is required to be numeric for that reason -- a ``*.so*`` glob
#: on its own sweeps up debug sidecars and ``.so.orig`` backups, which are not
#: ELF and would take the whole scan to :func:`_unreadable_stack`.
_SHARED_OBJECT_RE = re.compile(r"\.so(\.\d+)*$")


def torch_shared_objects(torch_lib: Path) -> set[Path]:
    """Every shared object in torch's directory, versioned names included.

    ``glob("*.so")`` dropped the SONAME-versioned layout, which is not a
    hypothetical layout: ``instrumentation/environment.py`` matches
    ``libtorch_hip.so.2`` by name in ``_loaded_lib_path_from_maps`` precisely
    because torch ships it that way. A wheel carrying only the versioned file
    was therefore scanned as an *empty directory*, and this is the failure the
    scan set is least able to report -- :class:`ScanSet` only knows a ROCm
    soname resolved nowhere, and torch's side declares no sonames at all, so
    nothing was unresolved and nothing was missing. The scan simply got
    smaller, which moves both verdicts the wrong way: a variable only torch
    reads is reported unread, and a :data:`KNOWN_ABSENT` entry whose claimed
    consumer is a torch library stays excused by the binary that was not read.

    Resolved here rather than by the caller because that is what makes the
    widened pattern safe: ``libtorch_hip.so`` and ``libtorch_hip.so.2`` are two
    names for one inode on a devel install, and reading both would be the same
    double read the ROCm side already refuses. A set, so the dedup happens
    before the caller's ordering does.
    """
    return {
        lib.resolve()
        for lib in torch_lib.glob("*.so*")
        if _SHARED_OBJECT_RE.search(lib.name)
    }


def libraries_to_scan(dirs: list[Path]) -> ScanSet:
    """The shared objects to read names out of, resolved from ``dirs``.

    Declared sonames in the ROCm directories, and everything in torch's.
    :func:`scan_plan` appends torch's directory last and it is the one side
    with no soname declared for it -- torch's variables are spread across
    ``libtorch_*``/``libc10*`` rather than concentrated in a named library --
    so it is the only place a glob is the right instrument.

    **The ROCm side is not globbed, and that is what :func:`sonames_to_scan`
    means.** A ``*.so`` glob read ``librocblas``, ``libMIOpen`` and the rest
    of the GEMM stack -- 4.43 GB where 365 MB answers the question -- and
    re-read each declared soname through its unversioned symlink on top.
    Measured on ROCm 7.0.2: 25.9 s globbed against 2.5 s declared, with an
    identical verdict for all seventeen registry variables. The fixture is
    module-scoped and the GPU lane runs ``-n 4``, so the two ``rocm``-marked
    tests can land on different xdist workers and pay it twice.

    ``libhipblaslt`` is now in the named set and was not when those numbers
    were taken, so the declared figure is higher than 2.5 s by that library's
    read; it is there because :data:`KNOWN_ABSENT` names it, not because a
    glob swept it up. See :func:`sonames_to_scan` for why a claimed consumer
    has to be opened.

    **One library per soname, from the first directory that provides it.**
    The loop reads the soname on the outside and stops at the first hit, so
    the directory order decides -- which is the order the GPU lane puts on
    ``LD_LIBRARY_PATH``: ``gpu-tests.yml`` builds it from
    ``dict.fromkeys([core_lib_dir, lib_dir])`` and :func:`rocm_lib_dirs`
    returns that same pair in that same order. It still supports the split
    wheel layout, because there the two directories hold disjoint sonames and
    "first that provides it" is the only one that provides it.

    Taking every copy instead unioned the names out of libraries no process
    ever loads. See :func:`test_the_first_directory_that_provides_a_soname_wins`
    for why that moves both verdicts the wrong way.

    Deduplicated by resolved path even so, because the torch directory is
    globbed rather than resolved and an install whose torch libraries sit
    beside ROCm's reaches the same file both ways. The glob's results are
    resolved for that to work at all: ``libamdhip64.so`` and
    ``libamdhip64.so.7`` are two paths to one inode, and a dedup over
    unresolved paths kept both -- so the dedup's own claim, one file is one
    read, did not hold on the side that needed it.
    """
    audit = _load_audit_script()
    *rocm_dirs, torch_lib = dirs
    scanned: list[Path] = []
    resolved: set[str] = set()
    for soname in sonames_to_scan():
        for directory in rocm_dirs:
            lib = audit.resolve_library(directory, soname)
            if lib is not None:
                resolved.add(soname)
                scanned.append(lib)
                break
    scanned.extend(sorted(torch_shared_objects(torch_lib)))
    return ScanSet(
        list(dict.fromkeys(scanned)),
        tuple(soname for soname in sonames_to_scan() if soname not in resolved),
    )


def _unreadable_stack(why: str) -> NoReturn:
    """Fail where a lane promised a stack, skip where none was promised.

    The same sentence in both places that can reach it, so the two doors into
    the green skip cannot drift apart in wording or in policy.
    """
    if rocm_is_required():
        pytest.fail(
            f"{REQUIRE_ROCM_ENV} is set, so this lane promised a stack this "
            f"audit can read, and it cannot: {why}. Skipping here would report "
            "the audit as having run when it ran nothing."
        )
    pytest.skip(why)


def require_scanned_libraries(scan: ScanSet, dirs: list[Path]) -> list[Path]:
    """``scan.libraries``, or the fail-or-skip :func:`require_readable_stack` makes.

    Separate from the fixture for the same reason that one is: the decision is
    the point, so it has to be assertable off a GPU lane.

    This skip used to be unconditional, which put the green skip back right
    after the flag had closed it. ``rocm_lib_dirs`` only requires the
    directories to *exist*, so a base-image bump that moves the shared objects
    into a subdirectory leaves ``dirs`` non-empty and the scan empty, and the
    lane that promised a stack reported an audit that read nothing as having
    run.

    A *partly* readable stack is the same defect one size down, and used to
    pass: "did we open anything" is not "did we open the binaries this audit
    is about". :class:`ScanSet` says why a missing soname cannot be treated as
    a smaller scan. Both branches end at :func:`_unreadable_stack` because the
    remedy is the same one -- fix the image, then re-run -- and neither is a
    verdict about the registry.

    Nothing-at-all is reported first even though the unresolved list would
    also be non-empty there: on an empty tree every soname is missing, and
    naming them all describes the symptom where "no shared objects under
    ``<dirs>``" names the cause.
    """
    if not scan.libraries:
        _unreadable_stack(f"no shared objects under {[str(d) for d in dirs]}")
    if scan.unresolved:
        _unreadable_stack(
            f"{list(scan.unresolved)} resolved in none of "
            f"{[str(d) for d in dirs[:-1]]}, so the audit would read a union "
            "missing whatever only those libraries carry -- under which an "
            "unread variable looks unread and a KNOWN_ABSENT entry whose "
            "claimed consumer is one of them stays excused on no evidence"
        )
    return scan.libraries


def union_of_names(scanned: list[Path]) -> frozenset[str]:
    """Every audited name in ``scanned``, or the fail-or-skip an unreadable one earns.

    The read used to be wrapped in ``except (ValueError, OSError): continue``
    -- "not an ELF, or unreadable; neither is this test's business". It is
    this test's business, and for the reason :class:`ScanSet` gives about a
    soname that resolves nowhere: a library that was opened and could not be
    read contributes nothing to the union, and nothing is exactly what a
    library that was read and carried no audited name contributes. The two are
    the same value and different facts.

    :func:`require_scanned_libraries` refuses a stack that is missing a named
    library; a named library present but unreadable is that stack wearing a
    file. The rot check is again where it bites: an entry's
    ``claimed_consumer`` that lands here stays excused without the one binary
    that could retire it ever being read.

    Torch's globbed objects are held to the same rule rather than tolerated as
    incidental. Three of the eighteen variables under audit are read only by
    them, so an unreadable one weakens the same union -- and the failure names
    the file, which is what makes an image that ships a genuinely non-ELF
    ``.so`` a one-line fix instead of a mystery.
    """
    found: set[str] = set()
    unreadable: list[str] = []
    for lib in scanned:
        try:
            found |= names_in_binary(lib)
        except (ValueError, OSError) as exc:
            unreadable.append(f"{lib} ({exc})")
    if unreadable:
        _unreadable_stack(
            f"{len(unreadable)} of {len(scanned)} scanned objects could not be "
            f"read: {unreadable}. Their names are missing from the union, "
            "under which an unread variable looks unread and a KNOWN_ABSENT "
            "entry whose claimed consumer is one of them stays excused"
        )
    return frozenset(found)


@pytest.fixture(scope="module")
def present_names() -> frozenset[str]:
    dirs = require_readable_stack()
    return union_of_names(require_scanned_libraries(libraries_to_scan(dirs), dirs))


@pytest.mark.rocm
def test_every_mitigation_variable_is_read_by_something(present_names):
    offenders = find_unread_mitigations(present_names)
    assert not offenders, (
        "these mitigations set an environment variable that appears in no "
        f"scanned binary, so nothing reads it: {offenders}. A probe cell for "
        "one of them is a second baseline under a different name. Either fix "
        "the entry or add it to KNOWN_ABSENT with the evidence. See aorta#511."
    )


@pytest.mark.rocm
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


def test_a_second_variable_on_a_listed_mitigation_is_still_reported(monkeypatch):
    """The exemption covers the listed variable, not the mitigation.

    A mitigation is an env-var bundle. Adding a second, unread variable to
    ``tf32_off`` is a new defect of exactly the kind this guard exists to
    catch, and the mitigation-name key silenced it -- the one entry in the list
    would have excused a variable nobody had ever looked at.
    """
    monkeypatch.setitem(
        BUILTIN_MITIGATIONS, "tf32_off", {"DISABLE_TF32": "1", "NEWLY_ADDED": "1"}
    )
    present = set(mitigation_variables()) - {"DISABLE_TF32", "NEWLY_ADDED"}
    assert find_unread_mitigations(present) == {"NEWLY_ADDED": {"tf32_off"}}


def test_a_variable_shared_with_an_unlisted_mitigation_is_still_reported(monkeypatch):
    """The other direction: one variable, two mitigations, one of them listed.

    ``TORCH_ROCM_FA_PREFER_CK`` and ``HSA_DISABLE_CACHE`` are each set by two
    registry entries today, so this is the shape the registry already has.
    Exempting one entry must not answer for the other.
    """
    monkeypatch.setitem(BUILTIN_MITIGATIONS, "tf32_off_but_louder", {"DISABLE_TF32": "1"})
    present = set(mitigation_variables()) - {"DISABLE_TF32"}
    assert find_unread_mitigations(present) == {
        "DISABLE_TF32": {"tf32_off_but_louder"}
    }


def _synthetic_exemption(monkeypatch) -> tuple[str, str]:
    """A ``KNOWN_ABSENT`` entry supplied by the test rather than borrowed.

    The two tests below are about :func:`find_fixed_known_absent`, and the
    backlog is only their fixture. Borrowing it made them say nothing once it
    was empty -- and guarding that with ``assert KNOWN_ABSENT`` turned the
    empty backlog into a permanent red, which is worse: an empty backlog is
    this lane's *success* condition, both defects closed, and reaching it must
    not be the thing that breaks the test that watched for it.

    Added to the live mapping rather than replacing it, so the real entries are
    still covered by the equality below while there are any.
    """
    monkeypatch.setitem(
        BUILTIN_MITIGATIONS, "synthetic_mitigation", {"SYNTHETIC_ABSENT": "1"}
    )
    monkeypatch.setitem(
        KNOWN_ABSENT,
        ("synthetic_mitigation", "SYNTHETIC_ABSENT"),
        Exemption(
            claimed_consumer="libamdhip64.so",
            reason="fixture for this test; never reaches the real backlog",
        ),
    )
    return ("synthetic_mitigation", "SYNTHETIC_ABSENT")


def test_a_known_absent_entry_that_came_back_is_flagged(monkeypatch):
    """With everything readable, every listed pair is a pair that came back.

    Stated as ``sorted(KNOWN_ABSENT)`` rather than as the pairs in it today:
    hard-coding them makes fixing `tf32_off` break a test that is not about
    `tf32_off`, and the assertion being made here is about the function.
    """
    synthetic = _synthetic_exemption(monkeypatch)
    present = set(mitigation_variables())          # everything readable
    # Named explicitly as well as compared, so the test still asserts something
    # specific on the day the real backlog is empty.
    assert synthetic in find_fixed_known_absent(present)
    assert find_fixed_known_absent(present) == sorted(KNOWN_ABSENT)


def test_nothing_is_flagged_while_the_backlog_is_genuinely_absent(monkeypatch):
    """The narrowness control, on the same supplied entry for the same reason."""
    synthetic = _synthetic_exemption(monkeypatch)
    present = set(mitigation_variables()) - {v for _, v in KNOWN_ABSENT}
    assert synthetic[1] not in present
    assert find_fixed_known_absent(present) == []


def test_a_sibling_variable_becoming_readable_does_not_clear_the_entry(monkeypatch):
    """The same coarseness defect, on the rot check rather than the audit.

    Under the mitigation-name key this asked "is ANY of this mitigation's
    variables present", so a readable sibling declared the listed one fixed and
    the entry would have been removed while the variable it records is still
    absent -- deleting the only evidence aorta#500 is still open.
    """
    monkeypatch.setitem(
        BUILTIN_MITIGATIONS, "tf32_off", {"DISABLE_TF32": "1", "READABLE_SIBLING": "1"}
    )
    assert find_fixed_known_absent({"READABLE_SIBLING"}) == []
    assert find_fixed_known_absent({"DISABLE_TF32"}) == [("tf32_off", "DISABLE_TF32")]
