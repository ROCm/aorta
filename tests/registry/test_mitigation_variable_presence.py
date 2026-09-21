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
  audits nothing is worse than one that is not there.

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

import pytest

from aorta.instrumentation.rocm_paths import RocmRoots, resolve_rocm_roots, safe_is_dir
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

#: ``(mitigation, variable)`` pairs known to be absent from every scanned
#: binary, each with the reason. The lane must not go red over a backlog on day
#: one -- the point is to catch the *next* one.
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
KNOWN_ABSENT: dict[tuple[str, str], str] = {
    ("tf32_off", "DISABLE_TF32"): (
        "DISABLE_TF32 appears in no ROCm or torch binary; aorta#500. The "
        "registry attributes it to hipBLASLt and instrumentation/env_knobs.py "
        "attributes it to pytorch, and neither holds."
    ),
    ("rccl_gfx942_cheap_fence_off", "RCCL_GFX942_CHEAP_FENCE_OFF"): (
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
    """``(directories to scan, why not)``. A reason is non-empty iff the list is.

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


@pytest.fixture(scope="module")
def present_names() -> frozenset[str]:
    dirs = require_readable_stack()
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


def test_a_known_absent_entry_that_came_back_is_flagged():
    present = set(mitigation_variables())          # everything readable
    assert find_fixed_known_absent(present) == [
        ("rccl_gfx942_cheap_fence_off", "RCCL_GFX942_CHEAP_FENCE_OFF"),
        ("tf32_off", "DISABLE_TF32"),
    ]


def test_nothing_is_flagged_while_the_backlog_is_genuinely_absent():
    present = set(mitigation_variables()) - {"DISABLE_TF32", "RCCL_GFX942_CHEAP_FENCE_OFF"}
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
