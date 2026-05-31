"""Tier 3 kernel / GPU detectors via ``dmesg`` and ``amd-smi`` shims.

Two complementary signal sources:

* ``dmesg --since=<probe_start>`` -- kernel ring buffer entries
  emitted by the amdgpu / kfd drivers between trial start and trial
  end. Detector IDs cover the failure modes the AORTA team has seen
  enough times to bake into the platform:

  - ``tier3:amdgpu_reset`` -- "amdgpu: GPU reset" lines
  - ``tier3:sdma_timeout`` -- "SDMA semaphore timeout" / "SDMA hang"
  - ``tier3:vm_l2_fault`` -- "VM_L2_PROTECTION_FAULT" lines
  - ``tier3:xgmi_link_error`` -- "XGMI" link-error lines
  - ``tier3:pcie_aer_fatal`` -- "AER" fatal PCIe errors

* ``amd-smi`` -- counters polled twice (pre/post-trial) and diffed:

  - ``tier3:vram_growth`` -- VRAM used delta exceeds the trial's
    starting VRAM by more than :data:`VRAM_GROWTH_THRESHOLD_MIB`.
    Catches leaks the workload's own ``peak_vram_mib`` doesn't
    reveal because the workload exited cleanly but left the VRAM
    allocated under another process.
  - ``tier3:thermal_throttle`` -- thermal throttling counter
    incremented during the trial.

Both signal sources are **fail-soft**: a missing binary (``dmesg``,
``amd-smi`` not on ``PATH``), a non-zero exit status, an unparseable
line — every error path logs a single ``tier3 disabled: <reason>``
warning the first time it fires per ``aorta probe`` invocation and
otherwise returns no fired detectors. Tiers 1 + 2 + 4 continue;
operators see a clear "tier 3 unavailable" signal in the runner
log instead of a sea of per-trial failures (rubric §2.B FR 2.11).

The Tier 3 detector functions are intentionally pure given their
inputs (a captured ``dmesg`` text blob and an ``amd-smi`` snapshot
pair) so they're unit-testable without a real GPU / kernel
(rubric §2.B FR 2.13). The *invocation* code calls
``shutil.which`` and subprocess-runs the shim, but that wrapper is
covered by the missing-binary test which exercises the fail-soft
branch.
"""

from __future__ import annotations

import csv
import io
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

# Detector IDs (stable contract, see Tier 1 / Tier 4 docstrings).
DETECTOR_AMDGPU_RESET = "tier3:amdgpu_reset"
DETECTOR_SDMA_TIMEOUT = "tier3:sdma_timeout"
DETECTOR_VM_L2_FAULT = "tier3:vm_l2_fault"
DETECTOR_XGMI_LINK_ERROR = "tier3:xgmi_link_error"
DETECTOR_PCIE_AER_FATAL = "tier3:pcie_aer_fatal"
DETECTOR_VRAM_GROWTH = "tier3:vram_growth"
DETECTOR_THERMAL_THROTTLE = "tier3:thermal_throttle"

# VRAM growth that crosses this threshold (MiB) between pre- and
# post-trial polls fires :data:`DETECTOR_VRAM_GROWTH`. Chosen well
# above the noise floor on a multi-tenant host where unrelated
# processes can move VRAM by a few MiB during the trial.
VRAM_GROWTH_THRESHOLD_MIB = 256

# Cap on the amount of dmesg text we scan per trial. Same rationale
# as :data:`aorta.probe.sandbox.MAX_LOG_BYTES`: a runaway dmesg
# producing > 10MiB of output in one trial would point at a kernel
# log loop, not the trial; scanning more than this is wasted work.
MAX_DMESG_BYTES = 10 * 1024 * 1024

# Compiled patterns. Module-level so the cost is paid once per
# process. ``MULTILINE`` so ``^`` / ``$`` match per line.
_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        DETECTOR_AMDGPU_RESET,
        re.compile(r"amdgpu: GPU reset", re.MULTILINE | re.IGNORECASE),
    ),
    (
        DETECTOR_SDMA_TIMEOUT,
        re.compile(r"SDMA semaphore timeout|SDMA hang", re.MULTILINE | re.IGNORECASE),
    ),
    (
        DETECTOR_VM_L2_FAULT,
        re.compile(r"VM_L2_PROTECTION_FAULT", re.MULTILINE),
    ),
    (
        DETECTOR_XGMI_LINK_ERROR,
        # 'XGMI' lines that carry 'error' / 'link' / 'failed' --
        # narrow enough to avoid a healthy 'XGMI initialized'
        # message firing the detector.
        re.compile(r"XGMI.*(?:error|fail|link down)", re.MULTILINE | re.IGNORECASE),
    ),
    (
        DETECTOR_PCIE_AER_FATAL,
        re.compile(r"AER:?\s+Fatal", re.MULTILINE),
    ),
)


@dataclass
class Tier3State:
    """Per-``aorta probe`` invocation state for Tier 3.

    Tracks the "tier3 disabled" warning so it's emitted at most
    once even when scanning N cells x M trials (rubric §2.B FR
    2.11). Pass the SAME instance into every call to
    :func:`scan_dmesg` / :func:`scan_amd_smi` over the course of
    one ``aorta probe`` run.
    """

    dmesg_disabled_logged: bool = False
    amdsmi_disabled_logged: bool = False
    # Disabled reasons captured for ``result.json::capture`` if the
    # workload wants to surface them — not currently consumed but
    # recorded so a future audit can reconstruct what was skipped.
    disabled_reasons: list[str] = field(default_factory=list)


def scan_dmesg_text(text: str) -> list[str]:
    """Scan an already-captured ``dmesg`` text blob for tier-3 patterns.

    Pure: no subprocess, no FS. The :func:`scan_dmesg` wrapper does
    the subprocess and threads its output through here. Exposed
    separately so the patterns can be unit-tested against canned
    text fixtures (rubric §2.B FR 2.13).
    """
    if not text:
        return []
    if len(text) > MAX_DMESG_BYTES:
        # Keep the *tail*, not the head. The XGMI / HBM / MMU
        # kernel signatures Tier 3 looks for are almost always
        # emitted in the seconds before the trial ends -- i.e. at
        # the end of the dmesg ring. The previous head-slice
        # discarded exactly those lines on long-running trials
        # where the ring filled up. Per Sonbol's PR #197 review.
        text = text[-MAX_DMESG_BYTES:]
    fired: list[str] = []
    for detector_id, pattern in _PATTERNS:
        if pattern.search(text):
            fired.append(detector_id)
    return fired


def scan_dmesg(
    state: Tier3State,
    since_seconds: float | None = None,
) -> list[str]:
    """Invoke ``dmesg`` and return the fired tier-3 kernel detectors.

    Fail-soft on every error path. Returns ``[]`` when:

    * ``dmesg`` is not on ``PATH``.
    * ``dmesg`` exits non-zero (permission denied is the common
      case in unprivileged containers; ``CAP_SYSLOG`` is required
      on recent kernels).
    * The subprocess raises (timeout, ``OSError``).

    On the first failure for a given invocation, a single warning
    is logged via the module logger; subsequent calls during the
    same invocation are silent. The ``state`` object carries the
    "already logged" bit across calls.
    """
    binary = shutil.which("dmesg")
    if binary is None:
        _log_disabled_once(state, "dmesg", "dmesg not on PATH")
        return []

    # ``--since=<n>seconds`` is the standard idiom on util-linux
    # dmesg. We pass ``--no-pager`` for portability across systems
    # where the operator's PAGER is set.
    argv: list[str] = [binary, "--no-pager"]
    if since_seconds is not None and since_seconds > 0:
        # 'X seconds ago' is the human shape util-linux accepts.
        argv.extend(["--since", f"{int(since_seconds)} seconds ago"])

    try:
        completed = subprocess.run(  # noqa: S603 -- audited argv list
            argv,
            capture_output=True,
            text=True,
            check=False,
            timeout=10.0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        _log_disabled_once(state, "dmesg", f"{type(exc).__name__}: {exc}")
        return []
    if completed.returncode != 0:
        # Common case: 'dmesg: read kernel buffer failed: Operation
        # not permitted'. Logged once; tiers 1+2+4 continue.
        reason = (completed.stderr or completed.stdout or "non-zero exit").strip()
        _log_disabled_once(state, "dmesg", f"non-zero exit: {reason!r}")
        return []
    return scan_dmesg_text(completed.stdout)


@dataclass(frozen=True)
class AmdSmiSnapshot:
    """Minimum fields a Tier 3 amd-smi poll surfaces.

    ``vram_used_mib`` is the cumulative used VRAM across every GPU
    visible to ``amd-smi``; we don't try to attribute usage per
    process (a much harder problem). ``thermal_throttle_count`` is
    the all-GPU sum of the throttle counter -- in live polling we
    can't compute a true monotonic counter from
    ``amd-smi monitor`` alone (the CLI surfaces current % time in
    violation, not a cumulative count) so the live path leaves it
    at ``0``; the fake-shim env var keeps the diff-based
    ``DETECTOR_THERMAL_THROTTLE`` test path working as before.
    ``gpu_utilization_pct`` is the across-GPU max ``GFX%`` value
    (``None`` when amd-smi doesn't expose it or the column isn't
    in the CSV header). The hang monitor uses this as the third
    "GPU idle" leg of the two-of-three Tier 2 predicate.

    Frozen so callers can keep two snapshots side-by-side (pre /
    post) without one mutating the other.
    """

    vram_used_mib: int
    thermal_throttle_count: int
    gpu_utilization_pct: int | None = None


# GPU is considered idle if the max-GPU ``GFX%`` from
# ``amd-smi monitor`` is below this threshold. A running GPU
# workload typically pegs activity at 80-100%; a hung kernel sits at
# 0-2% (the residual is amd-smi's own polling). 5% is well below any
# real compute and well above the noise floor.
GPU_IDLE_UTILIZATION_THRESHOLD_PCT = 5


# Subprocess timeout for amd-smi monitor in the live polling path.
# Longer than dmesg's because monitor enumerates every GPU; an
# unresponsive driver shouldn't block the hang monitor forever.
_AMD_SMI_TIMEOUT_SEC = 10.0


def scan_amd_smi(
    state: Tier3State,
    pre: AmdSmiSnapshot | None,
    post: AmdSmiSnapshot | None,
) -> list[str]:
    """Diff two amd-smi snapshots and return the fired Tier 3 detectors.

    Either snapshot being ``None`` means the polling step failed —
    fail-soft: log once, return ``[]``. When both are present,
    fire :data:`DETECTOR_VRAM_GROWTH` if the delta crosses
    :data:`VRAM_GROWTH_THRESHOLD_MIB` and
    :data:`DETECTOR_THERMAL_THROTTLE` if the throttle counter went
    up during the trial.
    """
    if pre is None or post is None:
        _log_disabled_once(
            state,
            "amd-smi",
            "amd-smi not on PATH or polling failed (see runner log)",
        )
        return []
    fired: list[str] = []
    if post.vram_used_mib - pre.vram_used_mib >= VRAM_GROWTH_THRESHOLD_MIB:
        fired.append(DETECTOR_VRAM_GROWTH)
    if post.thermal_throttle_count > pre.thermal_throttle_count:
        fired.append(DETECTOR_THERMAL_THROTTLE)
    return fired


def poll_amd_smi(state: Tier3State) -> AmdSmiSnapshot | None:
    """Single ``amd-smi`` poll. Returns ``None`` when unavailable.

    Live path: runs ``amd-smi monitor --csv --gfx --vram-usage``
    and parses the (stable, documented) CSV output. ``monitor`` is
    used in preference to ``metric --json`` because the column
    layout is documented and stable across ROCm 6.x / 7.x while
    the ``metric --json`` shape has been observed to differ
    between point releases (e.g. socket-vs-partition layouts on
    MI300). The CSV path covers the two columns we actually need
    (VRAM_USED, GFX%); ``thermal_throttle_count`` is left at 0
    because ``monitor`` only exposes current % time in violation,
    not a cumulative counter -- the diff-based
    ``DETECTOR_THERMAL_THROTTLE`` continues to fire through the
    fake-shim env var path for tests.

    Test stub: ``AORTA_PROBE_AMDSMI_FAKE`` env var. When set to
    ``vram=<int>,throttle=<int>`` (optionally ``,util=<int>``)
    the value is parsed and returned without spawning a
    subprocess. Allows the unit suite to exercise the diff logic
    in :func:`scan_amd_smi` and the gpu-idle leg of
    :func:`tier2_hang.evaluate_predicate` without a real GPU.

    Any error in the live path (missing binary, non-zero exit,
    unparseable header, timeout) returns ``None`` and counts as
    "amd-smi disabled" via :func:`_log_disabled_once` for the
    rest of the invocation -- the runner sees a single
    ``tier3 disabled (amd-smi): <reason>`` warning and Tiers
    1+2+4 continue (rubric §2.B FR 2.11).
    """
    import os

    fake = os.environ.get("AORTA_PROBE_AMDSMI_FAKE")
    if fake is not None:
        return _parse_fake_snapshot(fake, state)

    binary = shutil.which("amd-smi")
    if binary is None:
        _log_disabled_once(state, "amd-smi", "amd-smi not on PATH")
        return None
    return _poll_amd_smi_live(binary, state)


def _poll_amd_smi_live(binary: str, state: Tier3State) -> AmdSmiSnapshot | None:
    """Run ``amd-smi monitor --csv --gfx --vram-usage`` and parse the result.

    Split out from :func:`poll_amd_smi` so the test suite can shim
    the binary path via PATH (same pattern as the dmesg shim
    test). Any subprocess failure or parse failure logs a single
    ``tier3 disabled (amd-smi): ...`` warning and returns
    ``None``.
    """
    argv = [binary, "monitor", "--csv", "--gfx", "--vram-usage"]
    try:
        completed = subprocess.run(  # noqa: S603 -- audited argv list
            argv,
            capture_output=True,
            text=True,
            check=False,
            timeout=_AMD_SMI_TIMEOUT_SEC,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        _log_disabled_once(state, "amd-smi", f"{type(exc).__name__}: {exc}")
        return None
    if completed.returncode != 0:
        reason = (completed.stderr or completed.stdout or "non-zero exit").strip()
        _log_disabled_once(state, "amd-smi", f"non-zero exit: {reason!r}")
        return None
    snapshot = _parse_amd_smi_monitor_csv(completed.stdout)
    if snapshot is None:
        _log_disabled_once(
            state,
            "amd-smi",
            f"unrecognised monitor CSV output: {completed.stdout[:200]!r}",
        )
    return snapshot


# Recognised column-header aliases. Kept here so the parser can
# absorb minor CSV header churn between ROCm releases without
# editing the function body. All comparisons are uppercased and
# stripped of whitespace.
_VRAM_USED_HEADERS = frozenset({"VRAM_USED"})
_GFX_HEADERS = frozenset({"GFX%", "GFX_ACTIVITY", "GFX"})


def _parse_amd_smi_monitor_csv(payload: str) -> AmdSmiSnapshot | None:
    """Parse the CSV from ``amd-smi monitor --csv --gfx --vram-usage``.

    Sums ``VRAM_USED`` across rows (one row per GPU) and takes the
    max of ``GFX%`` (the "idle" probe wants the busiest GPU; if any
    GPU is doing real work the workload isn't hung GPU-wise).
    ``N/A`` cells contribute nothing. Returns ``None`` when the
    payload is empty / missing a recognised header so the caller
    can log a single disabled warning.

    Pure function (no logging, no subprocess) so unit tests can
    exercise the parser directly with deterministic input.
    """
    reader = csv.reader(io.StringIO(payload))
    try:
        header = next(reader)
    except StopIteration:
        return None
    normalised = [cell.strip().upper() for cell in header]
    vram_idx = next((i for i, h in enumerate(normalised) if h in _VRAM_USED_HEADERS), None)
    gfx_idx = next((i for i, h in enumerate(normalised) if h in _GFX_HEADERS), None)
    if vram_idx is None and gfx_idx is None:
        return None

    vram_total_mib = 0
    util_max: int | None = None
    for row in reader:
        if not row:
            continue
        if vram_idx is not None and vram_idx < len(row):
            mib = _parse_mib(row[vram_idx])
            if mib is not None:
                vram_total_mib += mib
        if gfx_idx is not None and gfx_idx < len(row):
            pct = _parse_pct(row[gfx_idx])
            if pct is not None:
                util_max = pct if util_max is None else max(util_max, pct)
    return AmdSmiSnapshot(
        vram_used_mib=vram_total_mib,
        thermal_throttle_count=0,
        gpu_utilization_pct=util_max,
    )


# Permissive numeric-with-unit parsers. amd-smi cells look like
# ``14 MB``, ``96432 MB``, ``0 %``, ``N/A`` -- the regex accepts an
# optional unit suffix, ``MB``/``MiB``/``GB``/``GiB`` (case
# insensitive), and a trailing ``%`` for percentages. Any
# non-match returns ``None`` so the caller can skip the cell.
_MIB_VALUE_RE = re.compile(r"^\s*(\d+)\s*(MB|MIB|GB|GIB)?\s*$", re.IGNORECASE)
_PCT_VALUE_RE = re.compile(r"^\s*(\d+)\s*%?\s*$")


def _parse_mib(cell: str) -> int | None:
    """Parse a memory cell like ``'14 MB'``, returning MiB-or-None.

    ``N/A`` / empty / unrecognised returns ``None``. ``GB`` /
    ``GiB`` values are scaled by 1024; ``MB`` / ``MiB`` (the common
    case) pass through unchanged because at the precision amd-smi
    reports the two are interchangeable for hang-detection
    purposes.
    """
    text = cell.strip()
    if not text or text.upper() == "N/A":
        return None
    m = _MIB_VALUE_RE.match(text)
    if m is None:
        return None
    value = int(m.group(1))
    unit = (m.group(2) or "MB").upper()
    if unit in ("GB", "GIB"):
        return value * 1024
    return value


def _parse_pct(cell: str) -> int | None:
    """Parse a percentage cell like ``'42 %'``, returning the integer.

    ``N/A`` / empty / unrecognised returns ``None``.
    """
    text = cell.strip()
    if not text or text.upper() == "N/A":
        return None
    m = _PCT_VALUE_RE.match(text)
    if m is None:
        return None
    return int(m.group(1))


def _parse_fake_snapshot(spec: str, state: Tier3State) -> AmdSmiSnapshot | None:
    """Parse ``AORTA_PROBE_AMDSMI_FAKE=vram=N,throttle=M[,util=U]`` for tests.

    Returns ``None`` on parse failure (treats it as 'amd-smi not
    available'). The test-only env var is intentionally simple so
    a unit test can wire ``vram=100,throttle=0`` for the pre-poll
    and ``vram=600,throttle=1`` for the post-poll, and assert
    ``scan_amd_smi`` fires both detectors. The optional ``util``
    field feeds the GPU-idle leg of :func:`tier2_hang.evaluate_predicate`
    in tests that exercise the two-of-three predicate.
    """
    try:
        parts = dict(item.split("=") for item in spec.split(","))
        util_raw = parts.get("util")
        util = int(util_raw) if util_raw is not None else None
        return AmdSmiSnapshot(
            vram_used_mib=int(parts.get("vram", "0")),
            thermal_throttle_count=int(parts.get("throttle", "0")),
            gpu_utilization_pct=util,
        )
    except (ValueError, KeyError, AttributeError):
        _log_disabled_once(state, "amd-smi", f"unparseable fake spec: {spec!r}")
        return None


def gpu_idle_probe_from_state(state: Tier3State) -> "callable[[], bool]":  # noqa: UP037
    """Return a zero-arg closure that polls amd-smi and returns "GPU idle?".

    Intended to be wired into :class:`tier2_hang.HangMonitor`'s
    ``gpu_idle_probe`` constructor argument. The closure returns
    ``True`` iff the live amd-smi poll surfaces a max-GPU
    ``GFX%`` below :data:`GPU_IDLE_UTILIZATION_THRESHOLD_PCT`.
    Any None (binary missing, parse fail, utilization column
    absent) yields ``False`` so the GPU leg can never single-
    handedly trip the two-of-three predicate when telemetry is
    unavailable -- consistent with the I/O leg's
    ``current_io is None -> io_idle=False`` rule in
    :class:`HangMonitor._run`.

    Each call spawns one amd-smi subprocess; the HangMonitor polls
    at ``poll_interval_sec`` (default 5s) so we expect ~12 calls
    per minute under default settings, well within the budget for
    a hang-monitoring background thread.
    """

    def _probe() -> bool:
        snap = poll_amd_smi(state)
        if snap is None or snap.gpu_utilization_pct is None:
            return False
        return snap.gpu_utilization_pct < GPU_IDLE_UTILIZATION_THRESHOLD_PCT

    return _probe


def _log_disabled_once(state: Tier3State, source: str, reason: str) -> None:
    """Emit a single ``tier3 disabled:`` warning per source per invocation.

    Per the rubric (FR 2.11): the warning is logged at most once
    for the whole ``aorta probe`` invocation regardless of how
    many cells / trials hit the same failure. Subsequent calls
    silently update the captured reasons list for audit.
    """
    state.disabled_reasons.append(f"{source}: {reason}")
    flag_attr = f"{source.replace('-', '')}_disabled_logged"
    # ``source`` is one of 'dmesg', 'amd-smi'; map to the
    # corresponding flag attribute. ``amd-smi`` collapses to
    # ``amdsmi_disabled_logged`` after the strip.
    if source == "dmesg":
        if state.dmesg_disabled_logged:
            return
        state.dmesg_disabled_logged = True
    elif source == "amd-smi":
        if state.amdsmi_disabled_logged:
            return
        state.amdsmi_disabled_logged = True
    else:  # pragma: no cover - guards future renames
        log.warning("tier3 disabled (%s): %s", source, reason)
        return
    log.warning("tier3 disabled (%s): %s", source, reason)
    # Reference the helper attribute name to silence "unused" linters
    # when the dead branch is removed in future refactors.
    _ = flag_attr


ALL_DETECTOR_IDS = (
    DETECTOR_AMDGPU_RESET,
    DETECTOR_SDMA_TIMEOUT,
    DETECTOR_VM_L2_FAULT,
    DETECTOR_XGMI_LINK_ERROR,
    DETECTOR_PCIE_AER_FATAL,
    DETECTOR_VRAM_GROWTH,
    DETECTOR_THERMAL_THROTTLE,
)


__all__ = [
    "ALL_DETECTOR_IDS",
    "AmdSmiSnapshot",
    "DETECTOR_AMDGPU_RESET",
    "DETECTOR_PCIE_AER_FATAL",
    "DETECTOR_SDMA_TIMEOUT",
    "DETECTOR_THERMAL_THROTTLE",
    "DETECTOR_VM_L2_FAULT",
    "DETECTOR_VRAM_GROWTH",
    "DETECTOR_XGMI_LINK_ERROR",
    "GPU_IDLE_UTILIZATION_THRESHOLD_PCT",
    "MAX_DMESG_BYTES",
    "Tier3State",
    "VRAM_GROWTH_THRESHOLD_MIB",
    "gpu_idle_probe_from_state",
    "poll_amd_smi",
    "scan_amd_smi",
    "scan_dmesg",
    "scan_dmesg_text",
]
