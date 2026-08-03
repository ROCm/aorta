"""rocjitsu sanitizer backends: waitcheck (static) + ConSan (dynamic).

This is the ONE place that encodes how we invoke the rocjitsu sanitizers
and how we read their output. It is reconciled against
``rocm-systems/emulation/rocjitsu/docs/sanitizers.md`` (branch
``shared/rocjitsu/sanitizers``). If that doc changes, update only this
module -- everything else consumes the normalised structures here.

How the two checks actually ship (per the doc):

  * **waitcheck** -- STATIC. The standalone ``rj_waitcheck`` tool inspects a
    saved code object (``.hsaco``, HIP fat binary, executable, shared lib,
    or a directory corpus) WITHOUT running it, and reports missing AMDGPU
    waits. Diagnostics are non-fatal and print with a
    ``rocjitsu-waitcheck:`` prefix. Needs no GPU.

  * **ConSan** -- DYNAMIC. There is no separate ``consan`` binary. ConSan
    is the second stage of the combined HSA-tools hook
    (``librocjitsu_dbi_hooks.so``): you load it via ``HSA_TOOLS_LIB`` and
    run the real application (on hardware, or a native ISA in the RocJITsu
    simulator). The hook runs an exhaustive load-time waitcheck first, then
    instruments the code object for ConSan. Diagnostics print with a
    ``[rocjitsu-dbi-hooks] ConSan`` prefix.

Target support (verbatim from the doc's table; ``support()`` below):
``gfx950`` (MI350X/MI355X) and ``gfx1100`` are **waitcheck: Yes, ConSan:
Yes** -- full supported-form coverage, as of the ``shared/rocjitsu/sanitizers``
update that graduated them from Partial/dash to full. There is no gfx950
simulator, so a dynamic ConSan run on that target needs real hardware.
Neither tool translates between ISAs; support is for native code objects only.

Nothing here assumes a GPU: argv/env builders are pure; the orchestrator
decides whether a check can actually execute.
"""

from __future__ import annotations

import os
import re
import shutil
from typing import Any

# --- binary / artifact resolution ----------------------------------------
# The build produces two artifacts (see the doc's Build section):
#   build/tools/rj_waitcheck                                  (static tool)
#   build/lib/rocjitsu/src/rocjitsu/hooks/librocjitsu_dbi_hooks.so  (hook)
# We resolve them from an explicit override, then $ROCJITSU_BUILD, then PATH.
ENV_ROCJITSU_BUILD = "ROCJITSU_BUILD"
ENV_WAITCHECK_BIN = "RJ_WAITCHECK_BIN"
ENV_SANITIZER_HOOK = "ROCJITSU_SANITIZER_HOOK"
DEFAULT_WAITCHECK_BIN = "rj_waitcheck"
_WAITCHECK_RELPATH = "tools/rj_waitcheck"
_HOOK_RELPATH = "lib/rocjitsu/src/rocjitsu/hooks/librocjitsu_dbi_hooks.so"

# ConSan analysis engines (RJ_CONSAN_MODE); record/replay is the default.
CONSAN_MODES = frozenset({"record-replay", "inline-shadow", "sampled", "supercollider"})


def resolve_waitcheck() -> str | None:
    """Absolute path to ``rj_waitcheck``, or ``None`` if not locatable."""
    override = os.environ.get(ENV_WAITCHECK_BIN)
    if override:
        return override if os.path.isabs(override) else (shutil.which(override) or override)
    build = os.environ.get(ENV_ROCJITSU_BUILD)
    if build:
        candidate = os.path.join(build, _WAITCHECK_RELPATH)
        if os.path.isfile(candidate):
            return candidate
    return shutil.which(DEFAULT_WAITCHECK_BIN)


def resolve_hook() -> str | None:
    """Absolute path to the combined DBI hook ``.so``, or ``None``."""
    override = os.environ.get(ENV_SANITIZER_HOOK)
    if override:
        return override
    build = os.environ.get(ENV_ROCJITSU_BUILD)
    if build:
        candidate = os.path.join(build, _HOOK_RELPATH)
        if os.path.isfile(candidate):
            return candidate
    return None


# --- support / validation policy -----------------------------------------
# Verbatim from the doc's target-support table. "Yes" == supported-form
# coverage (not every ISA memory op); "Partial" == native instrumentation
# for a subset of ConSan forms; a dash == waitcheck-only target (ConSan
# leaves the object uninstrumented).
_WAITCHECK_TARGETS = frozenset({
    "gfx942", "gfx950", "gfx1100", "gfx1150", "gfx1151", "gfx1200", "gfx1201", "gfx1250",
})
# ConSan "Yes" targets per the doc's table. gfx950 (MI350X/MI355X) and gfx1100
# graduated from Partial/dash to full on the shared/rocjitsu/sanitizers update.
_CONSAN_FULL = frozenset({"gfx942", "gfx950", "gfx1100", "gfx1201", "gfx1250"})
# No target is ConSan-"partial" in the current matrix; kept so re-adding one is
# a one-line change and the "partial" level below stays a documented outcome.
_CONSAN_PARTIAL: frozenset[str] = frozenset()


def support(check: str, target: str) -> dict[str, Any]:
    """Describe how well ``check`` is supported on ``target``.

    Returns ``{level, runnable, requires, note}``:
      * ``level``    -- ``full`` / ``partial`` / ``unsupported`` / ``unknown``
      * ``runnable`` -- whether we should attempt the check on this target
                        (static waitcheck is always attemptable; ConSan is
                        gated on the target being in the support table)
      * ``requires`` -- ``none`` (static) or ``hardware-or-simulator``
      * ``note``     -- caveat surfaced in the report
    """
    if check == "waitcheck":
        if target in _WAITCHECK_TARGETS:
            return _s("full", True, "none",
                      "waitcheck supported (native code objects; supported-form "
                      "coverage, not every ISA memory op)")
        return _s("unknown", True, "none",
                  f"{target} not in the rocjitsu support table; static waitcheck "
                  "may still run on a native code object")
    if check == "consan":
        if target in _CONSAN_FULL:
            return _s("full", True, "hardware-or-simulator",
                      "ConSan supported-form coverage on this target")
        if target in _CONSAN_PARTIAL:
            return _s("partial", True, "hardware-or-simulator",
                      f"{target} ConSan is PARTIAL: native instrumentation for "
                      "a subset of ConSan forms only")
        if target in _WAITCHECK_TARGETS:
            return _s("unsupported", False, "hardware-or-simulator",
                      f"{target} is a waitcheck-only target; ConSan leaves the "
                      "code object uninstrumented")
        return _s("unknown", False, "hardware-or-simulator",
                  f"{target} not in the rocjitsu support table")
    raise ValueError(f"unknown check {check!r}")


def _s(level: str, runnable: bool, requires: str, note: str) -> dict[str, Any]:
    return {"level": level, "runnable": runnable, "requires": requires, "note": note}


# --- invocation builders --------------------------------------------------

def waitcheck_argv(binary: str, artifact_path: str,
                   extra_args: list[str] | None = None) -> list[str]:
    """argv for a static ``rj_waitcheck`` run over one saved code object.

    ``extra_args`` passes through target-selection / kernel-filtering /
    corpus flags documented in the waitcheck guide without hard-coding them.
    """
    return [binary, *(extra_args or []), artifact_path]


def consan_env(hook_path: str, *, base_env: dict[str, str] | None = None,
               mode: str | None = None, policy: str | None = None,
               log: bool = False) -> dict[str, str]:
    """Environment for a dynamic ConSan run via the combined hook.

    Mirrors the doc's "Run waitcheck and ConSan together" recipe:
    ``HSA_TOOLS_DISABLE_REGISTER=1`` + ``HSA_TOOLS_LIB=<hook>``, plus the
    optional ``RJ_CONSAN_*`` knobs. Do NOT also set ``ROCJITSU_WAITCHECK*``
    -- the combined hook always runs an exhaustive load-time waitcheck.

    Note: the per-conflict ConSan diagnostic lines the guardrail keys on are
    only emitted when ``RJ_CONSAN_LOG=1`` (``log=True``); pass it when you want
    a race to fail the gate rather than pass silently.
    """
    if mode is not None and mode not in CONSAN_MODES:
        raise ValueError(f"invalid RJ_CONSAN_MODE {mode!r}; pick from {sorted(CONSAN_MODES)}")
    env = dict(base_env or {})
    env["HSA_TOOLS_DISABLE_REGISTER"] = "1"
    env["HSA_TOOLS_LIB"] = hook_path
    if mode is not None:
        env["RJ_CONSAN_MODE"] = mode
    if policy is not None:
        env["RJ_CONSAN_POLICY"] = policy
    if log:
        env["RJ_CONSAN_LOG"] = "1"
    return env


def consan_argv(command: list[str], *, simulator: str | None = None,
                config: str | None = None) -> list[str]:
    """argv for the instrumented application ConSan wraps.

    On real hardware this is just the application command. For a native-ISA
    simulator run (e.g. gfx1250), route it through the RocJITsu binary:
    ``rocjitsu --config <cfg>.json -- <app>``.
    """
    if simulator:
        return [simulator, *(["--config", config] if config else []), "--", *command]
    return list(command)


# --- output parsers -------------------------------------------------------

_WAITCHECK_PREFIX = "rocjitsu-waitcheck:"
# A missing-wait diagnostic in either output shape:
#   * combined hook (doc): "rocjitsu-waitcheck: .text+0x..: missing s_wait..."
#   * standalone rj_waitcheck: "<obj>:gfx950[0]:.text+0x..: missing s_waitcnt lgkmcnt(0) ..."
_WAITCHECK_MISS = re.compile(r"missing\s+s_wait|hazard", re.IGNORECASE)
# Producer/consumer context lines (hook uses "consumer:", tool uses "  producer ..").
_WAITCHECK_CTX = re.compile(r"^(producer|consumer)\b", re.IGNORECASE)
_CONSAN_PREFIX = "[rocjitsu-dbi-hooks] ConSan"
_KV_RE = re.compile(r"(\w+)=(\S+)")


def parse_waitcheck(stdout: str) -> dict[str, Any]:
    """Normalise ``rj_waitcheck`` output into ``{findings, counts}``.

    Handles both output shapes -- the combined hook's ``rocjitsu-waitcheck:``
    lines (doc) and the standalone ``rj_waitcheck`` tool's
    ``<obj>:<target>[i]:.text+0x..: missing s_waitcnt ..`` lines -- by keying
    on the ``missing s_wait`` / ``hazard`` text rather than a fixed prefix.
    The following indented ``producer``/``consumer`` context lines are
    attached to the finding. waitcheck diagnostics are non-fatal, so we
    classify them as ``warning`` (advisory) severity.
    """
    findings: list[dict[str, Any]] = []
    for raw in stdout.splitlines():
        line = raw.strip()
        if line.startswith(_WAITCHECK_PREFIX):
            line = line[len(_WAITCHECK_PREFIX):].strip()
        if _WAITCHECK_CTX.match(line):
            if findings:
                findings[-1].setdefault("context", []).append(line)
            continue
        if _WAITCHECK_MISS.search(line):
            findings.append({"severity": "warning", "message": line})
    return _summarise(findings)


def parse_consan(output: str) -> dict[str, Any]:
    """Normalise combined-hook ConSan output into ``{findings, counts, verdict}``.

    Keys ONLY on real conflict markers, verified against live gfx950 output:
      * a per-conflict ``... ConSan MOI auto replay diagnostic ...`` line
        (the doc's race emission, with ``first_kind`` / ``second_kind``);
      * an ``... auto replay ...`` summary line reporting ``conflict=true`` or
        a nonzero ``diagnostics=N``.
    Each becomes a ``race`` finding (-> fail). The ``analysis verdict`` line
    is captured as completeness metadata. Crucially, the many benign
    inventory / report / plan lines that merely contain the substring
    ``diagnostics=N`` (capacity, not a finding) are NOT counted.
    """
    findings: list[dict[str, Any]] = []
    verdict: dict[str, str] = {}
    for raw in output.splitlines():
        line = raw.strip()
        if _CONSAN_PREFIX not in line:
            continue
        low = line.lower()
        if "analysis verdict" in low:
            verdict = dict(_KV_RE.findall(line))
        elif "auto replay diagnostic" in low:
            findings.append({"severity": "race", "message": line,
                             **dict(_KV_RE.findall(line))})
        elif "auto replay" in low:
            kv = dict(_KV_RE.findall(line))
            try:
                ndiag = int(kv.get("diagnostics", "0"))
            except ValueError:
                ndiag = 0
            if kv.get("conflict", "false").lower() == "true" or ndiag > 0:
                findings.append({"severity": "race", "message": line, **kv})
    summary = _summarise(findings)
    summary["verdict"] = verdict
    return summary


def _summarise(findings: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for finding in findings:
        sev = finding.get("severity", "info")
        counts[sev] = counts.get(sev, 0) + 1
    return {"findings": findings, "counts": counts}


def consan_effective(verdict: dict[str, str]) -> dict[str, Any]:
    """Summarise what ConSan ACTUALLY achieved this run, straight from its
    ``analysis verdict`` line -- as opposed to the static per-target
    ``support()`` policy label. This is the underlying layer's own answer:
    even where the static table is conservative, a given kernel's run may
    still yield complete coverage. Returns ``{complete, analysis_complete,
    dynamic_complete, unsupported, incomplete_code_objects, access, note}``;
    ``complete`` is ``None`` when the hook emitted no verdict line.
    """
    if not verdict:
        return {"complete": None,
                "note": "no ConSan verdict line (hook produced no analysis)"}

    def _b(key: str) -> bool:
        return str(verdict.get(key, "")).lower() == "true"

    def _i(key: str) -> int:
        try:
            return int(verdict.get(key, "0"))
        except ValueError:
            return 0

    unsupported = (_i("replay_unsupported_access") + _i("replay_unsupported_atomics")
                   + _i("replay_unsupported_fences"))
    incomplete = _i("incomplete_code_objects")
    complete = (_b("analysis_complete") and _b("dynamic_complete")
                and incomplete == 0 and unsupported == 0)
    return {
        "complete": complete,
        "analysis_complete": _b("analysis_complete"),
        "dynamic_complete": _b("dynamic_complete"),
        "unsupported": unsupported,
        "incomplete_code_objects": incomplete,
        "access": verdict.get("access"),
        "note": ("ConSan achieved complete coverage on this run"
                 if complete else
                 "ConSan coverage was incomplete on this run "
                 "(some forms unsupported on this target)"),
    }


# Guardrail policy layered on top of the tools' own (non-fatal) diagnostics:
# a ConSan race fails the gate; an advisory missing-wait warns.
FAIL_SEVERITIES = frozenset({"error", "hazard", "race", "violation"})
WARN_SEVERITIES = frozenset({"warning"})


def verdict_for_counts(counts: dict[str, int]) -> str:
    """Reduce severity counts to ``pass`` / ``warn`` / ``fail``."""
    if any(counts.get(s, 0) for s in FAIL_SEVERITIES):
        return "fail"
    if any(counts.get(s, 0) for s in WARN_SEVERITIES):
        return "warn"
    return "pass"
