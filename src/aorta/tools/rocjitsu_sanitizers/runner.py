#!/usr/bin/env python3
"""Run the rocjitsu sanitizers (waitcheck + ConSan) over a kernel worklist.

Orchestrates the tool end to end for an AMDGPU target:

    top kernels ->  waitcheck  (static; rj_waitcheck on saved objects)  -\
                                                                          >-- verdict
                    ConSan     (dynamic; combined hook around a run)     -/

The worklist comes from ``select.py`` (Magpie kernel summary + hipBLASLt GEMM
CSV) or any producer that emits the ``rocjitsu_sanitizers.kernels/1`` schema.
waitcheck runs ``rj_waitcheck`` per-kernel over ISA artifacts in ``--isa-dir``;
ConSan loads the combined DBI hook (``librocjitsu_dbi_hooks.so``) via
``HSA_TOOLS_LIB`` around a real application (``--consan-command``) on hardware
or the RocJITsu simulator. Any check that cannot run (missing artifact,
unsupported target, no command, ``--dry-run``) is recorded *skipped* with a
reason -- a skip is "not checked", not "clean".

Standalone usage::

    python -m aorta.tools.rocjitsu_sanitizers.runner \
        --gemm-csv gemm_shapes_unique.csv \
        --isa-dir ./kernel_isa \
        --target gfx950 \
        --checks waitcheck,consan \
        --consan-command "python my_repro.py" \
        --output-dir ./sanitizer-out

Exit code is non-zero only when a runnable check returns a ``fail`` verdict;
``warn`` and all-skipped runs exit 0 (the JSON report has detail).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from . import backends
from .select import select_kernels

_ISA_SUFFIXES = (".s", ".isa", ".asm", ".hsaco", ".txt", ".co")
_VERDICT_RANK = {"pass": 0, "warn": 1, "fail": 2}


def _resolve_isa(kernel: dict[str, Any], isa_dir: Path | None) -> Path | None:
    """Best-effort map a worklist kernel to an ISA artifact in ``isa_dir``.

    Matches on the Tensile solution index (GEMM kernels) or a sanitised
    substring of the kernel name. Returns ``None`` when nothing matches --
    the caller records that as ``isa_not_found`` rather than guessing.
    """
    if isa_dir is None or not isa_dir.is_dir():
        return None
    candidates = [p for p in sorted(isa_dir.iterdir())
                  if p.suffix.lower() in _ISA_SUFFIXES]
    sol = kernel.get("top_solution_idx")
    if sol is not None:
        for path in candidates:
            if str(sol) in path.stem:
                return path
    token = "".join(c for c in kernel.get("name", "") if c.isalnum()).lower()
    for path in candidates:
        stem = "".join(c for c in path.stem if c.isalnum()).lower()
        if token and (token in stem or stem in token):
            return path
    return None


def _exec(argv: list[str], timeout: int,
          env: dict[str, str] | None = None) -> tuple[int | None, str, str]:
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, env=env)
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:
        out = e.stdout if isinstance(e.stdout, str) else ""
        err = (e.stderr if isinstance(e.stderr, str) else "") + "\n[TIMEOUT]"
        return None, out, err


def _skip(check: str, target: str, reason: str, **extra: Any) -> dict[str, Any]:
    return {"check": check, "target": target, "status": "skipped",
            "verdict": "skipped", "reason": reason,
            "support": backends.support(check, target), **extra}


def run_waitcheck(worklist: dict[str, Any], target: str, isa_dir: Path | None,
                  *, dry_run: bool, timeout: int,
                  extra_args: list[str] | None = None) -> dict[str, Any]:
    """Static per-kernel ``rj_waitcheck`` pass; never needs a GPU."""
    sup = backends.support("waitcheck", target)
    binary = backends.resolve_waitcheck()
    if binary is None:
        return _skip("waitcheck", target,
                     f"rj_waitcheck not found (set {backends.ENV_WAITCHECK_BIN} or "
                     f"{backends.ENV_ROCJITSU_BUILD})")

    per_kernel: list[dict[str, Any]] = []
    worst = "pass"
    for kernel in worklist["kernels"]:
        isa = _resolve_isa(kernel, isa_dir)
        if isa is None:
            per_kernel.append({"kernel": kernel["name"], "status": "skipped",
                               "reason": "isa_not_found"})
            continue
        if dry_run:
            per_kernel.append({"kernel": kernel["name"], "status": "skipped",
                               "reason": "dry_run", "isa": str(isa)})
            continue
        argv = backends.waitcheck_argv(binary, str(isa), extra_args)
        rc, out, err = _exec(argv, timeout)
        result = backends.parse_waitcheck(out + "\n" + err)
        verdict = backends.verdict_for_counts(result["counts"])
        worst = _worse(worst, verdict)
        per_kernel.append({"kernel": kernel["name"], "status": "ran",
                           "isa": str(isa), "returncode": rc,
                           "verdict": verdict, "counts": result["counts"],
                           "findings": result["findings"],
                           "stderr_tail": err[-500:]})

    ran = [k for k in per_kernel if k["status"] == "ran"]
    return {"check": "waitcheck", "target": target, "mode": "static",
            "status": "ran" if ran else "skipped",
            "verdict": worst if ran else "skipped",
            "support": sup, "kernels_checked": len(ran),
            "kernels_total": len(per_kernel), "results": per_kernel}


def run_consan(worklist: dict[str, Any], target: str, command: list[str] | None,
               *, dry_run: bool, timeout: int, base_env: dict[str, str] | None = None,
               mode: str | None = None, policy: str | None = None, log: bool = False,
               simulator: str | None = None, config: str | None = None) -> dict[str, Any]:
    """Dynamic ConSan pass via the combined hook; needs hardware or a simulator."""
    sup = backends.support("consan", target)
    total = worklist["kernel_count"]
    if not sup["runnable"]:
        return _skip("consan", target, sup["note"], kernels_total=total)
    hook = backends.resolve_hook()
    if hook is None:
        return _skip("consan", target,
                     f"DBI hook .so not found (set {backends.ENV_SANITIZER_HOOK} or "
                     f"{backends.ENV_ROCJITSU_BUILD})", kernels_total=total)
    if not command:
        return _skip("consan", target,
                     "no --consan-command given (ConSan wraps a real application run "
                     "via HSA_TOOLS_LIB)", kernels_total=total)
    if dry_run:
        return _skip("consan", target, "dry_run", kernels_total=total, command=command)

    env = backends.consan_env(hook, base_env=base_env, mode=mode, policy=policy, log=log)
    argv = backends.consan_argv(command, simulator=simulator, config=config)
    rc, out, err = _exec(argv, timeout, env=env)
    result = backends.parse_consan(out + "\n" + err)
    verdict = backends.verdict_for_counts(result["counts"])
    return {"check": "consan", "target": target, "mode": "dynamic",
            "status": "ran", "verdict": verdict, "support": sup,
            "returncode": rc, "command": command, "hook": hook,
            "consan_mode": mode or "record-replay", "policy": policy,
            "kernels_total": total, "counts": result["counts"],
            "consan_verdict": result.get("verdict", {}),
            # Live coverage the hook actually achieved this run (not the static
            # per-target policy label) -- this is "how the layer responded".
            "consan_effective": backends.consan_effective(result.get("verdict", {})),
            "findings": result["findings"], "stderr_tail": err[-500:]}


def _worse(a: str, b: str) -> str:
    return a if _VERDICT_RANK[a] >= _VERDICT_RANK[b] else b


def run_sanitizers(*, worklist: dict[str, Any], target: str, checks: list[str],
                   isa_dir: Path | None = None,
                   consan_command: list[str] | None = None,
                   dry_run: bool = False, timeout: int = 900,
                   waitcheck_args: list[str] | None = None,
                   base_env: dict[str, str] | None = None,
                   consan_mode: str | None = None, consan_policy: str | None = None,
                   consan_log: bool = False, simulator: str | None = None,
                   simulator_config: str | None = None) -> dict[str, Any]:
    """Run the requested checks and fold them into one guardrail report."""
    check_reports: list[dict[str, Any]] = []
    if "waitcheck" in checks:
        check_reports.append(run_waitcheck(worklist, target, isa_dir,
                                           dry_run=dry_run, timeout=timeout,
                                           extra_args=waitcheck_args))
    if "consan" in checks:
        check_reports.append(run_consan(worklist, target, consan_command,
                                        dry_run=dry_run, timeout=timeout,
                                        base_env=base_env, mode=consan_mode,
                                        policy=consan_policy, log=consan_log,
                                        simulator=simulator, config=simulator_config))

    ran = [c["verdict"] for c in check_reports if c["status"] == "ran"]
    overall = "not_checked" if not ran else max(ran, key=lambda v: _VERDICT_RANK[v])
    return {
        "schema": "rocjitsu_sanitizers.report/1",
        "target": target,
        "checks_requested": checks,
        "overall_verdict": overall,
        "kernel_count": worklist["kernel_count"],
        "kernel_sources": worklist["sources"],
        "checks": check_reports,
    }


def _print_summary(report: dict[str, Any]) -> None:
    print(f"target           : {report['target']}")
    print(f"kernels          : {report['kernel_count']} "
          f"from {', '.join(report['kernel_sources'])}")
    for check in report["checks"]:
        print(f"  {check['check']:<10} [{check.get('mode', '-')}]"
              f" status={check['status']} verdict={check['verdict']}")
        eff = check.get("consan_effective")
        if check["status"] == "ran" and eff and eff.get("complete") is not None:
            # A check that ACTUALLY ran: lead with the live rocjitsu verdict;
            # the static per-target policy label is only a parenthetical.
            print(f"      live (rocjitsu): analysis_complete={eff['analysis_complete']} "
                  f"dynamic_complete={eff['dynamic_complete']} access={eff.get('access')} "
                  f"unsupported={eff['unsupported']} -> "
                  f"{'complete' if eff['complete'] else 'incomplete'} for this run "
                  f"(target policy: {check['support']['level']})")
        else:
            print(f"      support: {check['support']['level']} -- {check['support']['note']}")
        if check.get("reason"):
            print(f"      skipped: {check['reason']}")
    print(f"OVERALL VERDICT  : {report['overall_verdict'].upper()}")


def main(argv: list[str] | None = None) -> int:
    import os

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernels", type=Path, default=None,
                        help="pre-built worklist JSON (rocjitsu_sanitizers.kernels/1)")
    parser.add_argument("--magpie-report", type=Path, default=None)
    parser.add_argument("--gemm-csv", type=Path, default=None)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--isa-dir", type=Path, default=None,
                        help="directory of kernel ISA / .s / .hsaco / .co artifacts")
    parser.add_argument("--target", default="gfx950",
                        help="GPU target (gfx950 = MI350X/MI355X)")
    parser.add_argument("--checks", default="waitcheck,consan",
                        help="comma-separated subset of waitcheck,consan")
    parser.add_argument("--consan-command", default=None,
                        help="application command the ConSan hook wraps")
    parser.add_argument("--consan-mode", default=None,
                        help=f"RJ_CONSAN_MODE ({', '.join(sorted(backends.CONSAN_MODES))})")
    parser.add_argument("--consan-policy", default=None,
                        help="RJ_CONSAN_POLICY (e.g. 'strict')")
    parser.add_argument("--consan-log", action="store_true", help="RJ_CONSAN_LOG=1")
    parser.add_argument("--simulator", default=None,
                        help="RocJITsu simulator binary (native-ISA run, e.g. gfx1250)")
    parser.add_argument("--simulator-config", default=None,
                        help="RocJITsu --config JSON for the simulator")
    parser.add_argument("--dry-run", action="store_true",
                        help="plan only: resolve kernels/backends, run nothing")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.kernels is not None:
        worklist = json.loads(args.kernels.read_text())
    else:
        worklist = select_kernels(magpie_report=args.magpie_report,
                                  gemm_csv=args.gemm_csv, top_n=args.top_n)

    checks = [c.strip() for c in args.checks.split(",") if c.strip()]
    unknown = set(checks) - {"waitcheck", "consan"}
    if unknown:
        parser.error(f"unknown checks: {sorted(unknown)}")

    consan_command = args.consan_command.split() if args.consan_command else None
    report = run_sanitizers(worklist=worklist, target=args.target, checks=checks,
                            isa_dir=args.isa_dir, consan_command=consan_command,
                            dry_run=args.dry_run, timeout=args.timeout,
                            base_env=dict(os.environ), consan_mode=args.consan_mode,
                            consan_policy=args.consan_policy, consan_log=args.consan_log,
                            simulator=args.simulator, simulator_config=args.simulator_config)

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        out = args.output_dir / "sanitizer_report.json"
        out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote report -> {out}")
    _print_summary(report)
    return 1 if report["overall_verdict"] == "fail" else 0


if __name__ == "__main__":
    sys.exit(main())
