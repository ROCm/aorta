#!/usr/bin/env python3
"""Structural + consistency validator for aorta run outputs.

Detects the artifact type of any path produced by an aorta command
(``env probe``, ``probe``, ``triage run``, ``run``, or a single trial /
matrix file) and runs schema + internal-consistency checks against it.

Stdlib only. PyYAML is used opportunistically for ``*.yaml`` recipes and
is skipped gracefully when unavailable.

Design notes:
* Workload / mitigation / diagnostic / environment NAMES are never
  validated against a fixed registry. Names supplied by entry-point
  plugin packages or ``--mitigations-file`` sidecars are first-class and
  must not be flagged as unknown. This keeps the validator correct for
  externally-provided workloads, not just the public built-ins.
* Findings have three levels: ``ok`` (silent unless --verbose),
  ``warn`` (suspicious but not provably wrong), ``fail`` (contract
  violation). Exit code is 0 when no ``fail`` finding fired, else 1.

Usage:
    python verify_run.py PATH [PATH ...] [--verbose] [--json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

try:  # optional; only needed for recipe.resolved.yaml deep checks
    import yaml  # type: ignore
except Exception:  # pragma: no cover - environment dependent
    yaml = None  # type: ignore

# Recognised enums from the public schemas.
_TRIAL_EXIT_STATUS = {
    "ok",
    "workload_failed",
    "workload_setup_failed",
    "infrastructure_failed",
}
_PROBE_VERDICTS = {"pass", "fail", "error"}
# Top-level env.json keys that should always be present (a subset chosen
# to be stable across schema 1.x; absence is a real problem, not drift).
_ENV_REQUIRED_KEYS = {
    "schema_version",
    "captured_at",
    "partial",
    "partial_reasons",
    "rocm",
    "hip",
    "env_vars",
    "python_version",
}


class Report:
    """Accumulates findings for one or more artifacts."""

    def __init__(self) -> None:
        self.findings: list[dict[str, str]] = []

    def add(self, level: str, target: str, msg: str) -> None:
        self.findings.append({"level": level, "target": target, "msg": msg})

    def ok(self, target: str, msg: str) -> None:
        self.add("ok", target, msg)

    def warn(self, target: str, msg: str) -> None:
        self.add("warn", target, msg)

    def fail(self, target: str, msg: str) -> None:
        self.add("fail", target, msg)

    @property
    def failed(self) -> bool:
        return any(f["level"] == "fail" for f in self.findings)

    def counts(self) -> dict[str, int]:
        out = {"ok": 0, "warn": 0, "fail": 0}
        for f in self.findings:
            out[f["level"]] = out.get(f["level"], 0) + 1
        return out


# --------------------------------------------------------------------------
# Loading helpers
# --------------------------------------------------------------------------


def _load_json(path: Path, rep: Report) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        rep.fail(str(path), "file does not exist")
    except json.JSONDecodeError as exc:
        rep.fail(str(path), f"invalid JSON: {exc}")
    except OSError as exc:
        rep.fail(str(path), f"unreadable: {exc}")
    return None


# --------------------------------------------------------------------------
# Artifact-type detection
# --------------------------------------------------------------------------


def classify_json(doc: Any) -> str:
    """Return a best-effort artifact kind for a parsed JSON document."""
    if not isinstance(doc, dict):
        return "unknown"
    keys = doc.keys()
    if "verdict" in keys and "failure_detectors_fired" in keys:
        return "probe_result"
    if "trial_id" in keys and "exit_status" in keys and "result" in keys:
        return "trial_result"
    if "cells" in keys and "baseline_cell" in keys:
        return "matrix_json"
    if "partial" in keys and ("rocm" in keys or "captured_at" in keys):
        return "env_snapshot"
    return "unknown"


# --------------------------------------------------------------------------
# Per-artifact checks
# --------------------------------------------------------------------------


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def check_probe_result(doc: dict, target: str, rep: Report) -> None:
    """Validate a probe Phase-2 ``result.json`` (one trial)."""
    verdict = doc.get("verdict")
    if verdict not in _PROBE_VERDICTS:
        rep.fail(target, f"verdict={verdict!r} not in {sorted(_PROBE_VERDICTS)}")
    fired = doc.get("failure_detectors_fired")
    if not isinstance(fired, list):
        rep.fail(target, "failure_detectors_fired must be a list")
        fired = []
    # Issue #230: infra-error signals live in their own list. Optional for
    # legacy result.json that predates the three-way verdict.
    errored = doc.get("error_detectors_fired")
    if errored is None:
        errored = []
    elif not isinstance(errored, list):
        rep.warn(target, "error_detectors_fired present but not a list")
        errored = []
    warned = doc.get("warn_detectors_fired")
    if warned is not None and not isinstance(warned, list):
        rep.warn(target, "warn_detectors_fired present but not a list")

    # Verdict precedence (fail > error > pass, issue #230):
    # - any failure detector  => verdict must be ``fail``
    # - else any error detector => verdict must be ``error``
    # - else                    => verdict must be ``pass``
    if fired and verdict != "fail":
        rep.fail(
            target,
            f"verdict={verdict!r} but failure_detectors_fired is non-empty "
            f"({fired}); failures outrank everything -> expected fail",
        )
    if not fired and errored and verdict != "error":
        rep.fail(
            target,
            f"verdict={verdict!r} but only error detectors fired ({errored}); "
            "expected error",
        )
    if not fired and not errored and verdict not in ("pass",):
        # verdict=fail/error with nothing fired. fail keeps its legacy warn
        # (meta:missing_pass_signal may have been the cause and dropped);
        # error with no error detector is a contradiction.
        if verdict == "fail":
            rep.warn(
                target,
                "verdict=fail with empty failure_detectors_fired "
                "(expected at least one detector or meta:missing_pass_signal)",
            )
        elif verdict == "error":
            rep.fail(
                target,
                "verdict=error with empty error_detectors_fired "
                "(an error verdict must name its infra-error reason)",
            )

    # timeout consistency. ``tier1:timeout`` is an *error* detector, so look
    # in both lists (a recognised hang co-fires tier2:hang -> the timeout
    # may sit in error_detectors_fired even on a fail verdict).
    timed_out = doc.get("timed_out")
    has_timeout_detector = any(
        isinstance(d, str) and d.endswith("timeout") for d in (*fired, *errored)
    )
    if timed_out is True and not has_timeout_detector:
        rep.warn(target, "timed_out=true but no tier1:timeout detector fired")
    if has_timeout_detector and timed_out is False:
        rep.warn(target, "tier1:timeout fired but timed_out=false")

    # Field shapes.
    if not _is_number(doc.get("walltime_sec")):
        rep.warn(target, "walltime_sec missing or non-numeric")
    pv = doc.get("peak_vram_mib", "absent")
    if pv != "absent" and pv is not None and not _is_number(pv):
        rep.warn(target, "peak_vram_mib should be an int or null")
    if not isinstance(doc.get("argv"), list):
        rep.warn(target, "argv missing or not a list")
    if "exit_code" in doc and not isinstance(doc["exit_code"], int):
        rep.warn(target, "exit_code present but not an int")
    rep.ok(target, f"probe result.json verdict={verdict}")


def check_trial_result(doc: dict, target: str, rep: Report) -> None:
    """Validate a ``TrialResult`` JSON (aorta run / triage per-trial)."""
    exit_status = doc.get("exit_status")
    if exit_status not in _TRIAL_EXIT_STATUS:
        rep.warn(
            target,
            f"exit_status={exit_status!r} not in known set "
            f"{sorted(_TRIAL_EXIT_STATUS)} (newer schema or custom producer?)",
        )
    for key in ("trial_id", "workload", "result"):
        if key not in doc:
            rep.fail(target, f"missing required key {key!r}")
    result = doc.get("result")
    if not isinstance(result, dict):
        rep.fail(target, "result must be a WorkloadResult dict")
        return
    if "passed" not in result:
        rep.warn(target, "result.passed missing")
    passed = result.get("passed")
    # Consistency: exit_status==ok implies passed is True.
    if exit_status == "ok" and passed is False:
        rep.fail(target, "exit_status=ok but result.passed=false")
    if exit_status == "workload_failed" and passed is True:
        rep.warn(target, "exit_status=workload_failed but result.passed=true")
    # Semantic plausibility: a pass that never reached the workload's main
    # work tested nothing meaningful.
    if passed is True and (
        result.get("main_work_started") is False
        or result.get("executed_iterations") == 0
    ):
        rep.warn(
            target,
            "result.passed=true but the workload never started its main work "
            "(main_work_started=false / executed_iterations=0) -- the pass may "
            "not have tested anything",
        )
    # env snapshot should be embedded.
    env = doc.get("env")
    if not isinstance(env, dict):
        rep.warn(target, "env (embedded snapshot) missing or not a dict")
    elif env.get("partial") is True and not env.get("partial_reasons"):
        rep.warn(target, "embedded env.partial=true but partial_reasons empty")
    rep.ok(target, f"trial {doc.get('trial_id')} exit_status={exit_status}")


def check_env_snapshot(doc: dict, target: str, rep: Report) -> None:
    """Validate an ``env.json`` snapshot from ``collect_env`` / ``env probe``."""
    missing = sorted(_ENV_REQUIRED_KEYS - doc.keys())
    if missing:
        rep.fail(target, f"env snapshot missing keys: {missing}")
    sv = doc.get("schema_version")
    if not isinstance(sv, str):
        rep.warn(target, f"schema_version should be a string, got {sv!r}")
    partial = doc.get("partial")
    if not isinstance(partial, bool):
        rep.warn(target, "partial should be a bool")
    reasons = doc.get("partial_reasons")
    if not isinstance(reasons, list):
        rep.warn(target, "partial_reasons should be a list")
        reasons = []
    if partial is True and not reasons:
        rep.warn(target, "partial=true but partial_reasons is empty")
    if partial is False and reasons:
        rep.warn(target, "partial=false but partial_reasons is non-empty")
    rep.ok(
        target,
        f"env snapshot schema={sv} partial={partial} "
        f"({len(reasons)} reason(s))",
    )


def check_matrix_json(doc: dict, target: str, rep: Report) -> None:
    """Validate a triage/probe ``matrix.json`` and its per-cell aggregates."""
    cells = doc.get("cells")
    if not isinstance(cells, list):
        rep.fail(target, "cells must be a list")
        return
    baseline = doc.get("baseline_cell")
    names = {c.get("name") for c in cells if isinstance(c, dict)}
    if baseline is not None and baseline not in names:
        rep.warn(target, f"baseline_cell={baseline!r} not among cell names")
    for cell in cells:
        if not isinstance(cell, dict):
            rep.fail(target, "cell entry is not a dict")
            continue
        name = cell.get("name", "<unnamed>")
        ct = f"{target}::{name}"
        trials = cell.get("trials")
        passed = cell.get("passed_count")
        failed = cell.get("failed_count")
        # Issue #230: error trials are a third bucket. Optional/0 for legacy
        # matrix.json that predates the three-way verdict.
        errored = cell.get("error_count", 0)
        if not isinstance(errored, int):
            errored = 0
        err = cell.get("error")
        if err is None and all(isinstance(x, int) for x in (trials, passed, failed)):
            if passed + failed + errored != trials:
                rep.fail(
                    ct,
                    f"passed_count({passed})+failed_count({failed})+"
                    f"error_count({errored}) != trials({trials})",
                )
            rate = cell.get("failure_rate")
            if _is_number(rate):
                # Event rate excludes error trials from the denominator.
                valid = passed + failed
                expect = (failed / valid) if valid else 0.0
                if abs(rate - expect) > 1e-6:
                    rep.fail(
                        ct,
                        f"failure_rate={rate} != failed/valid_trials={expect:.6f} "
                        f"(valid = passed+failed = {valid}; errors excluded)",
                    )
            erate = cell.get("error_rate")
            if _is_number(erate) and trials:
                expect_e = errored / trials
                if abs(erate - expect_e) > 1e-6:
                    rep.fail(
                        ct,
                        f"error_rate={erate} != error/trials={expect_e:.6f}",
                    )
            hist = cell.get("exit_status_counts")
            if isinstance(hist, dict) and hist:
                total = sum(v for v in hist.values() if isinstance(v, int))
                if total != trials:
                    rep.warn(
                        ct,
                        f"exit_status_counts sums to {total}, trials={trials}",
                    )
        # trial_paths existence is checked by the directory walker, not here.
    rep.ok(target, f"matrix.json: {len(cells)} cell(s), baseline={baseline}")


# --------------------------------------------------------------------------
# Directory walkers
# --------------------------------------------------------------------------


def _iter_probe_trial_results(root: Path) -> Iterable[Path]:
    # Flat-resume layout: <ticket>/<cell>/trial_<n>/result.json
    yield from sorted(root.glob("*/trial_*/result.json"))


def _iter_triage_trial_results(root: Path) -> Iterable[Path]:
    # Triage/run layout: cells/<cell>/<workload>/trial_*.json
    #                or  <workload>/trial_*.json (bare `aorta run`)
    yield from sorted(root.glob("cells/*/*/trial_*.json"))
    yield from sorted(root.glob("*/trial_d*_m*_t*.json"))


def check_run_directory(root: Path, rep: Report) -> None:
    """Detect and validate a matrix/run output tree under ``root``."""
    matrix_json = root / "matrix.json"
    found_any = False

    if matrix_json.exists():
        found_any = True
        doc = _load_json(matrix_json, rep)
        if isinstance(doc, dict):
            check_matrix_json(doc, str(matrix_json), rep)
            _check_matrix_md_agreement(root, doc, rep)
            _check_trial_completeness(root, doc, rep)

    # Walk every trial artifact we can find, regardless of layout.
    probe_trials = list(_iter_probe_trial_results(root))
    triage_trials = list(_iter_triage_trial_results(root))
    for p in probe_trials:
        found_any = True
        doc = _load_json(p, rep)
        if isinstance(doc, dict):
            kind = classify_json(doc)
            if kind == "probe_result":
                check_probe_result(doc, str(p), rep)
            elif kind == "trial_result":
                check_trial_result(doc, str(p), rep)
            else:
                rep.warn(str(p), f"unrecognised trial artifact ({kind})")
    for p in triage_trials:
        found_any = True
        doc = _load_json(p, rep)
        if isinstance(doc, dict):
            kind = classify_json(doc)
            if kind == "trial_result":
                check_trial_result(doc, str(p), rep)
            elif kind == "probe_result":
                check_probe_result(doc, str(p), rep)
            else:
                rep.warn(str(p), f"unrecognised trial artifact ({kind})")

    # Sibling env snapshots (host_env.json, environments/<env>/env.json).
    for env_path in sorted(root.glob("host_env.json")) + sorted(
        root.glob("environments/*/env.json")
    ):
        found_any = True
        doc = _load_json(env_path, rep)
        if isinstance(doc, dict):
            check_env_snapshot(doc, str(env_path), rep)

    if not found_any:
        rep.warn(
            str(root),
            "no recognised aorta artifacts found (no matrix.json, trial JSONs, "
            "or env snapshots)",
        )


def _check_matrix_md_agreement(root: Path, mjson: dict, rep: Report) -> None:
    md = root / "matrix.md"
    if not md.exists():
        rep.warn(str(md), "matrix.json present but matrix.md missing")
        return
    text = md.read_text(encoding="utf-8", errors="replace")
    wl = mjson.get("workload")
    if isinstance(wl, str) and wl and wl not in text:
        rep.warn(str(md), f"workload {wl!r} from matrix.json not found in matrix.md")
    rep.ok(str(md), "matrix.md present")


def _check_trial_completeness(root: Path, mjson: dict, rep: Report) -> None:
    """Warn when a cell's on-disk trial count differs from its trials field."""
    cells = mjson.get("cells")
    if not isinstance(cells, list):
        return
    for cell in cells:
        if not isinstance(cell, dict) or cell.get("error") is not None:
            continue
        paths = cell.get("trial_paths")
        if not isinstance(paths, list):
            continue
        missing = 0
        for p in paths:
            if not isinstance(p, str):
                continue
            cand = Path(p)
            if not cand.is_absolute():
                cand = root / p
            if not cand.exists():
                missing += 1
        if missing:
            rep.warn(
                f"{root}::{cell.get('name')}",
                f"{missing} of {len(paths)} trial_paths not found on disk",
            )


# --------------------------------------------------------------------------
# Top-level dispatch
# --------------------------------------------------------------------------


def verify_path(path: Path, rep: Report) -> None:
    if path.is_dir():
        check_run_directory(path, rep)
        return
    if path.suffix == ".json":
        doc = _load_json(path, rep)
        if not isinstance(doc, dict):
            return
        kind = classify_json(doc)
        dispatch = {
            "probe_result": check_probe_result,
            "trial_result": check_trial_result,
            "env_snapshot": check_env_snapshot,
            "matrix_json": check_matrix_json,
        }
        if kind in dispatch:
            dispatch[kind](doc, str(path), rep)
        else:
            rep.warn(
                str(path),
                "unrecognised JSON artifact; applying no schema checks. "
                "Keys: " + ", ".join(sorted(doc.keys())[:12]),
            )
        return
    if path.suffix in (".yaml", ".yml"):
        _check_recipe_yaml(path, rep)
        return
    rep.warn(str(path), f"unsupported artifact extension {path.suffix!r}")


def _check_recipe_yaml(path: Path, rep: Report) -> None:
    if yaml is None:
        rep.warn(str(path), "PyYAML not available; skipped recipe checks")
        return
    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - surface any parse failure
        rep.fail(str(path), f"invalid YAML: {exc}")
        return
    if not isinstance(doc, dict):
        rep.fail(str(path), "recipe is not a mapping")
        return
    if "schema_version" not in doc:
        rep.warn(str(path), "recipe missing schema_version")
    mode = doc.get("mode")
    if mode == "probe":
        if not doc.get("mitigation_axis") and not doc.get("diagnostic_axis"):
            rep.warn(str(path), "probe recipe has neither mitigation_axis nor diagnostic_axis")
    else:
        if "workload" not in doc:
            rep.warn(str(path), "triage recipe missing workload")
    rep.ok(str(path), f"recipe parsed (mode={mode or 'triage'})")


def _print_report(rep: Report, verbose: bool, as_json: bool) -> None:
    if as_json:
        print(json.dumps({"findings": rep.findings, "counts": rep.counts()}, indent=2))
        return
    symbols = {"ok": "  ok ", "warn": "WARN ", "fail": "FAIL "}
    for f in rep.findings:
        if f["level"] == "ok" and not verbose:
            continue
        print(f"{symbols[f['level']]}| {f['target']}\n        {f['msg']}")
    c = rep.counts()
    verdict = "FAIL" if rep.failed else ("WARN" if c["warn"] else "PASS")
    print(
        f"\nVerdict: {verdict}  "
        f"(fail={c['fail']} warn={c['warn']} ok={c['ok']})"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="artifact files or run directories")
    parser.add_argument("-v", "--verbose", action="store_true", help="show ok findings")
    parser.add_argument("--json", action="store_true", help="emit findings as JSON")
    args = parser.parse_args(argv)

    rep = Report()
    for raw in args.paths:
        p = Path(raw)
        if not p.exists():
            rep.fail(raw, "path does not exist")
            continue
        verify_path(p, rep)

    _print_report(rep, args.verbose, args.json)
    return 1 if rep.failed else 0


if __name__ == "__main__":
    sys.exit(main())
