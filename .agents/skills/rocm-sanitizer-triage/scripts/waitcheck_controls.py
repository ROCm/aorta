#!/usr/bin/env python3
"""Draft gfx1250 control examples: prepare data, run AORTA, verify saved evidence.

Requires Python 3.10+. Run/verify use the same Python environment as AORTA.
Never builds or executes the compiled control programs. Authorization to execute
on a cluster must be obtained separately, including for read-only commands.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import platform
import re
import shutil
import signal
import subprocess
import sys
from datetime import datetime, timezone

LABELS = ("missing", "corrected", "mismatch")
TARGET = "gfx1250"
UPSTREAM = "d03c6fe410eed62b211d56bb7878dd1f494269fe"
ASSETS = Path(__file__).resolve().parent.parent / "assets" / "waitcheck-controls"
LOG_LIMIT = 1024 * 1024


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fresh(path):
    require(not path.exists() and not path.is_symlink(), f"output already exists: {path}")
    path = path.resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def contained(root, relative):
    path = root
    require(not root.is_symlink(), "evidence root must not be a symlink")
    for part in Path(relative).parts:
        require(part not in ("..", "/", "\\"), "invalid relative evidence path")
        path = path / part
        require(not path.is_symlink(), f"symlink evidence refused: {path}")
    require(path.resolve().is_relative_to(root.resolve()), "evidence path escaped root")
    require(path.is_file(), f"missing evidence: {path}")
    return path


def text(path):
    require(path.stat().st_size <= LOG_LIMIT, f"text evidence exceeds {LOG_LIMIT} bytes: {path}")
    with path.open("rb") as stream:
        raw = stream.read(LOG_LIMIT + 1)
    require(len(raw) <= LOG_LIMIT, "text evidence changed or exceeded limit")
    return raw.decode("utf-8-sig")


def load(path):
    return json.loads(text(path))


def save(path, value):
    # Used only inside a directory created exclusively by this invocation.
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def source_versions():
    return {"upstream_revision": UPSTREAM, "helper_sha256": digest(Path(__file__)),
            "assets": {name: digest(ASSETS / name) for name in
                       ("missing_wait.hip", "corrected_wait.hip", "LICENSE.upstream.txt")}}


def recipe(label, prepared):
    control = prepared["controls"]["corrected" if label == "mismatch" else label]
    return {"schema_version": 1, "mode": "sanitizer", "ticket": f"WAITCHECK-CONTROL-{label.upper()}",
            "description": "Draft offline gfx1250 control qualification; not a general kernel test.",
            "sanitizer_plan": {
                "target": TARGET,
                "source": {"kind": "kernel", "kernel": {
                    "name": prepared["kernel_name"], "code_object": control["snapshot"],
                    "code_object_sha256": "0" * 64 if label == "mismatch" else control["sha256"],
                    "code_object_index": control["index"], "entry_offset": control["entry"]}},
                "scope": {"kind": "kernel"}, "selection": {"requirement": "top_dispatch_count", "top_n": 1},
                "sanitizers": ["waitcheck"],
                "policy": {"consan_policy": "strict", "on_missing_backend": "fail",
                           "timeout_seconds": 120, "waitcheck_max_diagnostics": 64},
                "output": {"report": "sanitizer_report.json"}}}


def prepare(args):
    root = fresh(args.output)
    manifest = {"schema": "waitcheck-controls-prepared/1", "status": "preparing", "target": TARGET,
                "kernel_name": args.kernel_name, "build_id": args.build_id,
                "source_versions": source_versions(), "controls": {}}
    try:
        require(args.kernel_name.strip() and args.build_id.strip(), "kernel name and build identity must be nonempty")
        (root / "inputs").mkdir()
        for label in ("missing", "corrected"):
            source = getattr(args, f"{label}_object").resolve(strict=True)
            require(source.is_file() and source.stat().st_size > 0, "control must be a nonempty regular file")
            destination = root / "inputs" / f"{label}.bin"
            before = digest(source)
            with source.open("rb") as src, destination.open("xb") as dst:
                shutil.copyfileobj(src, dst)
            require(digest(destination) == before == digest(source), "input changed while preparing")
            manifest["controls"][label] = {"source": str(source), "snapshot": f"inputs/{label}.bin",
                                            "sha256": before, "index": getattr(args, f"{label}_index"),
                                            "entry": getattr(args, f"{label}_entry")}
        require(manifest["controls"]["missing"]["sha256"] != manifest["controls"]["corrected"]["sha256"],
                "missing and corrected controls must contain different bytes")
        for label in LABELS:
            save(root / f"{label}.json", recipe(label, manifest))
        manifest["status"] = "prepared"
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        raise
    finally:
        save(root / "prepared.json", manifest)
    return f"Prepared three draft recipes in {root}. No programs executed."


def check_prepared(root):
    data = load(contained(root, "prepared.json"))
    require(data["schema"] == "waitcheck-controls-prepared/1" and data["status"] == "prepared"
            and data["target"] == TARGET, "invalid or incomplete prepared controls")
    require(set(data["controls"]) == {"missing", "corrected"}, "expected two controls")
    for label, control in data["controls"].items():
        require(control["snapshot"] == f"inputs/{label}.bin", "unexpected snapshot path")
        require(type(control["index"]) is int and control["index"] >= 0
                and type(control["entry"]) is int and control["entry"] >= 0, "invalid explicit selectors")
        require(digest(contained(root, control["snapshot"])) == control["sha256"], "prepared input digest mismatch")
    require(data["controls"]["missing"]["sha256"] != data["controls"]["corrected"]["sha256"], "controls identical")
    for label in LABELS:
        require(load(contained(root, f"{label}.json")) == recipe(label, data), f"{label}: recipe changed")
    return data


def tool(path):
    require(path.is_absolute() and path.is_file() and os.access(path, os.X_OK), f"executable unavailable: {path}")
    return {"path": str(path), "resolved_path": str(path.resolve()), "sha256": digest(path)}


def recorded_path(value):
    """Interpret saved remote paths without applying the verifier host's syntax."""
    require(isinstance(value, str), "recorded path must be a string")
    path = PureWindowsPath(value) if PureWindowsPath(value).drive else PurePosixPath(value)
    require(path.is_absolute() and ".." not in path.parts, "recorded path must be absolute without parent traversal")
    return path


def capture(argv, root, label, env):
    row = {"argv": argv, "cwd": str(root), "timeout_seconds": 180, "returncode": None,
           "timed_out": False, "interrupted": False, "stdout": f"{label}.stdout.txt", "stderr": f"{label}.stderr.txt"}
    try:
        with (root / row["stdout"]).open("xb") as stdout, (root / row["stderr"]).open("xb") as stderr:
            process = subprocess.Popen(argv, cwd=root, env=env, stdin=subprocess.DEVNULL,
                                       stdout=stdout, stderr=stderr, shell=False,
                                       start_new_session=os.name == "posix")
            try:
                row["returncode"] = process.wait(timeout=180)
            except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
                row["timed_out"] = isinstance(exc, subprocess.TimeoutExpired)
                row["interrupted"] = isinstance(exc, KeyboardInterrupt)
                if os.name == "posix":
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                else:
                    process.kill()
                row["returncode"] = process.wait(timeout=10)
    except OSError as exc:
        row["launch_error"] = str(exc)
    return row


def run(args):
    prepared_root = args.prepared.resolve(strict=True)
    require(not args.output.resolve().is_relative_to(prepared_root), "run output must be outside prepared inputs")
    prepared_data = check_prepared(prepared_root)
    aorta = tool(args.aorta)
    require(args.prebuilt.is_absolute(), "prebuilt bundle root must be absolute")
    prebuilt = args.prebuilt.resolve(strict=True)
    backend = tool(prebuilt / "bin" / "rj_waitcheck")
    require(backend["sha256"] == args.tool_sha256, "Waitcheck tool hash differs from expected")
    input_hashes = {control["sha256"] for control in prepared_data["controls"].values()}
    require(aorta["sha256"] not in input_hashes and backend["sha256"] not in input_hashes,
            "AORTA/scanner executable must not be a control input or a copy of one")
    require(aorta["sha256"] != backend["sha256"], "AORTA and scanner must be distinct executables")
    from aorta.instrumentation.rocjitsu_sanitizers import report as reader
    package_version = importlib.metadata.version("amd-aorta")
    root = fresh(args.output)
    manifest = {"schema": "waitcheck-controls-run/1", "status": "running", "commands": {},
                "started_utc": datetime.now(timezone.utc).isoformat(), "hostname": platform.node(),
                "python": sys.version, "platform": platform.platform(), "aorta_python_package": package_version,
                "aorta_reader_path": str(Path(reader.__file__).resolve()), "source_versions": source_versions(),
                "aorta_before": aorta, "backend_before": backend,
                "environment_override": {"ROCJITSU_PREBUILT": str(prebuilt)}, "identities": {}}
    save(root / "run.json", manifest)
    try:
        snapshot = root / "snapshot"
        (snapshot / "inputs").mkdir(parents=True)
        for relative in ("prepared.json", *(f"{label}.json" for label in LABELS), "inputs/missing.bin", "inputs/corrected.bin"):
            with contained(prepared_root, relative).open("rb") as src, (snapshot / relative).open("xb") as dst:
                shutil.copyfileobj(src, dst)
        data = check_prepared(snapshot)
        for label in LABELS:
            identity = dict(recipe(label, data)["sanitizer_plan"]["source"]["kernel"])
            identity["code_object"] = str((snapshot / identity["code_object"]).resolve())
            identity["target"] = TARGET
            manifest["identities"][label] = identity
        manifest["input_hashes_before"] = {label: value["sha256"] for label, value in data["controls"].items()}
        env = dict(os.environ, **manifest["environment_override"])
        for label in LABELS:
            argv = [str(args.aorta), "sweep", "run", "--recipe", str(snapshot / f"{label}.json"),
                    "--output", str(root / label)]
            manifest["commands"][label] = capture(argv, root, label, env)
            save(root / "run.json", manifest)
            row = manifest["commands"][label]
            require(not row.get("launch_error") and not row["timed_out"] and not row["interrupted"],
                    f"{label}: launch failed, interrupted or timed out; evidence retained")
        manifest["aorta_after"] = tool(args.aorta)
        manifest["backend_after"] = tool(prebuilt / "bin" / "rj_waitcheck")
        manifest["input_hashes_after"] = {label: digest(contained(snapshot, value["snapshot"]))
                                          for label, value in data["controls"].items()}
        manifest["status"] = "captured"
        save(root / "run.json", manifest)
        summary = verify(root)
        (root / "summary.txt").write_text(summary, encoding="utf-8")
        manifest["status"] = "verified"
        return summary
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        raise
    finally:
        save(root / "run.json", manifest)


def verify(root):
    from aorta.instrumentation.rocjitsu_sanitizers.report import read_report
    manifest = load(contained(root, "run.json"))
    require(manifest["schema"] == "waitcheck-controls-run/1" and manifest["status"] in ("captured", "verified"),
            "run capture incomplete or failed")
    data = check_prepared(root / "snapshot")
    require(manifest["aorta_before"] == manifest["aorta_after"]
            and manifest["backend_before"] == manifest["backend_after"], "executable fingerprint changed")
    expected_hashes = {label: value["sha256"] for label, value in data["controls"].items()}
    require(manifest["input_hashes_before"] == manifest["input_hashes_after"] == expected_hashes,
            "input fingerprint changed")
    require(set(manifest["environment_override"]) == {"ROCJITSU_PREBUILT"}, "unexpected environment override")
    bundle = recorded_path(manifest["environment_override"]["ROCJITSU_PREBUILT"])
    require(recorded_path(manifest["backend_before"]["path"]) == bundle / "bin" / "rj_waitcheck",
            "recorded backend does not match selected prebuilt bundle")
    require(set(manifest["commands"]) == set(LABELS), "expected three recorded invocations")
    lines = ["Draft Waitcheck example verification", "Scope: gfx1250; two supplied controls only.",
             "Saved records checked offline; no live host or scanner was queried."]
    for label in LABELS:
        row = manifest["commands"][label]
        error_case = label == "mismatch"
        verdict, state, cli_exit, native_exit = (("error", "error", 1, None) if error_case else
                                                ("warn", "ran", 0, 4) if label == "missing" else
                                                ("pass", "ran", 0, 0))
        require(row["returncode"] == cli_exit and not row["timed_out"] and not row["interrupted"]
                and not row.get("launch_error"), f"{label}: unexpected CLI execution result")
        require(row["argv"][:4] == [manifest["aorta_before"]["path"], "sweep", "run", "--recipe"]
                and len(row["argv"]) == 7 and row["argv"][5] == "--output", f"{label}: wrong invocation")
        recorded_root = recorded_path(row["cwd"])
        require(recorded_path(row["argv"][4]) == recorded_root / "snapshot" / f"{label}.json"
                and recorded_path(row["argv"][6]) == recorded_root / label,
                f"{label}: invocation paths disagree")
        text(contained(root, row["stdout"]))
        stderr = text(contained(root, row["stderr"]))
        require(error_case or not stderr.strip(), f"{label}: unexpected CLI stderr")
        path = contained(root, f"{label}/sanitizer_report.json")
        require(path.stat().st_size <= LOG_LIMIT, "report too large")
        report = read_report(path)
        require((report.target, report.overall_verdict.value, report.execution_status.value) ==
                (TARGET, verdict, "error" if error_case else "complete"), f"{label}: report verdict mismatch")
        require(len(report.checks) == 1 and len(report.worklist.kernels) == 1, f"{label}: unexpected selection")
        check = report.checks[0]
        require((check.sanitizer, check.state.value, check.verdict.value) == ("waitcheck", state, verdict)
                and len(check.kernel_results) == 1, f"{label}: wrong checker/result")
        result = check.kernel_results[0]
        expected = dict(recipe(label, data)["sanitizer_plan"]["source"]["kernel"], target=TARGET)
        require(recorded_path(manifest["identities"][label]["code_object"]) ==
                recorded_root / "snapshot" / expected["code_object"], f"{label}: input path is outside recorded snapshot")
        expected["code_object"] = manifest["identities"][label]["code_object"]
        require(manifest["identities"][label] == expected, f"{label}: recorded identity differs from recipe")
        for identity in (report.worklist.kernels[0].identity, result.identity):
            require(identity.exact and not identity.code_object_scan and identity.to_dict() == expected,
                    f"{label}: report identity mismatch")
        require((result.state.value, result.verdict.value, result.returncode) == (state, verdict, native_exit),
                f"{label}: native result mismatch")
        require(not check.diagnostics_truncated and not result.diagnostics_truncated,
                f"{label}: truncated diagnostics")
        backend = dict(check.backend)
        require(backend.get("path") == manifest["backend_before"]["path"]
                and backend.get("sha256") == manifest["backend_before"]["sha256"], f"{label}: backend mismatch")
        require(check.findings == result.findings, f"{label}: findings disagree")
        log_path = root / label / "waitcheck" / "waitcheck-0.log"
        if error_case:
            require(result.reason == "code_object_digest_mismatch" and check.reason == "worklist_not_fully_checked"
                    and not result.findings and not log_path.exists() and not log_path.is_symlink(),
                    "mismatch: expected refusal before native scanner launch")
        else:
            require(result.reason is None and check.reason is None, f"{label}: unexpected reason")
            log = text(contained(root, f"{label}/waitcheck/waitcheck-0.log"))
            counts = (3, 2, 1) if label == "missing" else (4, 2, 0)
            pattern = (r"rj_waitcheck: " + re.escape(expected["code_object"]) + r":gfx1250\[" +
                       str(expected["code_object_index"]) + r"\]:kernel=\.text\+0x" +
                       format(expected["entry_offset"], "x") +
                       r": instructions=(\d+) memory-events=(\d+) diagnostics=(\d+)")
            matches = [re.fullmatch(pattern, line.strip()) for line in log.splitlines()]
            matches = [match for match in matches if match]
            require(len(matches) == 1 and tuple(map(int, matches[0].groups())) == counts,
                    f"{label}: unsupported or unexpected raw count/scope format")
            require(len(result.findings) == counts[2], f"{label}: wrong finding count")
            require(len([line for line in log.splitlines() if line.strip()]) == (4 if label == "missing" else 1),
                    f"{label}: unexpected additional raw output")
            if label == "missing":
                finding = result.findings[0]
                context = dict(finding.metadata)
                require(finding.code == "wait_hazard" and finding.kernel_name == expected["name"]
                        and finding.code_object == expected["code_object"] and finding.entry_offset == expected["entry_offset"]
                        and "missing s_wait_kmcnt <= 0 before use of s4" in finding.message
                        and finding.message in log, "missing: expected s4 wait hazard absent")
                require("s_load_b32 s4, s[0:1], 0x0" in context.get("context_1", "")
                        and "s_mov_b32 s5, s4" in context.get("context_2", "")
                        and all(context[key] in log for key in ("context_1", "context_2")),
                        "missing: producer/consumer evidence absent")
        lines.append(f"{label}: CLI={cli_exit}, native={native_exit}, state={state}, verdict={verdict}; expected result verified.")
    lines += ["AORTA CLI exit 0 on the missing-wait case is a warning, not a clean scan.",
              "The mismatch case was refused; it was not sanitized. The route remains draft pending its other qualification gates."]
    return "\n".join(lines) + "\n"


def nonnegative(value):
    parsed = int(value, 0)
    if parsed < 0:
        raise argparse.ArgumentTypeError("selector must be nonnegative")
    return parsed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    for label in ("missing", "corrected"):
        prep.add_argument(f"--{label}-object", required=True, type=Path)
        prep.add_argument(f"--{label}-index", required=True, type=nonnegative)
        prep.add_argument(f"--{label}-entry", required=True, type=nonnegative)
    prep.add_argument("--kernel-name", required=True)
    prep.add_argument("--build-id", required=True)
    prep.add_argument("--output", required=True, type=Path)
    runner = commands.add_parser("run")
    runner.add_argument("--prepared", required=True, type=Path)
    runner.add_argument("--aorta", required=True, type=Path)
    runner.add_argument("--prebuilt", required=True, type=Path, help="bundle root containing bin/rj_waitcheck")
    runner.add_argument("--tool-sha256", required=True)
    runner.add_argument("--output", required=True, type=Path)
    verifier = commands.add_parser("verify")
    verifier.add_argument("--run", required=True, type=Path)
    verifier.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            summary = prepare(args)
        elif args.command == "run":
            summary = run(args)
        else:
            require(not args.run.is_symlink(), "run directory symlink refused")
            root = args.run.resolve(strict=True)
            if args.output:
                require(not args.output.exists() and not args.output.is_symlink()
                        and not args.output.resolve().is_relative_to(root), "summary must be a new file outside evidence")
            summary = verify(root)
            if args.output:
                with args.output.open("x", encoding="utf-8") as stream:
                    stream.write(summary)
        print(summary)
        return 0
    except (ValueError, OSError, KeyError, TypeError, ImportError, subprocess.SubprocessError) as exc:
        print(f"Example validation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
