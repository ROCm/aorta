#!/usr/bin/env python3
"""Run one sanitizer recipe end to end: Launch -> Watch -> Autopsy.

Executed with the cluster-intelligence-agent virtualenv so it can reuse the
agents' own job-record and sbatch machinery instead of reimplementing them.
Emits a single JSON object on stdout; progress goes to stderr so the caller can
show it live without corrupting the result.

Launch here is the deterministic path: the recipe, node constraints and the
ConSan environment are passed in explicitly rather than discovered by the
planner LLM, because a demo cannot tolerate a run that silently omits
LD_PRELOAD and then reports a clean guardrail it never actually exercised.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import yaml

from aorta.cia.autopsy.orchestrator import run_autopsy
from aorta.cia.launch.cluster import submit_sbatch
from aorta.cia.watch.poll import poll_jobs
from aorta.cia.launch.job import (
    JobRecord, _utc_now, new_job_id, read_job_json, update_job_status, write_job_json,
)

TERMINAL_STATES = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY"}


def _default_aorta_root() -> str:
    """Where the AORTA CLI's recipes live: the installed package, not a site."""
    import aorta

    return str(Path(aorta.__file__).resolve().parent)


def venv_bin(name: str) -> str:
    """Absolute path to a console script in the venv running this driver.

    The batch job activates whichever virtualenv submitted it, which is the
    chatbot's rather than the agents'. Naming these tools bare would leave them
    resolved by an inherited PATH: it works from a shell that once activated the
    agents' venv and fails with 'command not found' from a clean one.
    """
    candidate = Path(sys.prefix) / "bin" / name
    if candidate.is_file():
        return str(candidate)
    return shutil.which(name) or name


def log(msg: str) -> None:
    print(f"[triage] {msg}", file=sys.stderr, flush=True)


def sacct_state(slurm_id: str) -> str:
    """Terminal-or-running state of a Slurm job, tolerating a lagging accounting DB."""
    try:
        r = subprocess.run(
            ["sacct", "-j", slurm_id, "--format=State", "--noheader", "--parsable2", "-X"],
            capture_output=True, text=True, timeout=30,
        )
        line = (r.stdout or "").strip().splitlines()
        if line:
            return line[0].strip().split()[0]
    except Exception:
        pass
    return "UNKNOWN"


def wait_for_job(slurm_id: str, timeout: int, interval: int = 5) -> str:
    deadline = time.time() + timeout
    state = "UNKNOWN"
    while time.time() < deadline:
        state = sacct_state(slurm_id)
        if state in TERMINAL_STATES:
            log(f"slurm {slurm_id} reached {state}")
            return state
        log(f"slurm {slurm_id} state={state} ...")
        time.sleep(interval)
    return f"TIMEOUT_WAITING({state})"


def run(cmd: list[str], timeout: int, env: dict | None = None) -> subprocess.CompletedProcess:
    log(f"$ {shlex.join(cmd)}")
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, env=env or os.environ.copy()
    )


_KERNEL_RE = re.compile(r'__global__\s+[\w\s:<>,*&]*?\b(\w+)\s*\(', re.MULTILINE)


def detect_kernel_name(source: str) -> str:
    """First __global__ function in the source, so the caller need not name it."""
    match = _KERNEL_RE.search(source)
    return match.group(1) if match else ""


def write_kernel_recipe(
    *, recipe_path: Path, kernel_name: str, command: Path, target: str, ticket: str
) -> Path:
    """Emit a sanitizer recipe for a single user-supplied kernel.

    Uses the 'kernel' source kind rather than the built-in consan_repro variants,
    since those hardcode the two fixture kernels. ConSan runs the program named by
    'command' and scopes its analysis to the one selected identity.
    """
    recipe = {
        "schema_version": 1,
        "mode": "sanitizer",
        "ticket": ticket,
        "description": f"Chat-submitted kernel {kernel_name} triaged on {target}.",
        "sanitizer_plan": {
            "target": target,
            "source": {
                "kind": "kernel",
                "kernel": {"name": kernel_name},
                "command": str(command),
                "consan_log": True,
            },
            "scope": {"kind": "kernel"},
            "selection": {"requirement": "top_dispatch_count", "top_n": 1},
            "sanitizers": ["consan"],
            "policy": {"consan_policy": "strict", "on_missing_backend": "fail"},
            "output": {"report": "sanitizer_report.json"},
        },
    }
    recipe_path.write_text(yaml.safe_dump(recipe, sort_keys=False), encoding="utf-8")
    return recipe_path


def reconcile_stale_jobs(jobs_root: Path) -> int:
    """Mark finished jobs whose record still claims 'running' as terminal.

    Watch only monitors jobs the registry considers active, so a record left
    'running' by an earlier interrupted run makes it spend every round tailing a
    dead job instead of the one we just launched.
    """
    fixed = 0
    for job_json in jobs_root.glob("*/job.json"):
        try:
            record = read_job_json(job_json)
        except Exception:
            continue
        if record.status != "running" or not record.scheduler_job_id:
            continue
        state = sacct_state(record.scheduler_job_id)
        if state in TERMINAL_STATES:
            update_job_status(jobs_root, record.job_id,
                              "completed" if state == "COMPLETED" else "failed")
            fixed += 1
    if fixed:
        log(f"reconciled {fixed} stale job record(s) to terminal")
    return fixed


def read_watch_events(job_dir: Path) -> list[dict]:
    """Watch's structured alerts: its signal, confidence and stated reasoning.

    Read from the events file rather than the loop's stdout, which is no longer
    a separate process to capture, and which only ever carried a prose tail of
    what these records hold as fields.
    """
    path = job_dir / "events.jsonl"
    if not path.is_file():
        return []
    events = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def summarize_sanitizer(report: dict) -> dict:
    """Flatten the sanitizer report into the few facts that explain a race."""
    checks = report.get("checks") or []
    summary: dict = {
        "overall_verdict": report.get("overall_verdict"),
        "execution_status": report.get("execution_status"),
        "target": report.get("target"),
        "checks": [],
        "total_findings": 0,
    }

    kernels = ((report.get("worklist") or {}).get("kernels") or [])
    if kernels:
        summary["kernel"] = ((kernels[0].get("identity") or {}).get("name"))

    for check in checks:
        findings = check.get("findings") or []
        summary["total_findings"] += len(findings)
        entry = {
            "sanitizer": check.get("sanitizer"),
            "state": check.get("state"),
            "verdict": check.get("verdict"),
            "findings": len(findings),
            "reason": check.get("reason"),
            "returncode": check.get("returncode"),
        }
        backend = check.get("backend") or {}
        if backend.get("selected_kernel"):
            entry["kernel"] = backend["selected_kernel"]
        # One representative conflict carries the wave/LDS detail an engineer
        # needs; the other 63 are the same race seen from other lanes.
        if findings:
            meta = findings[0].get("metadata") or {}
            entry["example"] = {
                k: meta[k]
                for k in ("first_owner", "second_owner", "first_lds", "second_lds",
                          "first_kind", "second_kind", "first_inst", "second_inst")
                if k in meta
            }
            entry["example_message"] = findings[0].get("message", "")[:400]
        summary["checks"].append(entry)
    return summary


def run_triage(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recipe", help="Path to an existing aorta sanitizer recipe YAML")
    ap.add_argument("--source", help="Path to a .hip file to compile and triage")
    ap.add_argument("--command", help=(
        "Raw command to launch instead of a sanitizer sweep, for workloads that "
        "are not expressible as a recipe. '{bundle}' is replaced with the job's "
        "bundle directory so the workload can drop artifacts where Autopsy reads "
        "them."
    ))
    ap.add_argument("--kernel-name", default="",
                    help="Kernel to analyse (auto-detected from --source when omitted)")
    ap.add_argument("--arch", default=os.environ.get("CIA_GPU_ARCH", "gfx950"))
    ap.add_argument("--jobs-root", default=os.environ.get("CIA_JOBS_ROOT", ""))
    ap.add_argument("--node", default=os.environ.get("CIA_DEMO_NODE", ""))
    ap.add_argument("--aorta-root", default=os.environ.get("AORTA_PATH", _default_aorta_root()))
    ap.add_argument("--job-timeout", type=int, default=900)
    ap.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                    help="Extra variable to export in the batch job. The sanitizers "
                         "need ROCJITSU_BUILD and LD_PRELOAD, which used to reach the "
                         "job by being set in a subprocess this driver no longer runs in.")
    ap.add_argument("--watch-rounds", type=int, default=20)
    ap.add_argument("--watch-grace", type=int, default=180,
                    help="Seconds to let Watch alert after the job ends")
    ap.add_argument("--label", default="", help="Human label for this run (racy / fixed)")
    args = ap.parse_args(argv)

    if not args.recipe and not args.source and not args.command:
        return {"ok": False, "error": "pass either --recipe or --source"}

    jobs_root = Path(args.jobs_root or (Path.home() / "cia-jobs")).expanduser().resolve()
    jobs_root.mkdir(parents=True, exist_ok=True)

    job_id = new_job_id()
    job_dir = jobs_root / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    log_path = str(job_dir / "watch.log")
    aorta_output = str(job_dir / "bundle" / "aorta")

    kernel_name = args.kernel_name
    compiled_from_source = bool(args.source)
    job_env_vars: dict[str, str] = {}
    for pair in args.env:
        key, _, value = pair.partition("=")
        if key and value:
            job_env_vars[key] = value

    # Watch reads the job log, but a sanitizer verdict lands in a JSON report, so
    # a hazard never reaches the stream Watch is tailing and the job looks
    # healthy. Echo the findings after the sweep, with ';' rather than '&&' so it
    # still runs when the guardrail exits non-zero, and '|| true' so a broken
    # summary can never fail the run.
    echo_findings = (
        f"; {shlex.quote(venv_bin('python'))} "
        f"{shlex.quote(str(Path(__file__).resolve().parents[1] / 'scripts' / 'echo_sanitizer_findings.py'))} "
        f"{shlex.quote(str(Path(aorta_output) / 'sanitizer_report.json'))} || true"
    )

    if compiled_from_source:
        src = Path(args.source).expanduser().resolve()
        if not src.is_file():
            print(json.dumps({"ok": False, "error": f"source not found: {src}"}))
            return 1
        source_text = src.read_text(encoding="utf-8", errors="replace")
        kernel_name = kernel_name or detect_kernel_name(source_text)
        if not kernel_name:
            print(json.dumps({
                "ok": False,
                "error": "could not find a __global__ kernel in the source; "
                         "pass --kernel-name explicitly",
            }))
            return 1

        # Keep the source with the job so the bundle is self-describing.
        staged = job_dir / "kernel.hip"
        staged.write_text(source_text, encoding="utf-8")
        binary = job_dir / "kernel.bin"
        recipe = write_kernel_recipe(
            recipe_path=job_dir / "recipe.yaml",
            kernel_name=kernel_name,
            command=binary,
            target=args.arch,
            ticket=f"CHAT-{kernel_name}",
        )
        # Compile on the compute node: hipcc lives with ROCm on the GPU nodes, not
        # on the login node where the chatbot runs.
        command = (
            f"hipcc --offload-arch={shlex.quote(args.arch)} "
            f"-o {shlex.quote(str(binary))} {shlex.quote(str(staged))} && "
            f"{shlex.quote(venv_bin('aorta'))} sweep run "
            f"--recipe {shlex.quote(str(recipe))} "
            f"--output {shlex.quote(aorta_output)}"
            + echo_findings
        )
        # ConSan samples workgroups with a large default stride, so a small repro
        # grid can be skipped entirely: every site gets patched but nothing is
        # recorded, and the run exits 86 with zero findings that look like a pass.
        # Pasted kernels are small by nature, so record every workgroup.
        job_env_vars["RJ_CONSAN_MOI_RUNTIME_SAMPLE_STRIDE"] = os.environ.get(
            "RJ_CONSAN_MOI_RUNTIME_SAMPLE_STRIDE", "1"
        )
        log(f"kernel={kernel_name} arch={args.arch} source={src.name}")
    elif args.command:
        # A raw workload: no recipe, no sanitizer sweep. Watch still tails the
        # log and Autopsy still classifies whatever artifacts the workload leaves
        # in the bundle, which is how a training run gets the same treatment as a
        # kernel sweep.
        recipe = None
        command = args.command.replace("{bundle}", str(job_dir / "bundle"))
        log(f"raw command: {command[:160]}")
    else:
        recipe = Path(args.recipe).expanduser().resolve()
        if not recipe.is_file():
            print(json.dumps({"ok": False, "error": f"recipe not found: {recipe}"}))
            return 1
        command = (
            f"{shlex.quote(venv_bin('aorta'))} sweep run "
            f"--recipe {shlex.quote(str(recipe))} "
            f"--output {shlex.quote(aorta_output)}"
            + echo_findings
        )

    # Watch and Autopsy both reach the LLM through llm/config.py, which defaults
    # to localhost:4000. The proxy lives in its own Slurm allocation, so without
    # this Watch's assessment raises a connection error, and because poll.py
    # saves its file cursor before the assessment runs, the consumed log is gone:
    # the job silently looks healthy no matter what was in it.
    # The sanitizer positive control exits non-zero by design ("guardrail not
    # clean"). Let the batch script swallow that so Slurm reports COMPLETED and
    # the verdict comes from the sanitizer report rather than the exit code.
    env = os.environ.copy()
    env["CIA_TOLERATE_NONZERO"] = "1"

    record = JobRecord(
        job_id=job_id,
        node=args.node,
        recipe=recipe.stem if recipe else (args.label or "raw-command"),
        launched_at=_utc_now(),
        log_path=log_path,
        aorta_output=aorta_output,
        status="running",
        launch_command=command,
        working_dir=args.aorta_root,
        scheduler="slurm",
        launcher="sbatch",
        env_vars=job_env_vars,
    )

    log(f"job_id={job_id} recipe={recipe.name if recipe else '(raw command)'} label={args.label or '-'}")
    reconcile_stale_jobs(jobs_root)
    log("── Launch ──")

    saved = os.environ.get("CIA_TOLERATE_NONZERO")
    os.environ["CIA_TOLERATE_NONZERO"] = "1"
    try:
        slurm_id, err = submit_sbatch(
            command=command,
            job_name=job_id,
            log_path=log_path,
            script_path=job_dir / "launch.sbatch",
            working_dir=args.aorta_root,
            env_vars=record.env_vars,
            node=args.node,
        )
    finally:
        if saved is None:
            os.environ.pop("CIA_TOLERATE_NONZERO", None)
        else:
            os.environ["CIA_TOLERATE_NONZERO"] = saved

    if err:
        return {"ok": False, "stage": "launch", "error": err, "job_id": job_id}

    record.scheduler_job_id = slurm_id
    write_job_json(record, jobs_root)
    log(f"submitted slurm job {slurm_id}")

    bundle = job_dir / "bundle"
    report_path = bundle / "report.json"

    # Watch has to start while the record still says 'running', because the job
    # registry is what makes it eligible for monitoring at all.
    log("── Watch ──")
    log(f"poll_jobs(rounds={args.watch_rounds})")
    watcher = threading.Thread(
        target=poll_jobs,
        kwargs={"jobs_root": jobs_root, "max_rounds": args.watch_rounds},
        daemon=True,
    )
    watcher.start()

    state = wait_for_job(slurm_id, timeout=args.job_timeout)

    # Give Watch a bounded window to notice the finished log, alert, assemble the
    # bundle and trigger Autopsy before falling back to doing it directly.
    grace = args.watch_grace
    log(f"waiting up to {grace}s for Watch to alert and assemble the bundle")
    deadline = time.time() + grace
    while time.time() < deadline:
        if report_path.is_file() or not watcher.is_alive():
            break
        time.sleep(5)

    watcher.join(timeout=30)
    watch_events = read_watch_events(job_dir)
    watch_tail = [
        f"{e.get('signal')} @ {e.get('confidence')}: {e.get('assessment', '')[:160]}"
        for e in watch_events
    ]
    for line in watch_tail:
        log(f"watch| {line}")

    # Watch only assembles a bundle after it raises an alert, so the bundle
    # existing before the fallback runs is the reliable signal that it fired.
    watch_alerted = report_path.is_file() or (bundle / "manifest.yaml").is_file()

    update_job_status(jobs_root, job_id, "completed" if state == "COMPLETED" else "failed")

    # Watch normally assembles the bundle and triggers Autopsy on alert. Do it
    # directly otherwise so the caller always gets a verdict rather than silence.
    if not report_path.is_file():
        if not (bundle / "manifest.yaml").is_file():
            log("── Bundle (direct) ──")
            try:
                from aorta.cia.watch.bundle_writer import write_bundle
                evidence = ""
                if Path(log_path).is_file():
                    evidence = "\n".join(
                        Path(log_path).read_text(errors="replace").splitlines()[-200:]
                    )
                write_bundle(record, job_dir, evidence, "sanitizer_guardrail_not_clean")
                log(f"assembled bundle at {bundle}")
            except Exception as exc:
                log(f"bundle assembly failed: {exc}")

        log("── Autopsy (direct) ──")
        try:
            report = run_autopsy(bundle, kb_version="kb-static-poc")
            report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            log(f"autopsy category={report.get('category')} "
                f"confidence={report.get('confidence')}")
        except Exception as exc:
            log(f"autopsy failed: {exc}")
    else:
        log("── Autopsy (via Watch) ──")

    result: dict = {
        "ok": True,
        "job_id": job_id,
        "slurm_job_id": slurm_id,
        "label": args.label,
        "recipe": str(recipe) if recipe else "(raw command)",
        "slurm_state": state,
        "job_dir": str(job_dir),
        "bundle": str(bundle),
        "log_path": log_path,
        "watch_alerted": watch_alerted,
        "watch_tail": watch_tail,
        "kernel": kernel_name,
        "compiled_from_source": compiled_from_source,
    }

    # A compile failure short-circuits the sweep, which otherwise looks like a
    # silent no-report run. Report it as such so the caller can show the diagnostics
    # instead of guessing at a sanitizer verdict that was never produced.
    if compiled_from_source and not binary.is_file():
        diagnostics = []
        if Path(log_path).is_file():
            diagnostics = [
                ln for ln in Path(log_path).read_text(errors="replace").splitlines()
                if "error:" in ln or "warning:" in ln
            ][:20]
        result["ok"] = False
        result["stage"] = "compile"
        result["error"] = "hipcc failed to build the submitted kernel"
        result["compile_diagnostics"] = diagnostics
        return result

    if report_path.is_file():
        report = json.loads(report_path.read_text())
        result["report_path"] = str(report_path)
        result["autopsy"] = {
            k: report.get(k)
            for k in ("category", "confidence", "rationale", "evidence",
                      "next_probes", "tooling_gaps", "signals")
            if k in report
        }
    else:
        result["autopsy"] = None
        result["warning"] = "no report.json produced"

    san = bundle / "aorta" / "sanitizer_report.json"
    if san.is_file():
        s = json.loads(san.read_text())
        result["sanitizer_report_path"] = str(san)
        result["sanitizer"] = summarize_sanitizer(s)

    return result


def main() -> int:
    """CLI wrapper: the chatbot calls run_triage() directly instead."""
    result = run_triage()
    print(json.dumps(result, indent=2))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
