"""Real GPU acceptance for the sanitizer -> Watch -> Autopsy chain.

The ordinary CIA suite mocks scheduler and sanitizer boundaries. This test
keeps only the scheduler transport local: Launch still renders and submits its
production batch script, while that script compiles and executes the committed
racy ConSan repro on the attached GPU. Watch consumes the resulting job log and
is the only caller that can assemble the bundle and trigger Autopsy.
"""

from __future__ import annotations

import inspect
import json
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

import aorta.cia.watch.poll as poll_mod
from aorta.cia.launch import launch
from aorta.cia.launch.job import JobRecord, read_job_json, write_job_json
from aorta.cia.watch.poll import autopsy_state, poll_jobs
from aorta.instrumentation.rocjitsu_sanitizers.consan import resolve_consan_hook

pytestmark = [
    pytest.mark.cia_hardware,
    pytest.mark.gpu,
    pytest.mark.rocm,
    pytest.mark.integration,
    pytest.mark.slow,
]

_REPO = Path(__file__).resolve().parents[2]
_RACY_SOURCE = (
    _REPO
    / "recipes"
    / "sanitizers"
    / "fixtures"
    / "repro"
    / "consan_lds_race_2wave.hip"
)
_REQUIRED = os.environ.get("AORTA_CIA_HARDWARE_SMOKE_REQUIRED", "").lower() in {
    "1",
    "true",
    "yes",
}


def _hardware_requirements() -> tuple[str, str, Path]:
    """Return target, hipcc, and ConSan hook; skip only outside the required lane."""
    try:
        from aorta.utils.gpu_control import GPUVendor, detect_gpu

        vendor, target = detect_gpu()
    except Exception as exc:  # pragma: no cover - depends on runner hardware
        if _REQUIRED:
            pytest.fail(f"required CIA hardware smoke cannot detect a GPU: {exc}")
        pytest.skip(f"no detectable GPU: {exc}")
    if vendor is not GPUVendor.AMD or not target or not target.startswith("gfx"):
        if _REQUIRED:
            pytest.fail(f"required CIA hardware smoke needs AMD ROCm, got {vendor}/{target}")
        pytest.skip(f"no AMD ROCm GPU detected: {vendor}/{target}")

    hipcc = shutil.which("hipcc")
    hook = resolve_consan_hook()
    missing = [
        name
        for name, present in (("hipcc", hipcc), ("ConSan hook", hook))
        if not present
    ]
    if missing:
        message = "CIA hardware smoke is missing " + ", ".join(missing)
        if _REQUIRED:
            pytest.fail(message)
        pytest.skip(message)
    assert hipcc is not None and hook is not None
    return target, hipcc, hook


def _write_recipe(path: Path, *, target: str, binary: Path) -> None:
    recipe = {
        "schema_version": 1,
        "mode": "sanitizer",
        "ticket": "CIA-HARDWARE-SMOKE",
        "sanitizer_plan": {
            "target": target,
            "source": {
                "kind": "consan_repro",
                "variant": "racy",
                "hip": str(_RACY_SOURCE),
                "command": str(binary),
                "consan_log": True,
            },
            "scope": {"kind": "kernel"},
            "selection": {"requirement": "top_dispatch_count", "top_n": 1},
            "sanitizers": ["consan"],
            "policy": {
                "consan_policy": "strict",
                "on_missing_backend": "fail",
                "timeout_seconds": 180,
            },
            "output": {"report": "sanitizer_report.json"},
        },
    }
    path.write_text(yaml.safe_dump(recipe, sort_keys=False), encoding="utf-8")


def _write_local_sbatch(path: Path) -> None:
    """Install a scheduler shim; the production Launch path remains unchanged."""
    path.write_text(
        f"""#!{sys.executable}
import os
import subprocess
import sys
from pathlib import Path

script = Path(sys.argv[-1])
output = next(
    line.split("=", 1)[1]
    for line in script.read_text(encoding="utf-8").splitlines()
    if line.startswith("#SBATCH --output=")
)
env = dict(os.environ)
env["SLURM_JOB_ID"] = "cia-hardware-smoke"
with open(output, "w", encoding="utf-8") as stream:
    result = subprocess.run(
        ["bash", str(script)],
        stdout=stream,
        stderr=subprocess.STDOUT,
        env=env,
        check=False,
    )
print("cia-hardware-smoke")
raise SystemExit(result.returncode)
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


def _launch_command(recipe: Path, output: Path, report: Path) -> str:
    """Run the real sanitizer CLI and echo its machine-readable result to Watch."""
    aorta = shutil.which("aorta")
    sibling = Path(sys.executable).with_name("aorta")
    if aorta is None and sibling.is_file():
        aorta = str(sibling)
    if aorta is None:
        pytest.fail("editable install has no aorta CLI on PATH")
    echo = (
        "import json,sys; "
        "d=json.load(open(sys.argv[1], encoding='utf-8')); "
        "c=next(x for x in d['checks'] if x['sanitizer']=='consan'); "
        "n=len(c.get('findings') or [])+sum(len(k.get('findings') or []) "
        "for k in c.get('kernel_results') or []); "
        "print('[sanitizer] consan: verdict=%s state=%s findings=%d' % "
        "(c.get('verdict'),c.get('state'),n))"
    )
    return (
        f"{shlex.quote(aorta)} sweep run --recipe {shlex.quote(str(recipe))} "
        f"--output {shlex.quote(str(output))}; "
        "_cia_smoke_rc=$?; "
        f"{shlex.quote(sys.executable)} -c {shlex.quote(echo)} "
        f"{shlex.quote(str(report))}; "
        "(exit $_cia_smoke_rc)"
    )


class _OfflineRouter:
    """Keep this CI smoke on Autopsy's production rule-based fallback."""

    def __init__(self, *_args, **_kwargs):
        raise RuntimeError("CIA hardware smoke does not require an external LLM")


def test_consan_launch_watch_autopsy_on_real_gpu(tmp_path: Path, monkeypatch) -> None:
    target, hipcc, _hook = _hardware_requirements()
    binary = tmp_path / "consan-racy"
    subprocess.run(
        [
            hipcc,
            f"--offload-arch={target}",
            "-O1",
            "-g",
            str(_RACY_SOURCE),
            "-o",
            str(binary),
        ],
        check=True,
        timeout=600,
    )

    jobs_root = tmp_path / "jobs"
    job_id = "cia-hardware-smoke"
    job_dir = jobs_root / job_id
    output = job_dir / "bundle" / "aorta"
    report_path = output / "sanitizer_report.json"
    recipe = tmp_path / "consan-racy.yaml"
    _write_recipe(recipe, target=target, binary=binary)

    shim_dir = tmp_path / "bin"
    shim_dir.mkdir()
    _write_local_sbatch(shim_dir / "sbatch")
    monkeypatch.setenv("PATH", f"{shim_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("CIA_TOLERATE_NONZERO", "1")
    monkeypatch.setenv("RJ_CONSAN_MOI_RUNTIME_SAMPLE_STRIDE", "1")

    watch_log = job_dir / "watch.log"
    job = JobRecord(
        job_id=job_id,
        node="",
        recipe=recipe.name,
        launched_at=datetime.now(timezone.utc).isoformat(),
        log_path=str(watch_log),
        aorta_output=str(output),
        watch_files=[str(watch_log)],
        launch_command=_launch_command(recipe, output, report_path),
        working_dir=str(_REPO),
        scheduler="ci-local",
        launcher="sbatch",
        recipe_path=str(recipe),
    )
    write_job_json(job, jobs_root)

    launch_options = {
        "command": job.launch_command,
        "job_name": job_id,
        "log_path": str(watch_log),
        "script_path": job_dir / "launch.sbatch",
        "working_dir": str(_REPO),
    }
    if "tolerate_nonzero" in inspect.signature(launch).parameters:
        # PR #424 moved this from process-global environment into the launch
        # call so concurrent triages cannot change one another's policy.
        launch_options["tolerate_nonzero"] = True
    scheduler_id, error = launch(
        **launch_options,
    )
    assert error == ""
    assert scheduler_id == "cia-hardware-smoke"
    assert watch_log.is_file(), "Launch produced no log for Watch"

    sanitizer_report = json.loads(report_path.read_text(encoding="utf-8"))
    consan = next(
        check
        for check in sanitizer_report["checks"]
        if check["sanitizer"] == "consan"
    )
    report_debug = json.dumps(sanitizer_report, indent=2)[:6000]
    assert sanitizer_report["execution_status"] == "complete", report_debug
    assert consan["state"] == "ran", "ConSan was selected but did not execute"
    assert consan["verdict"] == "fail", report_debug
    assert consan.get("findings"), report_debug

    monkeypatch.setattr(poll_mod, "LogFinder", lambda **_kwargs: object())
    monkeypatch.setattr(
        "aorta.cia.autopsy.router.TriageRouter",
        _OfflineRouter,
    )
    watch_config = tmp_path / "watch.yaml"
    watch_config.write_text(
        yaml.safe_dump(
            {
                "watch": {
                    "poll_interval_sec": 0,
                    "confidence_threshold": 0.7,
                    "expectations": ["ConSan must not report a race."],
                },
                "log_finder": {},
            }
        ),
        encoding="utf-8",
    )

    # This is the only Autopsy entry point in the smoke. A report therefore
    # proves Watch triggered it rather than the test invoking a fallback.
    poll_jobs(jobs_root, config_path=watch_config, max_rounds=1)

    event = json.loads(
        (job_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()[-1]
    )
    assert event["event_type"] == "watchdog_alert"
    assert event["signal"] == "WATCH_UNKNOWN_ERROR"
    assert "state=ran" in event["excerpt"]

    bundle = job_dir / "bundle"
    manifest = yaml.safe_load(
        (bundle / "manifest.yaml").read_text(encoding="utf-8")
    )
    assert manifest["metadata"]["watch_signal"] == "WATCH_UNKNOWN_ERROR"

    autopsy_report = json.loads(
        (bundle / "report.json").read_text(encoding="utf-8")
    )
    assert autopsy_report["category"] == "gpu_race"
    assert any(
        item.get("signal") == "SAN_CONSAN_RACE"
        for item in autopsy_report["evidence"]
    )
    assert autopsy_state(job_dir)["state"] == "done"
    assert read_job_json(job_dir / "job.json").status == "failed"
