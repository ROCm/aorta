"""The CIA launch seam works on a workstation without Slurm."""

from __future__ import annotations

import json
import os
import shlex
import threading
import time
from pathlib import Path

import pytest

from aorta.cia import launch as launch_pkg
from aorta.cia.launch.job import JobRecord, _utc_now, read_job_json, write_job_json
from aorta.cia.launch.local import cancel_local, local_state


def _wait_for_state(job_id: str, job_dir: Path, expected: str, timeout: float = 5) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if local_state(job_id, job_dir) == expected:
            return
        time.sleep(0.02)
    raise AssertionError(f"{job_id} did not reach {expected}: {local_state(job_id, job_dir)}")


def _single_json(root: Path, pattern: str) -> dict:
    paths = list(root.glob(pattern))
    assert len(paths) == 1, paths
    return json.loads(paths[0].read_text(encoding="utf-8"))


def test_auto_uses_local_when_sbatch_is_absent(tmp_path, monkeypatch):
    monkeypatch.setattr("aorta.cia.launch.cluster.sbatch_available", lambda: False)

    job_id, error = launch_pkg.launch(
        command="printf 'hello from local\\n'",
        job_name="local-success",
        log_path=str(tmp_path / "watch.log"),
        script_path=tmp_path / "launch.sh",
        backend="auto",
    )

    assert error == ""
    assert job_id.startswith("local:")
    _wait_for_state(job_id, tmp_path, "COMPLETED")
    assert "hello from local" in (tmp_path / "watch.log").read_text(encoding="utf-8")
    assert _single_json(tmp_path, "local.status.*.json")["exit_code"] == 0
    assert _single_json(tmp_path, "local.process.*.json")["job_id"] == job_id


def test_local_failure_is_durable(tmp_path):
    job_id, error = launch_pkg.launch(
        command="exit 17",
        job_name="local-failure",
        log_path=str(tmp_path / "watch.log"),
        script_path=tmp_path / "launch.sh",
        backend="local",
    )

    assert error == ""
    _wait_for_state(job_id, tmp_path, "FAILED")
    assert _single_json(tmp_path, "local.status.*.json")["exit_code"] == 17


def test_expected_nonzero_can_complete(tmp_path):
    job_id, error = launch_pkg.launch(
        command="exit 86",
        job_name="local-guardrail",
        log_path=str(tmp_path / "watch.log"),
        script_path=tmp_path / "launch.sh",
        tolerate_nonzero=True,
        backend="local",
    )

    assert error == ""
    _wait_for_state(job_id, tmp_path, "COMPLETED")
    assert "non-zero tolerated" in (tmp_path / "watch.log").read_text(encoding="utf-8")


def test_a_trailing_shell_comment_does_not_break_the_wrapper(tmp_path):
    job_id, error = launch_pkg.launch(
        command="printf 'comment-ok\\n' # the closing delimiter must be on its own line",
        job_name="local-comment",
        log_path=str(tmp_path / "watch.log"),
        script_path=tmp_path / "launch.sh",
        backend="local",
    )

    assert error == ""
    _wait_for_state(job_id, tmp_path, "COMPLETED")
    assert "comment-ok" in (tmp_path / "watch.log").read_text(encoding="utf-8")


def test_a_heredoc_terminator_is_preserved(tmp_path):
    job_id, error = launch_pkg.launch(
        command="cat <<'EOF'\nheredoc-ok\nEOF",
        job_name="local-heredoc",
        log_path=str(tmp_path / "watch.log"),
        script_path=tmp_path / "launch.sh",
        backend="local",
    )

    assert error == ""
    _wait_for_state(job_id, tmp_path, "COMPLETED")
    assert "heredoc-ok" in (tmp_path / "watch.log").read_text(encoding="utf-8")


def test_local_launch_preserves_working_directory_and_environment(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    job_id, error = launch_pkg.launch(
        command='printf "%s|%s" "$PWD" "$LOCAL_SENTINEL"',
        job_name="local-env",
        log_path=str(tmp_path / "watch.log"),
        script_path=tmp_path / "launch.sh",
        working_dir=str(work),
        env_vars={"LOCAL_SENTINEL": "value with spaces"},
        backend="local",
    )

    assert error == ""
    _wait_for_state(job_id, tmp_path, "COMPLETED")
    assert f"{work}|value with spaces" in (tmp_path / "watch.log").read_text(encoding="utf-8")


@pytest.mark.parametrize("script_path", [Path("launch.sh"), Path("jobs/launch.sh")])
def test_relative_launch_paths_survive_a_working_directory_change(
    tmp_path, monkeypatch, script_path
):
    monkeypatch.chdir(tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    job_dir = (tmp_path / script_path.parent).resolve()

    job_id, error = launch_pkg.launch(
        command="printf relative-ok",
        job_name="relative-path",
        log_path=str(script_path.parent / "watch.log"),
        script_path=script_path,
        working_dir=str(work),
        backend="local",
    )

    assert error == ""
    _wait_for_state(job_id, job_dir, "COMPLETED")
    assert (job_dir / script_path.name).is_file()
    assert "relative-ok" in (job_dir / "watch.log").read_text(encoding="utf-8")
    assert not list(work.glob("local.status*.json"))


def test_status_sidecars_are_isolated_for_overlapping_launches(tmp_path):
    first_id, first_error = launch_pkg.launch(
        command="sleep 30",
        job_name="first",
        log_path=str(tmp_path / "first.log"),
        script_path=tmp_path / "launch-first.sh",
        backend="local",
    )
    second_id, second_error = launch_pkg.launch(
        command="sleep 30",
        job_name="second",
        log_path=str(tmp_path / "second.log"),
        script_path=tmp_path / "launch-second.sh",
        backend="local",
    )
    assert first_error == second_error == ""
    try:
        cancelled, why = cancel_local(first_id, timeout=0.2)
        assert cancelled, why
        launch_pkg.record_cancellation(
            first_id,
            tmp_path,
            preserve_terminal=False,
        )

        assert local_state(first_id, tmp_path) == "CANCELLED"
        assert local_state(second_id, tmp_path) == "RUNNING"
        assert len(list(tmp_path.glob("local.process.*.json"))) == 2
    finally:
        cancel_local(second_id, timeout=0.2)


def test_late_cancellation_record_does_not_overwrite_completion(tmp_path):
    job_id, error = launch_pkg.launch(
        command="true",
        job_name="already-complete",
        log_path=str(tmp_path / "watch.log"),
        script_path=tmp_path / "launch.sh",
        backend="local",
    )
    assert error == ""
    _wait_for_state(job_id, tmp_path, "COMPLETED")

    cancelled, why = cancel_local(job_id)
    assert cancelled is True
    assert "finished before cancellation" in why
    launch_pkg.record_cancellation(job_id, tmp_path)

    assert local_state(job_id, tmp_path) == "COMPLETED"


def test_missing_working_directory_fails_instead_of_running_elsewhere(tmp_path):
    marker = tmp_path / "must-not-exist"
    job_id, error = launch_pkg.launch(
        command=f"touch {marker}",
        job_name="missing-cwd",
        log_path=str(tmp_path / "watch.log"),
        script_path=tmp_path / "launch.sh",
        working_dir=str(tmp_path / "missing"),
        backend="local",
    )

    assert error == ""
    _wait_for_state(job_id, tmp_path, "FAILED")
    assert not marker.exists()


def test_a_remote_node_is_not_silently_ignored(tmp_path):
    job_id, error = launch_pkg.launch(
        command="true",
        job_name="wrong-node",
        log_path=str(tmp_path / "watch.log"),
        script_path=tmp_path / "launch.sh",
        node="gpu-node-7",
        backend="local",
    )

    assert job_id == ""
    assert "cannot pin" in error


def test_an_unknown_backend_fails_before_starting_work(tmp_path):
    job_id, error = launch_pkg.launch(
        command="true",
        job_name="unknown-backend",
        log_path=str(tmp_path / "watch.log"),
        script_path=tmp_path / "launch.sh",
        backend="magic",
    )

    assert job_id == ""
    assert "expected auto, slurm, or local" in error
    assert not (tmp_path / "launch.sh").exists()


def test_local_environment_names_are_validated_before_launch(tmp_path):
    job_id, error = launch_pkg.launch(
        command="true",
        job_name="unsafe-env",
        log_path=str(tmp_path / "watch.log"),
        script_path=tmp_path / "launch.sh",
        env_vars={"X; touch /tmp/aorta-local-injection": "safe"},
        backend="local",
    )

    assert job_id == ""
    assert "invalid environment variable name" in error
    assert not (tmp_path / "launch.sh").exists()


def test_cancel_stops_the_whole_local_process_group(tmp_path):
    child_pid = tmp_path / "child.pid"
    job_id, error = launch_pkg.launch(
        command=f"sleep 30 & echo $! > {child_pid}; wait",
        job_name="local-cancel",
        log_path=str(tmp_path / "watch.log"),
        script_path=tmp_path / "launch.sh",
        backend="local",
    )
    assert error == ""

    deadline = time.monotonic() + 3
    while not child_pid.is_file() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert child_pid.is_file()

    cancelled, why = launch_pkg.cancel(job_id)

    assert cancelled, why
    _wait_for_state(job_id, tmp_path, "FAILED")
    pid = int(child_pid.read_text())
    deadline = time.monotonic() + 2
    while Path(f"/proc/{pid}").exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not Path(f"/proc/{pid}").exists()


def test_cancel_kills_a_child_that_ignores_term(tmp_path):
    child_pid = tmp_path / "stubborn-child.pid"
    child = (
        "trap '' TERM; " f"echo $$ > {shlex.quote(str(child_pid))}; " "while :; do sleep 1; done"
    )
    job_id, error = launch_pkg.launch(
        command=f"bash -c {shlex.quote(child)} & wait",
        job_name="local-stubborn-child",
        log_path=str(tmp_path / "watch.log"),
        script_path=tmp_path / "launch.sh",
        backend="local",
    )
    assert error == ""

    deadline = time.monotonic() + 3
    while not child_pid.is_file() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert child_pid.is_file()
    pid = int(child_pid.read_text())

    cancelled, why = cancel_local(job_id, timeout=0.2)

    assert cancelled, why
    deadline = time.monotonic() + 2
    while Path(f"/proc/{pid}").exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not Path(f"/proc/{pid}").exists()


def test_pid_reuse_is_never_signalled(monkeypatch):
    killed: list[tuple[int, int]] = []
    monkeypatch.setattr(
        "aorta.cia.launch.local._process_stat",
        lambda _pid: ("new-process-start", 123, 123, "R"),
    )
    monkeypatch.setattr(os, "kill", lambda pid, sig: killed.append((pid, sig)))

    cancelled, error = cancel_local("local:123:old-process-start")

    assert cancelled is False
    assert "reused" in error
    assert killed == []


def test_a_malformed_local_handle_is_not_sent_to_scancel(monkeypatch):
    monkeypatch.setattr(
        "aorta.cia.launch.cluster.cancel_sbatch",
        lambda _job_id: (_ for _ in ()).throw(AssertionError("scancel was used")),
    )

    cancelled, error = launch_pkg.cancel("local:not-a-process")

    assert cancelled is False
    assert "invalid local job id" in error


def test_forced_slurm_keeps_using_the_existing_submitter(tmp_path, monkeypatch):
    seen: dict = {}
    monkeypatch.setattr(
        "aorta.cia.launch.cluster.submit_sbatch",
        lambda **kwargs: seen.update(kwargs) or ("42", ""),
    )

    job_id, error = launch_pkg.launch(
        command="true",
        job_name="slurm",
        log_path=str(tmp_path / "watch.log"),
        script_path=tmp_path / "launch.sbatch",
        backend="slurm",
    )

    assert (job_id, error) == ("42", "")
    assert seen["command"] == "true"


def test_auto_keeps_slurm_when_sbatch_is_available(tmp_path, monkeypatch):
    monkeypatch.setattr("aorta.cia.launch.cluster.sbatch_available", lambda: True)
    monkeypatch.setattr(
        "aorta.cia.launch.cluster.submit_sbatch",
        lambda **kwargs: ("84", ""),
    )

    job_id, error = launch_pkg.launch(
        command="true",
        job_name="auto-slurm",
        log_path=str(tmp_path / "watch.log"),
        script_path=tmp_path / "launch.sbatch",
        backend="auto",
    )

    assert (job_id, error) == ("84", "")


def test_full_triage_pipeline_runs_locally_without_sbatch(tmp_path, monkeypatch):
    from aorta.cia import triage as triage_mod

    monkeypatch.setattr("aorta.cia.launch.cluster.sbatch_available", lambda: False)
    monkeypatch.setattr(triage_mod, "poll_jobs", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        triage_mod,
        "run_autopsy",
        lambda *args, **kwargs: {
            "category": "clean",
            "confidence": 1.0,
            "rationale": "local integration test",
            "evidence": [],
        },
    )

    result = triage_mod.run_triage(
        [
            "--command",
            "printf 'workstation path\\n'",
            "--jobs-root",
            str(tmp_path),
            "--job-backend",
            "auto",
            "--job-timeout",
            "5",
            "--watch-rounds",
            "1",
            "--watch-grace",
            "0",
        ]
    )

    assert result["ok"] is True
    assert result["scheduler"] == "local"
    assert result["scheduler_state"] == "COMPLETED"
    assert "slurm_job_id" not in result
    assert "workstation path" in Path(result["log_path"]).read_text(encoding="utf-8")
    record = read_job_json(tmp_path / result["job_id"] / "job.json")
    assert record.scheduler == "local"
    assert record.launcher == "aorta_direct"
    assert record.status == "completed"


def test_a_new_process_can_reconcile_a_finished_local_job(tmp_path):
    from aorta.cia.triage import reconcile_stale_jobs

    job_id = "cia-restart"
    job_dir = tmp_path / job_id
    native_id, error = launch_pkg.launch(
        command="true",
        job_name=job_id,
        log_path=str(job_dir / "watch.log"),
        script_path=job_dir / "launch.sh",
        backend="local",
    )
    assert error == ""
    _wait_for_state(native_id, job_dir, "COMPLETED")
    write_job_json(
        JobRecord(
            job_id=job_id,
            node="localhost",
            recipe="restart",
            launched_at=_utc_now(),
            log_path=str(job_dir / "watch.log"),
            aorta_output=str(job_dir / "bundle" / "aorta"),
            scheduler="local",
            launcher="aorta_direct",
            scheduler_job_id=native_id,
            status="running",
        ),
        tmp_path,
    )

    assert reconcile_stale_jobs(tmp_path) == 1
    assert read_job_json(job_dir / "job.json").status == "completed"


def test_local_reconciliation_never_queries_old_slurm_jobs(tmp_path, monkeypatch):
    from aorta.cia import triage as triage_mod

    write_job_json(
        JobRecord(
            job_id="old-slurm-job",
            node="node-1",
            recipe="old",
            launched_at=_utc_now(),
            log_path=str(tmp_path / "old.log"),
            aorta_output=str(tmp_path / "old-output"),
            scheduler="slurm",
            launcher="sbatch",
            scheduler_job_id="12345",
            status="running",
        ),
        tmp_path,
    )
    monkeypatch.setattr(
        triage_mod,
        "sacct_state",
        lambda _job_id: (_ for _ in ()).throw(AssertionError("Slurm was queried")),
    )

    assert triage_mod.reconcile_stale_jobs(tmp_path, "local") == 0


def test_cancelling_local_triage_stops_and_persists_the_process(tmp_path, monkeypatch):
    from aorta.cia import triage as triage_mod

    monkeypatch.setattr(triage_mod, "poll_jobs", lambda *args, **kwargs: None)
    stop = threading.Event()
    threading.Timer(0.15, stop.set).start()

    result = triage_mod.run_triage(
        [
            "--command",
            "sleep 30",
            "--jobs-root",
            str(tmp_path),
            "--job-backend",
            "local",
            "--job-timeout",
            "30",
            "--watch-rounds",
            "1",
        ],
        stop=stop,
    )

    assert result["ok"] is False
    assert result["stage"] == "wait"
    assert result["cancelled"] is True
    job_dir = tmp_path / result["job_id"]
    assert local_state(result["scheduler_job_id"], job_dir) == "CANCELLED"
    assert read_job_json(job_dir / "job.json").status == "cancelled"
