"""Every scheduler-specific decision is owned by the launch backend."""

from __future__ import annotations

from aorta.cia.launch import (
    LOCAL_BACKEND,
    SLURM_BACKEND,
    backend_for,
    resolve_backend,
)


def test_each_backend_exposes_the_complete_contract():
    attributes = (
        "name",
        "script_name",
        "launcher",
        "poll_interval",
        "wait_subject",
        "supports_remote_probe",
        "already_finished_reason",
    )
    methods = (
        "submit",
        "cancel",
        "state",
        "node",
        "record_node",
        "run_quick",
        "record_cancellation",
        "result_fields",
        "display_job_id",
        "launch_summary",
        "should_reconcile",
        "cancel_was_noop",
    )
    for backend in (LOCAL_BACKEND, SLURM_BACKEND):
        assert all(hasattr(backend, name) for name in attributes)
        assert all(callable(getattr(backend, name)) for name in methods)


def test_configured_names_resolve_to_backend_objects(monkeypatch):
    assert resolve_backend("local") is LOCAL_BACKEND
    assert resolve_backend("slurm") is SLURM_BACKEND

    monkeypatch.setattr(
        "aorta.cia.launch.cluster.sbatch_available",
        lambda: False,
    )
    assert resolve_backend("auto") is LOCAL_BACKEND


def test_native_handles_select_their_backend():
    assert backend_for("local:123:456:token") is LOCAL_BACKEND
    assert backend_for("12345") is SLURM_BACKEND


def test_backend_metadata_carries_the_old_scattered_choices():
    assert (LOCAL_BACKEND.script_name, LOCAL_BACKEND.launcher) == (
        "launch.sh",
        "aorta_direct",
    )
    assert (SLURM_BACKEND.script_name, SLURM_BACKEND.launcher) == (
        "launch.sbatch",
        "sbatch",
    )
    assert LOCAL_BACKEND.poll_interval == 0.2
    assert SLURM_BACKEND.poll_interval == 5.0
    assert LOCAL_BACKEND.supports_remote_probe is False
    assert SLURM_BACKEND.supports_remote_probe is True


def test_slurm_compatibility_fields_come_from_the_backend():
    assert SLURM_BACKEND.result_fields("123", "COMPLETED") == {
        "slurm_job_id": "123",
        "slurm_state": "COMPLETED",
    }
    assert LOCAL_BACKEND.result_fields("local:1:2:t", "COMPLETED") == {}


def test_local_backend_records_its_node_without_scheduler_queries(monkeypatch):
    monkeypatch.setattr("aorta.cia.launch.backend.socket.gethostname", lambda: "workstation")

    assert LOCAL_BACKEND.record_node("local:1:2:t", "") == "workstation"
    assert LOCAL_BACKEND.record_node("local:1:2:t", "requested") == "requested"
    assert SLURM_BACKEND.record_node("123", "node-1") == "node-1"
