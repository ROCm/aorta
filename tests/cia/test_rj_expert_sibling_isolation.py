"""Archive demos are stored beside CIA evidence and never enter Autopsy."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml

from aorta.cia.autopsy.orchestrator import run_autopsy
from aorta.rocjitsu.cia_report import attach_archive_run
from aorta.rocjitsu.runtime import (
    ArchiveOperation,
    ArchiveRunConfig,
    ArchiveRunResult,
)


def _run(tmp_path: Path) -> ArchiveRunResult:
    return ArchiveRunResult(
        config=ArchiveRunConfig(tmp_path / "archive", "ran@toy"),
        status="complete",
        answer={"class": "waitcnt", "kernel": "k000", "offset": "0x0"},
        outcome="exact",
        score=5.45,
        operations=(
            ArchiveOperation(
                step=1,
                seed=7,
                prompt="prompt",
                reply='{"tool":"waitcheck"}',
                kind="tool",
                tool="waitcheck",
                kernel="",
                parse_error="",
                machine_result=[],
                rendered_result="waitcheck: no diagnostics.",
                prompt_tokens=3,
                completion_tokens=2,
                latency_s=0.1,
            ),
        ),
        system_prompt_sha256="a" * 64,
        scenario_digest="b" * 64,
        endpoint="http://127.0.0.1:8000/v1",
        model="rjx",
        server={"flavor": "generic"},
        seconds=0.1,
    )


def _job(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "jobs"
    job = root / "cia-demo"
    bundle = job / "bundle"
    (bundle / "logs").mkdir(parents=True)
    (job / "job.json").write_text('{"job_id":"cia-demo","status":"failed"}\n')
    (bundle / "logs" / "watch.stderr.log").write_text("kernel failed\n")
    (bundle / "manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "0.1",
                "job_id": "cia-demo",
                "paths": {"stderr": "logs/watch.stderr.log"},
            }
        )
    )
    (bundle / "report.json").write_text('{"category":"unknown"}\n')
    return root, job


def test_attachment_is_storage_only_and_does_not_change_autopsy(tmp_path: Path) -> None:
    root, job = _job(tmp_path)
    bundle = job / "bundle"
    manifest_before = (bundle / "manifest.yaml").read_bytes()
    report_before = (bundle / "report.json").read_bytes()
    job_before = (job / "job.json").read_bytes()
    autopsy_before = run_autopsy(bundle, use_llm=False)

    path = attach_archive_run(root, "cia-demo", _run(tmp_path))

    assert path == job / "rj_expert" / "report.json"
    document = json.loads(path.read_text())
    assert document["cia_job_binding"] == "storage_only"
    assert document["live_job_input_analyzed"] is False
    assert document["affects_autopsy"] is False
    assert "api_key" not in path.read_text()
    assert (bundle / "manifest.yaml").read_bytes() == manifest_before
    assert (bundle / "report.json").read_bytes() == report_before
    assert (job / "job.json").read_bytes() == job_before

    autopsy_after = run_autopsy(bundle, use_llm=False)
    for key in ("category", "confidence", "rationale", "evidence", "tooling_gaps"):
        assert autopsy_after[key] == autopsy_before[key]
    assert "rj_expert" not in json.dumps(autopsy_after)


def test_existing_expert_report_is_not_overwritten(tmp_path: Path) -> None:
    root, _job_dir = _job(tmp_path)
    attach_archive_run(root, "cia-demo", _run(tmp_path))

    with pytest.raises(FileExistsError):
        attach_archive_run(root, "cia-demo", _run(tmp_path))


def test_running_job_cannot_receive_an_archive_attachment(tmp_path: Path) -> None:
    root, job = _job(tmp_path)
    (job / "job.json").write_text('{"job_id":"cia-demo","status":"running"}\n')

    with pytest.raises(ValueError, match="not terminal"):
        attach_archive_run(root, "cia-demo", _run(tmp_path))


@pytest.mark.skipif(os.name == "nt", reason="symlink creation requires privileges on Windows")
def test_symlinked_job_id_cannot_escape_jobs_root(tmp_path: Path) -> None:
    root = tmp_path / "jobs"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "job.json").write_text("{}")
    (root / "escape").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="must not be a symlink"):
        attach_archive_run(root, "escape", _run(tmp_path))


@pytest.mark.skipif(os.name == "nt", reason="symlink creation requires privileges on Windows")
def test_symlinked_expert_directory_cannot_escape_job(tmp_path: Path) -> None:
    root, job = _job(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (job / "rj_expert").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="must not be a symlink"):
        attach_archive_run(root, "cia-demo", _run(tmp_path))
