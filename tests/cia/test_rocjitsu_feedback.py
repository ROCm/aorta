"""Only verified, immutable outcomes can become offline training data."""

from __future__ import annotations

import json

import pytest

from aorta.rocjitsu.feedback import FeedbackOutcome, export_verified, record_feedback


def _job(tmp_path, job_id="cia-demo"):
    job = tmp_path / "jobs" / job_id
    expert = job / "rj_expert"
    expert.mkdir(parents=True)
    (job / "job.json").write_text(
        json.dumps({"job_id": job_id, "status": "completed"})
    )
    (expert / "report.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "answer": {"class": "waitcnt", "kernel": "k000", "offset": "0x0"},
                "operations": "operations.jsonl",
            }
        )
    )
    (expert / "operations.jsonl").write_text('{"step":1}\n')
    return job


def test_unverified_feedback_is_not_exported(tmp_path) -> None:
    job = _job(tmp_path)
    record_feedback(
        job.parent,
        job.name,
        FeedbackOutcome.NOT_FIXED,
        notes="reported but not reproduced",
    )

    output = tmp_path / "export.jsonl"
    assert export_verified(job.parent, output, include_operations=True) == 0
    assert output.read_text() == ""


def test_verified_feedback_exports_operations(tmp_path) -> None:
    job = _job(tmp_path)
    record_feedback(
        job.parent,
        job.name,
        FeedbackOutcome.FIXED,
        verified=True,
        verified_by="reproducer rerun",
    )

    output = tmp_path / "export.jsonl"
    assert export_verified(job.parent, output, include_operations=True) == 1
    exported = json.loads(output.read_text())
    assert exported["feedback"]["outcome"] == "fixed"
    assert exported["operations"] == [{"step": 1}]


def test_verified_feedback_requires_a_verifier(tmp_path) -> None:
    job = _job(tmp_path)

    with pytest.raises(ValueError, match="verified_by"):
        record_feedback(
            job.parent,
            job.name,
            FeedbackOutcome.FIXED,
            verified=True,
        )


def test_feedback_is_immutable(tmp_path) -> None:
    job = _job(tmp_path)
    record_feedback(job.parent, job.name, FeedbackOutcome.NOT_TESTED)

    with pytest.raises(FileExistsError):
        record_feedback(job.parent, job.name, FeedbackOutcome.FIXED)


def test_changed_expert_report_is_refused(tmp_path) -> None:
    job = _job(tmp_path)
    record_feedback(
        job.parent,
        job.name,
        FeedbackOutcome.FIXED,
        verified=True,
        verified_by="engineer review",
    )
    (job / "rj_expert" / "report.json").write_text('{"changed":true}\n')

    with pytest.raises(ValueError, match="changed after feedback"):
        export_verified(job.parent, tmp_path / "export.jsonl")
