"""Only independent, kernel-matched ConSan evidence confirms an expert claim."""

from __future__ import annotations

import json

from aorta.rocjitsu.comparison import (
    ComparisonStatus,
    ShadowComparison,
    compare_job,
    compare_reports,
    summarize,
)


def _expert(*, cls="barrier", kernel="kernel_a", live=True, status="complete"):
    return {
        "live_job_input_analyzed": live,
        "status": status,
        "answer": {"class": cls, "kernel": kernel, "offset": "0x40"},
    }


def _consan(*, kernel="kernel_a", finding=True):
    findings = (
        [
            {
                "sanitizer": "consan",
                "severity": "race",
                "code": "sampled_conflict",
                "message": "race",
                "kernel_name": kernel,
                "metadata": {},
            }
        ]
        if finding
        else []
    )
    return {
        "checks": [
            {
                "sanitizer": "consan",
                "state": "ran",
                "verdict": "fail" if finding else "pass",
                "findings": findings,
            }
        ]
    }


def test_matching_consan_finding_confirms_the_expert() -> None:
    compared = compare_reports(_expert(), _consan())

    assert compared.status is ComparisonStatus.CONFIRMED_BY_CONSAN


def test_clean_consan_does_not_disprove_the_expert() -> None:
    compared = compare_reports(_expert(), _consan(finding=False))

    assert compared.status is ComparisonStatus.NOT_CONFIRMED_BY_CONSAN


def test_unattributed_consan_finding_cannot_confirm() -> None:
    compared = compare_reports(_expert(), _consan(kernel=None))

    assert compared.status is ComparisonStatus.NOT_CONFIRMED_BY_CONSAN


def test_consan_finding_exposes_an_expert_miss() -> None:
    compared = compare_reports(_expert(cls="none", kernel=""), _consan())

    assert compared.status is ComparisonStatus.CONSAN_FINDING_EXPERT_NONE


def test_archive_demo_is_not_a_live_comparison() -> None:
    compared = compare_reports(_expert(live=False), _consan())

    assert compared.status is ComparisonStatus.NOT_APPLICABLE


def test_comparison_is_written_beside_not_inside_autopsy(tmp_path) -> None:
    job = tmp_path / "job"
    expert_dir = job / "rj_expert"
    sanitizer_dir = job / "bundle" / "aorta"
    expert_dir.mkdir(parents=True)
    sanitizer_dir.mkdir(parents=True)
    (expert_dir / "report.json").write_text(json.dumps(_expert()))
    (sanitizer_dir / "sanitizer_report.json").write_text(json.dumps(_consan()))

    output = compare_job(job)

    assert output == expert_dir / "shadow_comparison.json"
    assert not (job / "bundle" / "shadow_comparison.json").exists()


def test_summary_counts_statuses() -> None:
    summary = summarize(
        [
            ShadowComparison(ComparisonStatus.CONFIRMED_BY_CONSAN),
            ShadowComparison(ComparisonStatus.NO_CONSAN),
            ShadowComparison(ComparisonStatus.NO_CONSAN),
        ]
    )

    assert summary == {
        "total": 3,
        "statuses": {"confirmed_by_consan": 1, "no_consan": 2},
    }
