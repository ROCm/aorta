"""Chat presents rocjitsu output as advisory evidence, never as Autopsy."""

from __future__ import annotations

import json

from aorta.chat.config import settings
from aorta.chat.tools import cluster


def test_formatter_keeps_expert_and_autopsy_results_separate(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "jobs_path", str(tmp_path))
    job = tmp_path / "cia-demo"
    expert = job / "rj_expert"
    expert.mkdir(parents=True)
    (expert / "report.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "live_job_input_analyzed": True,
                "answer": {"class": "barrier", "kernel": "kernel_a", "offset": "0x40"},
            }
        )
    )
    (expert / "shadow_comparison.json").write_text(
        json.dumps(
            {
                "status": "confirmed_by_consan",
                "rationale": "ConSan independently observed a matching race",
            }
        )
    )

    lines = cluster._fmt_rocjitsu_expert({"job_dir": str(job)})

    assert lines[0] == "rocjitsu expert (advisory; separate from Autopsy):"
    assert any("class=barrier" in line and "offset=0x40" in line for line in lines)
    assert any("verification: confirmed_by_consan" in line for line in lines)


def test_archive_demo_is_labelled_as_not_live(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "jobs_path", str(tmp_path))
    job = tmp_path / "cia-demo"
    expert = job / "rj_expert"
    expert.mkdir(parents=True)
    (expert / "report.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "live_job_input_analyzed": False,
                "answer": {"class": "waitcnt", "kernel": "k000", "offset": "0x0"},
            }
        )
    )

    rendered = "\n".join(cluster._fmt_rocjitsu_expert({"job_dir": str(job)}))

    assert "recorded archive demo" in rendered
    assert "CIA job input was not analyzed" in rendered


def test_bounded_reader_uses_a_separate_tool(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "jobs_path", str(tmp_path))
    report = tmp_path / "cia-demo" / "rj_expert" / "report.json"
    report.parent.mkdir(parents=True)
    report.write_text('{"status":"complete"}\n')

    rendered = cluster.read_rocjitsu_expert_report.invoke({"job_id": "cia-demo"})

    assert "rocjitsu expert report" in rendered
    assert "Autopsy report" not in rendered
