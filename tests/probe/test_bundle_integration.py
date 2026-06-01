"""End-to-end ``aorta bundle`` + redaction integration (issue #188 Phase 3)."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from aorta.bundle import MANIFEST_FILENAME, Manifest, bundle_run_dir
from aorta.cli import main
from aorta.probe.redaction import RedactingRedactor, RedactionCfg


def _write_trial(cell_dir: Path, trial_idx: int = 0) -> None:
    trial = cell_dir / f"trial_{trial_idx}"
    trial.mkdir(parents=True, exist_ok=True)
    (trial / "stdout.log").write_text(
        "connect 192.168.1.1 from /home/user/secret/data\n",
        encoding="utf-8",
    )
    (trial / "stderr.log").write_text("", encoding="utf-8")
    (trial / "result.json").write_text(
        json.dumps(
            {
                "verdict": "pass",
                "exit_code": 0,
                "walltime_sec": 0.1,
                "argv": ["/home/user/secret/train.py"],
                "cell_name": cell_dir.name,
                "trial_index": trial_idx,
                "env": {"AWS_TOKEN": "secret", "HIP_VISIBLE_DEVICES": "0"},
                "env_passthrough_mode": "inherit",
                "timed_out": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (trial / "probe.env").write_text(
        "AWS_TOKEN=secret\nHIP_VISIBLE_DEVICES=0\n",
        encoding="utf-8",
    )


@pytest.fixture
def redaction_run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "probe-out" / "TKT-RED"
    run_dir.mkdir(parents=True)
    _write_trial(run_dir / "none-none")
    (run_dir / "host_env.json").write_text(
        json.dumps({"env": {"AWS_KEY": "x", "PATH": "/usr/bin"}}, indent=2),
        encoding="utf-8",
    )
    (run_dir / "recipe.resolved.yaml").write_text(
        """\
schema_version: 1
mode: probe
trials: 1
mitigation_axis: [none]
diagnostic_axis: [none]
redaction:
  scrub_env_keys: ["AWS_*"]
  scrub_paths: true
  scrub_ip_addresses: true
""",
        encoding="utf-8",
    )
    return run_dir


def test_refuses_no_ticket(tmp_path: Path):
    run_dir = tmp_path / "probe-out" / "_no_ticket_"
    run_dir.mkdir(parents=True)
    _write_trial(run_dir / "none-none")
    runner = CliRunner()
    result = runner.invoke(main, ["bundle", str(run_dir)])
    assert result.exit_code != 0
    assert "--ticket" in result.output


def test_review_pause_proceed(redaction_run_dir: Path, tmp_path: Path):
    runner = CliRunner()
    out = tmp_path / "bundle.tar.gz"
    result = runner.invoke(
        main,
        ["bundle", str(redaction_run_dir), "--review", "--output", str(out)],
        input="y\n",
    )
    assert result.exit_code == 0, result.output
    assert out.exists()


def test_review_pause_abort(redaction_run_dir: Path, tmp_path: Path):
    runner = CliRunner()
    out = tmp_path / "bundle.tar.gz"
    result = runner.invoke(
        main,
        ["bundle", str(redaction_run_dir), "--review", "--output", str(out)],
        input="n\n",
    )
    assert result.exit_code == 1
    assert not out.exists()


def test_manifest_records_per_file_counts(redaction_run_dir: Path, tmp_path: Path):
    runner = CliRunner()
    out = tmp_path / "bundle.tar.gz"
    result = runner.invoke(
        main,
        ["bundle", str(redaction_run_dir), "--output", str(out)],
    )
    assert result.exit_code == 0, result.output
    with tarfile.open(out, "r:gz") as tar:
        member = next(n for n in tar.getnames() if n.endswith(MANIFEST_FILENAME))
        manifest = Manifest.from_json(tar.extractfile(member).read().decode("utf-8"))
    assert manifest.redaction_applied is True
    assert manifest.redactor_kind == "probe.v1"
    stdout_row = next(f for f in manifest.files if f.path.endswith("stdout.log"))
    assert stdout_row.paths_rewritten >= 1
    assert stdout_row.ips_rewritten >= 1


def test_originals_untouched(redaction_run_dir: Path, tmp_path: Path):
    before = {
        p: p.read_bytes()
        for p in redaction_run_dir.rglob("*")
        if p.is_file()
    }
    out = tmp_path / "bundle.tar.gz"
    bundle_run_dir(
        redaction_run_dir,
        output=out,
        redactor=RedactingRedactor(
            RedactionCfg(
                scrub_env_keys=("AWS_*",),
                scrub_paths=True,
                scrub_ip_addresses=True,
            )
        ),
    )
    after = {
        p: p.read_bytes()
        for p in redaction_run_dir.rglob("*")
        if p.is_file()
    }
    assert before == after


def test_redaction_from_auto_fallback(redaction_run_dir: Path, tmp_path: Path):
    runner = CliRunner()
    out = tmp_path / "bundle.tar.gz"
    result = runner.invoke(
        main,
        ["bundle", str(redaction_run_dir), "--output", str(out)],
    )
    assert result.exit_code == 0, result.output
    with tarfile.open(out, "r:gz") as tar:
        names = tar.getnames()
        stdout_member = next(n for n in names if n.endswith("stdout.log"))
        text = tar.extractfile(stdout_member).read().decode("utf-8")
    assert "/home/user" not in text
    assert "192.168.1.1" not in text
