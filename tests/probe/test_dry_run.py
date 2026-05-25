"""Dry-run semantics for ``aorta probe`` (issue #188 Phase 1 FR 1.2)."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from aorta.cli.probe import probe

FIXTURES = Path(__file__).parent / "fixtures"


def test_dry_run_prints_cells_and_argv(tmp_path):
    """`aorta probe --dry-run -- echo hi` prints one line per cell + argv."""
    output = tmp_path / "out"
    runner = CliRunner()
    result = runner.invoke(
        probe,
        [
            "--recipe",
            str(FIXTURES / "probe_two_cell.yaml"),
            "--output",
            str(output),
            "--dry-run",
            "--",
            "echo",
            "hi",
        ],
    )
    assert result.exit_code == 0, result.output
    out = result.output
    # Both cells listed with their literal argv:
    assert "none-none" in out
    assert "tf32_off-none" in out
    assert "['echo', 'hi']" in out
    # Env-passthrough mode is shown:
    assert "inherit" in out.lower() or "inherit" in out
    # tf32_off sets DISABLE_TF32=1 -- the cell's env bundle must surface.
    assert "DISABLE_TF32" in out


def test_dry_run_does_not_write_disk(tmp_path):
    """Dry-run must not create the output directory."""
    output = tmp_path / "nope"
    runner = CliRunner()
    result = runner.invoke(
        probe,
        [
            "--recipe",
            str(FIXTURES / "probe_minimal.yaml"),
            "--output",
            str(output),
            "--dry-run",
            "--",
            "echo",
            "hi",
        ],
    )
    assert result.exit_code == 0
    assert not output.exists(), f"dry-run created {output}; expected no FS writes"
