"""CLI tests for ``aorta bundle`` (issue #196).

These are the acceptance-criteria-1-through-3 tests from issue
#196: `--help` lists the flags, `<dir>` without `--ticket` (when
basename is `_no_ticket_`) exits non-zero, and `--review` pauses
for confirmation.
"""

from __future__ import annotations

import tarfile
from pathlib import Path

from click.testing import CliRunner

from aorta.bundle.manifest import MANIFEST_FILENAME
from aorta.cli.bundle import bundle


def test_help_lists_documented_flags():
    """Acceptance criterion 1: ``aorta bundle --help`` lists the flags."""
    runner = CliRunner()
    result = runner.invoke(bundle, ["--help"])
    assert result.exit_code == 0, result.output
    for flag in ("--ticket", "--review", "--output", "--redaction-from"):
        assert flag in result.output, f"--help missing {flag!r}: {result.output!r}"
    # And the positional argument is documented.
    assert "RUN_DIR" in result.output.upper() or "<RUN_DIR>" in result.output.upper()


def test_refuses_no_ticket_basename(no_ticket_run_dir):
    """Acceptance criterion 2: refuses without --ticket when source has none."""
    runner = CliRunner()
    result = runner.invoke(bundle, [str(no_ticket_run_dir)])
    assert result.exit_code != 0
    assert "_no_ticket_" in result.output or "no_ticket" in result.output
    assert "--ticket" in result.output


def test_runs_with_ticket_flag_against_no_ticket_source(no_ticket_run_dir, tmp_path):
    out = tmp_path / "out.tar.gz"
    runner = CliRunner()
    result = runner.invoke(
        bundle,
        [str(no_ticket_run_dir), "--ticket", "OVERRIDE-1", "--output", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert out.is_file()


def test_runs_against_synthetic_run_dir(synthetic_run_dir, tmp_path):
    out = tmp_path / "out.tar.gz"
    runner = CliRunner()
    result = runner.invoke(bundle, [str(synthetic_run_dir), "--output", str(out)])
    assert result.exit_code == 0, result.output
    assert out.is_file()
    # Final stdout line tells the operator where the bundle landed.
    assert str(out.resolve()) in result.output


def test_review_y_proceeds(synthetic_run_dir, tmp_path):
    """Acceptance criterion 3a: --review with 'y' produces the tarball."""
    out = tmp_path / "out.tar.gz"
    runner = CliRunner()
    result = runner.invoke(
        bundle,
        [str(synthetic_run_dir), "--review", "--output", str(out)],
        input="y\n",
    )
    assert result.exit_code == 0, result.output
    assert out.is_file()
    # The review summary actually surfaced before the prompt.
    assert "review pause" in result.output
    assert "ticket" in result.output
    assert "TKT-1" in result.output


def test_review_n_aborts_cleanly(synthetic_run_dir, tmp_path):
    """Acceptance criterion 3b: --review with 'n' aborts with exit 1, no tarball."""
    out = tmp_path / "out.tar.gz"
    runner = CliRunner()
    result = runner.invoke(
        bundle,
        [str(synthetic_run_dir), "--review", "--output", str(out)],
        input="n\n",
    )
    assert result.exit_code == 1, result.output
    assert "aborted" in result.output.lower()
    assert not out.exists()


def test_review_default_no_aborts(synthetic_run_dir, tmp_path):
    """Pressing Enter at the [y/N] prompt aborts (default=False)."""
    out = tmp_path / "out.tar.gz"
    runner = CliRunner()
    result = runner.invoke(
        bundle,
        [str(synthetic_run_dir), "--review", "--output", str(out)],
        input="\n",
    )
    assert result.exit_code == 1, result.output
    assert not out.exists()


def test_cli_run_dir_missing_returns_click_usage_error(tmp_path):
    """Click rejects nonexistent <run-dir> at parse time (exit 2)."""
    runner = CliRunner()
    result = runner.invoke(bundle, [str(tmp_path / "no-such-dir")])
    assert result.exit_code != 0


def test_cli_empty_run_dir_returns_friendly_error(empty_run_dir):
    runner = CliRunner()
    result = runner.invoke(bundle, [str(empty_run_dir)])
    assert result.exit_code != 0
    assert "no 'trial_*/result.json' artifacts" in result.output


def test_cli_emits_top_level_bundle_in_tar(synthetic_run_dir, tmp_path):
    """Sanity: the tar's top-level directory name embeds the ticket."""
    out = tmp_path / "out.tar.gz"
    runner = CliRunner()
    result = runner.invoke(bundle, [str(synthetic_run_dir), "--output", str(out)])
    assert result.exit_code == 0, result.output
    with tarfile.open(out, "r:gz") as tar:
        names = tar.getnames()
    top_levels = {Path(n).parts[0] for n in names}
    assert len(top_levels) == 1
    only = next(iter(top_levels))
    assert only.startswith("TKT-1-")
    # manifest.json sits directly under the top-level dir.
    assert f"{only}/{MANIFEST_FILENAME}" in names


def test_cli_registered_as_subcommand_on_main_group():
    """``aorta`` group exposes the bundle subcommand."""
    from aorta.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["bundle", "--help"])
    assert result.exit_code == 0, result.output
    assert "Package an" in result.output  # docstring's first sentence
