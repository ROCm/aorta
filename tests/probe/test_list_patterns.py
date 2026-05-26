"""Tests for ``aorta probe --list-patterns`` flag (FR 2.5).

The CLI deviation from the rubric: ``--list-patterns`` is a flag on the
``aorta probe`` command (not a ``list-patterns`` subcommand) so the
Phase-1 ``aorta probe -- <argv>`` invocation surface stays byte-equivalent.
See the PR description for the rationale.
"""

from __future__ import annotations

from click.testing import CliRunner

from aorta.cli.probe import probe
from aorta.probe.classifier import tier4_patterns
from aorta.probe.classifier.tier4_patterns import BUILTIN_PATTERN_VERSION


def test_list_patterns_exits_zero():
    runner = CliRunner()
    result = runner.invoke(probe, ["--list-patterns"])
    assert result.exit_code == 0, result.output


def test_version_without_list_patterns_is_rejected_with_targeted_message():
    """Regression for PR #197 round-3 review: ``--version`` is
    documented as meaningful only when paired with
    ``--list-patterns``. Used to silently fall through to the
    ``--recipe`` check, producing a confusing "Missing option
    '--recipe'" error. Now we reject up-front with a targeted
    usage message so the operator sees the intended pairing.
    """
    runner = CliRunner()
    result = runner.invoke(probe, ["--version"])
    assert result.exit_code != 0
    # click.UsageError formats messages on stderr-equivalent; in
    # CliRunner that lands in ``output`` by default.
    assert "--version is only meaningful with --list-patterns" in result.output
    # Confirm we did NOT fall through to the recipe complaint.
    assert "Missing option '--recipe'" not in result.output


def test_list_patterns_prints_every_detector_id():
    """Every Tier 4 detector ID is in the output (rubric §2.B FR 2.5)."""
    runner = CliRunner()
    result = runner.invoke(probe, ["--list-patterns"])
    assert result.exit_code == 0
    for pattern in tier4_patterns.all_patterns():
        assert pattern.detector_id in result.output


def test_list_patterns_prints_sample_regex():
    """Each entry includes its sample regex so an operator can grep visually."""
    runner = CliRunner()
    result = runner.invoke(probe, ["--list-patterns"])
    for pattern in tier4_patterns.all_patterns():
        assert pattern.regex.pattern in result.output


def test_list_patterns_version_flag_prints_version_banner():
    """`--list-patterns --version` prints the rubric-specified version line."""
    runner = CliRunner()
    result = runner.invoke(probe, ["--list-patterns", "--version"])
    assert result.exit_code == 0
    expected_fragment = f"aorta probe pattern library v{BUILTIN_PATTERN_VERSION}"
    assert expected_fragment in result.output
    assert "aorta " in result.output  # contains the aorta package version too


def test_list_patterns_short_circuits_recipe_load():
    """Without --recipe, --list-patterns must still exit 0 (no recipe required)."""
    runner = CliRunner()
    result = runner.invoke(probe, ["--list-patterns"])
    assert result.exit_code == 0


def test_list_patterns_no_double_dash_required():
    """The Phase 1 require-double-dash check is short-circuited for --list-patterns."""
    runner = CliRunner()
    result = runner.invoke(probe, ["--list-patterns"])
    assert result.exit_code == 0
    assert "double-dash" not in result.output.lower()


def test_list_patterns_does_not_write_to_disk(tmp_path, monkeypatch):
    """A `--list-patterns` invocation MUST NOT touch the filesystem."""
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(probe, ["--list-patterns"])
    assert result.exit_code == 0
    assert list(tmp_path.iterdir()) == []
