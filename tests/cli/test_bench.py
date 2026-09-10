"""CLI tests for the ``aorta bench`` shim group.

Tests the shim contract only — not hw_queue_eval internals:
- ``aorta bench --help`` lists hw_queue_eval as a subcommand.
- ``aorta bench hw_queue_eval --help`` exposes the same commands that
  hw_queue_eval's own CLI group registers (derived at runtime, not hardcoded).
- When hw_queue_eval is unavailable, bare invoke exits non-zero with a clear
  install hint; ``--help`` exits 0 but shows the hint instead of empty help.
- Only a missing *external* dependency is read as "extra not installed": a
  missing aorta.hw_queue_eval sub-module, or any other ImportError, propagates
  — from the group and from this module's own availability probe alike.
- The resolved group is named ``hw_queue_eval``, the key it is registered
  under, not ``cli``, the name it is defined as.
- ``bench`` is correctly wired under the top-level ``aorta`` CLI.
"""

from __future__ import annotations

import importlib
import types

import click
import pytest
from click.testing import CliRunner

from aorta.cli import main
from aorta.cli.bench import bench


def _hw_queue_available() -> bool:
    """Report whether the [hw-queue] extra is installed.

    Availability is determined by actually attempting the import, not
    find_spec: aorta.hw_queue_eval files are always present in the source tree,
    but the import fails on a base install because hw_queue_eval.__init__
    pulls torch.

    Mirrors _BenchGroup.get_command's distinction: only an absent *external*
    dependency means "extra not installed". A missing aorta.hw_queue_eval
    sub-module is our own bug, and reading it as "extra absent" would skip the
    real-group tests below instead of failing them.
    """
    try:
        importlib.import_module("aorta.hw_queue_eval.cli")
    except ModuleNotFoundError as exc:
        if exc.name is not None and exc.name.startswith("aorta.hw_queue_eval"):
            raise
        return False
    return True


_HW_QUEUE_AVAILABLE = _hw_queue_available()


def _raise_on_import(monkeypatch: pytest.MonkeyPatch, exc: Exception) -> None:
    """Make importlib.import_module raise ``exc`` for the availability probe."""

    def raise_it(_name: str) -> None:
        raise exc

    monkeypatch.setattr(importlib, "import_module", raise_it)


def test_availability_probe_reads_an_absent_external_dep_as_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An absent third-party dep is what "extra not installed" means."""
    _raise_on_import(monkeypatch, ModuleNotFoundError("No module named 'torch'", name="torch"))
    assert _hw_queue_available() is False


def test_availability_probe_does_not_mask_an_internal_import_bug(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A broken aorta.hw_queue_eval sub-module must not read as "extra absent".

    Reporting it that way skips every real-group test in this module, so a
    packaging regression inside the subpackage would land green.
    """
    _raise_on_import(
        monkeypatch,
        ModuleNotFoundError(
            "No module named 'aorta.hw_queue_eval.sweep'",
            name="aorta.hw_queue_eval.sweep",
        ),
    )
    with pytest.raises(ModuleNotFoundError, match="aorta.hw_queue_eval.sweep"):
        _hw_queue_available()


def test_bench_help_lists_hw_queue_eval() -> None:
    """``aorta bench --help`` exits 0 and lists hw_queue_eval with its short help.

    The row's help text is the behaviour change the registry bought: the
    deleted proxy carried no help, so the row rendered blank. Sourced from the
    registry rather than hardcoded so the two cannot drift apart.
    """
    result = CliRunner().invoke(bench, ["--help"])
    assert result.exit_code == 0, result.output
    assert "hw_queue_eval" in result.output
    row = next(
        (line for line in result.output.splitlines() if line.strip().startswith("hw_queue_eval")),
        None,
    )
    assert row is not None, result.output
    rendered = row.split("hw_queue_eval", 1)[1].strip()
    assert rendered, f"help row is blank: {row!r}"
    # Tolerate Click's width-dependent "..." truncation instead of pinning a width.
    assert bench.lazy_commands["hw_queue_eval"].help.startswith(
        rendered.removesuffix("...").rstrip()
    ), rendered


def test_bench_wired_under_main() -> None:
    """``aorta bench --help`` via the top-level CLI exits 0 (validates __init__.py wiring)."""
    result = CliRunner().invoke(main, ["bench", "--help"])
    assert result.exit_code == 0, result.output
    assert "hw_queue_eval" in result.output


@pytest.mark.skipif(not _HW_QUEUE_AVAILABLE, reason="amd-aorta[hw-queue] not installed")
def test_hw_queue_eval_help_lists_subcommands() -> None:
    """``aorta bench hw_queue_eval --help`` lists every registered subcommand."""
    from aorta.hw_queue_eval.cli import cli as hw_queue_eval_cli

    expected = set(hw_queue_eval_cli.commands)
    result = CliRunner().invoke(bench, ["hw_queue_eval", "--help"])
    assert result.exit_code == 0, result.output
    for sub in expected:
        assert sub in result.output, f"--help missing {sub!r}: {result.output!r}"


@pytest.mark.skipif(_HW_QUEUE_AVAILABLE, reason="hw_queue_eval is installed — error path not active")
def test_hw_queue_unavailable_invoke_shows_install_hint() -> None:
    """Bare invoke when hw_queue_eval is missing exits non-zero with install hint."""
    result = CliRunner().invoke(bench, ["hw_queue_eval"])
    assert result.exit_code != 0
    assert "amd-aorta[hw-queue]" in result.output


@pytest.mark.skipif(_HW_QUEUE_AVAILABLE, reason="hw_queue_eval is installed — error path not active")
def test_hw_queue_unavailable_help_shows_install_hint() -> None:
    """``--help`` when hw_queue_eval is missing shows install hint.

    Click always exits 0 for --help; the hint text is the signal, not the
    exit code.
    """
    result = CliRunner().invoke(bench, ["hw_queue_eval", "--help"])
    assert result.exit_code == 0, result.output
    assert "amd-aorta[hw-queue]" in result.output


def _fail_import(monkeypatch: pytest.MonkeyPatch, exc: Exception) -> None:
    """Make resolving a lazy command raise ``exc``, whatever is installed locally."""

    def raise_it(_name: str) -> None:
        raise exc

    monkeypatch.setattr("aorta.cli._lazy_group.import_module", raise_it)


@pytest.mark.parametrize(
    "argv",
    [
        pytest.param(["hw_queue_eval"], id="bare"),
        pytest.param(["hw_queue_eval", "sweep"], id="trailing-subcommand"),
        pytest.param(["hw_queue_eval", "sweep", "--iters", "3"], id="trailing-subcommand-options"),
        pytest.param(["hw_queue_eval", "--bogus-flag"], id="trailing-unknown-option"),
    ],
)
def test_missing_external_dependency_becomes_an_install_hint(
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
) -> None:
    """An absent third-party dep means the extra is not installed: show the hint.

    The trailing-argument cases pin the stand-in's ``ignore_unknown_options``
    plus variadic ``UNPROCESSED`` argument: without them Click rejects the
    extra tokens with its own usage error and the user never sees the hint.
    """
    _fail_import(monkeypatch, ModuleNotFoundError("No module named 'torch'", name="torch"))
    result = CliRunner().invoke(bench, argv)
    assert result.exit_code != 0
    assert "amd-aorta[hw-queue]" in result.output


@pytest.mark.parametrize(
    "argv",
    [
        pytest.param(["hw_queue_eval", "--help"], id="bare"),
        pytest.param(["hw_queue_eval", "sweep", "--help"], id="trailing-subcommand"),
    ],
)
def test_missing_external_dependency_help_shows_the_install_hint(
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
) -> None:
    """``--help`` on the stand-in exits 0 and renders the hint as the help body.

    The environment-gated test above only covers this on a base install; this
    one stubs the import so the path stays covered wherever the suite runs.
    """
    _fail_import(monkeypatch, ModuleNotFoundError("No module named 'torch'", name="torch"))
    result = CliRunner().invoke(bench, argv)
    assert result.exit_code == 0, result.output
    assert "amd-aorta[hw-queue]" in result.output
    assert "Error" not in result.output


def test_missing_internal_module_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing aorta.hw_queue_eval sub-module is our bug, not a missing extra.

    Reporting it as "install the extra" would send users down a dead end, so it
    has to surface as the ModuleNotFoundError it is.
    """
    broken = ModuleNotFoundError(
        "No module named 'aorta.hw_queue_eval.sweep'",
        name="aorta.hw_queue_eval.sweep",
    )
    _fail_import(monkeypatch, broken)
    result = CliRunner().invoke(bench, ["hw_queue_eval"])
    assert isinstance(result.exception, ModuleNotFoundError)
    assert result.exception.name == "aorta.hw_queue_eval.sweep"


def test_other_import_errors_propagate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Anything that is not a ModuleNotFoundError is a real bug and must surface."""
    _fail_import(monkeypatch, ImportError("cannot import name 'cli'"))
    result = CliRunner().invoke(bench, ["hw_queue_eval"])
    assert isinstance(result.exception, ImportError)


def test_hw_queue_group_is_resolved_when_the_extra_is_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the import succeeding, ``bench`` hands Click the real inner group.

    Stubbed rather than skipped so the happy path stays covered on the base
    install, where torch (and so hw_queue_eval) is absent. The stub is named
    ``cli`` because the real group is (``aorta.hw_queue_eval.cli:cli``); a stub
    pre-named ``hw_queue_eval`` would not exercise the rename below.
    """
    stub = _stub_hw_queue_group(monkeypatch)
    result = CliRunner().invoke(bench, ["hw_queue_eval", "--help"])
    assert result.exit_code == 0, result.output
    assert "sweep" in result.output
    assert stub.name == "cli", "the loaded module's own group must not be renamed in place"


def _stub_hw_queue_group(monkeypatch: pytest.MonkeyPatch) -> click.Group:
    """Stand in for ``aorta.hw_queue_eval.cli:cli``, named as that group really is."""

    @click.group(name="cli")
    def stub() -> None:
        """Stand-in hw_queue_eval group."""

    @stub.command()
    def sweep() -> None:
        """Stand-in subcommand."""

    monkeypatch.setattr(
        "aorta.cli._lazy_group.import_module",
        lambda _name: types.SimpleNamespace(cli=stub),
    )
    return stub


def test_resolved_group_carries_the_registry_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """The group handed to Click must be named ``hw_queue_eval``, not ``cli``.

    Click 8.0.0 builds the child context from ``cmd.name`` rather than the
    invoked name, so a mismatch renders as ``Usage: aorta bench cli ...`` --
    a command line the user cannot type. ``click>=8.0.0`` still resolves 8.0.0,
    and the deleted proxy was explicitly named ``hw_queue_eval``, so this is
    the invariant that kept the rename invisible.
    """
    _stub_hw_queue_group(monkeypatch)
    resolved = bench.get_command(click.Context(bench), "hw_queue_eval")
    assert resolved is not None
    assert resolved.name == "hw_queue_eval"


def test_usage_line_uses_the_registry_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """``--help`` on the resolved group advertises the invocable path."""
    _stub_hw_queue_group(monkeypatch)
    result = CliRunner().invoke(bench, ["hw_queue_eval", "--help"])
    assert result.exit_code == 0, result.output
    usage = result.output.splitlines()[0]
    assert "hw_queue_eval" in usage, usage
    assert " cli " not in usage, usage
