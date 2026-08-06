from __future__ import annotations

import subprocess

import pytest

from aorta.instrumentation.rocjitsu_sanitizers.execution import run_argv


def test_run_argv_preserves_quoted_argument_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[tuple[str, ...]] = []

    def fake_run(
        argv: tuple[str, ...],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        captured.append(argv)
        return subprocess.CompletedProcess(argv, 0, "ok", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    command = ("python", "/path with spaces/model.py", "--label", "two words")

    result = run_argv(command, timeout_seconds=10)

    assert captured == [command]
    assert result.argv == command
    assert result.returncode == 0


def test_run_argv_normalizes_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(
        argv: tuple[str, ...],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(argv, 2, output=b"partial")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = run_argv(("app",), timeout_seconds=2)

    assert result.timed_out is True
    assert result.returncode is None
    assert result.stdout == "partial"


def test_run_argv_normalizes_launch_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(
        argv: tuple[str, ...],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError(argv[0])

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = run_argv(("missing-app",), timeout_seconds=2)

    assert result.returncode is None
    assert "FileNotFoundError" in str(result.launch_error)
