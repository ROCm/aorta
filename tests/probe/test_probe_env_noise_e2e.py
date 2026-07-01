"""End-to-end guard for the issue #220 minimal repro.

The issue's minimal repro is::

    aorta probe --recipe ... -- bash -c 'echo hi; sleep 100'
    # -> NUM_GPUS x "(null): No such file or directory" on the terminal

The real trigger is an in-process ``import torch`` -> HIP runtime ``dlopen``
on a multi-GPU ROCm host, which writes one ``(null)`` line per GPU straight to
fd 2 from inside the env probe that ``aorta probe`` runs at start. That trigger
needs a GPU host, so here we reproduce its *shape* on any host: we drive the
real ``aorta probe`` CLI -> ``run_recipe`` -> ``_capture_env`` -> ``collect_env``
path wrapping a trivial command, and inject synthetic raw ``write(2, ...)``
noise from inside the env probe. The fix (``_ProbeStdioRedirect``) must keep
that noise off the operator's terminal (fd 2), which ``contextlib`` /
``CliRunner`` capture cannot do because they only swap the Python ``sys.stderr``
object and never move fd 2.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from click.testing import CliRunner

import aorta.instrumentation.environment as env_mod
from aorta.cli.probe import probe

FIXTURES = Path(__file__).parent / "fixtures"

_NULL_LINE = b"(null): No such file or directory\n"


def _with_outer_terminal(tmp_path: Path):
    """Install a temp file on the real fds 1/2; return (path, restore()).

    Stands in for the operator's terminal: anything written straight to fd 1/2
    (bypassing Python-level ``sys.stdout`` / ``sys.stderr``) lands here. The
    fix must keep the probe noise out of this file.
    """
    outer = tmp_path / "outer_terminal.txt"
    sys.stdout.flush()
    sys.stderr.flush()
    outer_fd = os.open(str(outer), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    saved1, saved2 = os.dup(1), os.dup(2)
    os.dup2(outer_fd, 1)
    os.dup2(outer_fd, 2)

    def restore():
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved1, 1)
        os.dup2(saved2, 2)
        os.close(saved1)
        os.close(saved2)
        os.close(outer_fd)

    return outer, restore


def test_aorta_probe_does_not_leak_env_probe_noise_to_terminal(tmp_path, monkeypatch):
    """`aorta probe -- bash -c 'echo hi'` must not print `(null)` to fd 2.

    Mirrors the issue #220 minimal repro at the CLI/runner level (synthetic
    noise, since the real torch/ROCm trigger needs a GPU host).
    """
    # Inject the noise from inside the env probe body, exactly where the real
    # HIP dlopen noise originates -- the first probe call after the fd capture
    # context is entered in collect_env(). Counting calls proves the noisy
    # path actually ran, so a clean terminal means the noise was *captured*,
    # not merely never produced.
    original_detect = env_mod._detect_runtime_context
    calls = {"n": 0}

    def noisy_detect():
        calls["n"] += 1
        for _ in range(8):  # one (null) line per GPU on an 8-GPU host
            os.write(2, _NULL_LINE)
        return original_detect()

    monkeypatch.setattr(env_mod, "_detect_runtime_context", noisy_detect)

    outer, restore = _with_outer_terminal(tmp_path)
    try:
        result = CliRunner().invoke(
            probe,
            [
                "--recipe",
                str(FIXTURES / "probe_minimal.yaml"),
                "--output",
                str(tmp_path / "out"),
                "--",
                "bash",
                "-c",
                "echo hi",
            ],
        )
    finally:
        restore()

    assert result.exit_code == 0, result.output
    assert calls["n"] >= 1, "env probe (the noise source) never ran"
    terminal_text = outer.read_text()
    assert "(null): No such file or directory" not in terminal_text, (
        "env-probe noise leaked to the operator's terminal (fd 2):\n"
        f"{terminal_text!r}"
    )
