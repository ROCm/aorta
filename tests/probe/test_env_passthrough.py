"""Env-passthrough mode tests for ``aorta probe`` (FR 1.9 / 1.10)."""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from aorta.workloads._subprocess import (
    CONFIG_KEY_LOG_PREFIX,
    CONFIG_KEY_PROBE_EXTRAS,
    CONFIG_KEY_SUBPROCESS_ARGV,
    SubprocessWorkload,
)


def _make_workload(
    tmp_path: Path,
    argv: list[str],
    env_mode: str,
    cell_env_vars: dict[str, str] | None = None,
) -> SubprocessWorkload:
    workload_subdir = tmp_path / "_subprocess"
    workload_subdir.mkdir(parents=True, exist_ok=True)
    prefix = workload_subdir / "trial_d0_m0_t0"
    cfg = {
        CONFIG_KEY_SUBPROCESS_ARGV: argv,
        CONFIG_KEY_LOG_PREFIX: str(prefix),
        CONFIG_KEY_PROBE_EXTRAS: {
            "cell_name": "test-cell",
            "env_passthrough_mode": env_mode,
            "timeout_per_trial": None,
            "cell_env_vars": dict(cell_env_vars or {}),
        },
    }
    return SubprocessWorkload(cfg)


# ---- FR 1.9 (inherit mode does NOT write env file) -----------------------


def test_inherit_mode_does_not_write_env_file(tmp_path):
    """`inherit` mode runs the subprocess without dropping a probe.env."""
    wl = _make_workload(
        tmp_path,
        argv=["sh", "-c", "echo hi"],
        env_mode="inherit",
        cell_env_vars={"DISABLE_TF32": "1"},
    )
    wl.setup()
    wl.run()
    trial_dir = tmp_path / "trial_0"
    assert (trial_dir / "stdout.log").is_file()
    assert (trial_dir / "result.json").is_file()
    assert not (trial_dir / "probe.env").exists(), "inherit mode must NOT write probe.env"


# ---- FR 1.10 (file mode writes env file + exports AORTA_ENV_FILE) --------


def test_file_mode_writes_env_file_and_exports_pointer(tmp_path):
    """`file` mode writes probe.env (chmod 0600) and exports AORTA_ENV_FILE."""
    wl = _make_workload(
        tmp_path,
        argv=[
            "sh",
            "-c",
            'test -f "$AORTA_ENV_FILE" && cat "$AORTA_ENV_FILE"',
        ],
        env_mode="file",
        cell_env_vars={"DISABLE_TF32": "1", "HSA_XNACK": "1"},
    )
    wl.setup()
    wl.run()
    trial_dir = tmp_path / "trial_0"
    env_path = trial_dir / "probe.env"
    assert env_path.is_file(), "file mode must write probe.env"
    text = env_path.read_text(encoding="utf-8")
    # Sorted keys (deterministic) -> DISABLE_TF32 first, HSA_XNACK second.
    assert "DISABLE_TF32=1" in text
    assert "HSA_XNACK=1" in text
    # The subprocess saw AORTA_ENV_FILE and was able to cat it.
    stdout = (trial_dir / "stdout.log").read_text(encoding="utf-8")
    assert "DISABLE_TF32=1" in stdout
    assert "HSA_XNACK=1" in stdout


def test_env_file_is_0600(tmp_path):
    """probe.env must be chmod 0600 to keep secrets off the public read bit."""
    wl = _make_workload(
        tmp_path,
        argv=["true"],
        env_mode="file",
        cell_env_vars={"SECRET_TOKEN": "supersecret"},
    )
    wl.setup()
    wl.run()
    env_path = tmp_path / "trial_0" / "probe.env"
    assert env_path.is_file()
    mode = env_path.stat().st_mode
    perms = stat.S_IMODE(mode)
    assert perms == 0o600, f"expected 0600, got 0o{perms:o}"


def test_env_file_rejects_newline_in_value(tmp_path):
    """Newline in env value would corrupt the KEY=VALUE shape -- reject up-front."""
    wl = _make_workload(
        tmp_path,
        argv=["true"],
        env_mode="file",
        cell_env_vars={"MULTILINE": "line1\nline2"},
    )
    wl.setup()
    with pytest.raises(ValueError, match="newline"):
        wl.run()


@pytest.mark.parametrize(
    "bad_key",
    [
        "FOO\nBAR",       # newline in key
        "FOO\rBAR",       # carriage return in key
        "FOO=BAR",        # embedded '=' would rebind a later key
        "X\n=injected",   # newline-then-equals row-injection attempt
    ],
)
def test_env_file_rejects_unsafe_chars_in_key(tmp_path, bad_key):
    """Regression for PR #194 review: hostile mitigation sidecars
    could smuggle ``\\n``, ``\\r``, or ``=`` into env *keys* to
    inject extra KEY=VALUE rows or rebind a later key. The
    sidecar loader only enforces ``isinstance(key, str)``, so the
    env-file writer is the right place to catch this -- it is the
    layer that owns the on-disk KEY=VALUE row shape.
    """
    wl = _make_workload(
        tmp_path,
        argv=["true"],
        env_mode="file",
        cell_env_vars={bad_key: "safe-value"},
    )
    wl.setup()
    with pytest.raises(ValueError, match="env key"):
        wl.run()


def test_inherit_mode_passes_env_to_child(tmp_path, monkeypatch):
    """`inherit` mode's Popen env=os.environ.copy() snapshot includes our key.

    The runner stamps the cell's env vars on ``os.environ`` before
    calling ``run()``; the workload's Popen uses ``env=os.environ.copy()``
    so the child sees those keys. Simulate the runner overlay by
    setting the var via monkeypatch.
    """
    monkeypatch.setenv("PROBE_TEST_VAR", "from_inherit")
    wl = _make_workload(
        tmp_path,
        argv=["sh", "-c", "echo $PROBE_TEST_VAR"],
        env_mode="inherit",
    )
    wl.setup()
    wl.run()
    stdout = (tmp_path / "trial_0" / "stdout.log").read_text(encoding="utf-8")
    assert "from_inherit" in stdout
