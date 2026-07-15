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


def test_env_file_validation_failure_does_not_leave_partial_file(tmp_path):
    """Validation rejection must NOT leave a partial probe.env on disk.

    Regression for PR #194 round 6 (Copilot): the previous shape
    ``os.open(..., O_TRUNC) -> for-loop with mid-stream raise`` would
    write rows 0..k-1 to disk and then raise on row k. The caller
    recorded ``env_file_validation_failed`` in result.json, but a
    partial probe.env was already on disk -- a 0600 KEY=VALUE file
    that looked legitimate but contained only a subset of the
    cell's bundle. Operators inspecting the trial directory would
    see a probe.env that contradicts result.json's failure_type.

    The fix is two-pronged and this test pins both: validate-first
    in ``_write_env_file`` (atomic-on-rejection) AND a defensive
    unlink in the caller's failure path (so a *prior* run's stale
    probe.env from the same trial directory cannot survive either).
    """
    wl = _make_workload(
        tmp_path,
        argv=["true"],
        env_mode="file",
        # Sorted iteration: AAA, BBB, ZZZ_BAD (the bad key). If
        # validation were per-row-during-write, AAA + BBB would
        # land on disk before the ZZZ raise.
        cell_env_vars={
            "AAA_GOOD": "first",
            "BBB_GOOD": "second",
            "ZZZ_BAD\nKEY": "value",
        },
    )
    wl.setup()
    wl.run()
    trial_dir = tmp_path / "trial_0"
    env_path = trial_dir / "probe.env"
    assert not env_path.exists(), (
        "probe.env must not exist after env-file validation failure; "
        f"found contents: {env_path.read_text() if env_path.exists() else '(absent)'}"
    )
    # Sanity: the failure was still recorded.
    import json

    doc = json.loads((trial_dir / "result.json").read_text(encoding="utf-8"))
    assert doc["failure_type"] == "env_file_validation_failed"


def test_env_file_validation_failure_scrubs_stale_prior_file(tmp_path):
    """A stale probe.env from a previous run must be cleaned up on rejection.

    Resume + flat_resume reuse the same ``trial_<n>/`` directory across
    invocations. If a previous run wrote a valid probe.env and the
    current run's bundle fails validation, the on-disk artifact
    must agree with result.json -- a stale leftover would contradict
    ``failure_type=env_file_validation_failed``.
    """
    trial_dir = tmp_path / "trial_0"
    trial_dir.mkdir(parents=True, exist_ok=True)
    stale_path = trial_dir / "probe.env"
    stale_path.write_text("STALE=from_prior_run\n", encoding="utf-8")
    stale_path.chmod(0o600)

    wl = _make_workload(
        tmp_path,
        argv=["true"],
        env_mode="file",
        cell_env_vars={"BAD\nKEY": "value"},
    )
    wl.setup()
    wl.run()

    assert not stale_path.exists(), (
        "stale probe.env from a prior run was not scrubbed after the "
        "current run's env-file validation failure"
    )


def test_env_file_rejects_newline_in_value(tmp_path):
    """Newline in env value -> Tier-1 fail artifact (not unhandled raise).

    Previous contract (pre-PR-#194-review): ``_write_env_file`` raised
    ``ValueError`` and the exception escaped ``run()``. That broke the
    "every probe trial leaves an artifact" contract: the dispatcher
    recorded an ``infrastructure_failed`` TrialResult but no per-trial
    ``result.json`` was on disk, which made ``flat_resume``'s
    ``is_trial_complete`` predicate re-run the broken cell on every
    subsequent ``aorta probe`` invocation. The new contract synthesises
    a ``result.json`` (verdict=error, ``failure_type=env_file_validation_failed``;
    issue #230) so resume sees the cell as complete and the operator gets a
    grep-able cause in the artifact tree.
    """
    wl = _make_workload(
        tmp_path,
        argv=["true"],
        env_mode="file",
        cell_env_vars={"MULTILINE": "line1\nline2"},
    )
    wl.setup()
    result = wl.run()

    trial_dir = tmp_path / "trial_0"
    result_path = trial_dir / "result.json"
    assert result_path.is_file(), (
        "env-file validation failure must still leave result.json -- "
        "otherwise flat_resume re-runs the cell forever"
    )
    import json

    doc = json.loads(result_path.read_text(encoding="utf-8"))
    # Issue #230: env-file validation rejection is an infra/config error.
    assert doc["verdict"] == "error"
    assert doc["failure_type"] == "env_file_validation_failed"
    assert "newline" in doc["error_message"]
    assert doc["env_passthrough_mode"] == "file"
    assert doc["trial_index"] == 0
    # ``stderr.log`` carries the same diagnostic so operators inspecting
    # the trial without parsing JSON still see the cause.
    stderr_text = (trial_dir / "stderr.log").read_text(encoding="utf-8")
    assert "newline" in stderr_text

    # WorkloadResult must reflect "never launched" so the matrix
    # outcome classifier doesn't conflate this with a completed
    # 1/1 trial (the same invariant the Popen-exec-failed branch
    # already maintains).
    assert result.passed is False
    assert result.main_work_started is False
    assert result.executed_iterations == 0


@pytest.mark.parametrize(
    "bad_key",
    [
        "FOO\nBAR",  # newline in key
        "FOO\rBAR",  # carriage return in key
        "FOO=BAR",  # embedded '=' would rebind a later key
        "X\n=injected",  # newline-then-equals row-injection attempt
    ],
)
def test_env_file_rejects_unsafe_chars_in_key(tmp_path, bad_key):
    """Hostile sidecar keys -> ``error``-verdict artifact (issue #230).

    Original regression (PR #194 round 4): hostile mitigation sidecars
    could smuggle ``\\n``, ``\\r``, or ``=`` into env *keys* to inject
    extra KEY=VALUE rows or rebind a later key. The sidecar loader
    only enforces ``isinstance(key, str)``, so the env-file writer is
    the right place to catch this -- it is the layer that owns the
    on-disk KEY=VALUE row shape.

    Pre-fix behaviour was a raw ``ValueError`` escape; post-fix the
    rejection is recorded as ``verdict=error`` /
    ``failure_type=env_file_validation_failed`` in ``result.json`` so
    flat_resume can move past the broken cell.
    """
    wl = _make_workload(
        tmp_path,
        argv=["true"],
        env_mode="file",
        cell_env_vars={bad_key: "safe-value"},
    )
    wl.setup()
    result = wl.run()

    trial_dir = tmp_path / "trial_0"
    import json

    doc = json.loads((trial_dir / "result.json").read_text(encoding="utf-8"))
    assert doc["verdict"] == "error"
    assert doc["failure_type"] == "env_file_validation_failed"
    assert "env key" in doc["error_message"]
    assert result.passed is False
    assert result.main_work_started is False


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


def test_inherit_mode_scrubs_stale_probe_env(tmp_path):
    """A leftover ``probe.env`` from a prior ``file``-mode attempt is removed.

    Regression for PR #194 review: in ``flat_resume`` runs the trial
    directory is re-used when a prior attempt's ``result.json`` was
    truncated. If the prior attempt ran in ``env_passthrough_mode=file``
    it wrote a ``probe.env``; if the resumed attempt is in ``inherit``
    mode the file would otherwise remain on disk, contradicting the
    on-the-wire behaviour (no ``AORTA_ENV_FILE`` exported in the child).
    """
    workload_subdir = tmp_path / "_subprocess"
    workload_subdir.mkdir(parents=True, exist_ok=True)
    # Pre-create the trial dir + a stale probe.env from a "previous"
    # file-mode run, then invoke the workload in inherit mode.
    trial_dir = tmp_path / "trial_0"
    trial_dir.mkdir()
    stale = trial_dir / "probe.env"
    stale.write_text("STALE_KEY=stale_value\n", encoding="utf-8")
    assert stale.is_file()

    wl = _make_workload(
        tmp_path,
        argv=["true"],
        env_mode="inherit",
        cell_env_vars={"DISABLE_TF32": "1"},
    )
    wl.setup()
    wl.run()

    assert not stale.exists(), "inherit mode must scrub a stale probe.env"
    # Child must not have seen AORTA_ENV_FILE either; verify by
    # re-running with a probe that echoes the var to stdout.
    wl2 = _make_workload(
        tmp_path,
        argv=["sh", "-c", "echo AORTA_ENV_FILE=${AORTA_ENV_FILE:-unset}"],
        env_mode="inherit",
    )
    wl2.setup()
    wl2.run()
    stdout = (trial_dir / "stdout.log").read_text(encoding="utf-8")
    assert "AORTA_ENV_FILE=unset" in stdout


def test_inherit_mode_with_no_prior_probe_env_is_a_noop(tmp_path):
    """Scrubbing logic must not raise when there's no stale file to remove."""
    wl = _make_workload(
        tmp_path,
        argv=["true"],
        env_mode="inherit",
    )
    wl.setup()
    wl.run()
    assert not (tmp_path / "trial_0" / "probe.env").exists()
