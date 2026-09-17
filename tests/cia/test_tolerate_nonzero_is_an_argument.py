"""Whether a non-zero exit is expected is a fact about one job.

The sanitizer positive control exits non-zero on purpose -- "guardrail not
clean" is the finding, not a failure -- so its batch script swallows the code
and lets the verdict come from the sanitizer report. The driver said so by
setting ``CIA_TOLERATE_NONZERO`` in ``os.environ`` before calling launch and
restoring it in a ``finally``.

There is one environment per process and triages run concurrently on a pool.
Two of them interleave like this: B reads the variable and finds it unset, A
sets it, A starts rendering its script, B's finally removes it, and A -- still
rendering -- reads an environment that no longer says what A put there. A's
script omits its ``exit 0``, Slurm marks the run FAILED, and the job is
reported as broken for exiting exactly the way it was designed to.

Nothing about that is per-process, so it does not belong in the process. It is
an argument now.
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from aorta.cia import triage as triage_mod
from aorta.cia.launch.cluster import build_sbatch_script


def _script(**overrides) -> str:
    args = {
        "command": "python train.py",
        "job_name": "cia-test",
        "log_path": "/tmp/cia-test/watch.log",
    }
    return build_sbatch_script(**{**args, **overrides})


class TestTheFlagIsCarriedByTheCall:
    def test_a_tolerated_run_exits_zero(self):
        script = _script(tolerate_nonzero=True)

        assert "exit 0" in script
        assert "exit $rc" not in script

    def test_an_ordinary_run_still_reports_its_code(self):
        script = _script(tolerate_nonzero=False)

        assert "exit $rc" in script
        assert "exit 0" not in script

    def test_the_default_is_to_report_the_code(self):
        """Swallowing an exit status is the exception and should be asked for."""
        assert "exit $rc" in _script()

    def test_the_exit_code_is_still_recorded_either_way(self):
        """The watchdog reads this line; tolerating must not hide it."""
        for tolerate in (True, False):
            assert 'echo "[cia] workload exit=$rc"' in _script(tolerate_nonzero=tolerate)


class TestTheEnvironmentNoLongerDecides:
    def test_setting_the_old_variable_changes_nothing(self, monkeypatch):
        """It was an internal channel between two functions, not a knob."""
        monkeypatch.setenv("CIA_TOLERATE_NONZERO", "1")

        assert "exit $rc" in _script()

    def test_nor_does_unsetting_it_override_the_argument(self, monkeypatch):
        monkeypatch.delenv("CIA_TOLERATE_NONZERO", raising=False)

        assert "exit 0" in _script(tolerate_nonzero=True)

    def test_rendering_does_not_read_the_environment_for_it(self):
        import inspect

        from aorta.cia.launch import cluster

        source = inspect.getsource(cluster.build_sbatch_script)

        assert "CIA_TOLERATE_NONZERO" not in source


class TestTwoTriagesAtOnce:
    """The race, run as the pool actually runs them."""

    def test_each_gets_the_script_it_asked_for(self):
        ready = threading.Barrier(2, timeout=10)

        def render(tolerate: bool) -> str:
            ready.wait()  # both inside the call at the same moment
            return _script(job_name=f"job-{tolerate}", tolerate_nonzero=tolerate)

        with ThreadPoolExecutor(max_workers=2) as pool:
            tolerant = pool.submit(render, True)
            strict = pool.submit(render, False)
            tolerant_script, strict_script = tolerant.result(), strict.result()

        assert "exit 0" in tolerant_script
        assert "exit $rc" in strict_script

    def test_many_at_once_still_each_get_their_own(self):
        def render(i: int) -> tuple[bool, str]:
            tolerate = i % 2 == 0
            return tolerate, _script(job_name=f"job-{i}", tolerate_nonzero=tolerate)

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(render, range(24)))

        for tolerate, script in results:
            expected = "exit 0" if tolerate else "exit $rc"
            assert expected in script

    def test_the_process_environment_is_left_alone(self):
        before = os.environ.get("CIA_TOLERATE_NONZERO")
        _script(tolerate_nonzero=True)

        assert os.environ.get("CIA_TOLERATE_NONZERO") == before


class TestTheDriverAsksForIt:
    def test_run_triage_reaches_launch_with_the_flag(self, tmp_path, monkeypatch):
        """The positive control still has to come back COMPLETED."""
        seen: dict = {}

        def fake_launch(**kwargs):
            seen.update(kwargs)
            return "", "stopped before submitting"

        monkeypatch.setattr(triage_mod, "launch", fake_launch)
        source = tmp_path / "k.hip"
        source.write_text("__global__ void bump(float* o) { o[0] += 1; }\n", encoding="utf-8")

        triage_mod.run_triage(
            ["--source", str(source), "--jobs-root", str(tmp_path), "--kernel-name", "bump"]
        )

        assert seen.get("tolerate_nonzero") is True

    def test_it_does_not_touch_the_environment_to_say_so(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CIA_TOLERATE_NONZERO", raising=False)
        during: list = []

        def fake_launch(**kwargs):
            during.append(os.environ.get("CIA_TOLERATE_NONZERO"))
            return "", "stopped before submitting"

        monkeypatch.setattr(triage_mod, "launch", fake_launch)
        source = tmp_path / "k.hip"
        source.write_text("__global__ void bump(float* o) { o[0] += 1; }\n", encoding="utf-8")

        triage_mod.run_triage(
            ["--source", str(source), "--jobs-root", str(tmp_path), "--kernel-name", "bump"]
        )

        assert during == [None]


class TestOnlyTheSweepPathsGetIt:
    """A raw workload's exit code is the only thing anyone has to go on.

    Swallowing it is right for a sanitizer sweep, where exiting non-zero is the
    finding and the verdict lives in a JSON report. It is wrong for --command,
    which runs the user's own program: a training script that died came back
    exit 0, Slurm recorded COMPLETED, and the job status follows the scheduler's
    state, so the record agreed with it. Nothing anywhere said the run had
    failed.

    The flag had no effect on the paths it was written for anyway. Both sweep
    commands end in ``|| true`` from echo_findings, so their exit code is
    already zero before the script's own handling sees it. Set unconditionally,
    it changed the outcome of exactly one path -- the one it should never have
    applied to.
    """

    @staticmethod
    def _flag_for(argv: list[str], tmp_path, monkeypatch) -> object:
        seen: dict = {}

        def fake_launch(**kwargs):
            seen.update(kwargs)
            return "", "stopped before submitting"

        monkeypatch.setattr(triage_mod, "launch", fake_launch)
        triage_mod.run_triage([*argv, "--jobs-root", str(tmp_path)])
        return seen.get("tolerate_nonzero")

    def test_a_raw_command_reports_its_exit_code(self, tmp_path, monkeypatch):
        flag = self._flag_for(["--command", "python train.py"], tmp_path, monkeypatch)

        assert flag is False

    def test_a_pasted_kernel_still_tolerates_one(self, tmp_path, monkeypatch):
        source = tmp_path / "k.hip"
        source.write_text("__global__ void bump(float* o) { o[0] += 1; }\n", encoding="utf-8")

        flag = self._flag_for(
            ["--source", str(source), "--kernel-name", "bump"], tmp_path, monkeypatch
        )

        assert flag is True

    def test_and_so_does_a_recipe(self, tmp_path, monkeypatch):
        recipe = tmp_path / "r.yaml"
        recipe.write_text("schema_version: 1\n", encoding="utf-8")

        flag = self._flag_for(["--recipe", str(recipe)], tmp_path, monkeypatch)

        assert flag is True


class TestWhatTheScriptDoesWithIt:
    """The consequence, at the level the scheduler sees."""

    def test_a_tolerated_crash_is_reported_as_success(self):
        """Which is correct for a sweep and wrong for a user's program."""
        script = _script(command="(exit 1)", tolerate_nonzero=True)

        assert script.rstrip().endswith("exit 0")

    def test_an_untolerated_crash_keeps_its_code(self):
        script = _script(command="(exit 1)", tolerate_nonzero=False)

        assert script.rstrip().endswith("exit $rc")

    def test_the_sweep_command_already_ends_in_a_tolerant_shape(self):
        """echo_findings appends `|| true`, so rc is zero before the flag acts.

        This is why setting it unconditionally looked harmless: on the paths it
        was written for it does nothing at all.
        """
        import inspect

        source = inspect.getsource(triage_mod.run_triage)

        assert "|| true" in source


class TestTheDeadCodeIsGone:
    def test_the_unused_run_helper_was_removed(self):
        """It was the only reader of the env copy above it; both were dead."""
        assert not hasattr(triage_mod, "run")

    def test_and_the_env_copy_it_was_for(self):
        import inspect

        source = inspect.getsource(triage_mod.run_triage)

        assert "CIA_TOLERATE_NONZERO" not in source
