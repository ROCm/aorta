"""An environment variable name is rendered verbatim, so it has to be a name.

``build_sbatch_script`` quotes every value and wrote the key straight into the
script. The keys come from ``LaunchPlan.env_vars``, which is model output, so a
key like ``X; curl attacker.example #`` rendered as::

    export X; curl attacker.example #=safe

An export, then a command, then a comment eating the rest of the line. The
quoted value never mattered — the name had already left the assignment.

Rejecting rather than dropping is deliberate. A job that runs without a
variable it was told to set fails later, somewhere else, as something else —
the sanitizer silently absent because ROCJITSU_BUILD never reached the node is
exactly that failure, and this package has met it before.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aorta.cia.launch.cluster import build_sbatch_script, submit_sbatch


def _script(**env) -> str:
    return build_sbatch_script(
        command="echo hi", job_name="j", log_path="/tmp/l", env_vars=env
    )


class TestNamesThatAreNotNames:
    @pytest.mark.parametrize(
        "name",
        [
            "X; curl attacker.example #",
            "X && rm -rf /",
            "X$(whoami)",
            "X`whoami`",
            "X | tee /tmp/out",
            "X=Y",
            "X Y",
            "X\nY",
            "9X",
            "X-Y",
            "",
        ],
    )
    def test_are_refused(self, name):
        with pytest.raises(ValueError, match="invalid environment variable name"):
            _script(**{name: "safe"})

    def test_the_message_names_the_offender(self):
        with pytest.raises(ValueError, match="curl attacker.example"):
            _script(**{"X; curl attacker.example #": "safe"})

    def test_nothing_is_rendered_for_a_rejected_plan(self):
        """Not a partial script with the good half of the environment in it."""
        with pytest.raises(ValueError):
            _script(GOOD="1", **{"BAD; echo x": "2"})


class TestNamesThatAre:
    @pytest.mark.parametrize(
        "name", ["ROCJITSU_BUILD", "LD_PRELOAD", "_PRIVATE", "A1", "a_b_c9", "PATH"]
    )
    def test_are_exported(self, name):
        assert f"export {name}=" in _script(**{name: "value"})

    def test_a_value_needing_quotes_still_gets_them(self):
        script = _script(CIA_LABEL="two words; echo x")
        assert "export CIA_LABEL='two words; echo x'" in script

    def test_the_environment_the_sanitizers_need_survives(self):
        """The reason this path exists at all."""
        script = _script(ROCJITSU_BUILD="/apps/rocjitsu/build", LD_PRELOAD="/lib/x.so")
        assert "export ROCJITSU_BUILD=/apps/rocjitsu/build" in script
        assert "export LD_PRELOAD=/lib/x.so" in script


class TestTheCallerStillGetsAnError:
    """submit_sbatch promises (job_id, error). A traceback is neither."""

    def test_a_rejected_plan_comes_back_as_an_error_string(self, tmp_path, monkeypatch):
        monkeypatch.setattr("aorta.cia.launch.cluster.sbatch_available", lambda: True)

        job_id, error = submit_sbatch(
            command="echo hi",
            job_name="j",
            log_path=str(tmp_path / "l"),
            script_path=tmp_path / "s.sbatch",
            env_vars={"X; curl attacker.example #": "safe"},
        )

        assert job_id == ""
        assert "invalid environment variable name" in error

    def test_no_script_is_left_behind(self, tmp_path, monkeypatch):
        monkeypatch.setattr("aorta.cia.launch.cluster.sbatch_available", lambda: True)
        script_path = tmp_path / "s.sbatch"

        submit_sbatch(
            command="echo hi",
            job_name="j",
            log_path=str(tmp_path / "l"),
            script_path=script_path,
            env_vars={"X; echo pwned": "safe"},
        )

        assert not script_path.exists()

    def test_the_seam_reports_it_too(self, tmp_path, monkeypatch):
        """launch() is what the triage driver calls."""
        from aorta.cia.launch import launch

        monkeypatch.setattr("aorta.cia.launch.cluster.sbatch_available", lambda: True)
        job_id, error = launch(
            command="echo hi",
            job_name="j",
            log_path=str(tmp_path / "l"),
            script_path=tmp_path / "s.sbatch",
            env_vars={"X; echo pwned": "safe"},
        )
        assert job_id == "" and "invalid environment variable name" in error
