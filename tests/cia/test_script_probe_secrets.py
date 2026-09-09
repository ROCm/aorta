"""Reading how jobs are launched must not read what launches them.

``read_cluster_configs`` and ``read_existing_launch_scripts`` ran
``find $HOME ... | xargs head -25``, putting the first 25 lines of up to eight
of the user's shell scripts into an LLM prompt bound for whatever
LITELLM_API_BASE points at. The first lines of a personal ``.sh`` are where
``export ..._API_KEY=`` lives, so the part that got sent was the part worth
keeping.

What the planner needs from those files is the #SBATCH directives -- the
partition, the time limit, the gres. These tests run the probes against a real
directory with real-shaped credentials in it, because the property is about
what leaves the machine, and a mock cannot tell you that.
"""

from __future__ import annotations

import pytest

from aorta.cia.launch import discovery, planner
from aorta.cia.launch.cluster import scrub_secrets

#: Written the way people actually write them, in the first lines of a script.
PLANTED = {
    "hf_THISMUSTNOTLEAK1234567890": "export HF_API_KEY=hf_THISMUSTNOTLEAK1234567890",
    "tok_abcdef123456": 'export MY_SECRET_TOKEN="tok_abcdef123456"',
    "hunter2": "export DATABASE_PASSWORD=hunter2",
    "AKIAIOSFODNN7EXAMPLE": "export AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE",
    "ghp_0123456789abcdef": "GITHUB_TOKEN=ghp_0123456789abcdef",
}


@pytest.fixture
def home_with_secrets(tmp_path, monkeypatch):
    """A search root shaped like somebody's home directory."""
    work = tmp_path / "work"
    work.mkdir()
    (work / "train.sbatch").write_text(
        "#!/bin/bash\n"
        + "\n".join(PLANTED.values())
        + "\n#SBATCH --partition=meta64\n"
        "#SBATCH --time=04:00:00\n"
        "#SBATCH --gres=gpu:8\n"
        "srun python train.py\n"
    )
    (work / "env.sh").write_text("#!/bin/bash\nexport DATABASE_PASSWORD=hunter2\n")
    monkeypatch.setenv("CIA_SEARCH_ROOTS", str(tmp_path))
    return tmp_path


class TestNothingSecretLeavesTheMachine:
    @pytest.mark.parametrize("secret", sorted(PLANTED))
    def test_read_cluster_configs_does_not_carry_it(self, secret, home_with_secrets):
        assert secret not in discovery.read_cluster_configs("localhost")

    @pytest.mark.parametrize("secret", sorted(PLANTED))
    def test_read_existing_launch_scripts_does_not_carry_it(
        self, secret, home_with_secrets
    ):
        assert secret not in planner.read_existing_launch_scripts("localhost", "node-01")


class TestWhatTheProbeStillLearns:
    def test_the_directives_the_planner_needs_survive(self, home_with_secrets):
        out = discovery.read_cluster_configs("localhost")
        assert "#SBATCH --partition=meta64" in out
        assert "#SBATCH --time=04:00:00" in out
        assert "#SBATCH --gres=gpu:8" in out

    def test_the_filenames_survive_for_context(self, home_with_secrets):
        assert "train.sbatch" in discovery.read_cluster_configs("localhost")

    def test_the_body_of_the_script_does_not(self, home_with_secrets):
        """Nothing outside a directive line has a reason to be in the prompt."""
        out = discovery.read_cluster_configs("localhost")
        assert "srun python train.py" not in out


class TestTheScrubberAsASecondGate:
    @pytest.mark.parametrize(
        "line",
        [
            "export HF_API_KEY=hf_leak",
            "MY_TOKEN=abc",
            "readonly DB_PASSWORD=hunter2",
            "export AWS_SECRET_ACCESS_KEY=x",
            "SESSION_COOKIE=abc123",
            "export BEARER_TOKEN=xyz",
            "api_key = 'lower case counts too'",
            "PRIVATE_KEY=-----BEGIN",
        ],
    )
    def test_an_assignment_named_like_a_credential_is_dropped(self, line):
        assert scrub_secrets(line) == ""

    @pytest.mark.parametrize(
        "line",
        [
            "#SBATCH --partition=meta64",
            "export PATH=/usr/bin",
            "CUDA_VISIBLE_DEVICES=0,1",
            "export OMP_NUM_THREADS=8",
            "module load rocm",
            "#SBATCH --gres=gpu:8",
        ],
    )
    def test_an_ordinary_line_is_kept(self, line):
        assert scrub_secrets(line) == line

    def test_it_matches_on_the_name_not_the_value(self):
        """A token looks like any other opaque string; only the name tells you."""
        assert scrub_secrets("SOME_TOKEN=aaaaaaaa") == ""
        assert scrub_secrets("BUILD_ID=aaaaaaaa") == "BUILD_ID=aaaaaaaa"

    def test_it_drops_only_the_offending_line(self):
        text = "#SBATCH --time=1:00\nexport API_KEY=x\n#SBATCH --gres=gpu:8"
        assert scrub_secrets(text) == "#SBATCH --time=1:00\n#SBATCH --gres=gpu:8"
