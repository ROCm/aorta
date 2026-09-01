"""The published-index CI jobs, asserted as configuration.

These live under ``tests/cli/`` rather than ``tests/chat/`` because they parse
YAML with pyyaml -- a base dependency -- so they run on an install with no chat
extra, which is where a workflow regression would otherwise go unnoticed.

The public-tree guard is the reason this file exists. A published index stores
every chunk's source text verbatim, so an index built over an internal or
customer tree republishes that source, and a release asset cannot be recalled.
The guard therefore has to be enforced in code rather than remembered in review,
and its wiring in the workflow has to be enforced too -- deleting one `if:` line
is a one-character change with an unrecallable consequence.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

#: Two assertions below compare the workflow against the module that defines the
#: asset names, so they need the extra. The rest are pure YAML and must not.
_CHAT_AVAILABLE = importlib.util.find_spec("langchain_core") is not None
_needs_chat = pytest.mark.skipif(not _CHAT_AVAILABLE, reason="amd-aorta[chat-cli] not installed")

_WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"
_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load(name: str) -> dict:
    return yaml.safe_load((_WORKFLOWS / name).read_text(encoding="utf-8"))


@pytest.fixture(params=["nightly.yml", "release.yml"])
def index_job(request) -> tuple[str, dict]:
    """The index-build job from each workflow that publishes one."""
    workflow = _load(request.param)
    assert "chat-index" in workflow["jobs"], f"{request.param} has no chat-index job"
    return request.param, workflow["jobs"]["chat-index"]


def _steps_text(job: dict) -> str:
    return "\n".join(str(step.get("run", "")) for step in job["steps"])


class TestPublicTreeGuard:
    def test_the_job_is_pinned_to_the_public_repository(self, index_job):
        """First of the two guards, and the one that cannot be defended in code.

        A workflow pointed at a private fork of this file would otherwise build
        and publish an index over whatever that fork contains.
        """
        _, job = index_job
        assert "github.repository == 'ROCm/aorta'" in job["if"]

    def test_the_build_passes_public_only(self, index_job):
        """Second guard: re-checks origin and restricts the corpus to git-tracked files.

        The remote check catches a misconfigured workflow; the tracked-file
        filter catches an untracked internal reproducer or customer bundle in the
        working directory, which no remote check can see.
        """
        name, job = index_job
        text = _steps_text(job)
        assert "aorta chat index build --public-only" in text, name

    def test_the_built_index_is_re_verified_before_publishing(self, index_job):
        """Belt and braces: assert no chunk came from an untracked path."""
        name, job = index_job
        assert "scripts/verify_chat_index_sources.py" in _steps_text(job), name

    def test_the_verification_script_exists(self):
        assert (_SCRIPTS / "verify_chat_index_sources.py").exists()

    def test_verification_runs_before_publishing(self, index_job):
        """A check after the upload would be a post-mortem, not a guard."""
        _, job = index_job
        names = [step.get("name", "") for step in job["steps"]]
        verify = next(i for i, n in enumerate(names) if "untracked source" in n)
        publish = next(
            i
            for i, step in enumerate(job["steps"])
            if str(step.get("uses", "")).startswith("softprops/action-gh-release")
        )
        assert verify < publish


class TestTorchFreeGuard:
    def test_the_job_refuses_a_torch_or_cuda_resolution(self, index_job):
        """Decision 19a asserted in the one job that installs the extra fresh.

        A transitive dependency change is exactly how a CUDA wheel would come
        back, and this is the only place that would notice.
        """
        name, job = index_job
        text = _steps_text(job)
        assert "nvidia-" in text, name
        assert "torch" in text, name


class TestPublishTargets:
    @_needs_chat
    def test_the_nightly_publishes_the_rolling_main_asset(self):
        """A .devN install resolves to this tag; every internal install is one."""
        from aorta.chat.rag.index_ops import ASSET_NAME, ROLLING_TAG

        job = _load("nightly.yml")["jobs"]["chat-index"]
        publish = [s for s in job["steps"] if "action-gh-release" in str(s.get("uses", ""))]
        assert len(publish) == 1
        assert publish[0]["with"]["tag_name"] == ROLLING_TAG
        assert ASSET_NAME in publish[0]["with"]["files"]

    def test_the_release_attaches_a_per_release_asset(self):
        """Release assets are immutable, which is what makes version matching work."""
        job = _load("release.yml")["jobs"]["chat-index"]
        publish = [s for s in job["steps"] if "action-gh-release" in str(s.get("uses", ""))]
        assert len(publish) == 1
        assert publish[0]["with"]["tag_name"] == "${{ needs.build-and-release.outputs.tag }}"

    def test_the_release_job_reads_the_tag_the_build_job_resolved(self):
        """Re-deriving it would risk the asset landing on a different tag."""
        workflow = _load("release.yml")
        assert workflow["jobs"]["build-and-release"]["outputs"]["tag"]
        assert workflow["jobs"]["chat-index"]["needs"] == "build-and-release"

    @_needs_chat
    def test_all_three_files_are_published_together(self, index_job):
        """An index without its manifest is one a client has to refuse."""
        from aorta.chat.rag.manifest import CHECKSUM_SUFFIX, MANIFEST_SUFFIX

        name, job = index_job
        files = next(
            s["with"]["files"]
            for s in job["steps"]
            if "action-gh-release" in str(s.get("uses", ""))
        )
        assert MANIFEST_SUFFIX in files, name
        assert CHECKSUM_SUFFIX in files, name

    def test_the_nightly_index_job_does_not_race_the_wheel_job(self):
        """Both write to the same rolling release, and one force-moves its tag."""
        assert _load("nightly.yml")["jobs"]["chat-index"]["needs"] == "nightly"


class TestDigestSkip:
    def test_the_nightly_skips_a_rebuild_when_the_corpus_is_unchanged(self):
        """Otherwise an identical 18 MB asset is re-uploaded every night."""
        job = _load("nightly.yml")["jobs"]["chat-index"]
        text = _steps_text(job)
        assert "aorta chat index digest --public-only" in text
        build = next(s for s in job["steps"] if "index build" in str(s.get("run", "")))
        assert build["if"] == "steps.digest.outputs.changed == 'true'"

    def test_a_missing_baseline_rebuilds_rather_than_skips(self):
        """ "No published manifest" must not be read as "nothing changed"."""
        job = _load("nightly.yml")["jobs"]["chat-index"]
        digest = next(s for s in job["steps"] if s.get("id") == "digest")
        assert "changed=true" in digest["run"]
        assert "No published manifest yet" in digest["run"]

    def test_the_release_does_not_skip(self):
        """`index fetch` resolves by tag, so a release must carry its own asset.

        Even when the corpus is byte-identical to the previous release's.
        """
        job = _load("release.yml")["jobs"]["chat-index"]
        assert not any("digest" in str(step.get("id", "")) for step in job["steps"])


class TestJobHygiene:
    def test_the_index_job_is_separate_from_the_publish_job(self):
        """It must not be able to delay or fail the wheel/PyPI publish."""
        for name in ("nightly.yml", "release.yml"):
            jobs = _load(name)["jobs"]
            assert "chat-index" in jobs
            wheel = jobs["nightly" if name == "nightly.yml" else "build-and-release"]
            assert not any("chat index" in str(s.get("run", "")) for s in wheel["steps"])

    def test_python_311_is_used_because_that_is_chats_floor(self, index_job):
        name, job = index_job
        setup = next(s for s in job["steps"] if "setup-python" in str(s.get("uses", "")))
        assert setup["with"]["python-version"] == "3.11", name

    def test_the_eval_is_logged_rather_than_gated(self, index_job):
        """Decision 19b's baseline is young; a threshold today would be arbitrary."""
        name, job = index_job
        step = next(s for s in job["steps"] if "index eval" in str(s.get("run", "")))
        assert "|| echo" in step["run"], name

    def test_full_history_is_fetched_so_the_manifest_records_head(self, index_job):
        name, job = index_job
        checkout = next(s for s in job["steps"] if "actions/checkout" in str(s.get("uses", "")))
        assert checkout["with"]["fetch-depth"] == 0, name
