"""A staged workload is written inside staged/, under a name of its own.

``triage_workload`` took ``label`` from the tool call — the model's own words —
and made a filename of it with ``replace(" ", "_")``. That leaves ``/`` and
``..`` untouched, so ``label="../../../../.bashrc"`` wrote the user's pasted
source outside the staging directory.

The second half is quieter and more likely: two runs with the same label wrote
the same path, so a later triage overwrote an earlier one's script while the
node was still reading it. The kernel and assembly paths already stamp their
stems; this one did not.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import pytest

from aorta.chat.tools.cluster import _STAGED_STEM_RE

STAGED = Path("/jobs/staged")


def _staged_path(label: str) -> Path:
    """The filename the tool builds, without running a cluster job to see it."""
    name = label or "workload"
    stem = _STAGED_STEM_RE.sub("_", name).strip("._")[:60] or "workload"
    return STAGED / f"{stem}-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}.py"


class TestItStaysInsideStaged:
    @pytest.mark.parametrize(
        "label",
        [
            "../../../../.bashrc",
            "../outside",
            "a/b/c",
            "/etc/passwd",
            "..",
            "...",
            "./../x",
            "sub/dir/workload",
        ],
    )
    def test_a_label_that_means_somewhere_else_does_not_go_there(self, label):
        path = _staged_path(label)
        assert path.parent == STAGED
        assert ".." not in path.name
        assert "/" not in path.name

    def test_the_dangerous_label_from_the_finding(self):
        """It wrote user-pasted source over a dotfile in the home directory."""
        assert _staged_path("../../../../.bashrc").parent == STAGED

    @pytest.mark.parametrize("label", ["", "...", "///", "___"])
    def test_a_label_that_sanitises_to_nothing_still_has_a_name(self, label):
        assert _staged_path(label).name.startswith("workload-")


class TestEachRunGetsItsOwnFile:
    def test_the_same_label_twice_does_not_collide(self):
        """A second triage used to overwrite the first mid-flight."""
        first = _staged_path("my workload")
        second = _staged_path("my workload")
        assert first != second

    def test_the_name_still_carries_the_label(self):
        """Unique is not enough; a reader has to recognise their own run."""
        assert _staged_path("rms_norm NaN at step 5").name.startswith("rms_norm_NaN_at_step_5-")

    def test_it_ends_in_py(self):
        assert _staged_path("anything").suffix == ".py"


class TestTheNameStaysReadable:
    def test_spaces_become_underscores(self):
        assert "my_workload" in _staged_path("my workload").name

    def test_a_very_long_label_is_bounded(self):
        stem = _staged_path("x" * 400).name.split("-")[0]
        assert len(stem) <= 60

    @pytest.mark.parametrize(
        "label,expected",
        [
            ("rms_norm", "rms_norm"),
            ("triage-1", "triage-1"),
            ("model.v2", "model.v2"),
        ],
    )
    def test_an_ordinary_label_is_left_alone(self, label, expected):
        assert _staged_path(label).name.startswith(f"{expected}-")


def test_the_other_two_paths_already_stamped_theirs():
    """This finding was the odd one out, not a new policy."""
    source = Path("src/aorta/chat/tools/cluster.py").read_text(encoding="utf-8")
    assert source.count("strftime") >= 3, "kernel, assembly and workload should each stamp"
