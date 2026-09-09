"""The guard catches the values that got past it.

``test_settings_neutral.py`` exists to keep one site's layout out of a public
repository, and three values walked past it into one: a head-node IP shipped as
a default, the same IP in a comment, and a hostname with no hyphens in it. The
hostname pattern required hyphens and there was no address pattern at all.

A guard is only worth what it catches, so this asserts on the patterns directly
with the strings that beat them.
"""

from __future__ import annotations

import pytest

from tests.cia.test_settings_neutral import _SITE_SHAPED, _UNIVERSAL


def _flagged(line: str) -> list[str]:
    """What the guard would say about *line*, as the guard itself asks it."""
    if _UNIVERSAL.search(line):
        return []
    return [what for pattern, what in _SITE_SHAPED if pattern.search(line)]


class TestTheValuesThatGotThrough:
    def test_an_ip_shipped_as_a_default_argument(self):
        line = 'def run_aorta_probe(bundle_root: Path, head_node: str = "149.28.124.225"):'
        assert "an IP address" in _flagged(line)

    def test_an_ip_left_behind_in_a_comment(self):
        line = '    head_node: str = ""   # SSH host (e.g. 149.28.124.225)'
        assert "an IP address" in _flagged(line)

    def test_a_hostname_with_no_hyphens_to_give_it_away(self):
        line = "    # Default: proxy runs on whatever node the agent is on (chi2878)."
        assert "a bare cluster hostname" in _flagged(line)


class TestWhatItStillCatches:
    @pytest.mark.parametrize(
        "line",
        [
            'root = f"/apps/{os.environ.get(\'USER\')}"',
            'home = "/home/someone/jobs"',
            "node = 'cv350-rck-g03'",
            'cmd = "sbatch --partition=meta64 job.sh"',
            'PARTITION = "interactive"',
        ],
    )
    def test_the_original_shapes_still_fail(self, line):
        assert _flagged(line)


class TestWhatItMustNotCatch:
    """A guard that cries wolf gets an allowlist, and then it guards nothing."""

    @pytest.mark.parametrize(
        "line",
        [
            # Same address on every machine.
            'LOCAL_HOSTS = {"", "local", "localhost", "127.0.0.1"}',
            'BIND = "0.0.0.0"',
            # A version, not an address -- the hyphen is what tells them apart.
            'TOOLCHAIN = "/opt/rocm-7.0.2.2/lib/llvm/bin"',
            # Architectures, not hostnames: too few digits to be one.
            'ARCH = "gfx950"',
            'if arch == "gfx942": pass',
            'BOARD = "mi355x"',
            # Paths that mean the same thing anywhere.
            'TMP = "/tmp/aorta"',
            'DEV = "/dev/kfd"',
        ],
    )
    def test_benign_lines_are_not_flagged(self, line):
        assert not _flagged(line), f"false positive on: {line}"


def test_the_guard_reads_the_chat_tools_too():
    """The tools reach the same cluster and are written the same way."""
    from tests.cia.test_settings_neutral import _GUARDED

    assert "src/aorta/chat/tools" in _GUARDED
