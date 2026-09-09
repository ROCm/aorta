"""The guard catches the values that got past it.

``test_settings_neutral.py`` exists to keep one site's layout out of a public
repository, and several values walked past it into one: a head-node IP shipped
as a default, the same IP in a comment, a hostname with no hyphens in it, and a
toolchain path pinned to one machine's installed ROCm. The hostname pattern
required hyphens, there was no address pattern at all, and the scan read only
``src/aorta/cia`` while the chat tools reach the same cluster.

A guard is only worth what it catches, so this asserts on the patterns directly
with the strings that beat them -- and, just as much, on the benign lines a
widened guard must not start failing on. A guard that cries wolf earns an
allowlist, and then it guards nothing.
"""

from __future__ import annotations

import pytest

from tests.cia.test_settings_neutral import _offences


def _flagged(line: str) -> list[str]:
    """What the guard says about *line*, asked the way the guard asks it."""
    return _offences(line)


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

    def test_a_toolchain_pinned_to_one_installed_version(self):
        """/opt/rocm is everywhere; /opt/rocm-7.0.2.2 is one machine's."""
        line = '_ROCM_LLVM = os.environ.get("ROCM_LLVM_BIN", "/opt/rocm-7.0.2.2/lib/llvm/bin")'
        assert "a version-pinned toolchain path" in _flagged(line)

    def test_a_three_digit_hostname(self):
        """Four digits was a workaround for gfx950; hardware names are excluded now."""
        assert "a bare cluster hostname" in _flagged('node = "chi287"')


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
    @pytest.mark.parametrize(
        "line",
        [
            # Same address on every machine.
            'LOCAL_HOSTS = {"", "local", "localhost", "127.0.0.1"}',
            'BIND = "0.0.0.0"',
            # An unpinned toolchain root, which is where ROCm lives everywhere.
            'ROCM = "/opt/rocm"',
            # Paths that mean the same thing anywhere.
            'TMP = "/tmp/aorta"',
            'DEV = "/dev/kfd"',
        ],
    )
    def test_benign_lines_are_not_flagged(self, line):
        assert not _flagged(line), f"false positive on: {line}"

    @pytest.mark.parametrize(
        "line",
        [
            'arch = "gfx950"',
            'if arch == "gfx942": pass',
            'BOARD = "mi355x"',
            'variants = ["gfx90a", "gfx942", "gfx950"]',
            'if board.startswith("mi300"): pass',
            'CAP = "sm_90"',
            'GPU = "navi31"',
        ],
    )
    def test_hardware_names_are_not_hostnames(self, line):
        """gfx950 is an architecture; every MI355X in the world is an MI355X."""
        assert not _flagged(line), f"false positive on: {line}"

    def test_a_pinned_version_is_still_not_read_as_an_address(self):
        """7.0.2.2 is a dotted quad; the hyphen is what tells them apart."""
        line = 'TOOLCHAIN = "/opt/rocm-7.0.2.2/lib/llvm/bin"'
        assert "an IP address" not in _flagged(line)


def test_the_guard_reads_the_chat_tools_too():
    """The tools reach the same cluster and are written the same way."""
    from tests.cia.test_settings_neutral import _GUARDED

    assert "src/aorta/chat/tools" in _GUARDED
