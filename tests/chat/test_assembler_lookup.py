"""Finding the assembler, and saying so when it is not there.

The default assembler path named one machine's ROCm patch release. Anywhere
else the clang invocation failed, and the user was told "the pasted assembly
did not assemble" -- a missing toolchain reported as a problem with their
code, which sends them to rewrite assembly that was fine.

Where the lookup happens matters as much as what it falls back to. The
assemble runs under ``srun`` on a compute node, and the login node this is
typed on frequently has no ROCm at all: resolving the path in this process
would consult the wrong machine and refuse a setup that works. So the
node picks, and this process only reads the answer.
"""

from __future__ import annotations

import importlib
import os
import subprocess
from pathlib import Path

import pytest


@pytest.fixture()
def cluster(monkeypatch):
    pytest.importorskip("dspy", reason="cluster tools need the [cia] extra")

    def _load(rocm_llvm: str | None = None):
        if rocm_llvm is None:
            monkeypatch.delenv("ROCM_LLVM_BIN", raising=False)
            monkeypatch.delenv("ROCM_PATH", raising=False)
        else:
            monkeypatch.setenv("ROCM_LLVM_BIN", rocm_llvm)
        return importlib.reload(importlib.import_module("aorta.chat.tools.cluster"))

    yield _load
    monkeypatch.delenv("ROCM_LLVM_BIN", raising=False)
    importlib.reload(importlib.import_module("aorta.chat.tools.cluster"))


@pytest.fixture()
def fake_clang(tmp_path):
    """Stand-in clangs on two different paths, each announcing which it is."""

    def _make(where: str) -> Path:
        directory = tmp_path / where
        directory.mkdir(parents=True, exist_ok=True)
        clang = directory / "clang"
        clang.write_text(
            "#!/bin/sh\n"
            f'echo "{where}"\n'
            'for a in "$@"; do [ "$prev" = "-o" ] && touch "$a"; prev="$a"; done\n',
            encoding="utf-8",
        )
        clang.chmod(0o755)
        return directory

    return _make


def _run(command: str, path: str) -> subprocess.CompletedProcess:
    """Run the generated command the way srun would: a bare shell on a node."""
    return subprocess.run(
        ["bash", "-c", command], capture_output=True, text=True, env={"PATH": path}
    )


class TestTheDefaultIsNotOneMachine:
    def test_no_patch_version_is_pinned(self, cluster):
        """The version that was here existed on exactly one box."""
        module = cluster(None)

        assert module._ROCM_ROOT == "/opt/rocm"
        assert "/opt/rocm/lib/llvm/bin" == module._ROCM_LLVM

    def test_rocm_path_moves_the_whole_toolchain(self, cluster, monkeypatch):
        monkeypatch.setenv("ROCM_PATH", "/opt/rocm-8.1")
        module = importlib.reload(importlib.import_module("aorta.chat.tools.cluster"))

        assert module._ROCM_LLVM == "/opt/rocm-8.1/lib/llvm/bin"

    def test_the_bin_directory_can_be_named_outright(self, cluster):
        module = cluster("/usr/lib/llvm-18/bin")

        assert module._ROCM_LLVM == "/usr/lib/llvm-18/bin"


class TestTheNodeChoosesTheAssembler:
    """Three cases, run as a shell the way the compute node would."""

    def test_the_configured_path_is_preferred(self, cluster, fake_clang, tmp_path):
        rocm = fake_clang("rocm")
        other = fake_clang("onpath")
        module = cluster(str(rocm))

        done = _run(
            module._assemble_command(tmp_path / "in.s", tmp_path / "out.hsaco"),
            path=f"{other}:/usr/bin:/bin",
        )

        assert done.returncode == 0
        assert "rocm" in done.stdout

    def test_a_clang_on_the_path_is_used_when_it_is_absent(
        self, cluster, fake_clang, tmp_path
    ):
        """The version-independent default still misses on some installs."""
        other = fake_clang("onpath")
        module = cluster(str(tmp_path / "nowhere"))

        done = _run(
            module._assemble_command(tmp_path / "in.s", tmp_path / "out.hsaco"),
            path=f"{other}:/usr/bin:/bin",
        )

        assert done.returncode == 0
        assert "onpath" in done.stdout

    def test_with_neither_it_says_so_distinctly(self, cluster, tmp_path):
        module = cluster(str(tmp_path / "nowhere"))

        done = _run(
            module._assemble_command(tmp_path / "in.s", tmp_path / "out.hsaco"),
            path="/usr/bin:/bin",
        )

        assert done.returncode != 0
        assert module._NO_ASSEMBLER in done.stderr

    def test_the_paths_are_quoted(self, cluster, fake_clang, tmp_path):
        """Staged filenames are derived from a label the model supplies."""
        rocm = fake_clang("rocm")
        module = cluster(str(rocm))
        awkward = tmp_path / "a b; touch /tmp/aorta-pwned.s"

        done = _run(
            module._assemble_command(awkward, tmp_path / "out.hsaco"),
            path="/usr/bin:/bin",
        )

        assert not Path("/tmp/aorta-pwned.s").exists()
        assert done.returncode == 0


class TestTheMessageBlamesTheRightThing:
    def test_a_missing_toolchain_is_not_reported_as_bad_assembly(self, cluster):
        module = cluster(None)
        message = module._no_assembler_message()

        assert "did not assemble" not in message
        assert "Nothing is wrong with what you pasted" in message

    def test_it_names_what_was_looked_for(self, cluster):
        """Otherwise the reader cannot tell where to install or point it."""
        module = cluster("/opt/rocm-8.1/lib/llvm/bin")
        message = module._no_assembler_message()

        assert "/opt/rocm-8.1/lib/llvm/bin/clang" in message
        assert "PATH" in message

    def test_it_says_which_knob_fixes_it(self, cluster):
        message = cluster(None)._no_assembler_message()

        assert "ROCM_PATH" in message

    def test_the_sentinel_does_not_leak_into_the_message(self, cluster):
        """It is a wire signal between the node and this process."""
        module = cluster(None)

        assert module._NO_ASSEMBLER not in module._no_assembler_message()


class TestTheGuardWouldCatchARelapse:
    def test_this_file_is_scanned_for_site_specific_values(self):
        """The pin lived here precisely because the guard did not look here."""
        from tests.cia import test_settings_neutral

        assert "src/aorta/chat/tools" in test_settings_neutral._GUARDED
