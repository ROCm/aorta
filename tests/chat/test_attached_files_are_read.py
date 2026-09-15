"""An attached listing has to reach the tools, or be refused out loud.

The upload button was on and the handler read ``message.content`` and nothing
else, so an attached file was discarded in silence: the agent answered from
the covering sentence, and nothing anywhere said the file had not been read.
For a diagnostic product that is the worst shape a failure can take -- the
answer looks like a verdict on your code and is a verdict on your prose.

Folded into the prompt as a fenced block rather than routed somewhere new. A
fence is what the harness reads as "the user is pointing at this", and tool
selection already knows what to do with a message carrying code, so an
attachment takes the path a paste takes.
"""

from __future__ import annotations

import pytest

cl = pytest.importorskip("chainlit", reason="requires the chat-ui extra")

from aorta.chat.ui import app

_ASM = "\tglobal_load_dword v5, v[3:4], off\n\tv_add_f32 v6, v5, v5\n\ts_endpgm\n"


class _Element:
    def __init__(self, name: str, path: str | None):
        self.name = name
        self.path = path


class _Message:
    def __init__(self, content: str = "", elements=None):
        self.content = content
        self.elements = elements or []


def attach(tmp_path, name: str, body, suffix_ok: bool = True):
    target = tmp_path / name
    target.write_bytes(body if isinstance(body, bytes) else body.encode("utf-8"))
    return _Element(name, str(target))


class TestAListingIsRead:
    def test_its_contents_reach_the_prompt(self, tmp_path):
        msg = _Message("run waitcheck on this", [attach(tmp_path, "kernel.s", _ASM)])

        folded, skipped = app._attached_source(msg)

        assert "global_load_dword" in folded
        assert skipped == []

    def test_it_arrives_fenced(self, tmp_path):
        """The harness treats a fence as the user pointing at the code."""
        msg = _Message("check this", [attach(tmp_path, "kernel.s", _ASM)])

        folded, _ = app._attached_source(msg)

        assert folded.count("```") == 2

    def test_the_file_is_named(self, tmp_path):
        """So the answer can refer to it, and so several stay distinguishable."""
        msg = _Message("check", [attach(tmp_path, "gemm.isa", _ASM)])

        folded, _ = app._attached_source(msg)

        assert "gemm.isa" in folded

    def test_several_files_all_arrive(self, tmp_path):
        msg = _Message(
            "compare these",
            [attach(tmp_path, "a.s", _ASM), attach(tmp_path, "b.s", "s_endpgm\n")],
        )

        folded, skipped = app._attached_source(msg)

        assert "a.s" in folded and "b.s" in folded
        assert skipped == []

    def test_the_harness_accepts_what_comes_out(self, tmp_path):
        """End of the road: it has to survive prepare_asm, not merely look right."""
        from aorta.chat.tools.harness.assembly import prepare_asm

        msg = _Message("run waitcheck", [attach(tmp_path, "kernel.s", _ASM)])
        folded, _ = app._attached_source(msg)

        prepared = prepare_asm(f"{msg.content}\n\n{folded}", arch="gfx950")

        assert "global_load_dword" in prepared.program


class TestWhatIsRefusedIsSaidOutLoud:
    def test_a_binary_object_is_refused_by_name(self, tmp_path):
        """A .hsaco is the obvious thing to attach and the one that cannot work."""
        msg = _Message("check", [attach(tmp_path, "gemm.hsaco", b"\x7fELF\x02\x01")])

        folded, skipped = app._attached_source(msg)

        assert folded == ""
        assert any("gemm.hsaco" in note for note in skipped)

    def test_the_binary_note_says_to_disassemble(self, tmp_path):
        msg = _Message("check", [attach(tmp_path, "gemm.isa", b"\x00\x01\x02\xff\xfe")])

        _, skipped = app._attached_source(msg)

        assert any("disassemble" in note for note in skipped)

    def test_a_listing_is_never_truncated(self, tmp_path, monkeypatch):
        """Half a listing assembles into a different program.

        Large ones are staged whole rather than cut down -- see
        TestAListingTooBigToPasteIsStagedInstead. What must never happen is a
        partial listing reaching the assembler under the user's file name.
        """
        monkeypatch.setenv("AORTA_CHAT_JOBS_PATH", str(tmp_path / "jobs"))
        from aorta.chat.config import reset_settings

        reset_settings()
        big = "\ts_nop 0\n" * 40_000
        msg = _Message("check", [attach(tmp_path, "whole.disasm", big)])

        folded, skipped = app._attached_source(msg)

        assert skipped == [], skipped
        assert "s_nop" not in folded, "the listing was quoted into the prompt"
        reset_settings()

    def test_an_empty_file_is_reported(self, tmp_path):
        msg = _Message("check", [attach(tmp_path, "empty.s", "")])

        folded, skipped = app._attached_source(msg)

        assert folded == ""
        assert any("empty" in note for note in skipped)

    def test_an_element_with_no_path_is_reported(self):
        msg = _Message("check", [_Element("ghost.s", None)])

        folded, skipped = app._attached_source(msg)

        assert folded == ""
        assert skipped


class TestAMessageWithNoAttachment:
    def test_nothing_is_added(self):
        folded, skipped = app._attached_source(_Message("what is aorta?"))

        assert folded == ""
        assert skipped == []

    def test_a_message_without_the_attribute_is_fine(self):
        """Not every caller builds a full chainlit Message."""

        class Bare:
            content = "hello"

        folded, skipped = app._attached_source(Bare())

        assert folded == "" and skipped == []


class TestTheUploadConfigMatchesWhatIsAccepted:
    @staticmethod
    def _upload() -> dict:
        import tomllib
        from pathlib import Path

        shipped = (
            Path(__file__).resolve().parents[2]
            / "src" / "aorta" / "chat" / "ui" / "chainlit_config.toml"
        )
        settings = tomllib.loads(shipped.read_text(encoding="utf-8"))
        return settings["features"]["spontaneous_file_upload"]

    def test_the_browser_does_not_offer_everything(self):
        """"*/*" invited the one file the handler cannot use."""
        assert self._upload()["accept"] != ["*/*"]

    def test_the_cap_admits_a_disassembled_code_object(self):
        """About 9 MB for a GEMM, and that is the file worth attaching."""
        assert self._upload()["max_size_mb"] >= 16

    def test_the_cap_is_not_unbounded(self):
        assert self._upload()["max_size_mb"] <= 128

    def test_assembly_suffixes_are_offered(self):
        accepted = str(self._upload()["accept"])
        for suffix in (".s", ".asm", ".isa"):
            assert suffix in accepted


class TestAListingTooBigToPasteIsStagedInstead:
    """The size that matters is the model's, not the assembler's.

    ``source`` travels as a tool argument, so anything folded into the prompt
    has to be written out again in full by the model to make the call. A real
    disassembled kernel is megabytes and will never fit there -- and a listing
    too big to paste is exactly the one worth attaching. So past a threshold
    the file is staged and the model is handed its name; the tool reads the
    bytes off disk and they never cross the context window.
    """

    @staticmethod
    def _staged_name(folded: str) -> str:
        import re

        found = re.search(r"staged as `([^`]+)`", folded)
        assert found, f"no staged name in: {folded[:200]}"
        return found.group(1)

    @pytest.fixture()
    def jobs_root(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AORTA_CHAT_JOBS_PATH", str(tmp_path / "jobs"))
        from aorta.chat.config import reset_settings

        reset_settings()
        monkeypatch.setattr(app, "_INLINE_ATTACHMENT_BYTES", 64)
        yield tmp_path / "jobs"
        reset_settings()

    def test_the_prompt_does_not_grow_by_the_file(self, tmp_path, jobs_root):
        big = "\ts_nop 0\n" * 5_000
        msg = _Message("waitcheck this", [attach(tmp_path, "big.s", big)])

        folded, skipped = app._attached_source(msg)

        assert skipped == [], skipped
        assert len(folded) < 1_000, "the listing was folded into the prompt"

    def test_the_model_is_given_a_name(self, tmp_path, jobs_root):
        big = "\ts_nop 0\n" * 5_000
        msg = _Message("waitcheck", [attach(tmp_path, "big.s", big)])

        folded, _ = app._attached_source(msg)

        assert self._staged_name(folded).endswith("big.s")

    def test_the_name_is_not_run_into_punctuation(self, tmp_path, jobs_root):
        """A name copied with a trailing '.' is a name that does not resolve."""
        big = "\ts_nop 0\n" * 5_000
        msg = _Message("waitcheck", [attach(tmp_path, "big.s", big)])

        folded, _ = app._attached_source(msg)

        assert not self._staged_name(folded).endswith(".")

    def test_it_is_told_which_argument_to_use(self, tmp_path, jobs_root):
        big = "\ts_nop 0\n" * 5_000
        msg = _Message("waitcheck", [attach(tmp_path, "big.s", big)])

        folded, _ = app._attached_source(msg)

        assert "source_file=" in folded
        assert "not try to reproduce" in folded

    def test_the_staged_file_is_inside_the_jobs_root(self, tmp_path, jobs_root):
        big = "\ts_nop 0\n" * 5_000
        msg = _Message("waitcheck", [attach(tmp_path, "big.s", big)])

        folded, _ = app._attached_source(msg)
        staged = jobs_root / self._staged_name(folded)

        assert staged.is_file()
        assert staged.read_text(encoding="utf-8") == big

    def test_two_uploads_of_one_name_do_not_collide(self, tmp_path, jobs_root):
        big = "\ts_nop 0\n" * 5_000
        first, _ = app._attached_source(
            _Message("a", [attach(tmp_path, "same.s", big)])
        )
        second, _ = app._attached_source(
            _Message("b", [attach(tmp_path, "same.s", big + "\ts_endpgm\n")])
        )

        assert self._staged_name(first) != self._staged_name(second)

    def test_a_small_listing_is_still_folded_in(self, tmp_path, jobs_root):
        """Staging everything would make the common case harder to read."""
        msg = _Message("waitcheck", [attach(tmp_path, "tiny.s", "\ts_endpgm\n")])

        folded, _ = app._attached_source(msg)

        assert "s_endpgm" in folded
        assert "staged as" not in folded


class TestTheToolReadsAStagedListing:
    @pytest.fixture()
    def staged(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AORTA_CHAT_JOBS_PATH", str(tmp_path))
        from aorta.chat.config import reset_settings

        reset_settings()
        target = tmp_path / "chat-uploads" / "upload-x" / "kernel.s"
        target.parent.mkdir(parents=True)
        target.write_text(_ASM, encoding="utf-8")
        yield "chat-uploads/upload-x/kernel.s"
        reset_settings()

    def test_it_gets_past_reading_the_file(self, staged, monkeypatch):
        """What this feature owns is the read, not what the assembler then says.

        Asserting the cluster was reached would be asserting a toolchain is
        installed: assembly happens first and fails without clang, which says
        nothing about whether the staged name resolved.
        """
        pytest.importorskip("dspy", reason="the cluster tools need the [cia] extra")
        from aorta.chat.tools import cluster

        out = cluster.triage_assembly_source.func(source_file=staged)

        for refusal in ("no staged listing", "escapes the jobs root", "could not read"):
            assert refusal not in out, out[:160]

    @pytest.mark.parametrize(
        "escape",
        ["../../etc/passwd", "/etc/passwd", "chat-uploads/../../etc/passwd"],
    )
    def test_it_refuses_a_name_outside_the_jobs_root(self, staged, escape):
        pytest.importorskip("dspy", reason="the cluster tools need the [cia] extra")
        from aorta.chat.tools import cluster

        assert "escapes the jobs root" in cluster.triage_assembly_source.func(
            source_file=escape
        )

    def test_a_missing_name_says_so(self, staged):
        pytest.importorskip("dspy", reason="the cluster tools need the [cia] extra")
        from aorta.chat.tools import cluster

        assert "no staged listing" in cluster.triage_assembly_source.func(
            source_file="chat-uploads/upload-x/absent.s"
        )

    def test_neither_argument_is_an_error(self, staged):
        pytest.importorskip("dspy", reason="the cluster tools need the [cia] extra")
        from aorta.chat.tools import cluster

        assert "Error:" in cluster.triage_assembly_source.func()
