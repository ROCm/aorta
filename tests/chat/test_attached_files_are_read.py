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

    def test_an_oversized_listing_is_refused_not_truncated(self, tmp_path):
        """Half a listing assembles into a different program."""
        big = "\ts_nop 0\n" * 40_000
        msg = _Message("check", [attach(tmp_path, "whole.disasm", big)])

        folded, skipped = app._attached_source(msg)

        assert folded == "", "an oversized listing was silently included"
        assert any("KB" in note for note in skipped)

    def test_the_size_note_suggests_what_to_do(self, tmp_path):
        big = "\ts_nop 0\n" * 40_000
        msg = _Message("check", [attach(tmp_path, "whole.disasm", big)])

        _, skipped = app._attached_source(msg)

        assert any("kernel you care about" in note for note in skipped)

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

    def test_the_size_cap_is_not_half_a_gigabyte(self):
        """The handler refuses past 256 KB; the browser should not accept 500 MB."""
        assert self._upload()["max_size_mb"] <= 1

    def test_assembly_suffixes_are_offered(self):
        accepted = str(self._upload()["accept"])
        for suffix in (".s", ".asm", ".isa"):
            assert suffix in accepted
