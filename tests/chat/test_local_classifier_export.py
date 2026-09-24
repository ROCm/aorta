"""The export tool, everywhere that does not need torch or a checkpoint.

Neither exists on the machine this was written on, so this covers the plumbing
either side of ``torch.onnx.export``: resolving the render template, refusing to
invent one, probing the checkpoint for its encoder and head, validating the
temperature fit, and assembling and writing the manifest.

What is **not** covered, and is listed here rather than left to be discovered:
the ``torch.onnx.export`` call itself, whether the attribute names in
``_ENCODER_ATTRIBUTES`` and ``_HEAD_ATTRIBUTES`` are the ones a real agent
carries, whether the render template can in fact be read off
``rl_agent_config.json``, and whether the resulting graph agrees numerically
with the torch path. Every one of those needs weights.
"""

from __future__ import annotations

import json
import logging

import pytest

from aorta.chat.local_classifier.artifact import (
    GRAPH_NAME,
    MANIFEST_NAME,
    TOKENIZER_NAME,
    Manifest,
    RenderTemplate,
    load_artifact,
)
from aorta.chat.local_classifier.export import (
    ExportError,
    build_manifest,
    check_temperatures,
    export_artifact,
    resolve_modules,
    resolve_render,
    write_manifest,
)

_RENDER = {
    "state_template": "STATE: {state}",
    "question_template": "Q: {question}",
    "option_template": "{marker} {option} : {criteria}",
    "separator": "\n",
    "marker": "[OPT]",
}


class FakeAgent:
    """Stands in for a loaded Laya agent, carrying only what the export reads."""

    def __init__(self, *, config=None, encoder=None, head=None, revision="rev-1"):
        self.cfg = config if config is not None else {"max_len": 1024, **_RENDER}
        self.encoder = encoder if encoder is not None else "an-encoder"
        self.decision_head = head if head is not None else "a-head"
        self.revision = revision
        self.tok = None


# ── the render template ───────────────────────────────────────────────────


class TestResolvingTheRenderTemplate:
    def test_it_reads_the_checkpoints_own_format(self):
        template = resolve_render({"max_len": 512, **_RENDER})
        assert template.marker == "[OPT]"
        assert template.state_template == "STATE: {state}"

    def test_a_nested_render_block_is_read_too(self):
        template = resolve_render({"render": dict(_RENDER)})
        assert template.separator == "\n"

    def test_it_refuses_to_guess_a_format_it_cannot_find(self):
        """The heart of it.

        A plausible default here would be train/serve skew introduced by the
        one component whose job is to prevent it: the model would be asked in a
        wording it was never fine-tuned on, and nothing would fail -- the
        answers would just be slightly worse.
        """
        with pytest.raises(ExportError) as exc:
            resolve_render({"max_len": 512, "hidden_size": 768})
        message = str(exc.value)
        assert "will not guess" in message
        # It names what the config did carry, because the person hitting this
        # is holding the checkpoint this code has never seen.
        assert "hidden_size" in message

    def test_a_partial_config_is_still_a_refusal(self):
        partial = {"state_template": "{state}", "marker": "[OPT]"}
        with pytest.raises(ExportError) as exc:
            resolve_render(partial)
        assert "question_template" in str(exc.value)

    def test_an_override_replaces_the_config_entirely(self):
        """Not merged. Half from the checkpoint and half from the operator is
        the worst of both, and nothing would record which came from where."""
        template = resolve_render({"marker": "[FROM-CONFIG]"}, {**_RENDER, "marker": "[MINE]"})
        assert template.marker == "[MINE]"


# ── the modules to export ─────────────────────────────────────────────────


class TestResolvingTheModules:
    def test_it_finds_an_encoder_and_a_head(self):
        agent = FakeAgent(encoder="ENC", head="HEAD")
        assert resolve_modules(agent) == ("ENC", "HEAD")

    def test_a_missing_head_lists_what_the_object_actually_has(self):
        """These names are a probe, not a verified API.

        The first real run has to say what to fix, not export the wrong
        subgraph and report numbers from it.
        """

        class Bare:
            encoder = "ENC"
            something_else = 1

        with pytest.raises(ExportError) as exc:
            resolve_modules(Bare())
        message = str(exc.value)
        assert "decision head" in message
        assert "something_else" in message


# ── the temperature fit ───────────────────────────────────────────────────


class TestCheckingTheFit:
    def test_an_empty_fit_is_allowed(self):
        """The honest state of things until the Phase 1 harness has run."""
        assert check_temperatures({}) == {}

    def test_a_non_positive_temperature_is_refused(self):
        with pytest.raises(ExportError):
            check_temperatures({"noul:2": -1.0})

    def test_a_non_numeric_temperature_is_refused(self):
        with pytest.raises(ExportError):
            check_temperatures({"noul:2": "hot"})

    def test_a_sharpening_temperature_is_warned_about(self, caplog):
        """The 0.1006 case, which is the reason the [local-classifier] floor is 0.3.5.

        Below 1.0 sharpens logits rather than softening them, so an uncertain
        answer gets published as a confident one. Our own fit is not subject to
        the library's clamp, so nothing else would say a word about it.
        """
        with caplog.at_level(logging.WARNING):
            assert check_temperatures({"choice:21": 0.1006}) == {"choice:21": 0.1006}
        assert "sharpens" in caplog.text


# ── the manifest ──────────────────────────────────────────────────────────


class TestBuildingTheManifest:
    def _build(self, **overrides) -> Manifest:
        kwargs = {
            "source_checkpoint": "fine-tune",
            "source_revision": "sha256:abc",
            "onnx_sha256": "b" * 64,
            "laya_version": "0.3.5",
            "max_len": 1024,
            "temperatures": {},
            "render": RenderTemplate.from_dict(_RENDER),
        }
        kwargs.update(overrides)
        return build_manifest(**kwargs)

    def test_it_records_the_identity_it_was_given(self):
        manifest = self._build()
        assert manifest.source_revision == "sha256:abc"
        assert "fit:none" in manifest.identity()

    def test_an_artifact_with_no_revision_is_refused(self):
        """Decision 22 does not accept a name or a tag in place of a digest."""
        with pytest.raises(ExportError) as exc:
            self._build(source_revision="")
        assert "digest" in str(exc.value)

    def test_the_manifest_is_written_last(self, tmp_path):
        """load_artifact keys off the manifest's presence.

        Writing it first would make a half-copied directory look loadable and
        fail at the session instead -- several frames from the cause, inside a
        Chainlit handler, mid-conversation.
        """
        manifest = self._build()
        (tmp_path / GRAPH_NAME).write_bytes(b"graph")
        (tmp_path / TOKENIZER_NAME).write_text("{}", encoding="utf-8")
        path = write_manifest(tmp_path, manifest)
        assert path.name == MANIFEST_NAME
        assert load_artifact(tmp_path).manifest == manifest

    def test_what_it_writes_is_json_a_later_build_can_read(self, tmp_path):
        manifest = self._build()
        write_manifest(tmp_path, manifest)
        raw = json.loads((tmp_path / MANIFEST_NAME).read_text(encoding="utf-8"))
        assert Manifest.from_dict(raw) == manifest
        # The graph signature travels with it, so a reader can see which
        # inputs the file expects without loading it.
        assert raw["graph"]["output"] == "marker_logits"


# ── the whole export, with torch stubbed out ──────────────────────────────


class TestTheExportEndToEnd:
    """Everything except :func:`write_torch_graph`, which needs torch and weights.

    ``graph_writer`` and ``load`` are the injection points, and the first is
    drawn exactly around the torch surface so that what cannot be run here is
    one named function rather than a thread through the whole export. Faking it
    proves the ordering, the digests and the manifest; it proves nothing at all
    about the graph, which is stated in this module's docstring and in the
    report.
    """

    def _export(self, tmp_path, *, agent=None, **kwargs):
        written = {}

        def fake_writer(encoder, head, path, *, max_len, opset):
            written.update(encoder=encoder, head=head, path=path, max_len=max_len)
            path.write_bytes(b"pretend onnx bytes")

        target = tmp_path / "artifact"
        checkpoint = tmp_path / "checkpoint"
        (checkpoint / "tokenizer").mkdir(parents=True, exist_ok=True)
        (checkpoint / "tokenizer" / TOKENIZER_NAME).write_text("{}", encoding="utf-8")

        manifest = export_artifact(
            str(checkpoint),
            target,
            temperatures=kwargs.pop("temperatures", {}),
            load=lambda _name: agent or FakeAgent(),
            graph_writer=kwargs.pop("graph_writer", fake_writer),
            **kwargs,
        )
        return manifest, target, written

    def test_it_writes_a_directory_that_loads_back(self, tmp_path):
        manifest, target, _ = self._export(tmp_path)
        assert load_artifact(target, verify_digest=True).manifest == manifest

    def test_the_digest_it_records_is_of_the_file_it_wrote(self, tmp_path):
        """The pair Rule 2 is about has to be written as a pair."""
        _, target, _ = self._export(tmp_path)
        assert load_artifact(target, verify_digest=True)

    def test_the_tokenizer_travels_with_the_graph(self, tmp_path):
        """Copied, not referenced.

        The destination node has neither the checkpoint nor a HuggingFace
        cache, and a tokenizer resolved by name there would either fail or find
        a different revision and tokenise the same text into different ids.
        """
        _, target, _ = self._export(tmp_path)
        assert (target / TOKENIZER_NAME).is_file()

    def test_the_writer_is_handed_the_resolved_modules_and_the_real_window(self, tmp_path):
        """The plumbing either side of the one call that needs torch."""
        agent = FakeAgent(encoder="ENC", head="HEAD")
        _, target, written = self._export(tmp_path, agent=agent)
        assert (written["encoder"], written["head"]) == ("ENC", "HEAD")
        assert written["max_len"] == 1024
        assert written["path"] == target / GRAPH_NAME

    def test_a_writer_that_writes_nothing_is_refused(self, tmp_path):
        """A manifest describing a missing graph is worse than no manifest."""

        def writes_nothing(encoder, head, path, *, max_len, opset):
            return None

        with pytest.raises(ExportError) as exc:
            self._export(tmp_path, graph_writer=writes_nothing)
        assert "without writing" in str(exc.value)

    def test_the_fit_reaches_the_manifest(self, tmp_path):
        manifest, _, _ = self._export(tmp_path, temperatures={"noul:2": 1.6})
        assert manifest.temperatures == {"noul:2": 1.6}
        assert manifest.fit_id() != "none"

    def test_a_checkpoint_with_no_window_is_refused(self, tmp_path):
        """The predictor refuses to truncate, so it needs the real window."""
        agent = FakeAgent(config={**_RENDER})
        with pytest.raises(ExportError) as exc:
            self._export(tmp_path, agent=agent)
        assert "max_len" in str(exc.value)

    def test_a_checkpoint_that_names_no_revision_is_refused(self, tmp_path):
        agent = FakeAgent(revision="")
        with pytest.raises(ExportError) as exc:
            self._export(tmp_path, agent=agent)
        assert "source_revision" in str(exc.value)

    def test_an_explicit_revision_overrides_what_was_probed(self, tmp_path):
        manifest, _, _ = self._export(tmp_path, source_revision="sha256:supplied")
        assert manifest.source_revision == "sha256:supplied"
