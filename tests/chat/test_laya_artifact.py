"""What an exported Laya artifact has to say about itself before it is trusted.

Decision 22's Rule 2 in test form. A checkpoint swap changes verdicts without
changing a line of code, without an error, and without anything in a report
that says so, and this repository has already been bitten by the general shape
of that twice -- a stale ``ROCJITSU_PREBUILT`` directory that would have
reported fixed bugs as unfixed, and the retrieval index manifest of Decision
20a. So the manifest is the thing that refuses.

Every assertion here runs with no weights, no onnxruntime and no torch: an
artifact's *schema* is exactly the part that has to be checkable on a machine
that cannot load one.
"""

from __future__ import annotations

import json

import pytest

from aorta.chat.laya.artifact import (
    ARTIFACT_VERSION,
    GRAPH_NAME,
    MANIFEST_NAME,
    TOKENIZER_NAME,
    ArtifactUnavailableError,
    Manifest,
    RenderTemplate,
    calibration_key,
    load_artifact,
    sha256_file,
)
from aorta.laya.predictor import Choice, Noul

_RENDER = {
    "state_template": "STATE: {state}",
    "question_template": "Q: {question}",
    "option_template": "{marker} {option} : {criteria}",
    "separator": "\n",
    "marker": "[OPT]",
}


def _manifest_dict(**overrides) -> dict:
    payload = {
        "artifact_version": ARTIFACT_VERSION,
        "source_checkpoint": "laya-typed-decisions",
        "source_revision": "sha256:abc123",
        "onnx_sha256": "0" * 64,
        "laya_version": "0.3.5",
        "max_len": 1024,
        "temperatures": {"noul:2": 1.4},
        "render": dict(_RENDER),
        "exported_by": "amd-aorta test",
    }
    payload.update(overrides)
    return payload


def _stage(directory, **overrides) -> None:
    """Write a complete, loadable artifact directory."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / GRAPH_NAME).write_bytes(b"not really onnx")
    (directory / TOKENIZER_NAME).write_text("{}", encoding="utf-8")
    payload = _manifest_dict(**overrides)
    if "onnx_sha256" not in overrides:
        payload["onnx_sha256"] = sha256_file(directory / GRAPH_NAME)
    (directory / MANIFEST_NAME).write_text(json.dumps(payload), encoding="utf-8")


# ── the calibration bucket ────────────────────────────────────────────────


class TestTheTemperatureBucket:
    """The fit is written under one key and read under another, or it is not a fit."""

    def test_it_agrees_with_the_eval_harness(self):
        """The one that would fail silently.

        ``aorta.laya.eval`` fits a temperature per (question type, option
        count) and keys it off a ``LabelledExample``; inference has no labelled
        example, so the key is computed again here. If the two conventions ever
        drift, every lookup misses, every bucket falls back to temperature 1.0,
        and nothing anywhere raises -- the probabilities are simply the
        uncalibrated ones while the manifest says a fit was applied.
        """
        from aorta.laya.corpus.schema import LabelledExample
        from aorta.laya.eval import calibration_key as eval_key

        for question, label in (
            (Noul(question="is it clean?"), "true"),
            (Choice(question="which?", options=("a", "b", "c")), "a"),
        ):
            example = LabelledExample(
                decision="watch_healthy",
                state="a state",
                question=question,
                label=label,
                join_key="k",
            )
            assert calibration_key(question) == eval_key(example)

    def test_a_noul_is_always_the_two_option_bucket(self):
        assert calibration_key(Noul(question="anything")) == "noul:2"

    def test_a_choice_carries_its_width(self):
        assert calibration_key(Choice(question="q", options=("a", "b"))) == "choice:2"


# ── identity ──────────────────────────────────────────────────────────────


class TestIdentity:
    """What a report records, and what it records when there is nothing to record."""

    def test_it_names_the_checkpoint_the_graph_and_the_fit(self):
        manifest = Manifest.from_dict(_manifest_dict())
        identity = manifest.identity()
        assert "laya-typed-decisions" in identity
        assert manifest.onnx_sha256[:12] in identity
        assert manifest.fit_id() in identity

    def test_an_unfitted_artifact_says_so_rather_than_omitting_it(self):
        """The half-identity Decision 22 is about.

        A probability is a function of the checkpoint *and* the fit applied to
        it. An artifact exported before Phase 1 produced a fit is usable, and
        every line it appears in has to say that the number is the model's own
        rather than a calibrated one.
        """
        manifest = Manifest.from_dict(_manifest_dict(temperatures={}))
        assert manifest.fit_id() == "none"
        assert "fit:none" in manifest.identity()

    def test_two_different_fits_are_two_different_identities(self):
        one = Manifest.from_dict(_manifest_dict(temperatures={"noul:2": 1.4}))
        two = Manifest.from_dict(_manifest_dict(temperatures={"noul:2": 2.1}))
        assert one.fit_id() != two.fit_id()

    def test_the_same_fit_is_the_same_identity_whatever_order_it_was_written(self):
        one = Manifest.from_dict(_manifest_dict(temperatures={"noul:2": 1.4, "choice:3": 2.0}))
        two = Manifest.from_dict(_manifest_dict(temperatures={"choice:3": 2.0, "noul:2": 1.4}))
        assert one.fit_id() == two.fit_id()


class TestTheTemperatureLookup:
    def test_a_fitted_bucket_is_applied(self):
        manifest = Manifest.from_dict(_manifest_dict(temperatures={"noul:2": 1.4}))
        assert manifest.temperature_for(Noul(question="q")) == pytest.approx(1.4)

    def test_an_unfitted_bucket_is_the_identity_rather_than_an_error(self):
        """A fit covers the buckets the corpus happened to hold.

        Refusing a question type nobody had labelled yet would take the node
        down for a gap in the training data, which is a worse trade than
        answering uncalibrated and saying so in the identity.
        """
        manifest = Manifest.from_dict(_manifest_dict(temperatures={"noul:2": 1.4}))
        question = Choice(question="q", options=("a", "b", "c"))
        assert manifest.temperature_for(question) == 1.0


# ── refusing a manifest that cannot be trusted ────────────────────────────


class TestTheManifestRefuses:
    @pytest.mark.parametrize(
        "key",
        ["source_checkpoint", "source_revision", "onnx_sha256", "laya_version", "max_len",
         "temperatures", "render"],
    )
    def test_a_missing_field_is_refused_rather_than_defaulted(self, key: str):
        payload = _manifest_dict()
        del payload[key]
        with pytest.raises(ArtifactUnavailableError) as exc:
            Manifest.from_dict(payload)
        assert key in str(exc.value)

    def test_an_artifact_from_another_graph_signature_is_refused(self):
        """The quiet one: an older artifact still tokenises and still returns numbers."""
        with pytest.raises(ArtifactUnavailableError) as exc:
            Manifest.from_dict(_manifest_dict(artifact_version=ARTIFACT_VERSION - 1))
        assert "Re-export" in str(exc.value)

    def test_a_non_positive_temperature_is_refused(self):
        with pytest.raises(ArtifactUnavailableError):
            Manifest.from_dict(_manifest_dict(temperatures={"noul:2": 0.0}))

    def test_a_non_numeric_temperature_is_refused(self):
        with pytest.raises(ArtifactUnavailableError):
            Manifest.from_dict(_manifest_dict(temperatures={"noul:2": "warm"}))

    def test_a_zero_context_window_is_refused(self):
        with pytest.raises(ArtifactUnavailableError) as exc:
            Manifest.from_dict(_manifest_dict(max_len=0))
        assert "no question fits" in str(exc.value)

    def test_it_round_trips(self):
        manifest = Manifest.from_dict(_manifest_dict())
        assert Manifest.from_dict(manifest.to_dict()) == manifest


class TestTheRenderTemplateRefuses:
    """The template is captured from the checkpoint, so a broken one is a bug not a typo."""

    @pytest.mark.parametrize(
        "field,broken",
        [
            ("state_template", "STATE:"),
            ("question_template", "Q:"),
            ("option_template", "{marker} option"),
            ("option_template", "{option}"),
        ],
    )
    def test_a_template_that_drops_what_it_wraps_is_refused(self, field, broken):
        render = dict(_RENDER)
        render[field] = broken
        with pytest.raises(ArtifactUnavailableError):
            RenderTemplate.from_dict(render)

    def test_an_empty_marker_is_refused(self):
        with pytest.raises(ArtifactUnavailableError):
            RenderTemplate.from_dict({**_RENDER, "marker": ""})

    def test_a_missing_field_names_itself(self):
        render = dict(_RENDER)
        del render["separator"]
        with pytest.raises(ArtifactUnavailableError) as exc:
            RenderTemplate.from_dict(render)
        assert "separator" in str(exc.value)


# ── loading a directory ───────────────────────────────────────────────────


class TestLoadingAnArtifact:
    def test_a_complete_directory_loads(self, tmp_path):
        _stage(tmp_path / "laya")
        artifact = load_artifact(tmp_path / "laya")
        assert artifact.manifest.source_checkpoint == "laya-typed-decisions"
        assert artifact.graph_path.is_file()

    def test_an_absent_artifact_prints_the_staging_procedure(self, tmp_path):
        """Not a traceback. It cannot be fixed by a download either, so the
        message has to carry the export, not a pip install."""
        with pytest.raises(ArtifactUnavailableError) as exc:
            load_artifact(tmp_path / "nothing-here")
        message = str(exc.value)
        assert "export_artifact" in message
        assert "AORTA_CHAT_LAYA_ARTIFACT_PATH" in message

    @pytest.mark.parametrize("missing", [GRAPH_NAME, TOKENIZER_NAME])
    def test_a_manifest_without_its_files_is_refused(self, tmp_path, missing):
        directory = tmp_path / "laya"
        _stage(directory)
        (directory / missing).unlink()
        with pytest.raises(ArtifactUnavailableError) as exc:
            load_artifact(directory)
        assert "Copy the whole directory" in str(exc.value)

    def test_unparseable_json_names_the_file(self, tmp_path):
        directory = tmp_path / "laya"
        _stage(directory)
        (directory / MANIFEST_NAME).write_text("{ not json", encoding="utf-8")
        with pytest.raises(ArtifactUnavailableError) as exc:
            load_artifact(directory)
        assert MANIFEST_NAME in str(exc.value)

    def test_a_graph_that_is_not_the_one_described_is_refused_when_verified(self, tmp_path):
        """The case Rule 2 is actually about: one of the pair was replaced.

        Not corruption -- a *different* copy. It loads, it runs, and it answers
        from weights the manifest does not describe.
        """
        directory = tmp_path / "laya"
        _stage(directory)
        (directory / GRAPH_NAME).write_bytes(b"a different export entirely")
        with pytest.raises(ArtifactUnavailableError) as exc:
            load_artifact(directory, verify_digest=True)
        assert "without the other" in str(exc.value)

    def test_the_digest_is_not_checked_unless_asked(self, tmp_path):
        """A user is waiting on the first routed turn; hashing is opt-in."""
        directory = tmp_path / "laya"
        _stage(directory)
        (directory / GRAPH_NAME).write_bytes(b"a different export entirely")
        assert load_artifact(directory).manifest.max_len == 1024
