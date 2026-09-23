"""What an exported Laya artifact is, and which weights and fit produced it.

The chat path cannot load a Laya checkpoint the way every other track does.
``laya.load()`` exists and works -- ``LayaAgentPredictor`` uses it -- but it
goes through torch, and ``[chat-cli]`` is CI-gated against resolving torch at
all (``.github/workflows/nightly.yml`` and ``release.yml`` fail the build).
So the chat path runs the same model through ONNX on the onnxruntime that
``fastembed`` already installs, and the thing it loads is an artifact this
package exported rather than a checkpoint the library published.

That inverts who owns the format. A published checkpoint carries its own
identity; an artifact we produced carries whatever we decided to write down.
Decision 22 (``docs/laya-packaging.md``) says what has to be written down, and
this module is where that list becomes a schema:

* **The weights, by digest.** ``convaiinnovations/laya`` is a moving reference
  and a subfolder name is not a version, so the manifest records the source
  revision and the SHA-256 of the exported graph. The repository has been bitten
  by the general form of this twice -- a stale ``ROCJITSU_PREBUILT`` directory
  that would have reported fixed bugs as unfixed, and Decision 20a's index
  manifest -- and in both cases the failure was a plausible answer rather than
  an error.
* **The temperature fit, beside the weights.** The calibration is not in the
  weights. One temperature per (question type, option count) is what the Phase 1
  harness fits, so a probability is a function of the checkpoint *and* the fit.
  An artifact exported with no fit is allowed and says so: :meth:`Manifest.identity`
  reports ``fit:none``, which travels into every report a verdict reaches.
* **The render template, captured rather than guessed.** See :class:`RenderTemplate`.

Nothing here imports ``onnxruntime``, ``tokenizers`` or ``numpy``. Those live
inside :mod:`aorta.chat.laya.onnx_predictor`'s loaders, because ``onnxruntime``
and ``fastembed`` are both in ``_HEAVY_PREFIXES`` in
``tests/cli/test_chat_boundaries.py`` and the rule there is about start-up cost,
not about wheel size.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aorta.laya.predictor import LayaUnavailable, Noul, Question

#: Bumped when the graph signature or the manifest's required keys change in a
#: way that makes an older artifact unreadable. Checked on load rather than
#: assumed, because the failure it prevents is the quiet one: an artifact
#: exported against an older marker convention still tokenises, still runs, and
#: still returns numbers.
ARTIFACT_VERSION = 1

MANIFEST_NAME = "aorta-laya-onnx.json"
GRAPH_NAME = "model.onnx"
TOKENIZER_NAME = "tokenizer.json"

#: The graph signature this package exports and loads. One scalar per option
#: marker, and the marker positions passed in rather than found inside the
#: graph, so that locating them is Python that can be tested without weights
#: rather than ONNX ops that cannot.
GRAPH_INPUTS = ("input_ids", "attention_mask", "marker_positions")
GRAPH_OUTPUT = "marker_logits"

#: Printed when the artifact is absent. Modelled on ``PRE_SEED_PROCEDURE`` in
#: ``aorta/chat/rag/embeddings/fastembed_bge.py``, and for the same reason
#: Decision 22 gives: a customer node may have no egress, and surfacing a
#: HuggingFace connection error there reads as a bug in aorta rather than as a
#: missing artifact. It also cannot be fixed by a download, because this
#: artifact is not published anywhere -- somebody has to export it.
STAGING_PROCEDURE = (
    "No exported Laya artifact at {path}, so the chat router and selector "
    "cannot run on Laya.\n"
    "\n"
    "Unlike the embedding weights, this artifact is not downloadable: it is\n"
    "produced from a fine-tuned checkpoint by this repository's own export.\n"
    "\n"
    "To produce one, on a machine that has torch and the checkpoint:\n"
    "  pip install 'amd-aorta[laya]'\n"
    "  python - <<'EOF'\n"
    "  from aorta.chat.laya.export import export_artifact\n"
    "  export_artifact('<checkpoint-dir-or-name>', '{path}', temperatures={{}})\n"
    "  EOF\n"
    "\n"
    "Then copy that directory to this machine and point at it with\n"
    "  export AORTA_CHAT_LAYA_ARTIFACT_PATH={path}\n"
    "\n"
    "Until then, leave AORTA_CHAT_LAYA_ENABLED unset: the router and selector\n"
    "keep running on the LLM, which is what they do today."
)


class ArtifactUnavailable(LayaUnavailable):
    """The exported artifact is missing, incomplete, or not one of ours.

    A subclass of the seam's own error rather than a new exception type, so a
    caller that already handles "no predictor" handles this too with one
    ``except``. The chat nodes have exactly that shape: Laya is a tier, and
    every reason it cannot answer has to land in the same fallback.
    """


def sha256_file(path: str | Path) -> str:
    """The SHA-256 of a file, read in chunks.

    Chunked because the exported graph is hundreds of megabytes and this runs
    on the node that loads it, not only on the one that produced it.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def calibration_key(question: Question) -> str:
    """The temperature bucket *question* falls in: question type and option count.

    The same convention as :func:`aorta.laya.eval.calibration_key`, which keys
    on the exact option count rather than the library's coarser buckets. Spelled
    again here because that function takes a ``LabelledExample`` -- it is the
    scoring side, and there is no labelled example at inference time -- and a
    fit written under one key and read under another applies temperature 1.0 to
    every question while looking entirely normal.
    ``tests/chat/test_laya_artifact.py`` asserts the two agree.
    """
    kind = "noul" if isinstance(question, Noul) else "choice"
    return f"{kind}:{len(question.options)}"


@dataclass(frozen=True)
class RenderTemplate:
    """How a state and its questions become the one string the encoder reads.

    **This is captured from the source checkpoint at export time and never
    guessed here.** It is the part of the ONNX path most able to fail silently:
    fine-tuning on one wording and asking another does not raise, it just
    answers slightly worse, for a reason nobody would go looking for. Track B
    found the same defect in a corpus builder that had independently re-phrased
    the questions the inference path asks.

    So :mod:`aorta.chat.laya.export` reads these fields off the checkpoint's own
    configuration and refuses to export when it cannot find them, rather than
    substituting a plausible default -- a plausible default being exactly the
    failure this paragraph is about. An operator who knows the format can supply
    it explicitly; nothing infers it.

    *marker* is the token that precedes every option. It has to be a single
    token in the checkpoint's tokenizer, which :mod:`~aorta.chat.laya.onnx_predictor`
    checks on load: a marker that tokenises into three pieces would put three
    positions where the scorer expects one and silently misalign every answer
    after the first.
    """

    #: Wraps the state. Must contain ``{state}``.
    state_template: str
    #: Wraps one question's text. Must contain ``{question}``.
    question_template: str
    #: One option. Must contain ``{marker}`` and ``{option}``; ``{criteria}``
    #: receives the gloss, or the empty string for an option with none.
    option_template: str
    #: Between the state, each question, and each option.
    separator: str
    #: The literal option-marker token.
    marker: str

    def __post_init__(self) -> None:
        for field, placeholder in (
            ("state_template", "{state}"),
            ("question_template", "{question}"),
            ("option_template", "{option}"),
            ("option_template", "{marker}"),
        ):
            if placeholder not in getattr(self, field):
                raise ArtifactUnavailable(
                    f"the artifact's render template has a {field} with no "
                    f"{placeholder} in it, so the rendered text would drop the "
                    "thing the model is meant to read"
                )
        if not self.marker:
            raise ArtifactUnavailable("the artifact's render template has an empty marker token")

    @classmethod
    def from_dict(cls, raw: Any) -> RenderTemplate:
        if not isinstance(raw, dict):
            raise ArtifactUnavailable(
                f"the artifact manifest's 'render' is a {type(raw).__name__}, not an object"
            )
        missing = [
            key
            for key in (
                "state_template",
                "question_template",
                "option_template",
                "separator",
                "marker",
            )
            if key not in raw
        ]
        if missing:
            raise ArtifactUnavailable(
                f"the artifact manifest's 'render' is missing {missing}. It is captured "
                "from the checkpoint at export time and cannot be reconstructed here."
            )
        return cls(
            state_template=str(raw["state_template"]),
            question_template=str(raw["question_template"]),
            option_template=str(raw["option_template"]),
            separator=str(raw["separator"]),
            marker=str(raw["marker"]),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "state_template": self.state_template,
            "question_template": self.question_template,
            "option_template": self.option_template,
            "separator": self.separator,
            "marker": self.marker,
        }


@dataclass(frozen=True)
class Manifest:
    """The identity of one exported artifact, and everything needed to use it.

    Every field is required on load. A manifest with a missing key is refused
    rather than defaulted, which is Decision 20a's rule for the retrieval index
    applied to the same hazard here: a mismatched index does not raise, it
    answers fluently from vectors that were never comparable to the query's.
    """

    #: What was exported: a published checkpoint name, or the fine-tune directory.
    source_checkpoint: str
    #: The hub revision, or the digest of the local weights. Decision 22 asks for
    #: a digest rather than a tag because a retag must not silently change a verdict.
    source_revision: str
    #: SHA-256 of ``model.onnx``, verified on load.
    onnx_sha256: str
    #: The ``laya`` release the export ran against, recorded next to the weights
    #: because the library moved three times in three days and a report naming
    #: only the checkpoint names half of what produced its number.
    laya_version: str
    #: The longest sequence the encoder was exported for. Truncation past this
    #: is what drops option markers, so the predictor refuses rather than truncates.
    max_len: int
    #: One temperature per :func:`calibration_key`. Empty means no fit has been
    #: applied, which :meth:`identity` reports rather than hides.
    temperatures: dict[str, float]
    render: RenderTemplate
    artifact_version: int = ARTIFACT_VERSION
    #: Which aorta produced it. Free text; not load-bearing.
    exported_by: str = ""

    def identity(self) -> str:
        """What :meth:`OnnxLayaPredictor.model_id` reports, inline in every report.

        Carries the checkpoint, the graph digest and the fit, because Decision 22
        asks for all three and because a report is read on its own -- copied,
        archived, attached to a ticket -- where nothing beside it can answer
        "which model said this?".
        """
        return (
            f"laya-onnx:{self.source_checkpoint}"
            f"@{self.onnx_sha256[:12]}"
            f"+fit:{self.fit_id()}"
        )

    def fit_id(self) -> str:
        """A short stable name for the temperature fit, or ``none``.

        ``none`` rather than an omission, so an artifact exported before Phase 1
        produced a fit says so in every line it appears in. The probabilities
        are still probabilities; they are just the model's own, and a threshold
        picked against a fitted number does not transfer to an unfitted one.
        """
        if not self.temperatures:
            return "none"
        payload = json.dumps(self.temperatures, sort_keys=True)
        return hashlib.blake2b(payload.encode("utf-8"), digest_size=4).hexdigest()

    def temperature_for(self, question: Question) -> float:
        """The fitted temperature for *question*'s bucket, or 1.0 -- the identity.

        1.0 for an unfitted bucket rather than an error: a fit covers the buckets
        the corpus happened to contain, and refusing to answer a question type
        nobody had labelled yet would take the whole node down for a gap in the
        training data. :meth:`fit_id` is what makes the gap visible.
        """
        return float(self.temperatures.get(calibration_key(question), 1.0))

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_version": self.artifact_version,
            "source_checkpoint": self.source_checkpoint,
            "source_revision": self.source_revision,
            "onnx_sha256": self.onnx_sha256,
            "laya_version": self.laya_version,
            "max_len": self.max_len,
            "temperatures": dict(sorted(self.temperatures.items())),
            "render": self.render.to_dict(),
            "exported_by": self.exported_by,
            "graph": {"inputs": list(GRAPH_INPUTS), "output": GRAPH_OUTPUT},
        }

    @classmethod
    def from_dict(cls, raw: Any) -> Manifest:
        if not isinstance(raw, dict):
            raise ArtifactUnavailable(
                f"an artifact manifest must be a JSON object, got {type(raw).__name__}"
            )
        version = raw.get("artifact_version")
        if version != ARTIFACT_VERSION:
            raise ArtifactUnavailable(
                f"artifact version {version!r} was exported against a different graph "
                f"signature; this build reads version {ARTIFACT_VERSION}. Re-export it."
            )
        required = (
            "source_checkpoint",
            "source_revision",
            "onnx_sha256",
            "laya_version",
            "max_len",
            "temperatures",
            "render",
        )
        missing = [key for key in required if key not in raw]
        if missing:
            raise ArtifactUnavailable(
                f"the artifact manifest is missing {missing}. Every one of these is "
                "part of what produced a probability, and an artifact that cannot "
                "say which weights and which fit answered is not usable as evidence "
                "(see docs/laya-packaging.md, Rule 2)."
            )
        temperatures = raw["temperatures"]
        if not isinstance(temperatures, dict):
            raise ArtifactUnavailable(
                f"the artifact manifest's 'temperatures' is a {type(temperatures).__name__}, "
                "not a map of calibration bucket to temperature"
            )
        try:
            fitted = {str(key): float(value) for key, value in temperatures.items()}
        except (TypeError, ValueError) as exc:
            raise ArtifactUnavailable(
                f"the artifact manifest has a non-numeric temperature: {exc}"
            ) from exc
        for key, value in fitted.items():
            if value <= 0.0:
                raise ArtifactUnavailable(
                    f"the artifact manifest fits temperature {value} for bucket {key!r}; "
                    "a temperature must be positive"
                )
        try:
            max_len = int(raw["max_len"])
        except (TypeError, ValueError) as exc:
            raise ArtifactUnavailable(
                f"the artifact manifest's 'max_len' is not an integer: {exc}"
            ) from exc
        if max_len <= 0:
            raise ArtifactUnavailable(
                f"the artifact manifest declares max_len={max_len}, so no question fits in it"
            )
        return cls(
            source_checkpoint=str(raw["source_checkpoint"]),
            source_revision=str(raw["source_revision"]),
            onnx_sha256=str(raw["onnx_sha256"]),
            laya_version=str(raw["laya_version"]),
            max_len=max_len,
            temperatures=fitted,
            render=RenderTemplate.from_dict(raw["render"]),
            artifact_version=ARTIFACT_VERSION,
            exported_by=str(raw.get("exported_by", "")),
        )


@dataclass(frozen=True)
class Artifact:
    """A loaded artifact directory: the manifest plus the two files beside it."""

    directory: Path
    manifest: Manifest

    @property
    def graph_path(self) -> Path:
        return self.directory / GRAPH_NAME

    @property
    def tokenizer_path(self) -> Path:
        return self.directory / TOKENIZER_NAME


def load_artifact(directory: str | Path, *, verify_digest: bool = False) -> Artifact:
    """Read and validate the artifact in *directory*.

    *verify_digest* re-hashes the graph, which costs a second or two on a
    few-hundred-megabyte file. Off by default because the chat UI loads this on
    the first routed turn and a user is waiting; worth turning on wherever a
    verdict is going to be published, which is the case Decision 22's Rule 2 is
    actually about. A corrupt copy is not the hazard -- a *different* copy is,
    and the manifest travels with the file it describes, so the digest catches
    the case where one of the two was replaced and the other was not.
    """
    root = Path(directory).expanduser()
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise ArtifactUnavailable(STAGING_PROCEDURE.format(path=root))

    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactUnavailable(f"could not read {manifest_path}: {exc}") from exc

    manifest = Manifest.from_dict(raw)
    artifact = Artifact(directory=root, manifest=manifest)

    for path, what in (
        (artifact.graph_path, "the exported ONNX graph"),
        (artifact.tokenizer_path, "the checkpoint's tokenizer"),
    ):
        if not path.is_file():
            raise ArtifactUnavailable(
                f"{manifest_path} describes an artifact but {what} is not beside it "
                f"(expected {path}). Copy the whole directory, not the manifest alone."
            )

    if verify_digest:
        actual = sha256_file(artifact.graph_path)
        if actual != manifest.onnx_sha256:
            raise ArtifactUnavailable(
                f"{artifact.graph_path} hashes to {actual[:12]} but its manifest "
                f"records {manifest.onnx_sha256[:12]}. One of the two was replaced "
                "without the other, so nothing here can say which weights would answer."
            )
    return artifact


__all__ = [
    "ARTIFACT_VERSION",
    "GRAPH_INPUTS",
    "GRAPH_OUTPUT",
    "MANIFEST_NAME",
    "STAGING_PROCEDURE",
    "TOKENIZER_NAME",
    "Artifact",
    "ArtifactUnavailable",
    "Manifest",
    "RenderTemplate",
    "calibration_key",
    "load_artifact",
    "sha256_file",
]
