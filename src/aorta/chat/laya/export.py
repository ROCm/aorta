"""Produce the ONNX artifact the chat path loads, from a torch checkpoint.

This module is the *other* side of Decision 22's split. It runs where torch is
allowed -- an ``[laya]`` install, on whatever machine holds the fine-tune -- and
writes a directory that :mod:`aorta.chat.laya.onnx_predictor` can load on a
``[chat-cli]`` install that has no torch at all. It is never imported by the
chat graph, and importing it costs nothing: every torch and ``laya`` import
happens inside the function that needs it, the same discipline
:class:`~aorta.laya.predictor.LayaAgentPredictor` keeps.

**It lives under** ``aorta/chat/`` **rather than under** ``aorta/laya/``
**because the artifact it writes is the chat path's format.** The schema in
:mod:`aorta.chat.laya.artifact` -- the graph signature, the marker convention,
the render template, what counts as identity -- is owned here, and an exporter
living away from the schema it targets is how the two drift. The import
direction permits it: ``tests/cli/test_chat_boundaries.py`` lets ``aorta.chat``
import ``aorta.*`` and forbids the reverse, so this module reading
:mod:`aorta.laya.predictor` is fine and an exporter under ``aorta/laya/``
reading the chat schema would not be.

**What it exports.** The encoder plus the decision head -- two transformer
layers and the option-marker scorer. Deliberately *not* the act/escalate head:
nothing on the chat path reads it. ``router_node`` and ``selector_node`` decide
with a threshold of their own against a per-question probability, so exporting
a head whose output nothing consumes would add a second, unvalidated number to
every artifact and invite somebody to threshold on it later.

**None of this has been run.** There is no Laya checkpoint and no torch on the
machine this was written on. The pure parts -- resolving the render template,
validating the temperature fit, assembling and writing the manifest, computing
digests -- are unit-tested against fakes in ``tests/chat/test_laya_export.py``.
The parts that are not:

* :func:`resolve_modules` probes the loaded agent for the encoder and the head
  by attribute name. Those names are a *probe*, not a verified API: 0.3.5's
  internals were not read, only its ``load`` / ``predict`` surface was. The
  function raises and lists what the object actually carries rather than
  guessing, so the first real run says what to fix instead of exporting a graph
  with the wrong subgraph in it.
* The ``torch.onnx.export`` call itself, and whether the exported graph agrees
  numerically with the torch path.
* Whether the render template can in fact be read off ``rl_agent_config.json``.
  :func:`resolve_render` looks under documented keys and refuses when they are
  absent, because the alternative -- a plausible default -- is precisely the
  train/serve skew this whole module is trying not to introduce.
"""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from aorta.chat.laya.artifact import (
    GRAPH_INPUTS,
    GRAPH_NAME,
    GRAPH_OUTPUT,
    MANIFEST_NAME,
    TOKENIZER_NAME,
    ArtifactUnavailableError,
    Manifest,
    RenderTemplate,
    sha256_file,
)

logger = logging.getLogger(__name__)

#: Where :func:`resolve_render` looks in the checkpoint configuration, in order.
#: A tuple per field so a config that spells one of them differently can be
#: handled by adding a key here rather than by passing the whole template in.
_RENDER_KEYS: dict[str, tuple[str, ...]] = {
    "state_template": ("state_template", "state_format", "state"),
    "question_template": ("question_template", "question_format", "question"),
    "option_template": ("option_template", "option_format", "option"),
    "separator": ("separator", "sep"),
    "marker": ("option_marker", "marker", "marker_token"),
}

#: Attribute names :func:`resolve_modules` probes on a loaded agent. Ordered
#: most-specific first. See this module's docstring: a probe, not an API.
_ENCODER_ATTRIBUTES = ("encoder", "enc", "backbone", "model")
_HEAD_ATTRIBUTES = ("decision_head", "head", "scorer", "option_head")


class ExportError(ArtifactUnavailableError):
    """The export cannot proceed, and says which part of the checkpoint is why.

    A subclass of the artifact error so that a caller holding one ``except``
for "this artifact is not usable" catches an export failure too -- they are
    the same problem seen from the two ends of the same file.
    """


def resolve_render(config: Mapping[str, Any], override: Mapping[str, Any] | None = None) -> RenderTemplate:
    """The render template for this checkpoint, from its config or from *override*.

    *override* wins entirely when given -- it is not merged. A half-overridden
    template would be the worst of both: some fields from the checkpoint, some
    from whoever ran the export, and nothing recording which came from where.

    When neither the config nor an override supplies a field, this raises. It
    would be easy to substitute something that looks right, and that is exactly
    the failure mode Track B found in the corpus builder: fine-tuning on one
    wording and asking another does not raise, it answers slightly worse for a
    reason nobody would go looking for. An operator who has the checkpoint can
    read its formatter and pass ``render=``; this module will not invent one.
    """
    if override is not None:
        return RenderTemplate.from_dict(dict(override))

    found: dict[str, Any] = {}
    nested = config.get("render") if isinstance(config.get("render"), Mapping) else {}
    for field, keys in _RENDER_KEYS.items():
        for key in keys:
            for source in (nested, config):
                if key in source:
                    found[field] = source[key]
                    break
            if field in found:
                break

    missing = [field for field in _RENDER_KEYS if field not in found]
    if missing:
        raise ExportError(
            f"the checkpoint configuration does not say how it renders {missing}. "
            f"Keys it does carry: {sorted(config)}.\n"
            "The ONNX path has to reproduce the checkpoint's own prompt format "
            "exactly -- a different wording does not fail, it answers worse -- so "
            "this export will not guess one. Read the format off the library's "
            "formatter and pass render={...} with "
            f"{sorted(_RENDER_KEYS)}."
        )
    return RenderTemplate(
        state_template=str(found["state_template"]),
        question_template=str(found["question_template"]),
        option_template=str(found["option_template"]),
        separator=str(found["separator"]),
        marker=str(found["marker"]),
    )


def resolve_modules(agent: Any) -> tuple[Any, Any]:
    """The encoder and the decision head on a loaded Laya agent.

    Probed by name and refused loudly, for the reason in this module's
    docstring. The error lists the attributes the object actually has, because
    the person hitting it is holding the checkpoint this code has never seen and
    that listing is the whole of what they need to fix the tuple above.
    """
    encoder = _first_attribute(agent, _ENCODER_ATTRIBUTES)
    head = _first_attribute(agent, _HEAD_ATTRIBUTES)
    if encoder is None or head is None:
        public = sorted(name for name in dir(agent) if not name.startswith("_"))
        wanted = "encoder" if encoder is None else "decision head"
        raise ExportError(
            f"could not find the {wanted} on the loaded agent "
            f"({type(agent).__name__}). Probed "
            f"{list(_ENCODER_ATTRIBUTES if encoder is None else _HEAD_ATTRIBUTES)}; "
            f"the object carries {public}. Add the right name to "
            "aorta/chat/laya/export.py rather than exporting the wrong subgraph."
        )
    return encoder, head


def _first_attribute(obj: Any, names: tuple[str, ...]) -> Any:
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return None


def check_temperatures(temperatures: Mapping[str, Any]) -> dict[str, float]:
    """The fit, validated, or a loud refusal. Pure.

    An empty fit is allowed. It is the honest state of things until the Phase 1
    harness has run, and :meth:`~aorta.chat.laya.artifact.Manifest.fit_id`
    reports it as ``fit:none`` in every line the artifact's identity reaches --
    so an unfitted artifact is usable and visibly unfitted, rather than
    unusable or invisibly unfitted.

    A temperature below 0.5 is warned about rather than refused, because that is
    the direction that does damage: below 1.0 sharpens logits instead of
    softening them, and the shipped ``choice:11+`` bucket at 0.1006 is why Laya
    0.3.5 clamps fitted temperatures into [0.5, 5.0] at all. A fit produced by
    our own harness on our own corpus is not subject to that clamp, so nothing
    here would otherwise say a word about a value that multiplies logics tenfold.
    """
    checked: dict[str, float] = {}
    for key, value in temperatures.items():
        try:
            temperature = float(value)
        except (TypeError, ValueError) as exc:
            raise ExportError(f"temperature for bucket {key!r} is not a number: {exc}") from exc
        if temperature <= 0.0:
            raise ExportError(
                f"temperature for bucket {key!r} is {temperature}; it must be positive"
            )
        if not 0.5 <= temperature <= 5.0:
            logger.warning(
                "temperature %.4f for bucket %s is outside the [0.5, 5.0] range Laya "
                "0.3.5 clamps its own fitted temperatures into. Below 0.5 sharpens "
                "the distribution rather than softening it, which publishes an "
                "uncertain answer as a confident one.",
                temperature,
                key,
            )
        checked[str(key)] = temperature
    return checked


def build_manifest(
    *,
    source_checkpoint: str,
    source_revision: str,
    onnx_sha256: str,
    laya_version: str,
    max_len: int,
    temperatures: Mapping[str, Any],
    render: RenderTemplate,
    exported_by: str = "",
) -> Manifest:
    """Assemble the manifest. Pure, so the schema can be tested with no weights."""
    if not source_revision:
        raise ExportError(
            "refusing to export without a source revision. Decision 22 pins a "
            "checkpoint by digest rather than by name because a retag must not "
            "silently change a verdict, and an artifact that cannot name the "
            "weights it came from cannot be compared against the next one."
        )
    return Manifest(
        source_checkpoint=source_checkpoint,
        source_revision=source_revision,
        onnx_sha256=onnx_sha256,
        laya_version=laya_version,
        max_len=max_len,
        temperatures=check_temperatures(temperatures),
        render=render,
        exported_by=exported_by,
    )


def write_manifest(directory: str | Path, manifest: Manifest) -> Path:
    """Write the manifest last, after the files it describes are in place.

    Last, deliberately. :func:`~aorta.chat.laya.artifact.load_artifact` keys off
    the manifest's presence, so writing it first would make a half-copied or
    half-written directory look loadable and fail at the session instead --
    several frames from the cause, in a Chainlit handler, mid-conversation.
    """
    path = Path(directory).expanduser() / MANIFEST_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n", "utf-8")
    return path


def export_artifact(
    checkpoint: str,
    directory: str | Path,
    *,
    temperatures: Mapping[str, Any],
    source_revision: str | None = None,
    render: Mapping[str, Any] | None = None,
    device: str = "cpu",
    load: Any = None,
    graph_writer: Any = None,
    opset: int = 17,
) -> Manifest:
    """Export *checkpoint* into *directory* and return the manifest written.

    *temperatures* is required and may be empty; see :func:`check_temperatures`.
    Required rather than defaulted so that exporting without a fit is a sentence
    somebody typed, not an omission -- the fit is half of what produces a
    probability, and Decision 22 asks for it to be recorded beside the weights.

    *load* and *graph_writer* are the injection points, mirroring
    :class:`~aorta.laya.predictor.LayaAgentPredictor`'s ``load=``. They are
    drawn where they are so that the torch-requiring work is exactly one call:
    everything this function does around :func:`write_torch_graph` -- resolving
    the template, probing the modules, validating the fit, digesting, ordering
    the writes -- is then testable on a machine that has neither torch nor a
    checkpoint, which is every machine this has run on so far.
    """
    target = Path(directory).expanduser()
    target.mkdir(parents=True, exist_ok=True)

    agent, laya_version = _load_agent(checkpoint, device=device, load=load)
    config = _config_of(agent)
    template = resolve_render(config, render)
    max_len = int(config.get("max_len") or 0)
    if max_len <= 0:
        raise ExportError(
            "the checkpoint configuration declares no positive 'max_len'. The "
            "predictor refuses to truncate rather than silently dropping option "
            "markers off the end of a long state, so it needs the real window; "
            f"the config carries {sorted(config)}."
        )

    encoder, head = resolve_modules(agent)
    graph_path = target / GRAPH_NAME
    writer = graph_writer or write_torch_graph
    writer(encoder, head, graph_path, max_len=max_len, opset=opset)
    if not graph_path.is_file():
        raise ExportError(
            f"the graph writer returned without writing {graph_path}. Nothing "
            "downstream can hash a file that is not there, and a manifest "
            "describing a missing graph is worse than no manifest."
        )
    _copy_tokenizer(agent, checkpoint, target)

    manifest = build_manifest(
        source_checkpoint=checkpoint,
        source_revision=source_revision or _revision_of(agent, checkpoint),
        onnx_sha256=sha256_file(graph_path),
        laya_version=laya_version,
        max_len=max_len,
        temperatures=temperatures,
        render=template,
        exported_by=_aorta_version(),
    )
    write_manifest(target, manifest)
    logger.info("exported %s to %s", manifest.identity(), target)
    return manifest


def _load_agent(checkpoint: str, *, device: str, load: Any) -> tuple[Any, str]:
    """The loaded checkpoint and the ``laya`` release that loaded it.

    Goes through :class:`~aorta.laya.predictor.LayaAgentPredictor` rather than
    calling ``laya.load`` directly, so that the checkpoint-name-or-local-
    directory resolution, the ``USE_TF=0`` guard and the "could not load"
    message are the same ones every other track gets. Reaching past that class
    to the library would be a second loader to keep in step with it.
    """
    from aorta.laya.predictor import LayaAgentPredictor

    predictor = LayaAgentPredictor(checkpoint, device=device, load=load)
    agent = predictor._agent()  # noqa: SLF001 - the export is inside this seam, not a consumer
    return agent, _laya_version()


def _laya_version() -> str:
    """The installed ``laya`` release, or ``unknown``.

    Recorded in the manifest because Decision 22 asks for the library version
    next to the checkpoint: 0.3.3, 0.3.4 and 0.3.5 shipped on three consecutive
    days and moved a floor inside that window, so "which laya" is part of the
    identity of a number and not a footnote.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("laya")
    except PackageNotFoundError:
        return "unknown"


def _aorta_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return f"amd-aorta {version('amd-aorta')}"
    except PackageNotFoundError:
        return "amd-aorta (unpackaged)"


def _config_of(agent: Any) -> Mapping[str, Any]:
    """The checkpoint's own config dict, as ``LayaAgentPredictor`` reads it."""
    config = getattr(agent, "cfg", None)
    if not isinstance(config, Mapping):
        raise ExportError(
            f"the loaded agent ({type(agent).__name__}) exposes no 'cfg' mapping, so "
            "neither its context window nor its prompt format can be read off it."
        )
    return config


def _revision_of(agent: Any, checkpoint: str) -> str:
    """The digest to pin this artifact to.

    A hub revision when the loader recorded one, the digest of the local weights
    file otherwise. Raises when neither is available rather than writing a
    placeholder: an artifact whose provenance is the string "local" cannot be
    compared against the next artifact, which is the entire point of recording it.
    """
    for attribute in ("revision", "commit_hash", "sha"):
        value = getattr(agent, attribute, None)
        if isinstance(value, str) and value:
            return value
    weights = Path(checkpoint).expanduser() / "model.safetensors"
    if weights.is_file():
        return f"sha256:{sha256_file(weights)}"
    raise ExportError(
        f"neither the loaded agent nor {checkpoint} names a revision, and there is "
        "no model.safetensors beside it to hash. Pass source_revision= with the "
        "digest of the weights being exported -- Decision 22 does not accept a "
        "name or a tag in its place."
    )


def _copy_tokenizer(agent: Any, checkpoint: str, target: Path) -> Path:
    """Put the checkpoint's own tokenizer beside the graph.

    Copied rather than referenced. The artifact is going to be rsynced to a node
    that has neither the checkpoint nor a HuggingFace cache, and a tokenizer
    resolved by name at load time on that node would either fail or -- worse --
    find a different revision of the same name and tokenise the same text into
    different ids than the graph was exported against.
    """
    destination = target / TOKENIZER_NAME
    tokenizer = getattr(agent, "tok", None)
    save = getattr(getattr(tokenizer, "backend_tokenizer", tokenizer), "save", None)
    if callable(save):
        save(str(destination))
        return destination

    source = Path(checkpoint).expanduser() / "tokenizer" / TOKENIZER_NAME
    if source.is_file():
        shutil.copyfile(source, destination)
        return destination
    raise ExportError(
        f"could not write {destination}: the loaded agent's tokenizer offers no "
        f"save(), and there is no {source}. The artifact has to carry its own "
        "tokenizer, or the node that loads it will tokenise differently than the "
        "graph was exported against."
    )


def write_torch_graph(
    encoder: Any,
    head: Any,
    path: Path,
    *,
    max_len: int,
    opset: int = 17,
) -> None:
    """Write the ONNX graph implementing the signature in ``artifact.py``.

    **This is the one function in this module that has never been run.** It
    needs torch and a real checkpoint, and neither has been available. It is
    deliberately the whole of the torch surface, so that what is unexercised is
    one named function rather than a thread running through the export.

    The wrapper is what fixes the graph signature: marker positions come in as
    an input and per-marker scores go out, so that finding the markers is
    Python in :mod:`~aorta.chat.laya.onnx_predictor` -- testable with no
    weights -- rather than ops inside a graph that cannot be inspected without
    them.

    ``dynamic_axes`` covers the sequence length and the marker count, because
    the selector's question count varies with how many tools are registered and
    a graph frozen at one shape would refuse the turn after a plugin loaded.
    """
    import torch

    class _MarkerScorer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = encoder
            self.head = head

        def forward(
            self,
            input_ids: Any,
            attention_mask: Any,
            marker_positions: Any,
        ) -> Any:
            hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            # The encoder returns either a tensor or an object carrying one;
            # both shapes exist across transformers versions, and picking one
            # here would tie the export to whichever the checkpoint's pinned
            # release happens to be.
            states = getattr(hidden, "last_hidden_state", hidden)
            selected = states[0].index_select(0, marker_positions)
            return self.head(selected).reshape(-1)

    module = _MarkerScorer().eval()
    example = (
        torch.ones((1, min(16, max_len)), dtype=torch.int64),
        torch.ones((1, min(16, max_len)), dtype=torch.int64),
        torch.tensor([1, 2], dtype=torch.int64),
    )
    with torch.no_grad():
        torch.onnx.export(
            module,
            example,
            str(path),
            input_names=list(GRAPH_INPUTS),
            output_names=[GRAPH_OUTPUT],
            dynamic_axes={
                "input_ids": {1: "sequence"},
                "attention_mask": {1: "sequence"},
                "marker_positions": {0: "markers"},
                GRAPH_OUTPUT: {0: "markers"},
            },
            opset_version=opset,
        )


__all__ = [
    "ExportError",
    "build_manifest",
    "check_temperatures",
    "export_artifact",
    "resolve_modules",
    "resolve_render",
    "write_manifest",
    "write_torch_graph",
]
