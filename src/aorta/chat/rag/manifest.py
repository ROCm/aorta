"""The manifest that travels with a chat index, and the checks it makes possible.

Decision 20a. An index is valid for exactly one tuple of (embedding identity,
dimensions, chunk params, store version, source commit) -- identity rather than
model name, because a remote model name means only what its endpoint says it
means, so the endpoint is part of it. The failure mode
when that tuple is wrong is the reason this module exists: a mismatched index
does not raise. It answers, fluently, from vectors that were never comparable
to the query's. For a debugging assistant that is worse than an outage, because
the user has no signal that anything is wrong.

So the policy is asymmetric on purpose:

* **Source drift warns.** An index built two weeks and forty commits ago is
  still mostly right, and refusing would leave the user with nothing.
* **An embedding-model or dimension mismatch refuses.** There is no partially
  correct answer available, and a warning printed above a confident answer is a
  warning nobody reads.

The manifest is a sidecar JSON file next to the ``.sqlite``, not a table inside
it, so CI can publish and a user can inspect it without loading sqlite-vec.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Bumped when a field is removed or its meaning changes. A reader tolerates a
#: manifest carrying *extra* keys (a newer builder adding a field must not
#: strand an older client) but refuses one it cannot interpret at all.
SCHEMA_VERSION = 1

#: The on-disk store this manifest describes. Recorded rather than assumed
#: because the format changed once already (Chroma -> sqlite-vec) and a
#: published artifact outlives the decision that produced it.
STORE_NAME = "sqlite-vec"

#: Suffixes of the two files published alongside the index. Both are derived
#: from the index filename so a side-loaded set stays self-describing after a
#: user renames it.
MANIFEST_SUFFIX = ".manifest.json"
CHECKSUM_SUFFIX = ".sha256"

#: Read/written in 1 MiB blocks: the index is tens of megabytes, and hashing it
#: must not hold a second copy in memory on a node that is already tight.
_HASH_BLOCK = 1024 * 1024

#: How to name a parsed value's type in a message someone has to act on. JSON's
#: vocabulary rather than Python's, because the thing being described is a file
#: the user can open and edit.
_TYPE_NAMES: dict[type, str] = {
    str: "a string",
    bool: "a boolean",
    int: "a whole number",
    float: "a fractional number",
    list: "a list",
    dict: "an object",
    type(None): "null",
}

#: The same vocabulary for what a field wanted, keyed by the annotation
#: :class:`Manifest` declares it with. ``from __future__ import annotations``
#: makes every annotation a string, which is what makes this table checkable
#: against the dataclass -- see the import-time guard below the class.
_SHAPE_NAMES: dict[str, str] = {
    "str": "a string",
    "int": "a whole number",
    "list[str]": "a list of strings",
}


class ManifestError(RuntimeError):
    """The manifest is missing, unreadable, or not a manifest."""


class IndexMismatchError(RuntimeError):
    """The index cannot answer queries from this install's embedding provider.

    Raised rather than logged. See the module docstring: the alternative to
    raising is a plausible wrong answer.
    """


def _fits_shape(value: Any, shape: str) -> bool:
    """Whether a parsed JSON value can fill a field declared as ``shape``."""
    if shape == "str":
        return isinstance(value, str)
    if shape == "int":
        # ``bool`` is an ``int`` subclass, so ``true`` in a count or dimension
        # field would otherwise pass as the number 1 and produce a refusal
        # naming a value nobody wrote.
        return isinstance(value, int) and not isinstance(value, bool)
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def manifest_path(index_path: str | Path) -> Path:
    """Sidecar manifest path for an index file."""
    return Path(str(index_path) + MANIFEST_SUFFIX)


def checksum_path(index_path: str | Path) -> Path:
    """Sidecar SHA256 path for an index file."""
    return Path(str(index_path) + CHECKSUM_SUFFIX)


def sha256_file(path: str | Path) -> str:
    """Hex SHA256 of a file, read incrementally."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(_HASH_BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class Manifest:
    """What a published index says about itself.

    ``index_sha256`` covers the ``.sqlite`` only. The manifest cannot hash
    itself, which is why the published set also carries a ``.sha256`` file: the
    checksum is verified against the download before the manifest is trusted.

    It describes the index *as built or published*, and is a transport check:
    it is what proves a download or a hand-carried copy arrived intact. It is
    deliberately not what the load path verifies, because the file legitimately
    stops matching it -- the per-user run collection lives in the same
    ``.sqlite`` (``rag/runs.py``) and rewrites it on its own cadence. The
    load-time equivalent is ``chunk_count`` checked against the source
    collection's live row count, which is scoped to the half this manifest
    actually describes.
    """

    aorta_version: str
    aorta_sha: str
    embedding_provider: str
    embedding_model: str
    dimensions: int
    collection: str
    chunk_size: int
    chunk_overlap: int
    index_sha256: str
    schema_version: int = SCHEMA_VERSION
    aorta_tag: str = ""
    store: str = STORE_NAME
    store_version: str = ""
    built_at: str = ""
    #: The provider's full vector identity -- for a remote provider, endpoint
    #: and model rather than model alone. Defaulted rather than required so a
    #: manifest written before the field existed still parses; ``validate``
    #: therefore only compares it when it is present, and the collection check
    #: is what fails such a manifest closed, since the collection name carries
    #: a digest of this same identity.
    embedding_identity: str = ""
    corpus_digest: str = ""
    corpus_roots: list[str] = field(default_factory=list)
    file_count: int = 0
    chunk_count: int = 0

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Manifest:
        """Build from a parsed manifest, dropping keys this version predates.

        Forward-tolerant about *keys* by design: a newer CI job adding a field
        must not make its index unreadable to an older client, since the two are
        published and installed independently.

        Strict about *types*, which is the opposite of the coercion
        ``agent.llm.AgentStep.from_dict`` applies to untrusted model output, and
        deliberately so. That one is filling in a field it can do without; this
        one decides whether an index is trusted to answer questions, and a
        sidecar coerced into a plausible shape is precisely the silently-wrong
        answer this module exists to prevent. So a value of the wrong type is a
        broken manifest, reported as one.

        Every field here is declared ``str``, ``int`` or ``list[str]``, and
        nothing downstream re-checks: ``describe`` slices ``aorta_sha``,
        ``validate`` slices it twice more and calls ``startswith`` on it,
        ``ensure_supported_schema`` compares ``schema_version``, and
        ``index_ops`` walks ``corpus_roots``. Before this check, a sidecar
        carrying ``"aorta_sha": 42`` reached ``build_index`` as ``TypeError:
        'int' object is not subscriptable`` and one carrying ``{"sha": "abc"}``
        as ``KeyError: slice(None, 7, None)`` -- neither of which any caller
        handles, because callers handle :class:`ManifestError`. Checking the
        declared types once, here at the only boundary parsed JSON crosses,
        closes those and every future one: the alternative is a type guard at
        each use site, and the one that gets forgotten is the one a user hits.

        No field is exempt, including the ones only ever *reported* --
        ``built_at`` carrying a number refuses the manifest as surely as
        ``dimensions`` does. That is not the usual asymmetry (a reported field
        can survive being slightly wrong; a used one cannot) because the
        subject here is not the field, it is the file. A sidecar whose types do
        not match the format it declares is evidence the file did not arrive
        intact, and this module's whole policy is that an index it cannot
        vouch for must not answer questions -- see the module docstring. The
        remedy is one command and the message names it.

        Raises:
            ManifestError: If ``raw`` is not an object, is missing a required
                field, or carries a field whose type is not the declared one.
        """
        if not isinstance(raw, dict):
            raise ManifestError(f"manifest is a {type(raw).__name__}, not an object")
        known = {field_.name: field_.type for field_ in fields(cls)}
        unknown = sorted(set(raw) - set(known))
        if unknown:
            logger.debug("Ignoring unknown manifest key(s): %s", ", ".join(unknown))
        accepted = {key: value for key, value in raw.items() if key in known}

        wrong = [
            f"{key} is {_TYPE_NAMES.get(type(value), type(value).__name__)}, not "
            f"{_SHAPE_NAMES[known[key]]}"
            for key, value in sorted(accepted.items())
            if not _fits_shape(value, known[key])
        ]
        if wrong:
            raise ManifestError(
                f"manifest field(s) are not the type the format declares: {'; '.join(wrong)}. "
                f"Replace it with {_refresh_advice()}."
            )

        try:
            return cls(**accepted)
        except TypeError as exc:
            raise ManifestError(f"manifest is missing required field(s): {exc}") from exc

    def describe(self) -> str:
        """One line for ``doctor`` and for a warning that has to name the index."""
        source = self.aorta_tag or (self.aorta_sha[:7] if self.aorta_sha else "unknown")
        return (
            f"aorta {self.aorta_version} ({source}), {self.embedding_model} "
            f"@ {self.dimensions}d, chunks {self.chunk_size}/{self.chunk_overlap}, "
            f"built {self.built_at or 'unknown'}"
        )


# A field added in a shape ``_fits_shape`` does not know about would otherwise
# skip validation silently, which is the failure mode the check above exists to
# remove -- so it fails at import instead, where a test collection run finds it.
_UNKNOWN_SHAPES = sorted({field_.type for field_ in fields(Manifest)} - set(_SHAPE_NAMES))
if _UNKNOWN_SHAPES:  # pragma: no cover - a guard on this module's own edits
    raise AssertionError(
        f"Manifest declares field type(s) {_UNKNOWN_SHAPES} that _fits_shape does not "
        "check; add them to _SHAPE_NAMES and _fits_shape or from_dict will accept "
        "anything there"
    )


def now_stamp() -> str:
    """Build timestamp, UTC and second-resolution."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_manifest(index_path: str | Path, manifest: Manifest) -> Path:
    """Write the sidecar checksum and the sidecar manifest, returning the latter.

    The manifest goes last because it is the file every load path gates on. An
    interruption between the two therefore leaves the previous manifest over
    the new index, which the contents check refuses -- rather than a current
    manifest beside a checksum describing something else.
    """
    target = manifest_path(index_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Two spaces is sha256sum's own format, so `sha256sum -c` works on the
    # published file without reformatting.
    checksum_path(index_path).write_text(
        f"{manifest.index_sha256}  {Path(index_path).name}\n", encoding="utf-8"
    )
    target.write_text(manifest.to_json(), encoding="utf-8")
    return target


def read_manifest(index_path: str | Path) -> Manifest:
    """Read the sidecar manifest for an index file.

    Raises:
        ManifestError: If it is absent, unparseable, or from a schema this
            version cannot interpret.
    """
    path = manifest_path(index_path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ManifestError(
            f"no manifest beside the index at {index_path}.\n"
            "An index without one cannot be checked against this install's "
            "embedding model, which is the check that stops a silently "
            "mismatched index answering from the wrong vectors. Replace it "
            f"with {_refresh_advice()}."
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"could not read the manifest at {path}: {exc}") from exc

    manifest = Manifest.from_dict(raw)
    ensure_supported_schema(manifest, f"the index at {index_path}")
    return manifest


def ensure_supported_schema(manifest: Manifest, subject: str) -> None:
    """Raise unless this build can interpret ``manifest``'s schema version.

    Separate from :meth:`Manifest.from_dict` so parsing stays independent of
    policy, and shared so that every path which *adopts* a manifest applies it.
    The download path did not, so an older client installed an index it would
    then refuse on first load -- a successful fetch followed by a broken chat.

    A non-integer version is rejected explicitly rather than compared: ``>``
    against a string raises ``TypeError``, which escapes callers that handle
    only :class:`ManifestError`.

    That check is defence in depth now rather than the first line of it.
    :meth:`Manifest.from_dict` type-checks every declared field, so a *parsed*
    manifest cannot reach here with a non-integer version, and both callers
    parse. It is kept because this function is exported and the comparison is
    its own to make safe: a caller holding a hand-built manifest is the one
    case the parser never saw.
    """
    version = manifest.schema_version
    if isinstance(version, bool) or not isinstance(version, int):
        raise ManifestError(
            f"the manifest for {subject} carries a non-integer schema version "
            f"({version!r}), so it is malformed and cannot be interpreted. "
            f"Replace it with {_refresh_advice()}."
        )
    if version > SCHEMA_VERSION:
        raise ManifestError(
            f"{subject} carries manifest schema version {version}, but this "
            f"aorta understands {SCHEMA_VERSION}. Upgrade aorta, or rebuild the "
            "index locally with 'aorta chat index build'."
        )


@dataclass
class ValidationReport:
    """Outcome of checking a manifest against the configured provider."""

    manifest: Manifest
    warnings: list[str] = field(default_factory=list)
    refusals: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.refusals and not self.warnings

    def raise_if_refused(self, index_path: str | Path) -> None:
        if self.refusals:
            raise IndexMismatchError(_refusal_text(index_path, self.refusals, self.manifest))


def _configured_embedding_provider() -> str:
    """The provider this install embeds with, for choosing a remedy.

    Asked of the factory rather than read off ``settings.embedding_provider``,
    because that setting is a spelling, not the answer: ``onnx`` and
    ``fastembed`` are documented aliases of ``local``, so comparing the raw
    string would hand a perfectly ordinary local install the remote remedy --
    withholding the one command that fixes it. ``doctor`` already asks the
    built provider for its ``name``; this asks the same question the same way.

    Imported lazily so this module stays importable without the chat extra, and
    defaulted to ``local`` when the provider cannot be resolved at all: that is
    the shipped default and the one the published index is built with, so an
    unreadable configuration is far likelier to be a local one than a remote
    one.

    That fallback used to cover two states and reason about only one of them.
    A name the factory does not have is a *configuration error* -- deterministic,
    and the value is right there in the message -- and resolving it to ``local``
    made the advice arms offer index commands that die resolving the same
    setting. :func:`_unknown_embedding_provider` now answers that case before
    any caller reaches here, so what is left for this fallback is what its
    reasoning was always about: an environment that cannot construct a provider
    it can name.
    """
    try:
        from aorta.chat.rag.embeddings.factory import get_provider

        return get_provider().name
    except Exception:
        logger.debug("could not resolve the configured embedding provider", exc_info=True)
        return "local"


def _remote_embedder_error() -> str:
    """Why a remote embedding build would fail before it started, or "".

    Every remedy below that names ``aorta chat index build`` under a remote
    provider is naming a command that sends each chunk of the corpus through an
    embeddings API -- and ``RemoteApiProvider.get_embeddings`` raises on an
    empty ``remote_embedding_api_key`` before it sends anything. No profile
    template has ever prompted for that key and it does not fall back to the
    chat one, so the install being advised is very often the install that
    cannot take the advice. Offering it anyway is the same defect as offering
    ``index fetch`` to a remote embedder, one layer further in.

    Asked of the provider rather than read off ``settings`` so it cannot drift
    from the precondition the command will actually hit, and constructed
    directly rather than through the factory: a caller may be asking about
    ``remote`` while the configured provider is local, and instantiating *that*
    one would download a model to answer a question about wording. Building the
    client is offline either way.
    """
    try:
        # Imported inside the try, not above it: ``remote_api`` pulls in the
        # OpenAI client, and an install missing the chat-cli extra would
        # otherwise take a refusal message down with an ImportError.
        from aorta.chat.rag.embeddings.remote_api import RemoteApiProvider

        RemoteApiProvider().get_embeddings()
    except Exception as exc:
        # Broad on purpose: the question is whether the client can be built at
        # all, and every way it cannot is a build that fails on chunk one.
        return str(exc)
    return ""


def _unknown_embedding_provider() -> str:
    """The factory's complaint about ``embedding_provider``, or "".

    The third of these predicates, and the one furthest out: before asking
    whether a *provider* can embed, ask whether the setting names a provider at
    all. ``get_provider`` raises ``ValueError`` listing the registered names,
    which is a better message than anything reworded here, so it is returned
    verbatim.

    ``ValueError`` alone, deliberately. That is the exception the factory
    raises for a name it does not have, and the one
    :func:`~aorta.chat.doctor._check_embedding_model` already catches to report
    the same state. Anything else -- a missing extra, a provider whose
    constructor fails here -- is not a claim that the *name* is wrong, and is
    left to :func:`_configured_embedding_provider`'s fallback, whose reasoning
    still holds for it.
    """
    try:
        from aorta.chat.rag.embeddings.factory import get_provider

        get_provider()
    except ValueError as exc:
        return str(exc)
    except Exception:
        logger.debug("could not resolve the embedding provider", exc_info=True)
    return ""


def _custom_local_model() -> str:
    """The configured local model when it is not the published one, else "".

    The local half of :func:`_remote_embedder_error`, and the same defect one
    setting over. ``embedding_model`` is configurable, but CI publishes exactly
    one asset and builds it with ``fastembed_bge.DEFAULT_MODEL`` -- so
    ``fetch_index`` validates the published manifest against this install's
    provider identity and refuses on all three of embedding model, collection
    and embedding identity when they differ. Offering the fetch to such an
    install names a command whose only possible outcome is a refusal.

    Deliberately reads the model rather than the chunk settings.
    ``chunk_size``/``chunk_overlap`` drift is a *warning* in ``validate`` and
    ``fetch_index`` installs through it, so conditioning on those would
    withhold a fetch that works.
    """
    # Imported lazily for the same reason as its neighbours: this module stays
    # importable without the chat extra.
    from aorta.chat.config import settings
    from aorta.chat.rag.embeddings import fastembed_bge

    # Compared unstripped, and that is the point. ``model_id()``,
    # ``collection_name()`` and ``vector_identity()`` all read
    # ``settings.embedding_model`` verbatim, so "  BAAI/bge-small-en-v1.5  " is
    # a *different* model as far as ``fetch_index`` is concerned -- it refuses
    # all three of embedding model, collection and identity. Stripping here
    # would call that value the published default and offer the fetch it
    # refuses, which is the exact defect this function exists to prevent.
    configured = settings.embedding_model or fastembed_bge.DEFAULT_MODEL
    return "" if configured == fastembed_bge.DEFAULT_MODEL else configured


def _refresh_advice(embedding_provider: str | None = None) -> str:
    """:func:`_refresh_command`, quoted, with any precondition it depends on.

    The quoting lives here rather than at the four call sites so the
    precondition cannot be dropped by one of them: a bare ``'{command}'`` in an
    f-string reads as a command that runs, and for a keyless remote install it
    is not.
    """
    if embedding_provider is None and _unknown_embedding_provider():
        # ``_refresh_command`` is not consulted here at all. It answers "which
        # of the two index commands suits this provider", and for a provider
        # aorta does not have there is no answer -- returning one and then
        # disowning it in the same sentence is how these slots went wrong
        # before.
        return (
            "no index command can run until embedding_provider names a provider "
            "aorta has -- 'aorta chat doctor' lists them"
        )
    command = _refresh_command(embedding_provider)
    provider = (embedding_provider or _configured_embedding_provider()).strip().lower()
    if provider == "local":
        custom = _custom_local_model()
        if not custom:
            return f"'{command}'"
        return (
            f"'{command}' -- the published index is built with the default "
            f"embedding_model, so it cannot be read by {custom}"
        )
    if not _remote_embedder_error():
        return f"'{command}'"
    return (
        # Names the blocker and points at the full remedy rather than inlining
        # it. These are one-line slots inside a warning, and spelling out the
        # switch to local here would put ``index fetch`` back into a remote
        # install's messages -- the thing this predicate exists to keep out.
        f"'{command}', which cannot run until remote_embedding_api_key is set "
        "-- 'aorta chat doctor' names the alternative"
    )


def _refresh_command(embedding_provider: str | None = None) -> str:
    """The single command that gets this install a current index, named inline.

    ``remedy_lines`` is the block form, for the places that can spend several
    lines on it; this is for the sentence that only has room for one command.
    Conditional on the same two facts -- the provider and, for a local one, the
    model -- so the two cannot disagree about whether a fetch is worth
    suggesting.

    Callers want :func:`_refresh_advice`, which quotes this and adds any
    precondition the command depends on. This returns the bare command, so
    that a message can say which one it means without asserting it can run.

    Not asked at all when ``embedding_provider`` names a provider that does not
    exist: the question this answers is *which of the two index commands suits
    this provider*, and for that state the answer is neither.
    ``_refresh_advice`` short-circuits before it rather than taking a command
    from here and disowning it in the same sentence.
    """
    provider = (embedding_provider or _configured_embedding_provider()).strip().lower()
    if provider != "local" or _custom_local_model():
        return "aorta chat index build"
    return "aorta chat index fetch"


def remedy_lines(
    embedding_provider: str | None = None,
    *,
    include_doctor: bool = True,
) -> list[str]:
    """The commands that get this install an index it can query.

    Conditional on the provider because ``index fetch`` cannot help a remote
    embedder: CI publishes one asset, built with the local provider, so "the
    index matching this install" does not exist for a remote one and never
    will. Offering it first sends the user to a refusal with different wording,
    from which the reasonable conclusion is that chat is broken.

    Conditional on ``remote_embedding_api_key`` for the same reason one layer
    in: a remote provider that cannot build a client cannot run ``index build``
    either, so an install with neither key nor index is offered no index
    command at all, and led to the switch back to local instead.

    Conditional on ``embedding_model`` for the mirror image of the first
    reason. CI publishes one asset built with the default model, so a local
    install that queries with any other one has ``fetch_index`` refuse it --
    the fetch is withheld there too, and the build named instead. Not
    conditional on the chunk settings, whose drift ``validate`` warns about and
    ``fetch_index`` installs through.

    Conditional on ``embedding_provider`` naming a provider that *exists*,
    which is the one condition that withholds both commands rather than
    choosing between them: ``fetch_index`` and the build each resolve the
    provider first, so a name the factory does not have fails both
    identically. That state is answered before the provider is resolved at
    all, because resolving it is what used to turn a typo into two dead
    commands.

    Every one of these is about the same thing: the difference between advice
    and a dead end.

    Worded for an index that is absent as much as for one that is refused,
    because both states want the same list and a remedy that is only correct
    for one of them is how the second state keeps the impossible command.

    Args:
        embedding_provider: Override for the configured provider, as one of
            the factory's canonical names. ``None`` resolves it from the
            configured provider.
        include_doctor: Whether to suggest ``aorta chat doctor``. Off for
            ``doctor``'s own report, which is already that output.
    """
    doctor_line = ["  aorta chat doctor          show what the two sides currently disagree on"]

    # Asked before the provider is resolved, and only when the caller did not
    # name one: an explicit argument means "assume this provider", and reading
    # the setting underneath it would answer a question nobody asked.
    if embedding_provider is None:
        unknown = _unknown_embedding_provider()
        if unknown:
            # Neither index command is offered because neither can start:
            # ``fetch_index`` and the build both resolve the provider before
            # anything else and both raise this same error. Offering them was
            # how a typo in one setting turned into two dead commands.
            return [
                '  embedding_provider = "local"',
                "                             in chat.toml, or the environment variable",
                "                             AORTA_CHAT_EMBEDDING_PROVIDER=local for a",
                "                             single session",
                *(doctor_line if include_doctor else []),
                "",
                "No index command is offered, because neither can start:",
                # On its own line: the factory's message carries the offending
                # value and the valid set, so its length is not ours to wrap.
                f"  {unknown}",
                "Both 'aorta chat index fetch' and 'aorta chat index build' resolve",
                "the provider before anything else and fail with that same error. So",
                "this is a configuration error rather than a missing index -- naming a",
                "provider aorta has is the only remedy that runs, and both index",
                "commands become available again once it does.",
            ]

    provider = (embedding_provider or _configured_embedding_provider()).strip().lower()

    if provider == "local":
        custom = _custom_local_model()
        if not custom:
            return [
                "  aorta chat index fetch     replace this install's index with the\n"
                "                             published one, built for this version",
                "  aorta chat index build     build one locally with the configured provider",
                *(doctor_line if include_doctor else []),
            ]
        # Same shape as the remote arm: withhold the command that can only
        # refuse, and say why, because a list that silently drops the
        # documented first remedy reads as a list that forgot it.
        return [
            "  aorta chat index build     build one locally with the configured provider",
            *(doctor_line if include_doctor else []),
            "",
            "'aorta chat index fetch' is not offered here: CI publishes one asset",
            "and builds it with the default embedding_model, so fetching it under",
            f"embedding_model = {custom!r}",
            "is refused on the embedding model, the collection and the embedding",
            "identity alike. Building with the configured model is the remedy;",
            "setting embedding_model back to the default makes the fetch work again.",
        ]

    unavailable = _remote_embedder_error()
    if unavailable:
        # Leading with ``index build`` here would name the one command this
        # install is guaranteed to fail: the provider raises before the first
        # chunk is sent. So the order flips -- the setting change is the only
        # remedy that runs, and the build is listed as what becomes possible
        # rather than as what to do now.
        # The provider's first sentence, not its whole message: the rest of it
        # is remedies of its own, and repeating them would put the user in
        # front of two differently-worded lists for one problem.
        blocker = unavailable.split(". ")[0].rstrip(".")
        return [
            '  embedding_provider = "local"',
            "                             in chat.toml, or the environment variable",
            "                             AORTA_CHAT_EMBEDDING_PROVIDER=local for a",
            # Kept on one line: a command name broken across a wrap cannot be
            # copied out of the report in one go.
            "                             single session, after which",
            "                             'aorta chat index fetch' works",
            *(doctor_line if include_doctor else []),
            "",
            "Neither index command is offered as-is. The published index is built with",
            "the local embedder, so no published asset can match a remote one -- and",
            "'aorta chat index build' would embed the corpus through the embeddings",
            "API, which this install cannot reach:",
            f"  {blocker}.",
        ]

    return [
        "  aorta chat index build     embed the corpus with the configured",
        "                             provider -- slow, and every chunk goes",
        "                             through the embeddings API",
        *(doctor_line if include_doctor else []),
        "",
        "'aorta chat index fetch' is not offered here: the published index is built",
        "with the local embedder, so no published asset can match a remote one, and",
        f"fetching it under embedding_provider = {provider!r} would be refused in",
        'turn. Set embedding_provider = "local" (or the environment variable',
        "AORTA_CHAT_EMBEDDING_PROVIDER=local) and the fetch works, at no cost in",
        "embedding API calls.",
    ]


def _refusal_text(index_path: str | Path, refusals: list[str], manifest: Manifest) -> str:
    """Compose the refusal. Its job is to be impossible to skim past.

    It leads with the consequence rather than the mismatch, because the
    mismatch is not self-evidently serious to someone who just wants an answer,
    and it ends with concrete commands, because a refusal the user cannot act
    on gets worked around.
    """
    rule = "=" * 72
    lines = [
        "",
        rule,
        f"REFUSING to query the chat index at {index_path}",
        rule,
        "",
        "It was not built by the embedding provider this install queries with, so",
        "every retrieval would compare vectors that are not comparable.",
        "",
        "This would not error. You would get confident answers assembled from the",
        "wrong chunks, with nothing on screen to say so.",
        "",
        *(f"  - {line}" for line in refusals),
        "",
        f"Index built as: {manifest.describe()}",
        "",
        "Resolve it by one of:",
        *remedy_lines(),
        "",
    ]
    return "\n".join(lines)


def _readable_identity(identity: str) -> str:
    """Render a newline-joined vector identity on one line."""
    return " / ".join(part for part in identity.split("\n") if part)


def validate(
    manifest: Manifest,
    *,
    embedding_model: str,
    collection: str,
    embedding_identity: str = "",
    dimensions: int | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    installed_version: str = "",
    installed_sha: str = "",
    chunk_count: int | None = None,
) -> ValidationReport:
    """Check a manifest against what this install would query with.

    ``dimensions`` is optional because the caller may not know it before the
    model is loaded, and loading it costs a 65 MB download on a cold cache. The
    model and collection checks are the load-bearing ones; the dimension check
    is a second, cheaper net for a model whose dimensions changed under a
    stable name.

    ``chunk_count`` is the collection's live row count, when the caller has
    opened the store far enough to know it. It is what catches a manifest that
    survived while the index under it did not -- an interrupted build, a
    truncated copy -- which the field checks above cannot see, because they
    only ever read the sidecar.

    Deliberately schema-agnostic: this does not call
    :func:`ensure_supported_schema`, because both paths that turn bytes into a
    :class:`Manifest` already have -- ``read_manifest`` before returning one,
    and ``index_ops._parse_manifest`` before the fetched asset is installed. A
    manifest from a version this build cannot interpret therefore never reaches
    here, and repeating the check would put the policy in two places for a
    caller that cannot be reached without it. Field *types* are guaranteed by
    :meth:`Manifest.from_dict`, which is why the slices below need no guard.
    """
    report = ValidationReport(manifest=manifest)

    if manifest.embedding_model != embedding_model:
        report.refusals.append(
            f"embedding model: index was built with {manifest.embedding_model!r}, "
            f"this install queries with {embedding_model!r}"
        )
    if manifest.collection != collection:
        report.refusals.append(
            f"collection: index holds {manifest.collection!r}, this install reads "
            f"{collection!r}. The trailing digest covers the whole embedding "
            "identity, so the endpoint may have changed even where the model "
            "name did not"
        )
    # Only when the manifest carries one: an older manifest has no identity to
    # compare, and is already refused above on the collection name. Checking
    # this too is what turns that refusal from two opaque digests into a line
    # naming the endpoint that changed.
    if (
        manifest.embedding_identity
        and embedding_identity
        and manifest.embedding_identity != embedding_identity
    ):
        report.refusals.append(
            f"embedding identity: index was built against "
            f"{_readable_identity(manifest.embedding_identity)}, this install "
            f"embeds with {_readable_identity(embedding_identity)}"
        )
    if dimensions is not None and manifest.dimensions != dimensions:
        report.refusals.append(
            f"dimensions: index was built at {manifest.dimensions}, this install "
            f"produces {dimensions}"
        )

    if manifest.store != STORE_NAME:
        report.refusals.append(
            f"store format: index is {manifest.store!r}, this aorta reads {STORE_NAME!r}"
        )

    # A refusal, not a warning, and for the same reason as the model check: an
    # index holding a third of its chunks does not fail, it answers from the
    # third that survived. ``chunk_count`` of 0 in the manifest means a builder
    # that predates the field rather than an empty index, so it is not checked.
    if chunk_count is not None and manifest.chunk_count and manifest.chunk_count != chunk_count:
        report.refusals.append(
            f"contents: the manifest describes {manifest.chunk_count} chunks, but "
            f"the index holds {chunk_count}. The manifest and the index it "
            "describes are from different builds -- most likely a build or "
            "install that was interrupted part way"
        )

    # Chunk params do not invalidate an index -- the vectors are still the same
    # model's, and the chunker only ran at build time -- but a mismatch means
    # retrieved spans are not the size the prompt budget was tuned for.
    if chunk_size is not None and manifest.chunk_size != chunk_size:
        report.warnings.append(
            f"chunk size: index was built at {manifest.chunk_size}, configured "
            f"value is {chunk_size}; retrieved spans will not be the size this "
            "install expects"
        )
    if chunk_overlap is not None and manifest.chunk_overlap != chunk_overlap:
        report.warnings.append(
            f"chunk overlap: index was built at {manifest.chunk_overlap}, "
            f"configured value is {chunk_overlap}"
        )

    # Named through ``_refresh_command`` for the reason ``remedy_lines`` is
    # conditional: on a remote embedder a fetch is refused rather than stale,
    # so advising it here would answer a warning about an index that still
    # works with a command that cannot run.
    if installed_version and manifest.aorta_version != installed_version:
        report.warnings.append(
            f"source drift: index was built from aorta {manifest.aorta_version}"
            f"{f' ({manifest.aorta_sha[:7]})' if manifest.aorta_sha else ''}, this "
            f"install is {installed_version}. Answers may cite code that has "
            f"since changed; refresh with {_refresh_advice()}"
        )
    elif installed_sha and manifest.aorta_sha and not manifest.aorta_sha.startswith(installed_sha):
        report.warnings.append(
            f"source drift: index was built at {manifest.aorta_sha[:7]}, this "
            f"install reports {installed_sha}. Refresh with {_refresh_advice()}"
        )

    return report


__all__ = [
    "CHECKSUM_SUFFIX",
    "MANIFEST_SUFFIX",
    "SCHEMA_VERSION",
    "STORE_NAME",
    "IndexMismatchError",
    "Manifest",
    "ManifestError",
    "ValidationReport",
    "checksum_path",
    "ensure_supported_schema",
    "manifest_path",
    "now_stamp",
    "read_manifest",
    "remedy_lines",
    "sha256_file",
    "validate",
    "write_manifest",
]
