"""``aorta chat doctor`` -- what is installed, what is reachable, what is stale.

No Click; ``cli/chat.py`` renders :class:`Check` records.

Every check reports rather than raises, and each one runs even if an earlier one
failed. That is the whole point of a doctor command: a user whose chat session
just failed wants the full list, not the first item on it. The command's own
exit status is derived at the end, in the CLI.

The embedding-model check is the one that earns this command a place in Phase 4.
Decision 21b publishes the index but not the model, so an air-gapped user is
blocked twice and only discovers the second blocker when ``fastembed`` raises a
HuggingFace connection error -- which reads as a bug in aorta, not as "pre-seed
a cache". So when the weights are absent, this probes HuggingFace, and when that
probe fails it prints the exact procedure. Documentation does not reach someone
whose command just failed.
"""

from __future__ import annotations

import json
import logging
import re
import shlex
import socket
import sys
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Status values, worst last. The CLI exits non-zero on ``fail``.
OK = "ok"
WARN = "warn"
FAIL = "fail"
SKIP = "skip"

#: HuggingFace host the model would be downloaded from, and how long it gets to
#: answer. Short on purpose: this runs while the user waits, and a slow probe
#: and an unreachable host lead to the same advice.
_HF_HOST = "huggingface.co"
_HF_PORT = 443
_HF_PROBE_TIMEOUT = 3.0

#: The LLM backend's budget, for the same reason and on the same scale. A local
#: server that needs longer than this to answer ``/health`` is not one a chat
#: session can use yet either, so waiting minutes here only delays the same
#: advice.
_BACKEND_PROBE_TIMEOUT = 5.0

#: Distributions the chat extras install, grouped by the extra that provides
#: them. Reported by import name because that is what actually determines
#: whether a code path works -- a distribution can be installed for a different
#: interpreter, or half-uninstalled.
_EXTRA_MODULES: dict[str, tuple[tuple[str, str], ...]] = {
    "chat-cli": (
        ("langchain", "langchain"),
        ("langgraph", "langgraph"),
        ("langchain_openai", "langchain-openai"),
        ("openai", "openai"),
        ("pydantic_settings", "pydantic-settings"),
        ("sqlite_vec", "sqlite-vec"),
        ("fastembed", "fastembed"),
        ("onnxruntime", "onnxruntime"),
        ("rich", "rich"),
    ),
    "chat-ui": (("chainlit", "chainlit"),),
    "chat-all": (("litellm", "litellm"), ("langchain_litellm", "langchain-litellm")),
    "chat-sqlite": (("pysqlite3", "pysqlite3-binary"),),
}

#: Extras whose absence is not a problem. ``chat-cli`` is required; the rest are
#: opt-in surfaces, so "not installed" is a fact rather than a finding.
_REQUIRED_EXTRAS = frozenset({"chat-cli"})

#: LLM providers that talk to a remote OpenAI-compatible endpoint, as opposed
#: to the local vLLM one. Which of the two decides where the served model's
#: name is configured and what ``native`` costs to turn on. It does *not*
#: decide whether ``text`` works: a reasoning model breaks it wherever it is
#: served, so both flows are checked.
_REMOTE_LLM_PROVIDERS = frozenset({"openai", "litellm"})

#: What ``native`` needs beyond the setting, per flow. A stock vLLM rejects the
#: ``tools`` parameter until it is started for it, so the local remedy is two
#: server flags on top of the setting rather than the setting alone. These
#: restate the ``Endpoint requirement`` and ``Local vLLM`` rows of the table in
#: ``docs/chat/providers.md``.
_REMOTE_NATIVE_NOTE = (
    "It needs an endpoint that accepts the 'tools' parameter, which a remote\n"
    "OpenAI-compatible gateway normally does."
)
_VLLM_NATIVE_NOTE = (
    "It needs the vLLM server restarted with --enable-auto-tool-choice and a\n"
    "--tool-call-parser; a stock server does not accept the 'tools' parameter."
)

#: The half of the remote-embedder story ``manifest.remedy_lines`` cannot tell.
#:
#: That function stops *offering* ``index fetch`` where it is guaranteed to
#: refuse, and names ``embedding_provider = "local"`` as what makes the fetch
#: work. What it cannot say is that the setting is probably not a decision
#: anyone made. Every profile template ``aorta chat config init`` shipped for a
#: remote LLM wrote ``embedding_provider = "remote"`` explicitly, and
#: ``config.write_profile`` runs only on ``config init`` -- so a profile created
#: before the templates changed still carries it, and nothing rewrites it. The
#: published index cannot be read that way, which leaves those installs in the
#: state this command exists to end: a fetch that refuses, and a build that
#: sends every chunk of the corpus through the embeddings API.
#:
#: So this names the edit rather than only the consequence. ``--force`` comes
#: second on purpose: it rewrites the whole profile from the current template,
#: discarding hand edits and re-prompting for the key, where the one-line
#: change keeps everything else. And it ends by saying what to do if the remote
#: embedder *was* deliberate, because for that install flipping to local would
#: be the wrong advice and the line has no way to tell the two apart.
#:
#: Deliberately does **not** restate why the fetch refuses. ``remedy_lines``
#: already prints that paragraph on the row that owns the index verdict, and
#: rendering the two together showed the same argument twice, forty lines
#: apart. A report that repeats itself is one people learn to skim, which is
#: the failure this whole batch is about -- so this carries only the half
#: ``remedy_lines`` structurally cannot: that the setting came from a template,
#: and what to do about it. The setting *name* appears in both, because that is
#: the action rather than the argument.
_REMOTE_EMBEDDING_MIGRATION = (
    # Names its own subject rather than opening on "this": --json emits hint
    # and procedure as separate fields, so a pronoun here points at nothing
    # for a reader who has only the one.
    "A remote embedding provider is very likely not a choice anyone made here.\n"
    "A profile for a remote LLM is where the setting usually comes from, and no\n"
    "profile template selects a remote embedder any more. Nothing rewrites a\n"
    "chat.toml that already exists, though -- write_profile runs only on 'config\n"
    "init' -- so a profile written by an older install still carries it, and will\n"
    "until it is edited or regenerated.\n"
    "\n"
    "Three ways to change it, cheapest first:\n"
    '  embedding_provider = "local"                edit chat.toml, keeping the rest\n'
    # ``export``, not a bare assignment: the promise is a session, and an
    # unexported shell variable does not reach the aorta process the user runs
    # next. A line someone can paste and have nothing happen is the same defect
    # as a command that cannot run, in a smaller package.
    "  export AORTA_CHAT_EMBEDDING_PROVIDER=local  for this shell session\n"
    "  aorta chat config init --force --profile <name>\n"
    "                                              rewrite the profile from the\n"
    "                                              current template; --profile is\n"
    "                                              required and chat.toml does not\n"
    "                                              record which one wrote it, so the\n"
    "                                              wrong name changes the chat\n"
    "                                              provider too. It also discards\n"
    "                                              hand edits and asks for the API\n"
    "                                              key again\n"
    "Local embedding runs on CPU, makes no API calls, and downloads ~65 MB of\n"
    "weights once. It does not change which LLM you talk to -- only how the\n"
    "corpus and your questions are turned into vectors."
)

#: Closes the block when the remote client builds, so keeping it is a real
#: option. Named separately from the tail below because which of the two is
#: true decides whether "build locally instead" is advice or a dead end.
_REMOTE_KEEPING_IT_WORKS = (
    "\n\nIf the remote embedder *was* deliberate, keep it and build the index\n"
    "locally; the index checks below say what this install currently needs."
)

#: And when it does not build. Review raised the same gap from #462's side --
#: that recommending ``index build`` on an empty ``remote_embedding_api_key``
#: is guaranteed to fail -- and this is the half of it that belongs here: the
#: paragraph above must not offer a local build to an install whose provider
#: row has just failed. Worth stating that nothing asked for the key, because
#: the natural reading of "not set" is that the user cleared it.
_REMOTE_KEEPING_IT_NEEDS_A_KEY = (
    "\n\nKeeping the remote embedder means setting remote_embedding_api_key\n"
    "first. The provider row above could not build a client without it, so\n"
    "'aorta chat index build' would fail on the first chunk rather than\n"
    "slowly -- and nothing has asked you for that key: no profile template\n"
    "prompts for it, and it does not fall back to the one the chat model uses."
)

#: Model names that mark a reasoning model. A heuristic -- a gateway can call a
#: deployment anything, and a vLLM server is launched under whatever name its
#: operator gave it -- so it only decides whether the tool-mode check warns or
#: merely informs. Every branch names ``native`` and the symptom, because the
#: case this cannot recognise is exactly the one a user reaches after hitting it.
#:
#: Measured against 13 real reasoning names and 31 ordinary ones, it separates
#: them cleanly -- ``gpt-4o`` and ``gpt-4o-mini`` do not match, ``o4-mini`` and
#: ``DeepSeek-R1`` do. What it cannot separate is a bespoke deployment whose
#: name ends in a revision suffix: ``internal-llama-70b-r1`` matches. That is
#: the reason this only ever chooses the wording, never withholds advice -- a
#: false positive costs a warning whose remedy is the same one the ``OK``
#: branch prints anyway.
_REASONING_MODEL_PATTERN = re.compile(r"gpt-oss|qwq|reasoner|reasoning|\b(?:o[1-4]|r1)\b")

#: The pointer every ``text``-mode line carries, whatever the check could work
#: out about the model. ``native`` is what an action-routed question that comes
#: back empty needs, and a deployment can be served under any name -- so the
#: line that has no name to read is the one whose reader most needs the pointer,
#: not the one that can be left green and silent. That user is already
#: misconfigured, and ``text`` names the symptom they will actually hit.
_TEXT_MODE_HINT = (
    "If an action-routed question comes back with no answer, the first thing\n"
    'to change is llm_tool_mode = "native". Reasoning models cannot write the\n'
    "ACTION: lines text mode parses, and a deployment can be served under any\n"
    "name.\n"
)

#: And the pointer every ``native`` line carries, found by sweeping this
#: function for the same green-with-no-advice shape review found in the
#: no-model-name branch. ``native`` is the mode with an endpoint requirement,
#: and nothing in this report tests it: ``_check_backend`` asks the backend for
#: ``/health``, which a server that rejects the ``tools`` parameter answers
#: perfectly well. So this line read green over the one tool-mode setting that
#: cannot work at all -- ``native`` on a stock local vLLM -- and the user who
#: reaches it has already configured the failure.
_NATIVE_MODE_HINT = (
    "Nothing here confirms the endpoint accepts it: the backend probe asks\n"
    "for /health, which a server that rejects the 'tools' parameter answers\n"
    'normally. If action-routed questions fail, "text" is the mode to try.\n'
)

#: The same warning for a model that has no such fallback. Review found the
#: line above being printed over a reasoning model, where it is not merely
#: unhelpful but points at the dead end this check was added to prevent: the
#: ``text`` branch two screens down warns that a reasoning model cannot write
#: the ``ACTION:`` lines that mode parses, so sending one there on the way out
#: of a failing ``native`` walks it into the failure the other branch exists to
#: describe. The module knows which it is -- ``model`` is resolved before the
#: mode is branched on -- so the only reason it said it was that nobody asked
#: the question on this path.
_NATIVE_REASONING_HINT = (
    "Nothing here confirms the endpoint accepts it: the backend probe asks\n"
    "for /health, which a server that rejects the 'tools' parameter answers\n"
    "normally. This model has no second option if it does not -- a reasoning\n"
    'model cannot drive "text" mode, which is why that mode warns about it,\n'
    "so the endpoint is the thing to fix rather than the setting.\n"
)


@dataclass
class Check:
    """One line of the report."""

    name: str
    status: str
    detail: str = ""
    hint: str = ""
    #: Long-form remediation, printed as its own block. Used for the pre-seed
    #: procedure, which is a paragraph rather than a sentence.
    procedure: str = ""


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)

    def add(self, *args: Any, **kwargs: Any) -> Check:
        check = Check(*args, **kwargs)
        self.checks.append(check)
        return check

    @property
    def failed(self) -> bool:
        return any(check.status == FAIL for check in self.checks)

    @property
    def warned(self) -> bool:
        return any(check.status == WARN for check in self.checks)


def _dist_version(dist: str) -> str:
    try:
        return version(dist)
    except PackageNotFoundError:
        return ""


def _check_python(report: Report) -> None:
    have = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    # 3.11 is chat's floor (Decision 13a): onnxruntime and stdlib tomllib.
    status = OK if sys.version_info >= (3, 11) else FAIL
    report.add(
        "python",
        status,
        have,
        hint="" if status == OK else "aorta chat needs Python 3.11 or newer.",
    )
    report.add("aorta", OK, _dist_version("amd-aorta") or "not installed (raw source tree?)")


def _check_extras(report: Report) -> None:
    for extra, modules in _EXTRA_MODULES.items():
        missing = [dist for module, dist in modules if find_spec(module) is None]
        present = [
            f"{dist} {_dist_version(dist) or '?'}"
            for module, dist in modules
            if find_spec(module) is not None
        ]
        if not missing:
            report.add(f"extra {extra}", OK, ", ".join(present))
        elif extra in _REQUIRED_EXTRAS:
            report.add(
                f"extra {extra}",
                FAIL,
                f"missing: {', '.join(missing)}",
                hint=f"pip install 'amd-aorta[{extra}]'",
            )
        elif present:
            report.add(f"extra {extra}", WARN, f"partial; missing {', '.join(missing)}")
        else:
            report.add(f"extra {extra}", SKIP, "not installed (optional)")


def _check_sqlite(report: Report) -> None:
    """sqlite version and loadable-extension support, sqlite-vec's two needs."""
    from aorta.chat.rag import sqlite_compat

    floor = ".".join(str(part) for part in sqlite_compat.MIN_SQLITE_VERSION)
    try:
        sqlite_compat.ensure_modern_sqlite()
        sqlite_compat.ensure_loadable_extensions()
    except RuntimeError as exc:
        report.add("sqlite", FAIL, f"floor is {floor}", hint=str(exc))
        return
    import sqlite3

    report.add("sqlite", OK, f"{sqlite3.sqlite_version} (>= {floor}, extensions loadable)")


def _warm_command(model: str, cache: str) -> str:
    """Downloading the embedding weights and nothing else -- what a "pre-warm" is.

    The same invocation ``fastembed_bge.PRE_SEED_PROCEDURE`` uses in its first
    step, rebuilt rather than imported because that procedure is a paragraph and
    this is a line; a test pins the two together.

    Deliberately *not* ``aorta chat index build``, which this check used to
    advise. That command's ``--output`` defaults to the index this install
    already reads and its corpus defaults to ``src/aorta`` alone, so as a
    pre-warm it overwrites a fetched index with one that has no ``docs/`` and no
    ``README.md`` in it -- and says nothing about having done either.

    Both values reach a shell and neither is this module's to trust: ``model``
    is a configured setting, and ``cache`` follows ``HF_HOME``. Interpolated
    raw, a path holding an apostrophe -- ``/home/o'brien/.cache`` -- closes the
    surrounding single-quoted argument early, so the remedy printed to someone
    whose setup is already broken is a command that will not parse.

    ``json.dumps`` for the two Python string literals, then ``shlex.quote`` for
    the shell argument, in that order, because the inner literals have to be
    escaped before the outer quoting measures them. For a path that needs no
    escaping the result is byte-identical to the hand-quoted form, which is what
    lets the test pinning this against ``PRE_SEED_PROCEDURE`` compare text.
    """
    snippet = (
        "from fastembed import TextEmbedding; "
        f"TextEmbedding({json.dumps(model)}, cache_dir={json.dumps(cache)})"
    )
    return f"python -c {shlex.quote(snippet)}"


def _probe_huggingface() -> bool:
    """Whether the HuggingFace CDN answers. A TCP connect, not a model download."""
    try:
        with socket.create_connection((_HF_HOST, _HF_PORT), timeout=_HF_PROBE_TIMEOUT):
            return True
    except OSError as exc:
        logger.debug("HuggingFace probe failed: %s", exc)
        return False


def _store_defect(index_file: Path, dimensions: int) -> str:
    """Why this install's collection cannot be read out of ``index_file``, or "".

    ``dimensions`` is the width the manifest records, which both callers
    already hold from the ``check_index`` they just ran, so it costs no extra
    read. Pass ``0`` for "not known", which skips only the width comparison.

    The half ``check_index`` does not report. It escalates an unopenable index
    to a refusal only when the manifest claims a chunk count to contradict, and
    a manifest written before that field existed claims none -- so a clobbered
    legacy index comes back from it with an empty refusal list, even though the
    query path refuses that file unconditionally
    (``retriever._check_manifest``).

    Both readers of index health below go through this one function, because a
    report that says the index matches on one line and is unusable on another
    is worse than either answer on its own.

    A non-empty chunk table is necessary but not sufficient, which is why the
    schema check follows it -- see :func:`_collection_schema_defect`.

    **Why this models the read path instead of asking it, and what stops that
    modelling drifting.** Three rounds of review have now found the same shape
    here -- the chunks table was necessary-not-sufficient, then the rest of the
    seven-part contract, then the registry width inside it -- so the question
    is not which condition to add next but whether a *derived* answer is the
    right kind of answer at all. It is, at this price, and the reasons are
    properties of the read path rather than preferences:

    * ``SqliteVecStore._connection`` calls ``ensure_loadable_extensions``. This
      probe has to report on the install that is *missing* sqlite-vec, and the
      report already has a ``sqlite`` row saying so. Routing it through the
      store would restate one environment fault as a defect in every index --
      the mislabelling caught in ``_check_remote_embedding_profile``, where a
      raising probe put two ``embedding provider`` rows in one report.
    * That same method connects read-write. A doctor must not be the thing that
      writes a journal beside a user's already-damaged index; this reads
      ``mode=ro``.
    * A true end-to-end query also needs a query vector, which locally means
      downloading the model and remotely means a billed API call.

    The first two are unavoidable, so the modelling stays. What changes is that
    it is no longer trusted to be complete: ``TestStoreProbeAgreesWithTheReadPath``
    runs a real ``similarity_search`` against every state in ``STORE_DAMAGE``
    and fails if any of them answers differently from this function. A test can
    pay all three prices above, because it controls the environment a user's
    machine does not. That test is what found the non-integer width -- the
    review named the mismatched one.

    That oracle then missed twice itself, on ``metadata`` and then on
    ``content``, and both times because its list of states was hand-written
    beside the read path rather than taken from it. So the value states are now
    derived from the columns ``_knn`` selects, and a column nothing describes
    raises rather than going unswept. The practical consequence for anyone
    editing *this* function: the set it is measured against is closed by
    construction, so a column added to the read path fails the oracle until
    this probe has something to say about it.

    The agreement required is one-directional: no state where retrieval fails
    may read as healthy here. The reverse is allowed and one case uses it --
    see the row-parity check in :func:`_collection_schema_defect`.

    **On #465, which fixes that suppression at the root.** It removes the
    ``manifest.chunk_count`` gate, so ``check_index`` will refuse *a file that
    cannot be opened as sqlite at all* on its own. This probe is kept anyway,
    and not as belt-and-braces: measured against every state in
    ``STORE_DAMAGE`` -- the closed set the oracle enumerates, rather than a
    count maintained by hand here, which is what this paragraph used to carry
    and drifted twice -- ``check_index`` refuses none of them today, and #465
    changes exactly one, the file that cannot be opened as sqlite at all. All
    the rest are facts about the schema underneath the sidecar: no chunk table
    for this install's collection, an empty one, a missing collection registry,
    an unregistered collection, absent ``content``/``metadata`` columns,
    content that is not text and metadata that is not a JSON object, a missing
    or short vector table, and a registry width that is either not a number or
    not the one the index was built at. ``check_index`` never looks at any of them, because it
    compares a sidecar against a row count. So there is no behaviour to make
    this conditional on: after #465 the unopenable case short-circuits at the
    refusal check in both callers and never reaches here, and the states that
    do reach here are why the function exists. Merge order does not matter
    either way.
    """
    from aorta.chat.rag.embeddings.factory import get_provider

    # The registry name comes from the module that writes it, imported above the
    # ``except`` below so that a rename raises here -- a visible failed check --
    # rather than being caught and reported as a defect in every index.
    from aorta.chat.rag.retriever import _REGISTRY_TABLE, collection_chunk_count

    try:
        collection = get_provider().collection_name()
        chunks = collection_chunk_count(index_file, collection)
        # ``None`` is no chunk table for this install's collection and ``0`` is
        # an empty one -- an interrupted build, or a file indexed by another
        # provider. Neither can answer a question.
        if not chunks:
            return f"no chunks for this install's collection in {index_file}"
        return _collection_schema_defect(
            index_file, collection, _REGISTRY_TABLE, chunks, dimensions
        )
    except Exception as exc:
        # Deliberately broad, for the same reason as ``_check_backend``: a
        # damaged index surfaces as IndexUnreadableError, sqlite3.Error or
        # OSError depending on how it is damaged, and a doctor that propagates
        # one of those has failed at its only job. The message is the hint.
        logger.debug("index store probe failed", exc_info=True)
        return str(exc)


def _loads_or_none(value: object) -> object:
    """``json.loads`` that answers "not an object" instead of raising.

    Only used by the JSON1-less branch of the metadata check, where the
    question is whether the read path would survive the value, and every way
    of not surviving it has the same answer.
    """
    if not isinstance(value, (str, bytes)):
        return None
    try:
        return json.loads(value)
    except ValueError:
        return None


def _decodes_as_content(value: bytes) -> bool:
    """Whether ``Document`` would accept these bytes as page content.

    Only blob rows reach here -- the query selects on
    ``typeof(content) = 'blob'``, and sqlite3 hands blobs back as ``bytes`` --
    so the only question left is whether they decode. ``Document`` coerces
    bytes through UTF-8, so ones that do not raise on the first hit.

    Narrow on purpose. An earlier version dispatched on ``str`` and on "not
    bytes" as well, and mutation testing showed both branches were unreachable
    from the one caller: inverting the ``str`` answer changed no test. A guard
    no caller can exercise is not defence, it is a claim about the callers that
    nothing checks.
    """
    try:
        value.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def _collection_schema_defect(
    index_file: Path, collection: str, registry_table: str, chunks: int, dimensions: int
) -> str:
    """Which part of the readable-collection contract ``index_file`` fails, or "".

    A non-empty ``chunks_<collection>`` table is one part of what the read path
    needs, and on its own it is the weakest part. ``_get_vectorstore``
    also requires the collection registry and a row in it for this install's
    collection (``retriever.collection_exists``), and ``retriever._knn`` then
    reads ``content`` and ``metadata`` off the chunk table and joins it against
    ``vec_<collection>`` on ``c.id = m.rowid``. So a store holding nothing but
    the chunk table -- a partial copy, or a build interrupted between the two
    inserts -- passes a count and fails every query, which is the state this
    function exists to name.

    The registry row is read for its *width* and not merely for its presence,
    because ``_knn`` reads that number before it reads any table and refuses
    the search if it does not match the vector the provider just produced. A
    store can satisfy every structural part of this contract and still answer
    nothing on that one integer.

    Row parity is checked as well, because the join is an inner one: chunks
    with no vectors beside them are rows no retrieval can reach, and the query
    returns empty rather than raising. This is the one place the function is
    deliberately stricter than the read path -- the rows that do have vectors
    still answer -- which is why ``TestStoreProbeAgreesWithTheReadPath``
    requires one-way agreement and names this state as the only exception.

    Raw read-only sqlite and no sqlite-vec load, for the same reason
    ``collection_chunk_count`` uses it: this has to report on a broken install
    without the extension that install may be missing, and without an
    embedding provider it would have to download a model to instantiate.

    ``collection`` reaches the table names unparameterised, which sqlite does
    not allow to be bound. It is safe here for the reason it is safe there:
    ``collection_chunk_count`` has already matched it against
    ``retriever._SAFE_COLLECTION`` and returned ``None`` for anything else, so
    a name that got this far has produced a row count.
    """
    import sqlite3

    conn = sqlite3.connect(f"file:{index_file}?mode=ro", uri=True)
    try:
        tables = {
            name for (name,) in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        if registry_table not in tables:
            return (
                f"{index_file} holds chunks for this install's collection but no "
                f"{registry_table} registry, so the query path reports a missing "
                "collection"
            )
        registered = conn.execute(
            f'SELECT dimension FROM "{registry_table}" WHERE collection = ?',
            (collection,),
        ).fetchone()
        if registered is None:
            return (
                f"this install's collection is not registered in {registry_table} in "
                f"{index_file}, so the query path reports a missing collection"
            )

        # The registry width is not merely a field that should be populated: it
        # is the number ``retriever.SqliteVecStore._knn`` coerces with ``int()``
        # and compares against the width of the vector the provider just
        # produced, before it touches a table. So a width that is not a number,
        # or is a number the index was not built at, fails every search while
        # every other part of the contract here still holds -- registry, chunk
        # columns, vector table and row parity all intact.
        try:
            width = int(registered[0])
        except (TypeError, ValueError):
            return (
                f"this install's collection is registered in {index_file} with a "
                f"dimension of {registered[0]!r}, which is not a number; the query "
                "path reads it before every search and cannot"
            )
        if dimensions > 0 and width != dimensions:
            return (
                f"this install's collection is registered at {width} dimensions in "
                f"{index_file} but the manifest records {dimensions}; the query path "
                f"compares the two and refuses every {dimensions}-dimension search"
            )

        columns = {row[1] for row in conn.execute(f'PRAGMA table_info("chunks_{collection}")')}
        missing = [column for column in ("id", "content", "metadata") if column not in columns]
        if missing:
            return (
                f"the chunk table for this install's collection in {index_file} is "
                f"missing the {', '.join(missing)} column(s) that retrieval reads"
            )

        # The column existing is not the same as the read path surviving it,
        # and that holds for every column ``_knn`` selects rather than for the
        # one a review happened to name. ``content`` goes straight into
        # ``Document(page_content=...)``, which requires a string: a NULL, an
        # integer and a real each raise there. A BLOB does not -- it is decoded
        # and accepted -- so the defect is the stored type and not "is it
        # text", which is why this asks ``typeof`` rather than a cast. A store
        # aorta wrote cannot reach this (the column is ``TEXT NOT NULL``); one
        # carried in by hand or built by another tool can.
        bad_content = conn.execute(
            f'SELECT COUNT(*) FROM "chunks_{collection}" '
            "WHERE typeof(content) NOT IN ('text', 'blob')"
        ).fetchone()[0]
        # A BLOB counts as content only if it decodes. ``Document`` coerces
        # bytes through UTF-8, so ``x'FF'`` raises there exactly as a NULL
        # does, while ``typeof`` reports the same 'blob' for both it and a
        # perfectly readable one. SQLite has no UTF-8 predicate to ask in SQL,
        # so the decode happens here -- streamed off the cursor, and over blob
        # rows alone, of which a store aorta wrote has none.
        bad_content += sum(
            not _decodes_as_content(value)
            for (value,) in conn.execute(
                f"SELECT content FROM \"chunks_{collection}\" WHERE typeof(content) = 'blob'"
            )
        )
        if bad_content:
            return (
                f"{bad_content} chunk row(s) for this install's collection in "
                f"{index_file} hold no text in the content column; retrieval builds "
                "a document out of it for every hit and raises on the first one"
            )

        # ``_knn`` calls ``json.loads`` on every metadata value it selects and
        # hands the result to ``Document(metadata=...)``, which needs a mapping
        # -- so a NULL, a non-JSON string and a valid non-object JSON value are
        # each enough to make every matching retrieval raise while the chunk
        # count, the columns and the vectors all look right. Asked in SQL, and
        # of the whole column rather than a sample: one bad row is one query
        # that raises, and the row that raises is chosen by the query vector.
        try:
            # One CASE rather than a chain of ORs: ``json_type`` raises
            # "malformed JSON" on a value ``json_valid`` rejects, so the guard
            # has to be one SQLite promises not to evaluate past. CASE is;
            # OR's evaluation order is an implementation detail.
            bad_metadata = conn.execute(
                f'SELECT COUNT(*) FROM "chunks_{collection}" WHERE '
                "CASE WHEN metadata IS NOT NULL AND json_valid(metadata) "
                "THEN json_type(metadata) ELSE 'null' END <> 'object'"
            ).fetchone()[0]
        except sqlite3.OperationalError:
            # A build without JSON1. Counted in Python instead of skipped: the
            # question the read path asks does not go away because this SQLite
            # cannot answer it in SQL. Streamed off the cursor rather than
            # fetched, so a large index costs no more memory than one row.
            bad_metadata = sum(
                not isinstance(_loads_or_none(value), dict)
                for (value,) in conn.execute(f'SELECT metadata FROM "chunks_{collection}"')
            )
        if bad_metadata:
            return (
                f"{bad_metadata} chunk row(s) for this install's collection in "
                f"{index_file} carry metadata that is not a JSON object; retrieval "
                "parses it into a document for every hit and raises on the first one"
            )

        vectors = f"vec_{collection}"
        if vectors not in tables:
            return (
                f"{index_file} has no {vectors} vector table beside its chunk table, "
                "so there is nothing for a query to search"
            )

        # A table of that *name* is not a vector index. An ordinary table
        # called ``vec_<collection>`` satisfies the check above, and the read
        # path's ``embedding MATCH ?`` then raises ``OperationalError`` on it --
        # so the name test alone reported a store as queryable that answers
        # nothing.
        #
        # Asked as "is it virtual", not "is it vec0", on purpose -- and this is
        # a judgement rather than a tested property, so it is worth being plain
        # about: requiring the ``vec0`` module name passes the whole suite too,
        # because nothing today builds a vector table any other way. It is
        # written loose to match the decision in the next comment, where the
        # shadow table's absence is deliberately not a defect so that a future
        # sqlite-vec layout cannot make every healthy index read as broken. The
        # same reasoning applies to the module name, which is sqlite-vec's to
        # change. What this has to separate is an ordinary table from a virtual
        # one, and an ordinary table can never answer ``MATCH`` whatever
        # implements it.
        #
        # ``fetchone()`` cannot be None: the name came from the ``tables`` read
        # above, which is this same catalogue. ``or ""`` covers a NULL ``sql``
        # only so the comparison is total.
        sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?", (vectors,)
        ).fetchone()[0]
        if not (sql or "").upper().startswith("CREATE VIRTUAL TABLE"):
            return (
                f"{vectors} in {index_file} is an ordinary table, not a vector "
                "index; retrieval matches the query vector against it and every "
                "search fails on the table itself"
            )

        # ``vec0`` keeps its rows in a shadow table, which is countable without
        # loading the extension where the virtual table itself is not. Its
        # absence is not reported as a defect: the vector table is already
        # known to be there, and a future sqlite-vec layout would otherwise
        # make every healthy index read as broken.
        if f"{vectors}_rowids" in tables:
            rowids = f"{vectors}_rowids"
            embedded = conn.execute(f'SELECT COUNT(*) FROM "{rowids}"').fetchone()[0]
            if embedded != chunks:
                return (
                    f"{chunks} chunks but {embedded} vectors for this install's "
                    f"collection in {index_file}; retrieval joins the two and can only "
                    "reach the rows that have both"
                )
            rowid_columns = [name for _, name, *_ in conn.execute(f'PRAGMA table_info("{rowids}")')]
            rowid_column = (
                "rowid"
                if "rowid" in rowid_columns
                else (
                    "id" if "id" in rowid_columns else (rowid_columns[0] if rowid_columns else "")
                )
            )
            if rowid_column:
                quoted = rowid_column.replace('"', '""')
                missing_vector = conn.execute(
                    f'SELECT 1 FROM "chunks_{collection}" c LEFT JOIN "{rowids}" r '
                    f'ON c.id = r."{quoted}" WHERE r."{quoted}" IS NULL LIMIT 1'
                ).fetchone()
                if missing_vector:
                    return (
                        f"chunk and vector row IDs disagree for this install's collection "
                        f"in {index_file}; retrieval joins by row ID and misses unmatched rows"
                    )
                missing_chunk = conn.execute(
                    f'SELECT 1 FROM "{rowids}" r LEFT JOIN "chunks_{collection}" c '
                    f'ON c.id = r."{quoted}" WHERE c.id IS NULL LIMIT 1'
                ).fetchone()
                if missing_chunk:
                    return (
                        f"vector and chunk row IDs disagree for this install's collection "
                        f"in {index_file}; retrieval joins by row ID and misses unmatched rows"
                    )
        return ""
    finally:
        conn.close()


def _index_is_healthy() -> bool:
    """Whether an index is present and this install can query it as-is.

    Asked by both halves of ``_check_embedding_model``, so that neither the
    cold-cache hint nor the remote-profile advice fires on an install that is
    already fine: the first would say "nothing to do" over an unusable index,
    the second would tell a working remote setup to abandon it.
    ``_check_embedding_model`` runs before ``_check_index`` because
    provider-before-index reads better in the report, so it cannot read the
    later check's result; running the same validation twice costs two sqlite
    opens and no network, which is cheaper than reordering the output. Twice is
    the ceiling either way -- the two callers are on mutually exclusive
    branches of the provider check.

    Warnings do not disqualify an index. Source drift is a reason to refresh
    it, not a reason for advice that would replace it with a worse one.

    An empty refusal list is not on its own enough -- see ``_store_defect`` --
    so the file is read as well. Getting this wrong is asymmetric: a spurious
    ``False`` costs a warning the user can ignore, while a spurious ``True``
    tells someone whose index is unusable that there is nothing to do, and
    withholds the remedy.

    The ``_store_defect`` call on the last line is **deliberately not** inside
    the guard above, and review has asked about it, so: the guard exists to
    turn *a damaged index* into ``False``, which is a fact about the file. A
    raise out of ``_store_defect`` is not that -- it means this module's own
    imports moved, since the two it needs sit above its ``except`` precisely so
    a rename cannot be reported as a defect in every index. Wrapping it here
    would convert that bug into a quiet ``False``, and a quiet health probe
    over an index nobody checked is the exact shape of the defect this whole
    check exists to close.

    Nor would wrapping it contain anything. ``run_checks`` already catches per
    check and records the exception as a ``fail`` line, so one bad index cannot
    abort the report -- pinned by
    ``TestStructure.test_a_check_that_explodes_becomes_a_finding_not_a_traceback``
    and, for this path specifically, by
    ``test_a_raising_store_probe_is_reported_and_the_run_continues``. And
    ``_check_index`` calls ``_store_defect`` unguarded in the same run, so the
    same raise surfaces there regardless: a guard here could only make the two
    readers of one helper disagree about one failure, which is the drift this
    PR removed.

    ``_check_remote_embedding_profile`` *does* catch around its call, which is
    a different thing and not a re-litigation of the above. A guard in here
    would change what every caller learns about the file. A guard out there
    changes only what that one check is willing to conclude when the probe did
    not finish -- it abstains, having no premise, and the exception still
    reaches the report through ``_check_index`` under the name it belongs to.
    Letting it escape *that* call instead was measured, and put two
    ``embedding provider`` rows in one report.
    """
    from aorta.chat.config import settings
    from aorta.chat.rag.index_ops import check_index

    index_file = settings.index_file
    if not index_file.exists():
        return False
    try:
        result = check_index(index_file, strict=False)
        if result.refusals:
            return False
    except Exception:
        # Unreadable, unparseable, no manifest: all mean the same thing here,
        # which is that there is nothing worth protecting from a rebuild.
        logger.debug("index health probe failed", exc_info=True)
        return False
    return not _store_defect(index_file, result.manifest.dimensions)


def _check_remote_embedding_profile(report: Report, *, usable: bool) -> None:
    """Whether a remote embedding profile is one this install can answer from.

    ``usable`` says whether the remote client could be built at all, which only
    changes the closing paragraph: an install with no embedding key cannot be
    told to build the index locally instead, because that is the one remedy its
    missing key guarantees will fail.

    The migration gap raised on #462. That PR flips the profile templates to
    local embeddings, but ``config.write_profile`` runs only on ``config
    init``, so it cannot reach a ``chat.toml`` that already exists. Everyone who
    ran ``config init`` with a remote-LLM profile before the change keeps
    ``embedding_provider = "remote"`` on disk, gets no signal from that PR, and
    stays in the state that motivated this one: no published index they can
    read, and nothing naming the one line that fixes it.

    ``config validate`` was the other candidate and is the wrong home twice
    over -- it is rarely run, and a remote embedding profile is not *invalid*.
    It is valid and incompatible with the published index, which is a different
    statement and the kind a doctor makes. This command is also the only one
    that has already resolved the provider and read the index, which are the
    two facts the diagnosis needs.

    **Conditioned on the index, not on the provider.** An install that chose a
    remote embedder deliberately and built a matching index locally is correct,
    and telling it to flip to local would be wrong -- so this fires only where
    there is no index this install can query. That is the same discipline as
    the cold-cache line above: a warning that fires on a correct setup is how
    people learn to skim the command that also reports the fatal mismatches,
    and gap 2a -- the reason this PR exists -- was exactly that mistake.

    ``WARN`` rather than ``FAIL`` because the provider itself is fine and
    working. ``_check_index`` owns the verdict on the index and reports it
    below, with the remedies; this line explains why the obvious remedy is
    missing from that list and what to change so it comes back.

    Costs one more ``check_index`` and sqlite open, on the remote path only,
    which previously did no index work at all. That makes two per run at worst
    -- the same ceiling the local cold-cache path already has, and for the same
    reason: reordering the report to share one result reads worse than the
    second read costs, and neither touches the network.
    """
    try:
        healthy = _index_is_healthy()
    except Exception as exc:
        # A probe that did not finish establishes nothing, so this abstains
        # rather than guessing which way it would have gone. Letting it escape
        # instead was measured: a broken sqlite-vec install put *two*
        # ``embedding provider`` rows in one report, an ``ok`` and a ``fail``,
        # because ``run_checks`` labels a raising check with its own name --
        # and blamed the embedding provider for a sqlite fault. ``_check_index``
        # already reports that same exception one row down, under the name it
        # belongs to.
        #
        # Not a reversal of the argument about ``_index_is_healthy``'s last
        # line: that was against wrapping the line *inside* the helper, which
        # would have changed what ``_check_index`` reports. This is a caller
        # declining to draw a conclusion, and it leaves the helper's contract
        # and the index verdict exactly as they were.
        logger.debug("index probe failed; no profile advice: %s", exc)
        return
    if healthy:
        # A deliberate remote embedder with an index built to match it. Nothing
        # to migrate, and nothing this line could say that would be true.
        return
    report.add(
        "embedding profile",
        WARN,
        "remote embeddings, and no index this install can query",
        hint=(
            "This profile selects a remote embedding provider, and the\n"
            "published index cannot be read that way -- which is why\n"
            "'aorta chat index fetch' is not offered as an index remedy.\n"
            'Setting embedding_provider = "local" brings it back; see below.'
        ),
        procedure=_REMOTE_EMBEDDING_MIGRATION
        + (_REMOTE_KEEPING_IT_WORKS if usable else _REMOTE_KEEPING_IT_NEEDS_A_KEY),
    )


def _check_embedding_model(report: Report) -> None:
    """Whether queries can be embedded at all, and what to do when they cannot."""
    from aorta.chat.rag import manifest as manifest_mod
    from aorta.chat.rag.embeddings.factory import get_provider

    try:
        provider = get_provider()
    except ValueError as exc:
        # This row already made the distinction ``manifest`` was missing -- it
        # catches the factory's ``ValueError`` rather than letting an unknown
        # name resolve to local. What it did not have was a route to a fix: a
        # FAIL with an empty hint, which is the same complaint this PR started
        # from one row over. The remedy comes from ``remedy_lines`` rather than
        # being written here, so this row and the index row cannot disagree
        # about what to do -- they are now the same text.
        report.add(
            "embedding provider",
            FAIL,
            str(exc),
            hint="\n".join(manifest_mod.remedy_lines(include_doctor=False)),
        )
        return

    if provider.name != "local":
        # Building the client validates the key and the auth headers without
        # touching the network, and a query cannot run without it, so a remote
        # provider that cannot be built is reported rather than described.
        try:
            provider.get_embeddings()
        except ValueError as exc:
            usable = False
            report.add("embedding provider", FAIL, str(exc))
        else:
            usable = True
            report.add("embedding provider", OK, provider.describe())

        # A remote embedder needs a key and an endpoint, not a cache; the
        # provider reports its own configuration problems when built.
        report.add("embedding model cache", SKIP, "remote provider; no local weights needed")

        # Runs either way, and that is the whole point. Returning early on the
        # failure -- which is what the first version of this check did -- lost
        # the migration advice for precisely the install it was written for: no
        # profile template has ever prompted for ``remote_embedding_api_key``,
        # and it defaults to empty with no fallback to the chat key, so the
        # pre-#462 remote profiles this warning exists to rescue are keyless
        # almost by definition. They would have been told to buy an embeddings
        # subscription, when their actual problem is a stale template.
        _check_remote_embedding_profile(report, usable=usable)
        return

    report.add("embedding provider", OK, provider.describe())

    from aorta.chat.rag.embeddings import fastembed_bge

    state = fastembed_bge.describe_model_state()
    if state["cached"]:
        report.add(
            "embedding model cache",
            OK,
            f"{state['model']} present under {state['cache_dir']}",
        )
        return

    if _probe_huggingface():
        warm = _warm_command(state["model"], state["cache_dir"])
        if _index_is_healthy():
            # Not a warning. ``index fetch`` downloads somebody else's vectors
            # and never needs the local weights, so a correctly completed fetch
            # -- the documented normal path -- always lands here. A warning that
            # fires on every correct setup is how people learn to skim the one
            # command that also reports the fatal mismatches.
            report.add(
                "embedding model cache",
                SKIP,
                f"{state['model']} is not cached, and does not need to be yet",
                hint=(
                    "Your index is readable and matches this install, and the "
                    "weights\n"
                    "download themselves (~65 MB) on the first query. Nothing "
                    "to do.\n"
                    "To get them ahead of that without touching the index:\n"
                    f"  {warm}"
                ),
            )
            return
        report.add(
            "embedding model cache",
            WARN,
            f"{state['model']} is not cached, but HuggingFace is reachable",
            hint=(
                "It will be downloaded (~65 MB) the first time anything embeds "
                "text.\n"
                "To get it now, without building anything:\n"
                f"  {warm}"
            ),
        )
        return

    report.add(
        "embedding model cache",
        FAIL,
        f"{state['model']} is not cached and {_HF_HOST} is unreachable",
        hint="Queries and index builds will both fail until the cache is seeded.",
        procedure=fastembed_bge.PRE_SEED_PROCEDURE.format(
            model=state["model"], cache=state["cache_dir"]
        ),
    )


def _check_index(report: Report) -> None:
    """Index presence, and whether the manifest and the store under it fit this install."""
    from aorta.chat.config import settings
    from aorta.chat.rag import manifest as manifest_mod

    index_file = settings.index_file
    if not index_file.exists():
        # The same conditional list the refusal below uses, and for the same
        # reason: a fresh install on a remote embedder is the state most likely
        # to reach this branch, and it is the one for which `index fetch` is
        # guaranteed to refuse. Naming it first there would put the impossible
        # remedy in front of the user who has done nothing wrong yet.
        report.add(
            "chat index",
            FAIL,
            f"absent at {index_file}",
            hint="\n".join(manifest_mod.remedy_lines(include_doctor=False)),
        )
        return

    size_mb = index_file.stat().st_size / (1024 * 1024)
    report.add("chat index", OK, f"{index_file} ({size_mb:.1f} MB)")

    try:
        from aorta.chat.rag.index_ops import check_index

        result = check_index(index_file, strict=False)
    except manifest_mod.ManifestError as exc:
        report.add(
            "index manifest",
            WARN,
            "cannot be verified",
            hint=str(exc),
        )
        return

    if result.refusals:
        # Remedies come from the manifest module so this and the query-time
        # refusal cannot drift apart, and they are conditional for the same
        # reason: on a remote embedder, `index fetch` is guaranteed to refuse in
        # turn, and a first remedy that cannot work gets the whole refusal
        # worked around.
        remedies = "\n".join(manifest_mod.remedy_lines(include_doctor=False))
        report.add(
            "index manifest",
            FAIL,
            "does not match this install; queries are refused",
            hint="\n".join(result.refusals),
            procedure=(
                "This is not a cosmetic mismatch. The index holds vectors from a "
                "different embedding model, so retrieval would compare numbers "
                "that are not comparable and answer confidently from the wrong "
                "chunks.\n" + remedies
            ),
        )
        return

    # After the refusals, which already cover an unopenable store when the
    # manifest claims a chunk count -- and more specifically, since they name
    # the number. Two cases get past them: a manifest predating that field over
    # a store that cannot be read, and a store whose chunk count is right while
    # the rest of the collection is not, which no manifest describes at all.
    # The query path refuses both regardless, so reporting either as a matching
    # index would contradict both that path and the cold-cache line above,
    # which gates on the same helper.
    defect = _store_defect(index_file, result.manifest.dimensions)
    if defect:
        report.add(
            "index manifest",
            FAIL,
            "describes an index this install cannot read",
            hint=defect,
            procedure=(
                "The manifest and the store under it are from different builds, "
                "or the file did not arrive intact. Queries are refused rather "
                "than answered from whatever survived.\n"
                + "\n".join(manifest_mod.remedy_lines(include_doctor=False))
            ),
        )
        return

    if result.warnings:
        report.add(
            "index manifest",
            WARN,
            result.manifest.describe(),
            hint="\n".join(result.warnings),
        )
        return
    report.add("index manifest", OK, result.manifest.describe())


def _with_native_note(hint: str, native_note: str) -> str:
    """``hint``, plus what ``native`` requires here where the provider is known.

    ``native_note`` is empty for a provider no backend is registered for, and
    the generic pointer is still worth printing there -- so the note is appended
    rather than required.
    """
    return hint + native_note if native_note else hint.rstrip("\n")


def _check_tool_mode(report: Report) -> None:
    """Which protocol the act loop will use to call tools, and whether it fits.

    ``text`` is the default because it has no endpoint requirement, so it is the
    only mode a stock local vLLM can drive. A reasoning model cannot drive it:
    it puts its working in a channel of its own and returns empty ``content``
    where the ``ACTION:`` line was expected, so every action-routed query spends
    inference rounds on a reply this protocol cannot read -- billed for a remote
    provider, self-hosted capacity for a local one, wasted in both. Until this check existed
    the first signal of that was the failed query.

    That holds for a locally served reasoning model as much as a remote one --
    the channel is the model's, not the endpoint's -- so both flows are read.
    What the provider changes is the remedy: turning ``native`` on costs a
    setting remotely and a setting plus two server flags on vLLM.

    The warning names the *cost* and gives the outcome as a disjunction --
    late, degraded, or nothing -- rather than predicting one, and that is load
    bearing rather than vague. What happens after the empty reply is not a
    property of the configuration this check can read: #464 adds a one-shot
    escalation to ``native`` on that exact signature, which only moves the
    built-in default and not a mode the user set, plus a labelled
    retrieval-only fallback when neither protocol answers. So the same ``text``
    setting can end in a good answer one round late, a degraded answer, or
    nothing at all, and ``settings.llm_tool_mode`` does not say which. Every
    one of those is worse than not needing the round, which is the thing the
    user can act on. Do not narrow this back to a single outcome.
    """
    from aorta.chat.config import settings

    mode = str(settings.llm_tool_mode or "").strip().lower()
    provider = str(settings.llm_provider or "").strip().lower()

    if mode not in ("native", "text"):
        report.add(
            "llm tool mode",
            FAIL,
            f"{settings.llm_tool_mode!r} is not a tool mode",
            hint=(
                'Set llm_tool_mode to "text" or "native". Any other value '
                "raises on the\n"
                "first action-routed question rather than at startup."
            ),
        )
        return
    model = native_note = ""
    if provider in _REMOTE_LLM_PROVIDERS:
        model, native_note = str(settings.remote_llm_model or ""), _REMOTE_NATIVE_NOTE
    elif provider == "vllm":
        model, native_note = str(settings.vllm_model or ""), _VLLM_NATIVE_NOTE

    if mode == "native":
        # Resolved after the provider, not before it, so this line can say what
        # the mode requires here. See :data:`_NATIVE_MODE_HINT`.
        #
        # The model is read on this path too, and not only on the ``text`` one:
        # the advice to fall back to ``text`` is wrong for exactly the models
        # the ``text`` branch refuses to recommend, and the name is already in
        # hand. See :data:`_NATIVE_REASONING_HINT`.
        reasoning = bool(model) and bool(_REASONING_MODEL_PATTERN.search(model.lower()))
        report.add(
            "llm tool mode",
            OK,
            f"native (the provider's function-calling API), {model} is a reasoning model"
            if reasoning
            else "native (the provider's function-calling API)",
            hint=_with_native_note(
                _NATIVE_REASONING_HINT if reasoning else _NATIVE_MODE_HINT, native_note
            ),
        )
        return
    if not model:
        # A provider no backend is registered for -- ``_check_backend`` reports
        # that -- or one whose model setting is empty. Either way there is no
        # name to read, and guessing which setting holds it would invent one.
        # The *hint* does not depend on the name, though, so it is attached
        # here too: this was the one green line in the report carrying no route
        # to a fix, and it belongs to a user who is already misconfigured.
        report.add(
            "llm tool mode",
            OK,
            "text (ACTION: lines parsed out of the reply); no model name to check",
            hint=_with_native_note(_TEXT_MODE_HINT, native_note),
        )
        return

    if _REASONING_MODEL_PATTERN.search(model.lower()):
        report.add(
            "llm tool mode",
            WARN,
            f"text, and {model} is a reasoning model",
            hint=(
                "In text mode the model has to write 'ACTION: tool(arg=\"v\")' "
                "for aorta\n"
                "to parse. A reasoning model writes that in a channel of its "
                "own and\n"
                "returns empty content instead, so every action-routed "
                "question spends\n"
                "inference rounds on a reply aorta cannot read -- billed or "
                "self-hosted,\n"
                "they are wasted either way. The answer then comes back\n"
                "late, degraded, or not at all.\n"
                'Set llm_tool_mode = "native" in chat.toml, or\n'
                # ``export`` kept in the same literal as the variable, which is
                # also what lets the property test see the pair.
                "export AORTA_CHAT_LLM_TOOL_MODE=native.\n" + native_note
            ),
        )
        return
    report.add(
        "llm tool mode",
        OK,
        f"text, on {provider} ({model})",
        hint=_with_native_note(_TEXT_MODE_HINT, native_note),
    )


def _check_backend(report: Report) -> None:
    """Whether the configured LLM backend answers.

    Calls ``probe`` rather than ``preflight``. The local backend's preflight is
    deliberately permissive -- it waits five minutes and then starts anyway, so
    the REPL survives a server that is still loading weights -- which made this
    check report a confident ``ok`` for the single most likely failure, and
    spend preflight's whole budget doing it -- 302s measured against a closed
    port. ``probe`` raises instead, on a diagnostic's budget.
    """
    import asyncio

    try:
        # Both imports belong under this guard: `unreachable` reaches httpx and
        # openai, so on the install this check exists to diagnose -- the one
        # missing the chat extra -- importing it at function scope would crash
        # the command instead of reporting the missing dependency.
        from aorta.chat.inference.providers.factory import get_backend
        from aorta.chat.inference.unreachable import BackendUnreachableError

        backend = get_backend()
    except (ImportError, ValueError) as exc:
        report.add("llm backend", FAIL, str(exc))
        return

    try:
        asyncio.run(backend.probe(timeout=_BACKEND_PROBE_TIMEOUT))
    except Exception as exc:
        # Deliberately broad: a backend may raise anything from httpx, openai or
        # litellm, and a doctor command that propagates one of those has failed
        # at its only job.
        #
        # BackendUnreachableError's message is already written to be read by the
        # operator whose command just stopped, so prefixing it with a class name
        # would only add noise. Anything else needs its type named.
        hint = str(exc)
        if not isinstance(exc, BackendUnreachableError):
            hint = f"{type(exc).__name__}: {exc}"
        report.add(
            "llm backend",
            FAIL,
            f"{backend.describe()} did not answer",
            hint=hint,
        )
        return
    report.add("llm backend", OK, backend.describe())


def run_checks(*, backend: bool = True) -> Report:
    """Run every check and return the report.

    Args:
        backend: Whether to probe the LLM backend. Off in tests, and worth
            skipping when the user only wants the local picture.
    """
    report = Report()
    _check_python(report)
    _check_extras(report)
    # Labelled rather than derived from ``__name__`` so a check that raises is
    # still reported under the name the user is looking for.
    for label, check in (
        ("sqlite", _check_sqlite),
        ("embedding provider", _check_embedding_model),
        ("chat index", _check_index),
        ("llm tool mode", _check_tool_mode),
    ):
        try:
            check(report)
        except Exception as exc:  # a doctor must not die on its own diagnostics
            logger.debug("doctor check %s raised", label, exc_info=True)
            report.add(label, FAIL, f"{type(exc).__name__}: {exc}")
    if backend:
        _check_backend(report)
    else:
        report.add("llm backend", SKIP, "not checked (--no-backend)")
    return report


__all__ = ["FAIL", "OK", "SKIP", "WARN", "Check", "Report", "run_checks"]
