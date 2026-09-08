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

import logging
import re
import socket
import sys
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
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

#: Downloading the embedding weights and nothing else -- which is what a
#: "pre-warm" is. The same invocation ``fastembed_bge.PRE_SEED_PROCEDURE`` uses
#: in its first step, quoted rather than imported because that procedure is a
#: paragraph and this is a line; a test pins the two together.
#:
#: Deliberately *not* ``aorta chat index build``, which this check used to
#: advise. That command's ``--output`` defaults to the index this install
#: already reads and its corpus defaults to ``src/aorta`` alone, so as a
#: pre-warm it overwrites a fetched index with one that has no ``docs/`` and no
#: ``README.md`` in it -- and says nothing about having done either.
_WARM_COMMAND = (
    "python -c 'from fastembed import TextEmbedding; "
    'TextEmbedding("{model}", cache_dir="{cache}")\''
)

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

#: Model names that mark a reasoning model. A heuristic -- a gateway can call a
#: deployment anything, and a vLLM server is launched under whatever name its
#: operator gave it -- so it only decides whether the tool-mode check warns or
#: merely informs. Both branches name ``native`` and the symptom, because the
#: case this cannot recognise is exactly the one a user reaches after hitting it.
_REASONING_MODEL_PATTERN = re.compile(r"gpt-oss|qwq|reasoner|reasoning|\b(?:o[1-4]|r1)\b")


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


def _probe_huggingface() -> bool:
    """Whether the HuggingFace CDN answers. A TCP connect, not a model download."""
    try:
        with socket.create_connection((_HF_HOST, _HF_PORT), timeout=_HF_PROBE_TIMEOUT):
            return True
    except OSError as exc:
        logger.debug("HuggingFace probe failed: %s", exc)
        return False


def _index_is_healthy() -> bool:
    """Whether an index is present and this install can query it as-is.

    Asked so the cold-cache hint can be conditioned on what the user already
    has. ``_check_embedding_model`` runs before ``_check_index`` because
    provider-before-index reads better in the report, so it cannot read the
    later check's result; running the same validation twice costs one sqlite
    open and no network, which is cheaper than reordering the output.

    Warnings do not disqualify an index. Source drift is a reason to refresh
    it, not a reason for advice that would replace it with a worse one.
    """
    from aorta.chat.config import settings
    from aorta.chat.rag.index_ops import check_index

    if not settings.index_file.exists():
        return False
    try:
        return not check_index(settings.index_file, strict=False).refusals
    except Exception:
        # Unreadable, unparseable, no manifest: all mean the same thing here,
        # which is that there is nothing worth protecting from a rebuild.
        logger.debug("index health probe failed", exc_info=True)
        return False


def _check_embedding_model(report: Report) -> None:
    """Whether queries can be embedded at all, and what to do when they cannot."""
    from aorta.chat.rag.embeddings.factory import get_provider

    try:
        provider = get_provider()
    except ValueError as exc:
        report.add("embedding provider", FAIL, str(exc))
        return
    report.add("embedding provider", OK, provider.describe())

    if provider.name != "local":
        # A remote embedder needs a key and an endpoint, not a cache; the
        # provider reports its own configuration problems when built.
        report.add("embedding model cache", SKIP, "remote provider; no local weights needed")
        return

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
        warm = _WARM_COMMAND.format(model=state["model"], cache=state["cache_dir"])
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
                    "Your index is present and matches this install, and the "
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
    """Index presence, and whether its manifest matches this install."""
    from aorta.chat.config import settings
    from aorta.chat.rag import manifest as manifest_mod

    index_file = settings.index_file
    if not index_file.exists():
        report.add(
            "chat index",
            FAIL,
            f"absent at {index_file}",
            hint=(
                "aorta chat index fetch     download the prebuilt index\n"
                "aorta chat index build     build one from local code"
            ),
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
    if result.warnings:
        report.add(
            "index manifest",
            WARN,
            result.manifest.describe(),
            hint="\n".join(result.warnings),
        )
        return
    report.add("index manifest", OK, result.manifest.describe())


def _check_tool_mode(report: Report) -> None:
    """Which protocol the act loop will use to call tools, and whether it fits.

    ``text`` is the default because it has no endpoint requirement, so it is the
    only mode a stock local vLLM can drive. A reasoning model cannot drive it:
    it puts its working in a channel of its own and returns empty ``content``
    where the ``ACTION:`` line was expected, so every action-routed query spends
    its whole retry budget and answers nothing. Until this check existed the
    first signal of that was the failed query.

    That holds for a locally served reasoning model as much as a remote one --
    the channel is the model's, not the endpoint's -- so both flows are read.
    What the provider changes is the remedy: turning ``native`` on costs a
    setting remotely and a setting plus two server flags on vLLM.
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
    if mode == "native":
        report.add("llm tool mode", OK, "native (the provider's function-calling API)")
        return
    model = native_note = ""
    if provider in _REMOTE_LLM_PROVIDERS:
        model, native_note = str(settings.remote_llm_model or ""), _REMOTE_NATIVE_NOTE
    elif provider == "vllm":
        model, native_note = str(settings.vllm_model or ""), _VLLM_NATIVE_NOTE
    if not model:
        # A provider no backend is registered for -- ``_check_backend`` reports
        # that -- or one whose model setting is empty. Either way there is no
        # name to read, and guessing which setting holds it would invent one.
        report.add("llm tool mode", OK, "text (ACTION: lines parsed out of the reply)")
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
                "returns empty content instead, so the act loop re-prompts "
                "until it gives\n"
                "up and the question is answered with nothing.\n"
                'Set llm_tool_mode = "native" in chat.toml, or '
                "AORTA_CHAT_LLM_TOOL_MODE=native.\n" + native_note
            ),
        )
        return
    report.add(
        "llm tool mode",
        OK,
        f"text, on {provider} ({model})",
        hint=(
            "If an action-routed question comes back with no answer, the first "
            "thing\n"
            'to change is llm_tool_mode = "native". Reasoning models cannot '
            "write the\n"
            "ACTION: lines text mode parses, and a deployment can be served "
            "under any\n"
            "name.\n" + native_note
        ),
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
