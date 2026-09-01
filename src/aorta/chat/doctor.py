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
        report.add(
            "embedding model cache",
            WARN,
            f"{state['model']} is not cached, but HuggingFace is reachable",
            hint=(
                "It will be downloaded (~65 MB) on the first query or index "
                "build. Pre-warm it now with: aorta chat index build"
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
        report.add(
            "index manifest",
            FAIL,
            "does not match this install; queries are refused",
            hint="\n".join(result.refusals),
            procedure=(
                "This is not a cosmetic mismatch. The index holds vectors from a "
                "different embedding model, so retrieval would compare numbers "
                "that are not comparable and answer confidently from the wrong "
                "chunks.\n"
                "  aorta chat index fetch     get the index matching this install\n"
                "  aorta chat index build     rebuild with the configured provider"
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


def _check_backend(report: Report) -> None:
    """Whether the configured LLM backend answers."""
    import asyncio

    try:
        from aorta.chat.inference.providers.factory import get_backend

        backend = get_backend()
    except (ImportError, ValueError) as exc:
        report.add("llm backend", FAIL, str(exc))
        return

    try:
        asyncio.run(backend.preflight())
    except Exception as exc:
        # Deliberately broad: a backend may raise anything from httpx, openai or
        # litellm, and a doctor command that propagates one of those has failed
        # at its only job.
        report.add(
            "llm backend",
            FAIL,
            f"{backend.describe()} did not answer",
            hint=f"{type(exc).__name__}: {exc}",
        )
        return
    report.add("llm backend", OK, backend.describe())


def run_checks(*, backend: bool = True) -> Report:
    """Run every check and return the report.

    Args:
        backend: Whether to preflight the LLM backend. Off in tests, and worth
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
