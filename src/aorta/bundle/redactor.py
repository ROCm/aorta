"""Redactor protocol + identity (no-op) implementation for ``aorta bundle``.

The bundle CLI does NOT own the scrubbers. Per issue #196 and the
``aorta probe`` Phase 3 rubric (#188), the actual env-key glob,
path, and IP scrubbers live in ``aorta.probe.redaction`` (Phase 3
of #188). Until that module lands, every bundle uses the
:class:`IdentityRedactor` here -- a no-op that copies bytes
through and reports zero counts.

The :class:`Redactor` ABC is the public hand-off point. Phase 3 of
#188 will register a ``RedactingRedactor`` that the CLI loads via
``--redaction-from <recipe>``; everything else in the bundle
pipeline (staging, manifest, tarballing) is redactor-agnostic.

The contract is intentionally small (one method, one return type)
so a future ``redaction_v2`` implementation does not require a
breaking change to the bundle module.
"""

from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RedactionCounts:
    """Per-file scrubber output -- accumulated into the manifest.

    Values are non-negative ``int``. The default-constructed
    instance is the "identity" outcome (no scrubbing applied) and
    is what :class:`IdentityRedactor` returns. Issue #196 documents
    these three counters as the manifest's per-file shape; Phase 3
    of #188 fills them in with real scrubber output.

    ``bytes_in`` / ``bytes_out`` are NOT documented in the issue
    body but are needed to make the manifest self-describing
    (the bytes-in / bytes-out columns in the issue example come
    from these). We keep them on the same dataclass because every
    redactor knows them as a side effect of the copy and recording
    them elsewhere would force a separate ``stat()`` on every file.
    """

    env_keys_removed: int = 0
    paths_rewritten: int = 0
    ips_rewritten: int = 0
    bytes_in: int = 0
    bytes_out: int = 0


class Redactor(ABC):
    """Strategy for copying a single file with optional scrubbing.

    Implementations:

    * :class:`IdentityRedactor` -- the default. Used when no
      ``redaction:`` block is present in the source recipe (which
      is every probe recipe in Phase 1/2 of #188 -- the recipe
      loader rejects ``redaction:`` until Phase 3 of #188 lands).
    * ``aorta.probe.redaction.RedactingRedactor`` (Phase 3 of #188,
      not yet shipped) -- applies env-key / path / IP scrubbers
      configured from the recipe's ``redaction:`` block.

    The :attr:`kind` string is recorded in the manifest so a
    downstream consumer can tell at a glance which redactor wrote
    the bundle. Stable identifiers; see the manifest schema in
    ``docs/probe-188/bundle.md``.
    """

    #: Stable string identifier recorded under ``manifest.json::redactor_kind``.
    kind: str = "abstract"

    @abstractmethod
    def scrub_file(self, src: Path, dst: Path) -> RedactionCounts:
        """Copy ``src`` to ``dst`` (creating parent dirs as needed).

        Returns the per-file count summary. Implementations MUST:

        * Read from ``src`` only -- never modify the source tree
          (issue #196 acceptance criterion 5: "originals
          untouched"). The bundle pipeline tests this directly.
        * Create the destination's parent directory if missing.
        * Return :class:`RedactionCounts` with ``bytes_in`` /
          ``bytes_out`` populated.

        Raises:
            OSError: forwarded from the underlying copy. The bundle
                writer wraps these into a single
                ``BundleError``-shaped failure rather than letting
                the operator see a raw stack trace.
        """


class IdentityRedactor(Redactor):
    """No-op redactor: ``shutil.copyfile`` plus zero counts.

    Used by every ``aorta bundle`` invocation today (issue #196).
    Will be replaced by a Phase-3-of-#188 redactor when the
    ``--redaction-from <recipe>`` flag is wired through. The
    ``manifest.json`` written under this redactor reports
    ``redaction_applied: false`` and ``redactor_kind: "identity"``.

    Copy semantics: ``shutil.copyfile`` (vs ``copy`` / ``copy2``)
    moves bytes only, then ``shutil.copymode`` carries the source's
    permission bits onto the staged copy. Preserving the mode is a
    defense-in-depth requirement, not a nicety: ``aorta probe``
    creates ``probe.env`` at ``0600`` (owner-only), and a plain
    ``copyfile`` would land it at the process umask default
    (typically ``0644``) -- *widening* a secret-bearing file as it
    enters a shareable bundle. We never want the bundle copy to be
    less restrictive than the original (PR #199 review). We do not
    use ``copy2`` because timestamps/xattrs are not part of the
    bundle contract -- only the access mode matters here.
    """

    kind = "identity"

    def scrub_file(self, src: Path, dst: Path) -> RedactionCounts:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        # Carry the source mode so a 0600 probe.env is not widened to
        # the umask default inside the shareable bundle.
        shutil.copymode(src, dst)
        size = dst.stat().st_size
        return RedactionCounts(bytes_in=size, bytes_out=size)


__all__ = [
    "IdentityRedactor",
    "RedactionCounts",
    "Redactor",
]
