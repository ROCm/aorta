"""Filesystem-safe slug helper -- stdlib-only copy for ``aorta.bundle``.

Deliberately duplicated from :func:`aorta.triage.output.safe_slug` (and
its ``NO_TICKET_SLUG`` constant) so that importing ``aorta.bundle`` does
NOT transitively pull in PyYAML and the whole ``aorta.triage`` /
``aorta.registry`` stack at import time. Issue #196 acceptance
criterion 7 pins the bundle package to stdlib + ``click`` only;
``aorta.triage.output`` imports ``yaml`` (PR #199 review).

Keep this behaviourally in sync with the canonical
``aorta.triage.output.safe_slug``. ``tests/bundle/test_no_new_deps.py``
asserts byte-for-byte parity between the two implementations so this
copy cannot silently drift (e.g. if triage tightens the slug regex for
a path-traversal fix).
"""

from __future__ import annotations

import re

#: Basename a run dir carries when ``aorta probe`` ran without a ticket.
NO_TICKET_SLUG = "_no_ticket_"

# Replace anything that isn't ``[A-Za-z0-9_.-]`` with ``_`` so a ticket
# never creates surprise subdirectories.
_SAFE_RE = re.compile(r"[^A-Za-z0-9_.\-]")
# ``.`` / ``..`` survive the character scrub but are still path-meaningful;
# keep them out so a ticket like ``..`` cannot escape the output tree.
_RESERVED_SLUGS = frozenset({".", ".."})


def safe_slug(value: str) -> str:
    """Turn a ticket / name into a safe directory component.

    Replaces anything outside ``[A-Za-z0-9_.-]`` with ``_`` and rewrites
    the reserved ``.`` / ``..`` components (and empty input) to ``_`` so
    the result can never refer to the current or parent directory.
    """
    cleaned = _SAFE_RE.sub("_", value)
    if not cleaned or cleaned in _RESERVED_SLUGS:
        return "_"
    return cleaned


__all__ = ["NO_TICKET_SLUG", "safe_slug"]
