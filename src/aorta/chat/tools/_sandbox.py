"""The one path-containment rule the chat tools share.

Every tool that takes a path takes it from the *model*, so each one has to
refuse anything outside its sandbox. That rule was written three times -- in
:mod:`~aorta.chat.tools.files`, in :mod:`~aorta.chat.tools.artifacts` and
inline in :mod:`~aorta.chat.tools.search` -- and the three copies had already
drifted to three spellings of the refusal, which is what a shared rule kept in
three places does next.

The *root* stays the caller's: the two sandboxes are genuinely different (the
codebase is the installed package by default, while run artifacts live wherever
``--output`` pointed), and every caller already reads its own root for the
relative paths it prints. Passing it in keeps one read per call site, so the
path a tool checks cannot differ from the path it renders against.
"""

from __future__ import annotations

from pathlib import Path

#: Sandbox names used in the refusal. Constants rather than literals at each
#: call site, because the wording drifting per module is the defect this
#: module exists to close.
AORTA_ROOT_LABEL = "AORTA root"
RUNS_ROOT_LABEL = "run root"


def resolve_within(root: Path, path: str, root_label: str) -> Path:
    """Resolve *path* under *root*, refusing anything that escapes it.

    ``relative_to`` on the resolved path rather than a string prefix test:
    ``/aorta-old`` starts with the characters of ``/aorta`` without being
    inside it. ``resolve()`` runs first, so a symlink is judged by where it
    points rather than by where it sits, and an absolute *path* -- which
    ``root / path`` discards the root for -- is caught by the same test.

    *root_label* names the sandbox in the refusal and affects nothing else.

    Raises:
        ValueError: *path* resolves outside *root*. The tools turn this into an
            ``Error: ...`` string rather than letting it propagate, so the
            model can correct its own argument instead of aborting the run.
    """
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        raise ValueError(f"path escapes the {root_label}: {path}") from None
    return resolved


__all__ = ["AORTA_ROOT_LABEL", "RUNS_ROOT_LABEL", "resolve_within"]
