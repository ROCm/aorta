"""Laya on the chat path: the ONNX runtime, its artifact, and the two questions.

Four modules, in the order they matter to a reader:

* :mod:`~aorta.chat.laya.questions` -- the exact text ``router_node`` and
  ``selector_node`` ask, in one place, so nothing re-phrases it.
* :mod:`~aorta.chat.laya.artifact` -- what an exported artifact is, and which
  weights and which temperature fit produced it.
* :mod:`~aorta.chat.laya.onnx_predictor` -- the
  :class:`~aorta.laya.predictor.LayaPredictor` implementation that runs it.
* :mod:`~aorta.chat.laya.export` -- the torch-side tool that produces one.

**Why this package is under** ``aorta/chat/`` **and not under** ``aorta/laya/``.
Decision 22 splits the integration by runtime: the CIA tracks run torch out of
the ``[laya]`` extra, and the chat path runs ONNX on the onnxruntime
``fastembed`` already installs, because ``[chat-cli]`` is CI-gated against
resolving torch at all. ``onnxruntime`` and ``fastembed`` are themselves in
``_HEAVY_PREFIXES``, so the runtime is a chat-extra dependency and putting its
loader in the core ``aorta.laya`` package would either drag a chat dependency
into core or force ``aorta.laya`` to import ``aorta.chat`` -- which
``tests/cli/test_chat_boundaries.py`` forbids outright. The import arrow points
chat -> core and this package is the chat end of it: it reads
:mod:`aorta.laya.predictor`'s Protocol and types, and nothing in core reads
anything here.

Importing this package costs an import of ``dataclasses`` and ``json``.
``onnxruntime``, ``tokenizers``, ``numpy`` and ``torch`` are all imported inside
the functions that need them.
"""


from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

from aorta.chat.laya.artifact import ArtifactUnavailable

if TYPE_CHECKING:  # the Protocol only; no runtime import of anything heavy
    from aorta.laya.predictor import LayaPredictor

logger = logging.getLogger(__name__)

#: One predictor per process, because an ONNX session costs seconds to build
#: and a chat session routes on every turn. Guarded because two Chainlit
#: sessions can reach it at once; the same reason ``FastembedBgeEmbeddings``
#: locks its model.
_predictor: LayaPredictor | None = None
_predictor_lock = threading.Lock()


def reset_predictor() -> None:
    """Drop the cached predictor. For tests, and for a settings change.

    Exported for the reason ``reset_tool_mode_escalation`` is: the cache is
    deliberately process-wide, the whole test run is one process, and an
    artifact path set by one test would otherwise still be loaded in the next.
    """
    global _predictor
    with _predictor_lock:
        _predictor = None


def chat_predictor() -> LayaPredictor:
    """The predictor the chat nodes ask, built from settings on first use.

    Always the ONNX one. There is deliberately no way to select
    :class:`~aorta.laya.predictor.FakeLayaPredictor` from configuration, even
    though the seam's own ``make_predictor`` offers it: the fake answers from a
    hash of the question and the state, its own docstring says no number it
    returns may be reported as a measurement of anything, and a chat turn is a
    place where such a number would be shown to a user as a routing decision.
    Tests inject a predictor directly instead, which is a thing a test can do
    and an operator cannot.

    Raises :class:`~aorta.chat.laya.artifact.ArtifactUnavailable` when the
    artifact is absent or unreadable. Callers treat that as "no Laya tier" and
    fall back, which is why it is raised rather than logged: a node that got a
    predictor answering 0.5 to everything would gate on noise and look merely
    inaccurate rather than broken.
    """
    global _predictor
    if _predictor is None:
        with _predictor_lock:
            if _predictor is None:
                from aorta.chat.config import settings
                from aorta.chat.laya.onnx_predictor import OnnxLayaPredictor

                _predictor = OnnxLayaPredictor(
                    settings.laya_artifact_path,
                    verify_digest=settings.laya_verify_digest,
                )
    return _predictor


__all__ = [
    "ArtifactUnavailable",
    "chat_predictor",
    "reset_predictor",
]
