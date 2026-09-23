"""Typed, calibrated decisions from a local encoder (Laya), and the offline
measurement that decides whether they are worth integrating.

Three things live here, in the order they have to happen:

* :mod:`aorta.laya.predictor` -- the seam. A narrow Protocol, a fake, and a
  real checkpoint-backed implementation. Watch's clean-gate, the probe agent's
  proposer and the log finder are each meant to be written against this one
  contract.
* :mod:`aorta.laya.corpus` -- three builders that turn artifacts already on
  disk into labelled JSONL. Nothing downstream is measurable without them.
* :mod:`aorta.laya.eval` and :mod:`aorta.laya.gate` -- the scoring, which is
  pure and unit-tested, and the go/no-go comparison against three baselines,
  which needs a model and therefore is not a test.

Nothing in this package imports torch, ``laya`` or a tokenizer at module scope.
``[chat-cli]`` is CI-gated against resolving torch at all, and
``_HEAVY_PREFIXES`` in ``tests/cli/test_chat_boundaries.py`` asserts that
``import aorta.cli`` pulls in neither torch nor onnxruntime; the same file
asserts the agents stay free of torch too. Every model import therefore sits
inside the function that needs it.
"""

from __future__ import annotations

from aorta.laya.predictor import (
    CHECKPOINTS,
    VERIFIED_LAYA_VERSION,
    Answer,
    Calibration,
    Choice,
    ChoiceAnswer,
    ClampedBucket,
    FakeLayaPredictor,
    LayaAgentPredictor,
    LayaPredictor,
    LayaUnavailable,
    Noul,
    NoulAnswer,
    Question,
    ask_choice,
    ask_noul,
    ask_one,
    bucket_for,
    make_predictor,
)

__all__ = [
    "CHECKPOINTS",
    "VERIFIED_LAYA_VERSION",
    "Answer",
    "Calibration",
    "Choice",
    "ChoiceAnswer",
    "ClampedBucket",
    "FakeLayaPredictor",
    "LayaAgentPredictor",
    "LayaPredictor",
    "LayaUnavailable",
    "Noul",
    "NoulAnswer",
    "Question",
    "ask_choice",
    "ask_noul",
    "ask_one",
    "bucket_for",
    "make_predictor",
]
