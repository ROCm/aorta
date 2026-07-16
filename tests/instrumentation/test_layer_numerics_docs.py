"""Docs regression check for the ``NANLOG_SPEC`` copy-paste examples.

Pure JSON parsing -- deliberately does NOT import torch, so it runs in minimal
CI too (unlike ``test_layer_numerics_spec.py``, which needs torch to import the
logger). Guards against a doc example drifting into invalid JSON that would
silently fall back to the flat defaults when pasted.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

_DOC = Path(__file__).resolve().parents[2] / "docs" / "layer-numerics.md"


def test_docs_spec_examples_are_valid_json():
    """Every NANLOG_SPEC example in the customer doc must parse as JSON — both the
    ``NANLOG_SPEC='...'`` command forms and the backticked ``{...}`` table cells —
    so a user who copies one gets the intended config, not a silent fallback."""
    text = _DOC.read_text(encoding="utf-8")
    specs = re.findall(r"NANLOG_SPEC='([^']+)'", text) + re.findall(r"`(\{.*?\})`", text)
    assert specs, "no NANLOG_SPEC examples found in the doc"
    for s in specs:
        json.loads(s)   # raises if a doc example is not valid JSON
