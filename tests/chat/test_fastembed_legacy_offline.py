"""FastEmbed 0.8.1 can reuse a complete cache written with 0.8.0 casing."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from aorta.chat.rag.embeddings import fastembed_bge
from aorta.chat.rag.embeddings.fastembed_bge import (
    DEFAULT_MODEL,
    FastembedBgeEmbeddings,
)

_TINY_ONNX = (
    "CAg60QIKJwoJaW5wdXRfaWRzEglpZHNfZmxvYXQiBENhc3QqCQoCdG8YAaABAgomCglp"
    "ZHNfZmxvYXQKBGF4ZXMSCGV4cGFuZGVkIglVbnNxdWVlemUKLAoIZXhwYW5kZWQKB3Jl"
    "cGVhdHMSEWxhc3RfaGlkZGVuX3N0YXRlIgRUaWxlEhxhb3J0YS1mYXN0ZW1iZWQtb2Zm"
    "bGluZS10ZXN0Kg0IARAHOgECQgRheGVzKhMIAxAHOgQBAYADQgdyZXBlYXRzWigKCWlu"
    "cHV0X2lkcxIbChkIBxIVCgcSBWJhdGNoCgoSCHNlcXVlbmNlWi0KDmF0dGVudGlvbl9t"
    "YXNrEhsKGQgHEhUKBxIFYmF0Y2gKChIIc2VxdWVuY2ViNQoRbGFzdF9oaWRkZW5fc3Rh"
    "dGUSIAoeCAESGgoHEgViYXRjaAoKEghzZXF1ZW5jZQoDCIADQgQKABAN"
)


def _write_legacy_snapshot(cache_root: Path) -> Path:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    repository = cache_root / "models--qdrant--bge-small-en-v1.5-onnx-q"
    snapshot = repository / "snapshots" / "legacy-revision"
    snapshot.mkdir(parents=True)
    (repository / "refs").mkdir()
    (repository / "refs" / "main").write_text("legacy-revision\n", encoding="utf-8")
    (snapshot / "model_optimized.onnx").write_bytes(base64.b64decode(_TINY_ONNX))
    (snapshot / "config.json").write_text(
        json.dumps({"pad_token_id": 0}),
        encoding="utf-8",
    )
    tokenizer = Tokenizer(
        WordLevel(
            {"[PAD]": 0, "[UNK]": 1, "hello": 2},
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(snapshot / "tokenizer.json"))
    (snapshot / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "model_max_length": 32,
                "pad_token": "[PAD]",
                "unk_token": "[UNK]",
            }
        ),
        encoding="utf-8",
    )
    (snapshot / "special_tokens_map.json").write_text(
        json.dumps({"pad_token": "[PAD]", "unk_token": "[UNK]"}),
        encoding="utf-8",
    )
    return snapshot


def test_fastembed_081_loads_a_legacy_lowercase_cache_offline(
    monkeypatch,
    tmp_path: Path,
):
    """Detection and construction resolve the same snapshot without a download."""
    pytest.importorskip("fastembed", reason="the integration needs the chat-cli extra")
    from fastembed.common import model_management

    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert fastembed_bge._source_repo(DEFAULT_MODEL) == "Qdrant/bge-small-en-v1.5-onnx-Q"
    snapshot = _write_legacy_snapshot(tmp_path)

    def network_is_a_failure(*_args, **_kwargs):
        raise AssertionError("offline legacy loading reached snapshot_download")

    monkeypatch.setattr(model_management, "snapshot_download", network_is_a_failure)

    vector = FastembedBgeEmbeddings().embed_query("hello")

    assert fastembed_bge._cached_model_path(DEFAULT_MODEL) == snapshot
    assert len(vector) == 384
    assert sum(value * value for value in vector) == pytest.approx(1.0)
