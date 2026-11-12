"""Shared fixtures for the WP2 rollout tests (CPU only)."""

from __future__ import annotations

import os
import re
import types
from pathlib import Path

import pytest

from aorlta.ids import sha256_ids
from aorlta.rollout.chain import RolloutManager, RolloutSettings
from aorlta.rollout.compile_check import FakeCompiler
from aorlta.rollout.engine_pool import EnginePool
from aorlta.rollout.engine_proc import InProcessEngine
from aorlta.rollout.fake_engine import BYTE_TOKEN_BASE, FakeEngine, FakeEngineConfig
from aorlta.rollout.templates import THINK_PREFIX, render_turn0
from aorlta.schemas import PromptRecord, TaskRef

# The Qwen3.5-4B tokenizer from the Hugging Face cache (HF_HOME), and the stand-in dataset (AORLTA_STANDIN_DATASET);
# the tests that need either skip without it.
QWEN_DIR = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))) / "hub" / "models--Qwen--Qwen3.5-4B" / "snapshots" / "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
DATASET = Path(os.environ.get("AORLTA_STANDIN_DATASET", "/data/aorlta/standin-pool"))
CLANG_DATA = Path(__file__).resolve().parent / "data" / "clang"

ENGINE_KW = {"dtype": "bfloat16", "sampling_backend": "triton", "enable_output_logprobs": True,
             "enable_prefix_caching": True}
SPECIAL = {"<|endoftext|>": 248044, "<|im_start|>": 248045, "<|im_end|>": 248046, "<think>": 248068,
           "</think>": 248069}
_SPECIAL_RE = re.compile("|".join(re.escape(s) for s in SPECIAL))
_SPECIAL_TEXT = {v: k for k, v in SPECIAL.items()}


class SpecialByteTokenizer:
    """The fake engine's byte tokenizer (one id per UTF-8 byte) plus the five control tokens, with the
    system/user part of Qwen3.5's chat template. `log`, if given, records every encode call."""

    def __init__(self, log: list | None = None):
        self.log = log

    def encode(self, text: str) -> list[int]:
        if self.log is not None:
            self.log.append(("encode", text))
        out, i = [], 0
        for m in _SPECIAL_RE.finditer(text):
            out += [BYTE_TOKEN_BASE + b for b in text[i:m.start()].encode("utf-8")]
            out.append(SPECIAL[m.group(0)])
            i = m.end()
        out += [BYTE_TOKEN_BASE + b for b in text[i:].encode("utf-8")]
        return out

    def decode(self, ids) -> str:
        parts, buf = [], bytearray()
        for t in ids:
            if t in _SPECIAL_TEXT:
                parts.append(buf.decode("utf-8", errors="replace"))
                buf.clear()
                parts.append(_SPECIAL_TEXT[t])
            else:
                buf.append(t - BYTE_TOKEN_BASE)
        parts.append(buf.decode("utf-8", errors="replace"))
        return "".join(parts)

    def render(self, messages, thinking, add_generation_prompt=True) -> str:
        text = "".join(f"<|im_start|>{m['role']}\n{m['content'].strip()}<|im_end|>\n" for m in messages)
        return text + ("<|im_start|>assistant\n" + THINK_PREFIX[thinking] if add_generation_prompt else "")


def make_prompts(tok, n: int, thinking: str = "off", source: str = "standin", tag: str = "t") -> list[PromptRecord]:
    out = []
    for i in range(n):
        messages = [{"role": "system", "content": "You are a test."},
                    {"role": "user", "content": f"Fix kernel {tag}{i}: " + "listing line\n" * (3 + i)}]
        ids = render_turn0(tok, messages, thinking)
        task = TaskRef(contract_id=f"{source}:{tag}{i}", parent_id=None, parent_healthy=None, reference_kind=None,
                       task_sha256="0" * 64)
        out.append(PromptRecord(prompt_id=f"{source}:{tag}{i}", turn0_ids=None, messages=messages,
                                prompt_ids_sha256=sha256_ids(ids), task=task, source=source))
    return out


def fake_engines(n: int, cfg: FakeEngineConfig | None = None, **kw) -> list[FakeEngine]:
    cfg = cfg or FakeEngineConfig(seed=11, min_tokens=120, max_tokens=300)
    return [FakeEngine(cfg, engine_id=i, **{**ENGINE_KW, **kw}) for i in range(n)]


def make_pool(engines, *, run_seed: int = 7, selftest_tok=None, start: bool = True, **kw) -> EnginePool:
    handles = [e if not isinstance(e, FakeEngine) else InProcessEngine(i, e) for i, e in enumerate(engines)]
    pool = EnginePool(handles, run_seed=run_seed, **kw)
    if start:
        tok = selftest_tok or SpecialByteTokenizer()
        pool.start([tok.encode(f"self-test prompt {i}") for i in range(4)])
    return pool


def settings(**kw) -> RolloutSettings:
    base = dict(run_id="wp2-test", run_seed=7, source="standin", check_profile="standin-static", K=4,
                max_repair_turns=2, thinking="off", temperature=0.7, top_p=0.95, top_k=-1, max_new_tokens=600,
                max_feedback_chars=2000, max_model_len=20000, P=None)
    base.update(kw)
    return RolloutSettings(**base)


def make_manager(pool, *, tok=None, compiler=None, leak_check=None, **kw) -> RolloutManager:
    return RolloutManager(settings(**kw), pool, tok or SpecialByteTokenizer(), compiler or FakeCompiler(),
                          leak_check=leak_check, compile_workers=8)


@pytest.fixture(scope="session")
def qwen():
    pytest.importorskip("transformers")
    if not (QWEN_DIR / "tokenizer.json").exists():
        pytest.skip("the Qwen3.5-4B tokenizer is not cached")
    from aorlta.rollout.templates import HFTokenizer

    return HFTokenizer.load(QWEN_DIR)


@pytest.fixture
def rt():
    """Helpers as one namespace (test modules cannot import conftest under --import-mode=importlib)."""
    return types.SimpleNamespace(
        SpecialByteTokenizer=SpecialByteTokenizer, make_prompts=make_prompts, fake_engines=fake_engines,
        make_pool=make_pool, settings=settings, make_manager=make_manager, ENGINE_KW=ENGINE_KW,
        CLANG_DATA=CLANG_DATA, DATASET=DATASET, QWEN_DIR=QWEN_DIR,
    )
