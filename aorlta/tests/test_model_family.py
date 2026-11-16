"""The model families (model_family): told apart by config.json, each with its loader, decoder prefix and the
engines' attention kernels."""

from __future__ import annotations

import json

import pytest

from aorlta import model_family as MF
from aorlta.orchestrator import wiring


def _config_dir(tmp_path, model_type, arch):
    d = tmp_path / model_type
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": model_type, "architectures": [arch]}))
    return d


def test_a_dense_qwen3_runs_on_tokenspeeds_triton_attention_and_qwen3_5_on_its_own_choice(tmp_path):
    q3 = _config_dir(tmp_path, "qwen3", "Qwen3ForCausalLM")
    q35 = _config_dir(tmp_path, "qwen3_5", "Qwen3_5ForConditionalGeneration")
    other = _config_dir(tmp_path, "llama", "LlamaForCausalLM")
    assert MF.engine_attention_backend(q3) == "triton" and MF.read_family(q3).attention_backend == "triton"
    assert MF.engine_attention_backend(q35) is None
    assert MF.engine_attention_backend(other) is None  # an unknown family keeps TokenSpeed's choice
    assert MF.engine_attention_backend(tmp_path / "missing") is None
    with pytest.raises(MF.ModelFamilyError):
        MF.read_family(other)


@pytest.mark.parametrize("model_type, arch, backend", [("qwen3", "Qwen3ForCausalLM", "triton"),
                                                       ("qwen3_5", "Qwen3_5ForConditionalGeneration", None)])
def test_the_run_starts_its_engines_with_the_base_familys_attention_kernels(tmp_path, model_type, arch, backend):
    import types

    base = _config_dir(tmp_path, model_type, arch)
    cfg = types.SimpleNamespace(  # what engine_handles reads of a RunConfig
        model=types.SimpleNamespace(base_dir=str(base), adapter_dir=None),
        rollout=types.SimpleNamespace(P=4, K=2, engines=(0, 1), gpu_memory_utilization=0.45, port_base=None),
        orchestrator=types.SimpleNamespace(run_seed=5))
    handles = wiring.engine_handles(cfg, 4096)
    assert [h.engine_kwargs.get("attention_backend") for h in handles] == [backend, backend]
