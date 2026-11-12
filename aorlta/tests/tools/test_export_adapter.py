"""tools/export_adapter.py: a training run's checkpoint as a PEFT adapter. One wired iteration on CPU
(test_wiring_e2e's tiny model and engines) writes a checkpoint; its export, merged into the base by
tools/merge_base_adapter.py, is exactly the weights the engines were served, and peft loads it."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "orchestrator"))
import test_wiring_e2e as E  # noqa: E402 - the tiny model, its twin engines, the probes and the dataset

pytestmark = pytest.mark.heavy

tiny_base, dataset = E.tiny_base, E.dataset


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


X = _load("export_adapter", ROOT / "tools" / "export_adapter.py")
M = _load("merge_base_adapter_for_export", ROOT / "tools" / "merge_base_adapter.py")


@pytest.fixture(scope="module")
def trained(tmp_path_factory, tiny_base, dataset):
    """One wired training iteration: its run config, its checkpoint and the weights it served."""
    tmp = tmp_path_factory.mktemp("export")
    cfg = E.run_config(tmp, tiny_base, dataset, run_id="exported", iterations=1)
    orch, engines = E.build(cfg, tiny_base)
    result = orch.run()
    assert result["status"] == "completed", result
    weights = {k: v.detach().clone() for k, v in orch.trainer.merged.weights.items()}
    assert all(E.serves(e, weights) for e in engines)
    return cfg, cfg.run_dir / "ckpt" / "iter-000001", weights


def read_all(d: Path) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    out = {}
    for f in sorted(d.glob("*.safetensors")):
        out.update(load_file(str(f)))
    return out


def test_the_export_merged_is_what_the_engines_served(trained, tiny_base, tmp_path):
    cfg, ckpt, served = trained
    out = tmp_path / "adapter"
    assert X.main(["--ckpt", str(ckpt), "--out", str(out)]) == 0
    modes = {(out / n).stat().st_mode & 0o777 for n in (X.ADAPTER_WEIGHTS, X.ADAPTER_CONFIG)}
    assert len(modes) == 1  # the weights as readable as the config: the umask's, not owner-only
    manifest = json.loads((out / X.MANIFEST_NAME).read_text())
    assert manifest["pass"] and manifest["checkpoint"]["state"]["iteration"] == 1
    assert manifest["run_plan_base_dir"] == str(tiny_base)
    adapter_cfg = json.loads((out / X.ADAPTER_CONFIG).read_text())
    assert (adapter_cfg["r"], adapter_cfg["lora_alpha"], adapter_cfg["base_model_name_or_path"]) == (16, 32,
                                                                                                    str(tiny_base))
    merged = tmp_path / "merged"
    assert M.main(["merge", "--base", str(tiny_base), "--adapter", str(out), "--out", str(merged),
                   "--expect-scale", "2.0", "--threads", str(torch.get_num_threads())]) == 0
    got = read_all(merged)
    assert set(served) <= set(got)
    for k, w in served.items():
        assert torch.equal(got[k].view(torch.int16), w.to(got[k].dtype).view(torch.int16)), k


def test_peft_loads_the_export_with_the_checkpoints_tensors(trained, tiny_base, tmp_path):
    from peft import PeftModel

    _, ckpt, _ = trained
    out = tmp_path / "adapter"
    assert X.main(["--ckpt", str(ckpt), "--out", str(out), "--base-model", "tiny"]) == 0
    lora = torch.load(ckpt / "trainer" / "trainer.pt", map_location="cpu", weights_only=True)["lora"]
    model = PeftModel.from_pretrained(E.load_tiny(tiny_base).float(), str(out))
    params = dict(model.named_parameters())
    for name, t in lora.items():
        p = params[f"{name}.default.weight"]
        assert torch.equal(p.detach().float(), t.float()), name


def test_a_tampered_checkpoint_or_an_existing_output_is_refused(trained, tmp_path):
    import shutil

    _, ckpt, _ = trained
    existing = tmp_path / "exists"
    existing.mkdir()
    assert X.main(["--ckpt", str(ckpt), "--out", str(existing)]) == 2
    bad = tmp_path / "ckpt" / ckpt.name
    shutil.copytree(ckpt, bad)
    pt = bad / "trainer" / "trainer.pt"
    raw = bytearray(pt.read_bytes())
    raw[-100] ^= 0xFF
    pt.write_bytes(bytes(raw))
    assert X.main(["--ckpt", str(bad), "--out", str(tmp_path / "never")]) == 2
    assert not (tmp_path / "never").exists() and not (tmp_path / "never.partial").exists()
