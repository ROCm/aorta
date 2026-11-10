"""Stochastic rounding in the merge (`sync.stochastic_rounding`; design Section 6.5), with the trainer: the tensors the
trainer's forward uses are, by sha256, the ones the writer puts in the slot, with the switch on as with it off (C4.T3's
rule), and a restored latent re-merges to them. The seed, the cast and `merge_all` are in `test_rounding.py`.
"""

from __future__ import annotations

import hashlib

import pytest
import torch

from aorlta.sync import landing as L
from aorlta.sync import writer as W
from aorlta.sync.merge import bits, merge_weight
from aorlta.sync.reload import VersionBook
from aorlta.trainer.merged_lora import MergedLoRA

SEED = 20261102


# --- MergedLoRA, the writer and the landing check on the tiny model ---------------------------------------------


def _sha(t: torch.Tensor) -> str:
    return hashlib.sha256(W.tensor_bytes(t.detach().cpu())).hexdigest()


def _step(pm, merged, x, lr=1e-2):
    opt = torch.optim.Adam([p for p in pm.parameters() if p.requires_grad], lr=lr)
    pm(input_ids=x, labels=x).loss.backward()
    return merged.step(opt)


def _state(engines, probes, pm, tmp_path, keys, base_dir):
    prev = {e.engine_id: L.engine_logprobs(e, probes) for e in engines}
    cal = L.calibrate_tau(prev[0], L.trainer_logprobs(pm, probes), tau_floor=0.01)
    return L.SyncState(run_id="cpu", sync_dir=str(tmp_path / "shm"), base_dir=str(base_dir), probes=probes,
                       tau={e.engine_id: cal.tau for e in engines}, previous=prev, in_flight=lambda: 0,
                       expected=W.expected_count(keys, partial=True), book=VersionBook())


@pytest.mark.parametrize("rounding", ["nearest", "stochastic"])
def test_c4_t3_the_forward_runs_on_the_written_tensors_with_the_switch_off_and_on(
        rounding, tiny_base, tmp_path, tiny_probes, engine_factory, peft_factory):
    pm = peft_factory()
    keys = list(W.read_base_index(tiny_base))
    merged = MergedLoRA(pm, keys, rounding=rounding, run_seed=SEED).install()
    assert merged.version == 0 and all(torch.equal(bits(merged.weights[k]), bits(merged.base[k])) for k in merged.keys)
    engines = [engine_factory(0)]
    state = _state(engines, tiny_probes, pm, tmp_path, keys, tiny_base)
    targets = W.target_keys(keys)
    x = torch.randint(0, 320, (2, 40), generator=torch.Generator().manual_seed(11))
    for version in (1, 2):
        result = _step(pm, merged, x)
        assert result.stats.version == version and result.stats.rounding == rounding
        pub = L.publish(result.weights, version, result.stats.changed, engines=engines, state=state,
                        trainer_side=lambda p: L.trainer_logprobs(pm, p))
        assert pub.verdicts() == {0: "LANDED"}
        with merged.capture() as used, torch.no_grad():
            pm(input_ids=x)
        written = W.load_slot_tensors(pub.slot)
        assert set(used) == set(written) == targets and len(targets) == 31
        assert {k: _sha(t) for k, t in used.items()} == {k: _sha(t) for k, t in written.items()}
        # and the written tensors are this rounding's merge of the latent, for this version
        for key in targets:
            layer = merged.layers[key]
            want = merge_weight(layer.base_layer.weight, layer.lora_A["default"].weight, layer.lora_B["default"].weight,
                                float(layer.scaling["default"]), key, rounding=rounding, run_seed=SEED,
                                version=version)
            assert torch.equal(bits(written[key]), bits(want))
    nearest = {key: merge_weight(merged.layers[key].base_layer.weight, merged.layers[key].lora_A["default"].weight,
                                 merged.layers[key].lora_B["default"].weight, float(merged.layers[key].scaling["default"]))
               for key in merged.keys}
    differs = sum(not torch.equal(bits(written[k]), bits(nearest[k])) for k in merged.keys)
    assert differs == (0 if rounding == "nearest" else 31)


def test_a_restored_latent_re_merges_to_the_written_tensors(tiny_base, tmp_path, peft_factory):
    """Resume: a fresh model given the saved latent, merged at the saved version, serves the same bits."""
    keys = list(W.read_base_index(tiny_base))
    pm = peft_factory()
    merged = MergedLoRA(pm, keys, rounding="stochastic", run_seed=SEED).install()
    x = torch.randint(0, 320, (2, 40), generator=torch.Generator().manual_seed(12))
    _step(pm, merged, x)
    result = _step(pm, merged, x)
    info = W.write_slot(result.weights, base_dir=tiny_base, sync_dir=tmp_path / "shm", run_id="cpu", version=2)
    pm2 = peft_factory(seed=99)  # a different initial latent, then the saved one loaded over it
    lora = {n: p.detach().clone() for n, p in pm.named_parameters() if "lora_" in n}
    with torch.no_grad():
        for n, p in pm2.named_parameters():
            if n in lora:
                p.copy_(lora[n])
    m2 = MergedLoRA(pm2, keys, rounding="stochastic", run_seed=SEED)
    m2.version = 1
    m2.refresh()
    written = W.load_slot_tensors(info.path)
    assert m2.version == 2 and {k: _sha(t) for k, t in m2.weights.items()} == {k: _sha(t) for k, t in written.items()}
