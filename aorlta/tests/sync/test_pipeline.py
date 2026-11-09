"""C5 end to end on the CPU: trainer step -> merge -> write -> reload -> landing check, and the round trip.

The tiny model plays the trainer (PEFT + MergedLoRA) and `TinyEngine` plays TokenSpeed (it reads the slot's
files by name, skips unknown names silently, and answers prompt log-probs as a fresh prefill). These are the CPU
twins of C4.T3, C5.T1 and C5.T4; the GPU runs repeat them on Qwen3.5-4B and TokenSpeed.
"""

from __future__ import annotations

import hashlib
import json

import pytest
import torch

from aorlta.sync import landing as L
from aorlta.sync import writer as W
from aorlta.sync.reload import VersionBook, reload_all
from aorlta.sync.roundtrip import round_trip, synthetic_perturbation
from aorlta.trainer.merged_lora import MergedLoRA

TINY_TARGETS = 31  # the tiny model's LoRA targets (tests/sync/conftest.py)


def sha(t: torch.Tensor) -> str:
    return hashlib.sha256(W.tensor_bytes(t.detach().cpu())).hexdigest()


class Rig:
    def __init__(self, tiny_base, tmp_path, probes, engines, peft_factory):
        self.base_dir = tiny_base
        self.pm = peft_factory()
        self.keys = list(W.read_base_index(tiny_base))
        self.merged = MergedLoRA(self.pm, self.keys).install()
        self.opt = torch.optim.Adam([p for p in self.pm.parameters() if p.requires_grad], lr=1e-2)
        self.engines = engines
        self.probes = probes
        self.x = torch.randint(0, 320, (2, 40), generator=torch.Generator().manual_seed(11))
        prev = {e.engine_id: L.engine_logprobs(e, probes) for e in engines}
        cal = L.calibrate_tau(prev[0], self.trainer(probes), tau_floor=0.01)
        self.state = L.SyncState(run_id="cpu", sync_dir=str(tmp_path / "shm"), base_dir=str(tiny_base), probes=probes,
                                 tau={e.engine_id: cal.tau for e in engines}, previous=prev,
                                 in_flight=lambda: sum(e.busy for e in engines),
                                 expected=W.expected_count(self.keys, partial=True), book=VersionBook())

    def trainer(self, probes):
        return L.trainer_logprobs(self.pm, probes)

    def step(self, lr=1e-2):
        for g in self.opt.param_groups:
            g["lr"] = lr
        self.pm(input_ids=self.x, labels=self.x).loss.backward()
        result = self.merged.step(self.opt)
        self.opt.zero_grad(set_to_none=True)
        return result

    def publish(self, result):
        return L.publish(result.weights, result.stats.version, result.stats.changed, engines=self.engines,
                         state=self.state, trainer_side=self.trainer)


@pytest.fixture
def rig(tiny_base, tmp_path, tiny_probes, engine_factory, peft_factory):
    return Rig(tiny_base, tmp_path, tiny_probes, [engine_factory(0), engine_factory(1)], peft_factory)


def test_a_real_step_lands_and_the_trainer_ran_on_the_written_tensors(rig):
    r1 = rig.step()
    assert r1.stats.changed > 0
    pub = rig.publish(r1)
    assert pub.verdicts() == {0: "LANDED", 1: "LANDED"}
    assert all(e.reloads == [pub.slot] for e in rig.engines)
    assert rig.state.book.all_at(1)
    # C4.T3 in miniature: the W_eff each layer's forward used equals, by sha256, the tensor written to the slot.
    with rig.merged.capture() as used, torch.no_grad():
        rig.pm(input_ids=rig.x)
    written = W.load_slot_tensors(pub.slot)
    assert len(used) == TINY_TARGETS == len(written)
    assert {k: sha(t) for k, t in used.items()} == {k: sha(t) for k, t in written.items()}
    r2 = rig.step()
    pub2 = rig.publish(r2)
    assert pub2.verdicts() == {0: "LANDED", 1: "LANDED"}
    assert pub2.pruned == [pub.slot]  # the older slot goes only after the new one landed


def test_a_zero_update_gives_noop(rig):
    rig.publish(rig.step())
    r = rig.step(lr=0.0)
    assert r.stats.changed == 0 and r.stats.changed_vs_base > 0
    assert rig.publish(r).verdicts() == {0: "NOOP", 1: "NOOP"}


def test_an_update_below_bf16_resolution_gives_noop(rig):
    rig.publish(rig.step())
    r = rig.step(lr=1e-12)
    assert r.stats.changed == 0
    assert rig.publish(r).verdicts() == {0: "NOOP", 1: "NOOP"}


def test_skipping_the_reload_gives_not_landed(rig):
    r = rig.step()
    info = W.write_slot(r.weights, base_dir=rig.base_dir, sync_dir=rig.state.sync_dir, run_id="cpu",
                        version=r.stats.version)
    checked = L.landing_check(rig.engines, rig.probes, r.stats.changed, rig.state.previous,
                              rig.trainer(rig.probes), rig.state.tau)
    assert {e: res.verdict for e, (res, _) in checked.items()} == {0: "NOT_LANDED", 1: "NOT_LANDED"}
    assert info.path.is_dir()


def test_slightly_different_weights_give_mismatch(rig):
    r = rig.step()
    g = torch.Generator().manual_seed(5)
    wrong = {k: (t.float() * (1 + 0.3 * torch.randn(t.shape, generator=g))).to(torch.bfloat16)
             for k, t in r.weights.items()}
    with pytest.raises(L.LandingFailed) as exc:
        L.publish(wrong, r.stats.version, r.stats.changed, engines=rig.engines, state=rig.state,
                  trainer_side=rig.trainer)
    pub = exc.value.publication
    assert pub.verdicts() == {0: "MISMATCH", 1: "MISMATCH"}
    assert all(res.mean_gap > res.tau for res in pub.results.values())
    assert pub.pruned == [] and (rig.state.sync_dir and W.read_manifest(pub.slot)["version"] == 1)


def test_a_failed_check_keeps_every_slot(rig):
    pub1 = rig.publish(rig.step())
    r = rig.step()
    wrong = {k: torch.zeros_like(t) for k, t in r.weights.items()}
    with pytest.raises(L.LandingFailed) as exc:
        L.publish(wrong, r.stats.version, r.stats.changed, engines=rig.engines, state=rig.state,
                  trainer_side=rig.trainer)
    assert exc.value.publication.pruned == []
    assert W.read_manifest(pub1.slot)["version"] == 1
    assert W.read_manifest(exc.value.publication.slot)["version"] == 2


def test_a_misnamed_tensor_is_counted_as_read_but_never_lands(rig, tmp_path):
    """Why the name check exists (A6): TokenSpeed-like loaders skip an unknown name and still count it."""
    from safetensors.torch import save_file

    r = rig.step()
    slot = tmp_path / "hand" / "w1"
    slot.mkdir(parents=True)
    tensors = {k.replace("model.language_model.", "model."): t for k, t in r.weights.items()}  # text-only names
    save_file(tensors, str(slot / "model-00001-of-00001.safetensors"))
    (slot / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {k: "model-00001-of-00001.safetensors" for k in tensors}}))
    (slot / "manifest.json").write_text("{}")
    with pytest.raises(W.NameMapError):
        W.check_names(tensors, rig.keys)
    reload_all(rig.engines, slot, TINY_TARGETS, in_flight=lambda: 0)  # every engine reports 31 read
    assert all(len(e.skipped) == TINY_TARGETS for e in rig.engines)
    checked = L.landing_check(rig.engines, rig.probes, r.stats.changed, rig.state.previous,
                              rig.trainer(rig.probes), rig.state.tau)
    assert {e: res.verdict for e, (res, _) in checked.items()} == {0: "NOT_LANDED", 1: "NOT_LANDED"}


# --- the round trip (C5.T1 in miniature) -----------------------------------------------------------------


def _targets(tiny_base):
    from safetensors import safe_open

    with safe_open(str(tiny_base / "model.safetensors"), framework="pt") as f:
        return {k: f.get_tensor(k) for k in f.keys() if k in W.target_keys(f.keys())}


@pytest.mark.parametrize("partial", [True, False])
def test_round_trip_is_proven_full_and_partial(tiny_base, tmp_path, tiny_probes, engine_factory, partial):
    engines = [engine_factory(0), engine_factory(1)]
    base = _targets(tiny_base)
    keys = list(W.read_base_index(tiny_base))
    others = None if partial else W.load_base_tensors(tiny_base, [k for k in keys if k not in base])
    pert = W.write_slot(synthetic_perturbation(base), base_dir=tiny_base, sync_dir=tmp_path, run_id="rt",
                        version=1, partial=partial, others=others)
    rest = W.write_slot(base, base_dir=tiny_base, sync_dir=tmp_path, run_id="rt", version=2, partial=partial,
                        others=others)
    res = round_trip(engines, tiny_probes, perturbed_slot=pert.path, restore_slot=rest.path,
                     expected=W.expected_count(keys, partial=partial), in_flight=lambda: 0, reps=2, greedy_tokens=4)
    assert res.verdict == "PROVEN", res.as_dict()
    assert res.draws_per_phase == 2 * 2 * 3 == res.perturbed_draws_differing == res.restore_draws_matching


def test_round_trip_not_proven_when_the_restore_is_not_the_baseline(tiny_base, tmp_path, tiny_probes, engine_factory):
    engines = [engine_factory(0)]
    base = _targets(tiny_base)
    keys = list(W.read_base_index(tiny_base))
    pert = W.write_slot(synthetic_perturbation(base), base_dir=tiny_base, sync_dir=tmp_path, run_id="rt", version=1)
    off = {k: (t.float() * 1.01).to(torch.bfloat16) for k, t in base.items()}
    rest = W.write_slot(off, base_dir=tiny_base, sync_dir=tmp_path, run_id="rt", version=2)
    res = round_trip(engines, tiny_probes, perturbed_slot=pert.path, restore_slot=rest.path,
                     expected=W.expected_count(keys, partial=True), in_flight=lambda: 0, reps=2, greedy_tokens=4)
    assert res.verdict == "NOT_PROVEN" and res.restore_draws_matching < res.draws_per_phase


def test_round_trip_reload_rejected(tiny_base, tmp_path, tiny_probes, engine_factory):
    e = engine_factory(0)
    base = _targets(tiny_base)
    keys = list(W.read_base_index(tiny_base))
    pert = W.write_slot(synthetic_perturbation(base), base_dir=tiny_base, sync_dir=tmp_path, run_id="rt", version=1)
    e.fail_next = [(False, "refused", 0), (False, "refused", 0)]
    res = round_trip([e], tiny_probes, perturbed_slot=pert.path, restore_slot=pert.path,
                     expected=W.expected_count(keys, partial=True), in_flight=lambda: 0, reps=2, greedy_tokens=2)
    assert res.verdict == "RELOAD_REJECTED"
