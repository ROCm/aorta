"""Score mode on the wired components, on CPU (`wiring.build` with `orchestrator.mode: score`).

As in test_wiring_e2e.py, everything is the integrated code except three stand-ins: a tiny Qwen3.5 with the real
vocabulary, `TwinEngine`s that load and reload it as TokenSpeed does, and the fake checks. Here the twins load their
model when they start, from the directory the test names, so a test can have them serve something other than what the
config says. The real tools/merge_base_adapter.py merges a tiny PEFT adapter."""

import json
import shutil
import sys
import threading
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_wiring_e2e as E  # noqa: E402 - the tiny model, its twin engines, the probes and the dataset

from aorlta.config import ConfigError, config_from_dict  # noqa: E402
from aorlta.ids import sha256_file  # noqa: E402
from aorlta.orchestrator import wiring  # noqa: E402
from aorlta.orchestrator.ports import RunStopped  # noqa: E402
from aorlta.rollout.compile_check import FakeCompiler  # noqa: E402
from aorlta.rollout.engine_proc import InProcessEngine  # noqa: E402
from aorlta.sync.merge import is_target_key  # noqa: E402
from aorlta.sync.writer import load_base_tensors, read_base_index, target_keys  # noqa: E402

tiny_base, dataset = E.tiny_base, E.dataset  # module-scoped fixtures
TARGETS = ["in_proj_qkv", "out_proj", "in_proj_z", "down_proj", "o_proj", "in_proj_a", "v_proj", "in_proj_b", "q_proj",
           "gate_proj", "up_proj", "k_proj"]
LM_PREFIX = "base_model.model.model.language_model.layers."


class LazyTwin(E.TwinEngine):
    """A TwinEngine that loads its model when it starts, from `load_from()`: a merge does not exist before the score
    run's setup. The pool starts its engines in parallel, and from_pretrained sets torch's global default dtype while
    it loads, so a load beside another engine's forward pass can change that pass's dtype: the first twin to start
    loads every twin's model, one after another, while the others wait."""

    def __init__(self, load_from, fake, engine_id: int, family: list):
        self.fake = E.FakeEngine(fake, engine_id=engine_id, **E.ENGINE_KW)
        self.tokenizer_manager = self.fake.tokenizer_manager
        self.load_from, self.model, self.params, self.reloads = load_from, None, {}, []
        self.started_on: list[Path] = []
        self.family = family  # every twin of the pool, then the lock they load under
        self._lock = threading.Lock()

    def restart(self):
        *twins, lock = self.family
        with lock:
            for twin in twins:
                if twin.model is None:
                    path = Path(twin.load_from())
                    twin.model = E.load_tiny(path)
                    twin.params = dict(twin.model.named_parameters())
                    twin.started_on.append(path)
        self.fake.restart()


def score_config(tmp_path, base, ds, *, run_id="scored", adapter=None, prompts=4, scoring_s=None):
    model = {"base_dir": str(base), "source": "standin", **({"adapter_dir": str(adapter)} if adapter else {})}
    return config_from_dict({
        "run": {"run_id": run_id, "source": "standin", "root": str(tmp_path / "runs")},
        "model": model,
        "data": {"prompt_pools": [{"path": str(ds / "listings"), "source": "standin"}]},
        "rollout": {"engines": [0, 1], "K": 4, "P": 4, "max_new_tokens": 512, "request_timeout_s": 300,
                    "compile_adapter": "fake.compile"},
        "reward": {"profile": "fake", "reward_fn": "standin.v11"},
        "orchestrator": {"mode": "score", "run_seed": 7, "score": {"prompts": prompts, "passes": 1},
                         **({"deadlines": {"scoring_s": scoring_s}} if scoring_s else {})},
        "sync": {"dir": str(tmp_path / "shm")},
    })


def build(cfg, load_from, reward_factories=None):
    family: list = []
    engines = [LazyTwin(load_from, E.FAKE, i, family) for i in range(len(cfg.rollout.engines))]
    family += engines + [threading.Lock()]
    parts = wiring.Parts(tokenizer=E.ByteTokenizer(), handles=[InProcessEngine(i, e) for i, e in enumerate(engines)],
                         compiler=FakeCompiler(), reward_factories=reward_factories, probes=E.PROBES)
    return wiring.build(cfg, parts=parts), engines


def targets_of(model_dir: Path) -> dict[str, torch.Tensor]:
    return load_base_tensors(model_dir, sorted(target_keys(read_base_index(model_dir))))


def read_json(path: Path):
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def tiny_adapter(tiny_base, tmp_path_factory) -> Path:
    """An r4/alpha8 PEFT adapter on the tiny base (alpha/r = 2)."""
    from peft import LoraConfig, get_peft_model
    from safetensors.torch import load_file
    from transformers import Qwen3_5ForConditionalGeneration

    model = Qwen3_5ForConditionalGeneration.from_pretrained(str(tiny_base), dtype=torch.float32)
    pm = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=TARGETS, bias="none"))
    g = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for n, p in pm.named_parameters():
            if "lora_A" in n:
                p.copy_((torch.rand(p.shape, generator=g) * 2 - 1) / p.shape[1] ** 0.5)
            elif "lora_B" in n:
                p.copy_(torch.randn(p.shape, generator=g) * 0.05)
    out = tmp_path_factory.mktemp("tiny-adapter") / "adapter"
    pm.save_pretrained(str(out))
    assert all(k.startswith(LM_PREFIX) for k in load_file(str(out / "adapter_model.safetensors")))
    return out


@pytest.fixture(scope="module")
def other_base(tiny_base, tmp_path_factory) -> Path:
    """The tiny base with every LoRA target scaled by 1.5: another model in the same layout."""
    from safetensors.torch import load_file, save_file

    d = tmp_path_factory.mktemp("tiny-other")
    for p in tiny_base.iterdir():
        if p.suffix == ".safetensors":
            t = load_file(str(p))
            save_file({k: (v.float() * 1.5).to(v.dtype) if is_target_key(k) else v for k, v in t.items()},
                      str(d / p.name), metadata={"format": "pt"})
        elif p.is_file():
            shutil.copy(p, d / p.name)
    return d


@pytest.fixture(scope="module")
def other_norm(tiny_base, tmp_path_factory) -> tuple[Path, str]:
    """The tiny base with one tensor that is not a LoRA target changed: the same targets, another model. Returns the
    directory and the changed key."""
    from safetensors.torch import load_file, save_file

    keys = list(read_base_index(tiny_base))
    key = sorted(k for k in keys if k not in target_keys(keys) and "visual" not in k and "norm" in k)[0]
    d = tmp_path_factory.mktemp("tiny-other-norm")
    for p in tiny_base.iterdir():
        if p.suffix == ".safetensors":
            t = load_file(str(p))
            save_file({k: (v.float() + 0.25).to(v.dtype) if k == key else v for k, v in t.items()}, str(d / p.name),
                      metadata={"format": "pt"})
        elif p.is_file():
            shutil.copy(p, d / p.name)
    return d, key


def test_a_wired_score_run_serves_the_model_directory_confirms_it_and_trains_nothing(tmp_path, tiny_base, dataset,
                                                                                     monkeypatch):
    monkeypatch.setattr(wiring, "TrainerAdapter", lambda *a, **k: pytest.fail("a score run built a trainer"))
    monkeypatch.setattr(wiring, "SyncAdapter", lambda *a, **k: pytest.fail("a score run built the weight sync"))
    cfg = score_config(tmp_path, tiny_base, dataset)
    scorer, engines = build(cfg, lambda: wiring.served_model_dir(cfg))
    assert type(scorer).__name__ == "Scorer" and wiring.served_model_dir(cfg) == tiny_base
    result = scorer.run()
    assert result["status"] == "completed", result
    assert all(e.started_on == [tiny_base] for e in engines)
    assert all([Path(p).name for p in e.reloads] == ["w1", "w0"] for e in engines)  # the round trip; no publication
    base = targets_of(tiny_base)
    assert all(E.serves(e, base) for e in engines)
    model = read_json(cfg.run_dir / "score" / "model.json")
    landing = model["landing"]
    assert landing["verdict"] == "PROVEN" and landing["round_trip"]["verdict"] == "PROVEN"
    assert model["model"]["kind"] == "merged_dir" and model["model"]["files"] == wiring.top_files(tiny_base)
    assert landing["files_sha256"] == model["model"]["files_sha256"]
    assert landing["slot_tensors"] == landing["tensors_read"] == len(read_base_index(tiny_base)) > len(base)
    assert landing["partial"] is False
    plan = read_json(cfg.run_dir / "run_plan.json")
    assert {k.partition("/")[2]: v["sha256"] for k, v in plan["extra_files"].items()} == model["model"]["files"]
    rows = [json.loads(x) for x in (cfg.run_dir / "score" / "report.jsonl").read_text().splitlines()]
    assert rows[-1]["samples"] == 16 and len(rows) == 1 + 4 + 1
    assert rows[0]["landing"]["verdict"] == "PROVEN" and "round_trip" not in rows[0]["landing"]
    assert not wiring.node_dir(cfg).exists()  # the slots are removed at shutdown


def test_score_json_names_the_checks_still_running_at_the_scoring_deadline_and_at_the_retrys(tmp_path, tiny_base,
                                                                                            dataset):
    from aorlta.ids import candidate_id
    from aorlta.reward.adapters.fake import FakeAdapterConfig, FakeCheckAdapter, FakeLegProfile, FakePlan
    from aorlta.reward.clock import Clock

    class Stalling(FakeCheckAdapter):  # the parse check of each sample k = 0 runs until the service closes it
        def plan(self, job):
            if job.leg == "gate.parse" and (job.candidate_id or "").endswith("/0"):
                return FakePlan("unsettled", "worker_stalled", (), self.config.stall_s, True, "override")
            return super().plan(job)

    cfg = score_config(tmp_path, tiny_base, dataset, run_id="late", scoring_s=3.0)
    checker = FakeAdapterConfig(seed=cfg.orchestrator.run_seed, default=FakeLegProfile(fail_rate=wiring.FAKE_FAIL_RATE),
                                time_scale=1.0)
    factories = {**wiring.fake_reward_factories(cfg, Clock()), "fake": lambda w: Stalling(checker)}
    scorer, _ = build(cfg, lambda: tiny_base, reward_factories=factories)
    result = scorer.run()
    assert result["status"] == "completed", result
    batch = read_json(cfg.run_dir / "iterations" / "iter-000001" / "score.json")
    late = sorted(candidate_id("late", 1, p, 0) for p in batch["prompts"])
    assert batch["retry"]["candidates"] >= len(late)
    out = batch["deadlines"]["outstanding_at_deadline"]
    assert sorted(o["candidate_id"] for o in out if o["leg"] == "gate.parse" and o["candidate_id"] in late) == sorted(
        late * 2)  # once at the scoring deadline, once at the retry's


def test_a_wired_score_run_merges_an_adapter_with_the_tool_and_serves_the_merge(tmp_path, tiny_base, dataset,
                                                                                tiny_adapter, monkeypatch):
    cfg = score_config(tmp_path, tiny_base, dataset, run_id="adapted", adapter=tiny_adapter)
    served = wiring.served_model_dir(cfg)
    assert served == wiring.node_dir(cfg) / "model" and not served.exists()
    merged, seen = {}, {}
    real = wiring.ServedModel.confirm

    def confirm(self):  # the merge lives in the node directory, which shutdown removes: read it while it is there
        merged.update(targets_of(self.dir))
        seen["manifest"] = sha256_file(self.dir / wiring.MERGE_MANIFEST)
        return real(self)

    monkeypatch.setattr(wiring.ServedModel, "confirm", confirm)
    scorer, engines = build(cfg, lambda: served)
    result = scorer.run()
    assert result["status"] == "completed", result
    base = targets_of(tiny_base)
    assert merged.keys() == base.keys() and not any(torch.equal(merged[k], base[k]) for k in base)
    assert all(e.started_on == [served] and E.serves(e, merged) for e in engines)
    model = read_json(cfg.run_dir / "score" / "model.json")["model"]
    assert (model["kind"], model["adapter_dir"], model["dir"]) == ("base_plus_adapter", str(tiny_adapter), str(served))
    m = model["merge"]
    assert (m["rank"], m["lora_alpha"], m["targets"], m["targets_changed"]) == (4, 8.0, len(base), len(base))
    assert m["manifest_sha256"] == seen["manifest"]
    plan = read_json(cfg.run_dir / "run_plan.json")["extra_files"]
    assert {"adapter/adapter_model.safetensors", "adapter/adapter_config.json"} <= set(plan)
    # shutdown removes the merge with the node directory (here the twins still map its shard, so on NFS the emptied
    # directory itself can outlive them; a job's node directory is in /dev/shm)
    assert not (served / "model.safetensors").exists() and not (served / wiring.MERGE_MANIFEST).exists()


def test_engines_that_loaded_another_model_stop_the_score_run_before_anything_is_scored(tmp_path, tiny_base, dataset,
                                                                                        other_base):
    cfg = score_config(tmp_path, tiny_base, dataset, run_id="other")
    scorer, engines = build(cfg, lambda: other_base)
    result = scorer.run()
    assert result["status"] == "stopped" and result["stop"]["reason"] == "served_model_unconfirmed", result
    rt = result["stop"]["detail"]["landing"]["round_trip"]
    assert rt["verdict"] == "NOT_PROVEN" and rt["baseline_self_consistent"]
    assert rt["restore_draws_matching"] < rt["draws_per_phase"]  # the model's tensors are not what was loaded
    assert not (cfg.run_dir / "iterations").exists() and result["iterations_completed"] == 0


def test_engines_that_loaded_a_model_with_the_same_lora_targets_and_another_tensor_stop_the_score_run(
        tmp_path, tiny_base, dataset, other_norm):
    other, key = other_norm
    cfg = score_config(tmp_path, tiny_base, dataset, run_id="other-norm")
    scorer, engines = build(cfg, lambda: other)
    result = scorer.run()
    assert result["status"] == "stopped" and result["stop"]["reason"] == "served_model_unconfirmed", result
    rt = result["stop"]["detail"]["landing"]["round_trip"]
    assert rt["verdict"] == "NOT_PROVEN" and rt["baseline_self_consistent"]
    assert not (cfg.run_dir / "iterations").exists()
    right = load_base_tensors(tiny_base, [key])
    assert all(E.serves(e, right) for e in engines)  # whatever they loaded, the round trip leaves the model's tensors


def test_a_model_file_that_changes_after_the_run_plan_stops_the_score_run(tmp_path, tiny_base, dataset, monkeypatch):
    base = tmp_path / "base-copy"
    shutil.copytree(tiny_base, base)
    cfg = score_config(tmp_path, base, dataset, run_id="changed")
    real = wiring.ServedModel.prepare

    def prepare(self, recorded):
        out = real(self, recorded)
        with open(base / "config.json", "a") as f:
            f.write("\n")
        return out

    monkeypatch.setattr(wiring.ServedModel, "prepare", prepare)
    scorer, engines = build(cfg, lambda: base)
    result = scorer.run()
    assert result["stop"]["reason"] == "served_model_changed", result
    assert result["stop"]["detail"]["files"] == ["config.json"] and all(e.reloads == [] for e in engines)


def test_a_model_whose_index_names_a_shard_below_its_top_level_is_refused_before_the_engines_start(tmp_path, tiny_base,
                                                                                                    dataset):
    base = tmp_path / "nested"
    shutil.copytree(tiny_base, base)
    (base / "weights").mkdir()
    for shard in base.glob("*.safetensors"):
        shard.rename(base / "weights" / shard.name)
    index = base / "model.safetensors.index.json"
    data = read_json(index)
    data["weight_map"] = {k: f"weights/{v}" for k, v in data["weight_map"].items()}
    index.write_text(json.dumps(data))
    cfg = score_config(tmp_path, base, dataset, run_id="nested")
    scorer, engines = build(cfg, lambda: base)
    result = scorer.run()
    assert result["status"] == "stopped" and result["stop"]["reason"] == "model_shards_not_top_level", result
    assert result["stop"]["detail"]["shards"] == ["weights/model.safetensors"]
    assert all(e.started_on == [] for e in engines) and not (cfg.run_dir / "score").exists()


def test_a_second_score_run_under_the_same_run_id_is_refused_and_removes_nothing_of_the_first(tmp_path, tiny_base,
                                                                                              dataset):
    cfg = score_config(tmp_path, tiny_base, dataset, run_id="twice", prompts=5)
    node = wiring.node_dir(cfg)
    first, _ = build(cfg, lambda: tiny_base)
    seen = {}

    def second(name, n):  # between the first run's two batches, while its engines hold slot w0
        if name == "record.committed" and n == 1:
            again, engines = build(cfg, lambda: tiny_base)
            seen["before"] = sorted(p.relative_to(node).as_posix() for p in node.rglob("*"))
            with pytest.raises(ConfigError, match="is already set"):
                again.run()
            seen["after"] = sorted(p.relative_to(node).as_posix() for p in node.rglob("*")) if node.exists() else []
            seen["started"] = [e.started_on for e in engines]

    first._point = second
    result = first.run()
    assert result["status"] == "completed" and result["iterations_completed"] == 2, result
    assert seen["started"] == [[], []] and "w0/manifest.json" in seen["before"]
    assert seen["after"] == seen["before"]


def test_a_merge_that_fails_or_merged_other_inputs_is_refused(tmp_path, tiny_base, dataset, tiny_adapter):
    def served(run_id, tool=wiring.MERGE_TOOL):
        cfg = score_config(tmp_path, tiny_base, dataset, run_id=run_id, adapter=tiny_adapter)
        return wiring.ServedModel(cfg, None, None, spans=None, dataset=None, probes=E.PROBES, merge_tool=tool)

    recorded = {f"base/{n}": h for n, h in wiring.top_files(tiny_base).items()}
    recorded.update({f"adapter/{n}": sha256_file(tiny_adapter / n)
                     for n in ("adapter_model.safetensors", "adapter_config.json")})
    refusing = tmp_path / "refusing_tool.py"
    refusing.write_text("import sys\nprint('REFUSED: not a LoRA', file=sys.stderr)\nsys.exit(2)\n")
    with pytest.raises(RunStopped) as stop:
        served("refused", refusing).prepare(recorded)
    assert stop.value.reason == "model_merge_failed" and stop.value.detail["rc"] == 2
    assert "REFUSED: not a LoRA" in stop.value.detail["stderr"]
    for run_id, key, problem in (("other-adapter", "adapter/adapter_model.safetensors", "the merged adapter is not"),
                                 ("other-base", "base/config.json", "the merged base is not")):
        with pytest.raises(RunStopped) as stop:
            served(run_id).prepare(dict(recorded, **{key: "0" * 64}))
        assert stop.value.reason == "model_merge_unverified" and stop.value.detail["problems"] == [
            f"{problem} the {'adapter' if 'adapter' in key else 'base'} the run plan hashed"]
    identity = served("fine").prepare(recorded)
    assert identity["kind"] == "base_plus_adapter" and identity["merge"]["targets_changed"] > 0


def test_a_weight_slot_whose_files_are_not_its_manifest_stops_the_score_run(tmp_path, tiny_base, dataset,
                                                                             monkeypatch):
    from aorlta.sync import writer

    cfg = score_config(tmp_path, tiny_base, dataset, run_id="slot")
    model = wiring.ServedModel(cfg, None, None, spans=None, dataset=None, probes=E.PROBES)
    model.prepare({f"base/{n}": h for n, h in wiring.top_files(tiny_base).items()})
    monkeypatch.setattr(writer, "verify_slot", lambda slot: {"ok": False, "tensors": 1, "problems": ["k: differs"]})
    with pytest.raises(RunStopped) as stop:
        model.confirm()  # before any engine is asked anything
    assert stop.value.reason == "served_slot_mismatch" and stop.value.detail["problems"] == ["k: differs"]
