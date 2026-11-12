"""The integration wiring's own pieces (aorlta.orchestrator.wiring), each against the real component it adapts.
The whole wiring, end to end on CPU, is in test_wiring_e2e.py."""

import dataclasses
import json
import threading
import types
from pathlib import Path

import pytest

from aorlta.config import config_from_dict, load_config, parse_yaml
from aorlta.orchestrator import wiring
from aorlta.rollout.engine_proc import InProcessEngine, reply_value
from aorlta.rollout.templates import TemplateError, standin_training_kernels

REPO = Path(__file__).resolve().parents[2]


def _cfg(tmp_path, **sections):
    base = {
        "run": {"run_id": "w", "source": "standin", "root": str(tmp_path / "runs")},
        "model": {"base_dir": str(tmp_path / "base"), "source": "standin"},
        "data": {"prompt_pools": [{"path": str(tmp_path / "ds" / "listings"), "source": "standin"}]},
        "rollout": {"engines": [0, 1], "K": 4, "P": 4, "compile_adapter": "fake.compile"},
        "reward": {"profile": "fake", "reward_fn": "standin.v11"},
        "orchestrator": {"iterations": 1, "run_seed": 7},
        "trainer": {"ranks": 1},
        "sync": {"dir": str(tmp_path / "shm")},
    }
    for name, value in sections.items():
        base[name] = {**base.get(name, {}), **value}
    return config_from_dict(base)


# --- configs ----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name, ranks", [("dev-standin.yaml", 1), ("standin.yaml", 4)])
def test_the_stand_in_configs_are_ones_the_wiring_runs(name, ranks):
    cfg = load_config(REPO / "configs" / name)
    wiring.check_supported(cfg)
    assert cfg.trainer.ranks == ranks and cfg.trainer.micro_batch_tokens == 20480


def test_the_stand_in_run_enforces_t10s_gap_gate_with_a_warmup():
    """configs/standin.yaml enforces T10's gate (per-sample mean |delta|, 0.05 nats, 2%) and takes a one-step training
    warm-up at start-up; its checker sockets are kept apart from another run's by its run id."""
    cfg = load_config(REPO / "configs" / "standin.yaml")
    t = cfg.trainer
    assert (t.gap_gate, t.gap_statistic, t.max_generation_gap, t.max_off_policy_fraction) == (
        "enforce", "mean_abs", 0.05, 0.02)
    assert cfg.run.source == "standin" and t.startup_warmup is True
    assert cfg.reward.remote.socket_dir.endswith(f"/{cfg.run.run_id}")


def test_the_stand_in_run_puts_a_rank_beside_each_engine_and_the_checkers_on_the_spare_gpus():
    """configs/standin.yaml: four colocated ranks, one beside each engine on GPUs 0-3 (D10), ConSan on GPUs 4 and 6,
    ASan on 5 and 7, and each ConSan run of one kernel limited to 30 s. D10's fallback layout, built from the same
    config, puts two engines on GPUs 0-1 and two lone ranks on GPUs 2-3, away from the checkers."""
    cfg = load_config(REPO / "configs" / "standin.yaml")
    wiring.check_supported(cfg)
    assert cfg.trainer.ranks == 4 and cfg.trainer.devices is None
    assert wiring.trainer_gpus(cfg) == cfg.rollout.engines == (0, 1, 2, 3)
    assert wiring.TrainerAdapter(cfg, spans=None).devices == ("cuda:0", "cuda:1", "cuda:2", "cuda:3")
    assert [wiring.trainer_memory_fraction(cfg, g) for g in range(4)] == [0.55] * 4
    remote = cfg.reward.remote
    assert remote.gpus("standin.aorta_consan") == (4, 6) and remote.gpus("standin.aorta_asan") == (5, 7)
    assert cfg.reward.gpu_check_s == {"san.consan": 30.0}
    data = parse_yaml((REPO / "configs" / "standin.yaml").read_text())
    data["rollout"]["engines"] = [0, 1]
    data["trainer"].update(ranks=2, devices=[2, 3])
    fallback = config_from_dict(data)
    wiring.check_supported(fallback)
    assert fallback.rollout.engines == (0, 1) and wiring.trainer_gpus(fallback) == (2, 3)
    assert [wiring.trainer_memory_fraction(fallback, g) for g in (2, 3)] == [None, None]  # no engine beside them


def test_a_trainer_layout_the_wiring_cannot_place_is_refused(tmp_path):
    with pytest.raises(wiring.WiringError, match="trainer.ranks is 3: colocated, each rank sits beside one engine "
                                                 "and there are 2"):
        wiring.check_supported(_cfg(tmp_path, trainer={"ranks": 3}))
    wiring.check_supported(_cfg(tmp_path, trainer={"ranks": 2}))  # rank r beside engine r
    wiring.check_supported(_cfg(tmp_path, trainer={"ranks": 3, "devices": [2, 3, 6]}))  # a separate layout
    remote = {"socket_dir": "/dev/shm/x", "adapters": {"fake-asan": 5}}
    with pytest.raises(wiring.WiringError, match=r"trainer GPUs \[5\] serve the remote checkers"):
        wiring.check_supported(_cfg(tmp_path, trainer={"ranks": 2, "devices": [4, 5]}, reward={"remote": remote}))
    remote = {"socket_dir": "/dev/shm/x", "adapters": {"fake-asan": [5, 7]}}  # every GPU of an adapter counts
    with pytest.raises(wiring.WiringError, match=r"trainer GPUs \[7\] serve the remote checkers"):
        wiring.check_supported(_cfg(tmp_path, trainer={"ranks": 2, "devices": [6, 7]}, reward={"remote": remote}))
    wiring.check_supported(_cfg(tmp_path, trainer={"ranks": 2, "devices": [4, 6]}, reward={"remote": remote}))
    with pytest.raises(wiring.WiringError, match="one rank"):
        wiring.TrainerAdapter(_cfg(tmp_path, trainer={"ranks": 2}), spans=None, device="cuda:1")
    assert wiring.TrainerAdapter(_cfg(tmp_path, trainer={"ranks": 2}), spans=None, device="cpu").devices == ("cpu",
                                                                                                           "cpu")


def test_the_stand_in_run_is_the_development_run_at_256_samples_per_iteration_with_the_gpu_checks():
    cfg = load_config(REPO / "configs" / "standin.yaml")
    dev = load_config(REPO / "configs" / "dev-standin.yaml")
    assert (cfg.rollout.P, cfg.rollout.K, cfg.orchestrator.prompts_per_iteration) == (16, 16, 16)
    assert cfg.rollout.P * cfg.rollout.K == 256
    assert (dev.rollout.P, dev.rollout.K) == (32, 8)  # the development shape stays
    # the model, the data, the selector, the sync and every other rollout setting are the development run's; the
    # checks differ: the stand-in run uses standin-gpu, with its GPU legs served from their own process
    for section in ("model", "data", "sync", "selector"):
        assert getattr(cfg, section) == getattr(dev, section), section
    assert (dev.reward.profile, cfg.reward.profile) == ("standin-static", "standin-gpu") and dev.reward.remote is None
    assert dev.reward.reward_fn == cfg.reward.reward_fn == "standin.v11"
    assert dataclasses.replace(cfg.reward, profile=dev.reward.profile, remote=None, gpu_check_s={}) == dev.reward
    same = {k: v for k, v in dataclasses.asdict(cfg.rollout).items() if k not in ("P", "K")}
    assert same == {k: v for k, v in dataclasses.asdict(dev.rollout).items() if k not in ("P", "K")}
    assert cfg.run.run_id != dev.run.run_id


def test_every_unsupported_setting_is_named_at_once(tmp_path):
    cfg = _cfg(tmp_path, trainer={"ranks": 4, "steps_per_iteration": 2},
               orchestrator={"eval_every": 1}, selector={"endpoint": "http://planner:1"})
    with pytest.raises(wiring.WiringError) as e:
        wiring.check_supported(cfg)
    msg = str(e.value)
    for part in ("trainer.ranks is 4", "steps_per_iteration is 2", "eval_every is set", "selector.endpoint is set"):
        assert part in msg
    wiring.check_supported(_cfg(tmp_path))  # and the base config passes


def test_fewer_prompts_than_the_engine_self_test_needs_are_refused(tmp_path):
    with pytest.raises(wiring.WiringError, match="rollout.P is 3: .* needs 4"):
        wiring.check_supported(_cfg(tmp_path, rollout={"P": 3}))
    wiring.check_supported(_cfg(tmp_path, rollout={"P": 4}))


def test_streaming_off_is_refused(tmp_path):
    with pytest.raises(wiring.WiringError, match="streaming_accumulation is false"):
        wiring.check_supported(_cfg(tmp_path, trainer={"streaming_accumulation": False}))


def test_the_rounding_switch_reaches_the_merge(tmp_path):
    assert wiring.merged_lora_kwargs(_cfg(tmp_path)) == {"rounding": "nearest", "run_seed": 7}
    on = _cfg(tmp_path, sync={"stochastic_rounding": True})
    assert wiring.merged_lora_kwargs(on) == {"rounding": "stochastic", "run_seed": 7}


# --- the stand-in prompts -------------------------------------------------------------------------------------


def write_dataset(root: Path, kernels: dict) -> Path:
    (root / "listings").mkdir(parents=True)
    for kid in kernels:
        (root / "listings" / f"{kid}.json").write_text(json.dumps([[0, "s_load_dword s0, s[0:1], 0x0"],
                                                                   [8, "s_endpgm"]]))
    full = {kid: dict(kid=kid, demangled=f"kernel_{kid}", family="gemm", library="rocblas", **v)
            for kid, v in kernels.items()}
    (root / "kernels.json").write_text(json.dumps(full))
    return root


def test_the_dataset_is_found_from_the_configured_listings_path(tmp_path):
    ds = write_dataset(tmp_path / "ds", {"k1": {"n_insns": 100, "heldout": False}})
    assert wiring.standin_dataset_dir(ds / "listings") == ds
    assert wiring.standin_dataset_dir(ds) == ds
    with pytest.raises(wiring.WiringError, match="kernels.json"):
        wiring.standin_dataset_dir(tmp_path)


def test_training_prompts_leave_out_held_out_kernels(tmp_path):
    ds = write_dataset(tmp_path / "ds", {
        "k3": {"n_insns": 100, "heldout": False}, "k1": {"n_insns": 60, "heldout": False},
        "k2": {"n_insns": 400, "heldout": True}, "k4": {"n_insns": 59, "heldout": False},
        "k5": {"n_insns": 401, "heldout": False}, "k6": {"n_insns": 200},
    })
    assert [k["kid"] for k in standin_training_kernels(ds)] == ["k1", "k3"]
    assert all(k["listing"].startswith("0x0000: s_load_dword") for k in standin_training_kernels(ds))
    only_heldout = write_dataset(tmp_path / "ds2", {"k1": {"n_insns": 100, "heldout": True}})
    with pytest.raises(TemplateError, match="held-out"):
        standin_training_kernels(only_heldout)


def test_the_real_stand_in_pool_has_no_held_out_kernel():
    from aorlta.rollout.templates import STANDIN_DATASET as ds
    if not (ds / "kernels.json").is_file():
        pytest.skip("the stand-in dataset is not on this host (AORLTA_STANDIN_DATASET)")
    every = json.loads((ds / "kernels.json").read_text())
    pool = standin_training_kernels(ds)
    assert pool and all(k["heldout"] is False and 60 <= k["n_insns"] <= 400 for k in pool)
    held = {k for k, v in every.items() if v.get("heldout") and 60 <= v["n_insns"] <= 400}
    assert held and not held & {k["kid"] for k in pool}


# --- C1 pieces --------------------------------------------------------------------------------------------------


class BlockingHandle:
    def __init__(self):
        self.go = threading.Event()
        self.entered = threading.Event()
        self.ready = {"x": 1}

    def generate(self, ids, sp, **kw):
        self.entered.set()
        self.go.wait(10)
        if sp.get("fail"):
            raise RuntimeError("boom")
        return {"output_ids": [1]}


def test_a_counted_handle_counts_requests_in_flight_and_passes_the_rest_through():
    inner = BlockingHandle()
    h = wiring.CountedHandle(inner)
    assert h.ready == {"x": 1} and h.busy == 0
    t = threading.Thread(target=h.generate, args=([1], {}))
    t.start()
    assert inner.entered.wait(10) and h.busy == 1
    inner.go.set()
    t.join(10)
    assert h.busy == 0
    with pytest.raises(RuntimeError):
        h.generate([1], {"fail": True})
    assert h.busy == 0  # a failed request is no longer in flight


class ReloadingEngine:
    def __init__(self, reply):
        self.reply = reply
        self.tokenizer_manager = types.SimpleNamespace(server_args=types.SimpleNamespace(x=1))

    def update_weights_from_disk(self, path, flush_cache=True):
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def test_an_in_process_engine_reloads_and_reports_the_reply_as_values():
    msg = "Succeeded (31 checkpoint tensors read) " + "x" * 600
    e = InProcessEngine(0, ReloadingEngine((True, msg, 0)))
    e.start()
    res = e.reload("/slot/w1")
    assert res["ok"] is True and res["value"] == [True, msg, 0] and len(res["reply"]) == 500
    assert InProcessEngine(0, ReloadingEngine((False, "no", 0))).reload("/s")["error"].startswith("engine 0 is dead")
    bad = InProcessEngine(1, ReloadingEngine(OSError("disk")))
    bad.start()
    assert "OSError: disk" in bad.reload("/s")["error"]
    plain = InProcessEngine(2, types.SimpleNamespace(tokenizer_manager=ReloadingEngine(None).tokenizer_manager))
    plain.start()
    assert plain.reload("/s") == {"error": "the in-process engine does not reload", "ok": False}


def test_reply_values_keep_only_tuples_of_scalars():
    assert reply_value((True, "ok", 0)) == [True, "ok", 0]
    assert reply_value((True, "a" * 5000))[1] == "a" * 4000
    assert reply_value([True, "ok"]) is None and reply_value((True, {"a": 1})) is None and reply_value(None) is None


class RecordingPool:
    def __init__(self, handle):
        self.handles = [handle]
        self.flushed = []

    def note_flush(self, i):
        self.flushed.append(i)


class SyncHandle:
    def __init__(self, reload_res=None, gen_res=None):
        self.reload_res, self.gen_res = reload_res, gen_res
        self.calls = []

    def reload(self, path, *, flush_cache, timeout_s):
        self.calls.append(("reload", path, flush_cache, timeout_s))
        return self.reload_res

    def generate(self, ids, sp, *, timeout_s, extra):
        self.calls.append(("generate", ids, sp, extra))
        return self.gen_res


def test_a_sync_engine_reload_returns_the_engines_reply_and_notes_the_flush():
    h = SyncHandle(reload_res={"ok": True, "value": [True, "read 31", 0], "reply": "(True, ...)", "sec": 0.1})
    pool = RecordingPool(h)
    e = wiring.SyncEngine(pool, 0, timeout_s=5.0)
    assert e.update_weights_from_disk(Path("/shm/r/w1")) == (True, "read 31", 0)
    assert h.calls == [("reload", "/shm/r/w1", True, 5.0)] and pool.flushed == [0]
    assert e.update_weights_from_disk("/shm/r/w0", flush_cache=False) == (True, "read 31", 0)
    assert pool.flushed == [0]  # no flush asked, none noted
    h.reload_res = {"ok": False, "value": None, "reply": "'weird'"}
    assert e.update_weights_from_disk("/x") == "'weird'" and pool.flushed == [0]
    h.reload_res = {"error": "engine 0 is dead", "kind": "dead"}
    with pytest.raises(wiring.EngineCallError, match="engine 0 reload: engine 0 is dead"):
        e.update_weights_from_disk("/x")


def test_a_sync_engine_generate_is_tokenspeed_shaped():
    slim = {"output_ids": [7], "cached_tokens": 0, "finish_reason": {"type": "length"},
            "in_lp_vals": [None, -1.5, -0.25], "in_lp_ids": [3, 4, 5]}
    h = SyncHandle(gen_res=slim)
    e = wiring.SyncEngine(RecordingPool(h), 0, timeout_s=5.0)
    out = e.generate(input_ids=(3, 4, 5), sampling_params={"temperature": 0.0, "max_new_tokens": 1},
                     return_logprob=True, logprob_start_len=0)
    assert h.calls == [("generate", [3, 4, 5], {"temperature": 0.0, "max_new_tokens": 1}, {"logprob_start_len": 0})]
    assert out == {"output_ids": [7], "meta_info": {"cached_tokens": 0, "finish_reason": {"type": "length"},
                                                    "input_token_logprobs": [(None, 3, None), (-1.5, 4, None),
                                                                             (-0.25, 5, None)]}}
    h.gen_res = {"output_ids": [9]}
    assert "input_token_logprobs" not in e.generate([1], {"max_new_tokens": 1})["meta_info"]
    assert h.calls[-1][3] == {}
    h.gen_res = {"error": "timeout", "kind": "timeout"}
    with pytest.raises(wiring.EngineCallError, match="generate: timeout"):
        e.generate([1], {"max_new_tokens": 1})


# --- C2 + C3 ----------------------------------------------------------------------------------------------------


def test_the_reward_adapter_runs_the_fake_profile_with_the_log_only_hook(tmp_path):
    from aorlta.reward.clock import Clock
    from aorlta.spans import SpanRecorder

    cfg = _cfg(tmp_path)
    clock = Clock()
    r = wiring.RewardAdapter(cfg, clock=clock, spans=SpanRecorder(monotonic=clock.now, wall=clock.wall))
    assert r._svc is None  # nothing starts before the run plan is written
    try:
        r.reopen()
        r.stats()
        assert r.plan_log.is_file()
        assert r.work_dir.is_relative_to(wiring.scratch_dir(cfg))
        # D7's retry reaches the service's resubmit (a malformed candidate is unscored there, never a default)
        res = r.resubmit(types.SimpleNamespace(candidate_id="w/1/p/0")).result(timeout=30)
        assert (res.status, res.reason) == ("unscored", "bad_candidate")
    finally:
        r.close()
    assert not r.work_dir.exists()


def test_the_reward_adapter_returns_the_services_drain_report(tmp_path):
    """The loop and score mode record the legs a drain's deadline caught from the report `drain` returns."""
    from aorlta.reward.service import DrainReport

    r = wiring.RewardAdapter(_cfg(tmp_path), clock=None, spans=None)
    report = DrainReport(resolved=1, scored=0, unscored=1, unscored_by_reason={"sanitizer_not_settled": 1},
                         outstanding_at_deadline=[{"candidate_id": "w/1/p/0", "leg": "san.asan", "where": "running"}],
                         hit_deadline=True, t_drain=10.0, t_done=40.0)
    deadlines = []
    r._svc = types.SimpleNamespace(drain=lambda deadline: deadlines.append(deadline) or report)
    assert r.drain(40.0) is report and deadlines == [40.0]


def test_a_profile_scored_by_another_reward_function_is_refused(tmp_path, monkeypatch):
    from aorlta.reward import profiles

    real = profiles.load_profile

    def other(name):
        return types.SimpleNamespace(**{**vars(real(name)), "reward_fn": "other.v1"}) if name == "fake" else real(name)

    monkeypatch.setattr(profiles, "load_profile", other)
    with pytest.raises(wiring.WiringError, match="scored by other.v1"):
        wiring.RewardAdapter(_cfg(tmp_path), clock=None, spans=None)


# --- the glue for the GPU stand-in runs -------------------------------------------------------------------------


def test_the_trainer_takes_require_fast_path_from_the_config_unless_a_test_overrides_it(tmp_path):
    assert wiring.TrainerAdapter(_cfg(tmp_path), spans=None).require_fast_path is True
    off = _cfg(tmp_path, trainer={"require_fast_path": False})
    assert wiring.TrainerAdapter(off, spans=None).require_fast_path is False
    assert wiring.TrainerAdapter(off, spans=None, require_fast_path=True).require_fast_path is True
    assert wiring.Parts().require_fast_path is None  # a run passes nothing, so the config decides


def test_remote_adapters_replace_their_in_process_factories_and_reach_the_server(tmp_path):
    import tempfile

    from aorlta.reward.adapters.base import SequentialBatch
    from aorlta.reward.adapters.fake import FakeAdapterConfig, FakeCheckAdapter
    from aorlta.reward.clock import Clock
    from aorlta.reward.remote import RemoteAdapter, RemoteUnavailable, Server, socket_path

    socks = Path(tempfile.mkdtemp(prefix="rmt-"))  # a socket path is limited to ~107 bytes
    cfg = _cfg(tmp_path, reward={"remote": {"socket_dir": str(socks), "adapters": {"fake-asan": 5},
                                            "connect_timeout_s": 0.5}})
    wiring.check_supported(cfg)
    factories = wiring.remote_factories(cfg, Clock())
    assert set(factories) == {"fake-asan"} and wiring.remote_factories(_cfg(tmp_path), Clock()) == {}
    with pytest.raises(RemoteUnavailable):
        factories["fake-asan"]("asan/1")  # no server: an error at construction, never a silent local fallback
    server = Server(socket_path(socks, "fake-asan"), "fake-asan", lambda wid: SequentialBatch(
        FakeCheckAdapter(FakeAdapterConfig(legs=("san.asan",))))).start()
    try:
        a = factories["fake-asan"]("asan/1")
        assert isinstance(a, RemoteAdapter) and a.legs == ("san.asan",) and a.max_packages == 16
        a.close()
    finally:
        server.stop()


def test_a_remote_adapter_the_profile_does_not_use_is_refused(tmp_path):
    cfg = _cfg(tmp_path, reward={"remote": {"socket_dir": "/dev/shm/x", "adapters": {"standin.aorta_asan": 5}}})
    with pytest.raises(wiring.WiringError, match="does not use"):
        wiring.check_supported(cfg)


class _Manager:
    def __init__(self):
        self.kw = None

    def run(self, prompts, iteration, policy_version, **kw):
        self.kw = kw
        return "result"


def test_the_rollout_adapter_hands_over_each_chain_and_puts_the_deadline_on_the_rollouts_clock(tmp_path):
    from aorlta.orchestrator.ports import ChainEnd
    from aorlta.rollout.chain import ChainOutcome

    class Clock:
        def now(self):
            return 100.0

    class Spans:
        def now(self):
            return 1.8e9

    r = wiring.RolloutAdapter.__new__(wiring.RolloutAdapter)
    r.manager, r.clock, r.spans = _Manager(), Clock(), Spans()
    got = []
    assert r.run([], 1, "v", on_candidate=got.append, on_chain=got.append, deadline=130.0) == "result"
    kw = r.manager.kw
    assert kw["on_candidate"] is None and kw["deadline"] == pytest.approx(1.8e9 + 30.0)
    outcome = ChainOutcome(record="rec", candidate="cand", sample="sample", group=0)
    kw["on_chain_end"](outcome)
    assert got == [ChainEnd("rec", "cand", "sample")]
    r.run([], 1, "v", on_candidate=got.append)  # no chain callback: candidates still stream, and no deadline
    assert r.manager.kw["on_candidate"] == got.append and r.manager.kw["on_chain_end"] is None
    assert r.manager.kw["deadline"] is None


def test_an_adapter_on_several_gpus_gets_one_factory_over_its_servers(tmp_path):
    from aorlta.reward.clock import Clock
    from aorlta.reward.remote import RemoteFactory

    cfg = _cfg(tmp_path, reward={"remote": {"socket_dir": "/dev/shm/x", "adapters": {"fake-asan": [5, 7]}}})
    f = wiring.remote_factories(cfg, Clock())["fake-asan"]
    assert isinstance(f, RemoteFactory) and f.replicas == 2
    assert [p.name for p in f.paths] == ["fake-asan.gpu5.sock", "fake-asan.gpu7.sock"]
    one = wiring.remote_factories(_cfg(tmp_path, reward={"remote": {"socket_dir": "/dev/shm/x",
                                                                    "adapters": {"fake-asan": 5}}}), Clock())
    assert not isinstance(one["fake-asan"], RemoteFactory)  # one GPU: the plain factory on <adapter>.sock


def test_the_reward_adapter_builds_the_remote_adapters_in_place_of_its_own(tmp_path):
    import tempfile

    from aorlta.reward.clock import Clock
    from aorlta.reward.remote import RemoteUnavailable

    socks = Path(tempfile.mkdtemp(prefix="rmt-"))
    cfg = _cfg(tmp_path, reward={"remote": {"socket_dir": str(socks), "adapters": {"fake-asan": 5},
                                            "connect_timeout_s": 0.2}})
    r = wiring.RewardAdapter(cfg, clock=Clock(), spans=None)
    with pytest.raises(RemoteUnavailable):  # the ASan pool's adapter is the remote one, and nothing serves it
        r.factories["fake-asan"]("asan/1")
    assert r.factories["fake"]("w/1").name.startswith("fake")  # the in-process adapters stay
