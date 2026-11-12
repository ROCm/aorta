"""The worked example of docs/TUTORIAL.md (examples/hip-kernels), on CPU: its configs load with the plugin, its
prompts and probes build, its parse asks for a repair turn, its check maps each harness outcome to a leg state
without a GPU (a fake hipcc stands in for the compiler and the GPU run), its reward keeps the service's invariants,
and every task's harness compiles with its reference kernel where hipcc is installed."""

from __future__ import annotations

import itertools
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from aorlta import plugins
from aorlta.config import config_from_dict, load_config, parse_yaml

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "hip-kernels"
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tests" / "orchestrator"))
import test_wiring_e2e as E  # noqa: E402 - the tiny model and its engines, for the wired test
from test_plugins import CharChat, registry  # noqa: E402,F401 - the probe tokenizer and the registry's restore

tiny_base = E.tiny_base

HIPCC = os.environ.get("HIPCC", "/opt/rocm/bin/hipcc")


@pytest.fixture
def hipk(registry):  # noqa: F811
    """The example plugin, loaded as a run config loads it; its module."""
    plugins.load_plugins(["examples/hip-kernels/hip_kernels.py"])
    return sys.modules[plugins.identity()["examples/hip-kernels/hip_kernels.py"]["module"]]


def site_config(tmp_path, name: str):
    site = tmp_path / "site.yaml"
    site.write_text(json.dumps({"run": {"root": str(tmp_path / "runs")}, "model": {"base_dir": str(tmp_path / "base")},
                                "data": {"prompt_pools": [str(EXAMPLE)]}}))
    return load_config(ROOT / "configs" / f"{name}.yaml", site=site)


@pytest.mark.parametrize("name", ["hipk-score", "hipk-train"])
def test_the_example_configs_load_with_the_plugin(tmp_path, registry, name):  # noqa: F811
    cfg = site_config(tmp_path, name)
    assert cfg.plugins == ("examples/hip-kernels/hip_kernels.py",) and cfg.data.prompt_pools[0].task == "hipk"
    assert (cfg.reward.profile, cfg.reward.reward_fn, cfg.rollout.parse_adapter) == ("hipk-gpu", "hipk.v1",
                                                                                     "hipk.parse")
    from aorlta.orchestrator import wiring

    wiring.check_supported(cfg)


def test_the_tasks_give_prompts_and_probes(tmp_path, hipk):
    from aorlta.orchestrator import wiring

    cfg = site_config(tmp_path, "hipk-score")
    tok = CharChat()
    recs = wiring.TaskPromptPool(plugins.task("hipk"), tok, "off", EXAMPLE).records()
    tasks = hipk.load_tasks(EXAMPLE)
    assert [r.prompt_id for r in recs] == [f"hipk:{t['id']}" for t in tasks] and len(recs) == 8
    assert {r.task.task_sha256 for r in recs} == {hipk.task_sha256(t) for t in tasks}
    for r, t in zip(recs, tasks):
        assert t["signature"] in r.messages[1]["content"]
        assert (EXAMPLE / "reference" / f"{t['id']}.hip").is_file()
    probes = wiring.task_probes(cfg, tok, plugins.task("hipk"), EXAMPLE, recs)
    assert len(probes.ids) == cfg.sync.probes


def prompt(hipk, kernel="saxpy"):
    from aorlta.orchestrator import wiring

    return next(r for r in wiring.TaskPromptPool(plugins.task("hipk"), CharChat(), "off", EXAMPLE).records()
                if r.prompt_id == f"hipk:{kernel}")


def test_the_parse_gives_a_repair_turn_for_no_block_or_no_kernel(hipk):
    from aorlta.rollout.chain import make_feedback_template, make_parse_adapter

    parse, fb = make_parse_adapter("hipk.parse"), make_feedback_template("hipk.feedback")
    p = prompt(hipk)
    no_block = parse.check("I would write a loop.", "off", p)
    assert no_block.kind == "rejected" and no_block.rejection[1] == "no_code_block"
    other = parse.check("```cpp\n__global__ void axpy(float a, const float* x, float* y, int n) {}\n```", "off", p)
    assert other.kind == "rejected" and other.rejection[1] == "no_kernel" and "`saxpy`" in other.rejection[0]
    good = parse.check("x\n```cpp\n#include <hip/hip_runtime.h>\n__global__ void saxpy(float a, const float* x, "
                       "float* y, int n) { }\n```\n", "off", p)
    assert good.kind == "edits" and "__global__ void saxpy(" in good.code
    turn = fb.build_apply(*other.rejection)
    assert turn.text == other.rejection[0] and len(turn.text) <= 2000


def proc(rc=0, stdout="", stderr="", timed_out=False, killed=False, launch_error=None):
    from aorlta.reward.adapters.base import ProcResult

    return ProcResult(rc, stdout, stderr, 0.1, timed_out, killed, launch_error)


def result_line(status, **kw):
    return "AORLTA_RESULT " + json.dumps({"status": status, "checked": 10, "mismatches": 0, **kw})


@pytest.mark.parametrize("build, run, want", [
    ((0, ""), proc(stdout=result_line("ok")), ("clean", None)),
    ((0, ""), proc(stdout=result_line("mismatch", mismatches=3)), ("fail", "wrong_output")),
    ((0, ""), proc(stdout=result_line("out_of_bounds")), ("fail", "wrong_output")),
    ((0, ""), proc(stdout=result_line("hip_error")), ("fail", "crash")),
    ((0, ""), proc(rc=-11), ("fail", "crash")),
    ((0, ""), proc(timed_out=True, rc=-9), ("fail", "timeout")),
    ((0, ""), proc(stdout="nothing"), ("unsettled", "no_result")),
    ((0, ""), proc(rc=1, stdout=result_line("ok")), ("unsettled", "no_result")),
    ((0, ""), proc(stdout=result_line("ok", checked=0)), ("unsettled", "no_result")),
    ((0, ""), proc(killed=True), ("unsettled", "run_not_started")),
    ((0, ""), proc(launch_error="OSError"), ("unsettled", "run_not_started")),
    ((1, "error: no matching function for call to 'saxpy'"), None, ("fail", "bad_signature")),
    ((1, "error: redefinition of 'main'"), None, ("fail", "bad_signature")),
    ((1, "ld.lld: error: unable to find library"), None, ("unsettled", "harness_build_failed")),
])
def test_each_harness_outcome_has_one_leg_state(hipk, build, run, want):
    task = next(t for t in hipk.load_tasks(EXAMPLE) if t["id"] == "saxpy")
    state, reason, _ = hipk.verdict(build[0], build[1], run, task)
    assert (state, reason) == want


FAKE_HIPCC = """#!/bin/bash
# a stand-in for hipcc: the 'compiled' program prints what FAKE_RUN says (ok, mismatch, crash, hang)
out=""; while [ $# -gt 0 ]; do [ "$1" = -o ] && { out=$2; shift; }; shift; done
[ "${FAKE_BUILD:-ok}" = sig ] && { echo "error: no matching function for call to 'saxpy'" >&2; exit 1; }
printf '#!/bin/bash\\necho "gpu $HIP_VISIBLE_DEVICES" >&2\\n' > "$out"
case "${FAKE_RUN:-ok}" in
  ok) echo 'echo "AORLTA_RESULT {\\"status\\": \\"ok\\", \\"checked\\": 5}"' >> "$out" ;;
  mismatch) echo 'echo "AORLTA_RESULT {\\"status\\": \\"mismatch\\", \\"checked\\": 5, \\"mismatches\\": 2}"' >> "$out" ;;
  crash) echo 'kill -SEGV $$' >> "$out" ;;
  hang) echo 'sleep 30' >> "$out" ;;
esac
chmod +x "$out"
"""


@pytest.fixture
def fake_hipcc(tmp_path) -> Path:
    p = tmp_path / "hipcc"
    p.write_text(FAKE_HIPCC)
    p.chmod(p.stat().st_mode | stat.S_IXUSR)
    return p


def run_job(hipk, tmp_path, fake_hipcc, *, worker="run/2", task_id="saxpy", timeout=5.0, run_timeout_s=2.0):
    from aorlta.schemas import LegJob

    task = next(t for t in hipk.load_tasks(EXAMPLE) if t["id"] == task_id)
    src = tmp_path / "source.hip"
    src.write_text("__global__ void saxpy(float a, const float* x, float* y, int n) {}\n")
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    ctx = plugins.CheckContext(name="hipk.run", worker_id=worker, settings={"hipcc": str(fake_hipcc), "gpus": [4, 6]},
                               clock=lambda: 0.0, timeout_s=timeout, pools=(str(EXAMPLE),))
    adapter = plugins.CHECK_ADAPTERS["hipk.run"](ctx)
    job = LegJob(job_id="j", leg="ref.native", candidate_id="c", control_id=None, source_sha256="s" * 64,
                 task_sha256=hipk.task_sha256(task), leg_settings={"target": "gfx950", "run_timeout_s": run_timeout_s},
                 inputs={"source": str(src)}, output_dir=str(out))
    return adapter, adapter.run(job)


@pytest.mark.parametrize("build, run, want", [
    ("ok", "ok", ("clean", None)), ("ok", "mismatch", ("fail", "wrong_output")), ("ok", "crash", ("fail", "crash")),
    ("ok", "hang", ("fail", "timeout")), ("sig", "ok", ("fail", "bad_signature")),
])
def test_the_check_runs_the_harness_on_its_workers_gpu(hipk, tmp_path, fake_hipcc, monkeypatch, build, run, want):
    monkeypatch.setenv("FAKE_BUILD", build)
    monkeypatch.setenv("FAKE_RUN", run)
    adapter, res = run_job(hipk, tmp_path, fake_hipcc)
    assert (res.state, res.reason) == want and res.leg == "ref.native"
    assert adapter.gpu == 6  # worker 2 of gpus [4, 6]
    if build == "ok" and run == "ok":
        (log,) = tmp_path.glob("out/*/run.log")
        assert "gpu 6" in log.read_text()


def test_a_task_the_check_does_not_know_is_unsettled(hipk, tmp_path, fake_hipcc):
    from aorlta.schemas import LegJob

    ctx = plugins.CheckContext(name="hipk.run", worker_id="run/1", settings={"hipcc": str(fake_hipcc)},
                               clock=lambda: 0.0, timeout_s=5.0, pools=(str(tmp_path),))
    adapter = plugins.CHECK_ADAPTERS["hipk.run"](ctx)
    job = LegJob(job_id="j", leg="ref.native", candidate_id="c", control_id=None, source_sha256="s" * 64,
                 task_sha256="0" * 64, inputs={"source": str(tmp_path / "x.hip")}, output_dir=str(tmp_path))
    res = adapter.run(job)
    assert (res.state, res.reason) == ("unsettled", "adapter_error")


def leg(name, state, reason=None):
    from aorlta.schemas import LegResult

    return LegResult(leg=name, state=state, reason=reason, findings=[], repeats=1 if state != "unsettled" else 0,
                     attempts=1, worker_id=None, bracket=None, cache_hit=False, t_queued=0.0, t_start=0.0, t_end=0.0)


def cand(tokens=100):
    from aorlta.schemas import Candidate, TaskRef

    return Candidate(candidate_id="c", run_id="r", iteration=1, prompt_id="hipk:saxpy", k=0, policy_version="v",
                     response_text="", response_tokens=tokens, max_new_tokens=2048,
                     task=TaskRef("hipk:saxpy", None, None, None, "t" * 64), check_profile="hipk-gpu",
                     rollout_compile=None)


OUTCOMES = {
    "gate.parse": [("clean", None), ("fail", "parse"), ("unsettled", "timeout")],
    "gate.compile": [("clean", None), ("fail", "compile_error"), ("unsettled", "adapter_error"), ("not_run", None)],
    "ref.native": [("clean", None), ("fail", "wrong_output"), ("fail", "crash"), ("fail", "timeout"),
                   ("fail", "bad_signature"), ("fail", "other"), ("unsettled", "no_result"), ("not_run", None),
                   ("skipped_by_plan", None)],
}


def test_the_reward_ladder(hipk):
    fn = plugins.REWARD_FNS["hipk.v1"](None)
    c = cand(tokens=0)

    def score(**states):
        legs = {n: leg(n, *states.get(n.replace(".", "_"), ("clean", None))) for n in fn.legs}
        return fn.compute(legs, c)[0]

    assert score() == 1.0
    assert score(gate_parse=("fail", "parse")) == -1.0
    assert score(gate_compile=("fail", "compile_error")) == -0.9
    assert score(ref_native=("fail", "bad_signature")) == -0.8
    assert score(ref_native=("fail", "crash")) == score(ref_native=("fail", "timeout")) == -0.6
    assert score(ref_native=("fail", "wrong_output")) == -0.5
    assert score(ref_native=("unsettled", "no_result")) is None and score(ref_native=("fail", "other")) is None


def test_the_reward_keeps_the_services_invariants_and_decide_agrees_with_compute(hipk):
    """Every combination of leg outcomes, and every prefix of it as a partial record: decide never contradicts
    compute, and a score is a float in the service's range whose parts are base, broke_healthy and length and add
    up to it (reward/service.py `_invariant_problems`)."""
    from aorlta.reward.rewardfn.base import Decided, NeedMore, Unscored
    from aorlta.reward.service import REWARD_MAX, REWARD_MIN

    fn = plugins.REWARD_FNS["hipk.v1"](None)
    for tokens in (0, 1024, 2048):
        c = cand(tokens)
        for combo in itertools.product(*(OUTCOMES[n] for n in fn.legs)):
            legs = {n: leg(n, *o) for n, o in zip(fn.legs, combo)}
            reward, parts = fn.compute(legs, c)
            if reward is not None:
                assert type(reward) is float and REWARD_MIN <= reward <= REWARD_MAX
                assert set(parts) == {"base", "broke_healthy", "length"} and all(type(v) is float for v in parts.values())
                assert parts["base"] + parts["broke_healthy"] + parts["length"] == reward
                assert -0.05 <= parts["length"] <= 0.0 and parts["broke_healthy"] == 0.0
                assert fn.claims_clean(reward, parts) == all(legs[n].state == "clean" for n in fn.legs)
            else:
                assert isinstance(parts.get("reason"), str) and not fn.claims_clean(reward, parts)
            for k in range(len(fn.legs) + 1):
                d = fn.decide({n: legs[n] for n in fn.legs[:k]}, c)
                if isinstance(d, Decided):
                    assert (d.reward, d.parts) == (reward, parts)
                elif isinstance(d, Unscored):
                    assert reward is None
                else:
                    assert isinstance(d, NeedMore) and d.legs == (fn.legs[k],)


@pytest.mark.skipif(not Path(HIPCC).exists(), reason=f"no hipcc at {HIPCC} (HIPCC)")
def test_every_harness_compiles_with_its_reference_kernel(tmp_path):
    """The self-check's build step, CPU only: nothing is run."""
    proc_ = subprocess.run([sys.executable, str(EXAMPLE / "hip_kernels.py"), "selfcheck", "--hipcc", HIPCC,
                            "--build-only"], capture_output=True, text=True, env=dict(os.environ, TMPDIR=str(tmp_path)))
    rows = [json.loads(x) for x in proc_.stdout.splitlines()]
    assert proc_.returncode == 0 and len(rows) == 8 and all(r["pass"] for r in rows), proc_.stdout + proc_.stderr


def test_the_profile_names_only_what_the_plugin_and_the_package_provide(hipk):
    from aorlta.reward.profiles import load_profile

    profile = load_profile("hipk-gpu")
    assert profile.reward_fn == "hipk.v1" and profile.legs == ("gate.parse", "gate.compile", "ref.native")
    assert {p.adapter for p in profile.pools.values()} == {"standin.parse", "standin.hipcc", "hipk.run"}
    raw = parse_yaml((EXAMPLE / "hipk-gpu.yaml").read_bytes())
    assert raw["adapters"]["hipk.run"]["gpus"] == [4, 5, 6, 7]


def test_the_score_config_keeps_the_checks_off_the_engines(tmp_path, registry):  # noqa: F811
    cfg = site_config(tmp_path, "hipk-score")
    from aorlta.reward.profiles import load_profile

    gpus = set(load_profile(cfg.reward.profile).adapters["hipk.run"]["gpus"])
    assert not gpus & set(cfg.rollout.engines)


def test_the_tutorial_names_files_that_exist():
    text = (ROOT / "docs" / "TUTORIAL.md").read_text()
    for rel in ("examples/hip-kernels/hip_kernels.py", "examples/hip-kernels/tasks.json",
                "examples/hip-kernels/hipk-gpu.yaml", "configs/hipk-score.yaml", "configs/hipk-train.yaml"):
        assert rel in text and (ROOT / rel).exists(), rel


def test_config_from_dict_also_takes_the_plugin(tmp_path, registry):  # noqa: F811
    data = parse_yaml((ROOT / "configs" / "hipk-score.yaml").read_bytes())
    data["run"]["root"] = str(tmp_path)
    data["data"]["prompt_pools"][0]["path"] = str(EXAMPLE)
    assert config_from_dict(data).reward.profile == "hipk-gpu"


class _CpuLeg:
    """A CPU stand-in for one of the example's checks that need hipcc or a GPU (the wired test below): the compile
    gate is clean, and `ref.native` passes half the sources by hash, so the rewards in a group differ."""

    version, resource = "1", "local_cpu"

    def __init__(self, name, leg):
        self.name, self.legs = name, (leg,)

    def run(self, job):
        from aorlta.reward.adapters.base import make_result

        if job.leg == "gate.compile" or job.source_sha256[-1] in "02468ace":
            return make_result(job, "clean", t_start=0.0, t_end=0.0)
        return make_result(job, "fail", "wrong_output", t_start=0.0, t_end=0.0)

    def controls(self):
        return []

    def health(self):
        return {"responsive": True}


@pytest.mark.heavy
def test_the_example_trains_on_cpu_through_the_real_wiring(tmp_path, hipk, tiny_base):
    """The example's prompts, parse, feedback, profile and reward in the real loop, with test_wiring_e2e's tiny model
    and engines. The fake engine never writes the kernel asked for, so every chain gets hipk.parse's repair turn."""
    from aorlta.orchestrator import wiring
    from aorlta.reward.adapters.standin_parse import StandinParse
    from aorlta.rollout.engine_proc import InProcessEngine

    data = parse_yaml((ROOT / "configs" / "hipk-train.yaml").read_bytes())
    data["run"].update(root=str(tmp_path / "runs"), run_id="hipk-cpu")
    data["model"]["base_dir"] = str(tiny_base)
    data["data"]["prompt_pools"][0]["path"] = str(EXAMPLE)
    data["rollout"].update(engines=[0, 1], K=4, P=4, max_new_tokens=512, request_timeout_s=300,
                           compile_adapter="fake.compile")
    data["orchestrator"]["iterations"] = 1
    data["trainer"] = {"ranks": 1, "lr": 1e-2, "micro_batch_tokens": 2048}
    data["sync"] = {"dir": str(tmp_path / "shm")}
    cfg = config_from_dict(data)
    engines = [E.TwinEngine(tiny_base, E.FAKE, i) for i in range(2)]
    factories = {"standin.parse": lambda w: StandinParse(), "standin.hipcc": lambda w: _CpuLeg("standin.hipcc",
                                                                                                 "gate.compile"),
                 "hipk.run": lambda w: _CpuLeg("hipk.run", "ref.native")}
    parts = wiring.Parts(tokenizer=E.ByteTokenizer(), handles=[InProcessEngine(i, e) for i, e in enumerate(engines)],
                         load_model=E.load_trainer_model, trainer_device="cpu", require_fast_path=False,
                         probes=E.PROBES, reward_factories=factories)
    orch = wiring.build(cfg, parts=parts)
    result = orch.run()
    assert result["status"] == "completed", result
    s = E.summary(cfg, 1)
    assert s["trainer"]["step_taken"] and s["samples"]["trained"] > 0 and s["sync"]["landing"] == "LANDED"
    eps = E.episodes(cfg, 1)
    assert {e["prompt_id"].split(":")[0] for e in eps} == {"hipk"}
    assert all(len(e["turns"]) == 3 and e["end_reason"] == "no_code_block" for e in eps if e["end_reason"] != "engine_error")
    bases = {round(e["reward"] + 0.05, 1) for e in eps if e["status"] == "scored"}  # base, give or take the length term
    assert bases <= {1.0, -0.5, -1.0} and {1.0, -0.5} <= bases  # correct, wrong output, and no code block at all
