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


# --- the timing check (hipk-speed.yaml) ----------------------------------------------------------

READELF = """
Symbol table '.symtab' contains 22 entries:
   Num:    Value          Size Type    Bind   Vis       Ndx Name
     1: 0000000000000000     0 FILE    LOCAL  DEFAULT   ABS cand.hip
    11: 0000000000000000   138 FUNC    GLOBAL DEFAULT     2 _Z25__device_stub__vector_addPKfS0_Pfi
    12: 0000000000000000     0 NOTYPE  GLOBAL DEFAULT   UND __hipPopCallConfiguration
    13: 0000000000000000     8 OBJECT  GLOBAL DEFAULT     4 _Z10vector_addPKfS0_Pfi
    14: 0000000000000000     0 NOTYPE  GLOBAL DEFAULT   UND hipLaunchKernel
    15: 0000000000000090   233 FUNC    GLOBAL DEFAULT     2 aorlta_launch_candidate
    21: 0000000000000008     1 OBJECT  GLOBAL DEFAULT    10 __hip_cuid_f5937891a52fad34
"""


def timing_line(status="ok", cand_us=None, ref_us=None, **kw):
    body = {"status": status, "warmup": 5, "launches": 20, **kw}
    if cand_us is not None:
        body.update(candidate_us=cand_us, reference_us=ref_us)
    return "AORLTA_TIMING " + json.dumps(body)


def steady(us, n=35):
    return [100.0] * 5 + [us] * (n - 5)


class FakeRunner:
    """The timing check's subprocesses, without hipcc or a GPU: the compile, the symbol read, the link and the timed
    run each answer as the test sets them; every call is recorded."""

    def __init__(self, run, *, build=None, syms=None, link=None):
        self.answers = {"build": build or proc(), "syms": syms or proc(stdout=READELF), "link": link or proc(),
                        "run": run}
        self.calls = []

    def run(self, argv, *, timeout_s, env=None, cwd=None):
        step = ("syms" if argv[0].endswith("readelf") else "build" if "-c" in argv
                else "link" if argv[0].endswith("hipcc") else "run")
        self.calls.append((step, list(argv), timeout_s, env))
        return self.answers[step]

    def close(self):
        pass


def time_job(hipk, tmp_path, runner, *, settings=None, task_id="vector_add", pools=None):
    from aorlta.reward.adapters.base import checked_result
    from aorlta.schemas import LegJob

    task = next(t for t in hipk.load_tasks(EXAMPLE) if t["id"] == task_id)
    src = tmp_path / "source.hip"
    src.write_text((EXAMPLE / "reference" / f"{task_id}.hip").read_text())
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    ctx = plugins.CheckContext(name="hipk.time", worker_id="time/1",
                               settings={"hipcc": "/x/bin/hipcc", "readelf": "/x/llvm/bin/llvm-readelf", "gpus": [7]},
                               clock=lambda: 0.0, timeout_s=30.0, pools=pools or (str(EXAMPLE),))
    adapter = plugins.CHECK_ADAPTERS["hipk.time"](ctx)
    adapter.runner = runner
    leg_settings = {"target": "gfx950", "ratio": 1.1, "warmup": 5, "repeats": 30, "launches": 20, **(settings or {})}
    job = LegJob(job_id="j", leg="ref.speedup", candidate_id="c", control_id=None, source_sha256="s" * 64,
                 task_sha256=hipk.task_sha256(task), leg_settings=leg_settings, inputs={"source": str(src)},
                 output_dir=str(out))
    return checked_result(job, adapter.run(job), 0.0, 0.0)


TIMED = [
    ("faster", proc(stdout=timing_line(cand_us=steady(8.0), ref_us=steady(10.0))), ("clean", None)),
    ("slower", proc(stdout=timing_line(cand_us=steady(12.0), ref_us=steady(10.0))), ("fail", "not_faster")),
    ("itself", proc(stdout=timing_line(cand_us=steady(10.0), ref_us=steady(10.0))), ("fail", "not_faster")),
    ("within_noise", proc(stdout=timing_line(cand_us=steady(9.0), ref_us=steady(10.0))),
     ("unsettled", "timing_within_noise")),
    ("noisy", proc(stdout=timing_line(cand_us=[100.0] * 5 + [5.0, 15.0] * 15, ref_us=steady(10.0))),
     ("unsettled", "timing_noisy")),
    ("too_few", proc(stdout=timing_line(cand_us=steady(8.0, 20), ref_us=steady(10.0, 20))),
     ("unsettled", "timing_too_few")),
    ("timeout", proc(timed_out=True, rc=-9, stdout=""), ("unsettled", "timing_timeout")),
    ("crash", proc(rc=-11), ("unsettled", "timing_crash")),
    ("crash_after_ok", proc(rc=-11, stdout=timing_line(cand_us=steady(8.0), ref_us=steady(10.0))),
     ("unsettled", "no_result")),
    ("silent", proc(rc=0, stdout="nothing"), ("unsettled", "no_result")),
    ("wrong_output", proc(stdout=timing_line("candidate_wrong_output")), ("unsettled", "timing_wrong_output")),
    ("reference_wrong", proc(stdout=timing_line("reference_wrong_output")), ("unsettled", "reference_wrong_output")),
    ("hip_error", proc(stdout=timing_line("hip_error")), ("unsettled", "timing_hip_error")),
    ("too_slow", proc(stdout=timing_line("too_slow", candidate_us=9e5, reference_us=10.0)),
     ("unsettled", "too_slow_to_measure")),
    ("unknown_status", proc(stdout=timing_line("maybe")), ("unsettled", "no_result")),
    ("not_lists", proc(stdout=timing_line(cand_us="8", ref_us=steady(10.0))), ("unsettled", "no_result")),
    ("not_started", proc(launch_error="OSError"), ("unsettled", "run_not_started")),
    ("killed", proc(killed=True), ("unsettled", "run_not_started")),
]


@pytest.mark.parametrize("case, run, want", TIMED, ids=[c[0] for c in TIMED])
def test_each_timing_outcome_has_one_leg_state(hipk, tmp_path, case, run, want):
    from aorlta.reward import timing

    runner = FakeRunner(run)
    res = time_job(hipk, tmp_path, runner)
    assert (res.state, res.reason) == want and res.leg == "ref.speedup"
    assert (res.state == "clean") == (case == "faster")  # nothing it could not measure is clean
    rec = timing.measurement(res)
    assert (rec is not None) == (res.state != "unsettled")
    if case == "faster":
        assert rec["speedup"] == 1.25 and rec["samples"] == 30
    steps = [c[0] for c in runner.calls]
    assert steps == ["build", "syms", "link", "run"]
    (_, argv, _, env), = [c for c in runner.calls if c[0] == "run"]
    assert argv[1:4] == ["5", "30", "20"] and float(argv[4]) == 10000.0 and env["HIP_VISIBLE_DEVICES"] == "7"


SYMBOLS = [
    ("interposed", READELF + "    16: 00000000000000a0     9 FUNC    GLOBAL DEFAULT     2 hipEventElapsedTime\n",
     "defines_reserved_symbol"),
    ("hsa", READELF + "    16: 00000000000000a0     9 FUNC    WEAK   DEFAULT     2 hsa_signal_wait_scacquire\n",
     "defines_reserved_symbol"),
    ("main", READELF + "    16: 00000000000000a0     9 FUNC    GLOBAL DEFAULT     2 main\n", "defines_reserved_symbol"),
    ("hip_object", READELF + "    16: 00000000000000a0     8 OBJECT  GLOBAL DEFAULT     4 hipStreamPerThread\n",
     "defines_reserved_symbol"),
    ("c_function", READELF + "    16: 00000000000000a0     9 FUNC    GLOBAL DEFAULT     2 clock_gettime\n",
     "defines_reserved_symbol"),
    ("no_launcher", READELF.replace("aorlta_launch_candidate", "_Z6helperv"), "symbol_check_failed"),
    ("undefined_launcher", READELF.replace("     2 aorlta_launch_candidate", "   UND aorlta_launch_candidate"),
     "symbol_check_failed"),
]


@pytest.mark.parametrize("case, table, reason", SYMBOLS, ids=[c[0] for c in SYMBOLS])
def test_a_candidate_that_could_take_the_runtimes_place_is_not_timed(hipk, tmp_path, case, table, reason):
    runner = FakeRunner(proc(stdout=timing_line(cand_us=steady(1.0), ref_us=steady(10.0))), syms=proc(stdout=table))
    res = time_job(hipk, tmp_path, runner)
    assert (res.state, res.reason) == ("unsettled", reason)
    assert [c[0] for c in runner.calls] == ["build", "syms"]


def test_the_symbol_rules(hipk):
    syms = hipk.defined_symbols(READELF + "    16: 0 9 FUNC GLOBAL DEFAULT 2 __device_stub__k\n"
                                "    17: 0 4 OBJECT GLOBAL DEFAULT 4 table\n    18: 0 0 FUNC LOCAL DEFAULT 2 helper\n"
                                "    19: 0 9 FUNC GLOBAL DEFAULT 2 hipFoo@VERS_1\n")
    assert [n for n, _ in syms] == ["_Z25__device_stub__vector_addPKfS0_Pfi", "_Z10vector_addPKfS0_Pfi",
                                    "aorlta_launch_candidate", "__hip_cuid_f5937891a52fad34", "__device_stub__k",
                                    "table", "hipFoo"]
    assert hipk.reserved_symbols(syms) == ["hipFoo"]


@pytest.mark.parametrize("which, answer", [
    ("build", proc(rc=1, stderr="error: x")), ("build", proc(timed_out=True)), ("build", proc(killed=True)),
    ("syms", proc(rc=1)), ("syms", proc(launch_error="OSError")), ("link", proc(rc=1, stderr="ld.lld: error")),
    ("link", proc(timed_out=True)),
])
def test_a_harness_that_does_not_build_is_unsettled(hipk, tmp_path, which, answer):
    runner = FakeRunner(proc(stdout=timing_line(cand_us=steady(8.0), ref_us=steady(10.0))), **{which: answer})
    res = time_job(hipk, tmp_path, runner)
    assert res.state == "unsettled"
    assert res.reason == ("symbol_check_failed" if which == "syms" else "harness_build_failed")
    assert "run" not in [c[0] for c in runner.calls]


@pytest.mark.parametrize("settings", [{"ratio": 0}, {"repeats": 2}, {"launches": 0}, {"gpu_budget_s": 0},
                                      {"max_rel_mad": 2.0}, {"scalars": ["n"]}, {"scalars": {"n": 1.5}}])
def test_settings_that_cannot_decide_are_an_adapter_error(hipk, tmp_path, settings):
    runner = FakeRunner(proc(stdout=timing_line(cand_us=steady(8.0), ref_us=steady(10.0))))
    res = time_job(hipk, tmp_path, runner, settings=settings)
    assert (res.state, res.reason) == ("unsettled", "adapter_error") and not runner.calls


def test_a_task_without_its_reference_is_an_adapter_error(hipk, tmp_path):
    pool = tmp_path / "pool"
    pool.mkdir()
    (pool / "tasks.json").write_text((EXAMPLE / "tasks.json").read_text())
    runner = FakeRunner(proc(stdout=timing_line(cand_us=steady(8.0), ref_us=steady(10.0))))
    res = time_job(hipk, tmp_path, runner, pools=(str(pool),))
    assert (res.state, res.reason) == ("unsettled", "adapter_error")


def test_the_timing_sources_keep_the_candidate_apart(hipk):
    task = next(t for t in hipk.load_tasks(EXAMPLE) if t["id"] == "saxpy")
    cand_src = hipk.launcher_source(task, "#define float double\n__global__ void saxpy(float a, const float* x, "
                                          "float* y, int n) {}\n")
    assert cand_src.splitlines()[-1] == "}" and 'extern "C" void aorlta_launch_candidate(float* x, float* y, int n, ' \
                                                  'float alpha) {' in cand_src
    assert "saxpy<<<dim3((n + 255) / 256), dim3(256)>>>(alpha, x, y, n);" in cand_src
    ref = (EXAMPLE / "reference" / "saxpy.hip").read_text()
    harness = hipk.timing_source(task, ref)
    assert harness.startswith("#include <hip/hip_runtime.h>\n") and "#define float" not in harness
    body = harness.split("namespace aorlta_ref {", 1)[1].split("}  // namespace aorlta_ref", 1)[0]
    assert "#include" not in body and "__global__ void saxpy(" in body
    assert "aorlta_ref::saxpy<<<" in harness and "aorlta_launch_candidate(d_x, d_y, n, alpha)" in harness


def test_the_timing_sizes_are_the_settings_scalars(hipk, tmp_path):
    task = next(t for t in hipk.load_tasks(EXAMPLE) if t["id"] == "block_sum")
    ref = (EXAMPLE / "reference" / "block_sum.hip").read_text()
    harness = hipk.timing_source(task, ref, {"n": "(1 << 24) + 3", "rows": "7"})
    assert "  const int n = (1 << 24) + 3;" in harness and "  const int blocks = (n + 255) / 256;" in harness
    assert "rows" not in harness and "const int n = 100003;" in hipk.timing_source(task, ref)
    runner = FakeRunner(proc(stdout=timing_line(cand_us=steady(8.0), ref_us=steady(10.0))))
    assert time_job(hipk, tmp_path, runner, settings={"scalars": {"n": 4096}}, task_id="block_sum").state == "clean"
    (written,) = tmp_path.glob("out/*/timing.hip")
    assert "  const int n = 4096;" in written.read_text()


@pytest.mark.skipif(not Path(HIPCC).exists(), reason=f"no hipcc at {HIPCC} (HIPCC)")
@pytest.mark.parametrize("task_id", ["saxpy", "block_sum", "transpose"])
def test_the_timing_harness_builds_with_the_reference(hipk, tmp_path, task_id):
    """Its two translation units compile and link, CPU only: nothing is run."""
    task = next(t for t in hipk.load_tasks(EXAMPLE) if t["id"] == task_id)
    ref = (EXAMPLE / "reference" / f"{task_id}.hip").read_text()
    (tmp_path / "cand.hip").write_text(hipk.launcher_source(task, ref))
    from aorlta.reward.profiles import load_profile

    scalars = load_profile("hipk-speed").leg_settings["ref.speedup"]["scalars"]
    (tmp_path / "timing.hip").write_text(hipk.timing_source(task, ref, scalars))
    for argv in (["-c", "cand.hip", "-o", "cand.o"], ["cand.o", "timing.hip", "-o", "timing"]):
        p = subprocess.run([HIPCC, "--offload-arch=gfx950", "-O2", *argv], cwd=tmp_path, capture_output=True,
                           text=True, env=dict(os.environ, TMPDIR=str(tmp_path)))
        assert p.returncode == 0, p.stderr[-3000:]


def test_the_speed_profile_and_config(tmp_path, hipk, registry):  # noqa: F811
    from aorlta.reward.profiles import load_profile
    from aorlta.reward.rewardfn.for_profile import standin_rewardfn

    profile = load_profile("hipk-speed")
    assert profile.reward_fn == "hipk.speed"
    assert profile.legs == ("gate.parse", "gate.compile", "ref.native", "ref.speedup")
    assert profile.requires["ref.speedup"] == ("ref.native",)
    assert {p.adapter for p in profile.pools.values()} == {"standin.parse", "standin.hipcc", "hipk.run", "hipk.time"}
    from aorlta.reward import timing

    timing.TimingPolicy.from_settings(profile.leg_settings["ref.speedup"])
    time_gpus, run_gpus = set(profile.adapters["hipk.time"]["gpus"]), set(profile.adapters["hipk.run"]["gpus"])
    assert not time_gpus & run_gpus
    fn = standin_rewardfn(profile)
    assert fn.name == "hipk.speed" and fn.legs == profile.legs
    cfg = site_config(tmp_path, "hipk-speed-score")
    assert (cfg.reward.profile, cfg.reward.reward_fn) == ("hipk-speed", "hipk.speed")
    assert not (time_gpus | run_gpus) & set(cfg.rollout.engines)
    from aorlta.orchestrator import wiring

    wiring.check_supported(cfg)


def test_the_speed_reward_on_the_example_legs(hipk):
    from aorlta.reward import timing

    fn = plugins.REWARD_FNS["hipk.speed"](None)
    c = cand(tokens=0)
    good = timing.record(timing.judge([9.0] * 35, [10.0] * 5 + [12.0] * 30, timing.TimingPolicy()))
    legs = {n: leg(n, "clean") for n in ("gate.parse", "gate.compile", "ref.native")}
    clean = dict(legs, **{"ref.speedup": leg("ref.speedup", "clean")})
    clean["ref.speedup"].findings.append(good)
    reward, parts = fn.compute(clean, c)
    assert fn.claims_clean(reward, parts) and 0.75 < reward < 1.0
    slow = dict(legs, **{"ref.speedup": leg("ref.speedup", "unsettled", "timing_noisy")})
    assert fn.compute(slow, c) == (None, {"reason": "ref.speedup:timing_noisy"})
    wrong = dict(legs, **{"ref.native": leg("ref.native", "fail", "wrong_output"),
                          "ref.speedup": leg("ref.speedup", "not_run")})
    assert fn.compute(wrong, c)[0] == -0.5
