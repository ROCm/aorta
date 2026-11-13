"""The worked example of docs/TUTORIAL.md (examples/hip-kernels), on CPU: its configs load with the plugin, its
prompts and probes build, its parse asks for a repair turn, its check maps each harness outcome to a leg state
without a GPU (a fake hipcc stands in for the compiler and the GPU run), its reward keeps the service's invariants,
and every task's harness compiles with its reference kernel where hipcc is installed."""

from __future__ import annotations

import dataclasses
import hashlib
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


def rec(status=None, stage=None, **kw):
    out = {"stage": stage} if stage else {}
    if status:
        out.update({"status": status, "checked": 10, "mismatches": 0, **kw})
    return out


SPOOF = 'AORLTA_RESULT {"status": "ok", "checked": 1}'


@pytest.mark.parametrize("run, record, want", [
    (proc(), rec("ok"), ("clean", None)),
    (proc(), rec("mismatch", mismatches=3), ("fail", "wrong_output")),
    (proc(), rec("out_of_bounds"), ("fail", "wrong_output")),
    (proc(), rec("no_kernel", stage="setup"), ("fail", "bad_signature")),
    (proc(), rec("hip_error", stage="launch"), ("fail", "crash")),
    (proc(), rec("hip_error", stage="run"), ("fail", "crash")),
    (proc(), rec("hip_error", stage="setup"), ("unsettled", "setup_failed")),  # hipMalloc, the first copy, the load
    (proc(rc=-11), rec(stage="launched"), ("fail", "crash")),
    (proc(timed_out=True, rc=-9), rec(stage="launched"), ("fail", "timeout")),
    (proc(rc=-11), rec(stage="setup"), ("unsettled", "setup_failed")),
    (proc(timed_out=True, rc=-9), rec(stage="setup"), ("unsettled", "setup_timeout")),
    (proc(rc=-11), None, ("unsettled", "setup_failed")),
    (proc(stdout=SPOOF), None, ("unsettled", "setup_failed")),  # stdout is never read
    (proc(stdout=SPOOF), rec(stage="launched"), ("unsettled", "no_result")),
    (proc(rc=1), rec("ok"), ("unsettled", "no_result")),
    (proc(), rec("ok", checked=0), ("unsettled", "no_result")),
    (proc(killed=True), None, ("unsettled", "run_not_started")),
    (proc(launch_error="OSError"), None, ("unsettled", "run_not_started")),
])
def test_each_harness_outcome_has_one_leg_state(hipk, run, record, want):
    state, reason, _ = hipk.verdict(run, record)
    assert (state, reason) == want


def test_the_harness_holds_none_of_the_candidates_code(hipk):
    """Built from the task alone: the candidate's code object is loaded at run time, and the verdict goes to a file."""
    for task in hipk.load_tasks(EXAMPLE):
        src = hipk.harness_source(task)
        assert "hipModuleLoad(&mod_, argv[1])" in src and " printf(" not in src and "put(b_);" in src
        assert f"__builtin_get_device_side_mangled_name({task['kernel']})" in src
        assert src.count("put(") >= 4


FAKE_HIPCC = """#!/bin/bash
# a stand-in for hipcc: --genco writes a code object; otherwise the 'harness' writes the record FAKE_RUN says
out=""; genco=0; for a in "$@"; do [ "$a" = --genco ] && genco=1; done
while [ $# -gt 0 ]; do [ "$1" = -o ] && { out=$2; shift; }; shift; done
if [ "$genco" = 1 ]; then
  [ "${FAKE_BUILD:-ok}" = co ] && { echo "error: cannot compile" >&2; exit 1; }
  echo object > "$out"; exit 0
fi
[ "${FAKE_BUILD:-ok}" = harness ] && { echo "error: harness" >&2; exit 1; }
printf '#!/bin/bash\necho "gpu $HIP_VISIBLE_DEVICES co $(cat $1)" >&2\necho %s\n' "'$SPOOF'" > "$out"
case "${FAKE_RUN:-ok}" in
  ok) echo 'echo "{\\"status\\": \\"ok\\", \\"checked\\": 5}" > "$2"' >> "$out" ;;
  mismatch) echo 'echo "{\\"status\\": \\"mismatch\\", \\"checked\\": 5, \\"mismatches\\": 2}" > "$2"' >> "$out" ;;
  nokernel) echo 'echo "{\\"status\\": \\"no_kernel\\", \\"stage\\": \\"setup\\"}" > "$2"' >> "$out" ;;
  setup) echo 'echo "{\\"status\\": \\"hip_error\\", \\"stage\\": \\"setup\\", \\"at\\": \\"hipMalloc\\"}" > "$2"' >> "$out" ;;
  crash) echo 'echo "{\\"stage\\": \\"launched\\"}" > "$2"; kill -SEGV $$' >> "$out" ;;
  hang) echo 'echo "{\\"stage\\": \\"launched\\"}" > "$2"; sleep 30' >> "$out" ;;
  spoof) echo 'echo "{\\"stage\\": \\"launched\\"}" > "$2"' >> "$out" ;;
esac
chmod +x "$out"
""".replace("$SPOOF", SPOOF.replace('"', '\\"'))


@pytest.fixture
def fake_hipcc(tmp_path) -> Path:
    p = tmp_path / "hipcc"
    p.write_text(FAKE_HIPCC)
    p.chmod(p.stat().st_mode | stat.S_IXUSR)
    return p


def make_adapter(hipk, fake_hipcc, worker="run/2", gpus=(4, 6), timeout=5.0):
    ctx = plugins.CheckContext(name="hipk.run", worker_id=worker, settings={"hipcc": str(fake_hipcc),
                                                                           "gpus": list(gpus)},
                               clock=lambda: 0.0, timeout_s=timeout, pools=(str(EXAMPLE),))
    return plugins.CHECK_ADAPTERS["hipk.run"](ctx)


def run_job(hipk, tmp_path, fake_hipcc, *, worker="run/2", task_id="saxpy", timeout=5.0, run_timeout_s=2.0):
    from aorlta.schemas import LegJob

    task = next(t for t in hipk.load_tasks(EXAMPLE) if t["id"] == task_id)
    src = tmp_path / "source.hip"
    src.write_text("__global__ void saxpy(float a, const float* x, float* y, int n) {}\n")
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    adapter = make_adapter(hipk, fake_hipcc, worker=worker, timeout=timeout)
    job = LegJob(job_id="j", leg="ref.native", candidate_id="c", control_id=None, source_sha256="s" * 64,
                 task_sha256=hipk.task_sha256(task), leg_settings={"target": "gfx950", "run_timeout_s": run_timeout_s},
                 inputs={"source": str(src)}, output_dir=str(out))
    return adapter, adapter.run(job)


@pytest.mark.parametrize("build, run, want", [
    ("ok", "ok", ("clean", None)), ("ok", "mismatch", ("fail", "wrong_output")), ("ok", "crash", ("fail", "crash")),
    ("ok", "hang", ("fail", "timeout")), ("ok", "nokernel", ("fail", "bad_signature")),
    ("ok", "setup", ("unsettled", "setup_failed")), ("ok", "spoof", ("unsettled", "no_result")),
    ("co", "ok", ("unsettled", "candidate_build_failed")), ("harness", "ok", ("unsettled", "harness_build_failed")),
])
def test_the_check_runs_the_harness_on_its_workers_gpu(hipk, tmp_path, fake_hipcc, monkeypatch, build, run, want):
    monkeypatch.setenv("FAKE_BUILD", build)
    monkeypatch.setenv("FAKE_RUN", run)
    adapter, res = run_job(hipk, tmp_path, fake_hipcc)
    try:
        assert (res.state, res.reason) == want and res.leg == "ref.native"
        assert adapter.gpu == 6  # worker 2 of gpus [4, 6], no other worker live
        if build == "ok" and run == "ok":
            (log,) = tmp_path.glob("out/*/run.log")
            assert "gpu 6 co object" in log.read_text()
            assert {p.name for p in log.parent.iterdir()} >= {"build.log", "harness.hip", "record.json", "run.log"}
    finally:
        adapter.close()


def test_a_replaced_worker_takes_the_gpu_its_predecessor_gave_back(hipk, fake_hipcc):
    """The service names a replacement by the next sequence number (run/5 for run/2), so a GPU from the id alone
    would land run/5 on run/1's GPU and leave run/2's idle."""
    probe = make_adapter(hipk, fake_hipcc, worker="run/probe", gpus=(4, 5, 6, 7))
    probe.close()
    workers = {n: make_adapter(hipk, fake_hipcc, worker=f"run/{n}", gpus=(4, 5, 6, 7)) for n in (1, 2, 3, 4)}
    assert [w.gpu for w in workers.values()] == [4, 5, 6, 7]
    workers.pop(2).close()
    workers[5] = make_adapter(hipk, fake_hipcc, worker="run/5", gpus=(4, 5, 6, 7))
    assert workers[5].gpu == 5
    workers[6] = make_adapter(hipk, fake_hipcc, worker="run/6", gpus=(4, 5, 6, 7))  # more workers than GPUs: shared
    assert workers[6].gpu in (4, 5, 6, 7)
    for w in workers.values():
        w.close()
    workers[1].close()  # a second close gives nothing back twice
    assert hipk._LEASED == {}


def test_the_checker_version_names_the_compiler_and_this_file(hipk, fake_hipcc, tmp_path):
    """Cached ref.native legs are keyed by the checker's version: another compiler, or an edited harness, is a new
    version, so an old verdict is not reused."""
    a = make_adapter(hipk, fake_hipcc)
    other = tmp_path / "other" / "hipcc"
    other.parent.mkdir()
    other.write_text(FAKE_HIPCC + "# another build\n")
    other.chmod(0o755)
    b = make_adapter(hipk, other)
    again = make_adapter(hipk, fake_hipcc)
    try:
        assert a.version == again.version != b.version
        assert a.version.endswith(hashlib.sha256(Path(hipk.__file__).read_bytes()).hexdigest()[:16])
    finally:
        for x in (a, b, again):
            x.close()


def test_the_reward_parse_gate_reads_the_answer_as_the_rollout_does(hipk, tmp_path):
    """A final reply whose last ```cpp block never closes is no answer to the rollout; the stand-in gate would score
    the earlier, complete block in its place."""
    from aorlta.reward.adapters.standin_parse import StandinParse
    from aorlta.rollout.compile_check import ParseOutcome  # noqa: F401
    from aorlta.schemas import LegJob

    cut = ("Here:\n```cpp\n__global__ void saxpy(float a, const float* x, float* y, int n) {}\n```\n"
           "Better:\n```cpp\n__global__ void saxpy(float a, const")
    whole = "```cpp\n__global__ void saxpy(float a, const float* x, float* y, int n) {}\n```\n"
    gate = plugins.CHECK_ADAPTERS["hipk.gate"](plugins.CheckContext(name="hipk.gate", worker_id="gates/1",
                                                                    settings={}, clock=lambda: 0.0, timeout_s=5.0))
    for i, (text, want) in enumerate([(cut, ("fail", "parse")), (whole, ("clean", None))]):
        (tmp_path / f"r{i}.txt").write_text(text)
        job = LegJob(job_id="j", leg="gate.parse", candidate_id="c", control_id=None, source_sha256="s" * 64,
                     task_sha256="t" * 64, inputs={"response": str(tmp_path / f"r{i}.txt")},
                     output_dir=str(tmp_path / f"o{i}"))
        res = gate.run(job)
        assert (res.state, res.reason) == want
        assert (tmp_path / f"o{i}" / "source.hip").exists() == (want[0] == "clean")
        assert StandinParse().run(dataclasses.replace(job, output_dir=str(tmp_path / f"s{i}"))).state == "clean"
    assert hipk.HipkParse().check(cut, "off", None).kind == "rejected"


def test_a_task_the_check_does_not_know_is_unsettled(hipk, tmp_path, fake_hipcc):
    from aorlta.schemas import LegJob

    ctx = plugins.CheckContext(name="hipk.run", worker_id="run/1", settings={"hipcc": str(fake_hipcc)},
                               clock=lambda: 0.0, timeout_s=5.0, pools=(str(tmp_path),))
    adapter = plugins.CHECK_ADAPTERS["hipk.run"](ctx)
    job = LegJob(job_id="j", leg="ref.native", candidate_id="c", control_id=None, source_sha256="s" * 64,
                 task_sha256="0" * 64, inputs={"source": str(tmp_path / "x.hip")}, output_dir=str(tmp_path))
    try:
        res = adapter.run(job)
        assert (res.state, res.reason) == ("unsettled", "adapter_error")
    finally:
        adapter.close()


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
    assert {p.adapter for p in profile.pools.values()} == {"hipk.gate", "standin.hipcc", "hipk.run"}
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
    factories = {"hipk.gate": lambda w: hipk.HipkGate(), "standin.hipcc": lambda w: _CpuLeg("standin.hipcc",
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
