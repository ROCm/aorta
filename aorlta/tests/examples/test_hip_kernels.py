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
import re
import shutil
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


@pytest.mark.parametrize("thinking", ["off", "on"])
def test_the_checks_are_given_the_runs_thinking_flag(tmp_path, hipk, monkeypatch, thinking):
    """In-process and in the checker servers (`remote serve`), the parse gate is built with `rollout.thinking`."""
    from aorlta.orchestrator import wiring
    from aorlta.reward.clock import Clock
    from aorlta.reward.remote import standin_factory

    data = parse_yaml((ROOT / "configs" / "hipk-score.yaml").read_bytes())
    data["rollout"]["thinking"] = thinking
    (tmp_path / "run.yaml").write_text(json.dumps(data))
    site_config(tmp_path, "hipk-score")
    monkeypatch.setenv("AORLTA_SITE", str(tmp_path / "site.yaml"))
    cfg = load_config(tmp_path / "run.yaml")
    assert cfg.rollout.thinking == thinking
    in_process = wiring.RewardAdapter(cfg, clock=Clock(), spans=None).factories["hipk.gate"]("gates/1")
    assert in_process.thinking == thinking and in_process.kernels
    assert standin_factory(str(tmp_path / "run.yaml"), "hipk.gate")("gates/1").thinking == thinking


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


def test_the_parse_names_the_kernel_from_the_signature_not_the_task_id(hipk, tmp_path):
    """Shape variants of one kernel have their own task ids; the kernel the reply must define is the signature's."""
    from aorlta.orchestrator import wiring
    from aorlta.rollout.chain import make_parse_adapter

    task = dict(hipk.load_tasks(EXAMPLE)[1], id="saxpy_n1024")
    (tmp_path / "tasks.json").write_text(json.dumps([task]))
    (p,) = wiring.TaskPromptPool(plugins.task("hipk"), CharChat(), "off", tmp_path).records()
    parse = make_parse_adapter("hipk.parse")
    parse.check_prompts([p], "off")
    ref = (EXAMPLE / "reference" / "saxpy.hip").read_text()
    assert parse.check(f"```cpp\n{ref}\n```", "off", p).kind == "edits"
    other = parse.check("```cpp\n__global__ void saxpy_n1024(float a, const float* x, float* y, int n) {}\n```",
                        "off", p)
    assert other.kind == "rejected" and other.rejection[1] == "no_kernel" and "`saxpy`" in other.rejection[0]
    unnamed = dataclasses.replace(p, messages=[p.messages[0], {"role": "user", "content": "Write a kernel."}])
    with pytest.raises(ValueError, match="no __global__ signature"):
        parse.check_prompts([unnamed], "off")


def test_the_selfcheck_fails_a_task_with_no_reference_and_a_run_that_checked_nothing(hipk, tmp_path, capsys):
    """By default every task needs its reference kernel; with --sources a task without one is skipped, but a run
    that checked no task at all is not a pass."""
    (tmp_path / "pool").mkdir()
    (tmp_path / "pool" / "tasks.json").write_text(json.dumps([dict(hipk.load_tasks(EXAMPLE)[0], id="no_reference")]))
    assert hipk.selfcheck(["--pool", str(tmp_path / "pool"), "--build-only"]) == 1
    (row,) = [json.loads(x) for x in capsys.readouterr().out.splitlines()]
    assert (row["task"], row["pass"], row["error"]) == ("no_reference", False, "no reference kernel")
    (tmp_path / "sources").mkdir()
    assert hipk.selfcheck(["--pool", str(EXAMPLE), "--sources", str(tmp_path / "sources"), "--build-only"]) == 1
    out = capsys.readouterr()
    assert out.out == "" and "had a kernel to check" in out.err


def proc(rc=0, stdout="", stderr="", timed_out=False, killed=False, launch_error=None):
    from aorlta.reward.adapters.base import ProcResult

    return ProcResult(rc, stdout, stderr, 0.1, timed_out, killed, launch_error)


def rec(status=None, stage=None, **kw):
    out = {"stage": stage} if stage else {}
    if status:
        out.update({"status": status, "checked": 10, "mismatches": 0, "modified": 0, **kw})
    return out


NO_INPUT_CHECK = {"status": "ok", "checked": 10, "mismatches": 0}  # from a harness that never compared inputs


SPOOF = 'AORLTA_RESULT {"status": "ok", "checked": 1}'


@pytest.mark.parametrize("run, record, want", [
    (proc(), rec("ok"), ("clean", None)),
    (proc(), rec("mismatch", mismatches=3), ("fail", "wrong_output")),
    (proc(), rec("out_of_bounds"), ("fail", "wrong_output")),
    (proc(), rec("modified_input", modified=2), ("fail", "modified_input")),
    (proc(), rec("ok", modified=1), ("unsettled", "no_result")),  # never clean with an input changed
    (proc(), NO_INPUT_CHECK, ("unsettled", "no_result")),
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
        assert ", e_)" not in src  # CHECK declares e_ itself: `hipError_t e_ = (e_)` is undefined, and at -O2 it
        #                           made every lookup take the no_kernel branch (found on a GPU)


def test_the_harness_checks_exactly_the_read_only_arrays_and_snapshots_them_before_the_run(hipk):
    """Every role-`in` array is copied aside before anything touches it and compared after the last copy back; an
    `out` or `inout` array (saxpy's y) is the kernel's to write, so it is not."""
    import re

    for task in hipk.load_tasks(EXAMPLE):
        src = hipk.harness_source(task)
        want = {a["name"] for a in task["arrays"] if a["role"] == "in"}
        assert want and set(re.findall(r"o_(\w+) = h_\1;", src)) == want
        assert set(re.findall(r"modified \+= changed\(h_(\w+), o_\1,", src)) == want
        first_upload, last_download = src.index("hipMemcpyHostToDevice"), src.rindex("hipMemcpyDeviceToHost")
        assert max(src.index(f"o_{n} = h_{n};") for n in want) < first_upload
        assert min(src.index(f"changed(h_{n},") for n in want) > last_download
        assert src.index("modified += ") < src.index(hipk.STATUS) and "\\\"modified\\\": %ld" in src
    saxpy = next(t for t in hipk.load_tasks(EXAMPLE) if t["id"] == "saxpy")
    assert "changed(h_y," not in hipk.harness_source(saxpy)


def test_a_task_with_an_unknown_array_role_is_refused(hipk, tmp_path):
    """A typo in a role must not make a read-only array writable."""
    task = hipk.load_tasks(EXAMPLE)[0]
    task["arrays"][0]["role"] = "input"
    (tmp_path / "tasks.json").write_text(json.dumps([task]))
    with pytest.raises(ValueError, match="role 'input', not one of in, out, inout"):
        hipk.load_tasks(tmp_path)


CXX = shutil.which(os.environ.get("CXX", "c++")) or shutil.which("g++")


def input_check_program(hipk, task) -> str:
    """A host program around the harness's own snapshot, input comparison and status lines, with fake buffers in
    place of the device: it fills every array, takes the snapshot, then applies the writes given on its command line
    (array, index, bits) as if the kernel had made them, and prints `modified` and the status."""
    lines = ["#include <cstdint>", "#include <cstdio>", "#include <cstdlib>", "#include <cstring>",
             "#include <string>", "#include <vector>", "", *hipk.CHANGED, "", "int main(int argc, char** argv) {"]
    lines += [f"  {ctype} {name} = {value};" for ctype, name, value in task["scalars"]]
    for a in task["arrays"]:
        lines += [f"  std::vector<float> h_{a['name']}(({a['size']}) + {hipk.GUARD}, {hipk.GUARD_VALUE});",
                  f"  for (long i = 0; i < ({a['size']}); ++i) h_{a['name']}[i] = 0.5f * float(i);"]
    lines += hipk.snapshot_lines(task)
    lines += ["  for (int k = 1; k + 2 < argc; k += 3) {",
              "    std::string name = argv[k];",
              "    long i = std::atol(argv[k + 1]);",
              "    uint32_t bits = (uint32_t)std::strtoul(argv[k + 2], nullptr, 0);",
              "    std::vector<float>* v = nullptr;"]
    lines += [f"    if (name == \"{a['name']}\") v = &h_{a['name']};" for a in task["arrays"]]
    lines += ["    if (v == nullptr || i < 0 || i >= (long)v->size()) return 2;",
              "    std::memcpy(&(*v)[i], &bits, sizeof bits);",
              "  }",
              "  long oob = 0, mismatches = 0, modified = 0;"]
    lines += hipk.input_check_lines(task)
    lines += [hipk.STATUS, "  std::printf(\"%ld %s\\n\", modified, status);", "  return 0;", "}", ""]
    return "\n".join(lines)


def f32_bits(x: float) -> str:
    import struct

    return hex(struct.unpack("<I", struct.pack("<f", x))[0])


N = 100003  # vector_add's and saxpy's n
CHANGED_CASES = {  # task: [(writes the 'kernel' made, (modified, status))]
    "vector_add": [
        ([], (0, "ok")),
        ([("a", 5, f32_bits(123.0))], (1, "modified_input")),
        ([("b", N - 1, f32_bits(-1.0))], (1, "modified_input")),  # the last element
        ([("a", 0, "0x80000000")], (1, "modified_input")),  # +0.0 became -0.0: equal as floats, not as bits
        ([("a", 3, f32_bits(1.5))], (0, "ok")),  # the value it already held: nothing changed
        ([("a", N, f32_bits(0.0))], (0, "ok")),  # the guard past the end is out_of_bounds's to catch, not this
        ([("c", 0, f32_bits(9.0)), ("c", N - 1, f32_bits(9.0))], (0, "ok")),  # the output
        ([("a", 1, f32_bits(7.0)), ("b", 2, f32_bits(7.0)), ("b", 4, f32_bits(7.0))], (3, "modified_input")),
    ],
    "saxpy": [
        ([("y", 0, f32_bits(4.0)), ("y", N - 1, f32_bits(4.0))], (0, "ok")),  # in place: declared inout
        ([("x", 7, f32_bits(4.0))], (1, "modified_input")),
    ],
    "transpose": [
        ([("in", 300 * 517 - 1, f32_bits(-2.0))], (1, "modified_input")),  # its size is an expression: rows * cols
        ([("out", 0, f32_bits(-2.0))], (0, "ok")),
    ],
}


@pytest.mark.skipif(CXX is None, reason="no C++ compiler (CXX, c++ or g++)")
@pytest.mark.parametrize("task_id", sorted(CHANGED_CASES))
def test_the_harness_fails_a_changed_read_only_input_bit_for_bit(hipk, tmp_path, task_id):
    """The harness's own comparison lines, run on the host over fake buffers: a read-only input that changed in any
    bit is counted and makes the status `modified_input`; an output or a declared in-place array may change."""
    task = next(t for t in hipk.load_tasks(EXAMPLE) if t["id"] == task_id)
    (tmp_path / "check.cpp").write_text(input_check_program(hipk, task))
    build = subprocess.run([CXX, "-O2", "-std=c++17", "-o", str(tmp_path / "check"), str(tmp_path / "check.cpp")],
                           capture_output=True, text=True)
    assert build.returncode == 0, build.stderr
    for writes, (modified, status) in CHANGED_CASES[task_id]:
        argv = [str(x) for w in writes for x in w]
        out = subprocess.run([str(tmp_path / "check"), *argv], capture_output=True, text=True)
        assert out.returncode == 0 and out.stdout.split() == [str(modified), status], (writes, out.stdout)


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
  ok) echo 'echo "{\\"status\\": \\"ok\\", \\"checked\\": 5, \\"modified\\": 0}" > "$2"' >> "$out" ;;
  mismatch) echo 'echo "{\\"status\\": \\"mismatch\\", \\"checked\\": 5, \\"mismatches\\": 2}" > "$2"' >> "$out" ;;
  modified) echo 'echo "{\\"status\\": \\"modified_input\\", \\"modified\\": 1}" > "$2"' >> "$out" ;;
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
    ("ok", "modified", ("fail", "modified_input")),
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


SAXPY = "__global__ void saxpy(float a, const float* x, float* y, int n) {}"
GATE_CASES = [  # (thinking, final reply)
    ("off", f"```cpp\n{SAXPY}\n```\n"),
    ("off", f"Here:\n```cpp\n{SAXPY}\n```\nBetter:\n```cpp\n__global__ void saxpy(float a, const"),
    ("off", "```cpp\n#define K saxpy\n__global__ void K(float a, const float* x, float* y, int n) {}\n```\n"),
    ("off", "```cpp\n__global__ void axpy(float a, const float* x, float* y, int n) {}\n```\n"),
    ("off", f"Draft:\n```cpp\n{SAXPY}\n```\n</think>\nNo code here."),
    ("on", f"Let me think.\n</think>\n```cpp\n{SAXPY}\n```\n"),
    ("on", f"A draft:\n```cpp\n{SAXPY}\n```\nand the thinking never ends"),
    ("on", f"A draft:\n```cpp\n{SAXPY}\n```\n</think>\nI could not finish it."),
    ("on", "Let me think.\n</think>\n```cpp\n#define K saxpy\n__global__ void K(float a) {}\n```\n"),
]


@pytest.mark.parametrize("thinking, text", GATE_CASES)
def test_the_reward_parse_gate_reads_the_answer_as_the_rollout_does(hipk, tmp_path, thinking, text):
    """The reward's parse gate passes exactly the final replies `hipk.parse` accepted, read with the run's thinking
    flag: a rejected reply is still scored (its chain ends `no_code_block`), and must score as a parse failure, not
    on a block the rollout never accepted (one before a cut-off block, one in the thinking, or one that does not
    define the kernel), all of which the stand-in gate passes."""
    from aorlta.reward.adapters.standin_parse import StandinParse
    from aorlta.schemas import LegJob

    p = prompt(hipk)
    gate = plugins.CHECK_ADAPTERS["hipk.gate"](plugins.CheckContext(
        name="hipk.gate", worker_id="gates/1", settings={}, clock=lambda: 0.0, timeout_s=5.0, pools=(str(EXAMPLE),),
        thinking=thinking))
    (tmp_path / "r.txt").write_text(text)
    job = LegJob(job_id="j", leg="gate.parse", candidate_id="c", control_id=None, source_sha256="s" * 64,
                 task_sha256=p.task.task_sha256, inputs={"response": str(tmp_path / "r.txt")},
                 output_dir=str(tmp_path / "o"))
    accepted = hipk.HipkParse().check(text, thinking, p).kind == "edits"
    res = gate.run(job)
    assert (res.state, res.reason) == (("clean", None) if accepted else ("fail", "parse"))
    assert (tmp_path / "o" / "source.hip").exists() == accepted
    assert StandinParse().run(dataclasses.replace(job, output_dir=str(tmp_path / "s"))).state == "clean"


def test_the_parse_gate_cases_cover_both_answers(hipk):
    p = prompt(hipk)
    kinds = [hipk.HipkParse().check(text, thinking, p).kind for thinking, text in GATE_CASES]
    assert kinds == ["edits", "rejected", "rejected", "rejected", "edits", "edits", "rejected", "rejected", "rejected"]


def test_the_parse_gate_is_unsettled_for_a_task_it_does_not_know_and_versioned_by_thinking(hipk, tmp_path):
    from aorlta.schemas import LegJob

    def gate(thinking):
        return plugins.CHECK_ADAPTERS["hipk.gate"](plugins.CheckContext(
            name="hipk.gate", worker_id="gates/1", settings={}, clock=lambda: 0.0, timeout_s=5.0,
            pools=(str(EXAMPLE),), thinking=thinking))

    (tmp_path / "r.txt").write_text(f"```cpp\n{SAXPY}\n```\n")
    job = LegJob(job_id="j", leg="gate.parse", candidate_id="c", control_id=None, source_sha256="s" * 64,
                 task_sha256="0" * 64, inputs={"response": str(tmp_path / "r.txt")}, output_dir=str(tmp_path / "o"))
    assert (gate("off").run(job).state, gate("off").run(job).reason) == ("unsettled", "adapter_error")
    (tmp_path / "r.txt").write_text("No code.")  # not a parse failure either: the task is unknown, not the reply bad
    assert (gate("off").run(job).state, gate("off").run(job).reason) == ("unsettled", "adapter_error")
    assert gate("off").version != gate("on").version


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
    "ref.native": [("clean", None), ("fail", "wrong_output"), ("fail", "modified_input"), ("fail", "crash"),
                   ("fail", "timeout"),
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
    assert score(ref_native=("fail", "wrong_output")) == score(ref_native=("fail", "modified_input")) == -0.5
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
    and engines. The fake engine never writes the kernel asked for, so every chain gets hipk.parse's repair turn, and
    the reward scores the rejected final replies as the rollout read them."""
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
    gate_ctx = plugins.CheckContext(name="hipk.gate", worker_id="gates/1", settings={}, clock=lambda: 0.0,
                                    timeout_s=5.0, pools=(str(EXAMPLE),), thinking=cfg.rollout.thinking)
    factories = {"hipk.gate": lambda w: hipk.HipkGate(gate_ctx), "standin.hipcc": lambda w: _CpuLeg("standin.hipcc",
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
    scored = [e for e in eps if e["status"] == "scored"]
    # every final reply was rejected, so each scores as a parse failure (-1.0, give or take the length term)
    assert scored and all(e["legs"]["gate.parse"] == "fail" and -1.05 <= e["reward"] <= -1.0 for e in scored)


# --- the timing check (hipk-speed.yaml) ----------------------------------------------------------


def timing_record(status="ok", cand_us=None, ref_us=None, **kw):
    body = {"status": status, "warmup": 5, "launches": 20, **kw}
    if cand_us is not None:
        body.update(candidate_us=cand_us, reference_us=ref_us)
    return body


def steady(us, n=35):
    return [100.0] * 5 + [us] * (n - 5)


class FakeRunner:
    """The timing check's subprocesses, without hipcc or a GPU: the candidate's and the reference's code objects, the
    harness build and the timed run each answer as the test sets them, the run writing `record` (a dict) to the
    record file it is given; every call is recorded."""

    def __init__(self, run, record=None, *, co=None, ref=None, harness=None):
        self.answers = {"co": co or proc(), "ref": ref or proc(), "harness": harness or proc(), "run": run}
        self.record, self.calls = record, []

    def run(self, argv, *, timeout_s, env=None, cwd=None, input=None):
        if argv[0].endswith("hipcc"):
            step = ("co" if argv[-1].endswith("source.hip") else "ref") if "--genco" in argv else "harness"
        else:
            step = "run"
        self.calls.append((step, list(argv), timeout_s, env))
        if step == "run" and self.record is not None:
            Path(argv[3]).write_text(json.dumps(self.record))
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
    ctx = plugins.CheckContext(name="hipk.time", worker_id="time/1", settings={"hipcc": "/x/bin/hipcc", "gpus": [7]},
                               clock=lambda: 0.0, timeout_s=30.0, pools=pools or (str(EXAMPLE),))
    adapter = plugins.CHECK_ADAPTERS["hipk.time"](ctx)
    adapter.runner = runner
    leg_settings = {"target": "gfx950", "ratio": 1.1, "warmup": 5, "repeats": 30, "launches": 20, **(settings or {})}
    job = LegJob(job_id="j", leg="ref.speedup", candidate_id="c", control_id=None, source_sha256="s" * 64,
                 task_sha256=hipk.task_sha256(task), leg_settings=leg_settings, inputs={"source": str(src)},
                 output_dir=str(out))
    try:
        return checked_result(job, adapter.run(job), 0.0, 0.0)
    finally:
        adapter.close()


SPOOF_LINE = "AORLTA_TIMING " + json.dumps(timing_record(cand_us=steady(1.0), ref_us=steady(10.0)))
TIMED = [
    ("faster", proc(), timing_record(cand_us=steady(8.0), ref_us=steady(10.0)), ("clean", None)),
    ("slower", proc(), timing_record(cand_us=steady(12.0), ref_us=steady(10.0)), ("fail", "not_faster")),
    ("itself", proc(), timing_record(cand_us=steady(10.0), ref_us=steady(10.0)), ("fail", "not_faster")),
    ("within_noise", proc(), timing_record(cand_us=steady(9.0), ref_us=steady(10.0)),
     ("unsettled", "timing_within_noise")),
    ("noisy", proc(), timing_record(cand_us=[100.0] * 5 + [5.0, 15.0] * 15, ref_us=steady(10.0)),
     ("unsettled", "timing_noisy")),
    ("too_few", proc(), timing_record(cand_us=steady(8.0, 20), ref_us=steady(10.0, 20)),
     ("unsettled", "timing_too_few")),
    ("timeout", proc(timed_out=True, rc=-9), None, ("unsettled", "timing_timeout")),
    ("crash", proc(rc=-11), None, ("unsettled", "timing_crash")),
    ("crash_after_ok", proc(rc=-11), timing_record(cand_us=steady(8.0), ref_us=steady(10.0)),
     ("unsettled", "no_result")),
    ("silent", proc(rc=0, stdout="nothing"), None, ("unsettled", "no_result")),
    ("stdout_is_not_read", proc(rc=0, stdout=SPOOF_LINE), None, ("unsettled", "no_result")),
    ("wrong_output", proc(), timing_record("candidate_wrong_output"), ("unsettled", "timing_wrong_output")),
    ("reference_wrong", proc(), timing_record("reference_wrong_output"), ("unsettled", "reference_wrong_output")),
    ("no_kernel", proc(), timing_record("candidate_no_kernel"), ("unsettled", "timing_no_kernel")),
    ("hip_error", proc(), timing_record("hip_error"), ("unsettled", "timing_hip_error")),
    ("too_slow", proc(), timing_record("too_slow", candidate_us=9e5, reference_us=10.0),
     ("unsettled", "too_slow_to_measure")),
    ("unknown_status", proc(), timing_record("maybe"), ("unsettled", "no_result")),
    ("not_lists", proc(), timing_record(cand_us="8", ref_us=steady(10.0)), ("unsettled", "no_result")),
    ("not_started", proc(launch_error="OSError"), None, ("unsettled", "run_not_started")),
    ("killed", proc(killed=True), None, ("unsettled", "run_not_started")),
]


@pytest.mark.parametrize("case, run, record, want", TIMED, ids=[c[0] for c in TIMED])
def test_each_timing_outcome_has_one_leg_state(hipk, tmp_path, case, run, record, want):
    from aorlta.reward import timing

    runner = FakeRunner(run, record)
    res = time_job(hipk, tmp_path, runner)
    assert (res.state, res.reason) == want and res.leg == "ref.speedup"
    assert (res.state == "clean") == (case == "faster")  # nothing it could not measure is clean
    rec = timing.measurement(res)
    assert (rec is not None) == (res.state != "unsettled")
    if case == "faster":
        assert rec["speedup"] == 1.25 and rec["samples"] == 30
    assert [c[0] for c in runner.calls] == ["co", "ref", "harness", "run"]
    (_, argv, _, env), = [c for c in runner.calls if c[0] == "run"]
    assert argv[1].endswith("candidate.co") and argv[4:7] == ["5", "30", "20"] and float(argv[7]) == 10000.0
    assert env["HIP_VISIBLE_DEVICES"] == "7"


def test_the_timing_checkers_version_covers_the_references_and_the_judge(hipk, tmp_path, monkeypatch):
    """Cached ref.speedup legs are keyed by the checker's version: a changed reference kernel, or a changed timing
    judge, is a new version."""
    import shutil

    from aorlta.reward import timing

    pool = tmp_path / "pool"
    shutil.copytree(EXAMPLE, pool, ignore=shutil.ignore_patterns("__pycache__"))

    def version():
        ctx = plugins.CheckContext(name="hipk.time", worker_id="time/1", settings={"hipcc": "/x/bin/hipcc", "gpus": [7]},
                                   clock=lambda: 0.0, timeout_s=30.0, pools=(str(pool),))
        a = plugins.CHECK_ADAPTERS["hipk.time"](ctx)
        a.close()
        return a.version

    first = version()
    assert version() == first
    (pool / "reference" / "saxpy.hip").write_text((pool / "reference" / "saxpy.hip").read_text() + "// edited\n")
    second = version()
    judge = tmp_path / "timing.py"
    judge.write_text(Path(timing.__file__).read_text() + "# edited\n")
    monkeypatch.setattr(timing, "__file__", str(judge))
    assert len({first, second, version()}) == 3


def test_every_timed_launch_runs_on_freshly_reset_buffers(hipk):
    """A launch that found the previous launch's outputs could skip its work; each timed launch is preceded by its own
    reset, outside the timed interval."""
    for task in hipk.load_tasks(EXAMPLE):
        src = hipk.timing_source(task)
        body = src[src.index("auto batch = [&](int who)"):]
        body = body[:body.index("\n  };")]
        loop = body[body.index("for (int l = 0; l < launches; ++l) {"):]
        assert loop.index("reset()") < loop.index("hipEventRecord(t0)") < loop.index("launch(who)") < \
            loop.index("hipEventRecord(t1)")


def test_every_launch_runs_on_a_freshly_loaded_module(hipk):
    """A kernel keeps state in a device global across launches of one module, so it could compute on an untimed
    launch and copy the result on the timed ones. Each launch, timed or not, loads its kernel's module again first
    (outside the timed interval), and a load unloads the module before it; one load per batch would still let
    every launch after a batch's first copy what the first kept."""
    for task in hipk.load_tasks(EXAMPLE):
        src = hipk.timing_source(task)
        load = src[src.index("auto load = [&](int who)"):]
        load = load[:load.index("\n  };")]
        assert load.index("hipModuleUnload(mod_[who])") < load.index("hipModuleLoadData(&mod_[who], code_[who].data())")
        body = src[src.index("auto batch = [&](int who)"):]
        body = body[:body.index("\n  };")]
        head, loop = body.split("for (int l = 0; l < launches; ++l) {")
        assert "load(" not in head
        assert loop.count("load(who)") == 1 and loop.count("launch(who)") == 1
        assert loop.index("load(who)") < loop.index("hipEventRecord(t0)") < loop.index("launch(who)")
        fresh = src[src.index("auto fresh = [&](int who)"):]
        fresh = fresh[:fresh.index("\n  };")]
        assert fresh.count("load(who)") == 1 and fresh.index("load(who)") < fresh.index("launch(who)")
        # the kernel is launched nowhere else
        rest = src.replace(loop, "").replace(fresh, "")
        assert "launch(who)" not in rest.replace("auto launch = [&](int who)", "")


def test_every_timed_launchs_outputs_are_checked_after_its_end_event(hipk):
    """Resetting the buffers leaves a kernel's own device globals alone, so a kernel could do its work only on the
    launches that are checked: each timed launch's outputs and guards are checked after its end event (outside its
    timed interval) by the harness's check kernel, every array of the task, and a mismatch is the kernel's wrong
    output, not a hip error."""
    for task in hipk.load_tasks(EXAMPLE):
        src = hipk.timing_source(task)
        body = src[src.index("auto batch = [&](int who)"):]
        loop = body[body.index("for (int l = 0; l < launches; ++l) {"):body.index("\n  };")]
        assert loop.index("hipEventSynchronize(t1)") < loop.index("hipEventElapsedTime") < loop.index("checked()") \
            < loop.index("total += ms")
        assert "if (c != 0) return c < 0 ? -1.0f : -2.0f;" in loop
        check = src[src.index("auto checked = [&]()"):src.index("auto fresh = [&]")]
        launched = re.findall(r"aorlta_check_, dim3\(1024\), dim3\(256\), 0, 0, d_(\w+), (\w+), .*?\.size\(\), (\d),",
                              check)
        # a read-only array bit for bit against its pristine copy (as hipk.run checks it); an output against the
        # host reference's
        assert launched == [(a["name"], f"p_{a['name']}", "1") if a["role"] == "in"
                            else (a["name"], f"e_{a['name']}", "0") for a in task["arrays"]], task["id"]
        verify = src[src.index("auto verify = [&]()"):src.index("auto checked = [&]()")]
        for a in task["arrays"]:
            memcmp = f"std::memcmp(got.data(), h_{a['name']}.data(), ({a['size']}) * sizeof(float)) != 0"
            assert (memcmp in verify) == (a["role"] == "in"), (task["id"], a["name"])
        failure = src[src.index("auto timed_failure"):]
        assert 'us < -1.5f ? (who ? "reference_wrong_output" : "candidate_wrong_output") : "hip_error"' in failure
        rounds = src[src.index("for (int r = 0; r < warmup + repeats; ++r)"):]
        assert "if (us < 0.0f) return timed_failure(us, who);" in rounds
        assert "if (pc < 0.0f) return timed_failure(pc, 0);" in src and "if (pr < 0.0f) return timed_failure(pr, 1);" \
            in src


@pytest.mark.skipif(not Path(HIPCC).exists(), reason=f"no hipcc at {HIPCC} (HIPCC)")
def test_every_timing_harness_compiles(hipk, tmp_path):
    """The timing harness of each task, at hipk-speed.yaml's timing sizes, CPU only: nothing is run."""
    from aorlta.config import parse_yaml

    scalars = parse_yaml((EXAMPLE / "hipk-speed.yaml").read_bytes())["leg_settings"]["ref.speedup"]["scalars"]
    for task in hipk.load_tasks(EXAMPLE):
        src = tmp_path / f"{task['id']}.hip"
        src.write_text(hipk.timing_source(task, {k: str(v) for k, v in scalars.items()}))
        p = subprocess.run([HIPCC, "--offload-arch=gfx950", "-O2", "-o", str(tmp_path / task["id"]), str(src)],
                           capture_output=True, text=True)
        assert p.returncode == 0, (task["id"], p.stderr[-2000:])


# Correct when it is checked, idle otherwise: each block counts its own launches in a device global, which resetting
# the buffers does not touch, and computes only on launches 1, 22, 43, ... of one loaded module: with 20 launches per
# timed batch, the ones the harness checked before each round's batch, had it loaded the module only once.
STATEFUL_VECTOR_ADD = """#include <hip/hip_runtime.h>
__device__ unsigned calls_[1 << 20];
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
  __shared__ unsigned seen;
  if (threadIdx.x == 0) seen = calls_[blockIdx.x]++;
  __syncthreads();
  if (seen % 21 != 0) return;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}
"""


@pytest.mark.skipif(not os.environ.get("AORLTA_HIPK_GPU") or not os.environ.get("SLURM_JOB_ID")
                    or not Path(HIPCC).exists(),
                    reason="GPU test: runs only in a Slurm job that names a GPU in AORLTA_HIPK_GPU")
def test_on_a_gpu_a_kernel_idle_on_unchecked_launches_never_times_as_faster(hipk, tmp_path):
    """The stateful kernel above passes the correctness check (it computes on its first launch). Each launch now
    loads its module afresh, so its counter starts at zero every time and it computes on every launch (measured as
    what it is); were its state to survive, the check after each timed launch would make it wrong output. Never a
    speedup either way. The reference against itself is the control: measured, not faster."""
    gpu = os.environ["AORLTA_HIPK_GPU"]
    cand = tmp_path / "vector_add.hip"
    cand.write_text(STATEFUL_VECTOR_ADD)
    rows = []
    for extra in (["--candidate", str(cand)], []):
        p = subprocess.run([sys.executable, str(EXAMPLE / "hip_kernels.py"), "timecheck", "--task", "vector_add",
                            "--hipcc", HIPCC, "--gpu", gpu, *extra], capture_output=True, text=True,
                           env=dict(os.environ, TMPDIR=str(tmp_path)))
        assert p.returncode == 0, p.stderr[-2000:]
        rows.append(json.loads(p.stdout.splitlines()[-1]))
    assert (rows[0]["state"], rows[0]["reason"]) in {("unsettled", "timing_wrong_output"), ("fail", "not_faster")}, \
        rows[0]
    assert (rows[1]["state"], rows[1]["reason"]) == ("fail", "not_faster"), rows[1]


def test_the_timing_checkers_version_names_where_it_measures(hipk, monkeypatch):
    """A speedup is a fact about one node's GPUs: another host, or other GPUs, is another version, so a resumed run's
    cached leg and reward entries (whose keys include the leg's) are not reused there."""
    import socket

    def version(gpus):
        ctx = plugins.CheckContext(name="hipk.time", worker_id="time/1", settings={"hipcc": "/x/bin/hipcc", "gpus": gpus},
                                   clock=lambda: 0.0, timeout_s=30.0, pools=(str(EXAMPLE),))
        a = plugins.CHECK_ADAPTERS["hipk.time"](ctx)
        a.close()
        return a.version

    monkeypatch.setattr(socket, "gethostname", lambda: "node-a")
    a7 = version([7])
    assert version([7]) == a7 and version([6]) != a7
    monkeypatch.setattr(socket, "gethostname", lambda: "node-b")
    assert version([7]) != a7


def test_timecheck_resolves_a_relative_candidate_and_pool(hipk, tmp_path, monkeypatch):
    """The check runs hipcc from its own directory, so the paths it is given must not be relative."""
    seen = []

    class Recording:
        def __init__(self, ctx):
            seen.append(ctx.pools)
            self.ctx = ctx

        def run(self, job):
            seen.append(job.inputs["source"])
            from aorlta.reward.adapters.base import unsettled

            return unsettled(job, "adapter_error", t_start=0.0, t_end=0.0)

        def close(self):
            pass

    monkeypatch.setattr(hipk, "HipkTime", Recording)
    (tmp_path / "mine.hip").write_text("x")
    monkeypatch.chdir(tmp_path)
    rel_pool = os.path.relpath(EXAMPLE, tmp_path)
    assert hipk.timecheck(["--task", "saxpy", "--candidate", "mine.hip", "--pool", rel_pool]) == 0
    assert seen == [(str(EXAMPLE),), str(tmp_path / "mine.hip")]


def test_the_reference_and_the_harness_are_built_once_per_worker(hipk, tmp_path):
    """The candidate is compiled for every job; the reference's code object and the harness, per task and settings."""
    from aorlta.reward.adapters.base import checked_result
    from aorlta.schemas import LegJob

    runner = FakeRunner(proc(), timing_record(cand_us=steady(8.0), ref_us=steady(10.0)))
    ctx = plugins.CheckContext(name="hipk.time", worker_id="time/1", settings={"hipcc": "/x/bin/hipcc", "gpus": [7]},
                               clock=lambda: 0.0, timeout_s=30.0, pools=(str(EXAMPLE),))
    adapter = plugins.CHECK_ADAPTERS["hipk.time"](ctx)
    adapter.runner = runner
    task = next(t for t in hipk.load_tasks(EXAMPLE) if t["id"] == "vector_add")
    src = tmp_path / "source.hip"
    src.write_text("x")

    try:
        for i in range(2):
            (tmp_path / f"o{i}").mkdir()
            job = LegJob(job_id=f"j{i}", leg="ref.speedup", candidate_id="c", control_id=None, source_sha256="s" * 64,
                         task_sha256=hipk.task_sha256(task), leg_settings={"ratio": 1.1, "warmup": 5, "repeats": 30},
                         inputs={"source": str(src)}, output_dir=str(tmp_path / f"o{i}"))
            assert checked_result(job, adapter.run(job), 0.0, 0.0).state == "clean"
    finally:
        adapter.close()
    assert [c[0] for c in runner.calls] == ["co", "ref", "harness", "run", "co", "run"]


@pytest.mark.parametrize("which, answer", [
    ("co", proc(rc=1, stderr="error: x")), ("co", proc(timed_out=True)), ("ref", proc(rc=1)),
    ("ref", proc(launch_error="OSError")), ("harness", proc(rc=1, stderr="ld.lld: error")),
    ("harness", proc(killed=True)),
])
def test_a_harness_that_does_not_build_is_unsettled(hipk, tmp_path, which, answer):
    runner = FakeRunner(proc(), timing_record(cand_us=steady(8.0), ref_us=steady(10.0)), **{which: answer})
    res = time_job(hipk, tmp_path, runner)
    assert (res.state, res.reason) == ("unsettled", "harness_build_failed")
    assert "run" not in [c[0] for c in runner.calls]


@pytest.mark.parametrize("settings", [{"ratio": 0}, {"repeats": 2}, {"launches": 0}, {"gpu_budget_s": 0},
                                      {"max_rel_mad": 2.0}, {"scalars": ["n"]}, {"scalars": {"n": 1.5}}])
def test_settings_that_cannot_decide_are_an_adapter_error(hipk, tmp_path, settings):
    runner = FakeRunner(proc(), timing_record(cand_us=steady(8.0), ref_us=steady(10.0)))
    res = time_job(hipk, tmp_path, runner, settings=settings)
    assert (res.state, res.reason) == ("unsettled", "adapter_error") and not runner.calls


def test_a_task_without_its_reference_is_an_adapter_error(hipk, tmp_path):
    pool = tmp_path / "pool"
    pool.mkdir()
    (pool / "tasks.json").write_text((EXAMPLE / "tasks.json").read_text())
    runner = FakeRunner(proc(), timing_record(cand_us=steady(8.0), ref_us=steady(10.0)))
    res = time_job(hipk, tmp_path, runner, pools=(str(pool),))
    assert (res.state, res.reason) == ("unsettled", "adapter_error") and not runner.calls


def test_the_timing_harness_holds_none_of_the_candidates_code(hipk):
    """Built from the task alone: both kernels come from code objects named on its command line, and its verdict goes
    to the record file."""
    for task in hipk.load_tasks(EXAMPLE):
        src = hipk.timing_source(task)
        assert 'std::fopen(argv[1 + who], "rb")' in src and " printf(" not in src and "g_path = argv[3];" in src
        assert f"__builtin_get_device_side_mangled_name({task['kernel']})" in src and "<<<" not in src


def test_the_timing_sizes_are_the_settings_scalars(hipk, tmp_path):
    task = next(t for t in hipk.load_tasks(EXAMPLE) if t["id"] == "block_sum")
    harness = hipk.timing_source(task, {"n": "(1 << 24) + 3", "rows": "7"})
    assert "  int n = (1 << 24) + 3;" in harness and "  int blocks = (n + 255) / 256;" in harness
    assert "rows" not in harness and "  int n = 100003;" in hipk.timing_source(task)
    runner = FakeRunner(proc(), timing_record(cand_us=steady(8.0), ref_us=steady(10.0)))
    assert time_job(hipk, tmp_path, runner, settings={"scalars": {"n": 4096}}, task_id="block_sum").state == "clean"
    (written,) = tmp_path.glob("out/*/timing.hip")
    assert "  int n = 4096;" in written.read_text()


@pytest.mark.skipif(not Path(HIPCC).exists(), reason=f"no hipcc at {HIPCC} (HIPCC)")
@pytest.mark.parametrize("task_id", ["saxpy", "block_sum", "transpose"])
def test_the_timing_harness_builds_with_the_reference(hipk, tmp_path, task_id):
    """The reference's code object and the harness compile, CPU only: nothing is run."""
    from aorlta.reward.profiles import load_profile

    task = next(t for t in hipk.load_tasks(EXAMPLE) if t["id"] == task_id)
    scalars = load_profile("hipk-speed").leg_settings["ref.speedup"]["scalars"]
    (tmp_path / "timing.hip").write_text(hipk.timing_source(task, scalars))
    ref = str(EXAMPLE / "reference" / f"{task_id}.hip")
    for argv in (["--genco", ref, "-o", "ref.co"], ["timing.hip", "-o", "timing"]):
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
    assert {p.adapter for p in profile.pools.values()} == {"hipk.gate", "standin.hipcc", "hipk.run", "hipk.time"}
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
