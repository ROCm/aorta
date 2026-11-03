"""Unit tests for the FpSan reference/candidate driver (scripts/sanitizers/fpsan_compare.py).

No GPU, Triton or torch: the end-to-end tests run the real driver over a stub
``launch.py`` with stub ``torch`` and ``triton`` packages on ``--pkg-src``.
Each stub launch writes one object into ``TRITON_CACHE_DIR`` the way Triton's
cache does, with or without FpSan's TTGIR ops.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from aorta.instrumentation.rocjitsu_sanitizers import evaluate_fpsan_output, parse_fpsan_result
from aorta.instrumentation.rocjitsu_sanitizers.execution import ProcessResult
from aorta.instrumentation.rocjitsu_sanitizers.models import Verdict

_REPO = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO / "scripts" / "sanitizers" / "fpsan_compare.py"


def _load():
    spec = importlib.util.spec_from_file_location("fpsan_compare", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["fpsan_compare"] = module
    spec.loader.exec_module(module)
    return module


fc = _load()


# ------------------------------------------------------------ result line


def test_the_result_line_is_the_one_line_the_fpsan_contract_parses():
    result = {"reference_digest": "a" * 64, "candidate_digest": "b" * 64,
              "reference_instrumented": True, "candidate_instrumented": True, "target": "gfx950"}
    line = fc.result_line(result)
    assert line.startswith("AORTA_FPSAN_RESULT {")
    parsed = parse_fpsan_result(f"[candidate] noise\n{line}\n")
    assert (parsed.reference_digest, parsed.candidate_digest) == ("a" * 64, "b" * 64)
    assert parsed.instrumented and parsed.target == "gfx950"


# ------------------------------------------------------------ reference spec


def _arg(name: str, kind: str, value=None, **extra) -> dict:
    return {"name": name, "kind": kind, "value": value, **extra}


def _entry(kernel: str = "k", *, num_warps: int = 8, num_stages: int = 2, args=None, **launch) -> dict:
    args = args if args is not None else [
        _arg("x", "tensor", dtype="fp32", shape=[4], init="randn"),
        _arg("out", "tensor", dtype="fp32", shape=[4]),
        _arg("BLOCK", "constexpr", 128),
    ]
    return {"kernel": kernel, "role": "launched", "scenarios": {"S0": {"launch": {
        "grid": [4, 1, 1], "num_warps": num_warps, "num_stages": num_stages, "args": args, **launch,
    }}}}


def test_the_default_reference_is_the_same_kernel_at_a_conservative_launch():
    entries = [_entry(amd_kwargs={"waves_per_eu": 2})]
    spec, kernel, changes = fc.reference_spec(entries, kernel="k", scenario="S0")
    assert kernel == "k"
    assert changes == ["num_warps 8 -> 4", "num_stages 2 -> 1", "amd_kwargs dropped"]
    launch = spec[0]["scenarios"]["S0"]["launch"]
    assert (launch["num_warps"], launch["num_stages"], "amd_kwargs" in launch) == (4, 1, False)
    # The candidate's spec is left as it was.
    assert entries[0]["scenarios"]["S0"]["launch"]["num_warps"] == 8


def test_constexpr_and_grid_overrides_change_only_the_reference():
    entries = [_entry()]
    spec, _, changes = fc.reference_spec(
        entries, kernel="k", scenario="S0", num_warps=8, num_stages=2,
        constexprs={"BLOCK": 64}, grid=[8, 1, 1],
    )
    assert changes == ["BLOCK 128 -> 64", "grid [4, 1, 1] -> [8, 1, 1]"]
    launch = spec[0]["scenarios"]["S0"]["launch"]
    assert launch["args"][2]["value"] == 64 and launch["grid"] == [8, 1, 1]
    assert entries[0]["scenarios"]["S0"]["launch"]["args"][2]["value"] == 128


def test_another_spec_entry_can_be_the_reference():
    entries = [_entry("k"), _entry("k@small", num_warps=4)]
    spec, kernel, changes = fc.reference_spec(entries, kernel="k", scenario="S0", reference_kernel="k@small")
    assert kernel == "k@small" and changes == ["kernel k -> k@small"]
    assert spec[0]["kernel"] == "k@small"
    assert spec[0]["scenarios"]["S0"]["launch"]["num_warps"] == 4


@pytest.mark.parametrize(
    ("entries", "kwargs", "reason"),
    [
        pytest.param([_entry(num_warps=4, num_stages=1)], {}, "fpsan_reference_identical", id="identical"),
        pytest.param([_entry()], {"constexprs": {"BLOCK_K": 64}}, "fpsan_reference_invalid", id="unknown-constexpr"),
        pytest.param([_entry(args=[_arg("out", "tensor", dtype="fp32", shape=[4]),
                                   _arg("SPLIT_K", "constexpr", 4)])], {},
                     "fpsan_reference_invalid", id="split-k"),
        pytest.param([_entry(args=[_arg("table", "tensor", dtype="i64", shape=[2], init="ptr_of:a,b")])], {},
                     "fpsan_pointer_table", id="pointer-table"),
        pytest.param([_entry()], {"scenario": "S9"}, "fpsan_spec_mismatch", id="no-scenario"),
        pytest.param([_entry()], {"reference_kernel": "missing"}, "fpsan_spec_mismatch", id="no-reference"),
    ],
)
def test_a_reference_that_would_compare_nothing_is_refused(entries, kwargs, reason):
    kwargs = {"kernel": "k", "scenario": "S0", **kwargs}
    with pytest.raises(fc.CompareError, match=f"^{reason}:"):
        fc.reference_spec(entries, **kwargs)


# ------------------------------------------------------------ instrumentation


def _cache_object(cache: Path, name: str, *, mode="fpsan", arch="gfx950", ops=2) -> None:
    entry = cache / f"HASH_{name}"
    entry.mkdir(parents=True, exist_ok=True)
    (entry / f"{name}.hsaco").write_bytes(b"\x7fELF")
    (entry / f"{name}.json").write_text(json.dumps({
        "name": name, "instrumentation_mode": mode, "triton_version": "3.8.0",
        "target": {"backend": "hip", "arch": arch, "warp_size": 64},
    }), encoding="utf-8")
    (entry / f"{name}.ttgir").write_text(
        "tt.func @k() {\n" + '  %0 = "tti.experimental_fpsan_embed"(%x)\n' * ops + "}\n", encoding="utf-8"
    )


def test_an_object_is_instrumented_only_with_the_mode_an_fpsan_target_and_its_ttgir_ops(tmp_path):
    _cache_object(tmp_path / "ok", "k")
    checked = fc.instrumentation(tmp_path / "ok")
    assert checked["instrumented"] is True
    assert checked["objects"][0]["fpsan_ttgir_ops"] == 2
    _cache_object(tmp_path / "multi", "k", mode="consan,fpsan", arch="gfx942:sramecc+:xnack-")
    assert fc.instrumentation(tmp_path / "multi")["instrumented"] is True

    # Triton 3.6 and 3.7 record the mode but run no FpSan pass on AMD.
    _cache_object(tmp_path / "mode-only", "k", ops=0)
    assert fc.instrumentation(tmp_path / "mode-only")["instrumented"] is False
    _cache_object(tmp_path / "plain", "k", mode="")
    assert fc.instrumentation(tmp_path / "plain")["instrumented"] is False
    _cache_object(tmp_path / "arch", "k", arch="gfx90a")
    assert fc.instrumentation(tmp_path / "arch")["instrumented"] is False
    assert fc.instrumentation(tmp_path / "empty")["instrumented"] is False
    # One uninstrumented object makes the whole side uninstrumented.
    _cache_object(tmp_path / "mixed", "k")
    _cache_object(tmp_path / "mixed", "helper", ops=0)
    assert fc.instrumentation(tmp_path / "mixed")["instrumented"] is False


# ------------------------------------------------------------ end to end

_STUB_TORCH = '''
class _Dtype:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"torch.{self.name}"


float32, uint8 = _Dtype("float32"), _Dtype("uint8")


class Tensor:
    def __init__(self, values, dtype=float32):
        self.values, self.dtype = list(values), dtype

    @property
    def shape(self):
        return (len(self.values),)

    def detach(self):
        return self

    contiguous = cpu = numpy = detach

    def reshape(self, *shape):
        return self

    def view(self, dtype):
        return self

    def tobytes(self):
        return repr(self.values).encode()


class cuda:
    @staticmethod
    def synchronize():
        pass
'''

_STUB_JIT = '''
class KernelInterface:
    def __getitem__(self, grid):
        return lambda *args, **kwargs: self.run(*args, grid=grid, **kwargs)


class JITFunction(KernelInterface):
    def __init__(self, fn):
        self.fn = fn

    def run(self, *args, grid, **kwargs):
        return self.fn(*args, **kwargs)
'''

# The launch fields `launches`, `ops` and `skip_load` steer the stub; the real
# harness ignores keys it does not know.
_STUB_LAUNCH = '''
import argparse, json, os
from pathlib import Path

import torch
from triton.runtime.jit import JITFunction


def _kernel(x, out, SCALE=1):
    out.values = [v * SCALE for v in x.values]


def load_kernel(flat_file, name):
    return None, JITFunction(_kernel)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("spec")
    parser.add_argument("--kernel")
    parser.add_argument("--scenario")
    parser.add_argument("--flat-dir")
    args, _ = parser.parse_known_args()
    entry = next(e for e in json.load(open(args.spec)) if e["kernel"] == args.kernel)
    launch = entry["scenarios"][args.scenario]["launch"]
    consts = {a["name"]: a["value"] for a in launch["args"] if a["kind"] == "constexpr"}
    fn = JITFunction(_kernel) if launch.get("skip_load") else load_kernel(Path(args.flat_dir), args.kernel)[1]
    cache = Path(os.environ["TRITON_CACHE_DIR"]) / "HASH"
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "k.hsaco").write_bytes(b"x")
    (cache / "k.json").write_text(json.dumps({
        "name": "k", "instrumentation_mode": os.environ.get("TRITON_INSTRUMENTATION_MODE", ""),
        "target": {"backend": "hip", "arch": "gfx950", "warp_size": 64},
    }))
    (cache / "k.ttgir").write_text("tti.experimental_fpsan_embed\\n" * launch.get("ops", 1))
    x, out = torch.Tensor([1.0, 2.0, 3.0]), torch.Tensor([0.0, 0.0, 0.0])
    for _ in range(launch.get("launches", 1)):
        fn[tuple(launch["grid"])](x, out, SCALE=consts.get("SCALE", 1))
    print("stub launch ran")
'''


def _stub_asset(tmp_path: Path, *, launch_py: str = _STUB_LAUNCH, **launch) -> Path:
    model = tmp_path / "model"
    harness = model / "roofline" / "harness"
    harness.mkdir(parents=True)
    (model / "roofline" / "specs").mkdir()
    (model / "kernels").mkdir()
    pkg = model / "pkg_src"
    (pkg / "torch").mkdir(parents=True)
    (pkg / "torch" / "__init__.py").write_text(_STUB_TORCH, encoding="utf-8")
    (pkg / "triton" / "runtime").mkdir(parents=True)
    (pkg / "triton" / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "triton" / "runtime" / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "triton" / "runtime" / "jit.py").write_text(_STUB_JIT, encoding="utf-8")
    (harness / "launch.py").write_text(launch_py, encoding="utf-8")
    entry = {"kernel": "k", "role": "launched", "scenarios": {"S0": {"launch": {
        "grid": [1, 1, 1], "num_warps": 8, "num_stages": 2,
        "args": [_arg("SCALE", "constexpr", 1)], **launch,
    }}}}
    (model / "roofline" / "specs" / "toy.json").write_text(json.dumps([entry]), encoding="utf-8")
    return model


def _compare(tmp_path: Path, model: Path, *extra: str) -> tuple[subprocess.CompletedProcess, dict]:
    record = tmp_path / "record.json"
    proc = subprocess.run(
        [sys.executable, str(_SCRIPT),
         "--launch", str(model / "roofline" / "harness" / "launch.py"),
         "--spec", str(model / "roofline" / "specs" / "toy.json"),
         "--kernel", "k", "--scenario", "S0", "--flat-dir", str(model / "kernels"),
         "--pkg-src", str(model / "pkg_src"), "--work", str(tmp_path / "work"),
         "--record", str(record), "--side-timeout", "60", *extra],
        capture_output=True, text=True, timeout=120, check=False,
    )
    return proc, json.loads(record.read_text(encoding="utf-8"))


def _verdict(proc: subprocess.CompletedProcess):
    return evaluate_fpsan_output(
        ProcessResult(argv=("fpsan.sh",), returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr),
        sanitizer="fpsan", target="gfx950",
    )


def test_a_hook_that_fires_once_prints_one_result_line_and_passes(tmp_path):
    proc, record = _compare(tmp_path, _stub_asset(tmp_path))
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.count("AORTA_FPSAN_RESULT") == 1
    assert "stub launch ran" not in proc.stdout
    assert "[candidate] stub launch ran" in proc.stderr
    check = _verdict(proc)
    assert check.verdict is Verdict.PASS
    assert record["reference"] == {"kernel": "k", "changes": ["num_warps 8 -> 4", "num_stages 2 -> 1"]}
    assert record["differing_tensors"] == []
    for side in ("reference", "candidate"):
        assert record["sides"][side]["launches"] == 1
        assert record["sides"][side]["load_kernel_calls"] == 1
        assert [t["name"] for t in record["sides"][side]["tensors"]] == ["arg0", "arg1"]
    reference_spec = json.loads((tmp_path / "work" / "reference-spec.json").read_text())
    assert reference_spec[0]["scenarios"]["S0"]["launch"]["num_warps"] == 4


def test_outputs_that_differ_print_the_mismatch_and_name_the_tensor(tmp_path):
    proc, record = _compare(tmp_path, _stub_asset(tmp_path), "--reference-constexpr", "SCALE=2")
    assert proc.returncode == 0, proc.stderr
    assert _verdict(proc).verdict is Verdict.FAIL
    assert record["differing_tensors"] == ["arg1"]
    assert "digests differ in ['arg1']" in proc.stderr


def test_a_side_without_fpsan_ops_is_reported_uninstrumented(tmp_path):
    proc, record = _compare(tmp_path, _stub_asset(tmp_path, ops=0))
    result = parse_fpsan_result(proc.stdout)
    assert (result.reference_instrumented, result.candidate_instrumented) == (False, False)
    check = _verdict(proc)
    assert check.verdict is Verdict.ERROR
    assert str(check.reason).startswith("fpsan_not_instrumented:")
    assert record["sides"]["candidate"]["cache"]["objects"][0]["instrumentation_mode"] == "fpsan"


_SIDE_REFUSED = f"fpsan_reference_failed: exit {fc.EXIT_NO_RESULT}: fpsan_compare: "


@pytest.mark.parametrize(
    ("launch", "launch_py", "error"),
    [
        pytest.param({}, _STUB_LAUNCH.replace("def load_kernel", "def load_kernel_v2"),
                     f"{_SIDE_REFUSED}fpsan_hook_missing:", id="load_kernel-renamed"),
        pytest.param({"skip_load": True}, _STUB_LAUNCH, f"{_SIDE_REFUSED}fpsan_hook_not_fired:",
                     id="load_kernel-never-called"),
        pytest.param({"launches": 0}, _STUB_LAUNCH, f"{_SIDE_REFUSED}fpsan_launch_count: launch.py launched 0",
                     id="never-launched"),
        pytest.param({"launches": 2}, _STUB_LAUNCH, f"{_SIDE_REFUSED}fpsan_launch_count: launch.py launched 2",
                     id="launched-twice"),
        pytest.param({}, _STUB_LAUNCH.replace('print("stub launch ran")', "raise SystemExit(4)"),
                     "fpsan_reference_failed: exit 4", id="harness-exits"),
    ],
)
def test_a_hook_that_did_not_fire_exactly_once_is_an_error_never_a_pass(tmp_path, launch, launch_py, error):
    proc, record = _compare(tmp_path, _stub_asset(tmp_path, launch_py=launch_py, **launch))
    assert proc.returncode == fc.EXIT_NO_RESULT
    assert "AORTA_FPSAN_RESULT" not in proc.stdout
    assert record["error"].startswith(error)
    check = _verdict(proc)
    assert check.verdict is Verdict.ERROR
    assert str(check.reason).startswith(f"fpsan_command_exit_{fc.EXIT_NO_RESULT}")


def test_a_reference_identical_to_the_candidate_never_runs(tmp_path):
    proc, record = _compare(tmp_path, _stub_asset(tmp_path, num_warps=4, num_stages=1))
    assert proc.returncode == fc.EXIT_NO_RESULT and not proc.stdout
    assert record["error"].startswith("fpsan_reference_identical:")
    assert not (tmp_path / "work" / "reference").exists()
