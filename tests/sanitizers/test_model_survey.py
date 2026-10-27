"""Unit tests for the open-source model-kernel survey (scripts/sanitizers/model_survey.py)."""

from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import importlib.util
import io
import json
import re
import signal
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path

import pytest
import yaml

from aorta.instrumentation.rocjitsu_sanitizers.models import SanitizerReport
from aorta.instrumentation.rocjitsu_sanitizers.recipe import load_sanitizer_recipe

_REPO = Path(__file__).resolve().parents[2]
_MODELS = _REPO / "recipes" / "sanitizers" / "models"


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, _REPO / rel)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ms = _load("model_survey", "scripts/sanitizers/model_survey.py")
gen = _load("gen_sanitizer_dashboard_for_models", "scripts/sanitizers/gen_sanitizer_dashboard.py")
comparator = _load("compare_verdict_baselines_for_models", "scripts/sanitizers/compare_verdict_baselines.py")


# --------------------------------------------------------------- inventories


def _inventories() -> list[Path]:
    return sorted(_MODELS.glob("*.yaml"))


def test_every_committed_inventory_loads_and_names_the_pinned_asset():
    assert [p.stem for p in _inventories()] == ["gpt-oss", "qwen3"]
    for path in _inventories():
        inv = ms.load_inventory(path)
        assert inv.model == path.stem
        assert inv.asset_version == ms.ASSET_VERSION
        assert inv.roles == {"launched", "conditional"}


def test_dashboard_and_workflow_name_exactly_the_committed_models():
    models = {p.stem for p in _inventories()}
    assert set(gen._MODEL_SURVEY_GROUPS) == models
    workflow = (_REPO / ".github" / "workflows" / "sanitizers-nightly.yml").read_text(encoding="utf-8")
    listed = re.search(r'^\s*MODELS="([^"]+)"', workflow, re.MULTILINE)
    assert listed and set(listed.group(1).split()) == models


def test_the_asset_url_names_the_asset_version():
    assert f"model-kernels-v{ms.ASSET_VERSION}/" in ms.ASSET_URL
    assert ms.ASSET_URL.endswith(f"aorta-model-kernels-v{ms.ASSET_VERSION}.tar.gz")
    assert re.fullmatch(r"[0-9a-f]{64}", ms.ASSET_SHA256)


def _write_inventory(tmp_path: Path, name: str = "toy", **overrides) -> Path:
    doc = {
        "schema_version": 1,
        "model": name,
        "asset_version": "1",
        "specs": ["toy.json"],
        "roles": ["launched"],
        "expected_pairs": 3,
        "consan": {"timeout_seconds": 60, "exclude": {}},
    }
    doc.update(overrides)
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "overrides",
    [
        {"schema_version": 2},
        {"surprise": 1},
        {"model": "other"},
        {"model": "../toy"},
        {"expected_pairs": 0},
        {"expected_pairs": True},
        {"specs": []},
        {"roles": "launched"},
        {"asset_version": 1},
        {"consan": {"timeout_seconds": 60, "exclude": {"k": ""}}},
        {"consan": {"timeout_seconds": 60, "retries": 2}},
        {"consan": {"timeout_seconds": -1}},
    ],
)
def test_a_malformed_inventory_is_refused(tmp_path, overrides):
    with pytest.raises(ms.SurveyError):
        ms.load_inventory(_write_inventory(tmp_path, **overrides))


# ---------------------------------------------------------------- the asset


def _toy_model(root: Path, entries: list[dict]) -> Path:
    model = root / "toy"
    (model / "roofline" / "specs").mkdir(parents=True)
    (model / "roofline" / "harness").mkdir(parents=True)
    (model / "roofline" / "harness" / "launch.py").write_text("", encoding="utf-8")
    (model / "roofline" / "specs" / "toy.json").write_text(json.dumps(entries), encoding="utf-8")
    return model


_ENTRIES = [
    {"kernel": "a", "role": "launched", "scenarios": {"S0": {}, "S1": {}}},
    {"kernel": "dead", "role": "dead-for-toy", "scenarios": {"S0": {}}},
    {"kernel": "b", "role": "launched", "scenarios": {"S0": {}}},
    {"kernel": "", "role": "launched", "scenarios": {"S0": {}}},
]


def test_pairs_follow_spec_order_and_keep_only_inventory_roles(tmp_path):
    inv = ms.load_inventory(_write_inventory(tmp_path))
    pairs = ms.model_pairs(inv, _toy_model(tmp_path, _ENTRIES))
    assert [(p.kernel, p.scenario) for p in pairs] == [("a", "S0"), ("a", "S1"), ("b", "S0")]


def test_a_pair_count_that_drifted_from_the_inventory_is_an_error(tmp_path):
    inv = ms.load_inventory(_write_inventory(tmp_path, expected_pairs=4))
    with pytest.raises(ms.SurveyError, match="expects 4"):
        ms.model_pairs(inv, _toy_model(tmp_path, _ENTRIES))


def _manifest(root: Path, *, version: str = "1") -> None:
    files = {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }
    (root / "MANIFEST.json").write_text(
        json.dumps({"schema": ms.ASSET_SCHEMA, "version": version, "files": files}), encoding="utf-8"
    )


def test_check_asset_accepts_a_matching_manifest(tmp_path):
    _toy_model(tmp_path, _ENTRIES)
    _manifest(tmp_path)
    inv = ms.load_inventory(_write_inventory(tmp_path / "..", name="toy"))
    assert ms.check_asset(tmp_path, inv) == tmp_path / "toy"


def test_check_asset_refuses_a_version_or_file_that_does_not_match(tmp_path):
    model = _toy_model(tmp_path, _ENTRIES)
    inv = ms.load_inventory(_write_inventory(tmp_path / "..", name="toy"))
    _manifest(tmp_path, version="2")
    with pytest.raises(ms.SurveyError, match="v2"):
        ms.check_asset(tmp_path, inv)
    _manifest(tmp_path)
    (model / "roofline" / "specs" / "toy.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ms.SurveyError, match="does not match"):
        ms.check_asset(tmp_path, inv)


def _tarball(path: Path, members: dict[str, bytes]) -> Path:
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as tar:
        for name, body in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(body)
            tar.addfile(info, io.BytesIO(body))
    path.write_bytes(gzip.compress(raw.getvalue()))
    return path


def test_extract_asset_refuses_members_outside_its_root(tmp_path):
    for bad in ("../escape.txt", "elsewhere/file.txt", "aorta-model-kernels/../x.txt"):
        tarball = _tarball(tmp_path / "bad.tar.gz", {bad: b"x"})
        with pytest.raises(ms.SurveyError, match="not a plain file"):
            ms.extract_asset(tarball, tmp_path / "out")


def test_fetch_asset_refuses_a_download_with_the_wrong_digest(tmp_path):
    tarball = _tarball(tmp_path / "src.tar.gz", {"aorta-model-kernels/README.md": b"hi"})
    with pytest.raises(ms.SurveyError, match="refusing it"):
        ms.fetch_asset(tmp_path / "work", url=tarball.as_uri(), sha256="0" * 64)
    good = hashlib.sha256(tarball.read_bytes()).hexdigest()
    root = ms.fetch_asset(tmp_path / "work", url=tarball.as_uri(), sha256=good)
    assert (root / "README.md").read_text() == "hi"


# ------------------------------------------------------- launches and recipes


def test_cached_object_reads_the_symbol_triton_gave_the_kernel(tmp_path):
    entry = tmp_path / "cache" / "HASH"
    entry.mkdir(parents=True)
    (entry / "_matmul_ogs.hsaco").write_bytes(b"\x7fELF")
    (entry / "_matmul_ogs.json").write_text(json.dumps({"name": "_matmul_ogs_NNT_bf16"}), encoding="utf-8")
    obj, symbol = ms.cached_object(tmp_path / "cache")
    assert obj.name == "_matmul_ogs.hsaco"
    assert symbol == "_matmul_ogs_NNT_bf16"
    (tmp_path / "cache" / "OTHER").mkdir()
    (tmp_path / "cache" / "OTHER" / "second.hsaco").write_bytes(b"x")
    with pytest.raises(ms.SurveyError, match="found 2"):
        ms.cached_object(tmp_path / "cache")


def _warmed(tmp_path: Path, kernel: str = "k", scenario: str = "S0", *, ok: bool = True) -> object:
    item = ms.Warmed(ms.Pair("toy.json", kernel, scenario), tmp_path / f"{kernel}__{scenario}")
    item.case_dir.mkdir(parents=True, exist_ok=True)
    if ok:
        obj = item.case_dir / "dispatched.hsaco"
        obj.write_bytes(kernel.encode())
        item.ok, item.symbol, item.code_object = True, f"{kernel}_sym", obj
        item.sha256 = hashlib.sha256(obj.read_bytes()).hexdigest()
        (item.case_dir / "launch.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    return item


def test_generated_recipes_load_through_the_aorta_recipe_loader(tmp_path):
    items = [_warmed(tmp_path, "a"), _warmed(tmp_path, "b"), _warmed(tmp_path, "c", ok=False)]
    wc = tmp_path / "waitcheck.yaml"
    wc.write_text(ms.waitcheck_recipe("toy", items), encoding="utf-8")
    recipe = load_sanitizer_recipe(wc)
    assert recipe.sanitizers == ("waitcheck",)
    kernels = yaml.safe_load(wc.read_text())["sanitizer_plan"]["source"]["kernels"]
    assert [k["name"] for k in kernels] == ["a_sym", "b_sym"]
    # Whole-object scan: an entry offset would pin waitcheck to one entry.
    assert all("entry_offset" not in k for k in kernels)
    assert all(k["code_object_sha256"] for k in kernels)

    cs = tmp_path / "consan.yaml"
    cs.write_text(ms.consan_recipe("toy", items[0], timeout=60), encoding="utf-8")
    recipe = load_sanitizer_recipe(cs)
    assert recipe.sanitizers == ("consan",)
    assert recipe.consan_driver == "dispatch"


def test_the_shim_runs_the_harness_against_the_case_cache(tmp_path):
    shim = tmp_path / "launch.sh"
    ms.write_shim(
        shim,
        model_dir=tmp_path / "gpt oss",
        pair=ms.Pair("misc.json", "_matmul_ogs@w2", "P1"),
        cache=tmp_path / "cache",
        run_out=tmp_path / "run",
        python="/opt/venv/bin/python",
    )
    text = shim.read_text()
    assert f"export TRITON_CACHE_DIR={tmp_path / 'cache'}" in text
    assert "'" + str(tmp_path / "gpt oss" / "pkg_src") + "'" in text
    assert "--kernel _matmul_ogs@w2 --scenario P1 --mode run --arch gfx950" in text
    assert shim.stat().st_mode & 0o111


def test_on_gpus_never_runs_two_items_on_one_device():
    active: dict[str, int] = {}
    peak: dict[str, int] = {}
    lock = threading.Lock()

    def work(item, gpu):
        with lock:
            active[gpu] = active.get(gpu, 0) + 1
            peak[gpu] = max(peak.get(gpu, 0), active[gpu])
        time.sleep(0.01)
        with lock:
            active[gpu] -= 1
        return item

    assert ms.on_gpus(["0", "1"], list(range(10)), work) == list(range(10))
    assert set(peak) == {"0", "1"} and max(peak.values()) == 1


# --------------------------------------------------- rotation and budget


def test_rotation_starts_one_kernel_later_each_night(tmp_path):
    items = [_warmed(tmp_path, k) for k in ("c", "a", "b")]
    night = dt.date(2026, 10, 22)
    first = [w.pair.kernel for w in ms.rotation(items, night)]
    second = [w.pair.kernel for w in ms.rotation(items, night + dt.timedelta(days=1))]
    assert sorted(first) == ["a", "b", "c"]
    assert second == first[1:] + first[:1]
    assert ms.rotation([], night) == []


def test_budget_admits_a_case_only_while_its_timeout_still_fits():
    now = [0.0]
    budget = ms.Budget(100.0, clock=lambda: now[0])
    assert budget.admit(60)
    now[0] = 40.0
    assert budget.admit(60)
    now[0] = 41.0
    assert not budget.admit(60)
    # Time held for a model that has not run yet counts against the same clock.
    now[0] = 0.0
    assert budget.admit(60, reserve=40)
    assert not budget.admit(60, reserve=41)


# ------------------------------------------------------------------- merge


def _identity(name: str) -> dict:
    return {"name": name, "target": "gfx950", "code_object": f"/w/{name}.hsaco",
            "code_object_sha256": hashlib.sha256(name.encode()).hexdigest(),
            "code_object_index": 0, "entry_offset": None}


def _case_report(name: str, state: str, verdict: str, *, reason=None, findings=0, kernel_result=True) -> dict:
    found = [
        {"sanitizer": "consan", "severity": "race", "code": "sampled_conflict",
         "message": f"conflict reader=4242 in {name}", "kernel_name": name,
         "code_object": None, "entry_offset": None, "metadata": {}}
    ] * findings
    consan = {
        "sanitizer": "consan", "state": state, "verdict": verdict, "reason": reason,
        "returncode": 0, "findings": found, "coverage": [], "backend": {"hook": "hook.so"},
        "kernel_results": (
            [{"identity": _identity(name), "state": state, "verdict": verdict, "findings": found,
              "reason": reason, "returncode": 0, "diagnostics_truncated": False}]
            if kernel_result else []
        ),
    }
    preflight = {"sanitizer": "waitcheck_preflight", "state": "ran", "verdict": "pass", "reason": None,
                 "returncode": 0, "findings": [], "kernel_results": [], "coverage": [], "backend": {}}
    report = SanitizerReport.from_dict(
        {
            "schema": "aorta.sanitizer_report/0.1", "target": "gfx950",
            "worklist": {"schema": "aorta.kernel_worklist/0.1", "requirement": "top_dispatch_count",
                         "top_n": 1, "kernel_count": 1,
                         "kernels": [{"identity": _identity(name), "total_time_ms": 0.0,
                                      "dispatch_count": 1, "sources": ["kernel"]}]},
            "checks": [consan, preflight],
            "overall_verdict": max(verdict, "pass", key=ms._VERDICT_RANK.__getitem__),
            "execution_status": {"ran": "complete", "not_checked": "partial"}.get(state, "error"),
        }
    )
    return report.to_dict()


_NO_SITES = "consan_not_applicable: no LDS access, barrier, atomic or fence sites"


def test_merge_keeps_the_preflight_so_a_no_site_kernel_reads_partial_not_vacuous(tmp_path, monkeypatch, capsys):
    merged = ms.merge_consan([
        _case_report("a", "ran", "pass"),
        _case_report("b", "not_checked", "not_checked", reason=_NO_SITES),
    ])
    assert merged["overall_verdict"] == "not_checked"
    assert merged["execution_status"] == "partial"
    consan, preflight = merged["checks"]
    assert (consan["sanitizer"], consan["state"]) == ("consan", "not_checked")
    assert consan["reason"] == f"1 of 2 kernels not checked (1x {_NO_SITES})"
    assert [kr["verdict"] for kr in consan["kernel_results"]] == ["pass", "not_checked"]
    assert (preflight["sanitizer"], preflight["state"], preflight["verdict"]) == ("waitcheck_preflight", "ran", "pass")
    assert merged["worklist"]["kernel_count"] == 2

    case = tmp_path / "informational" / "consan-toy"
    case.mkdir(parents=True)
    (case / "sanitizer_report.json").write_text(json.dumps(merged), encoding="utf-8")
    monkeypatch.chdir(_REPO)
    assert comparator.main(["prog", "--vacuous-only", "--require", "informational/consan-toy", str(tmp_path)]) == 0


def test_merge_takes_the_worst_verdict_and_keeps_every_finding():
    merged = ms.merge_consan([
        _case_report("a", "ran", "pass"),
        _case_report("b", "ran", "fail", findings=2),
        _case_report("c", "not_checked", "not_checked", reason=_NO_SITES),
    ])
    consan = merged["checks"][0]
    assert (merged["overall_verdict"], merged["execution_status"]) == ("fail", "complete")
    assert (consan["state"], consan["verdict"], consan["reason"]) == ("ran", "fail", None)
    assert len(consan["findings"]) == 2
    assert [kr["verdict"] for kr in consan["kernel_results"]] == ["pass", "fail", "not_checked"]


def test_merge_counts_one_cause_once_whatever_reader_reported_it():
    merged = ms.merge_consan([
        _case_report("a", "error", "error", reason="consan_output_parse_error: reader 111 access site count mismatch"),
        _case_report("b", "error", "error", reason="consan_output_parse_error: reader 222 access site count mismatch"),
        _case_report("c", "error", "error", reason="combined_hook_exit_86", kernel_result=False),
        _case_report("d", "ran", "pass"),
    ])
    consan = merged["checks"][0]
    assert merged["execution_status"] == "error"
    assert consan["reason"] == (
        "3 of 4 kernels did not complete "
        "(2x consan_output_parse_error: access site count mismatch, 1x combined_hook_exit_86)"
    )
    # A hook that wrote no per-kernel result still gets a row for its kernel.
    assert [kr["identity"]["name"] for kr in consan["kernel_results"]] == ["a", "b", "c", "d"]


def test_merge_counts_a_timed_out_kernel_as_not_completed():
    merged = ms.merge_consan([
        _case_report("a", "ran", "pass"),
        _case_report("b", "timed_out", "error", reason="consan_timeout"),
    ])
    consan = merged["checks"][0]
    assert (merged["overall_verdict"], merged["execution_status"]) == ("error", "error")
    assert (consan["state"], consan["reason"]) == (
        "error", "1 of 2 kernels did not complete (1x consan_timeout)"
    )


# --------------------------------------------------------------- dashboard


def test_model_cases_link_their_inventory_and_a_runnable_command():
    for case in ("consan-gpt-oss", "waitcheck-gpt-oss", "consan-qwen3", "waitcheck-qwen3"):
        group = case.split("-", 1)[1]
        assert gen._survey_recipe_for(case) == f"models/{group}"
        assert (_REPO / "recipes" / "sanitizers" / f"models/{group}.yaml").is_file()
        assert gen._survey_command_for(case) == (
            "python scripts/sanitizers/model_survey.py run "
            f"--inventory recipes/sanitizers/models/{group}.yaml --out results"
        )
    assert gen._survey_command_for("waitcheck-gemm") == (
        "aorta sweep run --recipe recipes/sanitizers/daily-waitcheck-gemm-object.yaml"
    )


def test_merged_model_reports_render_one_tab_2_row_per_model(tmp_path):
    root = tmp_path / "informational"
    for case, report in (
        ("consan-qwen3", ms.merge_consan([_case_report("a", "ran", "pass"),
                                          _case_report("b", "not_checked", "not_checked", reason=_NO_SITES)])),
        ("consan-gpt-oss", ms.merge_consan([_case_report("c", "ran", "fail", findings=1)])),
    ):
        (root / case).mkdir(parents=True)
        (root / case / "sanitizer_report.json").write_text(json.dumps(report), encoding="utf-8")
    entries = gen.survey_cases_from_informational_dir(root)
    assert {(e["group"], e["sanitizer"]) for e in entries} == {("gpt-oss", "consan"), ("qwen3", "consan")}
    assert all("model_survey.py run" in e["command"] for e in entries)
    stats = gen._survey_summary_stats(gen._group_survey_entries(entries))
    assert (stats["workloads"], stats["kernels"], stats["runs"]) == (2, 3, 2)


# ------------------------------------------------------------ end to end
#
# These drive the real main() -> run_model() -> warm_one() -> consan_one() path.
# Only two things are stood in for: the asset's launch.py (a stub that refuses any
# path it cannot resolve from the directory the shim cds into, then writes one
# object into TRITON_CACHE_DIR the way Triton does) and the `aorta sweep run`
# boundary, which needs rj_waitcheck and the ConSan hook.

_STUB_LAUNCH = '''
import json, os, sys
from pathlib import Path
spec, rest = Path(sys.argv[1]), sys.argv[2:]
opts = dict(zip(rest[0::2], rest[1::2]))
for path in (spec, Path(opts["--flat-dir"]), Path(os.environ["PYTHONPATH"])):
    if not path.exists():
        print(f"not found from {os.getcwd()}: {path}", file=sys.stderr)
        sys.exit(3)
entry = Path(os.environ["TRITON_CACHE_DIR"]) / "HASH"
entry.mkdir(parents=True, exist_ok=True)
kernel = opts["--kernel"]
(entry / f"{kernel}.hsaco").write_bytes(kernel.encode())
(entry / f"{kernel}.json").write_text(json.dumps({"name": f"{kernel}_sym"}))
'''


def _toy_asset(tmp_path: Path, *, consan_timeout: int = 60) -> None:
    root = tmp_path / "asset" / "aorta-model-kernels"
    model = root / "toy"
    (model / "roofline" / "harness").mkdir(parents=True)
    (model / "roofline" / "specs").mkdir(parents=True)
    (model / "pkg_src").mkdir()
    (model / "kernels").mkdir()
    (model / "roofline" / "harness" / "launch.py").write_text(_STUB_LAUNCH, encoding="utf-8")
    (model / "roofline" / "specs" / "toy.json").write_text(json.dumps([
        {"kernel": "a", "role": "launched", "scenarios": {"S0": {}, "S1": {}}},
        {"kernel": "b", "role": "launched", "scenarios": {"S0": {}}},
        {"kernel": "c", "role": "launched", "scenarios": {"S0": {}}},
    ]), encoding="utf-8")
    _manifest(root)
    (tmp_path / "recipes").mkdir()
    _write_inventory(tmp_path / "recipes", expected_pairs=4,
                     consan={"timeout_seconds": consan_timeout, "exclude": {}})


def _fake_sweep(recipe: Path, out_dir: Path, *, env=None, timeout=None) -> int:
    """Stand-in for `aorta sweep run`: a, pass; b, writes nothing; c, swaps its object."""
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = yaml.safe_load(recipe.read_text())["sanitizer_plan"]
    source = plan["source"]
    if plan["sanitizers"] == ["waitcheck"]:
        # aorta dedups identical objects (both `a` scenarios compile to the same bytes).
        unique = {k["code_object_sha256"]: k for k in source["kernels"]}.values()
        kernels = [{"identity": {**k, "target": "gfx950", "entry_offset": None},
                    "total_time_ms": 0.0, "dispatch_count": 1, "sources": ["kernel"]}
                   for k in unique]
        report = SanitizerReport.from_dict({
            "schema": "aorta.sanitizer_report/0.1", "target": "gfx950",
            "worklist": {"schema": "aorta.kernel_worklist/0.1", "requirement": "top_dispatch_count",
                         "top_n": len(kernels), "kernel_count": len(kernels), "kernels": kernels},
            "checks": [{"sanitizer": "waitcheck", "state": "ran", "verdict": "pass", "reason": None,
                        "returncode": 0, "findings": [], "kernel_results": [], "coverage": [],
                        "backend": {}}],
            "overall_verdict": "pass", "execution_status": "complete",
        })
        (out_dir / "sanitizer_report.json").write_text(json.dumps(report.to_dict()))
        return 0
    name = source["kernel"]["name"]
    if name == "b_sym":
        return 1
    if name == "c_sym":
        Path(source["kernel"]["code_object"]).parent.joinpath("triton-cache", "HASH", "c.hsaco").write_bytes(b"recompiled")
    (out_dir / "sanitizer_report.json").write_text(json.dumps(_case_report(name, "ran", "pass")))
    return 0


def _run_toy(tmp_path: Path, monkeypatch, *extra: str) -> int:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ms, "sweep", _fake_sweep)
    return ms.main(["run", "--inventory", "recipes/toy.yaml", "--out", "results/informational",
                    "--work", "work", "--asset-dir", "asset/aorta-model-kernels", "--gpus", "0",
                    *extra])


def test_relative_cli_paths_still_reach_the_harness_and_cache(tmp_path, monkeypatch):
    # The shim cds into the asset before using any path it was given, so every
    # path must already be absolute. Relative ones made the nightly compile nothing.
    _toy_asset(tmp_path)
    assert _run_toy(tmp_path, monkeypatch) == 0
    record = json.loads((tmp_path / "results/informational/waitcheck-toy/model_survey.json").read_text())
    assert [p["ok"] for p in record["pairs"]] == [True, True, True, True]
    shim = (tmp_path / "work/toy/cases/a__S0/launch.sh").read_text()
    assert f"TRITON_CACHE_DIR={tmp_path / 'work/toy/cases/a__S0/triton-cache'}" in shim


def test_a_case_without_a_report_or_with_a_swapped_object_stays_on_the_row_as_an_error(
    tmp_path, monkeypatch
):
    _toy_asset(tmp_path)
    assert _run_toy(tmp_path, monkeypatch) == 0
    merged = json.loads((tmp_path / "results/informational/consan-toy/sanitizer_report.json").read_text())
    results = {kr["identity"]["name"]: kr for kr in merged["checks"][0]["kernel_results"]}
    assert set(results) == {"a_sym", "b_sym", "c_sym"}
    assert results["a_sym"]["verdict"] == "pass"
    assert results["b_sym"]["verdict"] == "error"
    assert results["b_sym"]["reason"] == "consan_case_produced_no_report: aorta sweep exited 1"
    assert results["c_sym"]["verdict"] == "error"
    assert results["c_sym"]["reason"].startswith("consan_object_mismatch:")
    assert merged["overall_verdict"] == "error"
    assert merged["worklist"]["kernel_count"] == 3


def test_launches_and_consan_cases_share_one_budget(tmp_path, monkeypatch):
    # A launch is admitted only if its timeout fits; with less than one launch
    # timeout of budget nothing compiles, which is an infrastructure fault.
    _toy_asset(tmp_path)
    assert _run_toy(tmp_path, monkeypatch, "--budget-seconds", str(ms.WARM_TIMEOUT_SECONDS - 1)) == 2

    # Enough for one launch but not for one 900 s ConSan case: the first wave
    # still compiles, later launches would spend that wave and are not started,
    # and no ConSan row is written to pass for the kernel that did compile.
    other = tmp_path / "second"
    other.mkdir()
    _toy_asset(other, consan_timeout=900)
    assert _run_toy(other, monkeypatch, "--budget-seconds", str(ms.WARM_TIMEOUT_SECONDS + 60)) == 0
    selection = json.loads((other / "results/informational/consan-toy/model_survey.json").read_text())
    assert selection.get("not_scheduled_tonight") == ["a"]
    assert sorted(selection.get("not_launched") or {}) == ["b", "c"]
    assert not (other / "results/informational/consan-toy/sanitizer_report.json").exists()


def test_slow_launches_leave_this_models_consan_wave(tmp_path, monkeypatch):
    # Every launch succeeds just inside its timeout. Without holding this
    # model's ConSan wave, four of them spend the budget and no ConSan case
    # starts, so --require finds no consan report.
    original = ms.Budget

    class _SlowLaunches(original):
        def admit(self, case_seconds: float, *, reserve: float = 0.0) -> bool:
            admitted = original.admit(self, case_seconds, reserve=reserve)
            if admitted and case_seconds == ms.WARM_TIMEOUT_SECONDS:
                with self._lock:
                    self._start -= ms.WARM_TIMEOUT_SECONDS - 1
            return admitted

    _toy_asset(tmp_path, consan_timeout=900)
    monkeypatch.setattr(ms, "Budget", _SlowLaunches)
    budget = ms.WARM_TIMEOUT_SECONDS * 2 + 900
    assert _run_toy(tmp_path, monkeypatch, "--budget-seconds", str(budget)) == 0
    informational = tmp_path / "results" / "informational"
    record = json.loads((informational / "waitcheck-toy" / "model_survey.json").read_text())
    assert [p.get("ok") for p in record.get("pairs") or []] == [True, True, False, False]
    selection = json.loads((informational / "consan-toy" / "model_survey.json").read_text())
    assert [r.get("kernel") for r in selection.get("ran") or []] == ["a"]
    assert (informational / "consan-toy" / "sanitizer_report.json").is_file()


def test_an_interrupt_stops_running_launches_and_drops_queued_ones(monkeypatch):
    # Only the main thread sees KeyboardInterrupt. Without stopping the groups
    # here, the pool would wait for each worker's subprocess to time out.
    if threading.current_thread() is not threading.main_thread():
        pytest.skip("SIGINT handlers can only be installed from the main thread")
    monkeypatch.setattr(ms, "_CHILDREN", ms._Children())
    started: list[int] = []
    procs: list[subprocess.Popen] = []
    lock = threading.Lock()
    both_running = threading.Event()

    def work(item: int, gpu: str) -> int:
        proc = ms._CHILDREN.start([sys.executable, "-c", "import time; time.sleep(60)"])
        with lock:
            started.append(item)
            procs.append(proc)
            if len(started) == 2:
                both_running.set()
        proc.wait()
        return item

    main = threading.main_thread().ident
    assert main is not None

    def interrupt() -> None:
        if both_running.wait(timeout=10):
            signal.pthread_kill(main, signal.SIGINT)

    previous = signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        threading.Thread(target=interrupt, daemon=True).start()
        begin = time.monotonic()
        with pytest.raises(KeyboardInterrupt):
            ms.on_gpus(["0", "1"], [0, 1, 2, 3], work)
        elapsed = time.monotonic() - begin
    finally:
        signal.signal(signal.SIGINT, previous)
    assert sorted(started) == [0, 1]
    assert all(proc.poll() is not None for proc in procs)
    assert elapsed < 30
    with pytest.raises(KeyboardInterrupt):
        ms._CHILDREN.start([sys.executable, "-c", "pass"])


def test_a_timed_out_consan_case_stays_on_the_row_as_an_error(tmp_path, monkeypatch):
    def _sweep(recipe: Path, out_dir: Path, *, env=None, timeout=None) -> int:
        plan = yaml.safe_load(recipe.read_text())["sanitizer_plan"]
        if plan["sanitizers"] == ["consan"] and plan["source"]["kernel"]["name"] == "b_sym":
            out_dir.mkdir(parents=True, exist_ok=True)
            return ms.SWEEP_TIMED_OUT
        return _fake_sweep(recipe, out_dir, env=env, timeout=timeout)

    _toy_asset(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ms, "sweep", _sweep)
    assert ms.main([
        "run", "--inventory", "recipes/toy.yaml", "--out", "results/informational",
        "--work", "work", "--asset-dir", "asset/aorta-model-kernels", "--gpus", "0",
    ]) == 0
    merged = json.loads((tmp_path / "results/informational/consan-toy/sanitizer_report.json").read_text())
    consan = (merged.get("checks") or [{}])[0]
    results = {kr["identity"]["name"]: kr for kr in consan.get("kernel_results") or []}
    assert results["b_sym"].get("verdict") == "error"
    assert str(results["b_sym"].get("reason") or "").startswith("consan_case_timed_out:")
    assert merged.get("overall_verdict") == "error"
    assert "consan_case_timed_out" in str(consan.get("reason") or "")
    monkeypatch.chdir(_REPO)
    assert comparator.main(
        ["prog", "--vacuous-only", "--require", "informational/consan-toy", str(tmp_path / "results")]
    ) == 1


@pytest.mark.parametrize(
    ("planted", "returncode", "reason"),
    [
        pytest.param('{"schema": "aorta.sanitizer_re', -9, "waitcheck_produced_no_report", id="truncated"),
        pytest.param('{"schema": "aorta.sanitizer_report/0.1"}', 1, "waitcheck_produced_no_report",
                     id="schema-invalid"),
        pytest.param("[]", ms.SWEEP_TIMED_OUT, "waitcheck_timed_out", id="not-a-mapping"),
    ],
)
def test_an_unreadable_waitcheck_report_is_replaced(tmp_path, monkeypatch, planted, returncode, reason):
    # The dashboard drops a report it cannot load, so a sweep killed mid-write
    # must not leave its partial file in place of the failure report.
    def _sweep(recipe: Path, out_dir: Path, *, env=None, timeout=None) -> int:
        plan = yaml.safe_load(recipe.read_text())["sanitizer_plan"]
        if plan["sanitizers"] == ["waitcheck"]:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "sanitizer_report.json").write_text(planted, encoding="utf-8")
            return returncode
        return _fake_sweep(recipe, out_dir, env=env, timeout=timeout)

    _toy_asset(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ms, "sweep", _sweep)
    assert ms.main([
        "run", "--inventory", "recipes/toy.yaml", "--out", "results/informational",
        "--work", "work", "--asset-dir", "asset/aorta-model-kernels", "--gpus", "0",
    ]) == 0
    scan = json.loads((tmp_path / "results/informational/waitcheck-toy/sanitizer_report.json").read_text())
    SanitizerReport.from_dict(scan)
    check = (scan.get("checks") or [{}])[0]
    assert str(check.get("reason") or "").startswith(f"{reason}:")
    assert scan.get("overall_verdict") == "error"


def _plant_model(root: Path, name: str, kernels: list[str]) -> None:
    model = root / name
    (model / "roofline" / "harness").mkdir(parents=True)
    (model / "roofline" / "specs").mkdir(parents=True)
    (model / "pkg_src").mkdir()
    (model / "kernels").mkdir()
    (model / "roofline" / "harness" / "launch.py").write_text(_STUB_LAUNCH, encoding="utf-8")
    (model / "roofline" / "specs" / f"{name}.json").write_text(
        json.dumps(
            [{"kernel": kernel, "role": "launched", "scenarios": {"S0": {}}} for kernel in kernels]
        ),
        encoding="utf-8",
    )


def test_a_later_model_keeps_a_consan_wave_when_earlier_cases_hang(tmp_path, monkeypatch):
    # One shared budget, and every admitted case hangs until its own timeout.
    # Without a hold for the later model its ConSan report is never written and
    # the nightly --require check fails; the hold gives that wave back.
    root = tmp_path / "asset" / "aorta-model-kernels"
    _plant_model(root, "early", ["e"])
    _plant_model(root, "late", ["l"])
    _manifest(root)
    recipes = tmp_path / "recipes"
    recipes.mkdir()
    for name in ("early", "late"):
        _write_inventory(
            recipes,
            name=name,
            specs=[f"{name}.json"],
            expected_pairs=1,
            consan={"timeout_seconds": 900, "exclude": {}},
        )

    original = ms.Budget

    class _Hung(original):
        def admit(self, case_seconds: float, *, reserve: float = 0.0) -> bool:
            admitted = original.admit(self, case_seconds, reserve=reserve)
            if admitted:
                with self._lock:
                    self._start -= case_seconds
            return admitted

    monkeypatch.setattr(ms, "Budget", _Hung)
    monkeypatch.setattr(ms, "sweep", _fake_sweep)
    monkeypatch.chdir(tmp_path)
    # Fits early's launch, exactly one 900s ConSan case, late's launch, and not
    # a second 900s case, plus the later model's waitcheck sweep. The hold is
    # what makes the one ConSan case be late's, and it leaves early no time to
    # scan without spending that hold.
    budget = ms.WARM_TIMEOUT_SECONDS * 2 + ms.SCAN_TIMEOUT_SECONDS + 900 + 100
    code = ms.main(
        [
            "run",
            "--inventory", "recipes/early.yaml",
            "--inventory", "recipes/late.yaml",
            "--out", "results/informational",
            "--work", "work",
            "--asset-dir", "asset/aorta-model-kernels",
            "--gpus", "0",
            "--budget-seconds", str(budget),
        ]
    )
    informational = tmp_path / "results" / "informational"
    assert code == 0
    early_scan = json.loads((informational / "waitcheck-early" / "model_survey.json").read_text())
    assert early_scan.get("sweep_timed_out") is True
    assert "not started" in (informational / "waitcheck-early" / "sweep.log").read_text()
    early_report = json.loads((informational / "waitcheck-early" / "sanitizer_report.json").read_text())
    SanitizerReport.from_dict(early_report)
    early_check = (early_report.get("checks") or [{}])[0]
    assert early_check.get("state") == "timed_out"
    assert str(early_check.get("reason") or "").startswith("waitcheck_not_started:")
    assert (early_report.get("worklist") or {}).get("kernel_count") == 1
    late_report = json.loads((informational / "waitcheck-late" / "sanitizer_report.json").read_text())
    assert late_report.get("overall_verdict") == "pass"
    assert not (informational / "consan-early" / "sanitizer_report.json").exists()
    assert (informational / "consan-late" / "sanitizer_report.json").is_file()
    monkeypatch.chdir(_REPO)
    results = str(tmp_path / "results")
    # The timeout report is published, and it is vacuous, so the nightly check
    # still fails until that case is named.
    assert comparator.main(
        ["prog", "--vacuous-only", "--require", "informational/consan-late", results]
    ) == 1
    assert comparator.main(
        [
            "prog", "--vacuous-only",
            "--require", "informational/consan-late",
            "--known-vacuous", "informational/waitcheck-early",
            results,
        ]
    ) == 0


def test_a_sweep_timeout_stops_the_process_group(tmp_path, monkeypatch):
    class _Proc:
        pid = 4321

        def __init__(self):
            self.calls = 0

        def wait(self, timeout=None):
            self.calls += 1
            if self.calls == 1:
                raise subprocess.TimeoutExpired("sweep", timeout)
            return -signal.SIGKILL

    opened = {}

    def _popen(*_args, **kwargs):
        opened.update(kwargs)
        return _Proc()

    killed = []
    monkeypatch.setattr(ms.subprocess, "Popen", _popen)
    monkeypatch.setattr(ms.os, "killpg", lambda pid, sig: killed.append((pid, sig)))
    rc = ms.sweep(tmp_path / "recipe.yaml", tmp_path / "out", timeout=3)
    assert rc == ms.SWEEP_TIMED_OUT
    assert opened.get("start_new_session") is True
    assert killed == [(4321, signal.SIGKILL)]
    assert "process group was stopped" in (tmp_path / "out" / "sweep.log").read_text()


def test_a_timed_out_waitcheck_is_reported_and_consan_still_runs(tmp_path, monkeypatch):
    def _sweep(recipe: Path, out_dir: Path, *, env=None, timeout=None) -> int:
        plan = yaml.safe_load(recipe.read_text())["sanitizer_plan"]
        if plan["sanitizers"] == ["waitcheck"]:
            assert timeout is not None and timeout > 0
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "sweep.log").write_text("sweep timed out\n")
            return ms.SWEEP_TIMED_OUT
        return _fake_sweep(recipe, out_dir, env=env, timeout=timeout)

    _toy_asset(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ms, "sweep", _sweep)
    assert ms.main([
        "run", "--inventory", "recipes/toy.yaml", "--out", "results/informational",
        "--work", "work", "--asset-dir", "asset/aorta-model-kernels", "--gpus", "0",
    ]) == 0
    record = json.loads((tmp_path / "results/informational/waitcheck-toy/model_survey.json").read_text())
    assert record.get("sweep_timed_out") is True
    assert record.get("sweep_returncode") == ms.SWEEP_TIMED_OUT
    scan = json.loads((tmp_path / "results/informational/waitcheck-toy/sanitizer_report.json").read_text())
    SanitizerReport.from_dict(scan)
    scan_check = (scan.get("checks") or [{}])[0]
    assert scan_check.get("sanitizer") == "waitcheck"
    assert scan_check.get("state") == "timed_out"
    assert scan.get("overall_verdict") == "error"
    assert scan.get("execution_status") == "error"
    assert str(scan_check.get("reason") or "").startswith("waitcheck_timed_out:")
    assert (scan.get("worklist") or {}).get("kernel_count") == 3
    assert "sweep timed out" in (tmp_path / "results/informational/waitcheck-toy/sweep.log").read_text()
    merged = json.loads((tmp_path / "results/informational/consan-toy/sanitizer_report.json").read_text())
    assert merged["checks"][0]["kernel_results"]
