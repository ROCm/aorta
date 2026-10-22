"""Unit tests for the open-source model-kernel survey (scripts/sanitizers/model_survey.py)."""

from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import importlib.util
import io
import json
import re
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
