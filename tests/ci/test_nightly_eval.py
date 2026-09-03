"""Integration test for the nightly-eval harness (scripts/ci/nightly_eval.py).

Monkeypatches the GPU / subprocess boundary (run_entry, gpu_count) so the
end-to-end evaluate() logic can be exercised on the CPU gate with a synthetic
matrix.json -- no torch / aorta / GPU required.
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from types import SimpleNamespace

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _REPO_ROOT / "scripts" / "ci" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


nightly_eval = _load("nightly_eval")


def _write_matrix(path: Path, cells):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"cells": cells}), encoding="utf-8")
    return path


def test_evaluate_record_pass_fail_and_skip(tmp_path, monkeypatch):
    matrix_doc = {
        "entries": [
            {"name": "gpu_smoke", "recipe": "r1.yaml"},
            {"name": "inference_offline", "recipe": "r2.yaml"},
            {"name": "race", "recipe": "r3.yaml", "nproc": 2, "min_gpus": 2},
        ]
    }
    baselines = {
        "baselines": {
            # inference has a blessed baseline it satisfies -> pass
            "inference_offline::baseline-local": {"passed": True, "step_time_ms": {"max": 10.0}},
        }
    }

    # Pretend the runner has exactly 1 GPU (so the 2-GPU race entry is skipped).
    monkeypatch.setattr(nightly_eval, "gpu_count", lambda: 1)

    def fake_run_entry(entry, out_dir):
        name = entry["name"]
        mpath = out_dir / name / "matrix.json"
        if name == "gpu_smoke":
            _write_matrix(mpath, [{"name": "baseline-local", "error": None,
                                   "passed_count": 1, "failed_count": 0, "error_count": 0,
                                   "mean_step_time_ms": 2.0, "metrics_summary": {}}])
        elif name == "inference_offline":
            _write_matrix(mpath, [{"name": "baseline-local", "error": None,
                                   "passed_count": 1, "failed_count": 0, "error_count": 0,
                                   "mean_step_time_ms": 4.0, "metrics_summary": {}}])
        return 0, mpath, False

    monkeypatch.setattr(nightly_eval, "run_entry", fake_run_entry)
    monkeypatch.setattr(nightly_eval, "build_metadata", lambda: {"amd_aorta_version": "x"})

    doc = nightly_eval.evaluate(matrix_doc, baselines, tmp_path)

    by_entry = {(e["entry"], e["cell"]): e for e in doc["entries"]}
    assert by_entry[("gpu_smoke", "baseline-local")]["verdict"] == "record"
    assert by_entry[("inference_offline", "baseline-local")]["verdict"] == "pass"
    assert by_entry[("race", None)]["verdict"] == "skip"
    assert doc["summary"]["record"] == 1
    assert doc["summary"]["pass"] == 1
    assert doc["summary"]["skip"] == 1
    assert doc["summary"]["fail"] == 0


def test_evaluate_fails_when_no_matrix_json(tmp_path, monkeypatch):
    matrix_doc = {"entries": [{"name": "broken", "recipe": "r.yaml"}]}
    monkeypatch.setattr(nightly_eval, "gpu_count", lambda: 1)
    monkeypatch.setattr(nightly_eval, "run_entry", lambda entry, out_dir: (17, None, False))
    monkeypatch.setattr(nightly_eval, "build_metadata", lambda: {})

    doc = nightly_eval.evaluate(matrix_doc, {"baselines": {}}, tmp_path)
    assert doc["summary"]["fail"] == 1
    assert doc["entries"][0]["verdict"] == "fail"


def test_evaluate_fails_when_all_entries_skip(tmp_path, monkeypatch):
    # Zero work (e.g. no GPUs) must FAIL, not go green.
    matrix_doc = {"entries": [{"name": "race", "recipe": "r.yaml", "nproc": 2, "min_gpus": 2}]}
    monkeypatch.setattr(nightly_eval, "gpu_count", lambda: 0)
    monkeypatch.setattr(nightly_eval, "build_metadata", lambda: {})

    doc = nightly_eval.evaluate(matrix_doc, {"baselines": {}}, tmp_path)
    assert doc["summary"]["fail"] == 1
    assert any(e["error"] == "zero_work" for e in doc["entries"])


def test_evaluate_timeout_records_failure(tmp_path, monkeypatch):
    matrix_doc = {"entries": [{"name": "hang", "recipe": "r.yaml"}]}
    monkeypatch.setattr(nightly_eval, "gpu_count", lambda: 1)
    monkeypatch.setattr(nightly_eval, "run_entry", lambda entry, out_dir: (124, None, True))
    monkeypatch.setattr(nightly_eval, "build_metadata", lambda: {})

    doc = nightly_eval.evaluate(matrix_doc, {"baselines": {}}, tmp_path)
    assert doc["summary"]["fail"] == 1
    assert doc["entries"][0]["error"] == "timeout"


def test_evaluate_timeout_is_authoritative_even_with_matrix(tmp_path, monkeypatch):
    # rank 0 wrote a matrix.json but a worker hung -> must still fail.
    matrix_doc = {"entries": [{"name": "hang", "recipe": "r.yaml"}]}
    monkeypatch.setattr(nightly_eval, "gpu_count", lambda: 1)
    monkeypatch.setattr(nightly_eval, "build_metadata", lambda: {})

    def fake(entry, out_dir):
        mpath = _write_matrix(out_dir / entry["name"] / "matrix.json",
                              [{"name": "c", "error": None, "passed_count": 1,
                                "failed_count": 0, "error_count": 0, "metrics_summary": {}}])
        return 124, mpath, True

    monkeypatch.setattr(nightly_eval, "run_entry", fake)
    doc = nightly_eval.evaluate(matrix_doc, {"baselines": {}}, tmp_path)
    assert doc["summary"]["fail"] == 1
    assert doc["entries"][0]["error"] == "timeout"


def test_evaluate_empty_matrix_entry_fails(tmp_path, monkeypatch):
    matrix_doc = {"entries": [{"name": "nocells", "recipe": "r.yaml"}]}
    monkeypatch.setattr(nightly_eval, "gpu_count", lambda: 1)
    monkeypatch.setattr(nightly_eval, "build_metadata", lambda: {})

    def fake(entry, out_dir):
        mpath = _write_matrix(out_dir / entry["name"] / "matrix.json", [])  # zero cells
        return 0, mpath, False

    monkeypatch.setattr(nightly_eval, "run_entry", fake)
    doc = nightly_eval.evaluate(matrix_doc, {"baselines": {}}, tmp_path)
    assert doc["summary"]["fail"] == 1
    assert doc["entries"][0]["error"] == "empty_matrix"


def test_evaluate_corrupt_matrix_json_fails_with_artifact(tmp_path, monkeypatch):
    # A truncated matrix.json must produce a per-entry fail (not abort evaluate()
    # before results JSON is written).
    matrix_doc = {"entries": [{"name": "corrupt", "recipe": "r.yaml"}]}
    monkeypatch.setattr(nightly_eval, "gpu_count", lambda: 1)
    monkeypatch.setattr(nightly_eval, "build_metadata", lambda: {})

    def fake(entry, out_dir):
        mpath = out_dir / entry["name"] / "matrix.json"
        mpath.parent.mkdir(parents=True, exist_ok=True)
        mpath.write_text('{"cells": [', encoding="utf-8")  # truncated JSON
        return 0, mpath, False

    monkeypatch.setattr(nightly_eval, "run_entry", fake)
    doc = nightly_eval.evaluate(matrix_doc, {"baselines": {}}, tmp_path)
    assert doc["summary"]["fail"] == 1
    assert doc["entries"][0]["error"] == "corrupt_matrix"


refresh_baselines = _load("refresh_baselines")


def test_refresh_carries_over_baseline_for_gpu_skipped_entry(tmp_path, monkeypatch):
    # An 8-GPU entry that can't run on a 1-GPU box must NOT poison the refresh:
    # its existing baseline is carried over instead of dropped to record-only.
    matrix_doc = {"entries": [
        {"name": "gpu_smoke", "recipe": "r1.yaml"},
        {"name": "race_8gpu", "recipe": "r8.yaml", "nproc": 8, "min_gpus": 8},
    ]}
    existing = {"race_8gpu::baseline-local": {"passed": True}}

    monkeypatch.setattr(refresh_baselines.nightly_eval, "gpu_count", lambda: 1)

    def fake_run_entry(entry, out_dir):
        mpath = _write_matrix(out_dir / entry["name"] / "matrix.json",
                              [{"name": "baseline-local", "error": None, "passed_count": 1,
                                "failed_count": 0, "error_count": 0, "metrics_summary": {}}])
        return 0, mpath, False

    monkeypatch.setattr(refresh_baselines.nightly_eval, "run_entry", fake_run_entry)

    doc = refresh_baselines.build_baselines(
        matrix_doc, tmp_path, 0.25, 0.15, False, existing_baselines=existing)

    assert doc["baselines"]["race_8gpu::baseline-local"] == {"passed": True}
    assert "gpu_smoke::baseline-local" in doc["baselines"]


def test_refresh_gpu_skipped_without_existing_is_not_fatal(tmp_path, monkeypatch):
    matrix_doc = {"entries": [
        {"name": "gpu_smoke", "recipe": "r1.yaml"},
        {"name": "race_8gpu", "recipe": "r8.yaml", "nproc": 8, "min_gpus": 8},
    ]}
    monkeypatch.setattr(refresh_baselines.nightly_eval, "gpu_count", lambda: 1)

    def fake_run_entry(entry, out_dir):
        mpath = _write_matrix(out_dir / entry["name"] / "matrix.json",
                              [{"name": "baseline-local", "error": None, "passed_count": 1,
                                "failed_count": 0, "error_count": 0, "metrics_summary": {}}])
        return 0, mpath, False

    monkeypatch.setattr(refresh_baselines.nightly_eval, "run_entry", fake_run_entry)

    doc = refresh_baselines.build_baselines(matrix_doc, tmp_path, 0.25, 0.15, False)
    assert "race_8gpu::baseline-local" not in doc["baselines"]
    assert "gpu_smoke::baseline-local" in doc["baselines"]


def test_refresh_refuses_on_corrupt_matrix_json(tmp_path, monkeypatch):
    # A corrupt matrix.json must refuse cleanly (atomic message), not traceback.
    import pytest
    matrix_doc = {"entries": [{"name": "gpu_smoke", "recipe": "r1.yaml"}]}
    monkeypatch.setattr(refresh_baselines.nightly_eval, "gpu_count", lambda: 8)

    def fake_run_entry(entry, out_dir):
        mpath = out_dir / entry["name"] / "matrix.json"
        mpath.parent.mkdir(parents=True, exist_ok=True)
        mpath.write_text('{"cells": [', encoding="utf-8")  # truncated JSON
        return 0, mpath, False

    monkeypatch.setattr(refresh_baselines.nightly_eval, "run_entry", fake_run_entry)

    with pytest.raises(SystemExit, match="unreadable matrix.json"):
        refresh_baselines.build_baselines(matrix_doc, tmp_path, 0.25, 0.15, False)


def test_refresh_refuses_when_entry_ran_but_failed(tmp_path, monkeypatch):
    # A genuine did-not-pass (ran but failed) must still refuse atomically.
    import pytest
    matrix_doc = {"entries": [{"name": "gpu_smoke", "recipe": "r1.yaml"}]}
    monkeypatch.setattr(refresh_baselines.nightly_eval, "gpu_count", lambda: 8)

    def fake_run_entry(entry, out_dir):
        mpath = _write_matrix(out_dir / entry["name"] / "matrix.json",
                              [{"name": "baseline-local", "error": None, "passed_count": 0,
                                "failed_count": 1, "error_count": 0, "metrics_summary": {}}])
        return 0, mpath, False

    monkeypatch.setattr(refresh_baselines.nightly_eval, "run_entry", fake_run_entry)

    with pytest.raises(SystemExit):
        refresh_baselines.build_baselines(matrix_doc, tmp_path, 0.25, 0.15, False)


# ---------------------------------------------------------------------------
# Scoping --perf-gate to one entry, and not auto-arming a near-zero metric.
# Both exist so serving perf gating can be rolled out on its own; see
# docs/tokenspeed-gating-rollout.md.
# ---------------------------------------------------------------------------


_SERVING_SUMMARY = {
    "median_ttft_ms": {"mean": 46.3},
    "median_tpot_ms": {"mean": 1.94},
    "median_itl_ms": {"mean": 0.0},
    "p99_itl_ms": {"mean": 21.4},
    "output_throughput": {"mean": 3538.0},
}


def _serving_matrix_runner(summary=None):
    summary = _SERVING_SUMMARY if summary is None else summary

    def fake_run_entry(entry, out_dir):
        mpath = _write_matrix(out_dir / entry["name"] / "matrix.json",
                              [{"name": "baseline", "error": None, "passed_count": 1,
                                "failed_count": 0, "error_count": 0,
                                "mean_step_time_ms": 1100.0,
                                "metrics_summary": summary}])
        return 0, mpath, False

    return fake_run_entry


def test_perf_gate_can_be_scoped_to_a_single_entry(tmp_path, monkeypatch):
    """Rolling gating out per workload must not arm every other entry too.

    refresh_baselines rewrites the whole baseline file, so an unscoped
    --perf-gate would derive step-time ceilings for every unrelated entry from
    whatever single run happened to be under way -- the one-observation
    threshold this rollout exists to avoid, applied to workloads nobody looked at.
    """
    matrix_doc = {"entries": [
        {"name": "tokenspeed_serve_smoke", "recipe": "ts.yaml"},
        {"name": "gpu_smoke", "recipe": "r1.yaml"},
    ]}
    monkeypatch.setattr(refresh_baselines.nightly_eval, "gpu_count", lambda: 1)
    monkeypatch.setattr(refresh_baselines.nightly_eval, "run_entry", _serving_matrix_runner())

    doc = refresh_baselines.build_baselines(
        matrix_doc, tmp_path, 0.25, 0.15, True,
        perf_gate_entries={"tokenspeed_serve_smoke"})

    gated = doc["baselines"]["tokenspeed_serve_smoke::baseline"]
    ungated = doc["baselines"]["gpu_smoke::baseline"]
    assert gated["step_time_ms"] == {"max": 1375.0}
    assert gated["metrics"]["median_tpot_ms"] == {"policy": "max", "value": 2.425}
    assert gated["metrics"]["output_throughput"] == {"policy": "min", "value": 3007.3}
    # The unscoped entry stays record-only for performance: no step-time ceiling
    # and no metric bounds at all.
    assert "step_time_ms" not in ungated
    assert "metrics" not in ungated


def test_perf_gate_without_a_scope_still_gates_every_entry(tmp_path, monkeypatch):
    """The default is unchanged, so existing callers keep their behaviour."""
    matrix_doc = {"entries": [
        {"name": "tokenspeed_serve_smoke", "recipe": "ts.yaml"},
        {"name": "gpu_smoke", "recipe": "r1.yaml"},
    ]}
    monkeypatch.setattr(refresh_baselines.nightly_eval, "gpu_count", lambda: 1)
    monkeypatch.setattr(refresh_baselines.nightly_eval, "run_entry", _serving_matrix_runner())

    doc = refresh_baselines.build_baselines(matrix_doc, tmp_path, 0.25, 0.15, True)
    for key in ("tokenspeed_serve_smoke::baseline", "gpu_smoke::baseline"):
        assert doc["baselines"][key]["step_time_ms"] == {"max": 1375.0}


def test_perf_gate_does_not_arm_a_near_zero_median_itl(tmp_path, monkeypatch):
    """`median_itl_ms` is measured at ~0, so `value * 1.25` is a ceiling of 0.0
    that no later run with any inter-token gap can satisfy. It stays charted and
    hand-gateable; the refresher must not bless a bound for it."""
    matrix_doc = {"entries": [{"name": "tokenspeed_serve_smoke", "recipe": "ts.yaml"}]}
    monkeypatch.setattr(refresh_baselines.nightly_eval, "gpu_count", lambda: 1)
    monkeypatch.setattr(refresh_baselines.nightly_eval, "run_entry", _serving_matrix_runner())

    doc = refresh_baselines.build_baselines(matrix_doc, tmp_path, 0.25, 0.15, True)
    metrics = doc["baselines"]["tokenspeed_serve_smoke::baseline"]["metrics"]
    assert "median_itl_ms" not in metrics
    # Its tail counterpart is a real number and stays gateable.
    assert metrics["p99_itl_ms"] == {"policy": "max", "value": 26.75}


def _refresh_cli(*argv):
    import subprocess
    import sys
    return subprocess.run(
        [sys.executable, str(_REPO_ROOT / "scripts" / "ci" / "refresh_baselines.py"), *argv],
        capture_output=True, text=True, timeout=120)


def test_perf_gate_entry_rejects_a_name_the_matrix_does_not_have():
    """A typo would scope perf gating to nothing and produce a correctness-only
    refresh that reads, in the PR diff, exactly like a successful bless."""
    out = _refresh_cli("--perf-gate", "--perf-gate-entry", "tokenspeed_serve_smoek")
    assert out.returncode != 0
    assert "no such matrix entry" in out.stderr
    # The message has to be actionable from the terminal it appeared in.
    assert "gpu_smoke" in out.stderr


def test_perf_gate_entry_without_perf_gate_is_rejected():
    """Silently ignoring it would leave the operator believing gates were armed."""
    out = _refresh_cli("--perf-gate-entry", "gpu_smoke")
    assert out.returncode != 0
    assert "no effect without --perf-gate" in out.stderr


def test_an_unknown_entry_is_not_a_valid_perf_gate_scope():
    """Scoping to a name that is not in `entries` -- a staged entry, or a typo --
    must fail rather than quietly refresh everything correctness-only."""
    out = _refresh_cli("--perf-gate", "--perf-gate-entry", "not_a_real_entry")
    assert out.returncode != 0
    assert "no such matrix entry" in out.stderr


# ---------------------------------------------------------------------------
# The committed matrix file itself, through the loader the nightly uses.
# ---------------------------------------------------------------------------


def _real_matrix():
    return nightly_eval._load_yaml(nightly_eval.MATRIX)


def test_the_committed_matrix_loads_and_every_recipe_exists():
    doc = _real_matrix()
    assert doc["version"] == 1
    entries = doc["entries"]
    assert entries and len({e["name"] for e in entries}) == len(entries)
    for entry in entries:
        assert (nightly_eval.REPO_ROOT / entry["recipe"]).is_file(), entry["recipe"]


def test_pending_entries_are_inert(tmp_path, monkeypatch):
    """`pending_entries` documents staged work; nothing may execute it.

    Both consumers iterate `entries` only, and this pins that -- a future reader
    who wires the key up would otherwise start a blocked entry by accident.
    """
    doc = _real_matrix()
    # Synthesised rather than read from the file: the key is legitimately empty
    # between staged entries, and the property being pinned -- that a future
    # reader who wires the key up starts a blocked entry -- must hold then too.
    doc = dict(doc)
    doc["pending_entries"] = [
        {"name": "staged_thing", "recipe": "recipes/ci/gpu-smoke.yaml", "blocked_on": "x"},
        *(doc.get("pending_entries") or []),
    ]
    pending = {e["name"] for e in doc["pending_entries"]}

    monkeypatch.setattr(nightly_eval, "gpu_count", lambda: 8)
    monkeypatch.setattr(nightly_eval, "build_metadata", lambda: {})
    ran: list[str] = []

    def fake(entry, out_dir):
        ran.append(entry["name"])
        return 0, _write_matrix(out_dir / entry["name"] / "matrix.json",
                                [{"name": "c", "error": None, "passed_count": 1,
                                  "failed_count": 0, "error_count": 0,
                                  "metrics_summary": {}}]), False

    monkeypatch.setattr(nightly_eval, "run_entry", fake)
    result = nightly_eval.evaluate(doc, {"baselines": {}}, tmp_path)

    assert not (pending & set(ran))
    assert not (pending & {e["entry"] for e in result["entries"]})


def _docker_matrix():
    return {"entries": [
        {"name": "needs_daemon", "recipe": "r.yaml", "needs_docker_daemon": True},
        {"name": "plain", "recipe": "r2.yaml"},
    ]}


def _fake_run(ran):
    def run(entry, out_dir):
        ran.append(entry["name"])
        return 0, _write_matrix(out_dir / entry["name"] / "matrix.json",
                                [{"name": "c", "error": None, "passed_count": 1,
                                  "failed_count": 0, "error_count": 0,
                                  "metrics_summary": {}}]), False
    return run


def test_an_entry_needing_a_daemon_skips_rather_than_fails_when_there_is_none(
        tmp_path, monkeypatch):
    """The socket is a per-lane opt-in that grants effective root, so no lane
    sets it and "no daemon" is the normal state. An entry that needs one must
    therefore skip -- a fail would redden every nightly until a security
    decision that is not the pipeline's to make."""
    monkeypatch.setattr(nightly_eval, "gpu_count", lambda: 8)
    monkeypatch.setattr(nightly_eval, "build_metadata", lambda: {})
    monkeypatch.setattr(nightly_eval, "docker_daemon",
                        lambda: (False, "no docker client on PATH"))
    ran: list[str] = []
    monkeypatch.setattr(nightly_eval, "run_entry", _fake_run(ran))

    result = nightly_eval.evaluate(_docker_matrix(), {"baselines": {}}, tmp_path)
    by_name = {e["entry"]: e for e in result["entries"]}

    assert by_name["needs_daemon"]["verdict"] == "skip"
    assert "no docker client on PATH" in by_name["needs_daemon"]["reasons"][0]
    assert "needs_daemon" not in ran
    # And the probe must not quarantine anything that did not ask for it.
    assert by_name["plain"]["verdict"] != "skip"
    assert ran == ["plain"]


def test_an_entry_needing_a_daemon_runs_when_one_is_reachable(tmp_path, monkeypatch):
    """The other half: the skip must disappear the moment a lane mounts the
    socket, with no further edit to the matrix."""
    monkeypatch.setattr(nightly_eval, "gpu_count", lambda: 8)
    monkeypatch.setattr(nightly_eval, "build_metadata", lambda: {})
    monkeypatch.setattr(nightly_eval, "docker_daemon", lambda: (True, "daemon 29.4.2"))
    ran: list[str] = []
    monkeypatch.setattr(nightly_eval, "run_entry", _fake_run(ran))

    nightly_eval.evaluate(_docker_matrix(), {"baselines": {}}, tmp_path)
    assert ran == ["needs_daemon", "plain"]


def test_the_daemon_probe_reports_a_client_that_cannot_reach_a_daemon(monkeypatch):
    """A mounted-but-dead socket is the interesting case: the client is on PATH,
    so a `which docker` check would call the capability present and the entry
    would fail on connect. The probe must actually talk to the daemon."""
    import subprocess as sp

    nightly_eval.docker_daemon.cache_clear()
    monkeypatch.setattr(nightly_eval.shutil, "which", lambda _: "/usr/local/bin/docker")
    monkeypatch.setattr(
        nightly_eval.subprocess, "run",
        lambda *a, **k: sp.CompletedProcess(
            a[0], 1, "", "Cannot connect to the Docker daemon at unix:///var/run/docker.sock."),
    )
    try:
        reachable, detail = nightly_eval.docker_daemon()
        assert reachable is False
        assert "Cannot connect to the Docker daemon" in detail
    finally:
        nightly_eval.docker_daemon.cache_clear()


def test_the_live_serving_entry_declares_the_capability_it_needs():
    """Promoted on a demonstrated launch, but only safe in `entries` because it
    declares the capability. Without the flag it is a guaranteed nightly failure
    on every runner that has not opted into the socket."""
    live = {e["name"]: e for e in _real_matrix()["entries"]}
    entry = live["tokenspeed_serve_smoke"]
    assert entry["needs_docker_daemon"] is True
    assert "blocked_on" not in entry
    assert int(entry["timeout_sec"]) >= 3600


def test_the_gating_recipe_discards_two_steps_before_measuring():
    """A step-0 Triton compile excursion reaches the metrics at `warmup_steps: 1`.

    Measured at 1 cell-run in 13: a ~10x first-token spike (465 ms against a
    43-47 ms clean range) and a 2825 ms step against 1108-1178 ms. The excursion
    lands on the first *measured* step, after the single warmup step has already
    been discarded, so it breaches three of the four originally-proposed gates on
    a run that is not a regression.

    `num_warmups` is a different knob and cannot substitute: it warms requests
    within one bench invocation, not the compile cache across invocations. The
    excursion is positional, so one more discarded step removes it by
    construction -- which is the only reason the duration-derived metrics can be
    promoted later at all.

    The matrix carries no `workload_config` override (nightly_eval.py reads only
    recipe/min_gpus/timeout_sec/needs_docker_daemon), so the recipe is the single
    place this can be set, and it is shared with the baseline refresher -- which
    is what keeps the blessed numbers and the nightly on the same measurement.
    """
    import yaml

    entry = {e["name"]: e for e in _real_matrix()["entries"]}["tokenspeed_serve_smoke"]
    recipe_path = nightly_eval.REPO_ROOT / entry["recipe"]
    config = yaml.safe_load(recipe_path.read_text("utf-8"))["workload_config"]

    assert config["warmup_steps"] == 2, (
        "the gating recipe must discard two bench steps; at 1 the measured "
        "step-0 compile excursion enters the metrics the nightly gates on"
    )

    # The workload derives its internal budget from
    # ready_timeout + (steps + warmup_steps) * bench_timeout, so an extra warmup
    # step lengthens it. The binding cap is the entry's own timeout_sec, and a
    # bench step here is ~1.1s against a 3600s budget dominated by bring-up --
    # but assert the relationship rather than trusting the arithmetic to hold.
    assert int(entry["timeout_sec"]) >= 3600


def test_the_rollout_doc_marks_its_variance_data_as_the_old_configuration():
    """The 13-cell-run variance table was measured at `warmup_steps: 1`, which the
    recipe no longer uses, so it is the rationale for the change and not the
    baseline to bless against. It is deliberately retained -- there is no
    measurement at the new setting yet, and the ten-night record-only window is
    what produces one. This guards the caveat, not the table."""
    doc = (nightly_eval.REPO_ROOT / "docs/tokenspeed-gating-rollout.md").read_text("utf-8")
    assert "warmup_steps: 1" in doc, "the provenance of the variance data is gone"
    assert "warmup_steps: 2" in doc, "the doc does not state the new setting"
    # The table itself must survive.
    assert "Clean range over 12 cell-runs" in doc
    # And the window has to be pinned to the setting the gate will run at.
    assert re.search(
        r"ten nightlies, at `warmup_steps: 2`", doc
    ), "the record-only window is not pinned to warmup_steps: 2"


def test_pending_entries_are_valid_and_loadable():
    """Validated exactly like a live entry, so promoting one is a move, not a bet.

    A staged entry that names a deleted recipe or a misspelled field would
    otherwise only be discovered by the first red nightly after promotion.
    """
    from aorta.triage.recipe import load_recipe

    known_fields = {"name", "recipe", "nproc", "min_gpus", "timeout_sec",
                    "needs_docker_daemon", "blocked_on"}
    pending = _real_matrix().get("pending_entries") or []
    live = {e["name"] for e in _real_matrix()["entries"]}

    for entry in pending:
        assert set(entry) <= known_fields, f"{entry['name']}: {set(entry) - known_fields}"
        assert entry["name"] not in live, f"{entry['name']} is both staged and live"
        # A staged entry exists because it cannot run yet; say why, in the file.
        assert entry.get("blocked_on"), f"{entry['name']} has no blocked_on"
        path = nightly_eval.REPO_ROOT / entry["recipe"]
        assert path.is_file(), entry["recipe"]
        recipe = load_recipe(path)
        assert recipe.cells, f"{entry['name']}: recipe has no cells"
        # Every entry's budget must cover the recipe it names; the default is
        # 1800s and a serving bring-up alone has been measured at 379s a cell.
        assert int(entry.get("timeout_sec", 1800)) >= 1800


def test_serving_entry_metric_names_resolve_against_the_policy_table():
    """The names `tokenspeed bench serve` exports, checked against the allowlist.

    The workload passes its export through verbatim, so a metric the plan intends
    to gate must resolve to the right direction -- and the ones it intends never
    to gate must resolve to nothing, since an allowlist entry is what --perf-gate
    arms from.
    """
    eval_lib = _load("eval_lib")
    live = {e["name"]: e for e in _real_matrix()["entries"]}
    assert "tokenspeed_serve_smoke" in live

    # Gated from the first bless (docs/tokenspeed-gating-rollout.md). Both are
    # per-token metrics: they are measured between tokens, so the step-0 compile
    # excursion measured on this cell does not enter them.
    for name in ("median_tpot_ms", "p99_itl_ms"):
        assert eval_lib.metric_policy(name) == "max", name
        assert eval_lib.is_auto_gateable(name) is True, name

    # Gateable and correctly directed, but held record-only for the first bless
    # because all three are duration-derived or wait on the compile, and the
    # excursion carries each of them through its threshold.
    assert eval_lib.metric_policy("median_ttft_ms") == "max"
    assert eval_lib.metric_policy("output_throughput") == "min"
    for name in ("median_ttft_ms", "output_throughput"):
        assert eval_lib.is_auto_gateable(name) is True, name

    # Recorded but never gated: bring-up time is the noisiest thing the workload
    # reports (189-379s on one node, nothing changed), and the counters/totals
    # restate the recipe rather than measuring the stack.
    for name in ("server_startup_sec", "container_elapsed_sec", "duration",
                 "completed_total", "failed_total", "total_output_tokens",
                 "max_output_tokens_per_s", "max_concurrent_requests",
                 "mean_ttft_ms", "std_ttft_ms", "p50_ttft_ms", "p90_ttft_ms"):
        assert eval_lib.metric_policy(name) is None, name

    # Gateable, but not armed by a refresh.
    assert eval_lib.is_auto_gateable("median_itl_ms") is False


# ---------------------------------------------------------------------------
# Dashboard metadata: the `rocm` column on both install layouts (issue #381)
# ---------------------------------------------------------------------------


def _roots(core: Path, libraries: Path | None = None):
    """Stand-in for rocm_paths.RocmRoots with only what build_metadata reads."""
    libraries = libraries or core
    return SimpleNamespace(
        core=core,
        libraries=libraries,
        version_file=core / ".info" / "version",
        version_dev_file=core / ".info" / "version-dev",
        lib_dir=libraries / "lib",
    )


def _write_version(root: Path, name: str, text: str) -> Path:
    info = root / ".info"
    info.mkdir(parents=True, exist_ok=True)
    path = info / name
    path.write_text(text, encoding="utf-8")
    return path


def test_metadata_reads_rocm_version_on_a_classic_layout(tmp_path, monkeypatch):
    root = tmp_path / "opt_rocm"
    _write_version(root, "version", "7.2.4\n")
    monkeypatch.setattr(nightly_eval, "_ROCM_ROOTS", _roots(root))
    assert nightly_eval.build_metadata()["rocm"] == "7.2.4"


def test_metadata_reads_rocm_version_on_a_wheel_layout(tmp_path, monkeypatch):
    """The regression #381 fixed: a wheel install left this column null.

    `torch` and `hip` still populated from the same run, so the dashboard row
    looked complete while the ROCm version -- the thing rows are compared
    across -- was silently missing.
    """
    core = tmp_path / "site-packages" / "_rocm_sdk_core"
    _write_version(core, "version", "7.14.0\n")
    libraries = tmp_path / "site-packages" / "_rocm_sdk_libraries"
    monkeypatch.setattr(nightly_eval, "_ROCM_ROOTS", _roots(core, libraries))
    assert nightly_eval.build_metadata()["rocm"] == "7.14.0"


def test_metadata_falls_back_to_version_dev(tmp_path, monkeypatch):
    root = tmp_path / "opt_rocm"
    _write_version(root, "version-dev", "7.2.4.50311-abc1234\n")
    monkeypatch.setattr(nightly_eval, "_ROCM_ROOTS", _roots(root))
    assert nightly_eval.build_metadata()["rocm"] == "7.2.4.50311-abc1234"


def test_metadata_rocm_is_null_when_no_install_is_found(tmp_path, monkeypatch):
    monkeypatch.setattr(nightly_eval, "_ROCM_ROOTS", _roots(tmp_path / "absent"))
    assert nightly_eval.build_metadata()["rocm"] is None


# ---------------------------------------------------------------------------
# Lane + base-image attribution for the latest-ROCm canary (issue #382)
# ---------------------------------------------------------------------------


def test_metadata_defaults_to_the_gate_lane(tmp_path, monkeypatch):
    """Existing callers keep describing themselves correctly, unchanged."""
    monkeypatch.delenv("AORTA_CI_LANE", raising=False)
    monkeypatch.delenv("AORTA_CI_BASE_IMAGE", raising=False)
    monkeypatch.setattr(nightly_eval, "_ROCM_ROOTS", _roots(tmp_path / "absent"))
    meta = nightly_eval.build_metadata()
    assert meta["lane"] == "gate"
    # The gated lane's digest is pinned in the Dockerfile and visible in review,
    # so there is nothing to record here.
    assert meta["base_image"] is None


def test_metadata_records_the_canary_lane_and_resolved_digest(tmp_path, monkeypatch):
    """"Ran :latest" is not attributable; the resolved digest is (#382)."""
    base = "rocm/pytorch:latest@sha256:" + "ab" * 32
    monkeypatch.setenv("AORTA_CI_LANE", "canary")
    monkeypatch.setenv("AORTA_CI_BASE_IMAGE", base)
    monkeypatch.setattr(nightly_eval, "_ROCM_ROOTS", _roots(tmp_path / "absent"))
    meta = nightly_eval.build_metadata()
    assert meta["lane"] == "canary"
    assert meta["base_image"] == base


def test_empty_lane_env_falls_back_to_gate(tmp_path, monkeypatch):
    """An exported-but-empty var must not produce a row labelled "" ."""
    monkeypatch.setenv("AORTA_CI_LANE", "")
    monkeypatch.setattr(nightly_eval, "_ROCM_ROOTS", _roots(tmp_path / "absent"))
    assert nightly_eval.build_metadata()["lane"] == "gate"


def test_an_empty_version_does_not_shadow_a_valid_version_dev(tmp_path, monkeypatch):
    """An interrupted install leaves a zero-byte marker (#387).

    The loop used to break on mere existence, so the empty file won and the
    dashboard's ROCm column went blank with a perfectly good `version-dev`
    sitting behind it -- indistinguishable from having no ROCm at all.
    """
    root = tmp_path / "opt_rocm"
    _write_version(root, "version", "")
    _write_version(root, "version-dev", "7.2.4.50311-abc1234\n")
    monkeypatch.setattr(nightly_eval, "_ROCM_ROOTS", _roots(root))
    assert nightly_eval.build_metadata()["rocm"] == "7.2.4.50311-abc1234"


def test_a_whitespace_only_version_does_not_shadow_a_valid_one(tmp_path, monkeypatch):
    root = tmp_path / "opt_rocm"
    _write_version(root, "version", "   \n\t\n")
    _write_version(root, "version-dev", "7.14.0\n")
    monkeypatch.setattr(nightly_eval, "_ROCM_ROOTS", _roots(root))
    assert nightly_eval.build_metadata()["rocm"] == "7.14.0"


def test_an_unreadable_first_marker_falls_through_instead_of_raising(tmp_path, monkeypatch):
    """A directory named `.info/version` must not take down build_metadata().

    read_text() raised IsADirectoryError straight out of build_metadata, and
    nothing above it catches -- so the whole results document was lost over a
    cosmetic dashboard column. Same for a permission-denied marker.
    """
    root = tmp_path / "opt_rocm"
    (root / ".info" / "version").mkdir(parents=True)
    _write_version(root, "version-dev", "7.2.4\n")
    monkeypatch.setattr(nightly_eval, "_ROCM_ROOTS", _roots(root))
    assert nightly_eval.build_metadata()["rocm"] == "7.2.4"


def test_a_non_utf8_marker_is_null_not_a_crash(tmp_path, monkeypatch):
    """Undecodable is unusable -- but it must not raise, either.

    Null matches what `environment.py` reports for the same file, so the
    dashboard column and `rocm.version` agree. The point of the test is that
    the read is fail-soft: `read_text()` raised `UnicodeDecodeError`, which is
    not an `OSError` and so escaped `build_metadata()` entirely.
    """
    root = tmp_path / "opt_rocm"
    info = root / ".info"
    info.mkdir(parents=True)
    (info / "version").write_bytes(b"\xff\xfe7.2.4")
    monkeypatch.setattr(nightly_eval, "_ROCM_ROOTS", _roots(root))
    assert nightly_eval.build_metadata()["rocm"] is None


def test_a_non_utf8_marker_does_not_shadow_a_valid_one(tmp_path, monkeypatch):
    root = tmp_path / "opt_rocm"
    info = root / ".info"
    info.mkdir(parents=True)
    (info / "version").write_bytes(b"\xff\xfe")
    _write_version(root, "version-dev", "7.2.4\n")
    monkeypatch.setattr(nightly_eval, "_ROCM_ROOTS", _roots(root))
    assert nightly_eval.build_metadata()["rocm"] == "7.2.4"


def test_a_multibyte_version_is_not_reported_as_corrupt(tmp_path, monkeypatch):
    """The bounded read must not turn a split character into "unreadable".

    Contrived content, but it pins the reason the decode is incremental: a
    plain bytes.decode() on a truncated buffer raises, which would report a
    perfectly readable marker as non-UTF-8.
    """
    root = tmp_path / "opt_rocm"
    _write_version(root, "version", "7.2.4-" + "\u00e9" * 3000)
    monkeypatch.setattr(nightly_eval, "_ROCM_ROOTS", _roots(root))
    value = nightly_eval.build_metadata()["rocm"]
    assert value is not None and value.startswith("7.2.4-")
