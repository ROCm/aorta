"""C7: the eval hook. C7.T1 (request and result files validate against the schema and round-trip) and C7.T3
(cleanup keeps at most 3 checkpoints); cadence, unknown results and resume. C7.T2 needs TokenSpeed (GPU)."""

import dataclasses
import json
from pathlib import Path

import pytest

from aorlta.eval.hook import (ADAPTER_WARNING, MAX_CHECKPOINTS, EvalHook, EvalRequest, EvalResult, EvalSchemaError,
                              FakeEval, read_request, read_result, request_from_dict, result_from_dict, verify_export,
                              write_request, write_result)

SHA = "a" * 64


def _request(**kw) -> EvalRequest:
    base = dict(request_id="run-1.it000002.validation", run_id="run-1", iteration=2, policy_version="run-1:v2",
                checkpoint={"path": "/scratch/x/eval/checkpoints/run-1.it000002/merged", "format": "hf_merged_bf16",
                            "manifest_sha256": SHA, "base_sha256": "b" * 64},
                eval={"set": "validation", "modes": ["full", "evidence_only"], "samples_per_item": 4},
                requested_at="2026-11-02T17:00:00.000Z")
    base.update(kw)
    return EvalRequest(**base)


def _result(**kw) -> EvalResult:
    base = dict(request_id="run-1.it000002.validation", status="done",
                aggregates={"verified_repairs": {"full": 0.31, "evidence_only": 0.27}, "best_of_4": 0.42,
                            "pass_count_histogram": [10, 4, 3, 1, 15]},
                eval_version="partner-eval/3", completed_at="2026-11-02T18:00:00Z")
    base.update(kw)
    return EvalResult(**base)


# --- C7.T1 -------------------------------------------------------------------------------------------------------


def test_c7_t1_a_request_and_a_result_round_trip_through_their_files(tmp_path):
    req, res = _request(), _result()
    write_request(tmp_path / "req.json", req)
    write_result(tmp_path / "res.json", res)
    assert read_request(tmp_path / "req.json") == req
    assert read_result(tmp_path / "res.json") == res
    assert request_from_dict(json.loads((tmp_path / "req.json").read_text())) == req
    assert set(json.loads((tmp_path / "req.json").read_text())) == {f.name for f in dataclasses.fields(EvalRequest)}


REQUEST_BREAKS = {
    "unknown key": lambda d: d.__setitem__("priority", 1),
    "missing key": lambda d: d.pop("requested_at"),
    "wrong type": lambda d: d.__setitem__("iteration", "2"),
    "iteration 0": lambda d: d.update(iteration=0, policy_version="run-1:v0"),
    "version not the iteration's": lambda d: d.__setitem__("policy_version", "run-1:v1"),
    "bad run id": lambda d: d.update(run_id="run/1", policy_version="run/1:v2"),
    "request id with a slash": lambda d: d.__setitem__("request_id", "a/b"),
    "request id ending .partial": lambda d: d.__setitem__("request_id", "x.partial"),
    "relative path": lambda d: d["checkpoint"].__setitem__("path", "eval/merged"),
    "unknown format": lambda d: d["checkpoint"].__setitem__("format", "gguf"),
    "short sha": lambda d: d["checkpoint"].__setitem__("manifest_sha256", "abc"),
    "checkpoint key": lambda d: d["checkpoint"].__setitem__("tensors", 738),
    "unknown set": lambda d: d["eval"].__setitem__("set", "test"),
    "no modes": lambda d: d["eval"].__setitem__("modes", []),
    "repeated mode": lambda d: d["eval"].__setitem__("modes", ["full", "full"]),
    "unknown mode": lambda d: d["eval"].__setitem__("modes", ["fast"]),
    "zero samples": lambda d: d["eval"].__setitem__("samples_per_item", 0),
    "eval key": lambda d: d["eval"].__setitem__("seed", 1),
    "local time": lambda d: d.__setitem__("requested_at", "2026-11-02 17:00:00"),
}


@pytest.mark.parametrize("name", sorted(REQUEST_BREAKS))
def test_c7_t1_a_request_that_breaks_the_schema_is_refused(name):
    d = json.loads(json.dumps(dataclasses.asdict(_request())))
    REQUEST_BREAKS[name](d)
    with pytest.raises(EvalSchemaError):
        request_from_dict(d)


RESULT_BREAKS = {
    "unknown key": lambda d: d.__setitem__("note", "x"),
    "missing key": lambda d: d.pop("eval_version"),
    "unknown status": lambda d: d.__setitem__("status", "ok"),
    "empty version": lambda d: d.__setitem__("eval_version", ""),
    "not a time": lambda d: d.__setitem__("completed_at", "yesterday"),
    "aggregates not an object": lambda d: d.__setitem__("aggregates", [1, 2]),
    "aggregates not finite": lambda d: d["aggregates"].__setitem__("best_of_4", float("inf")),
}


@pytest.mark.parametrize("name", sorted(RESULT_BREAKS))
def test_c7_t1_a_result_that_breaks_the_schema_is_refused(name):
    d = dataclasses.asdict(_result())
    RESULT_BREAKS[name](d)
    with pytest.raises(EvalSchemaError):
        result_from_dict(d)


# --- the hook, driven directly ------------------------------------------------------------------------------------


class Exporter:
    def __init__(self, fail=()):
        self.calls, self.fail = [], set(fail)

    def export(self, version, merged_dir: Path, adapter_dir: Path):
        self.calls.append(version)
        if version in self.fail:
            raise OSError("disk full")
        (merged_dir / "model.safetensors").write_bytes(b"w" * 32 + bytes([version]))
        (merged_dir / "config.json").write_text("{}")
        (adapter_dir / "adapter_model.safetensors").write_bytes(b"a" * 16 + bytes([version]))
        return {"tensors": 738}


def make_hook(tmp_path, every=1, final=99, exporter=None, **kw):
    return EvalHook(tmp_path / "eval", run_id="run-1", every=every, final_iteration=final,
                    exporter=exporter or Exporter(), base_sha256="b" * 64, background=False, **kw)


def held(hook):
    return sorted(p.name for p in hook.checkpoints.iterdir())


def test_c7_t1_the_files_the_hook_writes_validate_and_name_a_verifying_checkpoint(tmp_path):
    hook = make_hook(tmp_path, every=2, final=4)
    for n in (1, 2, 3, 4):
        hook.after_commit(n)
    reqs = [read_request(p) for p in sorted(hook.inbox.glob("*.json"))]
    assert [(r.iteration, r.eval["set"]) for r in reqs] == [(2, "validation"), (4, "heldout"), (4, "validation")]
    for r in reqs:
        export = Path(r.checkpoint["path"]).parent
        assert r.checkpoint["format"] == "hf_merged_bf16" and Path(r.checkpoint["path"]).name == "merged"
        assert verify_export(export, r.checkpoint["manifest_sha256"]) == []
        warning = json.loads((export / "adapter" / "aorlta_warning.json").read_text())
        assert warning["warning"] == ADAPTER_WARNING and warning["policy_version"] == r.policy_version
    assert hook.exporter.calls == [2, 4]  # one export per iteration, shared by its two requests
    FakeEval(hook.inbox, hook.outbox).process()
    for p in sorted(hook.outbox.glob("*.json")):
        assert read_result(p).status == "done"


def test_c7_t3_cleanup_keeps_at_most_3_checkpoints(tmp_path):
    hook = make_hook(tmp_path, every=1)
    for n in range(1, 7):
        hook.after_commit(n)
        assert len(held(hook)) <= MAX_CHECKPOINTS
    assert held(hook) == ["run-1.it000001", "run-1.it000002", "run-1.it000003"]
    skipped = [e for e in hook.events if e["event"] == "skipped"]
    assert [(e["iteration"], e["reason"]) for e in skipped] == [(4, "checkpoint_cap"), (5, "checkpoint_cap"),
                                                                (6, "checkpoint_cap")]
    # results arrive (one failed): their checkpoints are deleted, and the next due request goes out
    FakeEval(hook.inbox, hook.outbox).process(limit=1)
    FakeEval(hook.inbox, hook.outbox, status="failed").process(limit=1)
    seen = hook.poll(6)
    assert sorted(s["status"] for s in seen) == ["done", "failed"]
    assert held(hook) == ["run-1.it000003"]
    hook.after_commit(7)
    assert held(hook) == ["run-1.it000003", "run-1.it000007"]


def test_a_checkpoint_shared_by_two_requests_waits_for_both_results(tmp_path):
    hook = make_hook(tmp_path, every=2, final=2)  # iteration 2 is due for validation and held-out
    hook.after_commit(2)
    FakeEval(hook.inbox, hook.outbox).process(limit=1)
    hook.poll(2)
    assert held(hook) == ["run-1.it000002"]
    FakeEval(hook.inbox, hook.outbox).process()
    hook.poll(2)
    assert held(hook) == []


def test_a_result_for_an_unknown_request_or_a_bad_file_is_ignored_and_logged_once(tmp_path, caplog):
    hook = make_hook(tmp_path)
    hook.after_commit(1)
    hook.outbox.mkdir(parents=True, exist_ok=True)
    write_result(hook.outbox / "run-1.it000009.validation.json", _result(request_id="run-1.it000009.validation"))
    (hook.outbox / "run-1.it000001.validation.json").write_text('{"request_id": "run-1.it000001.validation"}')
    assert hook.poll(1) == [] and hook.poll(1) == []
    ignored = [(e["request_id"], e["reason"]) for e in hook.events if e["event"] == "ignored"]
    assert ignored == [("run-1.it000001.validation", "invalid"), ("run-1.it000009.validation", "unknown_request")]
    assert held(hook) == ["run-1.it000001"]  # an invalid result never frees a checkpoint


def test_the_cadence_is_every_n_plus_held_out_at_the_end_and_none_without_eval_every(tmp_path):
    hook = make_hook(tmp_path, every=3, final=7)
    assert [hook.due_sets(n) for n in range(1, 8)] == [[], [], ["validation"], [], [], ["validation"], ["heldout"]]
    assert make_hook(tmp_path / "x", every=None, final=7).due_sets(7) == []


def test_an_export_that_fails_is_logged_and_never_raises(tmp_path):
    hook = make_hook(tmp_path, exporter=Exporter(fail={2}))
    hook.after_commit(1)
    hook.after_commit(2)
    hook.after_commit(3)
    assert [(e["iteration"], e["reason"]) for e in hook.events if e["event"] == "export_failed"] == [(2, "OSError")]
    assert held(hook) == ["run-1.it000001", "run-1.it000003"]


def test_a_tampered_checkpoint_gets_a_failed_result_from_the_fake_eval(tmp_path):
    hook = make_hook(tmp_path)
    hook.after_commit(1)
    (hook.checkpoints / "run-1.it000001" / "merged" / "model.safetensors").write_bytes(b"x")
    FakeEval(hook.inbox, hook.outbox).process()
    assert read_result(hook.outbox / "run-1.it000001.validation.json").status == "failed"


def test_reconcile_finishes_what_a_crash_between_commit_and_request_left(tmp_path):
    hook = make_hook(tmp_path, every=1)
    hook.after_commit(1)
    # a crash after iteration 2's export was renamed but before its request: and a half-written one for 3
    eid = "run-1.it000002"
    hook._dirs()
    export = hook.checkpoints / eid
    (export / "merged").mkdir(parents=True)
    (export / "adapter").mkdir()
    Exporter().export(2, export / "merged", export / "adapter")
    from aorlta.eval.hook import files_manifest
    from aorlta.atomic import atomic_write_json
    from aorlta.ids import sha256_json
    files = files_manifest(export)
    atomic_write_json(export / "manifest.json", {"schema": "aorlta.eval_manifest/1", "export_id": eid, "iteration": 2,
                                                 "policy_version": "run-1:v2", "files": files,
                                                 "files_sha256": sha256_json(files)}, canonical=True)
    (hook.checkpoints / "run-1.it000003.partial").mkdir()
    fresh = make_hook(tmp_path, every=1)  # a new process
    issued = fresh.reconcile(2, can_export=lambda m: m == 2)
    assert issued == ["validation"] and fresh.exporter.calls == []  # reused, not re-exported
    assert held(fresh) == ["run-1.it000001", "run-1.it000002"]
    req = read_request(fresh.inbox / "run-1.it000002.validation.json")
    assert verify_export(export, req.checkpoint["manifest_sha256"]) == []
    assert [e["reused"] for e in fresh.events if e["event"] == "requested"] == [False, True]


def test_reconcile_skips_a_due_request_whose_weights_are_no_longer_held(tmp_path):
    hook = make_hook(tmp_path, every=1)
    fresh_issued = hook.reconcile(3, can_export=lambda m: m == 3)
    assert fresh_issued == ["validation"]
    assert [(e["iteration"], e["reason"]) for e in hook.events if e["event"] == "skipped"] == [
        (1, "weights_not_held_at_resume"), (2, "weights_not_held_at_resume")]


def test_the_hook_creates_nothing_before_it_is_used(tmp_path):
    make_hook(tmp_path)
    assert not (tmp_path / "eval").exists()  # the run directory must hold nothing before run_plan.json


# --- the optional `source` field -----------------------------------------------------------------------------------


def test_source_is_optional_on_a_request_and_round_trips_when_given(tmp_path):
    data = dataclasses.asdict(_request())
    del data["source"]
    assert request_from_dict(data).source is None  # a request written without it still reads
    for source in ("standin",):
        req = _request(source=source)
        write_request(tmp_path / f"{source}.json", req)
        assert read_request(tmp_path / f"{source}.json") == req


@pytest.mark.parametrize("bad", ["partner", "", 3])
def test_a_source_other_than_standin_is_refused(tmp_path, bad):
    data = dataclasses.asdict(_request())
    data["source"] = bad
    with pytest.raises(EvalSchemaError):
        request_from_dict(data)
    with pytest.raises(ValueError):
        make_hook(tmp_path, source="partner")


def test_the_hook_stamps_its_runs_source_on_every_request(tmp_path):
    hook = make_hook(tmp_path, every=1, final=2, source="standin")
    for n in (1, 2):
        hook.after_commit(n)
    reqs = [read_request(p) for p in sorted(hook.inbox.glob("*.json"))]
    assert reqs and {r.source for r in reqs} == {"standin"}
    unstamped = make_hook(tmp_path / "b", every=1, final=1)
    unstamped.after_commit(1)
    assert {read_request(p).source for p in unstamped.inbox.glob("*.json")} == {None}
