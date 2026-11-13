"""The Q&A example (examples/qa): a task with no compile step and no sanitizer, on CPU. Its configs load with the
plugin, its answers are read and normalised, its parse gives a repair turn to a reply with no answer line, its check
matches against the reference and asks the judge command, its reward keeps the service's invariants, and it trains
through the real wiring on CPU with test_wiring_e2e's tiny model and engines."""

from __future__ import annotations

import json
import stat
import sys
from pathlib import Path

import pytest

from aorlta import plugins
from aorlta.config import config_from_dict, load_config, parse_yaml

ROOT = Path(__file__).resolve().parents[2]
QA = ROOT / "examples" / "qa"
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tests" / "orchestrator"))
import test_wiring_e2e as E  # noqa: E402 - the tiny model and its engines
from test_plugins import CharChat, registry  # noqa: E402,F401

tiny_base = E.tiny_base


@pytest.fixture
def qa(registry):  # noqa: F811
    plugins.load_plugins(["examples/qa/qa_task.py"])
    return sys.modules[plugins.identity()["examples/qa/qa_task.py"]["module"]]


def site_config(tmp_path, name, pool, **model):
    site = tmp_path / "site.yaml"
    site.write_text(json.dumps({"run": {"root": str(tmp_path / "runs")},
                                "model": {"base_dir": str(tmp_path / "base"), **model},
                                "data": {"prompt_pools": [str(QA / pool)]}}))
    return load_config(ROOT / "configs" / f"{name}.yaml", site=site)


@pytest.mark.parametrize("name, pool", [("qa-train", "gpu-facts/train"), ("qa-score", "gpu-facts/heldout"),
                                        ("arith-train", "arith/train"), ("arith-score", "arith/heldout")])
def test_the_configs_load_and_run_on_one_gpu_with_no_compile(tmp_path, registry, name, pool):  # noqa: F811
    from aorlta.orchestrator import wiring

    cfg = site_config(tmp_path, name, pool)
    assert cfg.rollout.compile_adapter == "none" and cfg.rollout.engines == (0,)
    assert cfg.reward.profile == "qa-match" and (cfg.trainer.ranks == 1 or cfg.orchestrator.mode == "score")
    wiring.check_supported(cfg)


def test_the_profile_has_no_compile_or_sanitizer_leg(qa):
    from aorlta.reward.profiles import load_profile

    assert load_profile("qa-match").legs == ("ref.answer",)


def test_the_held_out_questions_are_not_trained_on(qa):
    for d in ("gpu-facts", "arith"):
        train = {r["question"] for r in qa.load_rows(QA / d / "train")}
        held = {r["question"] for r in qa.load_rows(QA / d / "heldout")}
        assert train and held and not train & held
    assert len(qa.load_rows(QA / "gpu-facts" / "train")) + len(qa.load_rows(QA / "gpu-facts" / "heldout")) == 24


def test_the_arith_files_are_what_make_arith_writes(tmp_path, monkeypatch):
    import importlib.util

    spec = importlib.util.spec_from_file_location("make_arith", QA / "make_arith.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "HERE", tmp_path)
    mod.main()
    for part in ("train", "heldout"):
        assert (tmp_path / "arith" / part / "qa.jsonl").read_bytes() == (QA / "arith" / part / "qa.jsonl").read_bytes()
    for row in json.loads("[" + ",".join((QA / "arith" / "heldout" / "qa.jsonl").read_text().splitlines()) + "]"):
        a, b = row["question"].removeprefix("What is ").removesuffix("?").split(" × ")
        assert int(row["answer"]) == int(a) * int(b)


@pytest.mark.parametrize("reply, answer", [
    ("Some working.\nAnswer: 1,024.", "1024"), ("**Answer:** `hipMalloc`", "hipmalloc"),
    ("answer: Local Data Share", "local data share"), ("Answer: 3\nthen\nAnswer:  42 ", "42"),
])
def test_the_final_answer_is_read_and_normalised(qa, reply, answer):
    assert qa.normalise(qa.final_answer(reply)) == answer


def test_no_answer_line_is_no_answer(qa):
    assert qa.final_answer("The answer is 42.") is None


def test_the_parse_gives_a_repair_turn_to_a_reply_without_an_answer(qa):
    from aorlta.rollout.chain import make_feedback_template, make_parse_adapter
    from aorlta.rollout.compile_check import make_compiler

    parse, fb = make_parse_adapter("qa.parse"), make_feedback_template("qa.feedback")
    out = parse.check("I think it is 64.", "off", None)
    assert out.kind == "rejected" and out.rejection[1] == "no_answer" and "Answer:" in fb.build_apply(*out.rejection).text
    good = parse.check("Working...\nAnswer: 64", "off", None)
    assert good.kind == "edits" and good.code == "64"
    assert make_compiler("none", Path("/nonexistent")).check(good.code, "c0-0-t0").ok


def job_for(qa, tmp_path, row, reply):
    from aorlta.schemas import LegJob

    resp = tmp_path / "response.txt"
    resp.write_text(reply)
    return LegJob(job_id="j", leg="ref.answer", candidate_id="c", control_id=None, source_sha256="s" * 64,
                  task_sha256=qa.row_sha256(row), inputs={"response": str(resp)}, output_dir=str(tmp_path))


def adapter(qa, settings=None):
    ctx = plugins.CheckContext(name="qa.match", worker_id="match/1", settings=settings or {}, clock=lambda: 0.0,
                               timeout_s=30.0, pools=(str(QA / "gpu-facts" / "train"),))
    return plugins.CHECK_ADAPTERS["qa.match"](ctx)


@pytest.mark.parametrize("reply, want", [
    ("Answer: 64", ("clean", None)), ("Answer: 64 threads", ("clean", None)), ("Answer: 32", ("fail", "wrong_answer")),
    ("sixty-four", ("fail", "no_answer")),
])
def test_the_check_matches_the_reference_and_its_aliases(qa, tmp_path, reply, want):
    row = next(r for r in qa.load_rows(QA / "gpu-facts" / "train") if r["id"] == "wave-cdna")
    res = adapter(qa).run(job_for(qa, tmp_path, row, reply))
    assert (res.state, res.reason) == want


@pytest.mark.parametrize("judge_out, want", [
    ('{"verdict": "correct"}', ("clean", None)), ('{"verdict": "wrong"}', ("fail", "wrong_answer")),
    ('not json', ("unsettled", "judge_no_verdict")), ('{"verdict": "maybe"}', ("unsettled", "judge_no_verdict")),
])
def test_a_judge_command_decides_an_answer_that_does_not_match(qa, tmp_path, judge_out, want):
    """The shape of a task whose reward is an external tool's verdict: the judge reads the question, the reference
    and the answer on stdin and prints its verdict; no verdict is unsettled, never clean."""
    judge = tmp_path / "judge"
    judge.write_text(f"#!/bin/bash\ncat > {tmp_path}/judge-input.json\necho '{judge_out}'\n")
    judge.chmod(judge.stat().st_mode | stat.S_IXUSR)
    row = next(r for r in qa.load_rows(QA / "gpu-facts" / "train") if r["id"] == "wave-cdna")
    res = adapter(qa, {"judge": [str(judge)]}).run(job_for(qa, tmp_path, row, "Answer: a wave of sixty-four"))
    assert (res.state, res.reason) == want
    seen = json.loads((tmp_path / "judge-input.json").read_text())
    assert seen == {"question": row["question"], "reference": "64", "answer": "a wave of sixty-four"}


def test_the_reward_keeps_the_services_invariants(qa):
    from aorlta.reward.service import REWARD_MAX, REWARD_MIN
    from aorlta.schemas import Candidate, LegResult, TaskRef

    fn = plugins.REWARD_FNS["qa.v1"](None)
    for state, reason, base in (("clean", None, 1.0), ("fail", "wrong_answer", -0.5), ("fail", "no_answer", -1.0),
                                ("fail", "other", None), ("unsettled", "x", None), ("not_run", None, None)):
        for tokens in (0, 512):
            leg = LegResult("ref.answer", state, reason, [], 1, 1, None, None, False, 0.0, 0.0, 0.0)
            c = Candidate("c", "r", 1, "qa:x", 0, "v", "", tokens, 512, TaskRef("qa:x", None, None, None, "t"),
                          "qa-match", None)
            reward, parts = fn.compute({"ref.answer": leg}, c)
            d = fn.decide({"ref.answer": leg}, c)
            if base is None:
                assert reward is None and type(d).__name__ == "Unscored"
                continue
            assert parts["base"] == base and REWARD_MIN <= reward <= REWARD_MAX
            assert parts["base"] + parts["broke_healthy"] + parts["length"] == reward and (d.reward, d.parts) == (
                reward, parts)
            assert fn.claims_clean(reward, parts) == (state == "clean")
    assert type(fn.decide({}, c)).__name__ == "NeedMore"


@pytest.mark.heavy
def test_the_qa_example_trains_on_cpu_through_the_real_wiring(tmp_path, qa, tiny_base):
    """No compile step, no sanitizer, the real answer check: the fake engine never writes an `Answer:` line, so
    every chain gets qa.parse's repair turn and is scored no_answer, and the iteration still trains and lands."""
    from aorlta.orchestrator import wiring
    from aorlta.rollout.engine_proc import InProcessEngine

    data = parse_yaml((ROOT / "configs" / "qa-train.yaml").read_bytes())
    data["run"].update(root=str(tmp_path / "runs"), run_id="qa-cpu")
    data["model"]["base_dir"] = str(tiny_base)
    data["data"]["prompt_pools"][0]["path"] = str(QA / "gpu-facts" / "train")
    data["rollout"].update(engines=[0, 1], K=4, P=4, request_timeout_s=300)
    data["orchestrator"]["iterations"] = 1
    data["trainer"] = {"ranks": 1, "lr": 1e-2, "micro_batch_tokens": 2048}
    data["sync"] = {"dir": str(tmp_path / "shm")}
    cfg = config_from_dict(data)
    engines = [E.TwinEngine(tiny_base, E.FAKE, i) for i in range(2)]
    parts = wiring.Parts(tokenizer=E.ByteTokenizer(), handles=[InProcessEngine(i, e) for i, e in enumerate(engines)],
                         load_model=E.load_trainer_model, trainer_device="cpu", require_fast_path=False,
                         probes=E.PROBES)
    result = wiring.build(cfg, parts=parts).run()
    assert result["status"] == "completed", result
    s = E.summary(cfg, 1)
    assert s["samples"]["scored"] == s["samples"]["total"] - sum(1 for e in E.episodes(cfg, 1)
                                                                 if e["end_reason"] == "engine_error")
    assert s["sync"]["landing"] == "LANDED"
    eps = [e for e in E.episodes(cfg, 1) if e["end_reason"] != "engine_error"]
    assert all(len(e["turns"]) == 2 for e in eps)  # the answer, and qa.parse's repair turn
    scored = [e for e in eps if e["status"] == "scored"]
    assert scored and all(e["legs"]["ref.answer"] == "fail" and -1.05 <= e["reward"] <= -1.0 for e in scored)


def test_the_judge_is_part_of_the_checkers_identity(qa, tmp_path):
    """Cached ref.answer legs are keyed by the checker's version: turning the judge on, changing its command, its
    timeout or its script is a new version, so a cached wrong_answer is not reused where the judge would decide."""
    judge = tmp_path / "judge"
    judge.write_text("#!/bin/bash\necho '{\"verdict\": \"correct\"}'\n")
    judge.chmod(0o755)
    versions = [adapter(qa).version, adapter(qa, {"judge": [str(judge)]}).version,
                adapter(qa, {"judge": [str(judge), "--strict"]}).version,
                adapter(qa, {"judge": [str(judge)], "judge_timeout_s": 5}).version]
    assert len(set(versions)) == 4 and versions[0] == "1"
    assert adapter(qa, {"judge": [str(judge)]}).version == versions[1]
    judge.write_text("#!/bin/bash\necho '{\"verdict\": \"wrong\"}'\n")
    assert adapter(qa, {"judge": [str(judge)]}).version != versions[1]


def test_a_judge_that_times_out_or_is_retired_leaves_nothing_running(qa, tmp_path):
    """The judge's whole process group is killed, on a timeout and when the worker is retired (close()), not only
    the command itself: a wrapper's children went on running before."""
    import os
    import threading
    import time

    row = next(r for r in qa.load_rows(QA / "gpu-facts" / "train") if r["id"] == "wave-cdna")
    for how in ("timeout", "close"):
        pid = tmp_path / f"{how}.pid"
        judge = tmp_path / f"judge-{how}"
        judge.write_text(f"#!/bin/bash\nsleep 60 &\necho $! > {pid}\nwait\n")
        judge.chmod(0o755)
        a = adapter(qa, {"judge": [str(judge)], "judge_timeout_s": 1.0 if how == "timeout" else 60.0})
        if how == "close":
            threading.Thread(target=lambda: (time.sleep(1.0), a.close()), daemon=True).start()
        t0 = time.monotonic()
        res = a.run(job_for(qa, tmp_path, row, "Answer: a wave of sixty-four"))
        assert (res.state, res.reason) == ("unsettled", "judge_no_verdict")
        assert time.monotonic() - t0 < 15, how  # not the adapter's own 30 s, nor the child's 60 s
        child = int(pid.read_text())
        deadline = time.monotonic() + 5
        while os.path.exists(f"/proc/{child}") and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not os.path.exists(f"/proc/{child}"), how
        a.close()


def test_a_judge_found_through_path_is_part_of_the_checkers_identity(qa, tmp_path, monkeypatch):
    """A bare command name is resolved as PATH finds it: replacing that executable, or which one PATH selects, is a new
    checker version."""
    bins = [tmp_path / "a", tmp_path / "b"]
    for d in bins:
        d.mkdir()
        (d / "my-judge").write_text("#!/bin/bash\necho '{\"verdict\": \"correct\"}'\n")
        (d / "my-judge").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bins[0]}:/usr/bin:/bin")
    first = adapter(qa, {"judge": ["my-judge"]}).version
    assert adapter(qa, {"judge": ["my-judge"]}).version == first
    (bins[0] / "my-judge").write_text("#!/bin/bash\necho '{\"verdict\": \"wrong\"}'\n")
    second = adapter(qa, {"judge": ["my-judge"]}).version
    monkeypatch.setenv("PATH", f"{bins[1]}:/usr/bin:/bin")
    assert len({first, second, adapter(qa, {"judge": ["my-judge"]}).version}) == 3
