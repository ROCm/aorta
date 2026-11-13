"""The vertical starters (examples/verticals): three plugin packages, each on its own. Their configs load with the
plugin, their pools keep the held-out prompts out of training, their parse gives a repair turn to a reply with nothing
to check, their placeholder check gives each failure its own reason and anything it cannot read is unsettled, their
placeholder reward keeps the service's invariants and leaves the unsettled unscored, and each scores and trains through
the real wiring on CPU with test_wiring_e2e's tiny model and engines, the engines' replies drawn from a few written for
the vertical."""

from __future__ import annotations

import dataclasses
import json
import re
import stat
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from aorlta import plugins
from aorlta.config import config_from_dict, load_config, parse_yaml
from aorlta.episodes.schema import LEG_RE

ROOT = Path(__file__).resolve().parents[2]
VERTICALS = ROOT / "examples" / "verticals"
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tests" / "orchestrator"))
import test_wiring_e2e as E  # noqa: E402 - the tiny model and its engines
from test_plugins import CharChat, registry  # noqa: E402,F401

tiny_base = E.tiny_base


@dataclass(frozen=True)
class Vertical:
    name: str  # the directory, the task, the profile and every component's prefix
    module: str  # the plugin file, beside its profile and configs
    pool_file: str
    leg: str
    check: str  # the check adapter's name
    good: str  # a reply the placeholder check passes, for the first training prompt
    bad: tuple[tuple[str, str], ...]  # (reply, fail reason) for the first training prompt; each reason at least once
    nothing: str  # a reply with nothing to check: the parse's repair turn, and the check's first reason

    @property
    def plugin(self) -> str:
        return f"examples/verticals/{self.name}/{self.module}"

    @property
    def reasons(self) -> set[str]:
        return {r for _, r in self.bad}


ISA = Vertical(
    "isa-fuzzer", "isa_fuzzer.py", "tasks.jsonl", "ref.program", "isa-fuzzer.check",
    good="Here it is.\n```toy-asm\nMOV r1, #3\nADD r2, r1, r1\n```",
    bad=(("No block here, sorry.", "no_program"), ("```\nMOV r1, #3\nADD r2, r1\n```", "malformed"),
         ("```\nMOV r1, #3\nJMP r2\n```", "malformed"), ("```\nMOV r9, #3\nADD r2, r1, r1\n```", "malformed"),
         ("```\nNOP r1\nADD r2, r1, r1\n```", "malformed"),
         ("```\nADD r1, r1, #1\nADD r1, r1, #1\nADD r1, r1, #1\nADD r1, r1, #1\nADD r1, r1, #1\n```", "wrong_length"),
         ("```\nMOV r1, #3\nSUB r2, r1, r1\n```", "missing_op")),
    nothing="I would add two registers.")
QA = Vertical(
    "qa-expert", "qa_expert.py", "questions.jsonl", "ref.answer", "qa-expert.match",
    good="Two and three make five.\n**Answer:** `Five.`",
    bad=(("It is five.", "no_answer"), ("Answer: 5\nOn second thought:\nAnswer: 6", "wrong_answer")),
    nothing="It is five.")
TRIAGE = Vertical(
    "sanitizer-triage", "sanitizer_triage.py", "reports.jsonl", "ref.verdict", "sanitizer-triage.check",
    good='Looks clean.\n```json\n{"category": "no_issue", "rationale": "The report lists no findings."}\n```',
    bad=(("No findings, so it is fine.", "no_verdict"), ('{"category": "no_issue"}', "bad_keys"),
         ('{"category": "fine", "rationale": "No findings."}', "bad_category"),
         ('{"category": "no_issue", "rationale": "Ok."}\nNo\n{"category": "fine", "rationale": "Ok."}', "bad_category"),
         ('{"category": "no_issue", "rationale": "No findings.", "extra": 1}', "bad_keys"),
         ('{"category": "no_issue", "rationale": "  "}', "bad_rationale"),
         ('{"category": "no_issue", "rationale": "%s"}' % ("x" * 401), "bad_rationale")),
    nothing="No findings, so it is fine.")
ALL = (ISA, QA, TRIAGE)
BY_NAME = pytest.mark.parametrize("v", ALL, ids=lambda v: v.name)


@pytest.fixture
def load(registry):  # noqa: F811
    """Load a vertical's plugin as a run config loads it; its module."""
    def _load(v):
        plugins.load_plugins([v.plugin])
        return sys.modules[plugins.identity()[v.plugin]["module"]]
    return _load


def site_config(tmp_path, v, kind):
    site = tmp_path / "site.yaml"
    split = "heldout" if kind == "score" else "train"
    site.write_text(json.dumps({"run": {"root": str(tmp_path / "runs")}, "model": {"base_dir": str(tmp_path / "base")},
                                "data": {"prompt_pools": [str(VERTICALS / v.name / "pool" / split)]}}))
    return load_config(VERTICALS / v.name / f"{kind}.yaml", site=site)


@BY_NAME
@pytest.mark.parametrize("kind", ["score", "train"])
def test_the_configs_load_with_the_plugin_and_run_on_one_gpu_with_no_compile(tmp_path, registry, v, kind):  # noqa: F811
    from aorlta.orchestrator import wiring

    cfg = site_config(tmp_path, v, kind)
    assert cfg.plugins == (v.plugin,) and cfg.data.prompt_pools[0].task == v.name
    assert (cfg.reward.profile, cfg.reward.reward_fn) == (v.name, f"{v.name}.v1")
    assert (cfg.rollout.parse_adapter, cfg.rollout.feedback_template) == (f"{v.name}.parse", f"{v.name}.feedback")
    assert cfg.rollout.compile_adapter == "none" and cfg.rollout.engines == (0,)
    assert cfg.orchestrator.mode == kind if kind == "score" else cfg.trainer.ranks == 1
    wiring.check_supported(cfg)


@BY_NAME
def test_the_profile_has_one_leg_and_no_compile_or_sanitizer(load, v):
    from aorlta.reward.profiles import load_profile

    load(v)
    profile = load_profile(v.name)
    assert profile.legs == (v.leg,) and {p.adapter for p in profile.pools.values()} == {v.check}
    assert LEG_RE.fullmatch(v.leg)  # an episode record names its legs gate.*, ref.* or san.*


@BY_NAME
def test_the_pools_are_placeholders_and_the_held_out_prompts_are_not_trained_on(load, v):
    mod = load(v)
    train, held = (mod.load_rows(VERTICALS / v.name / "pool" / s) for s in ("train", "heldout"))
    assert len(train) == len(held) == 4  # the self-test and a score batch need 4 prompts
    assert not {r["id"] for r in train} & {r["id"] for r in held}
    assert not {mod.row_sha256(r) for r in train} & {mod.row_sha256(r) for r in held}
    assert all("placeholder" in json.dumps(r).lower() for r in train + held)


@BY_NAME
def test_the_records_name_the_task_and_hash_what_the_check_reads(load, v):
    from aorlta.ids import sha256_ids
    from aorlta.rollout.templates import render_turn0

    mod = load(v)
    pool = VERTICALS / v.name / "pool" / "train"
    recs = plugins.task(v.name).records(pool, CharChat(), "off")
    rows = mod.load_rows(pool)
    assert [r.prompt_id for r in recs] == [f"{v.name}:{row['id']}" for row in rows]
    assert [r.task.task_sha256 for r in recs] == [mod.row_sha256(row) for row in rows]
    for key in rows[0]:  # every field the check reads moves the hash, so a cached result is never another task's
        assert mod.row_sha256({**rows[0], key: "changed"}) != mod.row_sha256(rows[0]), key
    assert all(r.prompt_ids_sha256 == sha256_ids(render_turn0(CharChat(), list(r.messages), "off")) for r in recs)
    assert len({r.messages[0]["content"] for r in recs}) == 1  # one system message, as the probes need


@BY_NAME
def test_the_parse_gives_a_repair_turn_to_a_reply_with_nothing_to_check(load, v):
    from aorlta.rollout.chain import make_feedback_template, make_parse_adapter

    load(v)
    parse, fb = make_parse_adapter(f"{v.name}.parse"), make_feedback_template(f"{v.name}.feedback")
    out = parse.check(v.nothing, "off", None)
    assert out.kind == "rejected" and out.rejection[1] == v.bad[0][1]
    assert fb.build_apply(*out.rejection).text == out.rejection[0]
    assert parse.check(v.good, "off", None).kind == "edits"
    assert parse.check(f"<think>{v.nothing}</think>\n{v.good}", "on", None).kind == "edits"
    assert parse.check(f"<think>{v.good}</think>\n{v.nothing}", "on", None).kind == "rejected"
    for reply, _ in v.bad[1:]:  # a reply that is wrong but present is the check's to score, not the parse's
        assert parse.check(reply, "off", None).kind == "edits"


def job_for(mod, tmp_path, row, reply, leg):
    from aorlta.schemas import LegJob

    resp = tmp_path / "response.txt"
    resp.write_text(reply)
    return LegJob(job_id="j", leg=leg, candidate_id="c", control_id=None, source_sha256="s" * 64,
                  task_sha256=mod.row_sha256(row), inputs={"response": str(resp)}, output_dir=str(tmp_path))


def adapter(v, settings=None):
    ctx = plugins.CheckContext(name=v.check, worker_id="w/1", settings=settings or {}, clock=lambda: 0.0,
                               timeout_s=30.0, pools=(str(VERTICALS / v.name / "pool" / "train"),))
    return plugins.CHECK_ADAPTERS[v.check](ctx)


@BY_NAME
def test_the_check_gives_each_failure_its_own_reason(load, tmp_path, v):
    mod = load(v)
    row = mod.load_rows(VERTICALS / v.name / "pool" / "train")[0]
    check = adapter(v)
    assert (check.run(job_for(mod, tmp_path, row, v.good, v.leg)).state, check.legs) == ("clean", (v.leg,))
    for reply, reason in v.bad:
        res = check.run(job_for(mod, tmp_path, row, reply, v.leg))
        assert (res.state, res.reason) == ("fail", reason), reply


@BY_NAME
def test_what_the_check_cannot_read_is_unsettled_never_clean(load, tmp_path, v):
    mod = load(v)
    row = mod.load_rows(VERTICALS / v.name / "pool" / "train")[0]
    check = adapter(v)
    unknown = dict(row, id="not-in-any-pool")
    held = mod.load_rows(VERTICALS / v.name / "pool" / "heldout")[0]  # a pool the check was not given
    jobs = [job_for(mod, tmp_path, unknown, v.good, v.leg), job_for(mod, tmp_path, held, v.good, v.leg),
            job_for(mod, tmp_path, row, v.good, "some.other.leg")]
    good = job_for(mod, tmp_path, row, v.good, v.leg)
    jobs += [dataclasses.replace(good, inputs={}),
             dataclasses.replace(good, inputs={"response": str(tmp_path / "missing.txt")})]
    for job in jobs:
        res = check.run(job)
        assert (res.state, res.reason) == ("unsettled", "adapter_error"), job


def tool(tmp_path, out: str, rc: int = 0) -> list[str]:
    t = tmp_path / "tool"
    t.write_text(f"#!/bin/bash\ncat > {tmp_path}/tool-input.json\necho '{out}'\nexit {rc}\n")
    t.chmod(t.stat().st_mode | stat.S_IXUSR)
    return [str(t)]


@pytest.mark.parametrize("out, rc, want", [
    ('{"verdict": "pass"}', 0, ("clean", None)), ('{"verdict": "fail"}', 0, ("fail", "tool_rejected")),
    ("not json", 0, ("unsettled", "tool_no_verdict")), ('{"verdict": "maybe"}', 0, ("unsettled", "tool_no_verdict")),
    ('{"verdict": "pass"}', 3, ("unsettled", "tool_no_verdict")),
])
def test_the_isa_tool_decides_a_well_formed_program_and_no_verdict_is_unsettled(load, tmp_path, out, rc, want):
    mod = load(ISA)
    row = mod.load_rows(VERTICALS / ISA.name / "pool" / "train")[0]
    check = adapter(ISA, {"tool": tool(tmp_path, out, rc)})
    res = check.run(job_for(mod, tmp_path, row, ISA.good, ISA.leg))
    assert (res.state, res.reason) == want
    assert json.loads((tmp_path / "tool-input.json").read_text()) == {"goal": row["goal"],
                                                                      "program": "MOV r1, #3\nADD r2, r1, r1\n"}
    (tmp_path / "tool-input.json").unlink()
    malformed = check.run(job_for(mod, tmp_path, row, ISA.bad[1][0], ISA.leg))
    assert malformed.reason == "malformed" and not (tmp_path / "tool-input.json").exists()  # the tool sees only these


@BY_NAME
def test_the_reward_keeps_the_services_invariants_and_leaves_the_unsettled_unscored(load, v):
    from aorlta.reward.service import REWARD_MAX, REWARD_MIN
    from aorlta.schemas import Candidate, LegResult, TaskRef

    load(v)
    fn = plugins.REWARD_FNS[f"{v.name}.v1"](None)
    assert fn.legs == (v.leg,) and fn.name == f"{v.name}.v1" and len(fn.code_sha256) == 64
    # every reason the check gives has its own value, below a clean answer's
    named = v.reasons | ({"tool_rejected"} if v is ISA else set())
    assert set(fn.BASE) == named and len(set(fn.BASE.values())) == len(fn.BASE)
    cases = [("clean", None, 1.0)] + [("fail", r, b) for r, b in fn.BASE.items()] + [
        ("fail", "other", None), ("fail", None, None), ("finding", None, None), ("unsettled", "x", None),
        ("not_run", None, None), ("skipped_by_plan", None, None)]
    for state, reason, base in cases:
        for tokens in (0, 256, 512):
            leg = LegResult(v.leg, state, reason, [], 1, 1, None, None, False, 0.0, 0.0, 0.0)
            c = Candidate("c", "r", 1, f"{v.name}:x", 0, "v", "", tokens, 512, TaskRef("x", None, None, None, "t"),
                          v.name, None)
            reward, parts = fn.compute({v.leg: leg}, c)
            d = fn.decide({v.leg: leg}, c)
            if base is None:
                assert reward is None and type(d).__name__ == "Unscored" and not fn.claims_clean(reward, parts)
                continue
            assert set(parts) == {"base", "broke_healthy", "length"} and parts["base"] == base
            assert REWARD_MIN <= reward <= REWARD_MAX and base <= 1.0
            assert parts["base"] + parts["broke_healthy"] + parts["length"] == reward
            assert (d.reward, d.parts) == (reward, parts)
            assert fn.claims_clean(reward, parts) == (state == "clean")
    assert type(fn.decide({}, c)).__name__ == "NeedMore" and fn.compute({}, c)[0] is None


@BY_NAME
def test_each_package_stands_alone_and_its_readme_is_short(v):
    d = VERTICALS / v.name
    files = {p.name for p in d.iterdir()} - {"__pycache__"}
    assert files == {v.module, f"{v.name}.yaml", "score.yaml", "train.yaml", "pool", "README.md"}
    text = (d / v.module).read_text()
    imports = re.findall(r"^\s*(?:from|import)\s+([\w.]+)", text, re.MULTILINE)
    assert all(m.split(".")[0] in {"__future__", "hashlib", "json", "re", "subprocess", "unicodedata", "pathlib",
                                   "aorlta"} for m in imports), imports
    assert "TODO" in text and "PLACEHOLDER" in text
    readme = (d / "README.md").read_text()
    assert len(readme.splitlines()) < 30
    for name in (v.module, f"{v.name}.yaml", "score.yaml", "train.yaml", "tests/examples/test_verticals.py"):
        assert name in readme


def replies_from(monkeypatch, replies):
    """The fake engines answer with one of `replies`, drawn from the request's own random stream, after a line of its
    own: two seeds then give different text, as the engines' start-up self-test needs."""
    from aorlta.rollout.fake_engine import FakeEngine

    monkeypatch.setattr(FakeEngine, "_answer_text",
                        lambda self, rng: f"({rng.hex('tag')})\n" + replies[int(rng.uniform("reply") * len(replies))])


def wired(tmp_path, base, v, kind, monkeypatch):
    from aorlta.orchestrator import wiring
    from aorlta.rollout.engine_proc import InProcessEngine

    replies_from(monkeypatch, [v.good, *(r for r, _ in v.bad)])
    data = parse_yaml((VERTICALS / v.name / f"{kind}.yaml").read_bytes())
    data["run"].update(root=str(tmp_path / "runs"), run_id=f"{v.name}-cpu")
    data["model"]["base_dir"] = str(base)
    data["data"]["prompt_pools"][0]["path"] = str(VERTICALS / v.name / "pool" / ("heldout" if kind == "score"
                                                                                  else "train"))
    data["rollout"].update(engines=[0, 1], K=4, P=4, request_timeout_s=300)
    if kind == "train":
        data["orchestrator"]["iterations"] = 1
        data["trainer"] = {"ranks": 1, "lr": 1e-2, "micro_batch_tokens": 2048}
    data["sync"] = {"dir": str(tmp_path / "shm")}
    cfg = config_from_dict(data)
    engines = [E.TwinEngine(base, E.FAKE, i) for i in range(2)]
    parts = wiring.Parts(tokenizer=E.ByteTokenizer(), handles=[InProcessEngine(i, e) for i, e in enumerate(engines)],
                         load_model=E.load_trainer_model, trainer_device="cpu", require_fast_path=False,
                         probes=E.PROBES)
    return cfg, wiring.build(cfg, parts=parts)


def scored_bases(eps, v):
    """Each scored episode's base reward (give or take the length term), and its leg state."""
    return {(round(e["reward"] + 0.05, 1), e["legs"][v.leg]) for e in eps if e["status"] == "scored"}


@pytest.mark.heavy
@BY_NAME
def test_a_vertical_trains_one_iteration_through_the_real_wiring(tmp_path, load, tiny_base, v, monkeypatch):
    load(v)
    cfg, orch = wired(tmp_path, tiny_base, v, "train", monkeypatch)
    result = orch.run()
    assert result["status"] == "completed", result
    s = E.summary(cfg, 1)
    assert s["trainer"]["step_taken"] and s["samples"]["trained"] > 0 and s["sync"]["landing"] == "LANDED"
    eps = [e for e in E.episodes(cfg, 1) if e["end_reason"] != "engine_error"]
    assert eps and {e["prompt_id"].partition(":")[0] for e in eps} == {v.name}
    bases = scored_bases(eps, v)
    fn = plugins.REWARD_FNS[f"{v.name}.v1"](None)
    assert {b for b, _ in bases} <= {1.0, *(round(b, 1) for b in fn.BASE.values())}
    assert len({b for b, _ in bases}) >= 2  # the answers get different rewards, so there is something to learn


@pytest.mark.heavy
@BY_NAME
def test_a_vertical_scores_its_held_out_prompts_through_the_real_wiring(tmp_path, load, tiny_base, v, monkeypatch):
    load(v)
    cfg, scorer = wired(tmp_path, tiny_base, v, "score", monkeypatch)
    assert type(scorer).__name__ == "Scorer"
    result = scorer.run()
    assert result["status"] == "completed", result
    rows = [json.loads(x) for x in (cfg.run_dir / "score" / "report.jsonl").read_text().splitlines()]
    assert rows[-1]["samples"] == 16
    held = {f"{v.name}:{r['id']}" for r in load(v).load_rows(VERTICALS / v.name / "pool" / "heldout")}
    assert {r["prompt_id"] for r in rows if "prompt_id" in r} == held
