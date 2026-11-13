"""Three task shapes registered from a package outside aorlta/, through an installed entry point: (a) code rewarded
by a compile gate and a reference check, (b) a question and answer with no compile step and no sanitizer, (c) a
reward that is an external tool's verdict. Each loads as a run config, passes the wiring's checks, and scores a
candidate through its own check adapter and reward function."""

from __future__ import annotations

import json
import stat
import sys
from pathlib import Path

import pytest

from aorlta import plugins
from aorlta.config import config_from_dict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_plugins import CharChat, registry  # noqa: E402,F401

PLUGIN = '''
import json
import subprocess
from pathlib import Path

from aorlta import plugins
from aorlta.ids import sha256_ids, sha256_text
from aorlta.reward.adapters.base import make_result, unsettled
from aorlta.reward.rewardfn.base import Decided, NeedMore, Unscored
from aorlta.rollout.compile_check import ParseOutcome
from aorlta.rollout.templates import render_turn0
from aorlta.schemas import PromptRecord, TaskRef

HERE = Path(__file__).resolve().parent


def records(pool_dir, tokenizer, thinking):
    out = []
    for i in range(6):
        msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": f"task {i}"}]
        out.append(PromptRecord(f"team:{i}", None, msgs, sha256_ids(render_turn0(tokenizer, msgs, thinking)),
                                TaskRef(f"team:{i}", None, None, None, sha256_text(f"task {i}")), "standin"))
    return out


class Parse:
    name = "team.parse"
    reply_stops = (248046,)

    def check_prompts(self, prompts, thinking, max_repair_turns=0):
        return None

    def check(self, text, thinking, prompt):
        return ParseOutcome("edits", code=text.strip())


class Check:
    version, resource = "1", "local_cpu"

    def __init__(self, ctx, leg):
        self.name, self.ctx, self.legs = ctx.name, ctx, (leg,)

    def answer(self, job):
        return Path(job.inputs["response"]).read_text().strip()

    def controls(self):
        return []

    def health(self):
        return {"responsive": True}


class Reference(Check):  # shape (a), after the stand-in compile gate: the source must hold the reference's marker
    def __init__(self, ctx):
        super().__init__(ctx, "ref.native")

    def run(self, job):
        t = self.ctx.clock()
        ok = "MARK" in Path(job.inputs["source"]).read_text()
        return make_result(job, "clean" if ok else "fail", None if ok else "wrong_output", t_start=t, t_end=t)


class Exact(Check):  # shape (b): the answer against a reference, no compile
    def __init__(self, ctx):
        super().__init__(ctx, "ref.answer")

    def run(self, job):
        t = self.ctx.clock()
        ok = self.answer(job) == "42"
        return make_result(job, "clean" if ok else "fail", None if ok else "wrong_answer", t_start=t, t_end=t)


class Tool(Check):  # shape (c): an external command's verdict; no verdict is unsettled
    def __init__(self, ctx):
        super().__init__(ctx, "ext.verdict")

    def run(self, job):
        t = self.ctx.clock()
        p = subprocess.run(list(self.ctx.settings["tool"]), input=self.answer(job), capture_output=True, text=True)
        v = p.stdout.strip()
        if v in ("PASS", "FAIL"):
            return make_result(job, "clean" if v == "PASS" else "fail", None if v == "PASS" else "rejected",
                               t_start=t, t_end=t)
        return unsettled(job, "no_verdict", t_start=t, t_end=t)


class Reward:
    version, code_sha256 = "1", "0" * 64

    def __init__(self, name, legs):
        self.name, self.legs = name, legs

    def compute(self, legs, cand):
        if any(legs[n].state in ("unsettled", "not_run", "skipped_by_plan", "finding") for n in self.legs):
            return None, {"reason": "unsettled"}
        base = 1.0 if all(legs[n].state == "clean" for n in self.legs) else -0.5
        return base + 0.0 + 0.0, {"base": base, "broke_healthy": 0.0, "length": 0.0}

    def decide(self, partial, cand):
        pending = [n for n in self.legs if n not in partial]
        if pending:
            return NeedMore((pending[0],))
        r, p = self.compute(partial, cand)
        return Unscored(p["reason"]) if r is None else Decided(r, p)

    def claims_clean(self, reward, parts):
        return reward is not None and parts["base"] > 0


plugins.register_task(plugins.Task("team", records))
plugins.register_parse_adapter("team.parse", Parse)
for shape, adapter, leg in (("code", Reference, "ref.native"), ("qa", Exact, "ref.answer"),
                            ("tool", Tool, "ext.verdict")):
    plugins.register_check_adapter(f"team.{shape}", adapter)
    legs = ("gate.parse", "gate.compile", leg) if shape == "code" else (leg,)
    plugins.register_reward_fn(f"team.{shape}.v1", lambda profile, legs=legs, shape=shape: Reward(f"team.{shape}.v1", legs))
    plugins.register_profile(f"team-{shape}", HERE / f"team-{shape}.yaml")
'''

PROFILES = {
    "code": """name: team-code
reward_fn: team.code.v1
materialize: true
pools:
  gates: {resource: local_cpu, adapter: standin.parse, timeout: gate}
  compile: {resource: local_cpu, adapter: standin.hipcc, timeout: compile}
  ref: {resource: local_cpu, adapter: team.code, timeout: sim_run}
stages:
  - {name: S0, pool: gates, legs: [gate.parse]}
  - {name: S1, pool: compile, legs: [gate.compile]}
  - {name: S2, pool: ref, legs: [ref.native]}
requires: {gate.compile: [gate.parse], ref.native: [gate.compile]}
leg_settings: {gate.compile: {offload_arch: gfx950, flags: ["-c"]}}
""",
    "qa": """name: team-qa
reward_fn: team.qa.v1
materialize: true
pools:
  ref: {resource: local_cpu, adapter: team.qa, timeout: gate}
stages:
  - {name: S0, pool: ref, legs: [ref.answer]}
""",
    "tool": """name: team-tool
reward_fn: team.tool.v1
materialize: true
pools:
  ext: {resource: local_cpu, adapter: team.tool, timeout: sim_run}
stages:
  - {name: S0, pool: ext, legs: [ext.verdict]}
adapters:
  team.tool: {tool: [TOOL]}
""",
}


@pytest.fixture
def installed(tmp_path, registry, monkeypatch):  # noqa: F811
    """A package of a team's own, `team_tasks`, installed (a dist-info with an `aorlta.plugins` entry point) in a
    directory outside aorlta/, and a stand-in for the team's verdict tool."""
    site = tmp_path / "site-packages"
    pkg = site / "team_tasks"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "plugin.py").write_text(PLUGIN)
    tool = tmp_path / "verdict-tool"
    tool.write_text("#!/bin/bash\nread a\n[ \"$a\" = ok ] && echo PASS || { [ \"$a\" = bad ] && echo FAIL || echo ?; }\n")
    tool.chmod(tool.stat().st_mode | stat.S_IXUSR)
    for shape, text in PROFILES.items():
        (pkg / f"team-{shape}.yaml").write_text(text.replace("TOOL", str(tool)))
    dist = site / "team_tasks-0.1.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: team-tasks\nVersion: 0.1\n")
    (dist / "entry_points.txt").write_text("[aorlta.plugins]\nteam = team_tasks.plugin\n")
    monkeypatch.syspath_prepend(str(site))
    yield tmp_path
    sys.modules.pop("team_tasks.plugin", None)
    sys.modules.pop("team_tasks", None)


def config(tmp_path, shape):
    compile_adapter = {"code": "fake.compile", "qa": "none", "tool": "none"}[shape]
    return config_from_dict({
        "plugins": ["team"],
        "run": {"run_id": f"team-{shape}", "source": "standin", "root": str(tmp_path / "runs")},
        "model": {"base_dir": str(tmp_path / "base"), "source": "standin"},
        "data": {"prompt_pools": [{"path": str(tmp_path / "pool"), "source": "standin", "task": "team"}]},
        "rollout": {"K": 2, "P": 4, "compile_adapter": compile_adapter, "parse_adapter": "team.parse"},
        "reward": {"profile": f"team-{shape}", "reward_fn": f"team.{shape}.v1"},
        "orchestrator": {"iterations": 1, "run_seed": 1},
    })


def leg_job(tmp_path, leg, response, source=None):
    from aorlta.schemas import LegJob

    (tmp_path / "response.txt").write_text(response)
    inputs = {"response": str(tmp_path / "response.txt")}
    if source is not None:
        (tmp_path / "source.hip").write_text(source)
        inputs["source"] = str(tmp_path / "source.hip")
    return LegJob(job_id="j", leg=leg, candidate_id="c", control_id=None, source_sha256="s" * 64, task_sha256="t" * 64,
                  inputs=inputs, output_dir=str(tmp_path))


@pytest.mark.parametrize("shape, leg, good, bad, extra", [
    ("code", "ref.native", ("x", "// MARK"), ("x", "// none"), ("gate.parse", "gate.compile")),
    ("qa", "ref.answer", ("42", None), ("41", None), ()),
    ("tool", "ext.verdict", ("ok", None), ("bad", None), ()),
])
def test_a_shape_registered_from_its_own_package_scores_through_its_own_check(installed, shape, leg, good, bad, extra):
    from aorlta.orchestrator import wiring
    from aorlta.reward.factories import standin_factories
    from aorlta.reward.profiles import load_profile
    from aorlta.reward.rewardfn.for_profile import standin_rewardfn
    from aorlta.schemas import Candidate, LegResult, TaskRef

    cfg = config(installed, shape)
    assert plugins.identity()["team"]["module"] == "team_tasks.plugin"
    wiring.check_supported(cfg)
    profile = load_profile(cfg.reward.profile)
    assert profile.legs == (*extra, leg)
    if shape != "code":
        assert not any(p.adapter.startswith("standin.") for p in profile.pools.values())  # no compile, no sanitizer

    class Clock:
        wall = staticmethod(lambda: 0.0)

    adapter = standin_factories(profile, Clock(), pools=(str(installed / "pool"),))[f"team.{shape}"]("w/1")
    fn = standin_rewardfn(profile)
    cand = Candidate("c", "r", 1, "team:0", 0, "v", "", 1, 10, TaskRef("team:0", None, None, None, "t"), profile.name,
                     None)
    clean = LegResult("x", "clean", None, [], 1, 1, None, None, False, 0.0, 0.0, 0.0)
    for (answer, source), want in ((good, 1.0), (bad, -0.5)):
        res = adapter.run(leg_job(installed, leg, answer, source))
        legs = {n: clean for n in extra} | {leg: res}
        assert fn.compute(legs, cand)[0] == want, (shape, answer, res)
    if shape == "tool":
        assert adapter.run(leg_job(installed, leg, "huh")).state == "unsettled"  # no verdict is never clean


def test_two_packages_declaring_one_entry_point_name_are_refused(installed, monkeypatch, tmp_path):
    other = tmp_path / "other-site"
    dist = other / "other-0.1.dist-info"
    dist.mkdir(parents=True)
    (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: other\nVersion: 0.1\n")
    (dist / "entry_points.txt").write_text("[aorlta.plugins]\nteam = other.plugin\n")
    monkeypatch.syspath_prepend(str(other))
    with pytest.raises(plugins.PluginError, match="2 installed packages declare the entry point 'team'"):
        plugins.load_plugins(["team"])


def test_an_unknown_entry_point_name_falls_back_to_a_module_import(installed):
    with pytest.raises(plugins.PluginError, match="importing no_such_team_module failed"):
        plugins.load_plugins(["no_such_team_module"])


def test_compile_adapter_none_passes_every_answer_and_only_it_may_be_none(tmp_path):
    from aorlta.config import ConfigError
    from aorlta.rollout.compile_check import make_compiler

    res = make_compiler("none", tmp_path).check("not code at all", "c0-0-t0")
    assert res.ok and res.rc == 0 and res.errors == []
    base = {"run": {"run_id": "r", "source": "standin", "root": str(tmp_path)},
            "model": {"base_dir": str(tmp_path / "b"), "source": "standin"},
            "data": {"prompt_pools": [{"path": str(tmp_path / "p"), "source": "standin"}]},
            "reward": {"profile": "fake", "reward_fn": "standin.v11"},
            "orchestrator": {"iterations": 1, "run_seed": 1}}
    assert config_from_dict({**base, "rollout": {"compile_adapter": "none"}}).rollout.compile_adapter == "none"
    with pytest.raises(ConfigError, match="parse_adapter: cannot be 'none'"):
        config_from_dict({**base, "rollout": {"parse_adapter": "none"}})
