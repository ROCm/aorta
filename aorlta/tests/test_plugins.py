"""Plugins (`aorlta.plugins`): a task's prompts, parse, feedback, compile check, profile, check adapter and reward
function, registered by a module the run config lists, and found by every place that looks a name up."""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import pytest

from aorlta import plugins
from aorlta.config import ConfigError, config_from_dict, write_run_plan
from aorlta.ids import sha256_file

ROOT = Path(__file__).resolve().parents[1]

PLUGIN = '''
from pathlib import Path

from aorlta import plugins
from aorlta.ids import sha256_ids, sha256_text
from aorlta.reward.adapters.base import make_result
from aorlta.reward.rewardfn.base import Decided, NeedMore
from aorlta.rollout.compile_check import FakeCompiler, ParseOutcome
from aorlta.rollout.feedback import Feedback, build_feedback
from aorlta.rollout.templates import render_turn0
from aorlta.schemas import PromptRecord, TaskRef

HERE = Path(__file__).resolve().parent


def records(pool_dir, tokenizer, thinking):
    out = []
    for line in (Path(pool_dir) / "prompts.jsonl").read_text().splitlines():
        row = __import__("json").loads(line)
        messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": row["ask"]}]
        out.append(PromptRecord(prompt_id=f"toy:{row['id']}", turn0_ids=None, messages=messages,
                                prompt_ids_sha256=sha256_ids(render_turn0(tokenizer, messages, thinking)),
                                task=TaskRef(contract_id=f"toy:{row['id']}", parent_id=None, parent_healthy=None,
                                             reference_kind=None, task_sha256=sha256_text(row["ask"])),
                                source="standin"))
    return out


class Parse:
    name = "toy.parse"
    reply_stops = (248046,)

    def check_prompts(self, prompts, thinking, max_repair_turns=0):
        return None

    def check(self, text, thinking, prompt):
        return ParseOutcome("edits", code=text) if "```" in text else ParseOutcome("rejected", rejection=("no block", "x"))


class Fb:
    name = "toy.feedback"

    def build(self, errors, max_chars):
        return build_feedback(errors, max_chars)

    def build_apply(self, message, reason):
        return Feedback(message, 0, 0)


class Adapter:
    version = "1"
    legs = ("ref.native",)
    resource = "local_cpu"

    def __init__(self, ctx):
        self.name, self.ctx = ctx.name, ctx

    def run(self, job):  # half the sources pass, by hash: rewards differ within a group
        t = self.ctx.clock()
        if job.source_sha256[-1] in "02468ace":
            return make_result(job, "clean", t_start=t, t_end=t)
        return make_result(job, "fail", "wrong_output", t_start=t, t_end=t)

    def controls(self):
        return []

    def health(self):
        return {"responsive": True}


class Reward:
    name = "toy.v1"
    version = "1"
    code_sha256 = "0" * 64
    legs = ("ref.native",)

    def __init__(self, profile):
        self.profile = profile

    def decide(self, partial, cand):
        if "ref.native" not in partial:
            return NeedMore(("ref.native",))
        return Decided(*self.compute(partial, cand))

    def compute(self, legs, cand):
        base = 1.0 if legs["ref.native"].state == "clean" else -1.0
        return base + 0.0 + 0.0, {"base": base, "broke_healthy": 0.0, "length": 0.0}

    def claims_clean(self, reward, parts):
        return reward is not None and parts["base"] > 0


plugins.register_task(plugins.Task("toy", records))
plugins.register_parse_adapter("toy.parse", Parse)
plugins.register_feedback_template("toy.feedback", Fb)
plugins.register_compile_adapter("toy.compile", lambda workdir, evidence=None, structure=None: FakeCompiler(
    evidence=evidence))
plugins.register_profile("toy-profile", HERE / "toy-profile.yaml")
plugins.register_check_adapter("toy.check", Adapter)
plugins.register_reward_fn("toy.v1", Reward)
'''

PROFILE = """
name: toy-profile
reward_fn: toy.v1
materialize: true
pools:
  run: {resource: local_cpu, adapter: toy.check, timeout: compile, workers: 2}
stages:
  - {name: S0, pool: run, legs: [ref.native]}
adapters:
  toy.check: {tool: "${AORLTA_ROCM_ROOT}/bin/hipcc", gpus: [4, 5]}
"""


@pytest.fixture
def registry():
    """The registries as they were, restored after the test: a plugin's registrations are global to the process."""
    tables = [plugins.TASKS, plugins.PARSE_ADAPTERS, plugins.FEEDBACK_TEMPLATES, plugins.COMPILE_ADAPTERS,
              plugins.PROFILES, plugins.REWARD_FNS, plugins.CHECK_ADAPTERS, plugins.LOADED]
    saved = [dict(t) for t in tables]
    yield
    for entry in set(plugins.LOADED) - set(saved[-1]):  # a plugin loaded here: imported afresh by the next test
        sys.modules.pop(plugins.LOADED[entry]["module"], None)
    for t, s in zip(tables, saved):
        t.clear()
        t.update(s)


@pytest.fixture
def plugin(tmp_path, registry) -> Path:
    d = tmp_path / "toyplugin"
    d.mkdir()
    (d / "toy_plugin.py").write_text(PLUGIN)
    (d / "toy-profile.yaml").write_text(PROFILE)
    pool = tmp_path / "pool"
    pool.mkdir()
    (pool / "prompts.jsonl").write_text("".join(json.dumps({"id": f"p{i}", "ask": f"do {i}"}) + "\n" for i in range(6)))
    return d / "toy_plugin.py"


def toy_config(tmp_path, plugin_entry: str, **over) -> dict:
    cfg = {
        "plugins": [plugin_entry],
        "run": {"run_id": "toy-1", "source": "standin", "root": str(tmp_path / "runs")},
        "model": {"base_dir": str(tmp_path / "base"), "source": "standin"},
        "data": {"prompt_pools": [{"path": str(tmp_path / "pool"), "source": "standin", "task": "toy"}]},
        "rollout": {"K": 2, "P": 4, "compile_adapter": "toy.compile", "parse_adapter": "toy.parse",
                    "feedback_template": "toy.feedback"},
        "reward": {"profile": "toy-profile", "reward_fn": "toy.v1"},
        "orchestrator": {"iterations": 1, "run_seed": 1},
        "sync": {"dir": str(tmp_path / "shm")},
    }
    for k, v in over.items():
        cfg[k] = v
    return cfg


def test_a_config_names_what_its_plugin_registered(tmp_path, plugin):
    cfg = config_from_dict(toy_config(tmp_path, str(plugin)))
    assert cfg.plugins == (str(plugin),) and cfg.data.prompt_pools[0].task == "toy"
    assert (cfg.reward.profile, cfg.reward.reward_fn) == ("toy-profile", "toy.v1")
    assert (cfg.rollout.parse_adapter, cfg.rollout.feedback_template, cfg.rollout.compile_adapter) == (
        "toy.parse", "toy.feedback", "toy.compile")


def test_without_the_plugin_its_names_are_refused(tmp_path, registry):
    data = toy_config(tmp_path, "unused")
    del data["plugins"]
    with pytest.raises(ConfigError, match="'toy' is not 'standin' or a task a plugin registered"):
        config_from_dict(data)
    data["data"]["prompt_pools"][0]["task"] = "standin"
    with pytest.raises(ConfigError, match="'toy-profile' is not one of standin-static, standin-gpu, fake or a profile a "
                                          "plugin registered"):
        config_from_dict(data)
    data["reward"] = {"profile": "fake", "reward_fn": "standin.v11"}
    with pytest.raises(ConfigError, match="'toy.compile' must be 'none', start with 'standin' or 'fake', or be "
                                          "registered by a plugin"):
        config_from_dict(data)


@pytest.mark.parametrize("key, value, message", [
    ("task", "nobody", "task: 'nobody' is not 'standin' or a task a plugin registered"),
])
def test_an_unregistered_task_is_refused(tmp_path, plugin, key, value, message):
    data = toy_config(tmp_path, str(plugin))
    data["data"]["prompt_pools"][0][key] = value
    with pytest.raises(ConfigError, match=message):
        config_from_dict(data)


def test_an_unregistered_profile_and_reward_fn_are_refused(tmp_path, plugin):
    data = toy_config(tmp_path, str(plugin))
    data["reward"]["profile"] = "nobody"
    with pytest.raises(ConfigError, match="profile: 'nobody' is not one of"):
        config_from_dict(data)
    data = toy_config(tmp_path, str(plugin))
    data["reward"]["reward_fn"] = "standin.v11"
    with pytest.raises(ConfigError, match="profile 'toy-profile' is scored by 'toy.v1'"):
        config_from_dict(data)


def test_a_relative_plugin_path_is_read_from_the_package_root(tmp_path, registry, monkeypatch):
    d = tmp_path / "pkg"
    (d / "examples").mkdir(parents=True)
    (d / "examples" / "p.py").write_text("from aorlta import plugins\nplugins.register_parse_adapter('rel.parse', "
                                         "lambda: None)\n")
    monkeypatch.setattr(plugins, "PACKAGE_ROOT", d)
    plugins.load_plugins(["examples/p.py"])
    assert "rel.parse" in plugins.PARSE_ADAPTERS
    assert plugins.identity()["examples/p.py"]["sha256"] == sha256_file(d / "examples" / "p.py")


def test_a_module_name_is_imported(tmp_path, registry, monkeypatch):
    (tmp_path / "toy_mod_plugin.py").write_text("from aorlta import plugins\nplugins.register_feedback_template("
                                                "'mod.feedback', lambda: None)\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    plugins.load_plugins(["toy_mod_plugin"])
    assert "mod.feedback" in plugins.FEEDBACK_TEMPLATES and plugins.identity()["toy_mod_plugin"]["module"] == (
        "toy_mod_plugin")


@pytest.mark.parametrize("source, message", [
    ("raise RuntimeError('boom')", "importing .* failed: RuntimeError: boom"),
    ("from aorlta import plugins\nplugins.register_parse_adapter('standin.parse', lambda: None)", "is built in"),
    ("from aorlta import plugins\nplugins.register_profile('p', '/nonexistent/p.yaml')", "no file at"),
])
def test_a_plugin_that_fails_stops_the_config(tmp_path, registry, source, message):
    p = tmp_path / "bad.py"
    p.write_text(source)
    data = toy_config(tmp_path, str(p))
    with pytest.raises(ConfigError, match=message):
        config_from_dict(data)


def test_a_missing_plugin_file_stops_the_config(tmp_path, registry):
    with pytest.raises(ConfigError, match="no file at"):
        config_from_dict(toy_config(tmp_path, str(tmp_path / "missing.py")))


def test_a_name_is_registered_once(registry):
    a, b = (lambda: None), (lambda: None)
    plugins.register_parse_adapter("once.parse", a)
    plugins.register_parse_adapter("once.parse", a)  # the same object: a no-op
    with pytest.raises(plugins.PluginError, match="already registered"):
        plugins.register_parse_adapter("once.parse", b)
    for bad in ("", " x", "none"):
        with pytest.raises(plugins.PluginError, match="non-empty name"):
            plugins.register_parse_adapter(bad, a)


def test_a_plugin_is_imported_once(tmp_path, registry):
    p = tmp_path / "count.py"
    p.write_text("import builtins\nbuiltins.__dict__.setdefault('_aorlta_count', []).append(1)\n")
    import builtins

    builtins.__dict__.pop("_aorlta_count", None)
    plugins.load_plugins([str(p)])
    plugins.load_plugins([str(p)])
    assert builtins.__dict__.pop("_aorlta_count") == [1]


@pytest.mark.parametrize("form", ["file", "module"])
def test_a_plugin_that_raises_leaves_nothing_registered(tmp_path, registry, monkeypatch, form):
    """What a plugin registered before raising is undone, so nothing sees its components and, fixed, it loads in
    the same process (its factories are new objects, which a leftover registration would refuse)."""
    body = ("from aorlta import plugins\nplugins.register_parse_adapter('half.parse', lambda: None)\n"
            "plugins.register_reward_fn('half.v1', lambda profile: None)\n")
    p = tmp_path / "half_plugin.py"
    p.write_text(body + "raise RuntimeError('boom')\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    entry = str(p) if form == "file" else "half_plugin"
    with pytest.raises(plugins.PluginError, match="RuntimeError: boom"):
        plugins.load_plugins([entry])
    assert not plugins.registered("half.parse") and not plugins.registered("half.v1") and entry not in plugins.LOADED
    p.write_text(body)
    plugins.load_plugins([entry])
    assert plugins.registered("half.parse") and plugins.registered("half.v1") and entry in plugins.LOADED


def test_a_registered_profile_must_carry_its_registered_name(tmp_path, plugin):
    """The candidates carry the registered name and the reward service serves the profile file's: a file naming
    anything else (here a built-in profile's name) would have every candidate refused as profile_mismatch."""
    from aorlta.reward.profiles import load_profile

    other = tmp_path / "other.yaml"
    other.write_text(PROFILE.replace("name: toy-profile", "name: fake"))
    plugins.register_profile("toy-other", other)
    with pytest.raises(ConfigError, match="'toy-other' names itself 'fake'"):
        load_profile("toy-other")
    data = toy_config(tmp_path, str(plugin))
    data["reward"]["profile"] = "toy-other"
    with pytest.raises(ConfigError, match="'toy-other' names itself 'fake'"):
        config_from_dict(data)
    assert load_profile("toy-profile").name == "toy-profile" and load_profile(other).name == "fake"


def test_the_run_plan_records_each_plugin_file(tmp_path, plugin):
    cfg = config_from_dict(toy_config(tmp_path, str(plugin)))
    plan = json.loads(write_run_plan(cfg, commit="a" * 40).read_text())
    assert plan["plugins"][str(plugin)]["sha256"] == sha256_file(plugin)
    assert plan["config"]["plugins"] == [str(plugin)]


def test_every_lookup_finds_the_registered_component(tmp_path, plugin, monkeypatch):
    from aorlta.reward.factories import standin_factories
    from aorlta.reward.profiles import load_profile
    from aorlta.reward.rewardfn.for_profile import standin_rewardfn
    from aorlta.rollout.chain import check_adapter_names, make_feedback_template, make_parse_adapter
    from aorlta.rollout.compile_check import FakeCompiler, make_compiler

    monkeypatch.setenv("AORLTA_ROCM_ROOT", "/rocm-here")
    config_from_dict(toy_config(tmp_path, str(plugin)))
    check_adapter_names("toy.parse", "toy.feedback")
    assert make_parse_adapter("toy.parse").name == "toy.parse"
    assert make_feedback_template("toy.feedback").build_apply("m", "r").text == "m"
    assert isinstance(make_compiler("toy.compile", tmp_path / "c"), FakeCompiler)
    profile = load_profile("toy-profile")
    assert profile.name == "toy-profile" and profile.adapters["toy.check"]["tool"] == "/rocm-here/bin/hipcc"
    assert standin_rewardfn(profile).name == "toy.v1"

    class Clock:
        @staticmethod
        def wall():
            return 5.0

    factories = standin_factories(profile, Clock(), timeouts={"compile": 100.0}, pools=("/pool/a",))
    adapter = factories["toy.check"]("w3")
    ctx = adapter.ctx
    assert (ctx.name, ctx.worker_id, ctx.pools, ctx.timeout_s) == ("toy.check", "w3", ("/pool/a",), 90.0)
    assert ctx.settings["gpus"] == [4, 5] and ctx.clock() == 5.0


def test_a_component_registered_under_another_name_is_refused(registry):
    from aorlta.rollout.engine_pool import RolloutStop
    from aorlta.rollout.chain import make_parse_adapter

    class Misnamed:
        name = "other"

    plugins.register_parse_adapter("wrong.parse", Misnamed)
    with pytest.raises(RolloutStop, match="is named 'other'"):
        make_parse_adapter("wrong.parse")


def test_the_task_pool_and_its_probes(tmp_path, plugin):
    from aorlta.orchestrator import wiring

    cfg = config_from_dict(toy_config(tmp_path, str(plugin)))
    tok = CharChat()
    pool = wiring.TaskPromptPool(plugins.task("toy"), tok, "off", tmp_path / "pool")
    recs = pool.records()
    assert [r.prompt_id for r in recs] == [f"toy:p{i}" for i in range(6)]
    probes = wiring.task_probes(cfg, tok, plugins.task("toy"), tmp_path / "pool", recs)
    assert len(probes.ids) == cfg.sync.probes == 4 and all(len(p) == cfg.sync.probe_tokens for p in probes.ids)


def test_the_task_pool_refuses_duplicate_ids_and_too_few_probes(tmp_path, plugin):
    from aorlta.orchestrator import wiring

    cfg = config_from_dict(toy_config(tmp_path, str(plugin)))
    (tmp_path / "pool" / "prompts.jsonl").write_text(json.dumps({"id": "a", "ask": "x"}) + "\n"
                                                     + json.dumps({"id": "a", "ask": "y"}) + "\n")
    with pytest.raises(wiring.WiringError, match="gave a prompt id twice"):
        wiring.TaskPromptPool(plugins.task("toy"), CharChat(), "off", tmp_path / "pool").records()
    (tmp_path / "pool" / "prompts.jsonl").write_text(json.dumps({"id": "a", "ask": "x"}) + "\n")
    recs = wiring.TaskPromptPool(plugins.task("toy"), CharChat(), "off", tmp_path / "pool").records()
    with pytest.raises(wiring.WiringError, match="needs 4 probe chats"):
        wiring.task_probes(cfg, CharChat(), plugins.task("toy"), tmp_path / "pool", recs)


def test_check_supported_takes_a_registered_reward_fn(tmp_path, plugin):
    from aorlta.orchestrator import wiring

    cfg = config_from_dict(toy_config(tmp_path, str(plugin)))
    wiring.check_supported(cfg)


class CharChat:
    """The rollout's Tokenizer and the probes' Hugging Face calls, one id per character, with Qwen3.5's generation
    prompt so `render_turn0` accepts it."""

    SPECIAL = {"<|im_start|>": 248045, "<|im_end|>": 248046, "<think>": 248068, "</think>": 248069}

    def encode(self, text):
        out, i = [], 0
        while i < len(text):
            for s, t in self.SPECIAL.items():
                if text.startswith(s, i):
                    out.append(t)
                    i += len(s)
                    break
            else:
                out.append(ord(text[i]) + 10000)
                i += 1
        return out

    def render(self, messages, thinking, add_generation_prompt=True):
        return self.apply_chat_template(messages, enable_thinking=thinking == "on")

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": self.encode(text)}

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        body = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs)
        return body + "<|im_start|>assistant\n" + ("<think>\n" if enable_thinking else "<think>\n\n</think>\n\n")


def test_the_module_docstring_names_every_registry():
    doc = textwrap.dedent(plugins.__doc__)
    for fn in ("register_task", "register_parse_adapter", "register_feedback_template", "register_compile_adapter",
               "register_profile", "register_reward_fn", "register_check_adapter"):
        assert fn in doc and callable(getattr(plugins, fn))
