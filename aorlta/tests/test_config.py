"""The config loader (Section 5 item 4; Section 6.10 item 8): frozen dataclasses, unknown keys and
mixed sources refused, run_plan.json written first."""

import copy
import dataclasses
import json
import pickle
import subprocess
import threading
from pathlib import Path

import pytest

from aorlta import config as C
from aorlta.config import ConfigError, config_from_dict, load_config, parse_yaml, write_run_plan
from aorlta.ids import sha256_file, sha256_json

STANDIN_YAML = """\
run:
  run_id: dev-standin-001
  source: standin
  root: {root}
model:
  base_dir: /data/hf/hub/models--Qwen--Qwen3.5-4B
  source: standin
data:
  prompt_pools:
    - path: /data/aorlta/standin-pool/listings
      source: standin
rollout:
  engines: [0, 1, 2, 3]
  K: 8
  P: 32
  thinking: off
reward:
  profile: standin-static
  reward_fn: standin.v11
trainer:
  lr: 1e-5
orchestrator:
  iterations: 3
  run_seed: 20261102
"""



def standin(root: str = "/scratch/aorlta/runs") -> dict:
    return parse_yaml(STANDIN_YAML.format(root=root))


def edit(base: dict, dotted: str, value) -> dict:
    data = json.loads(json.dumps(base))
    *parents, last = dotted.split(".")
    node = data
    for key in parents:
        node = node[int(key)] if isinstance(node, list) else node.setdefault(key, {})
    if isinstance(node, list):
        node[int(last)] = value
    else:
        node[last] = value
    return data


# --- loading and defaults -------------------------------------------------------------------------


def test_the_example_config_loads_with_doc_defaults(tmp_path: Path):
    path = tmp_path / "dev.yaml"
    path.write_text(STANDIN_YAML.format(root=tmp_path / "runs"))
    cfg = load_config(path)
    assert cfg.run.run_id == "dev-standin-001"
    assert cfg.rollout.thinking == "off" and cfg.rollout.engines == (0, 1, 2, 3)
    assert cfg.trainer.lr == 1e-05 and type(cfg.trainer.lr) is float
    assert cfg.rollout.temperature == 0.7 and cfg.rollout.max_new_tokens == 4096 and cfg.rollout.max_repair_turns == 2
    assert cfg.reward.timeouts.asan_request == 900.0 and cfg.reward.asan.max_packages == 16
    assert cfg.selector.mode == "log_only" and cfg.sync.slots == 2 and cfg.orchestrator.min_scored_fraction == 0.9
    assert cfg.trainer.pg_loss == "ppo_clip" and cfg.trainer.aggregation == "sample_mean" and cfg.trainer.kl_beta == 0.0
    # derived: the loss temperature is the rollout temperature, and P is the prompts per iteration
    assert cfg.trainer.loss_temperature == 0.7
    assert cfg.orchestrator.prompts_per_iteration == 32
    assert cfg.run_dir == tmp_path / "runs" / "standin" / "dev-standin-001"


# --- site files -----------------------------------------------------------------------------------


def test_a_site_file_sets_the_site_paths_and_nothing_else_changes(tmp_path: Path, monkeypatch):
    path = tmp_path / "run.yaml"
    path.write_text(json.dumps(edit(standin(), "reward.remote", REMOTE)))
    site = tmp_path / "site.yaml"
    site.write_text("run: {root: /local/runs}\nmodel: {base_dir: /local/hf/model}\n"
                    "data: {prompt_pools: [/local/pool/listings]}\nsync: {dir: /dev/shm/me}\n"
                    "reward: {remote: {socket_dir: /dev/shm/aorlta-checkers/me}}\n")
    expected = C.plain(load_config(path))
    expected["run"]["root"], expected["model"]["base_dir"] = "/local/runs", "/local/hf/model"
    expected["data"]["prompt_pools"][0]["path"], expected["sync"]["dir"] = "/local/pool/listings", "/dev/shm/me"
    expected["reward"]["remote"]["socket_dir"] = "/dev/shm/aorlta-checkers/me"
    assert C.plain(load_config(path, site=site)) == expected
    monkeypatch.setenv(C.SITE_ENV, str(site))
    assert C.plain(load_config(path)) == expected


@pytest.mark.parametrize("site, message", [
    ("rollout: {K: 16}\n", "rollout.K: a site file sets only"),
    ("run: {run_id: other}\n", "run.run_id: a site file sets only"),
    ("data: {prompt_pools: [/a, /b]}\n", "data.prompt_pools: give a list of 1 path"),
    ("data: {prompt_pools: {path: /a}}\n", "data.prompt_pools.path: a site file sets only"),
    ("reward: {remote: {socket_dir: /dev/shm/x}}\n", "reward.remote.socket_dir: the config has no reward.remote"),
    ("- /a\n", "a site file must be a mapping"),
    ("run: {root: relative/runs}\n", "root"),
])
def test_a_site_file_cannot_change_what_the_run_does(tmp_path: Path, site, message):
    path = tmp_path / "run.yaml"
    path.write_text(STANDIN_YAML.format(root="/scratch/runs"))
    (tmp_path / "site.yaml").write_text(site)
    with pytest.raises(ConfigError, match=message):
        load_config(path, site=tmp_path / "site.yaml")


def test_config_is_frozen():
    cfg = config_from_dict(standin())
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.rollout.K = 16
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.rollout = None


def test_mappings_in_the_config_are_frozen_too():
    data = edit(edit(standin(), "reward.profile", "standin-gpu"), "reward.gpu_check_s", {"san.consan": 30.0})
    cfg = config_from_dict(edit(data, "reward.remote", REMOTE))
    resolved = C.plain(cfg)
    with pytest.raises(TypeError):
        cfg.reward.gpu_check_s["san.consan"] = -1.0
    with pytest.raises(TypeError):
        cfg.reward.gpu_check_s["not.a.leg"] = 1.0
    with pytest.raises(AttributeError):
        cfg.reward.remote.adapters.clear()
    with pytest.raises(TypeError):
        del cfg.reward.remote.adapters["standin.aorta_asan"]
    assert C.plain(cfg) == resolved
    assert cfg.reward.gpu_check_s == {"san.consan": 30.0} and cfg.reward.remote.gpus("standin.aorta_asan") == (5,)
    assert C.RewardSection(profile="fake", reward_fn="standin.v11", gpu_check_s={}).gpu_check_s == {}


def test_a_config_with_mappings_round_trips():
    data = edit(edit(standin(), "reward.profile", "standin-gpu"), "reward.gpu_check_s", {"san.consan": 30.0})
    cfg = config_from_dict(edit(data, "reward.remote", REMOTE))
    assert config_from_dict(C.plain(cfg)) == cfg and json.loads(json.dumps(C.plain(cfg))) == C.plain(cfg)
    assert pickle.loads(pickle.dumps(cfg)) == cfg and copy.deepcopy(cfg) == cfg
    assert dataclasses.replace(cfg.reward, gpu_check_s={"san.asan": 45.0}).gpu_check_s == {"san.asan": 45.0}
    with pytest.raises(ConfigError, match=r"gpu_check_s\.san\.asan: must be > 0"):
        dataclasses.replace(cfg.reward, gpu_check_s={"san.asan": 0.0})


def test_sections_built_in_code_are_checked_and_normalised_too():
    assert C.RolloutSection(temperature=1).temperature == 1.0
    assert type(C.RolloutSection(temperature=1).temperature) is float
    assert C.RolloutSection(engines=[2, 3]).engines == (2, 3)
    with pytest.raises(ConfigError, match="K"):
        C.RolloutSection(K=1)
    with pytest.raises(ConfigError):
        C.RolloutSection(K="8")


# --- unknown keys ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dotted, where",
    [
        ("rolout", "config: unknown key(s) 'rolout'"),
        ("rollout.KK", "rollout: unknown key(s) 'KK'"),
        ("reward.asan.max_packges", "reward.asan: unknown key(s) 'max_packges'"),
        ("data.prompt_pools.0.src", "data.prompt_pools[0]: unknown key(s) 'src'"),
        ("orchestrator.deadlines.generation", "orchestrator.deadlines: unknown key(s) 'generation'"),
    ],
)
def test_unknown_keys_are_an_error_at_every_depth(dotted, where):
    with pytest.raises(ConfigError) as err:
        config_from_dict(edit(standin(), dotted, 1))
    assert where in str(err.value)


def test_a_merge_key_is_not_a_way_around_unknown_keys():
    with pytest.raises(ConfigError, match="rollout: unknown key.*'<<'"):
        config_from_dict(parse_yaml(STANDIN_YAML.format(root="/r").replace("  K: 8\n", "  <<: {K: 8}\n")))


def test_missing_required_keys_are_named():
    data = standin()
    del data["reward"]["profile"]
    with pytest.raises(ConfigError, match="reward: missing required key.*'profile'"):
        config_from_dict(data)
    data = standin()
    del data["orchestrator"]
    with pytest.raises(ConfigError, match="config: missing required key.*'orchestrator'"):
        config_from_dict(data)


# --- sources (Section 6.10 item 8) ----------------------------------------------------------------


def test_only_the_standin_source_is_known():
    for dotted in ("run.source", "model.source", "data.prompt_pools.0.source"):
        with pytest.raises(ConfigError, match="expected one of"):
            config_from_dict(edit(standin(), dotted, "other"))


@pytest.mark.parametrize("dotted", ["rollout.compile_adapter", "rollout.parse_adapter", "rollout.feedback_template",
                                    "rollout.leak_check_adapter"])
def test_a_component_of_no_known_source_is_refused(dotted):
    with pytest.raises(ConfigError, match="must be 'none', start with 'standin' or 'fake', or be registered by a plugin"):
        config_from_dict(edit(standin(), dotted, "other.component"))








# --- YAML strictness ------------------------------------------------------------------------------


def test_duplicate_keys_are_an_error():
    with pytest.raises(ConfigError, match="duplicate key 'K'"):
        parse_yaml("rollout:\n  K: 8\n  K: 16\n")


def test_yaml_1_2_scalars():
    assert parse_yaml("a: off\nb: on\nc: yes\nd: 1e-5\ne: 010\nf: 2026-11-02\ng: true\nh: 1.5\ni: ~") == {
        "a": "off", "b": "on", "c": "yes", "d": 1e-05, "e": 10, "f": "2026-11-02", "g": True, "h": 1.5, "i": None
    }


@pytest.mark.parametrize(
    "dotted, value, message",
    [
        ("rollout.K", True, "rollout.K: expected an int"),
        ("rollout.K", 8.0, "rollout.K: expected an int"),
        ("trainer.lr", "1e-5", "trainer.lr: expected a finite number"),
        ("trainer.lr", float("inf"), "trainer.lr: expected a finite number"),
        ("trainer.lr", 10 ** 400, "trainer.lr: expected a finite number"),  # too large for a float
        ("rollout.thinking", False, "quote it"),
        ("rollout.engines", 0, "rollout.engines: expected a list"),
        ("run.run_id", 7, "run.run_id: expected a string"),
        ("reward.speculative_parallel", "yes", "reward.speculative_parallel: expected a bool"),
    ],
)
def test_types_are_strict(dotted, value, message):
    with pytest.raises(ConfigError) as err:
        config_from_dict(edit(standin(), dotted, value))
    assert message in str(err.value)


@pytest.mark.parametrize("bad", ["", "a: [1\n", "- 1\n- 2\n", "a: 1\n---\nb: 2\n", "x: !!python/object:os.system {}\n"])
def test_unusable_files_are_refused(bad):
    with pytest.raises(ConfigError):
        config_from_dict(parse_yaml(bad))


# --- value rules ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dotted, value, message",
    [
        ("rollout.K", 1, "rollout.K: must be >= 2"),
        ("rollout.P", 0, "rollout.P: must be >= 1"),
        ("rollout.engines", [], "rollout.engines: at least one"),
        ("rollout.engines", [0, 0], "rollout.engines: a GPU is listed twice"),
        ("rollout.temperature", 0, "rollout.temperature: must be > 0"),
        ("rollout.top_p", 1.5, "rollout.top_p"),
        ("rollout.top_k", 0, "rollout.top_k"),
        ("rollout.max_repair_turns", 4, "rollout.max_repair_turns"),
        ("rollout.gpu_memory_utilization", 0, "rollout.gpu_memory_utilization"),
        ("rollout.compile_adapter", "hipcc", "rollout.compile_adapter: component name 'hipcc'"),
        ("rollout.parse_adapter", "none", "rollout.parse_adapter: cannot be 'none'"),
        ("reward.reward_fn", "other.v1", "reward.reward_fn: 'other.v1' is not standin.v11 or a reward function a plugin"),
        ("reward.asan.max_packages", 17, "reward.asan.max_packages"),
        ("reward.timeouts.gate", 0, "reward.timeouts.gate: must be > 0"),
        ("selector.mode", "plan", "selector.mode: 'plan' is v2"),
        ("selector.scoring", "capped", "selector.cap_value"),
        ("selector.cap_value", 0.5, "selector.cap_value: is set but scoring is 'unscored'"),
        ("trainer.aggregation", "mean", "trainer.aggregation"),
        ("trainer.aggregation", "constant(0)", "trainer.aggregation"),
        ("trainer.betas", [0.9], "trainer.betas: expected 2 items"),
        ("trainer.betas", [0.9, 1.0], "trainer.betas: each must be in [0, 1)"),
        ("trainer.lr", 0, "trainer.lr: must be > 0"),
        ("trainer.tis", 0, "trainer.tis"),
        ("sync.slots", 1, "sync.slots: must be >= 2"),
        ("sync.dir", "shm/aorlta", "sync.dir: must be an absolute path"),
        ("orchestrator.min_scored_per_group", 1, "orchestrator.min_scored_per_group: must be >= 2"),
        ("orchestrator.min_scored_fraction", 0, "orchestrator.min_scored_fraction"),
        ("orchestrator.prompts_per_iteration", 16, "must equal rollout.P"),
        ("orchestrator.min_scored_per_group", 9, "rollout.K (8) is below orchestrator.min_scored_per_group (9)"),
        ("orchestrator.run_seed", -1, "orchestrator.run_seed"),
        ("run.run_id", "a/b", "run.run_id"),
        ("run.run_id", "..", "run.run_id"),
        ("run.root", "runs", "run.root: must be an absolute path"),
    ],
)
def test_value_rules(dotted, value, message):
    with pytest.raises(ConfigError) as err:
        config_from_dict(edit(standin(), dotted, value))
    assert message in str(err.value)


def test_accepted_alternatives():
    cfg = config_from_dict(edit(edit(standin(), "trainer.aggregation", "constant(4096)"), "trainer.tis", 2.0))
    assert cfg.trainer.aggregation == "constant(4096)" and cfg.trainer.tis == 2.0
    cfg = config_from_dict(edit(edit(standin(), "selector.scoring", "capped"), "selector.cap_value", 0.3))
    assert cfg.selector.cap_value == 0.3
    assert config_from_dict(edit(standin(), "trainer.loss_temperature", 1.0)).trainer.loss_temperature == 1.0


# --- run_plan.json --------------------------------------------------------------------------------

COMMIT = "0123456789abcdef0123456789abcdef01234567"


def test_run_plan_records_config_commit_and_file_hashes(tmp_path: Path):
    path = tmp_path / "dev.yaml"
    path.write_text(STANDIN_YAML.format(root=tmp_path / "runs"))
    pool = tmp_path / "pool.json"
    pool.write_text("[]")
    cfg = load_config(path)
    plan_path = write_run_plan(cfg, config_path=path, extra_files={"pool": pool}, commit=COMMIT)
    assert plan_path == cfg.run_dir / "run_plan.json"
    assert sorted(p.name for p in cfg.run_dir.iterdir()) == ["run_plan.json"]  # nothing else, no .partial
    plan = json.loads(plan_path.read_text())
    assert plan["schema"] == "aorlta.run_plan/1" and plan["run_id"] == "dev-standin-001" and plan["source"] == "standin"
    assert plan["config"] == C.plain(cfg) and plan["config_sha256"] == sha256_json(C.plain(cfg))
    assert plan["config"]["trainer"]["loss_temperature"] == 0.7  # the resolved config, not the file
    assert plan["config_file"]["sha256"] == sha256_file(path)
    assert plan["extra_files"]["pool"]["sha256"] == sha256_file(pool)
    code = plan["code"]
    assert code["commit"] == COMMIT and code["dirty"] is None
    pkg = Path(C.__file__).resolve().parent
    assert code["files"]["config.py"] == sha256_file(pkg / "config.py")
    assert code["files"]["ids.py"] == sha256_file(pkg / "ids.py")
    assert code["files_sha256"] == sha256_json(code["files"])


def test_run_plan_is_written_once(tmp_path: Path):
    cfg = config_from_dict(standin(str(tmp_path)))
    first = write_run_plan(cfg, commit=COMMIT)
    before = first.read_bytes()
    with pytest.raises(ConfigError, match="already set"):
        write_run_plan(cfg, commit=COMMIT)
    assert write_run_plan(cfg, commit=COMMIT, exist_ok=True) == first and first.read_bytes() == before
    changed = config_from_dict(edit(standin(str(tmp_path)), "rollout.K", 16))
    with pytest.raises(ConfigError, match="different config"):
        write_run_plan(changed, commit=COMMIT, exist_ok=True)


def test_run_plan_comes_before_anything_else(tmp_path: Path):
    cfg = config_from_dict(standin(str(tmp_path)))
    cfg.run_dir.mkdir(parents=True)
    (cfg.run_dir / "episodes.jsonl").write_text("")
    with pytest.raises(ConfigError, match="plan must be written first"):
        write_run_plan(cfg, commit=COMMIT)


COMMIT_B = "fedcba9876543210fedcba9876543210fedcba98"


@pytest.mark.parametrize("changed, exist_ok, refusal", [
    (True, False, "already set"),
    (False, False, "already set"),
    (True, True, "different config"),
    (False, True, None),
])
def test_launchers_racing_to_start_one_run_publish_one_plan(tmp_path: Path, monkeypatch, changed, exist_ok, refusal):
    """Both launchers pass the checks before either publishes; the first then publishes, and the second is judged
    against that plan as if it had started later. The published plan is never replaced."""
    first = config_from_dict(standin(str(tmp_path)))
    second = config_from_dict(edit(standin(str(tmp_path)), "rollout.K", 16)) if changed else first
    checked = {COMMIT: threading.Event(), COMMIT_B: threading.Event()}
    go = {COMMIT: threading.Event(), COMMIT_B: threading.Event()}
    real = C.code_identity

    def code_identity(package_dir=None, *, commit=None):  # runs after the checks, before the publication
        checked[commit].set()
        assert go[commit].wait(30)
        return real(package_dir, commit=commit)

    monkeypatch.setattr(C, "code_identity", code_identity)
    out = {}

    def launch(cfg, commit):
        try:
            out[commit] = write_run_plan(cfg, commit=commit, exist_ok=exist_ok)
        except ConfigError as exc:
            out[commit] = exc

    a = threading.Thread(target=launch, args=(first, COMMIT))
    b = threading.Thread(target=launch, args=(second, COMMIT_B))
    a.start()
    assert checked[COMMIT].wait(30)
    b.start()
    assert checked[COMMIT_B].wait(30)
    go[COMMIT].set()
    a.join(30)
    go[COMMIT_B].set()
    b.join(30)
    plan_path = first.run_dir / "run_plan.json"
    plan = json.loads(plan_path.read_text())
    assert out[COMMIT] == plan_path
    assert plan["code"]["commit"] == COMMIT and plan["config_sha256"] == sha256_json(C.plain(first))
    if refusal is None:
        assert out[COMMIT_B] == plan_path
    else:
        assert isinstance(out[COMMIT_B], ConfigError) and refusal in str(out[COMMIT_B])
    assert sorted(p.name for p in first.run_dir.iterdir()) == ["run_plan.json"]


@pytest.mark.parametrize("exist_ok", [False, True])
def test_a_plan_published_between_the_checks_is_judged_like_an_existing_one(tmp_path: Path, monkeypatch, exist_ok):
    cfg = config_from_dict(standin(str(tmp_path)))
    plan_path = write_run_plan(cfg, commit=COMMIT)  # the racing launcher's, published after this one looked
    before = plan_path.read_bytes()
    real = Path.exists
    looked = []

    def exists(self, **kwargs):
        if self == plan_path and not looked:
            looked.append(self)
            return False
        return real(self, **kwargs)

    monkeypatch.setattr(Path, "exists", exists)
    if exist_ok:
        assert write_run_plan(cfg, commit=COMMIT_B, exist_ok=True) == plan_path
    else:
        with pytest.raises(ConfigError, match="already set"):
            write_run_plan(cfg, commit=COMMIT_B)
    assert looked and plan_path.read_bytes() == before


def test_a_racing_launchers_temporary_file_is_not_a_file_of_the_run(tmp_path: Path):
    cfg = config_from_dict(standin(str(tmp_path)))
    cfg.run_dir.mkdir(parents=True)
    (cfg.run_dir / "run_plan.json.4242-17-0a1b2c3d.partial").write_text("{")
    assert write_run_plan(cfg, commit=COMMIT) == cfg.run_dir / "run_plan.json"


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True, check=True).stdout.strip()


def test_code_identity_reads_the_commit_and_dirty_state(tmp_path: Path):
    repo = tmp_path / "repo"
    pkg = repo / "src" / "pkg"
    pkg.mkdir(parents=True)
    (pkg / "m.py").write_text("x = 1\n")
    _git(repo, "init", "-q")
    _git(repo, "add", ".")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-q", "-m", "init")
    ident = C.code_identity(pkg)
    assert ident["commit"] == _git(repo, "rev-parse", "HEAD") and ident["dirty"] is False
    assert ident["files"] == {"m.py": sha256_file(pkg / "m.py")}
    (pkg / "m.py").write_text("x = 2\n")
    assert C.code_identity(pkg)["dirty"] is True


def test_code_identity_without_git_needs_an_explicit_commit(tmp_path: Path):
    (tmp_path / "m.py").write_text("")
    with pytest.raises(ConfigError, match="pass commit= explicitly"):
        C.code_identity(tmp_path)
    assert C.code_identity(tmp_path, commit=COMMIT)["commit"] == COMMIT
    with pytest.raises(ConfigError, match="not a full git commit"):
        C.code_identity(tmp_path, commit="abc123")


def test_code_identity_of_this_package():
    ident = C.code_identity()
    assert len(ident["commit"]) == 40 and "config.py" in ident["files"]


# --- the first GPU stand-in run (decisions of 3 Nov, 18:05) ------------------------------------------------------
# The standin-gpu profile, for stand-in runs only; `reward.remote` for checkers in their own process;
# `trainer.require_fast_path`; the gap gate's report-only mode, also for stand-in runs only; and the run's own config.

REMOTE = {"socket_dir": "/dev/shm/aorlta-checkers/x", "adapters": {"standin.aorta_consan": 4, "standin.aorta_asan": 5}}


def test_a_standin_run_can_select_standin_gpu_with_its_checkers_remote():
    data = edit(standin(), "reward.profile", "standin-gpu")
    cfg = config_from_dict(edit(data, "reward.remote", REMOTE))
    assert cfg.reward.profile == "standin-gpu" and cfg.reward.reward_fn == "standin.v11"
    assert cfg.reward.remote.adapters == {"standin.aorta_consan": 4, "standin.aorta_asan": 5}
    assert cfg.reward.remote.connect_timeout_s == 120.0
    assert config_from_dict(standin()).reward.remote is None






@pytest.mark.parametrize("remote, message", [
    ({"socket_dir": "relative/dir", "adapters": {"standin.aorta_asan": 5}}, "absolute path"),
    ({"socket_dir": "/dev/shm/c", "adapters": {}}, "name at least one adapter"),
    ({"socket_dir": "/dev/shm/c", "adapters": {"standin.aorta_asan": -1}}, "GPU index must be >= 0"),
    ({"socket_dir": "/dev/shm/c", "adapters": {"nobody": 1}}, "must be 'none', start with 'standin' or 'fake'"),
    ({"socket_dir": "/dev/shm/c", "adapters": {"standin.aorta_asan": 5}, "connect_timeout_s": 0}, "must be > 0"),
    ({"socket_dir": "/dev/shm/c", "adapters": {"standin.aorta_asan": 5}, "extra": 1}, "unknown key"),
])
def test_reward_remote_is_checked(remote, message):
    with pytest.raises(ConfigError, match=message):
        config_from_dict(edit(standin(), "reward.remote", remote))


def test_startup_warmup_defaults_to_off_and_can_be_set():
    assert config_from_dict(standin()).trainer.startup_warmup is False
    assert config_from_dict(edit(standin(), "trainer.startup_warmup", True)).trainer.startup_warmup is True
    with pytest.raises(ConfigError, match="expected a bool"):
        config_from_dict(edit(standin(), "trainer.startup_warmup", 1))


def test_trainer_devices_default_to_colocated_and_must_give_one_distinct_gpu_per_rank():
    assert config_from_dict(standin()).trainer.devices is None
    two = edit(edit(standin(), "trainer.ranks", 2), "trainer.devices", [2, 3])
    assert config_from_dict(two).trainer.devices == (2, 3)
    for devices, message in (([2], "one GPU per rank"), ([2, 2], "listed twice"), ([2, -1], ">= 0")):
        with pytest.raises(ConfigError, match=message):
            config_from_dict(edit(two, "trainer.devices", devices))


def test_require_fast_path_defaults_to_true_and_can_be_set():
    assert config_from_dict(standin()).trainer.require_fast_path is True
    assert config_from_dict(edit(standin(), "trainer.require_fast_path", False)).trainer.require_fast_path is False
    with pytest.raises(ConfigError, match="expected a bool"):
        config_from_dict(edit(standin(), "trainer.require_fast_path", "yes"))


def test_the_gap_gate_is_enforced_by_default_and_can_be_report_only():
    assert config_from_dict(standin()).trainer.gap_gate == "enforce"
    assert config_from_dict(edit(standin(), "trainer.gap_gate", "report_only")).trainer.gap_gate == "report_only"
    with pytest.raises(ConfigError, match="expected one of"):
        config_from_dict(edit(standin(), "trainer.gap_gate", "off"))
