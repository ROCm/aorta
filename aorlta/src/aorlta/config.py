"""Run configuration (design Section 5 item 4; Section 6.10 item 8).

One YAML file per run is loaded into frozen dataclasses. Every key any component reads is declared
here, so an unknown key anywhere is an error, as is a duplicate key, a wrong type, or a value outside
its range. The sections and their defaults are the "Configuration." lines of Sections 6.1-6.6, plus
`run`, `model` and `data`, which say what the run is and what it reads.

`write_run_plan` writes the resolved config, the code commit and the file hashes to
`<run_dir>/run_plan.json`, before anything else is written for the run.
"""

from __future__ import annotations

import copy
import dataclasses
import datetime as _dt
import functools
import json
import math
import os
import re
import shutil
import subprocess
import sys
import types
import typing
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from .atomic import atomic_write_json, is_partial
from .ids import IdError, sha256_file, sha256_json, validate_run_id
from .rollout.ports import PORT_STRIDE

Source = Literal["standin"]  # the stand-in data source; a new source adds a value here
RUN_PLAN_SCHEMA = "aorlta.run_plan/1"
RUN_PLAN_NAME = "run_plan.json"
REWARD_MIN, REWARD_MAX = -1.55, 1.0  # reward v11's range (Appendix B)


class ConfigError(ValueError):
    """The configuration is malformed, out of range or self-contradictory."""


# --- YAML ------------------------------------------------------------------------------------------


class _StrictLoader(yaml.SafeLoader):
    """SafeLoader with YAML 1.2 core-schema scalars, and duplicate keys refused.

    PyYAML follows YAML 1.1, under which `thinking: off` is the bool False, `lr: 1e-5` is the string
    "1e-5", `010` is 8 and a repeated key silently replaces the first. Here only true/false are
    bools, ints are decimal, `1e-5` is a float, dates stay strings, and merge keys (`<<`) are plain
    keys (so they are rejected as unknown).
    """


_YAML11_TAGS = {
    "tag:yaml.org,2002:bool",
    "tag:yaml.org,2002:int",
    "tag:yaml.org,2002:float",
    "tag:yaml.org,2002:timestamp",
    "tag:yaml.org,2002:merge",
    "tag:yaml.org,2002:value",
}
_StrictLoader.yaml_implicit_resolvers = {
    first: [(tag, rx) for tag, rx in resolvers if tag not in _YAML11_TAGS]
    for first, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
_StrictLoader.add_implicit_resolver(
    "tag:yaml.org,2002:bool", re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"), list("tTfF")
)
# int before float: the float pattern also matches plain digits.
_StrictLoader.add_implicit_resolver("tag:yaml.org,2002:int", re.compile(r"^[-+]?[0-9]+$"), list("-+0123456789"))
_StrictLoader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        r"^(?:[-+]?(?:\.[0-9]+|[0-9]+(?:\.[0-9]*)?)(?:[eE][-+]?[0-9]+)?"
        r"|[-+]?\.(?:inf|Inf|INF)|\.(?:nan|NaN|NAN))$"
    ),
    list("-+0123456789."),
)


def _construct_int(loader: _StrictLoader, node: yaml.ScalarNode) -> int:
    return int(loader.construct_scalar(node), 10)


def _construct_map(loader: _StrictLoader, node: yaml.MappingNode) -> dict:
    seen: set[Any] = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=True)
        if key in seen:
            raise ConfigError(f"duplicate key {key!r} at line {key_node.start_mark.line + 1}")
        seen.add(key)
    return loader.construct_mapping(node, deep=True)


_StrictLoader.add_constructor("tag:yaml.org,2002:int", _construct_int)
_StrictLoader.add_constructor("tag:yaml.org,2002:map", _construct_map)


def parse_yaml(text: str | bytes, source: str = "<config>") -> Any:
    try:
        return yaml.load(text, Loader=_StrictLoader)  # a SafeLoader subclass: no arbitrary objects
    except ConfigError as exc:
        raise ConfigError(f"{source}: {exc}") from None
    except yaml.YAMLError as exc:
        raise ConfigError(f"{source}: invalid YAML: {exc}") from None


# --- strict decoding into dataclasses --------------------------------------------------------------

_NoneType = type(None)


def _join(path: str, name: str) -> str:
    return f"{path}.{name}" if path else name


def _describe(value: Any) -> str:
    return f"{type(value).__name__} {value!r}"[:120]


@functools.cache
def _hints(cls: type) -> dict[str, Any]:
    return typing.get_type_hints(cls)


def strict_decode(tp: Any, value: Any, path: str = "") -> Any:
    """Check `value` against the annotation `tp` and convert it to that type.

    No coercion beyond int -> float for float fields: a bool is not an int, a string is not a
    number, floats must be finite, dict keys must be str, and a dataclass mapping must have no
    unknown keys and every required key. Lists become tuples where the annotation is a tuple.
    """
    where = path or "config"
    if tp is Any:
        return _check_json(value, where)
    if tp is _NoneType or tp is None:
        if value is None:
            return None
        raise ConfigError(f"{where}: expected null, got {_describe(value)}")
    origin = typing.get_origin(tp)
    if origin in (typing.Union, types.UnionType):
        args = typing.get_args(tp)
        if value is None:
            if _NoneType in args:
                return None
            raise ConfigError(f"{where}: must not be null")
        options = [a for a in args if a is not _NoneType]
        if len(options) == 1:
            return strict_decode(options[0], value, path)
        errors = []
        for option in options:
            try:
                return strict_decode(option, value, path)
            except ConfigError as exc:
                errors.append(str(exc))
        raise ConfigError(f"{where}: no alternative matched ({'; '.join(errors)})")
    if origin is Literal:
        options = typing.get_args(tp)
        for option in options:
            if type(value) is type(option) and value == option:
                return value
        hint = ""
        if type(value) is bool and any(isinstance(o, str) and o in ("on", "off") for o in options):
            hint = " (quote it: unquoted on/off are not strings in every YAML parser)"
        raise ConfigError(f"{where}: expected one of {', '.join(map(repr, options))}, got {_describe(value)}{hint}")
    if tp is bool:
        if type(value) is bool:
            return value
        raise ConfigError(f"{where}: expected a bool, got {_describe(value)}")
    if tp is int:
        if type(value) is int:
            return value
        raise ConfigError(f"{where}: expected an int, got {_describe(value)}")
    if tp is float:
        if type(value) in (int, float):
            try:
                f = float(value)
            except OverflowError:  # an int too large for a float
                f = math.inf
            if math.isfinite(f):
                return f
        raise ConfigError(f"{where}: expected a finite number, got {_describe(value)}")
    if tp is str:
        if type(value) is str:
            return value
        raise ConfigError(f"{where}: expected a string, got {_describe(value)}")
    if origin is list or tp is list:
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{where}: expected a list, got {_describe(value)}")
        (item,) = typing.get_args(tp) or (Any,)
        return [strict_decode(item, v, f"{path}[{i}]") for i, v in enumerate(value)]
    if origin is tuple or tp is tuple:
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{where}: expected a list, got {_describe(value)}")
        args = typing.get_args(tp)
        if not args:
            return tuple(_check_json(v, f"{path}[{i}]") for i, v in enumerate(value))
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(strict_decode(args[0], v, f"{path}[{i}]") for i, v in enumerate(value))
        if len(value) != len(args):
            raise ConfigError(f"{where}: expected {len(args)} items, got {len(value)}")
        return tuple(strict_decode(a, v, f"{path}[{i}]") for i, (a, v) in enumerate(zip(args, value)))
    if origin is dict or tp is dict:
        if not isinstance(value, Mapping):
            raise ConfigError(f"{where}: expected a mapping, got {_describe(value)}")
        key_tp, val_tp = typing.get_args(tp) or (str, Any)
        if key_tp is not str:
            raise TypeError(f"unsupported dict key annotation {key_tp!r}")
        out = {}
        for k, v in value.items():
            if type(k) is not str:
                raise ConfigError(f"{where}: keys must be strings, got {_describe(k)}")
            out[k] = strict_decode(val_tp, v, _join(path, k))
        return out
    if dataclasses.is_dataclass(tp):
        return _decode_dataclass(tp, value, path)
    raise TypeError(f"unsupported annotation {tp!r}")


def _check_json(value: Any, where: str) -> Any:
    """Accept any JSON value (str keys, finite floats) without converting it."""
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if math.isfinite(value):
            return value
        raise ConfigError(f"{where}: non-finite number {value!r}")
    if isinstance(value, (list, tuple)):
        return [_check_json(v, f"{where}[{i}]") for i, v in enumerate(value)]
    if isinstance(value, Mapping):
        out = {}
        for k, v in value.items():
            if type(k) is not str:
                raise ConfigError(f"{where}: keys must be strings, got {_describe(k)}")
            out[k] = _check_json(v, f"{where}.{k}")
        return out
    raise ConfigError(f"{where}: not a JSON value: {_describe(value)}")


def _decode_dataclass(cls: type, value: Any, path: str) -> Any:
    where = path or "config"
    if isinstance(value, cls):
        return value
    if not isinstance(value, Mapping):
        raise ConfigError(f"{where}: expected a mapping for {cls.__name__}, got {_describe(value)}")
    fields = {f.name: f for f in dataclasses.fields(cls) if f.init}
    for key in value:
        if type(key) is not str:
            raise ConfigError(f"{where}: keys must be strings, got {_describe(key)}")
    unknown = sorted(k for k in value if k not in fields)
    if unknown:
        raise ConfigError(
            f"{where}: unknown key(s) {', '.join(map(repr, unknown))}; allowed: {', '.join(fields)}"
        )
    hints = _hints(cls)
    kwargs, missing = {}, []
    for name, f in fields.items():
        if name in value:
            kwargs[name] = strict_decode(hints[name], value[name], _join(path, name))
        elif f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
            missing.append(name)
    if missing:
        raise ConfigError(f"{where}: missing required key(s) {', '.join(map(repr, missing))}")
    try:
        return cls(**kwargs)
    except ConfigError as exc:
        raise ConfigError(f"{path}.{exc}" if path else str(exc)) from None


def plain(obj: Any) -> Any:
    """A dataclass tree as plain JSON values (tuples become lists)."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: plain(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [plain(v) for v in obj]
    if isinstance(obj, Mapping):
        return {k: plain(v) for k, v in obj.items()}
    return obj


class _FrozenMapping(Mapping):
    """A read-only mapping, so a frozen section's mapping cannot change after its checks; unlike
    types.MappingProxyType it pickles and copies, as the sections do."""

    def __init__(self, data: Mapping) -> None:
        self._data = dict(data)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._data!r})"


def _frozen(value: Any) -> Any:
    if isinstance(value, dict):
        return _FrozenMapping({k: _frozen(v) for k, v in value.items()})
    return value


class _Section:
    """Frozen config section: normalises and checks its fields on construction, from YAML or code."""

    def __post_init__(self) -> None:
        hints = _hints(type(self))
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            normal = _frozen(strict_decode(hints[f.name], value, f.name))
            if normal is not value:
                object.__setattr__(self, f.name, normal)
        self._validate()

    def _validate(self) -> None:
        pass


def _check(ok: bool, message: str) -> None:
    if not ok:
        raise ConfigError(message)


def _absolute(name: str, value: str) -> None:
    _check(bool(value) and Path(value).is_absolute(), f"{name}: must be an absolute path, got {value!r}")


_COMPONENT_RE = re.compile(r"(standin|fake)(?:[.-][A-Za-z0-9_.-]+)?")


def component_source(name: str) -> str | None:
    """The data source a named component belongs to: standin* and fake* are stand-ins, and so is a component a
    plugin registered (`plugins.py`); "none" (only allowed for the leak check) belongs to none."""
    if name == "none":
        return None
    from . import plugins

    if plugins.registered(name):
        return "standin"
    m = _COMPONENT_RE.fullmatch(name)
    if m is None:
        raise ConfigError(f"component name {name!r} must be 'none', start with 'standin' or 'fake', or be registered "
                          f"by a plugin")
    return "standin"


# --- sections --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RunSection(_Section):
    run_id: str
    source: Source
    root: str = "/scratch/aorlta/runs"  # the run directory is <root>/<source>/<run_id>

    def _validate(self) -> None:
        try:
            validate_run_id(self.run_id)
        except IdError as exc:
            raise ConfigError(f"run_id: {exc}") from None
        _absolute("root", self.root)


@dataclass(frozen=True)
class ModelSection(_Section):
    base_dir: str  # the merged base directory: v0 weights and tokenizer (Section 6.1, engine start)
    source: Source
    # A PEFT LoRA adapter trained on base_dir, for a score run only: tools/merge_base_adapter.py merges the two into the
    # run's node-local directory before the engines start, and the engines serve the merge.
    adapter_dir: str | None = None

    def _validate(self) -> None:
        _absolute("base_dir", self.base_dir)
        if self.adapter_dir is not None:
            _absolute("adapter_dir", self.adapter_dir)


@dataclass(frozen=True)
class PromptPool(_Section):
    path: str
    source: Source
    # what reads the pool's directory: "standin" (a kernel dataset's listings, rollout/templates.py) or a task a
    # plugin registers (plugins.register_task)
    task: str = "standin"

    def _validate(self) -> None:
        _absolute("path", self.path)
        if self.task != "standin":
            from . import plugins

            _check(self.task in plugins.TASKS, f"task: {self.task!r} is not 'standin' or a task a plugin registered "
                                               f"(registered: {', '.join(sorted(plugins.TASKS)) or 'none'})")


@dataclass(frozen=True)
class DataSection(_Section):
    prompt_pools: tuple[PromptPool, ...]

    def _validate(self) -> None:
        _check(len(self.prompt_pools) >= 1, "prompt_pools: at least one pool is needed")
        paths = [p.path for p in self.prompt_pools]
        _check(len(set(paths)) == len(paths), "prompt_pools: a path is listed twice")


@dataclass(frozen=True)
class RolloutSection(_Section):
    """Section 6.1."""

    engines: tuple[int, ...] = (0, 1, 2, 3)  # GPU indices, one engine per GPU (D10)
    K: int = 8  # chains per prompt
    P: int = 32  # prompts per iteration
    max_repair_turns: int = 2
    thinking: Literal["off", "on"] = "off"  # D9
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = -1
    max_new_tokens: int = 4096
    max_feedback_chars: int = 2000
    request_timeout_s: float = 1800.0
    gpu_memory_utilization: float = 0.45
    compile_adapter: str = "standin.hipcc"  # "none": no compile step; a parsed answer ends its chain
    parse_adapter: str = "standin.parse"
    leak_check_adapter: str = "none"  # the stand-in prompts need none (Section 6.1, Leak check)
    feedback_template: str = "standin"
    # how a repair turn's ids are built: "append" (Appendix C) or "rerender" (the whole conversation through the chat
    # template again); null: "append"
    repair_turn_ids: Literal["append", "rerender"] | None = None
    # stop when more than this share of an iteration's groups have all K answers identical (sampler suspected). A task
    # with near-unique answers and few prompts may raise it: one such group in 8 is 12.5%. The engines' start-up
    # self-test checks the sampler on its own.
    identical_groups_stop: float = 0.05
    # the engines' host ports. null: each engine start picks a free window of 2000 ports at random, outside the
    # kernel's ephemeral range, and tries another if one is taken before the engine binds it (rollout/ports.py), so
    # two runs can share a node. An int in [1024, 65536 - 2000 x engines]: engine i on port_base + 2000 i, started
    # once; 20000 is the layout every run used before this key. The ports used are in <run_dir>/engines.jsonl.
    port_base: int | None = None

    def _validate(self) -> None:
        _check(len(self.engines) >= 1, "engines: at least one engine GPU is needed")
        if self.port_base is not None:
            top = 65536 - PORT_STRIDE * len(self.engines)
            _check(1024 <= self.port_base <= top,
                   f"port_base: must be null or in [1024, {top}] ({len(self.engines)} engines, {PORT_STRIDE} ports "
                   f"each), got {self.port_base}")
        _check(all(g >= 0 for g in self.engines), f"engines: GPU indices must be >= 0, got {list(self.engines)}")
        _check(len(set(self.engines)) == len(self.engines), f"engines: a GPU is listed twice: {list(self.engines)}")
        _check(self.K >= 2, f"K: must be >= 2 so a group can have an advantage, got {self.K}")
        _check(self.P >= 1, f"P: must be >= 1, got {self.P}")
        _check(0 <= self.max_repair_turns <= 3, f"max_repair_turns: must be in [0, 3], got {self.max_repair_turns}")
        _check(self.temperature > 0, f"temperature: must be > 0 (0 samples greedily), got {self.temperature}")
        _check(0 < self.top_p <= 1, f"top_p: must be in (0, 1], got {self.top_p}")
        _check(self.top_k == -1 or self.top_k >= 1, f"top_k: must be -1 (off) or >= 1, got {self.top_k}")
        _check(self.max_new_tokens >= 1, f"max_new_tokens: must be >= 1, got {self.max_new_tokens}")
        _check(self.max_feedback_chars >= 1, f"max_feedback_chars: must be >= 1, got {self.max_feedback_chars}")
        _check(self.request_timeout_s > 0, f"request_timeout_s: must be > 0, got {self.request_timeout_s}")
        _check(0 <= self.identical_groups_stop <= 1,
               f"identical_groups_stop: must be in [0, 1], got {self.identical_groups_stop}")
        _check(
            0 < self.gpu_memory_utilization <= 1,
            f"gpu_memory_utilization: must be in (0, 1], got {self.gpu_memory_utilization}",
        )
        for name in ("compile_adapter", "parse_adapter", "feedback_template", "leak_check_adapter"):
            value = getattr(self, name)
            _check(value != "none" or name in ("leak_check_adapter", "compile_adapter"), f"{name}: cannot be 'none'")
            try:
                component_source(value)
            except ConfigError as exc:
                raise ConfigError(f"{name}: {exc}") from None


@dataclass(frozen=True)
class RewardCache(_Section):
    enabled: bool = True
    scope: Literal["run", "shared"] = "run"  # Section 6.2


@dataclass(frozen=True)
class RewardTimeouts(_Section):
    gate: float = 60.0
    compile: float = 120.0
    sim_run: float = 120.0
    asan_request: float = 900.0

    def _validate(self) -> None:
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            _check(value > 0, f"{f.name}: must be > 0 seconds, got {value}")


@dataclass(frozen=True)
class RewardAsan(_Section):
    max_packages: int = 16
    max_wait_s: float = 20.0
    max_inflight: int | None = None  # null: no limit

    def _validate(self) -> None:
        _check(1 <= self.max_packages <= 16, f"max_packages: must be in [1, 16] (the endpoint limit), got {self.max_packages}")
        _check(self.max_wait_s > 0, f"max_wait_s: must be > 0, got {self.max_wait_s}")
        _check(self.max_inflight is None or self.max_inflight >= 1, f"max_inflight: must be >= 1, got {self.max_inflight}")


@dataclass(frozen=True)
class RewardControls(_Section):
    every_jobs: int = 32
    every_s: float = 300.0

    def _validate(self) -> None:
        _check(self.every_jobs >= 1, f"every_jobs: must be >= 1, got {self.every_jobs}")
        _check(self.every_s > 0, f"every_s: must be > 0, got {self.every_s}")


GPU_CHECK_LEGS = ("san.consan", "san.asan")  # the legs the standin-gpu profile runs on a GPU (reward.gpu_check_s)

_PROFILE_REWARD_FN = {
    "standin-static": "standin.v11",
    "standin-gpu": "standin.v11",
    "fake": "standin.v11",
}


def profile_reward_fn(profile: str) -> str:
    """The reward function a check profile names: the built-in map, or a registered profile's `reward_fn:`."""
    if profile in _PROFILE_REWARD_FN:
        return _PROFILE_REWARD_FN[profile]
    from . import plugins

    if profile not in plugins.PROFILES:
        raise ConfigError(f"profile: {profile!r} is not one of {', '.join(_PROFILE_REWARD_FN)} or a profile a plugin "
                          f"registered (registered: {', '.join(sorted(plugins.PROFILES)) or 'none'})")
    from .reward.profiles import load_profile

    return load_profile(profile).reward_fn


@dataclass(frozen=True)
class RewardRemote(_Section):
    """Adapters that run in their own checker process, reached over a Unix socket (`aorlta.reward.remote`). The
    stand-in GPU legs need the ROCm 7.0.2 ASan runtime and glibc >= 2.38, which only rocm/dev-ubuntu-24.04:7.0.2
    has, so `jobs/aorlta-run.sbatch` starts one server per adapter and GPU there, each seeing only its own GPU. An
    adapter given several GPUs gets one server on each, and its pool one worker per server for each worker the profile
    names (one hung kernel once held the only ConSan worker)."""

    socket_dir: str
    adapters: dict[str, int | tuple[int, ...]]  # adapter name -> the GPU index its server sees, or several GPUs
    connect_timeout_s: float = 120.0

    def gpus(self, name: str) -> tuple[int, ...]:
        """The GPUs serving `name`, one server each."""
        gpu = self.adapters[name]
        return (gpu,) if isinstance(gpu, int) else tuple(gpu)

    def _validate(self) -> None:
        _absolute("socket_dir", self.socket_dir)
        _check(bool(self.adapters), "adapters: name at least one adapter, or leave reward.remote out")
        for name in self.adapters:
            try:
                component_source(name)
            except ConfigError as exc:
                raise ConfigError(f"adapters: {exc}") from None
            gpus = self.gpus(name)
            _check(bool(gpus), f"adapters.{name}: name at least one GPU")
            for gpu in gpus:
                _check(gpu >= 0, f"adapters.{name}: the GPU index must be >= 0, got {gpu}")
            _check(len(set(gpus)) == len(gpus), f"adapters.{name}: a GPU is listed twice in {list(gpus)}")
        _check(self.connect_timeout_s > 0, f"connect_timeout_s: must be > 0, got {self.connect_timeout_s}")


@dataclass(frozen=True)
class RewardSection(_Section):
    """Section 6.2."""

    profile: str  # standin-static, standin-gpu, fake, or a profile a plugin registered
    reward_fn: str  # standin.v11, or a reward function a plugin registered; must be the one the profile names
    cache: RewardCache = field(default_factory=RewardCache)
    sim_workers: int | None = None  # None: sized by the reward service; "sized by measurement"
    speculative_parallel: bool = True
    timeouts: RewardTimeouts = field(default_factory=RewardTimeouts)
    stall_s: float = 300.0
    asan: RewardAsan = field(default_factory=RewardAsan)
    controls: RewardControls = field(default_factory=RewardControls)
    retries_infra: int = 1
    reuse_rollout_compile: bool = False
    remote: RewardRemote | None = None
    # The stand-in GPU legs' per-check wall-clock limit in seconds, per leg: one aorta run of one kernel. The program
    # is killed at the limit and the leg is unsettled, never clean. The pool timeouts stay the outer bound (a check
    # never gets more than its adapter's own share of them); a leg not named keeps that share as its limit.
    gpu_check_s: dict[str, float] = field(default_factory=dict)

    def _validate(self) -> None:
        expected = profile_reward_fn(self.profile)
        from . import plugins

        _check(self.reward_fn == "standin.v11" or self.reward_fn in plugins.REWARD_FNS,
               f"reward_fn: {self.reward_fn!r} is not standin.v11 or a reward function a plugin registered")
        _check(
            self.reward_fn == expected,
            f"reward_fn: profile {self.profile!r} is scored by {expected!r}, got {self.reward_fn!r}",
        )
        _check(self.sim_workers is None or self.sim_workers >= 1, f"sim_workers: must be >= 1, got {self.sim_workers}")
        _check(self.stall_s > 0, f"stall_s: must be > 0, got {self.stall_s}")
        _check(self.retries_infra >= 0, f"retries_infra: must be >= 0, got {self.retries_infra}")
        for leg, limit in self.gpu_check_s.items():
            _check(leg in GPU_CHECK_LEGS, f"gpu_check_s: {leg!r} is not a stand-in GPU leg {GPU_CHECK_LEGS}")
            _check(limit > 0, f"gpu_check_s.{leg}: must be > 0 seconds, got {limit}")


@dataclass(frozen=True)
class SelectorSection(_Section):
    """Section 6.3. v1 is log_only; plan mode is v2 (D6) and is refused until a reward version for
    it exists."""

    mode: Literal["off", "log_only", "plan"] = "log_only"
    endpoint: str | None = None
    plan_timeout_s: float = 5.0
    audit_fraction: float = 0.2
    scoring: Literal["unscored", "capped"] = "unscored"
    cap_value: float | None = None
    max_miss_rate: float = 0.02

    def _validate(self) -> None:
        _check(
            self.mode != "plan",
            "mode: 'plan' is v2 (decision D6) and needs a reward version that supports it; none does yet",
        )
        _check(self.endpoint is None or bool(self.endpoint), "endpoint: must be non-empty or null")
        _check(self.plan_timeout_s > 0, f"plan_timeout_s: must be > 0, got {self.plan_timeout_s}")
        _check(0 <= self.audit_fraction <= 1, f"audit_fraction: must be in [0, 1], got {self.audit_fraction}")
        _check(0 <= self.max_miss_rate <= 1, f"max_miss_rate: must be in [0, 1], got {self.max_miss_rate}")
        if self.scoring == "capped":
            _check(
                self.cap_value is not None and REWARD_MIN <= self.cap_value <= REWARD_MAX,
                f"cap_value: scoring 'capped' needs a cap in [{REWARD_MIN}, {REWARD_MAX}], got {self.cap_value}",
            )
        else:
            _check(self.cap_value is None, "cap_value: is set but scoring is 'unscored'")


_AGGREGATION_RE = re.compile(r"sample_mean|token_mean|prompt_token_mean|sample_sum|constant\([1-9][0-9]*\)")


@dataclass(frozen=True)
class TrainerSection(_Section):
    """Section 6.4."""

    ranks: int = 4
    # GPU indices, one per rank. Null: rank r sits beside engine r on its GPU (colocated, D10); a list gives a separate
    # layout, such as D10's fallback of 2 engine GPUs and 2 trainer GPUs.
    devices: tuple[int, ...] | None = None
    micro_batch_tokens: int = 65536
    grad_checkpointing: bool = False
    pg_loss: Literal["ppo_clip", "reinforce", "cispo"] = "ppo_clip"
    eps_clip_low: float = 0.2
    eps_clip_high: float = 0.2
    old_logprobs: Literal["trainer_recompute", "engine"] = "trainer_recompute"
    tis: float | None = None  # off, or the cap C
    aggregation: str = "sample_mean"  # sample_mean | token_mean | prompt_token_mean | constant(L_max) | sample_sum
    kl_beta: float = 0.0
    lr: float = 1e-5
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    grad_clip: float = 1.0  # 0 = off
    steps_per_iteration: int = 1
    loss_temperature: float | None = None  # null: the rollout temperature (Section 6.4, Numerics)
    lm_head_fp32: bool = False  # D4: only on both sides, after measuring
    max_generation_gap: float = 2.0
    max_off_policy_fraction: float = 0.02
    # "report_only": the gap is measured and the gate's verdict logged, but nothing is excluded and no step is
    # refused for it. Stand-in runs only, until T10 sets the gate's statistic and limits.
    gap_gate: Literal["enforce", "report_only"] = "enforce"
    require_fast_path: bool = True  # refuse to train on transformers' torch fallback for the linear attention
    # the gap gate's per-sample statistic of |trainer - engine| over the trained reply; T10 proposes mean_abs
    gap_statistic: Literal["max_abs", "mean_abs", "median_abs", "p99_abs"] = "max_abs"
    streaming_accumulation: bool = True
    # One forward and backward on a synthetic group at start-up, gradient dropped, so the first iteration does not pay
    # the kernels' one-time compilation (run 1's first two iterations trained at a third of iteration 3's speed).
    # Not a learning-rate warm-up: no step is taken and the trained state is unchanged.
    startup_warmup: bool = False

    def _validate(self) -> None:
        _check(self.ranks >= 1, f"ranks: must be >= 1, got {self.ranks}")
        if self.devices is not None:
            d = list(self.devices)
            _check(len(d) == self.ranks, f"devices: one GPU per rank ({self.ranks}), got {d}")
            _check(all(g >= 0 for g in d), f"devices: GPU indices must be >= 0, got {d}")
            _check(len(set(d)) == len(d), f"devices: a GPU is listed twice: {d}")
        _check(self.micro_batch_tokens >= 1, f"micro_batch_tokens: must be >= 1, got {self.micro_batch_tokens}")
        _check(self.eps_clip_low > 0, f"eps_clip_low: must be > 0, got {self.eps_clip_low}")
        _check(self.eps_clip_high > 0, f"eps_clip_high: must be > 0, got {self.eps_clip_high}")
        _check(self.tis is None or self.tis > 0, f"tis: the cap must be > 0 (or null for off), got {self.tis}")
        _check(
            _AGGREGATION_RE.fullmatch(self.aggregation) is not None,
            f"aggregation: expected sample_mean, token_mean, prompt_token_mean, constant(<L_max>) or sample_sum, "
            f"got {self.aggregation!r}",
        )
        _check(self.kl_beta >= 0, f"kl_beta: must be >= 0, got {self.kl_beta}")
        _check(self.lr > 0, f"lr: must be > 0, got {self.lr}")
        _check(all(0 <= b < 1 for b in self.betas), f"betas: each must be in [0, 1), got {list(self.betas)}")
        _check(self.eps > 0, f"eps: must be > 0, got {self.eps}")
        _check(self.weight_decay >= 0, f"weight_decay: must be >= 0, got {self.weight_decay}")
        _check(self.grad_clip >= 0, f"grad_clip: must be >= 0 (0 = off), got {self.grad_clip}")
        _check(self.steps_per_iteration >= 1, f"steps_per_iteration: must be >= 1, got {self.steps_per_iteration}")
        _check(
            self.loss_temperature is None or self.loss_temperature > 0,
            f"loss_temperature: must be > 0, got {self.loss_temperature}",
        )
        _check(self.max_generation_gap > 0, f"max_generation_gap: must be > 0, got {self.max_generation_gap}")
        _check(
            0 <= self.max_off_policy_fraction <= 1,
            f"max_off_policy_fraction: must be in [0, 1], got {self.max_off_policy_fraction}",
        )


@dataclass(frozen=True)
class SyncSection(_Section):
    """Section 6.5."""

    dir: str = "/dev/shm/aorlta"
    slots: int = 2
    partial: bool = True  # D11
    hash: Literal["xxh64"] = "xxh64"
    probes: int = 4
    probe_tokens: int = 1024
    tau_floor: float = 0.01
    round_trip_at_start: bool = True
    stochastic_rounding: bool = False

    def _validate(self) -> None:
        _absolute("dir", self.dir)
        _check(self.slots >= 2, f"slots: must be >= 2 so the slot being written is never the one served, got {self.slots}")
        _check(self.probes >= 1, f"probes: must be >= 1, got {self.probes}")
        _check(self.probe_tokens >= 1, f"probe_tokens: must be >= 1, got {self.probe_tokens}")
        _check(self.tau_floor > 0, f"tau_floor: must be > 0, got {self.tau_floor}")


@dataclass(frozen=True)
class OrchestratorDeadlines(_Section):
    generation_s: float = 1800.0  # floor; the deadline is max(this, 3 x rolling median) (Section 6.6)
    scoring_s: float = 1800.0  # floor; max(this, 2 x rolling median of the scoring phase)

    def _validate(self) -> None:
        _check(self.generation_s > 0, f"generation_s: must be > 0, got {self.generation_s}")
        _check(self.scoring_s > 0, f"scoring_s: must be > 0, got {self.scoring_s}")


@dataclass(frozen=True)
class ScoreOptions(_Section):
    """`orchestrator.mode: score`: which prompts are scored, and how often. Each pass gives every selected prompt
    `rollout.K` samples, with seeds of its own; `rollout.P` prompts are scored at a time."""

    prompts: int | None = None  # the first N prompts of the pool in the run seed's order; null: every prompt
    passes: int = 1

    def _validate(self) -> None:
        _check(self.prompts is None or self.prompts >= 1,
               f"prompts: must be >= 1 or null (every prompt), got {self.prompts}")
        _check(self.passes >= 1, f"passes: must be >= 1, got {self.passes}")


@dataclass(frozen=True)
class OrchestratorSection(_Section):
    """Section 6.6. `mode: score` generates and scores without training (orchestrator/score.py): no trainer step, no
    optimizer and no weight sync; its batches follow from `score`, so `iterations` stays unset."""

    run_seed: int
    iterations: int | None = None  # training iterations: required in mode v1, unset in mode score
    prompts_per_iteration: int | None = None  # null: rollout.P (the two keys name one number)
    mode: Literal["v1", "score"] = "v1"  # v2 (one-step-off-policy) is not built (D5)
    deadlines: OrchestratorDeadlines = field(default_factory=OrchestratorDeadlines)
    min_scored_fraction: float = 0.9  # D7
    min_scored_per_group: int = 2
    eval_every: int | None = None  # null: no eval requests
    keep_ckpts: int = 2
    score: ScoreOptions | None = None  # mode score only; null there: every prompt, one pass

    def _validate(self) -> None:
        if self.mode == "score":
            _check(self.iterations is None, "iterations: a score run's batches follow from orchestrator.score; "
                                            "leave iterations out")
            if self.score is None:
                object.__setattr__(self, "score", ScoreOptions())
        else:
            _check(self.iterations is not None, "iterations: a training run (mode v1) needs it")
            _check(self.iterations >= 1, f"iterations: must be >= 1, got {self.iterations}")
            _check(self.score is None, "score: is read only with mode: score")
        _check(0 <= self.run_seed < 2**63, f"run_seed: must be in [0, 2**63), got {self.run_seed}")
        _check(
            self.prompts_per_iteration is None or self.prompts_per_iteration >= 1,
            f"prompts_per_iteration: must be >= 1, got {self.prompts_per_iteration}",
        )
        _check(
            0 < self.min_scored_fraction <= 1,
            f"min_scored_fraction: must be in (0, 1], got {self.min_scored_fraction}",
        )
        _check(
            self.min_scored_per_group >= 2,
            f"min_scored_per_group: must be >= 2 (an advantage needs two scored samples), got {self.min_scored_per_group}",
        )
        _check(self.eval_every is None or self.eval_every >= 1, f"eval_every: must be >= 1, got {self.eval_every}")
        _check(self.keep_ckpts >= 1, f"keep_ckpts: must be >= 1, got {self.keep_ckpts}")


@dataclass(frozen=True)
class RunConfig(_Section):
    run: RunSection
    model: ModelSection
    data: DataSection
    reward: RewardSection
    orchestrator: OrchestratorSection
    rollout: RolloutSection = field(default_factory=RolloutSection)
    selector: SelectorSection = field(default_factory=SelectorSection)
    trainer: TrainerSection = field(default_factory=TrainerSection)
    sync: SyncSection = field(default_factory=SyncSection)
    # modules or .py files that register a task's components (plugins.py); config_from_dict imports them before it
    # checks the sections
    plugins: tuple[str, ...] = ()

    def _validate(self) -> None:
        if self.trainer.loss_temperature is None:
            object.__setattr__(
                self, "trainer", dataclasses.replace(self.trainer, loss_temperature=self.rollout.temperature)
            )
        if self.orchestrator.prompts_per_iteration is None:
            object.__setattr__(
                self, "orchestrator", dataclasses.replace(self.orchestrator, prompts_per_iteration=self.rollout.P)
            )
        if self.rollout.repair_turn_ids is None:
            object.__setattr__(self, "rollout", dataclasses.replace(self.rollout, repair_turn_ids="append"))
        _check(
            self.orchestrator.prompts_per_iteration == self.rollout.P,
            f"orchestrator.prompts_per_iteration ({self.orchestrator.prompts_per_iteration}) must equal rollout.P "
            f"({self.rollout.P}); both name the prompts per iteration",
        )
        _check(
            self.rollout.K >= self.orchestrator.min_scored_per_group,
            f"rollout.K ({self.rollout.K}) is below orchestrator.min_scored_per_group "
            f"({self.orchestrator.min_scored_per_group}), so no group could ever be trained",
        )
        mixed = source_mismatches(self)
        _check(
            not mixed,
            f"mixed sources (Section 6.10 item 8): run.source is {self.run.source!r} but " + "; ".join(mixed),
        )
        if self.orchestrator.mode == "score":
            _check(self.sync.round_trip_at_start, "sync.round_trip_at_start: a score run confirms what the engines "
                                                  "serve with the start-up round trip, so it cannot be off")
            _check(self.orchestrator.eval_every is None, "orchestrator.eval_every: a score run trains nothing to "
                                                         "evaluate")
        else:
            _check(self.model.adapter_dir is None, "model.adapter_dir: a training run starts from model.base_dir; "
                                                   "an adapter is scored with orchestrator.mode: score")

    @property
    def run_dir(self) -> Path:
        """`<run.root>/<run.source>/<run.run_id>`: runs on different sources never share a directory."""
        return Path(self.run.root) / self.run.source / self.run.run_id


def source_mismatches(cfg: RunConfig) -> list[str]:
    """Every source-bearing entry whose source differs from run.source.

    Data (model, prompt pools) declares its source; components carry it in their name.
    """
    want = cfg.run.source
    out = []
    if cfg.model.source != want:
        out.append(f"model.source is {cfg.model.source!r}")
    for i, pool in enumerate(cfg.data.prompt_pools):
        if pool.source != want:
            out.append(f"data.prompt_pools[{i}].source is {pool.source!r}")
    components = {
        "rollout.compile_adapter": cfg.rollout.compile_adapter,
        "rollout.parse_adapter": cfg.rollout.parse_adapter,
        "rollout.leak_check_adapter": cfg.rollout.leak_check_adapter,
        "rollout.feedback_template": cfg.rollout.feedback_template,
        "reward.profile": cfg.reward.profile,
        "reward.reward_fn": cfg.reward.reward_fn,
    }
    if cfg.reward.remote is not None:
        for adapter in cfg.reward.remote.adapters:
            components[f"reward.remote.adapters.{adapter}"] = adapter
    for name, value in components.items():
        got = component_source(value)
        if got is not None and got != want:
            out.append(f"{name} {value!r} is a {got} component")
    return out


def _load_plugins(entries: Any, where: str = "plugins") -> None:
    from . import plugins

    try:
        plugins.load_plugins(entries)
    except plugins.PluginError as exc:
        raise ConfigError(f"{where}: {exc}") from None


def config_from_dict(data: Any, source: str = "<config>") -> RunConfig:
    if not isinstance(data, Mapping):
        raise ConfigError(f"{source}: the config must be a mapping, got {_describe(data)}")
    if data.get("plugins"):  # before the sections, whose names a plugin may register
        _load_plugins(data["plugins"], f"{source}: plugins")
    return strict_decode(RunConfig, data, "")


SITE_ENV = "AORLTA_SITE"
# What a site file may set: where a run's data and scratch are, never what the run does.
SITE_KEYS = ("run.root", "model.base_dir", "model.adapter_dir", "data.prompt_pools", "reward.remote.socket_dir",
             "sync.dir")


def _leaves(node: Mapping, prefix: str = ""):
    for key, value in node.items():
        if isinstance(value, Mapping):
            yield from _leaves(value, f"{prefix}{key}.")
        else:
            yield f"{prefix}{key}", value


def apply_site(data: Any, site: Any, source: str = "<site>") -> Any:
    """The run config `data` with a site file's paths in place: a mapping of the keys in SITE_KEYS, where
    `data.prompt_pools` is one path per pool of the config, in its order. A config can then be committed without
    the site's paths, and the run plan still records the paths a run used."""
    if not isinstance(site, Mapping):
        raise ConfigError(f"{source}: a site file must be a mapping, got {_describe(site)}")
    if not isinstance(data, Mapping):
        return data
    out = copy.deepcopy(dict(data))
    for key, value in _leaves(site):
        if key not in SITE_KEYS:
            raise ConfigError(f"{source}: {key}: a site file sets only {', '.join(SITE_KEYS)}")
        if key == "data.prompt_pools":
            pools = out.get("data", {}).get("prompt_pools") if isinstance(out.get("data"), Mapping) else None
            if not (isinstance(value, list) and isinstance(pools, list) and len(value) == len(pools)
                    and all(isinstance(p, Mapping) for p in pools)):
                n = len(pools) if isinstance(pools, list) else 0
                raise ConfigError(f"{source}: data.prompt_pools: give a list of {n} path(s), one per prompt pool")
            out["data"]["prompt_pools"] = [{**pool, "path": path} for pool, path in zip(pools, value)]
            continue
        *parents, leaf = key.split(".")
        node = out
        for name in parents:
            if name == "sync" and node is out and node.get("sync") is None:
                node["sync"] = {}
            node = node.get(name) if isinstance(node, Mapping) else None
        if not isinstance(node, dict):
            raise ConfigError(f"{source}: {key}: the config has no {'.'.join(parents)} section to set it in")
        node[leaf] = value
    return out


def load_config(path: str | Path, *, site: str | Path | None = None) -> RunConfig:
    """The run config at `path`, with the site file `site` applied (apply_site), or the one AORLTA_SITE names."""
    path = Path(path)
    data = parse_yaml(path.read_bytes(), str(path))
    site = site if site is not None else os.environ.get(SITE_ENV) or None
    if site is not None:
        site = Path(site)
        data = apply_site(data, parse_yaml(site.read_bytes(), str(site)), str(site))
    return config_from_dict(data, str(path))


# --- run_plan.json ---------------------------------------------------------------------------------

_COMMIT_RE = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")


def _git(package_dir: Path, *args: str) -> str:
    git = shutil.which("git") or "/usr/bin/git"
    proc = subprocess.run(
        [git, "-C", str(package_dir), *args],
        capture_output=True,
        text=True,
        timeout=60,
        env={"GIT_OPTIONAL_LOCKS": "0", "PATH": "/usr/bin:/bin", "HOME": str(Path.home())},
    )
    if proc.returncode != 0:
        raise ConfigError(f"git {' '.join(args)} failed in {package_dir}: {proc.stderr.strip()[:300]}")
    return proc.stdout


def code_identity(package_dir: str | Path | None = None, *, commit: str | None = None) -> dict:
    """The code commit, whether the package differs from it, and sha256 of every package file.

    Without git (for example inside a container) pass `commit` explicitly; `dirty` is then unknown
    (None) and the file hashes are what identifies the code.
    """
    pkg = Path(package_dir) if package_dir is not None else Path(__file__).resolve().parent
    files = {
        p.relative_to(pkg).as_posix(): sha256_file(p)
        for p in sorted(pkg.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts and p.suffix not in (".pyc", ".pyo")
    }
    if commit is None:
        try:
            commit = _git(pkg, "rev-parse", "HEAD").strip()
            dirty: bool | None = bool(_git(pkg, "status", "--porcelain", "--untracked-files=all", "--", ".").strip())
        except (OSError, subprocess.SubprocessError, ConfigError) as exc:
            raise ConfigError(f"cannot read the code commit in {pkg} ({exc}); pass commit= explicitly") from None
    else:
        dirty = None
    if not _COMMIT_RE.fullmatch(commit):
        raise ConfigError(f"not a full git commit id: {commit!r}")
    return {"commit": commit, "dirty": dirty, "package_dir": str(pkg), "files": files, "files_sha256": sha256_json(files)}


def write_run_plan(
    cfg: RunConfig,
    *,
    config_path: str | Path | None = None,
    extra_files: Mapping[str, str | Path] | None = None,
    commit: str | None = None,
    package_dir: str | Path | None = None,
    exist_ok: bool = False,
) -> Path:
    """Write `<run_dir>/run_plan.json`, the first file of a run.

    A run_id is set once: an existing plan is never overwritten, and of launchers racing to write it
    exactly one publishes; the others are checked against its plan as if they came later. With
    `exist_ok` (resume) an existing plan is accepted only if its resolved config hashes the same, and
    it is left untouched. A run directory that already holds other files but no plan is refused,
    because the plan comes first.
    """
    run_dir = cfg.run_dir
    plan_path = run_dir / RUN_PLAN_NAME
    resolved = plain(cfg)
    config_sha256 = sha256_json(resolved)
    if plan_path.exists():
        return _existing_plan(plan_path, cfg, config_sha256, exist_ok)
    # a racing launcher's plan or temporary file is not another file of the run: the publication below decides
    if run_dir.exists() and any(p.name != RUN_PLAN_NAME and not is_partial(p) for p in run_dir.iterdir()):
        raise ConfigError(f"{run_dir} already holds files but no {RUN_PLAN_NAME}; the plan must be written first")
    plan = {
        "schema": RUN_PLAN_SCHEMA,
        "run_id": cfg.run.run_id,
        "source": cfg.run.source,
        "run_dir": str(run_dir),
        "written_at": _dt.datetime.now(_dt.UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "config": resolved,
        "config_sha256": config_sha256,
        "config_file": (
            {"path": str(Path(config_path).resolve()), "sha256": sha256_file(config_path)} if config_path else None
        ),
        "code": code_identity(package_dir, commit=commit),
        "extra_files": {
            name: {"path": str(Path(p).resolve()), "sha256": sha256_file(p)}
            for name, p in sorted((extra_files or {}).items())
        },
        "environment": {"python": sys.version.split()[0], "pyyaml": yaml.__version__},
    }
    if cfg.plugins:
        from . import plugins

        loaded = plugins.identity()
        plan["plugins"] = {name: loaded.get(name) for name in cfg.plugins}
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        atomic_write_json(plan_path, plan, replace=False)
    except FileExistsError:  # a launcher racing this one published first
        return _existing_plan(plan_path, cfg, config_sha256, exist_ok)
    return plan_path


def _existing_plan(plan_path: Path, cfg: RunConfig, config_sha256: str, exist_ok: bool) -> Path:
    if not exist_ok:
        raise ConfigError(f"{plan_path} exists: run_id {cfg.run.run_id!r} is already set")
    recorded = json.loads(plan_path.read_text(encoding="utf-8")).get("config_sha256")
    if recorded != config_sha256:
        raise ConfigError(f"{plan_path} was written for a different config ({recorded} != {config_sha256})")
    return plan_path
