"""Plugins: a task's own prompts, answer parsing, checks and reward function, by name, without editing this package.

A run config lists its plugins under `plugins:`. Each entry is, in this order: the path of a `.py` file (a relative
path is read from the package root, the directory that holds `src/aorlta`, so a plugin committed beside the package
runs from the run job's snapshot of the tree); the name of an entry point in the `aorlta.plugins` group, which an
installed package of your own declares (`[project.entry-points."aorlta.plugins"] my-task = "my_pkg.plugin"`), so a
private task never has to be copied into this repository; or an importable module name. Each is imported once, before the rest of the config is
checked, and registers its components with the `register_*` functions below. A config, a profile or the wiring
then finds a name among the built-in components first and the registered ones after:

| config key | registered with |
|---|---|
| `data.prompt_pools[i].task` | `register_task`: the pool's prompt records, from the pool's directory |
| `rollout.parse_adapter` | `register_parse_adapter` (`rollout.chain.ParseAdapter`) |
| `rollout.feedback_template` | `register_feedback_template` (`rollout.chain.FeedbackTemplate`) |
| `rollout.compile_adapter` | `register_compile_adapter` (`rollout.compile_check.make_compiler`'s arguments) |
| `reward.profile` | `register_profile`: a profile YAML file (`reward/profiles.py`) |
| `reward.reward_fn` | `register_reward_fn` (`reward.rewardfn.base.RewardFn`) |
| a profile's pool `adapter` | `register_check_adapter` (`reward.adapters.base.CheckAdapter`), one per worker |

A registered component belongs to the `standin` source: its data is public, and its text may appear in logs and
tracebacks as the stand-in data's does. Data that must not, and a reward outside the stand-in range and parts
(`reward/service.py`), still need a change to this package. A name is registered once; registering it again with
another object is refused, with the same object (a module imported twice) it is a no-op.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import sys
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[2]  # the directory that holds src/aorlta
ENTRY_POINT_GROUP = "aorlta.plugins"


class PluginError(ValueError):
    """A plugin could not be loaded, or registered a name twice."""


@dataclass(frozen=True)
class Task:
    """A prompt source for `data.prompt_pools[i].task`.

    `records(pool_dir, tokenizer, thinking)` returns the pool's `PromptRecord`s (`schemas.py`), each with
    `source="standin"`, a unique `prompt_id` and a `task.task_sha256` that hashes what the checks read for it.
    `probe_messages(pool_dir)`, if given, returns the chats (a system and a user message) the landing check and the
    trainer's warm-up use; without it they are the first records' own messages. `selftest_messages(pool_dir)`, if
    given, returns the 4 chats each engine's start-up self-test samples twice with different seeds; at least 3 of the
    4 pairs must differ, so they must be open-ended requests. Without it the self-test uses
    `orchestrator.wiring.SELF_TEST_CHATS`: a precise task (a short kernel with one obvious answer) can give the same
    128 tokens for both seeds and fail a sound sampler."""

    name: str
    records: Callable[[Path, Any, str], list]
    probe_messages: Callable[[Path], list[list[dict]]] | None = None
    selftest_messages: Callable[[Path], list[list[dict]]] | None = None


@dataclass(frozen=True)
class CheckContext:
    """What a registered check adapter's factory is given, once per worker: its `settings` (the profile's
    `adapters:` entry under its name, with `${...}` tool roots expanded), the worker id, the service's wall clock,
    `timeout_s` (a little under its pool's timeout, so the adapter reports before the watchdog), the run's prompt pool
    directories, and `reward.gpu_check_s`."""

    name: str
    worker_id: str
    settings: Mapping[str, Any]
    clock: Callable[[], float]
    timeout_s: float
    pools: tuple[str, ...] = ()
    check_s: Mapping[str, float] = field(default_factory=dict)


TASKS: dict[str, Task] = {}
PARSE_ADAPTERS: dict[str, Callable[[], Any]] = {}
FEEDBACK_TEMPLATES: dict[str, Callable[[], Any]] = {}
COMPILE_ADAPTERS: dict[str, Callable[..., Any]] = {}
PROFILES: dict[str, Path] = {}
REWARD_FNS: dict[str, Callable[[Any], Any]] = {}
CHECK_ADAPTERS: dict[str, Callable[[CheckContext], Any]] = {}
LOADED: dict[str, dict] = {}  # plugin as listed -> {"module", "file", "sha256"}
_LOCK = threading.RLock()

_BUILT_IN = {
    "task": ("standin",),
    "parse adapter": ("standin.parse",),
    "feedback template": ("standin",),
    "compile adapter": ("none", "standin.hipcc", "fake.compile"),
    "profile": ("standin-static", "standin-gpu", "fake"),
    "reward function": ("standin.v11",),
    "check adapter": ("standin.parse", "standin.hipcc", "standin.rj_waitcheck", "standin.aorta_consan",
                      "standin.aorta_asan", "fake", "fake-asan"),
}


def _register(kind: str, table: dict, name: str, value: Any) -> None:
    if not (isinstance(name, str) and name and name.strip() == name and name != "none"):
        raise PluginError(f"a {kind} needs a non-empty name without surrounding spaces, other than 'none'; got {name!r}")
    if name in _BUILT_IN[kind]:
        raise PluginError(f"{kind} {name!r} is built in; register it under another name")
    with _LOCK:
        if name in table and table[name] is not value and table[name] != value:
            raise PluginError(f"{kind} {name!r} is already registered")
        table[name] = value


def register_task(task: Task) -> None:
    _register("task", TASKS, task.name, task)


def register_parse_adapter(name: str, factory: Callable[[], Any]) -> None:
    """`factory()` -> a ParseAdapter whose `name` is `name`."""
    _register("parse adapter", PARSE_ADAPTERS, name, factory)


def register_feedback_template(name: str, factory: Callable[[], Any]) -> None:
    """`factory()` -> a FeedbackTemplate whose `name` is `name`."""
    _register("feedback template", FEEDBACK_TEMPLATES, name, factory)


def register_compile_adapter(name: str, factory: Callable[..., Any]) -> None:
    """`factory(workdir, evidence=..., structure=...)` -> a compiler with `check(code, tag) -> CompileResult`."""
    _register("compile adapter", COMPILE_ADAPTERS, name, factory)


def register_profile(name: str, path: str | Path) -> None:
    """A check profile YAML file whose `name:` is `name`."""
    p = Path(path).resolve()
    if not p.is_file():
        raise PluginError(f"profile {name!r}: no file at {p}")
    _register("profile", PROFILES, name, p)


def register_reward_fn(name: str, factory: Callable[[Any], Any]) -> None:
    """`factory(profile)` -> a RewardFn whose `name` is `name`."""
    _register("reward function", REWARD_FNS, name, factory)


def register_check_adapter(name: str, factory: Callable[[CheckContext], Any]) -> None:
    """`factory(ctx)` -> a CheckAdapter whose `name` is `name`, for one worker."""
    _register("check adapter", CHECK_ADAPTERS, name, factory)


def registered(name: str) -> bool:
    """Whether a plugin registered `name` as any kind of component."""
    return any(name in t for t in (TASKS, PARSE_ADAPTERS, FEEDBACK_TEMPLATES, COMPILE_ADAPTERS, PROFILES, REWARD_FNS,
                                   CHECK_ADAPTERS))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_plugins(plugins: Sequence[str]) -> None:
    """Import every plugin listed, once each (see the module docstring for the two forms)."""
    if isinstance(plugins, (str, bytes)) or not isinstance(plugins, (list, tuple)):
        raise PluginError(f"plugins: expected a list of module names or .py paths, got {type(plugins).__name__}")
    for entry in plugins:
        if not (isinstance(entry, str) and entry):
            raise PluginError(f"plugins: each entry is a module name or a .py path, got {entry!r}")
        with _LOCK:
            if entry in LOADED:
                continue
            LOADED[entry] = _load(entry)


def _load(entry: str) -> dict:
    if entry.endswith(".py") or "/" in entry:
        path = Path(entry)
        path = (path if path.is_absolute() else PACKAGE_ROOT / path).resolve()
        if not path.is_file():
            raise PluginError(f"plugins: no file at {path}")
        name = "aorlta_plugin_" + hashlib.sha256(str(path).encode()).hexdigest()[:12]
        if name in sys.modules:
            module = sys.modules[name]
        else:
            spec = importlib.util.spec_from_file_location(name, path)
            if spec is None or spec.loader is None:
                raise PluginError(f"plugins: cannot import {path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            try:
                spec.loader.exec_module(module)
            except Exception as exc:
                del sys.modules[name]
                raise PluginError(f"plugins: importing {path} failed: {type(exc).__name__}: {exc}") from exc
    else:
        from importlib.metadata import entry_points

        named = [ep for ep in entry_points(group=ENTRY_POINT_GROUP) if ep.name == entry]
        if len(named) > 1:
            raise PluginError(f"plugins: {len(named)} installed packages declare the entry point {entry!r} in "
                              f"{ENTRY_POINT_GROUP}")
        target = named[0].value.split(":", 1)[0].strip() if named else entry
        try:
            module = importlib.import_module(target)
        except Exception as exc:
            raise PluginError(f"plugins: importing {target} failed: {type(exc).__name__}: {exc}") from exc
    file = getattr(module, "__file__", None)
    return {"module": module.__name__, "file": file, "sha256": _sha256(Path(file)) if file else None}


def identity() -> dict:
    """The loaded plugins as the run plan records them: each entry's module, file and the file's sha256."""
    with _LOCK:
        return {k: dict(v) for k, v in sorted(LOADED.items())}


def profile_path(name: str) -> Path | None:
    return PROFILES.get(name)


def task(name: str) -> Task:
    try:
        return TASKS[name]
    except KeyError:
        raise PluginError(f"no task named {name!r}; registered: {', '.join(sorted(TASKS)) or 'none'}") from None
