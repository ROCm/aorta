"""Check profiles (design Section 6.2, Stages): a small YAML file naming the stages, the
legs each runs, the worker pool each uses, which legs need which, and the per-leg settings that are
part of the cache key. Stages can change without code changes.

    name: fake                    # the Candidate.check_profile it serves
    reward_fn: standin.v11
    clean_repeats: 1              # k settled-clean repeats
    materialize: false            # write each candidate's package to disk (real adapters read files)
    pools:   {<pool>: {resource, adapter, workers, timeout, controls, batched}}
    stages:  [{name, pool, legs: [...], parallel: true | false | speculative}]
    requires: {<leg>: [<leg>, ...]}   # a leg runs only after these are clean
    leg_settings: {<leg>: {...}}
    adapters: {<adapter name>: {...}} # settings for the adapter factories

Unknown keys are errors, as in the run config. Profiles shipped with the package are in
`profiles/*.yaml`; `load_profile("fake")` reads `profiles/fake.yaml`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from ..config import ConfigError, parse_yaml, strict_decode

PROFILE_DIR = Path(__file__).resolve().parent / "profiles"
PROFILE_NAMES = ("standin-static", "standin-gpu", "fake")
TIMEOUT_KEYS = ("gate", "compile", "sim_run", "asan_request")


@dataclass(frozen=True)
class PoolSpec:
    resource: Literal["local_cpu", "sim_cpu", "partner_gpu", "own_gpu"]
    adapter: str
    timeout: Literal["gate", "compile", "sim_run", "asan_request"]
    workers: int | None = None  # None: the sim pool uses reward.sim_workers; a batched pool grows on demand
    controls: bool = False  # per-worker trusted-control brackets (per request when batched)
    batched: bool = False  # the ASan endpoint: packages go in requests of up to asan.max_packages


@dataclass(frozen=True)
class StageSpec:
    name: str
    pool: str
    legs: tuple[str, ...]
    parallel: bool | Literal["speculative"] = False  # "speculative": reward.speculative_parallel decides


@dataclass(frozen=True)
class Profile:
    name: str
    reward_fn: str
    stages: tuple[StageSpec, ...]
    pools: dict[str, PoolSpec]
    clean_repeats: int = 1
    materialize: bool = False
    requires: dict[str, tuple[str, ...]] = field(default_factory=dict)
    leg_settings: dict[str, dict[str, Any]] = field(default_factory=dict)
    adapters: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        problems = profile_problems(self)
        if problems:
            raise ConfigError(f"profile {self.name!r}: " + "; ".join(problems))

    @property
    def legs(self) -> tuple[str, ...]:
        return tuple(leg for stage in self.stages for leg in stage.legs)

    def stage_of(self, leg: str) -> StageSpec:
        for stage in self.stages:
            if leg in stage.legs:
                return stage
        raise KeyError(leg)


def profile_problems(p: Profile) -> list[str]:
    out = []
    if p.name not in PROFILE_NAMES:
        out.append(f"name must be one of {PROFILE_NAMES}")
    if p.clean_repeats < 1:
        out.append("clean_repeats must be >= 1")
    if not p.stages:
        out.append("at least one stage is needed")
    seen: dict[str, str] = {}
    names = set()
    for stage in p.stages:
        if stage.name in names:
            out.append(f"stage {stage.name!r} is listed twice")
        names.add(stage.name)
        if stage.pool not in p.pools:
            out.append(f"stage {stage.name!r} uses unknown pool {stage.pool!r}")
        if not stage.legs:
            out.append(f"stage {stage.name!r} has no legs")
        for leg in stage.legs:
            if leg in seen:
                out.append(f"leg {leg!r} is in stages {seen[leg]!r} and {stage.name!r}")
            seen[leg] = stage.name
        pool = p.pools.get(stage.pool)
        if pool is not None and pool.batched and len(stage.legs) != 1:
            out.append(f"stage {stage.name!r}: a batched pool serves exactly one leg")
    for pname, pool in p.pools.items():
        if pool.workers is not None and pool.workers < 1:
            out.append(f"pool {pname!r}: workers must be >= 1")
        if not pool.adapter:
            out.append(f"pool {pname!r}: adapter must be named")
    order = {leg: i for i, leg in enumerate(p.legs)}
    for leg, needs in p.requires.items():
        if leg not in order:
            out.append(f"requires names leg {leg!r}, which no stage runs")
            continue
        for need in needs:
            if need not in order:
                out.append(f"{leg!r} requires {need!r}, which no stage runs")
            elif order[need] >= order[leg]:
                out.append(f"{leg!r} requires {need!r}, which does not come before it")
    for leg in p.leg_settings:
        if leg not in order:
            out.append(f"leg_settings names leg {leg!r}, which no stage runs")
    return out


def profile_from_dict(data: Mapping[str, Any], source: str = "<profile>") -> Profile:
    if not isinstance(data, Mapping):
        raise ConfigError(f"{source}: a profile is a mapping")
    try:
        return strict_decode(Profile, data, "")
    except ConfigError as exc:
        raise ConfigError(f"{source}: {exc}") from None


def load_profile(name_or_path: str | Path) -> Profile:
    path = Path(name_or_path)
    if path.suffix not in (".yaml", ".yml"):
        path = PROFILE_DIR / f"{name_or_path}.yaml"
    return profile_from_dict(parse_yaml(path.read_bytes(), str(path)), str(path))
