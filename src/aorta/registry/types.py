"""Data types for the mitigations + environments registry."""

import copy
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Mitigation:
    """A named bundle of environment variables that modifies workload behavior.

    `frozen=True` prevents reassigning attributes (e.g. `m.name = "x"` raises),
    but the `env` dict itself is still mutable in place. Callers should treat
    `env` as read-only; `get_mitigation()` returns a defensive copy.
    """

    name: str
    env: dict[str, str]
    source_package: str  # "aorta" for built-ins, dist name for entry-point contributors


@dataclass(frozen=True)
class Environment:
    """A baseline process / container recipe for a workload run.

    `docker`, `venv`, `buck_target`, and `mirage_profile` are independent ways
    of describing the baseline; any combination (or none) may be set — built-in
    `local` has none (current process).

    No `rocm` field: ROCm version is implicit in the docker image digest,
    the host the venv runs on, or the captured `revision` of the Buck checkout;
    capture it from `aorta env probe` at runtime.

    `buck_target` is a Buck2 target label (e.g. `"//workloads/recom_repro:recom_repro"`).
    Interpreted by Buck-aware workload wrappers analogous to how `docker` is
    interpreted by docker-aware wrappers: the platform threads the field; the
    wrapper decides to shell out to `buck2 run <label>`. The platform itself
    does not invoke Buck (mirrors the no-docker-launching-in-platform policy:
    the platform threads tier hints, wrappers decide how to launch).

    `mirage_profile` / `emulator` describe a GPU-emulated baseline driven by the
    mirage control plane + rocjitsu software emulator (run AORTA workloads with
    no physical GPU). `mirage_profile` is the name of a mirage profile (which
    itself encodes the emulator backend, topology, and exec mode) and is
    **required** for emulated launches. `emulator` is an optional backend hint
    (for example, `"rocjitsu"`, `"rocjitsu-dbt"`, or `"noop"`) for discovery
    and documentation; the launch path still requires `mirage_profile` because
    the profile is authoritative (see `aorta.emulation.mirage_launch`). Same
    threading contract as the fields above: the platform threads them into
    `_aorta_environment`; an emulation-aware launch backend decides how to
    launch (e.g. `mirage run --profile <p> -- <argv>`, or an `LD_PRELOAD` env
    overlay). The platform itself launches nothing.

    `env` is a mapping of baseline environment variables intrinsic to this
    environment -- the lowest layer of the platform-wide env contract
    (`Environment.env < mitigations < recipe extra_env < cell/CLI extra_env`;
    see the dispatcher). Empty by default so every plugin / sidecar that omits
    it round-trips unchanged. Values are kept strictly `str`; the loaders do
    NOT coerce YAML/JSON numbers or booleans. Defensively deep-copied on
    construction (`frozen=True` blocks attribute reassignment but not mutation
    of the mapping in place), mirroring the `RunRequest` pattern.
    """

    name: str
    docker: str | None = None
    venv: str | None = None
    buck_target: str | None = None
    emulator: str | None = None
    mirage_profile: str | None = None
    source_package: str = "aorta"
    env: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # ``frozen=True`` blocks reassigning ``self.env`` but not mutating the
        # passed-in dict in place. Deep-copy via ``object.__setattr__`` so a
        # caller cannot mutate an environment's baseline env after construction
        # (same defensive pattern as ``RunRequest.__post_init__``).
        object.__setattr__(self, "env", copy.deepcopy(self.env))
