"""Data types for the mitigations + environments registry."""

from dataclasses import dataclass


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

    `docker`, `venv`, and `buck_target` are independent ways of describing the
    baseline; any combination (or none) may be set — built-in `local` has none
    (current process). No `rocm` field: ROCm version is implicit in the docker
    image digest, the host the venv runs on, or the captured `revision` of the
    Buck checkout; capture it from `aorta env probe` at runtime.

    `buck_target` is a Buck2 target label (e.g. `"//workloads/recom_repro:recom_repro"`).
    Interpreted by Buck-aware workload wrappers analogous to how `docker` is
    interpreted by docker-aware wrappers: the platform threads the field; the
    wrapper decides to shell out to `buck2 run <label>`. The platform itself
    does not invoke Buck (mirrors the no-docker-launching-in-platform policy
    documented in `aorta-internal#14`).
    """

    name: str
    docker: str | None = None
    venv: str | None = None
    buck_target: str | None = None
    source_package: str = "aorta"
