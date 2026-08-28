"""Run dispatcher - orchestrates workload execution across trials.

The dispatcher is the core of `aorta run`. It:
1. Discovers and instantiates workloads
2. Validates launch mode before execution
3. Applies environment and mitigation configuration
4. Runs trials and collects results
5. Persists results as JSON (rank 0 only for distributed)
"""

import contextlib
import copy
import json
import logging
import math
import os
import re
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from aorta._env_rules import is_valid_env_name, value_has_nul
from aorta.instrumentation.environment import collect_env
from aorta.registry import Environment, get_environment, get_mitigation
from aorta.run import _fsafe
from aorta.run._process import TrialWorkerError, launch_trial_worker
from aorta.run.collectors import KNOWN_RECIPES, validate_collectors
from aorta.run.discovery import (
    get_workload_class,
    get_workload_policy,
    get_workload_startup_env,
)
from aorta.run.results import TrialResult, trial_verdict
from aorta.run.validation import (
    TrialIsolation,
    resolve_trial_isolation,
    resolve_trial_isolation_policy,
    validate_launch_mode,
)
from aorta.workloads import Workload, WorkloadResult

logger = logging.getLogger(__name__)

_LAUNCHER_IDENTITY_KEYS = frozenset({
    "RANK",
    "WORLD_SIZE",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
    "GROUP_RANK",
    "GROUP_WORLD_SIZE",
    "ROLE_NAME",
    "ROLE_RANK",
    "ROLE_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "AORTA_TRIAL_MASTER_PORT_BASE",
})
_ISOLATION_RUN_GENERATION = 0


def _next_isolation_generation() -> int:
    global _ISOLATION_RUN_GENERATION
    generation = _ISOLATION_RUN_GENERATION
    _ISOLATION_RUN_GENERATION += 1
    return generation


def _isolated_fallback_port(
    isolation_generation: int,
    trial_idx: int,
) -> str:
    raw = os.environ.get("AORTA_TRIAL_MASTER_PORT_BASE")
    if raw is None:
        raise RuntimeError(
            "distributed process isolation without a torchrun agent store "
            "requires AORTA_TRIAL_MASTER_PORT_BASE to reserve a unique port "
            "range for this launcher job"
        )
    try:
        base = int(raw)
    except ValueError as exc:
        raise RuntimeError(
            f"AORTA_TRIAL_MASTER_PORT_BASE must be an integer, got {raw!r}"
        ) from exc
    port = base + (isolation_generation * 97 + trial_idx) % 1000
    if port < 1024 or port > 65535:
        raise RuntimeError(
            f"derived isolated worker port {port} is outside 1024..65535"
        )
    return str(port)


def _validate_env_mapping(label: str, env: object) -> None:
    """Validate a controlled environment overlay without exposing values.

    Name shape and NUL-in-value rules come from the shared ``aorta._env_rules``
    predicates (the single source of truth also used by the recipe parser and
    the registry loaders). The CLI enforces the same at parse time; library
    callers pass ``extra_env`` directly so we re-validate here for parity.
    """
    if not isinstance(env, dict):
        raise ValueError(
            f"{label} must be a dict[str, str], got {type(env).__name__}"
        )
    for key, value in env.items():
        if not isinstance(key, str):
            raise ValueError(
                f"{label} keys must be str, got {type(key).__name__}"
            )
        if not isinstance(value, str):
            raise ValueError(
                f"{label} value for key {key!r} must be str, "
                f"got {type(value).__name__}"
            )
        if value_has_nul(value):
            # A NUL byte is a valid Python str character but cannot be stored
            # in an OS environment variable. ``os.environ.update`` applies the
            # overlay entry-by-entry, so a NUL value part-way through would
            # raise AFTER earlier entries are already set -- and the overlay is
            # applied via a single ``update`` that lives OUTSIDE the try/finally
            # restore block, so those earlier entries would leak into later
            # matrix cells. Reject before any mutation. Value is NOT echoed.
            raise ValueError(
                f"{label} value for key {key!r} contains a NUL byte and cannot "
                "be stored in an environment variable."
            )
        if not is_valid_env_name(key):
            raise ValueError(
                f"Invalid {label} keys [{key!r}]: each key must match "
                "[A-Za-z_][A-Za-z0-9_]* (POSIX env-var name shape)."
            )


def _validate_json_native(value: Any, path: str = "config_overrides") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_native(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(
                    f"{path} keys must be strings for process isolation, "
                    f"got {type(key).__name__}"
                )
            _validate_json_native(item, f"{path}.{key}")
        return
    raise ValueError(
        f"{path} must contain only JSON-native values; "
        f"got {type(value).__name__}"
    )


@dataclass(frozen=True)
class RunRequest:
    """Configuration for a run_trials() invocation.

    The dataclass is ``frozen=True`` to prevent attribute reassignment,
    but ``extra_env`` and ``config_overrides`` are dicts -- ``frozen``
    does not stop callers from mutating those nested structures.
    ``__post_init__`` therefore stores deep copies, so an in-flight
    request can never be mutated out from under the dispatcher.  This
    mirrors the same defensive pattern used by :class:`TrialResult`.

    Note:
        :class:`TrialResult.execution_env` records the *effective*
        recipe (the declared environment plus any runtime overlays
        from CLI flags such as ``--buck-target``).  Replays should
        read ``execution_env`` field-by-field, not re-resolve the
        named environment, when overlays were used -- otherwise the
        replay silently drops the overlay and runs against a
        different recipe than the original trial.

    Attributes:
        workload: Name of the workload to run (from entry-point group).
        trials: Number of trials to execute.
        environment: Environment name (default: local).
        image: Optional runtime overlay for the resolved
            :class:`Environment`'s ``docker`` field. Symmetric peer
            of ``buck_target`` below: each overlays one axis of the
            named environment's recipe at run time and preserves the
            other axes (a single-axis pin). When set, takes effect
            AFTER :func:`get_environment` resolves ``environment``;
            ``None`` (the default) means "no override" so every
            pre-existing ``RunRequest`` invocation behaves unchanged.
            **Naming asymmetry**: the FIELD overlays
            ``Environment.docker`` (the recipe slot's name, peer of
            ``venv`` / ``buck_target``); the FIELD itself is named
            ``image`` (after the VALUE the operator provides -- an
            OCI image reference, typically a digest pin like
            ``sha256:<64-hex>`` or ``<repo>@sha256:<digest>``). Same
            convention as the CLI flag (``--image``). Threaded into
            ``config['_aorta_environment']['docker']`` for the
            workload's wrapper to consume. **Keyword-only**: same
            ``kw_only=True`` rationale as ``buck_target`` -- adding
            this field before existing positional fields like
            ``mitigations`` would otherwise shift their positional
            slots and silently break external positional callers.
        buck_target: Optional runtime overlay for the resolved
            :class:`Environment`'s ``buck_target`` field (#182). When
            set, takes effect AFTER :func:`get_environment` resolves
            ``environment``, so the named environment's other fields
            (``docker`` / ``venv`` / ``source_package``) are
            preserved -- the override is a pin on the Buck axis only.
            ``None`` (the default) means "no override": a named env
            that already declares ``buck_target`` keeps its value,
            and every pre-existing ``RunRequest`` invocation behaves
            unchanged. Symmetric peer of ``aorta env probe
            --buck-target`` (#163). Threaded into
            ``config['_aorta_environment']['buck_target']`` for the
            workload's wrapper to consume. **Keyword-only**: this
            field is declared with ``kw_only=True`` so adding it
            before existing positional fields like ``mitigations``
            does NOT shift the positional ``__init__`` signature
            (positional callers continue to interpret the 4th arg as
            ``mitigations``).
        mitigations: Tuple of mitigation names to apply.
        extra_env: Additional environment variables (override mitigations).
        trial_isolation: ``auto`` follows workload metadata; ``process`` uses
            a fresh interpreter per trial; ``in_process`` keeps the legacy
            lifecycle when permitted by the workload.
        steps: Number of steps per trial (workload-specific).
        config_overrides: Additional workload configuration.
        results_dir: Directory to write per-trial JSON files.
        collect: Collector recipe names, threaded to
            ``config['_aorta_collect']`` for the workload to act on.
        collect_options: Per-collector option dicts (name -> {knob: value}),
            threaded to ``config['_aorta_collect_options']`` when non-empty.
        sidecar_files: JSON sidecar files describing ad-hoc mitigations
            and/or environments (B3.1).  Forwarded to
            ``aorta.registry.get_mitigation`` /
            ``aorta.registry.get_environment`` so that names declared in
            the sidecar resolve in the same call as built-ins and
            entry-point plugins.
        dataset_index: Cell coordinate on the dataset / environment axis,
            used in ``trial_id`` and the per-trial JSON filename.
            ``aorta run`` is one cell, so the default ``0`` is correct
            for direct CLI use; ``aorta triage`` (B2) calls
            ``run_trials`` once per cell and varies this index across
            its environment axis so cells in the matrix don't collide
            on disk.
        mitigation_index: Cell coordinate on the mitigation axis (same
            rationale as ``dataset_index``).  ``aorta triage`` varies
            this across its mitigation axis; ``aorta run`` always emits
            ``m0``.
        save_logs: When ``True``, capture the workload's in-process
            ``stdout`` / ``stderr`` writes to per-trial files alongside
            the trial JSON (``trial_d{d}_m{m}_t{t}.{stdout,stderr}.log``).
            Default ``False`` preserves today's behaviour (no capture).
            Both file capture and the reserved-key injection described
            below are **rank-0 only** -- matches the trial-JSON write
            guarantee. Wrappers running on non-rank-0 won't see the
            keys and should treat capture as off there.
            ``contextlib.redirect_*`` only catches Python-level writes;
            subprocesses are not captured. Wrappers that own a
            subprocess can opt in by reading the platform-supplied
            ``_aorta_save_logs`` / ``_aorta_log_prefix`` config keys
            this dispatcher injects; the prefix is an absolute
            path-with-stem rooted in the per-workload results
            subdirectory (e.g. ``<results_dir>/<workload>/trial_d0_m0_t0``,
            anchored via ``Path.absolute()`` so a relative
            ``RunRequest.results_dir`` still yields a usable prefix)
            and the wrapper derives a non-colliding sibling path such as
            ``<prefix>.subprocess.{stdout,stderr}.log`` -- the
            dispatcher already holds open the
            ``<prefix>.{stdout,stderr}.log`` paths and double-writing
            them would race.

            The ``redirect_*`` rebinding is process-wide. Today no
            caller invokes ``run_trials`` from multiple threads
            concurrently (``aorta run`` is one cell, ``aorta triage``
            iterates cells serially), so cross-thread crosstalk is
            theoretical. If that ever changes, this knob would need a
            different capture mechanism -- this mirrors the same
            single-caller assumption the env-restore block on this
            function relies on.
        subprocess_argv: Opaque ``argv`` forwarded byte-for-byte to a
            subprocess-shaped workload. The dispatcher injects this
            tuple into the workload config as the reserved
            ``_aorta_subprocess_argv`` key after ``config_overrides``
            is merged, so user-supplied ``config_overrides`` cannot
            collide (the reserved-prefix rejection at the top of
            ``run_trials`` enforces this). Only consumed today by
            :class:`aorta.workloads._subprocess.SubprocessWorkload`,
            which ``aorta probe`` wires up; other workloads ignore
            the key. ``None`` (the default) leaves the key unset so
            existing single-process workloads round-trip exactly as
            before. Carrying ``argv`` here -- rather than letting it
            leak into ``config_overrides`` -- preserves the "no
            user-supplied ``_aorta_*`` keys" invariant and makes the
            data-flow visible in the dataclass surface.
        probe_extras: Opaque probe-mode metadata bundle injected into
            the workload config as the reserved ``_aorta_probe_extras``
            key, post-``config_overrides`` merge. Consumed only by
            :class:`aorta.workloads._subprocess.SubprocessWorkload`
            (Phase 1) to know its cell name, the requested
            ``env_passthrough_mode``, the per-trial timeout, and the
            resolved cell env-var bundle. ``None`` (default) leaves
            the key unset so non-probe workloads see no change.
    """

    workload: str
    trials: int
    environment: str = "local"
    # Runtime override for the resolved :class:`Environment`'s
    # ``docker`` field. Symmetric peer of ``buck_target`` below
    # (both overlay one axis of the named environment's recipe at
    # run time, both preserve the other axes). The NAME asymmetry
    # ``image`` (here) vs ``docker`` (the Environment field) is
    # intentional: the FIELD names the recipe slot (peer of ``venv``
    # / ``buck_target``); the OVERLAY VALUE names what the operator
    # provides -- an OCI image reference (typically a digest pin
    # like ``sha256:<64-hex>`` or ``<repo>@sha256:<digest>``). Same
    # naming used by the CLI flag (``--image``) and by downstream
    # regression-gate dispatchers (which emit ``--image <digest>``
    # for DOCKER_ONLY and BUCK_IN_DOCKER tiers).
    # ``None`` means "leave the resolved environment's ``docker``
    # untouched" -- a named env that already declares ``docker``
    # keeps its value.
    #
    # ``kw_only=True`` is the backward-compat guard: same rationale
    # as ``buck_target`` below -- declaring this BEFORE
    # ``mitigations`` in the source (to keep the docstring
    # "Attributes:" grouping of env-tier overlays together) would
    # otherwise shift ``mitigations``'s positional slot and break
    # external positional callers. With kw_only, Python places this
    # field last in ``__init__``'s signature regardless of class-
    # body order. (Caught at PR #193 review; pinned by
    # ``tests/run/test_dispatcher.py::TestImageIsKeywordOnly``.)
    image: str | None = field(default=None, kw_only=True)
    # Runtime override for the resolved :class:`Environment`'s
    # ``buck_target`` field (#182 made it a first-class peer of
    # ``docker`` / ``venv``). When set, takes effect AFTER
    # :func:`get_environment` resolves ``environment``, so the named
    # environment's other fields (``docker``, ``venv``,
    # ``source_package``) are preserved. ``None`` means "leave the
    # resolved environment's ``buck_target`` untouched" -- a named env
    # that already declares ``buck_target`` keeps its value. This is
    # the symmetric peer of how ``aorta env probe --buck-target``
    # enriches the env snapshot; here it overlays the runtime recipe
    # the workload's ``run()`` reads via
    # ``config["_aorta_environment"]["buck_target"]``. Enables the
    # BUCK_ONLY / BUCK_IN_DOCKER tiers of downstream regression-gate
    # dispatchers without forcing operators to register a one-shot
    # named environment per gate.
    #
    # ``kw_only=True`` is the backward-compat guard: declaring this
    # field BEFORE ``mitigations`` (so the docstring "Attributes:"
    # order matches the conceptual "env-tier overlay then mitigations"
    # grouping) would otherwise shift ``mitigations`` from the 4th
    # positional slot to the 5th, silently breaking any external
    # caller that constructed a ``RunRequest`` positionally. With
    # ``kw_only=True``, Python places ``buck_target`` last in
    # ``__init__``'s signature regardless of class-body order, so
    # positional callers continue to receive ``mitigations`` at slot
    # 4. (Caught at review on the symmetric --image PR; applied here
    # too for symmetry and to address the same principle at its
    # root.)
    buck_target: str | None = field(default=None, kw_only=True)
    mitigations: tuple[str, ...] = ("none",)
    extra_env: dict[str, str] = field(default_factory=dict)
    trial_isolation: Literal["auto", "in_process", "process"] = field(
        default="auto",
        kw_only=True,
    )
    cell_name: str | None = field(default=None, kw_only=True)
    request_fingerprint: str | None = field(default=None, kw_only=True)
    steps: int | None = None
    config_overrides: dict[str, Any] = field(default_factory=dict)
    results_dir: Path = field(default_factory=lambda: Path("results"))
    collect: tuple[str, ...] = field(default_factory=tuple)
    # Per-collector options keyed by collector name -> dict[str, str] of
    # knobs (e.g. {"layer_numerics": {"NANLOG_SAMPLE_EVERY": "1"}}). Threaded
    # into config["_aorta_collect_options"] when non-empty so a workload (or a
    # shared collector helper) can apply them. Empty (default) = no options,
    # key absent from config -- back-compat with every existing run.
    collect_options: dict[str, dict[str, str]] = field(default_factory=dict)
    sidecar_files: tuple[Path, ...] = field(default_factory=tuple)
    dataset_index: int = 0
    mitigation_index: int = 0
    save_logs: bool = False
    subprocess_argv: tuple[str, ...] | None = None
    probe_extras: dict[str, Any] | None = None
    # Issue #232: collect-until-N stopping rule. When set (a
    # :class:`aorta.triage.recipe.StopAfter`), ``trials`` is the hard cap
    # (``max_trials``) and the loop breaks early once ``events``
    # qualifying verdicts are observed. Typed as ``Any`` to keep this
    # module's import graph cycle-free (same rationale as
    # ``Recipe.probe_extras``); the loop only duck-types ``.events`` /
    # ``.event_verdict``. ``kw_only`` so positional callers are unaffected.
    stop_after: Any | None = field(default=None, kw_only=True)
    # In-container env-probe contract for isolated (docker/venv) envs.
    # The triage runner sets this to ``{"src": <host aorta src>, "out":
    # <environments/<env>/env.json>}`` so a self-isolating wrapper can
    # bind-mount the aorta source and drop a real snapshot the runner
    # then promotes (else it falls back to a placeholder). ``None`` (the
    # default) means "no in-container probe requested" -- every
    # pre-existing caller and every non-isolated env round-trips with the
    # ``_aorta_env_probe`` key absent. ``kw_only`` so positional callers
    # are unaffected, matching ``stop_after`` above.
    env_probe: dict[str, str] | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        # Defensively deep-copy mutable dict fields.  ``frozen=True``
        # blocks attribute reassignment, so we use
        # ``object.__setattr__`` to install the copies.
        for field_name in ("extra_env", "config_overrides", "collect_options"):
            object.__setattr__(self, field_name, copy.deepcopy(getattr(self, field_name)))
        # ``probe_extras`` is the same pattern as the dict fields above --
        # frozen blocks attribute reassignment but does not
        # stop the caller from mutating the nested dict. Deep-copy on
        # construction so an in-flight request can never be mutated
        # out from under the dispatcher. ``None`` short-circuits.
        if self.probe_extras is not None:
            object.__setattr__(self, "probe_extras", copy.deepcopy(self.probe_extras))
        # ``env_probe`` is the same optional-dict pattern as ``probe_extras``:
        # deep-copy so a caller cannot mutate the {src, out} paths after
        # construction and steer an in-flight probe. ``None`` short-circuits.
        if self.env_probe is not None:
            object.__setattr__(self, "env_probe", copy.deepcopy(self.env_probe))


def run_trials(request: RunRequest) -> list[TrialResult]:
    """Run N trials for a single (workload, environment, mitigation-set) combination.

    This is the main entry point for the workload runner. It handles:
    - Workload discovery and instantiation
    - Launch mode validation
    - Environment and mitigation configuration
    - Trial execution with error handling
    - JSON result persistence (rank 0 only)

    Args:
        request: Configuration for the run.

    Returns:
        List of TrialResult objects, one per trial.

    Raises:
        ValueError: If ``trials`` is not positive, an unknown collector
            recipe is requested, or the workload is not found.
        UnknownEnvironmentError / UnknownMitigationError: If the
            requested environment or mitigation is not in the registry
            (both subclass ``KeyError`` -- callers can also catch
            ``LookupError`` to handle either).
        RuntimeError: If launch-mode validation fails.
    """
    # 1. Validate trial count.  ``trials <= 0`` would silently no-op,
    #    which is almost never what either the CLI or a library caller
    #    intended.
    if request.trials < 1:
        raise ValueError(f"trials must be >= 1 (got {request.trials})")
    if request.request_fingerprint is not None and re.fullmatch(
        r"[0-9a-f]{64}",
        request.request_fingerprint,
    ) is None:
        raise ValueError(
            "request_fingerprint must be a lowercase SHA-256 hex string"
        )

    # 2. Validate collector recipe names.  The CLI also validates this
    #    against KNOWN_RECIPES, but ``run_trials`` is a public library
    #    API consumed by B2 (triage matrix runner) -- programmatic
    #    callers deserve the same protection.
    invalid_collectors = set(request.collect) - KNOWN_RECIPES
    if invalid_collectors:
        raise ValueError(
            f"Unknown collector recipes: {sorted(invalid_collectors)}. "
            f"Valid: {sorted(KNOWN_RECIPES)}"
        )
    if not isinstance(request.collect_options, dict):
        raise ValueError(
            "collect_options must be a mapping of collector name -> dict[str, str]"
        )
    stale_collect_options = sorted(set(request.collect_options) - set(request.collect))
    if stale_collect_options:
        raise ValueError(
            "collect_options provided for collectors not enabled in collect: "
            f"{stale_collect_options}"
        )
    bad_collect_options = [
        name
        for name, opts in request.collect_options.items()
        if not isinstance(opts, dict)
        or not all(isinstance(k, str) and isinstance(v, str) for k, v in opts.items())
    ]
    if bad_collect_options:
        raise ValueError(
            "collect_options entries must be dict[str, str]; invalid collectors: "
            f"{bad_collect_options}"
        )
    # Per-collector option schemas + cross-collector conflicts. The recipe
    # loader already applied these, but ``run_trials`` is a public library API:
    # a programmatic caller deserves to learn its rocprof/proton pairing cannot
    # run before the first trial launches, not from an empty artifact dir.
    validate_collectors(request.collect, request.collect_options)

    # 3. Validate ``extra_env``. The CLI and recipe loader validate their
    #    inputs, but library callers can construct ``RunRequest`` directly.
    #    Fail before mutating ``os.environ`` and never include values in errors.
    _validate_env_mapping("extra_env", request.extra_env)

    # Validate ``env_probe`` shape early.  The triage runner is the only
    # in-tree producer and always passes ``{"src": str, "out": str}``, but
    # ``RunRequest`` is a public library API -- a programmatic caller passing
    # a bad shape (missing keys, Path values) would otherwise only fail much
    # later when the value is injected into ``TrialResult.config`` and
    # JSON-serialized, with a confusing ``TypeError``.  Fail fast with an
    # actionable message instead.
    if request.env_probe is not None:
        if (
            not isinstance(request.env_probe, dict)
            or set(request.env_probe) != {"src", "out"}
            or not all(isinstance(v, str) for v in request.env_probe.values())
        ):
            raise ValueError(
                "env_probe must be a dict with exactly {'src', 'out'} string "
                f"values; got {request.env_probe!r}."
            )

    # 4. Reject reserved ``_aorta_*`` keys in ``config_overrides``.
    #    The dispatcher writes platform-supplied values (currently
    #    ``_aorta_environment``) into ``config`` after merging
    #    ``config_overrides``, so a caller-supplied ``_aorta_*`` key
    #    would be silently clobbered.  Failing loudly here surfaces
    #    typos and prevents callers from depending on a slot that
    #    isn't actually theirs.
    reserved_keys = sorted(k for k in request.config_overrides if k.startswith("_aorta_"))
    if reserved_keys:
        raise ValueError(
            f"config_overrides keys {reserved_keys} use the reserved "
            "'_aorta_' prefix (platform-supplied; not a user override)."
        )

    # 5. Resolve isolation from entry-point metadata without importing workload
    # code. This is load-bearing for process mode: the controlled overlay must
    # be in the worker environment before plugin/native-library imports.
    policy = get_workload_policy(request.workload)
    effective_trial_isolation: TrialIsolation = resolve_trial_isolation_policy(
        request.workload,
        policy,
        request.trial_isolation,
    )

    # 6. In-process execution still discovers and validates the class here.
    # Process execution defers all implementation imports to the configured
    # worker, which repeats launch/class-metadata validation after startup.
    workload_cls: type[Workload] | None = None
    if effective_trial_isolation == "in_process":
        workload_cls = get_workload_class(request.workload)
        validate_launch_mode(workload_cls)
        class_isolation = resolve_trial_isolation(
            workload_cls,
            request.trial_isolation,
        )
        if class_isolation != effective_trial_isolation:
            raise ValueError(
                f"Workload {request.workload!r} isolation policy resolves to "
                f"{effective_trial_isolation!r}, but its class metadata resolves "
                f"to {class_isolation!r}; register matching metadata in the "
                "'aorta.workload_policies' entry-point group"
            )
    request = replace(request, trial_isolation=effective_trial_isolation)

    # 7. Resolve environment.  Forward ``sidecar_files`` so any
    #    operator-supplied JSON sidecars (B3.1) are merged with
    #    built-ins and entry-point plugins.
    sidecar_files = list(request.sidecar_files) or None
    env_descriptor = get_environment(request.environment, extra_files=sidecar_files)

    # 7-env. Validate the resolved environment's baseline ``env`` mapping.
    #     ``Environment.env`` is the lowest layer of the platform env
    #     contract and is applied to ``os.environ`` before the workload runs
    #     (see ``_run_single_trial``). The registry loaders already validate
    #     name shape + NUL for env declared through them; this re-check is
    #     defense-in-depth for ``Environment`` objects constructed
    #     programmatically (which bypass the loaders), so a malformed key or
    #     value never reaches ``os.environ.update`` with the opaque
    #     ``ValueError: illegal environment variable name``. Same rule set as
    #     the ``extra_env`` check above.
    _validate_env_mapping("Environment.env", env_descriptor.env)

    # 7a. Apply the per-axis runtime overrides (if any) AFTER
    #     resolving the named environment, so the named env's other
    #     fields (``venv`` / ``source_package`` / the axes not being
    #     overridden) are preserved.  Each override is independent:
    #     a BUCK_IN_DOCKER gate pins BOTH ``image`` and
    #     ``buck_target`` and expects them BOTH to flow through.
    #
    #     Falsy values (``None`` -- the default -- and ``""``) mean
    #     "no override": a named env that already declares the
    #     field keeps its value.  Empty string is never a valid
    #     value for either flag (no Buck2 label is empty; an OCI
    #     image reference of ``""`` is not a reference), so treating
    #     it as a no-op rather than silently overlaying ``""`` onto
    #     the resolved env avoids a downstream ``buck2 run ""`` /
    #     ``docker run ""``-style failure that's hard to attribute
    #     back to the flag.  This makes the new flags backward-
    #     compat with every pre-existing run.
    #
    #     ``image`` overlays the ``docker`` field of
    #     :class:`Environment` (the recipe slot's name -- ``image``
    #     names the value the operator provides). ``buck_target``
    #     overlays the like-named field. See the ``RunRequest``
    #     docstring for the cross-repo motivation (downstream
    #     regression-gate dispatchers).
    if request.image:
        env_descriptor = replace(env_descriptor, docker=request.image)
    if request.buck_target:
        env_descriptor = replace(env_descriptor, buck_target=request.buck_target)

    # 8. Resolve and union mitigations.  ``aorta.registry.get_mitigation``
    #    returns a defensive ``dict[str, str]`` per-call, so later
    #    mitigations naturally win over earlier ones in the union.
    mitigation_env: dict[str, str] = {}
    for name in request.mitigations:
        mitigation_env.update(get_mitigation(name, extra_files=sidecar_files))
    _validate_env_mapping("mitigation environment", mitigation_env)

    effective_overlay: dict[str, str] = {}
    effective_overlay.update(env_descriptor.env)
    effective_overlay.update(mitigation_env)
    effective_overlay.update(request.extra_env)
    startup_env: dict[str, str] = {}
    if effective_trial_isolation == "process":
        startup_config = dict(request.config_overrides)
        if request.steps is not None:
            startup_config["steps"] = request.steps
        startup_env = get_workload_startup_env(
            request.workload,
            startup_config,
        )
        _validate_env_mapping("workload startup environment", startup_env)
        unsafe_identity = sorted(
            key
            for key in set(effective_overlay) | set(startup_env)
            if (
                (key.upper() if os.name == "nt" else key)
                in _LAUNCHER_IDENTITY_KEYS
                or (key.upper() if os.name == "nt" else key).startswith(
                    "TORCHELASTIC_"
                )
            )
        )
        if unsafe_identity:
            raise ValueError(
                "process-isolated trials cannot override launcher identity "
                f"variables {unsafe_identity}; set them in torchrun/srun instead"
            )
        if request.probe_extras is not None or request.subprocess_argv is not None:
            raise ValueError(
                "process trial isolation is not supported for probe/subprocess workloads"
            )

    # 9. Determine if we should write (rank 0 only for distributed).
    #    Only rank 0 needs the output directory; creating it on every
    #    rank causes shared-FS contention and weakens the rank-0-only
    #    write guarantee.  Parse RANK defensively -- a misconfigured
    #    launcher passing a non-integer should not crash the run.
    raw_rank = os.environ.get("RANK", "0")
    try:
        rank = int(raw_rank)
    except ValueError:
        logger.warning(
            "Ignoring non-integer RANK=%r; treating this process as rank 0.",
            raw_rank,
        )
        rank = 0
    should_write = rank == 0
    # The operator owns ``--results-dir``; the workload name below it is
    # payload-controlled state left by earlier attempts. Canonicalize only the
    # operator boundary, then append the workload lexically so a pre-existing
    # ``<results>/<workload> -> /outside`` link is visible to the collector
    # guard instead of being folded into its trusted anchor.
    canonical_results_dir = (
        request.results_dir.resolve() if request.collect and should_write else None
    )
    results_dir = (
        canonical_results_dir if canonical_results_dir is not None else request.results_dir
    ) / request.workload
    if should_write:
        results_dir.mkdir(parents=True, exist_ok=True)
    # Freeze the collector trust anchor here, once, before the FIRST trial
    # runs -- not per trial. Trial 0's payload can replace ``results_dir``
    # itself with a symlink; a per-trial ``resolve()`` would then hand trial 1
    # a brand-new anchor pointing outside the operator's tree, and
    # ``_reset_output_dir`` would happily clear a planted trial directory
    # there. Every trial receives this same unchanged path.
    #
    # The freeze records the directory's inode as well as its path, which is
    # the part a pathname cannot carry: a payload can rename the results
    # directory aside and move a *real* directory into its place, leaving every
    # later ``O_NOFOLLOW`` check on that pathname satisfied. Taken after the
    # ``mkdir`` above so the directory exists to be pinned.
    collector_results_root = (
        _fsafe.TrustedAnchor.freeze(canonical_results_dir)
        if canonical_results_dir is not None
        else None
    )
    isolation_generation = (
        _next_isolation_generation()
        if effective_trial_isolation == "process"
        else None
    )

    # 10. Run trials
    # Gate progress logs on rank 0 -- the same predicate that gates JSON
    # writes -- so a torchrun-launched workload doesn't emit duplicate
    # "trial K/N starting" lines from every rank under -v. Non-rank-0
    # processes still execute the trial; they just don't narrate it.
    if should_write:
        logger.info(
            "run_trials: workload=%s environment=%s mitigations=%s trials=%d steps=%s",
            request.workload,
            request.environment,
            list(request.mitigations) or ["(none)"],
            request.trials,
            request.steps if request.steps is not None else "(workload default)",
        )
    results: list[TrialResult] = []
    # Issue #232: ``request.trials`` is the hard cap (``max_trials`` when a
    # stop_after rule is attached). ``events_seen`` tracks qualifying
    # verdicts so the loop can break early once the target is met.
    stop_after = request.stop_after
    events_seen = 0
    for trial_idx in range(request.trials):
        if should_write:
            logger.info("trial %d/%d: starting", trial_idx + 1, request.trials)
        trial_t0 = time.perf_counter()
        if effective_trial_isolation == "process":
            assert isolation_generation is not None
            try:
                result = _run_single_trial_in_process_worker(
                    trial_idx=trial_idx,
                    request=request,
                    env_descriptor=env_descriptor,
                    mitigation_env=mitigation_env,
                    effective_overlay=effective_overlay,
                    startup_env=startup_env,
                    isolation_generation=isolation_generation,
                    results_dir=results_dir,
                    results_root=collector_results_root,
                    should_write=should_write,
                )
            except TrialWorkerError as exc:
                exc.completed_results = tuple(results)
                raise
        else:
            assert workload_cls is not None
            result = _run_single_trial(
                trial_idx=trial_idx,
                workload_cls=workload_cls,
                request=request,
                env_descriptor=env_descriptor,
                mitigation_env=mitigation_env,
                results_dir=results_dir,
                results_root=collector_results_root,
                should_write=should_write,
            )
        if should_write:
            # ``TrialResult.result`` is the WorkloadResult-as-dict; .get() so
            # workloads that omit ``passed`` still classify cleanly.
            passed = bool(result.result.get("passed"))
            logger.info(
                "trial %d/%d: %s in %.1fs (exit_status=%s)",
                trial_idx + 1,
                request.trials,
                "passed" if passed else "FAILED",
                time.perf_counter() - trial_t0,
                result.exit_status,
            )
        results.append(result)
        if stop_after is not None and _trial_is_event(result, stop_after.event_verdict):
            events_seen += 1
            if events_seen >= stop_after.events:
                if should_write:
                    # "early" only when trials remain in the budget; hitting
                    # the target on the final allowed trial is a cap reach,
                    # not an early stop -- don't mislead operator logs.
                    stopped_early = len(results) < request.trials
                    logger.info(
                        "stop_after: %d %r event(s) observed in %d trial(s) "
                        "(target %d, cap %d) -- %s",
                        events_seen,
                        stop_after.event_verdict,
                        len(results),
                        stop_after.events,
                        request.trials,
                        "stopping cell early" if stopped_early else "cap reached",
                    )
                break

    return results


def _run_single_trial_in_process_worker(
    *,
    trial_idx: int,
    request: RunRequest,
    env_descriptor: Environment,
    mitigation_env: dict[str, str],
    effective_overlay: dict[str, str],
    startup_env: dict[str, str],
    isolation_generation: int,
    results_dir: Path,
    results_root: _fsafe.TrustedAnchor | None,
    should_write: bool,
) -> TrialResult:
    trial_id = (
        f"{request.workload}_d{request.dataset_index}_"
        f"m{request.mitigation_index}_t{trial_idx}"
    )
    _validate_json_native(request.config_overrides)
    run_request = {
        "workload": request.workload,
        "environment": request.environment,
        "image": request.image,
        "buck_target": request.buck_target,
        "mitigations": list(request.mitigations),
        "extra_env": dict(request.extra_env),
        "trial_isolation": "process",
        "cell_name": request.cell_name,
        "request_fingerprint": request.request_fingerprint,
        "steps": request.steps,
        "config_overrides": copy.deepcopy(request.config_overrides),
        "results_dir": str(request.results_dir),
        "collect": list(request.collect),
        "collect_options": copy.deepcopy(request.collect_options),
        "sidecar_files": [str(path) for path in request.sidecar_files],
        "dataset_index": request.dataset_index,
        "mitigation_index": request.mitigation_index,
        "save_logs": request.save_logs,
        "env_probe": copy.deepcopy(request.env_probe),
    }
    # Fail before spawning with an actionable error rather than letting the
    # worker fail while serializing its request.
    try:
        json.dumps(run_request)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "process-isolated workload config must be JSON-compatible"
        ) from exc

    child_env = dict(os.environ)
    for key, value in startup_env.items():
        child_env.setdefault(key, value)
    child_env.update(effective_overlay)
    try:
        world_size = int(child_env.get("WORLD_SIZE", "1"))
    except ValueError as exc:
        raise ValueError(
            f"WORLD_SIZE must be an integer, got {child_env.get('WORLD_SIZE')!r}"
        ) from exc
    if (
        world_size > 1
        and child_env.get("TORCHELASTIC_USE_AGENT_STORE", "").lower()
        != "true"
    ):
        child_env["MASTER_PORT"] = _isolated_fallback_port(
            isolation_generation,
            trial_idx,
        )
    worker_started = time.perf_counter()
    result_dict = launch_trial_worker(
        {
            "trial_idx": trial_idx,
            "run_request": run_request,
            "env_descriptor": asdict(env_descriptor),
            "mitigation_env": dict(mitigation_env),
            "results_dir": str(results_dir),
            # The frozen collector anchor from before the trial loop. Sent
            # explicitly so the worker inherits the parent's canonical path
            # instead of resolving one in a process that starts after an
            # earlier trial's payload has already run. The pinned inode travels
            # as a plain pair because the envelope is JSON and a file
            # descriptor cannot cross the process boundary.
            "results_root": str(results_root.path) if results_root is not None else None,
            "results_root_id": (
                list(results_root.identity)
                if results_root is not None and results_root.identity is not None
                else None
            ),
            "should_write": should_write,
            "store_prefix": (
                f"aorta/{os.environ.get('TORCHELASTIC_RUN_ID', 'static')}/"
                f"{os.environ.get('TORCHELASTIC_RESTART_COUNT', '0')}/"
                f"{isolation_generation}/"
                f"{request.cell_name or 'direct'}/{trial_id}"
            ),
        },
        child_env=child_env,
        trial_id=trial_id,
    )
    result = replace(
        TrialResult.from_dict(result_dict),
        wall_clock_sec=time.perf_counter() - worker_started,
    )
    if should_write:
        _write_trial_result(
            result=result,
            request=request,
            trial_idx=trial_idx,
            results_dir=results_dir,
        )
    return result


def _trial_is_event(result: TrialResult, event_verdict: str) -> bool:
    """Decide whether ``result`` counts as a ``stop_after`` event.

    Uses the shared three-way :func:`aorta.run.results.trial_verdict`
    predicate (issue #230) so the stop-count and the matrix pass / fail /
    error columns can never disagree:

    * ``event_verdict == "fail"`` -> the bug reproduced (genuine failure).
    * ``event_verdict == "pass"`` -> a clean trial.
    * ``event_verdict == "error"`` -> the trial never validly ran (infra
      crash, launch failure, timeout with no recognised hang). Useful to
      bail out of a sweep that's mostly flaking on infrastructure.

    Note this is a behaviour refinement over the pre-#230 predicate
    (``fail`` == any non-``ok`` exit): infra errors no longer count as
    ``fail`` events -- they count as ``error`` events instead.

    Raises:
        ValueError: if ``event_verdict`` is not one of the canonical
            :data:`aorta.triage.recipe._STOP_AFTER_EVENT_VERDICTS`
            values. The recipe loader validates this up front, so a bad
            value here means a programmatic caller bypassed the loader;
            fail loudly rather than silently treat an unknown/typo'd
            verdict as ``"fail"``.
    """
    # Validate against the canonical vocabulary the recipe schema exports
    # rather than re-hardcoding the verdict set here. Imported locally
    # because a module-level ``aorta.triage`` import would invert the
    # layering (aorta.triage.runner imports this function) -- same
    # rationale as ``RunRequest.stop_after`` being typed ``Any``.
    from aorta.triage.recipe import _STOP_AFTER_EVENT_VERDICTS

    if not isinstance(event_verdict, str) or event_verdict not in _STOP_AFTER_EVENT_VERDICTS:
        raise ValueError(
            f"event_verdict must be one of {sorted(_STOP_AFTER_EVENT_VERDICTS)}, "
            f"got {event_verdict!r}"
        )
    return trial_verdict(result) == event_verdict


def _run_single_trial(
    trial_idx: int,
    workload_cls: type[Workload],
    request: RunRequest,
    env_descriptor: Environment,
    mitigation_env: dict[str, str],
    results_dir: Path,
    should_write: bool,
    results_root: _fsafe.TrustedAnchor | None = None,
    persist_result: bool = True,
    result_transform: (
        Callable[[WorkloadResult, str], tuple[WorkloadResult, str]] | None
    ) = None,
    skip_cleanup_on_error: bool = False,
) -> TrialResult:
    """Execute a single trial.

    Args:
        trial_idx: Index of the current trial (0-based).
        workload_cls: The workload class to instantiate.
        request: The run request configuration.
        env_descriptor: Resolved environment descriptor.
        mitigation_env: Environment variables from mitigations.
        results_dir: Directory for JSON output.
        should_write: Whether to write JSON (rank 0 only).
        results_root: Canonical collector trust anchor -- path plus pinned
            inode -- frozen by :func:`run_trials` before the first trial.
            ``None`` for a caller that runs a single trial directly; this
            function then freezes ``results_dir`` itself, which is still
            pre-launch for that one trial. Only a multi-trial loop needs the
            earlier freeze, because the exposure is one trial's payload
            rewriting the tree the next trial would resolve.

    Returns:
        TrialResult with execution outcome.
    """
    # Spec format: ``<workload>_d<dataset>_m<mitigation>_t<trial>`` so
    # ``aorta triage`` (B2) can fan out across the dataset/mitigation
    # axes without per-cell trial files colliding.  ``aorta run`` is
    # one cell, so ``d``/``m`` default to 0 here; B2 sets them per
    # cell when it calls ``run_trials`` directly.
    trial_id = (
        f"{request.workload}_d{request.dataset_index}_m{request.mitigation_index}_t{trial_idx}"
    )
    # ``perf_counter`` is monotonic; ``time.time()`` can jump backward
    # or forward when the system clock is adjusted (NTP, suspend/resume),
    # which would corrupt ``wall_clock_sec``.
    start_time = time.perf_counter()

    # Build config
    config: dict[str, Any] = {**request.config_overrides}
    if request.steps is not None:
        config["steps"] = request.steps

    # Thread the resolved Environment descriptor into the workload's
    # config under a reserved underscore-prefixed key.  Workloads that
    # can isolate themselves (e.g., the recom_repro wrapper invoking
    # ``docker run`` instead of ``python``, or a buck-aware wrapper
    # invoking ``buck2 run <label>``) read this to pick the right
    # image / venv / buck target; workloads that don't ignore the key.
    # Recognized tier hints today: ``docker`` (image digest), ``venv``
    # (path), ``buck_target`` (#182 -- Buck2 target label).  The platform
    # itself launches none of these -- it threads metadata, the wrapper
    # decides.
    #
    # This is the dispatcher's way of telling the workload *which*
    # environment was selected for this cell -- the ``--environment``
    # flag and the ``environment:`` axis in a triage recipe both flow
    # through here.  Triage runs vary this per-cell, so emitting it
    # on every trial keeps cells independently runnable.
    #
    # The underscore prefix signals "platform-supplied; not a user
    # override" and matches the same convention ``TrialResult`` uses
    # for ``execution_env``.  ``run_trials`` rejects ``_aorta_*`` keys
    # in ``config_overrides`` so this assignment can't silently clobber
    # a caller-supplied value.
    config["_aorta_environment"] = asdict(env_descriptor)
    config["_aorta_trial_isolation"] = request.trial_isolation

    # In-container env-probe contract (isolated docker/venv envs only).
    # The triage runner supplies ``{"src": ..., "out": ...}`` so a
    # self-isolating wrapper can bind-mount the aorta source at ``src``
    # and, as the first step inside its ``docker run``, write a real
    # ``env.json`` to the host path ``out`` -- captured INSIDE the
    # container, so ``sys.prefix`` / ROCm / hipBLASLt reflect the image
    # rather than the runner's venv. The runner promotes that file if it
    # appears, else falls back to the placeholder. Same reserved-``_aorta_*``
    # convention + non-None-only injection as the keys above; the platform
    # launches nothing, the wrapper acts. Non-isolated envs and every
    # pre-existing caller leave this None so the key stays absent.
    if request.env_probe is not None:
        config["_aorta_env_probe"] = dict(request.env_probe)

    # Subprocess-shaped workloads (currently SubprocessWorkload, wired
    # by ``aorta probe``) receive their opaque user argv via a typed
    # ``RunRequest.subprocess_argv`` field rather than via
    # ``config_overrides`` -- the ``_aorta_*`` prefix is reserved and
    # the dispatcher rejects user-supplied keys carrying it.  Inject
    # AFTER the ``config_overrides`` spread so the same reserved-key
    # rejection at the top of ``run_trials`` continues to guard the
    # slot from accidental smuggling, then convert to a list so the
    # JSON-serialised ``TrialResult.config`` is round-trippable
    # (tuples are not JSON types).  When ``subprocess_argv`` is None
    # (every existing caller pre-#188), the key stays absent and
    # ``SubprocessWorkload.setup()`` raises a clear error if it ends
    # up running without one.
    if request.subprocess_argv is not None:
        config["_aorta_subprocess_argv"] = list(request.subprocess_argv)

    # ``probe_extras`` follows the same pattern: a typed RunRequest
    # field is the only legal channel; the dispatcher copies it into
    # the reserved ``_aorta_probe_extras`` slot post-merge.
    # ``SubprocessWorkload`` reads cell name / env-passthrough mode /
    # timeout / cell-env-bundle from this dict; non-probe workloads
    # ignore it.
    if request.probe_extras is not None:
        config["_aorta_probe_extras"] = dict(request.probe_extras)

    # Thread the validated collector recipe names into the workload config
    # under a reserved key so a workload can decide whether to attach a
    # collector (e.g. the recom_repro wrapper rewrites its launch argv to
    # run the entry through the layer_numerics NaN logger when
    # "layer_numerics" is present). Same reserved-``_aorta_*`` convention
    # as ``_aorta_environment`` above; ``run_trials`` rejects user-supplied
    # ``_aorta_*`` keys in ``config_overrides`` so this can't clobber a
    # caller value. Injected only when non-empty so existing trials (no
    # ``--collect``) round-trip with the key absent -- back-compat. Stored
    # as a plain ``list[str]`` (JSON-safe; the trial-JSON dump needs no
    # sanitizer for it, unlike ``_aorta_probe_extras``). The platform
    # itself launches no collector for subprocess workloads -- it threads
    # the names, the wrapper acts on them.
    if request.collect:
        config["_aorta_collect"] = list(request.collect)

    # Per-collector options (the mapping form of ``collect:``). Same reserved
    # ``_aorta_*`` convention + non-empty-only injection as ``_aorta_collect``
    # above. A plain nested ``dict[str, dict[str, str]]`` is JSON-safe, so the
    # trial-JSON dump needs no sanitizer. A workload (or a shared collector
    # helper) reads its own collector's options and applies them.
    if request.collect_options:
        config["_aorta_collect_options"] = {
            name: dict(opts) for name, opts in request.collect_options.items()
        }

    # When a collector is active, thread the per-trial output path stem so a
    # workload can write collector artifacts (e.g. the layer_numerics NaN
    # logger's summary/jsonl) into the results tree -- WITHOUT requiring
    # ``save_logs``. Previously a subprocess wrapper had to piggy-back on
    # ``_aorta_log_prefix``, which the dispatcher only sets when
    # ``save_logs=True``; that coupled an unrelated debug knob to collector
    # output landing in the right place (and being picked up by ``aorta
    # bundle``). This is an absolute path stem with no extension: the
    # operator-owned ``--results-dir`` prefix is canonical while the
    # payload-owned workload/trial components stay lexical so the symlink
    # guard can inspect them. Only set on rank 0 (matches the trial-JSON /
    # log-capture write gate) and only when a collector was requested, so
    # non-collector runs are unchanged.
    if request.collect and should_write:
        trial_basename = (
            f"trial_d{request.dataset_index}_m{request.mitigation_index}_t{trial_idx}"
        )
        # The trust anchor for the collector symlink guards. Taken from the
        # caller when it froze one before the trial loop -- resolving it here
        # would be *after* an earlier trial's payload ran, and that payload can
        # replace ``results_dir`` with a symlink. The caller resolved only the
        # operator-owned ``--results-dir`` boundary, folding legitimate links
        # above it (such as a mounted scratch path); ``collect_base`` preserves
        # the workload component lexically below that prefix so a stale
        # workload symlink is rejected rather than trusted.
        if results_root is not None:
            trusted_root = results_root
            collect_base = results_dir
        else:
            # A direct single-trial caller has no separately declared
            # operator boundary; retain the historical per-workload anchor.
            trusted_root = _fsafe.TrustedAnchor.freeze(results_dir.resolve())
            collect_base = trusted_root.path
        config["_aorta_collect_dir"] = str(collect_base / trial_basename)
        config["_aorta_results_root"] = str(trusted_root.path)
        if trusted_root.identity is not None:
            config["_aorta_results_root_id"] = list(trusted_root.identity)

    # Compute the effective controlled overlay in the platform env-precedence
    # order (lowest to highest):
    #
    #     Environment.env  <  mitigations  <  request.extra_env
    #
    # ``request.extra_env`` already carries the recipe-level + cell-level (or
    # direct-CLI ``--extra-env``) merge -- the runner unions those before
    # constructing the request, and direct ``aorta run --extra-env`` lands here
    # too -- so this single mapping is the top layer. The overlay contains ONLY
    # variables contributed by these three controlled sources; it is NEVER
    # populated from the ambient ``os.environ``. This is the exact bundle
    # threaded to workloads as ``config['_aorta_trial_env']`` (below) so a
    # Docker-aware wrapper can forward precisely these vars into its container.
    effective_overlay: dict[str, str] = {}
    effective_overlay.update(env_descriptor.env)
    effective_overlay.update(mitigation_env)
    effective_overlay.update(request.extra_env)

    # Thread the controlled overlay into the workload config under a reserved
    # key so a self-isolating wrapper (docker/venv/buck) can forward exactly
    # these vars -- and only these -- across the isolation boundary via the
    # shared ``aorta.run.docker_env_flags`` helper. Injected AFTER the
    # ``config_overrides`` spread (like the other ``_aorta_*`` keys) so the
    # reserved-key rejection at the top of ``run_trials`` guards the slot.
    # Always a plain ``dict[str, str]`` (JSON-safe; recorded verbatim in the
    # trial-JSON ``config``). Empty when no controlled source contributed --
    # back-compat with every existing run. Values are deliberately NOT logged.
    config["_aorta_trial_env"] = dict(effective_overlay)

    # Snapshot the env BEFORE applying the overlay so the ``finally`` block can
    # restore both the dispatcher's overlay and any workload-side mutations
    # introduced by ``setup()`` / ``run()``.
    pre_trial_env = dict(os.environ)

    # Apply the effective overlay BEFORE collecting ``env_snapshot`` below.
    # Unlike ``pre_trial_env`` above (the pre-overlay restore point), that
    # snapshot is supposed to describe the actual environment the workload ran
    # under -- including the environment's baseline vars, operator overrides
    # like ``HSA_XNACK=1`` from a mitigation, and one-off ``DISABLE_TF32=1``
    # from ``--extra-env``. Capturing it pre-override loses that signal for
    # reproducibility / debugging. A single ``update`` from the pre-merged
    # overlay applies all three layers in their resolved precedence.
    os.environ.update(effective_overlay)

    # Capture environment snapshot AFTER env-var application.
    # ``collect_env`` is fail-soft and never raises (see A1 docs).
    env_snapshot = collect_env()

    # Instantiate and run workload
    exit_status: str = "ok"
    workload_result = WorkloadResult(
        passed=False,
        failure_count=1,
        failure_details=[{"error": "trial interrupted before workload result"}],
    )
    workload: Workload | None = None
    transform_error: BaseException | None = None

    # ``save_logs`` opens per-trial log files and redirects
    # ``sys.stdout`` / ``sys.stderr`` for the duration of
    # ``setup()`` + ``run()`` + ``cleanup()``. The reserved
    # ``_aorta_save_logs`` / ``_aorta_log_prefix`` config keys let
    # subprocess-based wrappers (whose child output ``redirect_stdout``
    # doesn't catch) opt in and write their own capture to a sibling
    # path derived from the prefix -- the dispatcher already holds
    # open the ``<prefix>.{stdout,stderr}.log`` paths so wrappers
    # must NOT write to them directly. The prefix is an absolute
    # path-with-stem so wrappers don't need to know ``results_dir``.
    #
    # ``encoding="utf-8", errors="backslashreplace"`` is deliberate:
    # the platform default encoding is locale-dependent (cp1252 on
    # Windows, ASCII under ``LC_ALL=C``), and a workload printing a
    # non-ASCII glyph would otherwise raise ``UnicodeEncodeError``
    # inside ``print()`` -- which the trial's ``except Exception``
    # would catch and flip the run to ``infrastructure_failed``.
    # Enabling a debug knob must never break an otherwise-healthy
    # trial; ``backslashreplace`` keeps the file lossless-enough for
    # grep without ever raising.
    #
    # The opens happen up-front in their own ``try/except OSError``
    # for two reasons:
    #   1. An opt-in debug knob must not crash the run -- if the disk
    #      is full or the dir lost write permission, we warn and let
    #      the trial proceed without capture.
    #   2. We've already mutated ``os.environ`` above with the
    #      mitigation / extra_env overlay. If an OSError escaped the
    #      ``with log_stack:`` block below, the env-restore ``finally``
    #      inside that block would never run and the mitigation vars
    #      would leak into the caller's process -- corrupting
    #      subsequent triage cells.
    # The ``_aorta_*`` config keys are only injected on success so
    # that wrappers can trust "if you see the keys, capture is on".
    stdout_fh: Any = None
    stderr_fh: Any = None
    if request.save_logs and should_write:
        log_basename = f"trial_d{request.dataset_index}_m{request.mitigation_index}_t{trial_idx}"
        candidate_stdout = results_dir / f"{log_basename}.stdout.log"
        candidate_stderr = results_dir / f"{log_basename}.stderr.log"
        try:
            stdout_fh = open(candidate_stdout, "w", encoding="utf-8", errors="backslashreplace")
            stderr_fh = open(candidate_stderr, "w", encoding="utf-8", errors="backslashreplace")
        except OSError as exc:
            if stdout_fh is not None:
                stdout_fh.close()
                stdout_fh = None
            # Best-effort cleanup so a 0-byte stub doesn't masquerade
            # as the trial's captured output -- if stdout opened but
            # stderr failed, the empty stdout.log is still on disk.
            for path in (candidate_stdout, candidate_stderr):
                try:
                    path.unlink()
                except OSError:
                    pass
            logger.warning(
                "save_logs=True but failed to open log files in %s "
                "(%s: %s); trial '%s' will run without capture.",
                results_dir,
                type(exc).__name__,
                exc,
                trial_id,
            )
        else:
            config["_aorta_save_logs"] = True
            # Absolute path-with-stem: wrappers compose sibling files as
            # f"{prefix}.subprocess.{stdout,stderr}.log" without needing
            # to know ``results_dir``. ``.absolute()`` (not ``.resolve()``)
            # because we only need to anchor relative inputs against cwd
            # -- a default ``RunRequest(results_dir=Path("results"))``
            # would otherwise leak a relative prefix to wrappers whose
            # subprocesses run with a different cwd (docker bind mounts,
            # torchrun-launched workers). ``.resolve()`` would also walk
            # symlinks and touch the filesystem, which is unnecessary
            # here and surprising on Windows.
            config["_aorta_log_prefix"] = str((results_dir / log_basename).absolute())

    with contextlib.ExitStack() as log_stack:
        if stdout_fh is not None and stderr_fh is not None:
            log_stack.callback(stderr_fh.close)
            log_stack.callback(stdout_fh.close)
            log_stack.enter_context(contextlib.redirect_stdout(stdout_fh))
            log_stack.enter_context(contextlib.redirect_stderr(stderr_fh))

        try:
            # Construct positionally to match the documented Workload(config)
            # contract -- third-party plugins are free to name their first
            # parameter something other than ``config``.
            workload = workload_cls(config)
            # setup() is split into its own try so a setup-time exception
            # gets the "workload_setup_failed" bucket instead of being
            # lumped under "infrastructure_failed". The distinction
            # matters: a row of all-setup-failures means the workload
            # never got off the ground (missing dep, broken probe), not
            # that the measurement under test failed -- matrix.md readers
            # need to see those differently. Construction failures and
            # run()-time exceptions still flow to the outer except as
            # infrastructure_failed (unchanged).
            try:
                workload.setup()
            except Exception as e:
                exit_status = "workload_setup_failed"
                workload_result = WorkloadResult(
                    passed=False,
                    failure_count=1,
                    failure_details=[
                        {
                            "error": str(e),
                            "type": type(e).__name__,
                            "phase": "setup",
                        }
                    ],
                    main_work_started=False,
                )
            else:
                workload_result = workload.run()
                if not workload_result.passed:
                    exit_status = "workload_failed"

        except Exception as e:
            exit_status = "infrastructure_failed"
            # Create error WorkloadResult
            workload_result = WorkloadResult(
                passed=False,
                failure_count=1,
                failure_details=[{"error": str(e), "type": type(e).__name__}],
            )

        finally:
            if result_transform is not None and sys.exc_info()[0] is None:
                try:
                    workload_result, exit_status = result_transform(
                        workload_result,
                        exit_status,
                    )
                except BaseException as exc:
                    transform_error = exc
            # Always attempt cleanup if the workload was constructed, even
            # when setup()/run() raised -- otherwise we leak GPU memory,
            # process groups, file handles, etc.  Cleanup failures are not
            # allowed to mask the original exception/exit_status.
            cleanup_is_unsafe = skip_cleanup_on_error and (
                transform_error is not None
                or exit_status
                in {"workload_setup_failed", "infrastructure_failed"}
            )
            if workload is not None and not cleanup_is_unsafe:
                try:
                    workload.cleanup()
                except Exception as cleanup_exc:
                    # Log -- silently swallowing makes leaked GPU memory /
                    # process groups invisible to the operator.  Use
                    # ``exc_info=True`` so the original traceback survives.
                    logger.warning(
                        "workload.cleanup() raised %s during trial '%s'; "
                        "continuing so the original outcome is preserved.",
                        type(cleanup_exc).__name__,
                        trial_id,
                        exc_info=True,
                    )
            # Restore environment by diff against the pre-trial snapshot.
            # We deliberately do NOT use ``os.environ.clear() +
            # os.environ.update(snapshot)`` -- ``run_trials`` is a public
            # library API and ``clear()`` would, for an instant, blank the
            # entire environment for every other thread in the process.
            # The diff approach has no such window: each key transitions
            # at most once, directly to its target value.
            current_keys = set(os.environ)
            saved_keys = set(pre_trial_env)
            for key in current_keys - saved_keys:
                # Added during the trial (mitigation / extra_env / workload
                # setup) -- remove.
                del os.environ[key]
            for key, value in pre_trial_env.items():
                # Restore both the keys we overwrote and any workload-side
                # mutations to pre-existing keys.  ``os.environ.get`` is
                # cheap; this skip avoids a redundant write when the value
                # is already correct.
                if os.environ.get(key) != value:
                    os.environ[key] = value
            if transform_error is not None:
                raise transform_error

    wall_clock = time.perf_counter() - start_time

    # Build execution_env block.  Mirrors the public
    # ``aorta.registry.Environment`` shape (no ``kind`` / ``rocm`` --
    # those were stub-isms; ROCm version now lives inside
    # ``env_snapshot.rocm`` and the runtime kind in
    # ``env_snapshot.runtime_context.type``).  Same shape as
    # ``config["_aorta_environment"]`` above; sharing ``asdict`` keeps
    # the two in lockstep if ``Environment`` ever grows a field.
    execution_env = asdict(env_descriptor)

    # Build TrialResult
    trial_result = TrialResult(
        trial_id=trial_id,
        workload=request.workload,
        execution_env=execution_env,
        mitigations_applied=request.mitigations,
        config=config,
        env=env_snapshot.to_dict(),
        result=asdict(workload_result),
        wall_clock_sec=wall_clock,
        exit_status=exit_status,  # type: ignore[arg-type]
        request_fingerprint=request.request_fingerprint,
    )

    # Write JSON (rank 0 only).  Filename mirrors ``trial_id`` so the
    # cell coordinates (``d`` / ``m`` / ``t``) are visible on disk
    # without parsing the JSON -- B2's matrix collator can slice by
    # axis from the filename alone.
    if should_write and persist_result:
        _write_trial_result(
            result=trial_result,
            request=request,
            trial_idx=trial_idx,
            results_dir=results_dir,
        )

    return trial_result


def _write_trial_result(
    *,
    result: TrialResult,
    request: RunRequest,
    trial_idx: int,
    results_dir: Path,
) -> None:
    output_path = results_dir / (
        f"trial_d{request.dataset_index}_m{request.mitigation_index}_t{trial_idx}.json"
    )
    serialized = result.to_dict()
    _sanitize_probe_extras_for_json(serialized.get("config"))
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serialized, f, indent=2)


def _sanitize_probe_extras_for_json(config: Any) -> None:
    """Replace ``_aorta_probe_extras.custom_patterns`` with a JSON-safe summary.

    Mutates ``config`` in place. ``trial_result.to_dict()`` returns a
    deep copy so mutation is local to the about-to-be-written dict
    and does not affect the live :class:`TrialResult`.

    Non-probe-mode trials (no ``_aorta_probe_extras`` key) and
    probe-mode trials with no ``custom_patterns`` are no-ops.
    Custom-pattern entries that are already JSON-safe (plain dicts,
    e.g. from a future :func:`from_dict` round-trip) pass through
    unchanged.
    """
    if not isinstance(config, dict):
        return
    extras = config.get("_aorta_probe_extras")
    if not isinstance(extras, dict):
        return
    patterns = extras.get("custom_patterns")
    if not patterns:
        return
    summarized: list[dict[str, Any]] = []
    for p in patterns:
        if isinstance(p, dict):
            # Already JSON-safe (round-trip from disk); pass through.
            summarized.append(p)
            continue
        # ``CompiledPattern`` (or any duck-type with the same field
        # names). Surface the public attributes the operator cares
        # about on inspection; skip the compiled regex / CodeType
        # which are runtime-only.
        summarized.append(
            {
                "detector_id": getattr(p, "detector_id", None),
                "regex": getattr(getattr(p, "regex", None), "pattern", None),
                "on_match": getattr(p, "on_match", None),
                "required_for_pass": getattr(p, "required_for_pass", False),
                "condition_source": getattr(p, "condition_source", None),
            }
        )
    extras["custom_patterns"] = summarized


__all__ = ["RunRequest", "run_trials"]
