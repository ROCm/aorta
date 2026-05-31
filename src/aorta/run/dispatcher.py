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
import os
import re
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from aorta.instrumentation.environment import collect_env
from aorta.registry import Environment, get_environment, get_mitigation
from aorta.run.collectors import KNOWN_RECIPES
from aorta.run.discovery import get_workload_class
from aorta.run.results import TrialResult
from aorta.run.validation import validate_launch_mode
from aorta.workloads import Workload, WorkloadResult

logger = logging.getLogger(__name__)

# Conservative POSIX env-var name shape: must start with a letter or
# underscore and contain only [A-Za-z0-9_].  The CLI also enforces this
# at parse time; library callers pass ``extra_env`` directly so we
# re-validate here for parity.
_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


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
        steps: Number of steps per trial (workload-specific).
        config_overrides: Additional workload configuration.
        results_dir: Directory to write per-trial JSON files.
        collect: Collector recipe names (MVP: validated but no-op).
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
    steps: int | None = None
    config_overrides: dict[str, Any] = field(default_factory=dict)
    results_dir: Path = field(default_factory=lambda: Path("results"))
    collect: tuple[str, ...] = field(default_factory=tuple)
    sidecar_files: tuple[Path, ...] = field(default_factory=tuple)
    dataset_index: int = 0
    mitigation_index: int = 0
    save_logs: bool = False
    subprocess_argv: tuple[str, ...] | None = None
    probe_extras: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        # Defensively deep-copy mutable dict fields.  ``frozen=True``
        # blocks attribute reassignment, so we use
        # ``object.__setattr__`` to install the copies.
        for field_name in ("extra_env", "config_overrides"):
            object.__setattr__(self, field_name, copy.deepcopy(getattr(self, field_name)))
        # ``probe_extras`` is the same pattern as the two dict fields
        # above -- frozen blocks attribute reassignment but does not
        # stop the caller from mutating the nested dict. Deep-copy on
        # construction so an in-flight request can never be mutated
        # out from under the dispatcher. ``None`` short-circuits.
        if self.probe_extras is not None:
            object.__setattr__(self, "probe_extras", copy.deepcopy(self.probe_extras))


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

    # 3. Validate ``extra_env`` keys.  The CLI validates this at parse
    #    time, but library callers (B2, future programmatic users) pass
    #    ``extra_env`` directly -- without parity here, a bad key would
    #    only fail mid-trial inside ``os.environ.update`` with the much
    #    less friendly ``ValueError: illegal environment variable name``.
    bad_keys = [k for k in request.extra_env if not _ENV_KEY_RE.match(k)]
    if bad_keys:
        raise ValueError(
            f"Invalid extra_env keys {bad_keys}: each key must match "
            "[A-Za-z_][A-Za-z0-9_]* (POSIX env-var name shape)."
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

    # 5. Discover workload
    workload_cls = get_workload_class(request.workload)

    # 6. Validate launch mode BEFORE setup()
    validate_launch_mode(workload_cls)

    # 7. Resolve environment.  Forward ``sidecar_files`` so any
    #    operator-supplied JSON sidecars (B3.1) are merged with
    #    built-ins and entry-point plugins.
    sidecar_files = list(request.sidecar_files) or None
    env_descriptor = get_environment(request.environment, extra_files=sidecar_files)

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
    results_dir = request.results_dir / request.workload
    if should_write:
        results_dir.mkdir(parents=True, exist_ok=True)

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
    for trial_idx in range(request.trials):
        if should_write:
            logger.info("trial %d/%d: starting", trial_idx + 1, request.trials)
        trial_t0 = time.perf_counter()
        result = _run_single_trial(
            trial_idx=trial_idx,
            workload_cls=workload_cls,
            request=request,
            env_descriptor=env_descriptor,
            mitigation_env=mitigation_env,
            results_dir=results_dir,
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

    return results


def _run_single_trial(
    trial_idx: int,
    workload_cls: type[Workload],
    request: RunRequest,
    env_descriptor: Environment,
    mitigation_env: dict[str, str],
    results_dir: Path,
    should_write: bool,
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

    # Snapshot the env BEFORE applying mitigation / extra_env so the
    # ``finally`` block can restore both the dispatcher's overlay and
    # any workload-side mutations introduced by ``setup()`` / ``run()``.
    pre_trial_env = dict(os.environ)

    # Apply mitigation env + extra_env BEFORE the env snapshot.  The
    # snapshot is supposed to describe the actual environment the
    # workload ran under -- including operator overrides like
    # ``HSA_XNACK=1`` from a mitigation or one-off ``DISABLE_TF32=1``
    # from ``--extra-env``.  Capturing pre-override loses that signal
    # for reproducibility / debugging.
    os.environ.update(mitigation_env)
    os.environ.update(request.extra_env)

    # Capture environment snapshot AFTER env-var application.
    # ``collect_env`` is fail-soft and never raises (see A1 docs).
    env_snapshot = collect_env()

    # Instantiate and run workload
    exit_status: str = "ok"
    workload_result: WorkloadResult
    workload: Workload | None = None

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
            # Always attempt cleanup if the workload was constructed, even
            # when setup()/run() raised -- otherwise we leak GPU memory,
            # process groups, file handles, etc.  Cleanup failures are not
            # allowed to mask the original exception/exit_status.
            if workload is not None:
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
    )

    # Write JSON (rank 0 only).  Filename mirrors ``trial_id`` so the
    # cell coordinates (``d`` / ``m`` / ``t``) are visible on disk
    # without parsing the JSON -- B2's matrix collator can slice by
    # axis from the filename alone.
    if should_write:
        output_path = results_dir / (
            f"trial_d{request.dataset_index}_m{request.mitigation_index}_t{trial_idx}.json"
        )
        serialized = trial_result.to_dict()
        # ``_aorta_probe_extras.custom_patterns`` is a tuple of
        # :class:`aorta.probe.classifier.tier5_custom.CompiledPattern`
        # objects carrying a compiled ``re.Pattern`` and a
        # ``CodeType`` -- neither is JSON-serializable, so leaving
        # the tuple untouched would crash this ``json.dump`` for
        # every probe-mode trial that configured custom patterns
        # (#197 round-7 review). Sanitize down to a JSON-safe
        # summary list (detector id, regex source, on_match,
        # required_for_pass, condition source) so the on-disk
        # ``TrialResult.config`` still tells operators which
        # patterns the workload ran against; the compiled forms
        # were only ever needed by ``SubprocessWorkload.run()``,
        # which has already consumed them by the time we get here.
        _sanitize_probe_extras_for_json(serialized.get("config"))
        # ``encoding="utf-8"`` matches the stdout/stderr opens at L601-602
        # and the rest of the codebase's text-artifact writes -- avoids
        # platform-default-encoding surprises on Windows / containers
        # without a configured locale. Per Copilot's PR #197 review.
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(serialized, f, indent=2)

    return trial_result


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
