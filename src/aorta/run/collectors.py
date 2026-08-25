"""Collector recipe registry.

Collectors are profiling / instrumentation tools attached to a workload run via
``--collect`` or a recipe's ``collect:`` block. This module is the dispatch
layer between the reserved ``_aorta_collect*`` config keys the dispatcher
threads into every trial and the per-collector packages under
:mod:`aorta.instrumentation`.

Supported recipes:
    rocprof: ``rocprofv3`` kernel/API tracing (:mod:`aorta.instrumentation.rocprof`)
    proton: Triton Proton profiler (:mod:`aorta.instrumentation.proton`)
    numerics: Numeric health monitoring (NaN/Inf detection)
    layer_numerics: Per-layer NaN/magnitude logger (:mod:`aorta.instrumentation.layer_numerics`)
    amd_log: AMD internal logging collector

``rocprof`` and ``proton`` attach generically, by wrapping the launch argv --
the same seam :func:`aorta.emulation.mirage_launch.wrap_argv_for_environment`
uses -- so any subprocess-shaped workload, including an opaque ``aorta probe --
<command>``, can be profiled. The remaining names are still validated-only:
they are consumed by workload wrappers that opt in (see the ``layer_numerics``
package docstring).
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

KNOWN_RECIPES: frozenset[str] = frozenset(
    {
        "rocprof",
        "proton",
        "numerics",
        "layer_numerics",
        "amd_log",
    }
)

#: Reserved config keys the dispatcher threads into every trial. Mirrored as
#: literals (not imported from the dispatcher) to keep this module a leaf --
#: :mod:`aorta.triage.recipe` imports it during recipe load.
CONFIG_KEY_COLLECT = "_aorta_collect"
CONFIG_KEY_COLLECT_OPTIONS = "_aorta_collect_options"
CONFIG_KEY_COLLECT_DIR = "_aorta_collect_dir"
#: The dispatcher's ``--results-dir``, canonicalized with :meth:`Path.resolve`
#: **before any trial runs** and threaded so the symlink guards have a trust
#: anchor that does **not** come from the path being validated -- and that is
#: not re-resolved after the payload runs. Every directory at or below this
#: root is payload-writable during the run, including the results directory
#: inode itself, so a boundary derived from a live ``resolve()`` of this path
#: after launch proves nothing.
CONFIG_KEY_RESULTS_ROOT = "_aorta_results_root"

#: Argv-wrapping order, outermost first. ``rocprof`` runs a whole command under
#: the profiler while ``proton`` takes over a Python script's execution, so
#: rocprof must be the outer process for the pair to compose at all. Relative
#: to the emulator: collectors wrap first and the mirage wrap goes outside
#: them, i.e. the profiler runs *inside* the emulated environment.
WRAP_ORDER: tuple[str, ...] = ("rocprof", "proton")


@dataclass(frozen=True)
class CollectorSpec:
    """How one collector validates, attaches, and reports.

    Attributes:
        name: The recipe name (a member of :data:`KNOWN_RECIPES`).
        output_subdir: Subdirectory of the trial's collector directory the
            collector writes artifacts into.
        validate: Validates a recipe's option mapping, raising ``ValueError``
            with an actionable message. Every collector has one; a
            validated-only collector accepts anything a wrapper defines.
        wrap: Rewrites the launch argv to attach the collector, or ``None``
            for a collector the platform does not launch itself.
        summarize: Parses the collector's artifacts into flat trial metrics,
            or ``None`` for a collector the platform does not parse.
    """

    name: str
    output_subdir: str | None
    validate: Callable[[Mapping[str, str] | None], Any]
    wrap: Callable[..., list[str]] | None = None
    summarize: Callable[[Path], dict[str, Any]] | None = None


def _accept_any(options: Mapping[str, str] | None) -> dict[str, str]:
    """Option validator for collectors whose knobs belong to a wrapper.

    ``layer_numerics`` takes ``NANLOG_*`` env knobs interpreted by the workload
    wrapper, not by the platform, so the platform has no schema to check them
    against. The recipe loader has already enforced ``str -> str``.
    """
    return dict(options or {})


def _registry() -> dict[str, CollectorSpec]:
    """Build the collector registry.

    The instrumentation packages are imported here rather than at module scope
    because :mod:`aorta.triage.recipe` imports this module during recipe load;
    a call-time import keeps that path free of import-order coupling. Both
    packages are stdlib-only, so the import is cheap.
    """
    from aorta.instrumentation import proton, rocprof

    return {
        "rocprof": CollectorSpec(
            name="rocprof",
            output_subdir=rocprof.OUTPUT_SUBDIR,
            validate=rocprof.validate_options,
            wrap=rocprof.wrap_argv,
            summarize=rocprof.parse_summary,
        ),
        "proton": CollectorSpec(
            name="proton",
            output_subdir=proton.OUTPUT_SUBDIR,
            validate=proton.validate_options,
            wrap=proton.wrap_argv,
            summarize=proton.parse_summary,
        ),
        "numerics": CollectorSpec("numerics", None, _accept_any),
        "layer_numerics": CollectorSpec("layer_numerics", "layer_numerics", _accept_any),
        "amd_log": CollectorSpec("amd_log", None, _accept_any),
    }


def active_collectors(config: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the requested collector names, in :data:`WRAP_ORDER`.

    Unknown / non-string entries are dropped: ``run_trials`` and the recipe
    loader both reject them up front, so anything left here came from a
    hand-built config and must not crash a trial.
    """
    raw = config.get(CONFIG_KEY_COLLECT)
    if not isinstance(raw, (list, tuple)):
        return ()
    names = [n for n in raw if isinstance(n, str) and n in KNOWN_RECIPES]
    ordered = [n for n in WRAP_ORDER if n in names]
    ordered += [n for n in dict.fromkeys(names) if n not in WRAP_ORDER]
    return tuple(ordered)


def _options_for(config: Mapping[str, Any], name: str) -> dict[str, str]:
    all_options = config.get(CONFIG_KEY_COLLECT_OPTIONS)
    if not isinstance(all_options, dict):
        return {}
    per_collector = all_options.get(name)
    if not isinstance(per_collector, dict):
        return {}
    return {str(k): str(v) for k, v in per_collector.items()}


def _collect_root(config: Mapping[str, Any]) -> Path | None:
    """Resolve the per-trial collector output root, or ``None``.

    The dispatcher only injects :data:`CONFIG_KEY_COLLECT_DIR` on the
    artifact-writing rank, so a non-writing rank has nowhere to put artifacts
    and the collector is skipped rather than scattering files into the cwd.
    """
    raw = config.get(CONFIG_KEY_COLLECT_DIR)
    return Path(raw) if isinstance(raw, str) and raw else None


def _trusted_root(config: Mapping[str, Any], root: Path) -> Path:
    """The directory every collector path must stay inside.

    Prefers the dispatcher-threaded ``--results-dir``
    (:data:`CONFIG_KEY_RESULTS_ROOT`), already canonicalized at dispatch.
    Callers must **not** :meth:`~pathlib.Path.resolve` this value again: the
    profiled process can replace that directory with a symlink, and resolving
    both the candidate and the anchor through the same link would make
    containment succeed for a path that has left the operator's tree.

    Falls back to ``root.parent`` when the key is absent, which only happens for
    a direct programmatic caller -- the dispatcher always supplies it alongside
    ``_aorta_collect_dir``. That fallback still catches a swapped collector root
    or subdirectory; it cannot catch a swapped ancestor.
    """
    raw = config.get(CONFIG_KEY_RESULTS_ROOT)
    if isinstance(raw, str) and raw:
        return Path(raw)
    return root.parent


def unsafe_collector_paths(config: Mapping[str, Any]) -> list[Path]:
    """Every collector path for this trial that cannot be safely traversed.

    Returns the collector root plus each active collector's own output
    subdirectory -- ``<root>/rocprof``, ``<root>/proton`` -- because those are
    the paths the post-run passes actually walk. Guarding only the root is not
    enough: the profiled process can swap a *subdirectory* just as easily, and
    ``parse_summary`` globs the subdirectory while ``apply_retention`` recurses
    into it.

    Empty when there is nothing to guard (no collector active, no threaded
    collector directory) or when every path is a real directory.
    """
    root = _collect_root(config)
    if root is None:
        return []
    registry = _registry()
    trusted = _trusted_root(config, root)
    candidates = [root]
    for name in active_collectors(config):
        spec = registry.get(name)
        if spec is not None and spec.output_subdir is not None:
            candidates.append(root / spec.output_subdir)
    return [path for path in candidates if not collector_root_is_traversable(path, trusted)]


def collector_root_is_traversable(root: Path, trusted_root: Path | None = None) -> bool:
    """True when ``root`` can be walked without following a payload-owned symlink.

    Checked **again after the command has run**, not only before it launches.
    The profiled command is handed this path (``rocprofv3 -d``, ``proton -n``),
    so between the pre-launch reset and any post-run pass it can delete a
    directory and leave a symlink in its place. Every later step --
    ``Path.is_dir()``, ``rglob``, and :func:`aorta.run.retention.apply_retention`
    -- follows links in *every* path component, so traversing one would read,
    and for retention *delete*, through the link.

    Two checks, both required:

    * **Containment.** ``root.resolve()`` must stay inside ``trusted_root``.
      ``trusted_root`` is a pre-launch canonical path and is **not** resolved
      again: resolving both sides after the payload replaced the results
      directory with a symlink would make the escape look contained.
    * **No payload symlink at or below the anchor.** A link whose target is
      still inside the results tree (``trial -> sibling_trial``,
      ``rocprof -> <results>``) also fails: ``rmtree`` / ``rglob`` would then
      operate on the sibling. Operator-owned links *above* the anchor are
      already folded away because the dispatcher stores
      ``--results-dir.resolve()``.

    ``trusted_root`` defaults to ``root.parent`` when the caller omits it.
    That fallback is only for a programmatic caller that did not thread
    :data:`CONFIG_KEY_RESULTS_ROOT`; it still misses a swapped ancestor, which
    is why the dispatcher always supplies the operator's ``--results-dir``.
    """
    trusted = trusted_root if trusted_root is not None else root.parent
    try:
        if not root.resolve().is_relative_to(trusted):
            return False
        return not _payload_symlink_at_or_below(root, trusted)
    except (OSError, RuntimeError):
        # RuntimeError: a symlink loop that ``resolve()`` refuses to follow.
        return False


def _payload_symlink_at_or_below(path: Path, trusted: Path) -> bool:
    """True when any component of ``path`` below ``trusted`` is a symlink.

    ``trusted`` itself is not inspected: that inode is the operator's
    ``--results-dir``. Everything below it is payload-writable.
    """
    current = path
    while current != trusted:
        if current.parent == current:
            # Walked to the filesystem root without meeting the anchor, so
            # ``path`` is not lexically inside the canonical results tree.
            return True
        if current.is_symlink():
            return True
        current = current.parent
    return False


def _reset_output_dir(out_dir: Path, trusted_root: Path) -> None:
    """Create ``out_dir`` empty, discarding any earlier attempt's artifacts.

    Probe resume replays an interrupted trial onto the *same* paths, so a
    retry would otherwise inherit the previous attempt's profile files and
    :func:`summarize_collectors` would report the old run's numbers -- or a
    blend of both, when the retry writes fewer per-rank files than the attempt
    it replaces. Clearing is safe precisely because this directory belongs to
    one trial of one collector: nothing else writes here, and the trial record
    (``result.json``) lives in a different tree.

    Args:
        out_dir: The collector's output directory for this trial.
        trusted_root: Pre-launch canonical results directory. ``out_dir`` must
            stay inside it with no payload-owned symlink on the path.
            ``rmtree`` is recursive and ``is_dir()`` follows links in *every*
            component, so a link at or below this root -- even to a sibling
            inside the tree -- would redirect the delete.

    Raises:
        OSError: the directory could not be cleared or created, or it is not
            traversable inside ``trusted_root``.
    """
    if not collector_root_is_traversable(out_dir, trusted_root):
        raise OSError(
            f"refusing to prepare {out_dir}: a path component at or below "
            f"{trusted_root} is a symlink or resolves outside that directory, "
            "so clearing it would delete through the link. Remove the symlink "
            "in that path, or point --results-dir at a real directory."
        )
    if out_dir.is_symlink() or out_dir.is_file():
        out_dir.unlink()
    elif out_dir.is_dir():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)


def validate_collectors(
    names: Sequence[str],
    options: Mapping[str, Mapping[str, str]] | None = None,
) -> None:
    """Validate a collect request: per-collector options plus cross-collector conflicts.

    Called at recipe-load time (and by any programmatic caller building a run
    request) so a typo or an unrunnable combination fails before a single trial
    starts.

    Args:
        names: The requested collector names. Already checked against
            :data:`KNOWN_RECIPES` by the caller; unknown names are ignored here
            so the caller's own error message stays the one the operator sees.
        options: The per-collector option mapping (the recipe's mapping form).

    Raises:
        ValueError: an option is invalid for its collector, or ``rocprof`` is
            combined with a Proton backend that also intercepts HSA queues.
    """
    from aorta.instrumentation.proton import AUTO_BACKEND, QUEUE_INTERCEPTING_BACKENDS

    registry = _registry()
    requested = [n for n in names if n in registry]
    opts = options or {}
    effective: dict[str, dict[str, str]] = {}
    for name in requested:
        effective[name] = dict(registry[name].validate(opts.get(name)))

    if "rocprof" in effective and "proton" in effective:
        backend = effective["proton"]["backend"]
        if backend in QUEUE_INTERCEPTING_BACKENDS:
            resolved = (
                " (which resolves to rocprofiler or roctracer on AMD)"
                if backend == AUTO_BACKEND
                else ""
            )
            raise ValueError(
                "'rocprof' and 'proton' cannot run together with "
                f"proton backend {backend!r}{resolved} -- both install an HSA "
                "queue interceptor and the second one to attach will fail or "
                "report nothing. Use 'proton: {backend: instrumentation}' "
                "(intra-kernel measurement, no queue interception) to combine "
                "them, or run rocprof and proton as two separate cells."
            )


def wrap_argv_for_collectors(
    config: Mapping[str, Any],
    argv: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
) -> list[str]:
    """Return ``argv`` wrapped in every active collector that attaches via argv.

    The counterpart of
    :func:`aorta.emulation.mirage_launch.wrap_argv_for_environment`: it returns
    the argv unchanged when no attaching collector is active, so a run without
    ``--collect`` is byte-for-byte what it was. Collectors are applied in
    :data:`WRAP_ORDER` (rocprof outermost), and the caller should apply the
    emulation wrap *after* this one so the profiler runs inside the emulator.

    Each collector's output directory (``<collect_dir>/<subdir>``) is created
    empty here, so the trial tree has the same shape whether or not the
    workload produced any GPU activity -- ``rocprofv3`` writes nothing at all
    when the command does no GPU work -- and so a resumed trial never
    summarises the artifacts of the attempt it is replacing.

    Args:
        config: The trial config carrying the reserved ``_aorta_collect*`` keys.
        argv: The command the trial would otherwise have run.
        env: Environment the command will run with, forwarded to collectors
            that need to inspect it (Proton's device-variable translation).
            The default is load-bearing rather than a fallback: no production
            caller passes this. ``SubprocessWorkload`` wraps in ``setup()`` but
            assembles the child environment later in ``run()``, and that
            environment is ``os.environ`` plus ``AORTA_ENV_FILE``, so the
            collector's own :data:`os.environ` default already describes what
            the child gets. The parameter exists so a caller that does know its
            child env sooner can supply it, and so tests can.

    Raises:
        ValueError: a collector option is invalid.
        RuntimeError: a requested collector cannot attach (rocprofv3 missing,
            a Proton CLI wrap of a non-Python command, or an artifact
            directory that cannot be prepared). Requesting a measurement that
            cannot be taken is a clean setup failure, not a silently
            unprofiled run.
    """
    wrapped = list(argv)
    names = active_collectors(config)
    if not names:
        return wrapped
    root = _collect_root(config)
    registry = _registry()
    # Reversed so the first entry of WRAP_ORDER ends up outermost.
    for name in reversed(names):
        spec = registry.get(name)
        if spec is None or spec.wrap is None or spec.output_subdir is None:
            continue
        if root is None:
            log.warning(
                "collect: %s requested but no %s was threaded into the trial "
                "config (non-writing rank?); skipping.",
                name,
                CONFIG_KEY_COLLECT_DIR,
            )
            continue
        out_dir = root / spec.output_subdir
        try:
            _reset_output_dir(out_dir, _trusted_root(config, root))
        except OSError as exc:
            raise RuntimeError(
                f"collect: cannot prepare the {name} artifact directory "
                f"{out_dir}: {exc}. The collector has nowhere to write, so the "
                "trial would run unprofiled; fix the path or drop "
                f"'{name}' from the collect request."
            ) from exc
        wrapped = spec.wrap(wrapped, out_dir, _options_for(config, name), env=env)
    return wrapped


def summarize_collectors(config: Mapping[str, Any]) -> dict[str, Any]:
    """Parse every active collector's artifacts into flat trial metrics.

    Fail-soft by construction: returns ``{}`` when nothing was collected, and a
    collector whose artifacts are missing, partial, or malformed contributes
    fewer keys rather than raising. An opt-in measurement must never turn an
    otherwise-healthy trial into a failure.

    Returns:
        A flat mapping merged into ``WorkloadResult.metrics``. Numeric values
        (``rocprof_gpu_time_ms``, ``proton_kernel_count``, ...) are picked up by
        the ``perf.md`` metrics table and aggregated into
        ``matrix.json::cells[*].metrics_summary``. The non-numeric ones
        (top-kernel name lists, artifact directories) reach **neither**:
        ``_aggregate_metrics`` takes only real int/float scalars, and
        ``_aggregate_audit_metadata`` works from a fixed key allowlist that
        does not include collector keys. They live solely in the per-trial
        dispatcher JSON under ``.result.metrics``.
    """
    root = _collect_root(config)
    if root is None:
        return {}
    unsafe = unsafe_collector_paths(config)
    if unsafe:
        # Read-only here, but the parsers ``rglob`` these directories, so one
        # swapped for a symlink while the command ran would pull file contents
        # from outside the results tree into the trial metrics. Checked per
        # collector directory, not just the shared root -- the payload can swap
        # either. Same guard as the retention pass, which has the destructive
        # version of this exposure.
        log.warning(
            "collect: %s is (or is under) a symlink after the run; refusing to "
            "parse artifacts through it. No collector metrics for this trial.",
            ", ".join(str(path) for path in unsafe),
        )
        return {}
    metrics: dict[str, Any] = {}
    registry = _registry()
    for name in active_collectors(config):
        spec = registry.get(name)
        if spec is None or spec.summarize is None or spec.output_subdir is None:
            continue
        try:
            metrics.update(spec.summarize(root / spec.output_subdir))
        except Exception:
            # An opt-in measurement must never turn a healthy trial into a
            # failure, so the catch is deliberately unbounded.
            log.warning("collect: %s summary parsing failed; skipping.", name, exc_info=True)
    return metrics


__all__ = [
    "CONFIG_KEY_COLLECT",
    "CONFIG_KEY_COLLECT_DIR",
    "CONFIG_KEY_COLLECT_OPTIONS",
    "CONFIG_KEY_RESULTS_ROOT",
    "KNOWN_RECIPES",
    "WRAP_ORDER",
    "CollectorSpec",
    "active_collectors",
    "collector_root_is_traversable",
    "summarize_collectors",
    "unsafe_collector_paths",
    "validate_collectors",
    "wrap_argv_for_collectors",
]
