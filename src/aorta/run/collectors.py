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

import errno
import fnmatch
import logging
import os
import shutil
import stat
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from aorta.run import _fsafe

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
#: The ``(st_dev, st_ino)`` of :data:`CONFIG_KEY_RESULTS_ROOT`, read at the same
#: pre-launch moment and threaded as a two-element list so the trial JSON stays
#: plain JSON. The path alone is not a trust anchor: a payload can rename the
#: results directory aside and move a *real* directory into its pathname, which
#: leaves every ``O_NOFOLLOW`` check on that pathname satisfied. Naming the inode
#: is what makes the anchor immutable across the run.
CONFIG_KEY_RESULTS_ROOT_ID = "_aorta_results_root_id"

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
            or ``None`` for a collector the platform does not parse. Follows
            symlinks; used only on the non-POSIX fallback path where the
            fd-relative reader is unavailable.
        summarize_streams: Parses artifacts supplied as open text handles into
            the same metrics. The POSIX post-run path opens each matching file
            ``O_NOFOLLOW`` under a directory fd and calls this, so a payload
            symlink swapped in after the guard cannot redirect the read.
            Signature: ``(artifact_dir: str, *stream_groups) -> dict``, with one
            positional stream group per entry of ``glob_groups``. Each group is
            a **lazy iterator** that opens one file at a time -- a distributed
            capture can hold more artifacts than ``RLIMIT_NOFILE`` allows -- so
            an implementation must iterate each group at most once, must not
            call ``len()`` on it, and must not hold a group to consume after
            returning: the directory fds the handles come from are closed then.
            A group it never iterates is never opened at all.
        glob_groups: One basename glob per stream group the parser expects, in
            the order ``summarize_streams`` takes them. ``rocprof`` reads two
            families (stats, then trace); ``proton`` reads one (``*.hatchet``).
    """

    name: str
    output_subdir: str | None
    validate: Callable[[Mapping[str, str] | None], Any]
    wrap: Callable[..., list[str]] | None = None
    summarize: Callable[[Path], dict[str, Any]] | None = None
    summarize_streams: Callable[..., dict[str, Any]] | None = None
    glob_groups: tuple[str, ...] = ()


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
            summarize_streams=rocprof.parse_summary_from_streams,
            glob_groups=("*_kernel_stats.csv", "*_kernel_trace.csv"),
        ),
        "proton": CollectorSpec(
            name="proton",
            output_subdir=proton.OUTPUT_SUBDIR,
            validate=proton.validate_options,
            wrap=proton.wrap_argv,
            summarize=proton.parse_summary,
            summarize_streams=proton.parse_summary_from_streams,
            glob_groups=("*.hatchet",),
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


def trusted_results_anchor(config: Mapping[str, Any]) -> _fsafe.TrustedAnchor | None:
    """The dispatcher's frozen ``--results-dir`` anchor, or ``None`` if unthreaded.

    Reads :data:`CONFIG_KEY_RESULTS_ROOT` (canonicalized at dispatch) together
    with :data:`CONFIG_KEY_RESULTS_ROOT_ID` (that directory's inode identity,
    read at the same pre-launch moment). Callers must **not**
    :meth:`~pathlib.Path.resolve` the path again: the profiled process can
    replace that directory with a symlink, and resolving both the candidate and
    the anchor through the same link would make containment succeed for a path
    that has left the operator's tree.

    Returns ``None`` when the key is absent, which only happens for a direct
    programmatic caller -- the dispatcher always supplies it alongside
    ``_aorta_collect_dir``.
    """
    raw = config.get(CONFIG_KEY_RESULTS_ROOT)
    if not isinstance(raw, str) or not raw:
        return None
    return _fsafe.TrustedAnchor(Path(raw), _threaded_identity(config))


def _threaded_identity(config: Mapping[str, Any]) -> tuple[int, int] | None:
    """Parse :data:`CONFIG_KEY_RESULTS_ROOT_ID`, or ``None`` when unusable.

    A hand-built config (or one round-tripped through a lossy transport) may
    carry anything here. An unparseable value degrades to an unpinned anchor
    -- the no-follow descent still runs -- rather than raising in a guard whose
    job is to fail closed on filesystem shape, not on config shape.
    """
    raw = config.get(CONFIG_KEY_RESULTS_ROOT_ID)
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        return None
    try:
        return (int(raw[0]), int(raw[1]))
    except (TypeError, ValueError):
        return None


def _trusted_root(config: Mapping[str, Any], root: Path) -> _fsafe.TrustedAnchor:
    """:func:`trusted_results_anchor` with the historical ``root.parent`` fallback.

    The fallback only fires for a direct programmatic caller that threaded no
    anchor. Being lexical and unpinned, it fails closed rather than guessing: a
    symlink anywhere above the collector root makes the containment check
    refuse, even when the operator put it there.
    """
    anchor = trusted_results_anchor(config)
    if anchor is not None:
        return anchor
    return _fsafe.TrustedAnchor(root.parent)


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

    A path that does not exist is not reported: the caller treats any entry
    here as "keep every collector artifact", and a validated-only collector
    such as ``layer_numerics`` has no output directory unless a wrapper made
    one, so counting absence as unsafe would let one never-created directory
    retain the rocprof capture sitting beside it.
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


def collector_root_is_traversable(
    root: Path, trusted_root: _fsafe.TrustedAnchor | Path | None = None
) -> bool:
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

    A path that is simply **absent** -- every component that does exist checks
    out, the leaf was never created -- is traversable: there is nothing there to
    read or delete through, so absence is not an escape. (An absent leaf
    *under* a symlinked or escaping ancestor is still a refusal; the symlink is
    met first.) That distinction matters because
    :func:`unsafe_collector_paths` asks about every active collector's output
    subdirectory, and a validated-only collector such as ``layer_numerics``
    has no subdirectory unless a wrapper made one.

    ``trusted_root`` defaults to ``root.parent`` when the caller omits it.
    That fallback is only for a programmatic caller that did not thread
    :data:`CONFIG_KEY_RESULTS_ROOT`; it still misses a swapped ancestor, which
    is why the dispatcher always supplies the operator's ``--results-dir``.

    On POSIX this is a *probe*: it descends the path once with ``O_NOFOLLOW``
    and reports whether the leaf is reachable without crossing a symlink. It is
    a pre-filter and a log-message source -- the race-free guarantee comes from
    the caller (:func:`_reset_output_dir`, :func:`summarize_collectors`,
    :func:`aorta.run.retention.apply_retention`) *holding* the dir fd across the
    operation, not from this check. Where the platform lacks the fd primitives
    it keeps the historical lexical ``resolve()`` + component-walk.
    """
    anchor = _fsafe.as_anchor(trusted_root if trusted_root is not None else root.parent)
    if _fsafe.HAVE_FD_TRAVERSAL:
        components = _fsafe.relative_components(anchor.path, root)
        if components is None:
            return False
        try:
            with _fsafe.open_dir_nofollow(anchor, components):
                return True
        except _fsafe.UnsafePathError as exc:
            # ENOENT is "nothing is there", not "something hostile is there".
            # A symlink refused by O_NOFOLLOW surfaces as ELOOP even when it
            # dangles, so absence cannot be faked with a broken link.
            return exc.errno == errno.ENOENT
        except OSError:
            return False
    try:
        # A missing path needs no special case here: non-strict ``resolve()``
        # keeps the lexical tail, so containment and the symlink walk both still
        # apply -- an absent leaf under a symlinked ancestor stays a refusal.
        if not root.resolve().is_relative_to(anchor.path):
            return False
        return not _payload_symlink_at_or_below(root, anchor.path)
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


def _reset_output_dir(out_dir: Path, trusted_root: _fsafe.TrustedAnchor) -> None:
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
    if _fsafe.HAVE_FD_TRAVERSAL:
        _reset_output_dir_fd(out_dir, trusted_root)
        return
    if not collector_root_is_traversable(out_dir, trusted_root):
        raise OSError(
            f"refusing to prepare {out_dir}: a path component at or below "
            f"{trusted_root.path} is a symlink or resolves outside that "
            "directory, so clearing it would delete through the link. Remove "
            "the symlink in that path, or point --results-dir at a real "
            "directory."
        )
    if out_dir.is_symlink() or out_dir.is_file():
        out_dir.unlink()
    elif out_dir.is_dir():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)


def _reset_output_dir_fd(out_dir: Path, trusted_root: _fsafe.TrustedAnchor) -> None:
    """Race-free :func:`_reset_output_dir`: clear+recreate through directory fds.

    The parent chain below ``trusted_root`` (``<workload>/<trial>``) is opened
    -- and created if a first attempt has not made it yet -- with ``O_NOFOLLOW``
    on every component, so the reset holds a fd to the collector directory's
    parent that no ancestor swap can redirect. The leaf (``rocprof`` / ``proton``)
    is then inspected ``lstat``-style and removed **relative to that parent fd**:
    a symlink or file is unlinked, a real directory is recursively emptied and
    ``rmdir``-ed, and the fresh directory is ``mkdir``-ed back -- all by name
    under the held fd, never by re-resolving the pathname the payload can swap.
    """
    parent = out_dir.parent
    leaf = out_dir.name
    components = _fsafe.relative_components(trusted_root.path, parent)
    if components is None or not leaf:
        raise OSError(
            f"refusing to prepare {out_dir}: it is not lexically inside "
            f"{trusted_root.path}, so clearing it could delete outside that "
            "directory. Point --results-dir at a real directory."
        )
    try:
        with _fsafe.open_dir_nofollow(
            trusted_root, components, create_missing=True
        ) as parent_fd:
            info = _fsafe.stat_at(parent_fd, leaf)
            if info is not None and stat.S_ISLNK(info.st_mode):
                # A symlink where the collector output dir belongs is never a
                # legitimate prior attempt -- the payload planted it. Refuse
                # rather than unlink-and-recreate, matching the pre-launch
                # contract that a swapped leaf aborts the trial's collection.
                raise _fsafe.UnsafePathError(
                    f"{out_dir} is a symlink; refusing to clear it"
                )
            _fsafe.remove_entry_at(parent_fd, leaf)
            os.mkdir(leaf, dir_fd=parent_fd)
    except _fsafe.UnsafePathError as exc:
        raise OSError(
            f"refusing to prepare {out_dir}: a path component at or below "
            f"{trusted_root.path} is a symlink, resolves outside that "
            "directory, or is no longer the directory frozen before the run, "
            "so clearing it would delete through the swap. Remove the symlink "
            f"in that path, or point --results-dir at a real directory. ({exc})"
        ) from exc


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
    trusted = _trusted_root(config, root)
    metrics: dict[str, Any] = {}
    registry = _registry()
    for name in active_collectors(config):
        spec = registry.get(name)
        if spec is None or spec.output_subdir is None:
            continue
        subdir = root / spec.output_subdir
        try:
            if _fsafe.HAVE_FD_TRAVERSAL and spec.summarize_streams is not None:
                metrics.update(_summarize_streamed(spec, root, subdir, trusted))
            elif spec.summarize is not None:
                # Non-POSIX fallback: guard lexically, then parse by pathname.
                # The parsers ``rglob`` the subdirectory, so one swapped for a
                # symlink while the command ran would pull file contents from
                # outside the results tree into the trial metrics.
                if not collector_root_is_traversable(subdir, trusted):
                    log.warning(
                        "collect: %s is (or is under) a symlink after the run; "
                        "refusing to parse artifacts through it. No %s metrics "
                        "for this trial.",
                        subdir,
                        name,
                    )
                    continue
                metrics.update(spec.summarize(subdir))
        except _fsafe.UnsafePathError as exc:
            if exc.errno == errno.ENOENT:
                # The collector wrote nothing at all -- a validated-only
                # collector whose wrapper made no directory, or a command that
                # did no GPU work. Not a swap, and not worth an operator
                # warning: the parsers treat a missing tree as no metrics too.
                log.debug("collect: %s does not exist; no %s metrics.", subdir, name)
                continue
            # A component of the collector directory was a symlink after the
            # run: refuse to read through it (same exposure the destructive
            # retention pass guards). Skip this collector, keep the trial.
            log.warning(
                "collect: %s is (or is under) a symlink after the run; refusing "
                "to parse artifacts through it. No %s metrics for this trial.",
                subdir,
                name,
            )
        except Exception:
            # An opt-in measurement must never turn a healthy trial into a
            # failure, so the catch is deliberately unbounded.
            log.warning("collect: %s summary parsing failed; skipping.", name, exc_info=True)
    return metrics


def _summarize_streamed(
    spec: CollectorSpec, root: Path, subdir: Path, trusted: _fsafe.TrustedAnchor
) -> dict[str, Any]:
    """Parse one collector's artifacts through no-follow, fd-relative reads.

    Descends to the collector subdirectory holding a dir fd no ancestor swap can
    redirect, walks it without following any symlink to list the files whose
    basename matches one of ``spec.glob_groups``, then hands the parser one lazy
    iterator of open handles per group (in ``glob_groups`` order). Each handle is
    opened ``O_NOFOLLOW`` under the held directory fd and closed before the next
    one is opened, so a capture with more per-rank artifacts than
    ``RLIMIT_NOFILE`` allows still aggregates every file instead of silently
    dropping the ranks that came after the limit.

    Raises:
        UnsafePathError: a component of the collector directory is a symlink or
            does not exist.
    """
    components = _fsafe.relative_components(trusted.path, subdir)
    if components is None:
        raise _fsafe.UnsafePathError(
            f"{subdir} is not lexically inside the trusted root {trusted.path}"
        )
    assert spec.summarize_streams is not None  # guarded by the caller
    with _fsafe.open_dir_nofollow(trusted, components) as base_fd:
        matched: list[list[str]] = [[] for _ in spec.glob_groups]
        for rel, _dir_fd, fname, _size in _fsafe.iter_regular_files(base_fd):
            for index, pattern in enumerate(spec.glob_groups):
                if fnmatch.fnmatch(fname, pattern):
                    matched[index].append(rel)
                    break
        groups = [_open_artifacts(base_fd, rels) for rels in matched]
        return spec.summarize_streams(str(subdir), *groups)


def _open_artifacts(base_fd: int, relative_paths: Sequence[str]) -> Iterator[TextIO]:
    """Yield each artifact under ``base_fd`` as an open handle, one at a time.

    The handle is closed before the next is opened, bounding this pass to a
    single artifact descriptor no matter how many ranks the capture holds.
    Reopening a directory chain per file is safe because ``base_fd`` names an
    inode: the descent below it is fd-relative and ``O_NOFOLLOW``, so the swap
    the held fd protects against stays defeated.

    A file that cannot be opened at all -- it vanished, or a payload left a
    symlink where the walk saw a regular file -- is skipped. Descriptor
    exhaustion is **not** skipped: it means the remaining artifacts would be
    missing from the totals, and a confidently-wrong ``rocprof_gpu_time_ms``
    covering a prefix of the ranks is worse than no metric, so it propagates
    and the caller drops the collector's metrics entirely.
    """
    for rel in relative_paths:
        *parents, name = rel.split("/")
        try:
            with (
                _fsafe.open_dir_at(base_fd, parents) as dir_fd,
                _fsafe.secure_open_read(
                    dir_fd, name, encoding="utf-8", newline=""
                ) as stream,
            ):
                yield stream
        except OSError as exc:
            if exc.errno in (errno.EMFILE, errno.ENFILE):
                raise
            log.debug("collect: skipping unreadable artifact %s (%s)", rel, exc)


__all__ = [
    "CONFIG_KEY_COLLECT",
    "CONFIG_KEY_COLLECT_DIR",
    "CONFIG_KEY_COLLECT_OPTIONS",
    "CONFIG_KEY_RESULTS_ROOT",
    "CONFIG_KEY_RESULTS_ROOT_ID",
    "KNOWN_RECIPES",
    "WRAP_ORDER",
    "CollectorSpec",
    "active_collectors",
    "collector_root_is_traversable",
    "summarize_collectors",
    "trusted_results_anchor",
    "unsafe_collector_paths",
    "validate_collectors",
    "wrap_argv_for_collectors",
]
