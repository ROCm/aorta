"""Output layout + writers for the triage matrix.

Three artifacts per run, all in the same ``<output-dir>/<ticket>/<workload>/
<run-timestamp>/`` directory:

* ``matrix.md`` -- human-readable table matching the §"matrix.md target
  format" block in issue #151.
* ``matrix.json`` -- full machine-readable per-cell data: step-time stats,
  the env-var bundle as actually applied (``resolved_env_vars``), the
  resolved :class:`aorta.registry.Environment` descriptor
  (``resolved_environment``), trial JSON paths, exit-status histogram, and
  cell-level errors.
* ``recipe.resolved.yaml`` -- a strict, reloadable recipe snapshot. **Named
  mitigations and environments are deliberately NOT expanded** -- the file
  is the rerunnable artifact, not a drift-pinning artifact. Inline-docker
  cells are re-emitted as ``{docker: <ref>}`` so the same ``_inline_<hash>``
  is re-derived without a sidecar; named registry entries stay by name.
  See :func:`write_resolved_recipe` for the rationale and ``recipes/README.md``
  for replay caveats. To detect registry drift across runs, compare each
  run's ``matrix.json::cells[*].resolved_env_vars``.

Sibling files / directories (written by :mod:`aorta.triage.runner`):

* ``host_env.json`` -- one collect_env() snapshot taken at runner start.
* ``environments/<env-name>/env.json`` -- one collect_env() snapshot per
  unique environment, captured right before that env's first cell runs.
* ``cells/<cell-name>/`` -- B1's per-trial JSON output for that cell.
* ``sidecars/<basename>`` -- byte-identical copy of every operator-supplied
  ``--mitigations-file`` so the run dir is self-contained for replay.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import errno
import json
import logging
import os
import re
import socket
from collections.abc import Iterable, Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

import yaml

from aorta.registry import get_environment
from aorta.triage.confound import CONFOUND_DID_NOT_RUN, ConfoundTag
from aorta.triage.matrix import CellStats
from aorta.triage.recipe import Recipe

log = logging.getLogger(__name__)

NO_TICKET_SLUG = "_no_ticket_"

# ``flat_resume`` lockfile name. Lives at ``<run_dir>/.aorta-probe.lock``; the
# leading dot keeps it out of casual ``ls`` output and matches the convention
# used by other resume-state files in the run dir.
FLAT_RESUME_LOCKFILE = ".aorta-probe.lock"

# Filesystem-safe slug: replace anything that isn't [A-Za-z0-9_.-] with '_'.
# Ticket IDs like "PROJ-123" pass through unchanged; spaces, slashes, ':'
# etc. get sanitised so we never create surprise subdirectories.
_SAFE_RE = re.compile(r"[^A-Za-z0-9_.\-]")
# Even after character-class scrubbing, "." and ".." are still meaningful path
# components on every filesystem we care about.  Keep them out of the output
# tree so a ticket like ".." can't move the run directory up a level.
_RESERVED_SLUGS = frozenset({".", ".."})


def safe_slug(value: str) -> str:
    """Turn a ticket / workload / env name into a safe directory component.

    Replaces anything outside ``[A-Za-z0-9_.-]`` with ``_`` and rewrites the
    reserved ``.`` / ``..`` components so the result can never refer to the
    current or parent directory. Empty input also rewrites to ``_``.
    """
    cleaned = _SAFE_RE.sub("_", value)
    if not cleaned or cleaned in _RESERVED_SLUGS:
        return "_"
    return cleaned


def format_timestamp(now: _dt.datetime | None = None) -> str:
    """Return an ISO-8601-ish timestamp suitable as a directory name.

    ``2026-04-28T14-12-03`` matches the layout shown in issue #151 §"Output
    layout". Colons are replaced with dashes because Windows filesystems
    reject them and it makes the path easier to copy-paste from a shell.
    """
    now = now or _dt.datetime.now(_dt.timezone.utc)
    return now.strftime("%Y-%m-%dT%H-%M-%S")


def resolve_run_dir(
    output_dir: Path,
    recipe: Recipe,
    timestamp: str | None = None,
    layout: Literal["timestamped", "flat_resume"] = "timestamped",
) -> Path:
    """Return the per-run output directory for the given layout.

    Two layouts are supported:

    * ``"timestamped"`` (default) -- preserves ``aorta triage run``
      behaviour byte-equivalently. Returns
      ``<output-dir>/<ticket>/<workload>/<timestamp>[-N]/``. Creates
      parents as needed and **never overwrites an existing directory**:
      the base candidate is ``<timestamp>``; if that already exists
      (two runs in the same wall-clock second for the same
      ``(ticket, workload)`` -- common in CI loops or concurrent
      jobs), a numeric suffix ``-2``, ``-3``, ... is appended until
      ``mkdir(exist_ok=False)`` succeeds. The race between two
      parallel processes is resolved by ``mkdir`` itself: only one
      can win for a given suffix, the loser bumps and retries. The
      base directory ``<output_dir>/<ticket>/<workload>/`` IS created
      with ``exist_ok=True`` -- it's a shared parent across runs and
      the "no-overwrite" guarantee only applies to the per-run leaf.

    * ``"flat_resume"`` -- the layout ``aorta probe`` (issue #188)
      passes. Returns ``<output-dir>/<safe_slug(ticket)>/`` (or
      ``<output-dir>/_no_ticket_/`` when ``ticket`` is ``None``) with
      ``mkdir(parents=True, exist_ok=True)``. NO timestamp segment, NO
      ``<workload>`` segment -- those would defeat the resume model
      (re-running the same probe with the same ``--output`` and
      ``--ticket`` must land in the same directory so per-cell
      ``trial_<n>/result.json`` files can be detected as
      "already complete"). The ``timestamp`` argument is ignored in
      this branch but kept on the signature so callers don't need to
      know which layout they're invoking.
    """
    # ``layout`` is a typed Literal but the type guard only fires under
    # mypy --strict; a caller passing ``layout="flatresume"`` (typo)
    # would otherwise silently land in the timestamped branch. Reject
    # unknown values at runtime so probe-mode callers can't
    # accidentally get the wrong output tree.
    if layout not in ("timestamped", "flat_resume"):
        raise ValueError(
            f"resolve_run_dir: layout must be 'timestamped' or 'flat_resume', "
            f"got {layout!r}"
        )

    ticket_slug = safe_slug(recipe.ticket) if recipe.ticket else NO_TICKET_SLUG

    if layout == "flat_resume":
        # Idempotent: the same (output_dir, ticket) tuple always yields
        # the same leaf so re-runs can detect already-complete trials
        # via ``aorta.probe.resume.is_trial_complete``. No timestamp
        # segment because that would force a fresh leaf on every
        # invocation -- which is exactly what ``aorta triage run``'s
        # "timestamped" layout does and exactly what probe-mode is
        # explicitly opting out of.
        run_dir = Path(output_dir) / ticket_slug
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    workload_slug = safe_slug(recipe.workload)
    ts = timestamp or format_timestamp()
    parent = Path(output_dir) / ticket_slug / workload_slug
    parent.mkdir(parents=True, exist_ok=True)

    # Try the bare timestamp first, then -2, -3, ... -- bounded so a buggy
    # caller can't spin forever. The cap is generous (10k runs in the same
    # second-bucket would already imply something is very wrong upstream)
    # but finite so failures surface as a clean error rather than a hang.
    for suffix in range(1, 10_001):
        candidate = parent / (ts if suffix == 1 else f"{ts}-{suffix}")
        try:
            candidate.mkdir(exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError(
        f"resolve_run_dir: exhausted 10000 suffixes for {parent / ts!r}; "
        "something is wedged upstream (clock not advancing? runaway loop?)"
    )


class RunDirLockedError(RuntimeError):
    """Raised when a ``flat_resume`` run dir is already locked by another writer.

    Carries the parsed lock-holder identity so the CLI layer can render a
    targeted operator message (host, PID, start time) rather than a generic
    "something is wrong" stack trace.
    """

    def __init__(
        self,
        run_dir: Path,
        holder_host: str,
        holder_pid: int | None,
        holder_started_at: str | None,
        reason: str,
    ) -> None:
        self.run_dir = run_dir
        self.holder_host = holder_host
        self.holder_pid = holder_pid
        self.holder_started_at = holder_started_at
        self.reason = reason
        super().__init__(
            f"flat_resume run dir {run_dir} is locked: {reason} "
            f"(holder host={holder_host!r} pid={holder_pid} "
            f"started_at={holder_started_at!r}). If the holder is no longer "
            f"running, remove {run_dir / FLAT_RESUME_LOCKFILE} and retry."
        )


def _pid_alive(pid: int) -> bool:
    """Best-effort liveness probe for a same-host PID.

    ``os.kill(pid, 0)`` is the standard Unix idiom: it doesn't actually
    signal the process, just performs the permission/existence check. We
    treat ``EPERM`` (process exists but is owned by another user) as alive
    too -- that's still a live process that could be mid-write.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but we can't signal it (different uid).
        return True
    except OSError:
        # Defensive: any other errno -> assume alive to avoid stomping.
        return True
    return True


@contextlib.contextmanager
def acquire_flat_resume_lock(run_dir: Path) -> Iterator[None]:
    """Advisory PID+host lockfile for the ``flat_resume`` run directory.

    ``flat_resume`` reuses a stable ``<output>/<ticket>/`` leaf across
    invocations (intentional, so per-trial ``result.json`` files can be
    detected as already-complete). That same property means two concurrent
    ``aorta probe`` invocations against the same ``--output`` /
    ``--ticket`` would race on ``matrix.json``, ``matrix.md``, and per-cell
    artifacts.

    On entry we ``O_CREAT|O_EXCL`` a small JSON file holding our PID, host,
    and start timestamp. If the file already exists we read it and decide:

    * Same host **and** ``os.kill(pid, 0)`` succeeds -> raise
      :class:`RunDirLockedError`. The other writer is genuinely live;
      stomping their tree would corrupt both runs.
    * Same host **and** PID is dead -> stale lock from a crashed prior
      run. Log a warning, overwrite, and proceed (this is the recovery
      path -- it's the whole reason ``flat_resume`` exists).
    * Different host -> we have no way to verify liveness across hosts.
      Fail closed: raise :class:`RunDirLockedError` and ask the operator
      to remove the lockfile explicitly. (Shared NFS with concurrent
      multi-host writers is the worst case; rather than guess we surface
      the situation.)
    * Lockfile present but unparseable -> treat as stale (a partial write
      from a crashed prior run); warn + overwrite. The alternative is to
      wedge resume indefinitely on a corrupt lock file.

    On context exit the lockfile is best-effort removed; failure to
    remove (e.g. the run dir was deleted under us) does not raise so the
    caller's own exit path isn't masked.

    Limitations (deliberately documented rather than papered over):

    * Advisory only -- a non-aorta writer (or an aorta version predating
      this lock) into the same tree is undetected.
    * Same-host liveness is checked via PID; PID reuse on busy hosts
      with very long-running operator sessions is theoretically possible
      but unlikely within a probe sweep's lifetime.
    * Cross-host NFS semantics for ``O_EXCL`` are
      filesystem-implementation-dependent. We don't rely on the
      atomicity of the cross-host case; the same-host PID check is the
      load-bearing guarantee. Operators running concurrent multi-host
      probes against the same tree should not (Phase-1 scope).
    """
    lock_path = run_dir / FLAT_RESUME_LOCKFILE
    payload = {
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "started_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    serialised = json.dumps(payload, indent=2).encode("utf-8")

    while True:
        try:
            fd = os.open(
                str(lock_path),
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
        except FileExistsError:
            # Existing lock: inspect the holder and decide stale-vs-live.
            holder_host: str = "<unknown>"
            holder_pid: int | None = None
            holder_started_at: str | None = None
            try:
                holder_raw = lock_path.read_text(encoding="utf-8")
                holder = json.loads(holder_raw)
                holder_host = str(holder.get("host", "<unknown>"))
                pid_raw = holder.get("pid")
                holder_pid = int(pid_raw) if pid_raw is not None else None
                started_raw = holder.get("started_at")
                holder_started_at = str(started_raw) if started_raw is not None else None
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                # Corrupt or unreadable lock: treat as stale.
                log.warning(
                    "flat_resume lock at %s is unreadable (%s); "
                    "treating as stale and taking over",
                    lock_path,
                    exc,
                )
                try:
                    lock_path.unlink()
                except OSError:
                    pass
                continue

            our_host = socket.gethostname()
            if holder_host == our_host:
                if holder_pid is not None and _pid_alive(holder_pid):
                    # ``raise ... from None``: the originating
                    # ``FileExistsError`` from ``os.open`` is an expected
                    # signal, not an error; chaining it would mislead the
                    # operator into thinking something failed at the FS
                    # layer when the real cause is "another process owns
                    # this dir".
                    raise RunDirLockedError(
                        run_dir=run_dir,
                        holder_host=holder_host,
                        holder_pid=holder_pid,
                        holder_started_at=holder_started_at,
                        reason="another aorta probe process on this host "
                        "is still running",
                    ) from None
                log.warning(
                    "flat_resume lock at %s is stale (holder pid=%s on this "
                    "host is no longer running, started_at=%s); taking over",
                    lock_path,
                    holder_pid,
                    holder_started_at,
                )
                try:
                    lock_path.unlink()
                except OSError:
                    pass
                continue

            # See the same-host live branch above for the ``from None``
            # rationale.
            raise RunDirLockedError(
                run_dir=run_dir,
                holder_host=holder_host,
                holder_pid=holder_pid,
                holder_started_at=holder_started_at,
                reason=(
                    "lock was created on a different host; cross-host "
                    "liveness cannot be verified automatically"
                ),
            ) from None
        except OSError as exc:
            # Anything other than EEXIST (which FileExistsError covers)
            # is a genuine open() failure -- don't pretend we hold the
            # lock.
            raise RuntimeError(
                f"acquire_flat_resume_lock: could not create {lock_path} "
                f"(errno={errno.errorcode.get(exc.errno, exc.errno)}): {exc}"
            ) from exc
        else:
            break

    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(serialised)
        yield
    finally:
        try:
            lock_path.unlink()
        except OSError:
            # Best-effort: the dir may have been removed under us, or a
            # concurrent process raced us to cleanup. Either way, not
            # crashing here keeps the caller's exit path intact.
            pass


def _format_mitigations(mitigations: Iterable[str]) -> str:
    items = list(mitigations)
    return ", ".join(items) if items else "-"


def _format_failure_rate(cell: CellStats) -> str:
    if cell.error is not None:
        return "n/a"
    pct = int(round(cell.failure_rate * 100))
    return f"{pct}%"


def _format_failures(cell: CellStats) -> str:
    """Render the Failures column as ``failed / trials`` (e.g. ``3 / 8``).

    Both numerator and denominator are spelled out so the column header
    ("Failures") and the value together read as "3 failures out of 8 trials"
    -- the previous "Trials" header on this same value confused readers
    into reading 3/8 as a trial-count column (issue #160 review round 6).
    """
    if cell.error is not None:
        return "n/a"
    return f"{cell.failed_count} / {cell.trials}"


def _format_step_ms(cell: CellStats) -> str:
    if cell.error is not None or cell.mean_step_time_ms <= 0:
        return "n/a"
    return f"{cell.mean_step_time_ms:.1f}"


def _format_confound(tag: ConfoundTag) -> str:
    return str(tag)


_CONFIG_KEY_ABSENT = object()


def _varying_workload_config_keys(cell_stats: list[CellStats]) -> list[str]:
    """Sorted keys whose effective value differs across cells.

    "Effective value" includes absence -- a cell that omits a key is
    distinct from one that sets it. ``repr`` is used for the equality
    comparison so unhashable values (e.g. dict, list) don't blow up the
    set construction, and ``sort_keys=True`` keeps the canonical form
    stable across insertion-order differences -- ``{a:1,b:2}`` and
    ``{b:2,a:1}`` are semantically equal but ``repr``-different in
    Python 3.7+, which would otherwise flag spurious "varying" keys
    from programmatically-generated recipes. ``default=repr`` falls
    back for the absent-key sentinel and any non-JSON values.
    Returns ``[]`` when no cell has workload_config or when every cell
    agrees on every key.
    """
    all_keys: set[str] = set()
    for c in cell_stats:
        all_keys.update(c.workload_config.keys())
    if not all_keys:
        return []
    return sorted(
        k
        for k in all_keys
        if len({_canon(c.workload_config.get(k, _CONFIG_KEY_ABSENT)) for c in cell_stats}) > 1
    )


def _canon(value: Any) -> str:
    """Stable canonical form for cross-cell value comparison.

    ``json.dumps`` can still raise even with ``default=repr``: circular
    references are detected before the ``default`` callback runs, and a
    user-defined ``__repr__`` could itself raise from inside ``default``.
    Fall back to bare ``repr`` so a pathological workload_config value
    can never crash matrix rendering -- Python's ``repr`` handles
    circular structures natively (prints ``[[...]]``).
    """
    try:
        return json.dumps(value, sort_keys=True, default=repr)
    except (ValueError, TypeError):
        return repr(value)


def _escape_md_cell(s: str) -> str:
    """Escape characters that would break a markdown table row."""
    return s.replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>")


def _format_workload_config(cell: CellStats, varying_keys: list[str]) -> str:
    """Render only the varying-key subset of one cell's workload_config.

    Keys omitted from the cell render nothing (no ``key=—`` noise). When
    the cell has none of the varying keys at all, render ``"—"`` so the
    column stays width-aligned. ``str(value)`` keeps scalar values
    unquoted (``shampoo_api=old`` rather than ``shampoo_api='old'``).
    Values are passed through ``_escape_md_cell`` so a ``|`` or newline
    in a workload_config value (workload_config is not schema-validated)
    can't break the markdown table layout.
    """
    parts = [
        f"{k}={_escape_md_cell(str(cell.workload_config[k]))}"
        for k in varying_keys
        if k in cell.workload_config
    ]
    return ", ".join(parts) if parts else "—"


def _render_failure_hints(cell_stats: list[CellStats]) -> list[str]:
    """Build the ``## Failure hints`` section for matrix.md.

    Returns an empty list when no cell carries a hint -- the caller
    suppresses the section entirely in that case so the markdown does not
    sprout an empty header for runs where every workload either passed or
    failed without emitting an explanatory hint.

    One bullet per ``(cell, hint)`` pair: a cell that emits two distinct
    hints across its trials gets two bullets so each hint stays one line
    and operators can scan-read. The ``(N/M trials)`` parenthetical reads
    as "the hint fired in N of the cell's M trials" -- not the cell's
    failure rate; trials can fail without emitting a hint.
    """
    if not any(cell.failure_hints for cell in cell_stats):
        return []
    lines = ["## Failure hints", ""]
    for cell in cell_stats:
        for hint, count in cell.failure_hints:
            lines.append(f"- **{cell.name}** ({count}/{cell.trials} trials): {hint}")
    lines.append("")
    return lines


def write_matrix_md(
    path: Path,
    recipe: Recipe,
    cell_stats: list[CellStats],
    baseline: CellStats,
    confound_tags: dict[str, tuple[ConfoundTag, float | None]],
    warnings: list[str],
    run_timestamp: str,
) -> None:
    """Render matrix.md in the format from issue #151 §"matrix.md target format"."""
    lines: list[str] = []
    lines.append(f"# Triage Matrix - {recipe.workload}")
    lines.append("")
    if warnings:
        lines.append("> [!WARNING]")
        for w in warnings:
            lines.append(f"> {w}")
        lines.append("")

    lines.append(f"**Ticket**: {recipe.ticket or '(none)'}  ")
    lines.append(f"**Workload**: {recipe.workload}  ")
    recipe_line = "**Recipe**: "
    if recipe.source_path is not None:
        sha = (recipe.source_sha256 or "")[:10]
        recipe_line += f"{recipe.source_path} (sha256:{sha})"
    else:
        recipe_line += "(flag-mode; in-memory)"
    lines.append(recipe_line + "  ")
    lines.append(f"**Trials per cell**: {recipe.trials}  ")
    lines.append(f"**Steps per trial**: {recipe.steps}  ")
    lines.append(f"**Run timestamp**: {run_timestamp}  ")
    baseline_step = (
        f"{baseline.mean_step_time_ms:.1f} ms"
        if baseline.error is None and baseline.mean_step_time_ms > 0
        else "n/a"
    )
    lines.append(f"**Baseline cell**: {baseline.name} (mean step time = {baseline_step})")
    lines.append("")
    lines.append("## Reproduction Summary")
    lines.append("")

    # Iters column appears whenever any cell has iteration data worth
    # surfacing -- hidden only when *every* cell rendered as the em-dash
    # placeholder (legacy workloads that don't speak the new contract).
    # Gating on ``iters_display != "—"`` (rather than ``configured_iters
    # is not None``) is deliberate: the defensive "?/?" case sets
    # ``configured_iters=None`` to flag the contradiction, but the
    # contradiction itself is exactly what an operator needs to see --
    # hiding the column would silently swallow it.
    show_iters = any(c.iters_display != "—" for c in cell_stats)
    # Config column surfaces per-cell workload_config keys whose values
    # vary across cells -- e.g. ``shampoo_api=old`` when one cell flips
    # the optimizer API and others don't. Hidden entirely when no cell
    # has workload_config, or when every cell agrees on every key.
    # ``Confound`` stays unchanged: this column carries the disambiguation,
    # not a new confound tag.
    varying_config_keys = _varying_workload_config_keys(cell_stats)
    show_config = bool(varying_config_keys)
    # Phase 2 (issue #188): Top failure / Top warn columns appear
    # immediately after Failures when at least one cell in the
    # matrix populates the corresponding field. Triage-mode runs
    # never set them, so the columns stay hidden and the legacy
    # matrix.md layout is byte-equivalent.
    show_top_failure = any(c.top_failure_detector_id for c in cell_stats)
    show_top_warn = any(c.top_warn_detector_id for c in cell_stats)
    header_cells: list[str] = [
        "Cell",
        "Mitigations",
        "Environment",
    ]
    if show_config:
        header_cells.append("Config")
    header_cells.extend(["Failure rate", "Failures"])
    if show_top_failure:
        header_cells.append("Top failure")
    if show_top_warn:
        header_cells.append("Top warn")
    if show_iters:
        header_cells.append("Iters")
    header_cells.extend(["Mean step (ms)", "Confound"])
    header = tuple(header_cells)
    rows: list[tuple[str, ...]] = [header]
    for cell in cell_stats:
        tag, _ = confound_tags.get(cell.name, (cell.error and "error" or "-", None))
        row: list[str] = [
            cell.name,
            _format_mitigations(cell.mitigations),
            cell.environment,
        ]
        if show_config:
            row.append(_format_workload_config(cell, varying_config_keys))
        row.extend([_format_failure_rate(cell), _format_failures(cell)])
        if show_top_failure:
            row.append(cell.top_failure_detector_id or "—")
        if show_top_warn:
            row.append(cell.top_warn_detector_id or "—")
        if show_iters:
            row.append(cell.iters_display)
        row.extend([_format_step_ms(cell), _format_confound(tag)])
        rows.append(tuple(row))
    widths = [max(len(r[i]) for r in rows) for i in range(len(header))]

    def _row(cells: tuple[str, ...]) -> str:
        return "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells)) + " |"

    lines.append(_row(rows[0]))
    lines.append("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for row in rows[1:]:
        lines.append(_row(row))

    lines.append("")
    lines.extend(_render_failure_hints(cell_stats))
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- Cell name comes from the recipe; mitigations + environment columns "
        "disambiguate when names get terse."
    )
    lines.append("- Confound column legend:")
    lines.append("  - `(baseline)` -- the cell against which all step-time ratios are computed.")
    lines.append("  - `-` -- the mitigation appears to work without a speed cost. Trust this cell.")
    lines.append(
        "  - `speed (+N%)` -- the mitigation may be suppressing failure via slower "
        "iteration rather than a real fix. Verify with `rocprofv3` dispatch comparison "
        "before drawing causal conclusions."
    )
    lines.append(
        "  - `no effect` -- the mitigation neither changed the failure rate nor slowed "
        "iteration; it likely doesn't apply to this workload."
    )
    lines.append(
        "  - `n/a` -- the cell could not be compared against the baseline. Possible "
        "reasons: the baseline errored, the baseline produced no usable timing, the "
        "cell itself produced no usable timing, or the cell or baseline lacks "
        "per-step instrumentation (`step_time_source != per_step`) so the ratio "
        "would be dominated by setup / teardown / crash time rather than per-step "
        "cost. Distinct from `-`: these cells are **unclassified**, not "
        "trustworthy. Check `matrix.json::cells[*].step_time_source` to see which "
        "branch each row landed on."
    )
    lines.append("  - `error` -- the whole cell failed; row preserved so the matrix is complete.")
    # Gate the did_not_run legend on the tag actually appearing in the
    # rendered Confound column. Gating on ``any(c.outcome_counts ...)``
    # leaks the legend into matrices where every cell completed (the
    # outcome histogram is non-empty for every new-contract run, but
    # the tag itself only renders when ``is_did_not_run_cell`` returns
    # True). Reading from ``confound_tags`` keeps the legend in lockstep
    # with what the table actually shows.
    if any(tag == CONFOUND_DID_NOT_RUN for tag, _ in confound_tags.values()):
        lines.append(
            "  - `did_not_run` -- every trial in the cell ended before the workload's "
            "primary code path began (e.g. setup-time crash). The cell is excluded "
            "from confound classification entirely; `Mean step (ms)` is `n/a` because "
            "any number derived from setup-only wall clock would misrepresent "
            "iteration timing. Inspect `cells/<cell-name>/<workload>/trial_*.json` "
            "for the cause."
        )
    if show_iters:
        lines.append(
            "- `Iters` -- iterations actually executed vs. configured. `0/<N>` = workload "
            "never started its main work phase. `<min>..<max>/<N>` = trials in the cell "
            "completed different counts (e.g. crashed mid-run). `?/?` = trials disagreed "
            "on the configured count (defensive; should not happen under a single recipe). "
            "`—` = workload didn't track iterations. The column is hidden entirely "
            "when no cell in the matrix populates the new field."
        )
    if show_top_failure or show_top_warn:
        lines.append(
            "- `Top failure` / `Top warn` -- the most-frequently fired detector ID "
            "(probe-mode Phase 2; issue #188) across the cell's trials. Built-in "
            "(`tier1:`, `tier2:`, `tier3:`, `tier4:`) and user (`custom:`) IDs "
            "are listed as peers. The columns are hidden in triage-mode runs."
        )
    if show_config:
        lines.append(
            "- `Config` -- per-cell `workload_config` keys whose value differs across "
            "cells (rendered `key=value, key2=value2`). Only varying keys appear, so "
            "a key set identically by every cell stays hidden. `—` means the cell "
            "has none of the varying keys. Confound classification is unchanged; this "
            "column carries the disambiguation when two otherwise-identical rows "
            "behave differently because of a workload knob (e.g. `shampoo_api=old` "
            "selecting the V1 SHAMPOO entry script). The column is hidden entirely "
            "when no cell sets `workload_config`, or when every cell agrees on every key."
        )
    lines.append(
        "- `Failures` is `failed_count / trial_count` (e.g. `3 / 8` = three failed out "
        "of eight). `Failure rate` is the same data as a percentage and counts every "
        "trial whose `exit_status != ok` or whose `WorkloadResult.passed` is False; "
        "neither is NaN-specific. Use `matrix.json::cells[*].exit_status_counts` to "
        "break failures down by mode: `workload_failed` (run() reported "
        "passed=False), `workload_setup_failed` (setup() raised, so the "
        "workload never reached the measurement -- a 100% setup-fail row is "
        "NOT a 100% reproduction), `infrastructure_failed` (construction or "
        "run() itself raised), `unknown`, etc."
    )
    lines.append(
        "- Only `mean step (ms)` is shown here. Per-cell `std`, `min`, `max`, "
        "`p50`, `p90`, `p99`, raw step-time arrays, exit-status histogram, and "
        "per-trial JSON paths are in `matrix.json`."
    )
    lines.append(
        "- `recipe.resolved.yaml` (alongside this file) is a strict, reloadable "
        "recipe: inline-docker cells are pinned via `{docker: <ref>}`, but named "
        "mitigations / environments are NOT expanded -- a later registry change "
        "can produce a different run. Compare `matrix.json::cells[*].resolved_env_vars` "
        "between runs to detect drift. See `recipes/README.md` for full replay caveats."
    )
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def write_matrix_json(
    path: Path,
    recipe: Recipe,
    cell_stats: list[CellStats],
    baseline_name: str,
    confound_tags: dict[str, tuple[ConfoundTag, float | None]],
    run_timestamp: str,
    warnings: list[str],
    sidecar_files: tuple[Path, ...] | None = None,
) -> None:
    """Serialise the full per-cell matrix as JSON.

    Each cell entry carries (a) the aggregated stats from
    :class:`aorta.triage.matrix.CellStats`, (b) the resolved
    :class:`aorta.registry.Environment` descriptor (formerly written into
    ``recipe.resolved.yaml`` -- moved here so that file can stay schema-valid
    and re-loadable by :func:`aorta.triage.recipe.load_recipe`), and (c) the
    confound tag + step-time ratio.
    """
    doc: dict[str, Any] = {
        "schema_version": 1,
        "workload": recipe.workload,
        "ticket": recipe.ticket,
        "trials_per_cell": recipe.trials,
        "steps_per_trial": recipe.steps,
        "run_timestamp": run_timestamp,
        "baseline_cell": baseline_name,
        "confound": {
            "threshold": recipe.confound.threshold,
            "baseline_cell_configured": recipe.confound.baseline_cell,
        },
        "warnings": list(warnings),
        "recipe_source": {
            "path": str(recipe.source_path) if recipe.source_path else None,
            "sha256": recipe.source_sha256,
        },
        "cells": [],
    }
    for cell in cell_stats:
        tag, ratio = confound_tags.get(cell.name, ("-", None))
        entry = asdict(cell)
        entry["failure_rate"] = cell.failure_rate
        entry["confound"] = tag
        entry["step_time_ratio"] = ratio
        try:
            entry["resolved_environment"] = resolved_cell_environment(
                cell.environment,
                inline_environments=recipe.inline_environments,
                sidecar_files=sidecar_files,
            )
        except Exception as exc:  # pragma: no cover - belt-and-suspenders
            entry["resolved_environment"] = {
                "name": cell.environment,
                "_resolution_error": f"{type(exc).__name__}: {exc}",
            }
        doc["cells"].append(entry)

    path.write_text(json.dumps(doc, indent=2, sort_keys=False), encoding="utf-8")


def write_resolved_recipe(
    path: Path,
    recipe: Recipe,
    sidecar_files: tuple[Path, ...] | None = None,
) -> None:
    """Write a schema-valid ``recipe.resolved.yaml`` that can be re-loaded as-is.

    The output is a strict :func:`aorta.triage.recipe.load_recipe` input -- no
    debug-only fields, no per-cell ``resolved_mitigation_env`` block, no
    top-level ``inline_environments`` key. The "resolved" property comes from
    inline-docker cells being re-emitted in the ``{docker: <ref>}`` shorthand
    form (so re-loading on another machine reproduces the same
    ``_inline_<hash>`` env without needing a sidecar JSON next to the file).

    Per-cell debug expansions (the ``Environment`` descriptor and the unioned
    mitigation env-var bundle) live in ``matrix.json`` instead, where they
    belong as run-time state.

    For runs that used ``--mitigations-file``, the resolved YAML still
    references those mitigation/environment names by name -- it is **not**
    self-contained on its own. The runner snapshots the operator-supplied
    sidecars into ``<run_dir>/sidecars/<basename>`` (see
    :func:`aorta.triage.runner._copy_operator_sidecars`) so the rerun
    command is::

        aorta triage run --recipe recipe.resolved.yaml \\
            --mitigations-file sidecars/<basename>  # repeat per sidecar

    The resolved YAML schema is intentionally kept strict (no metadata
    namespace) -- the README documents the replay command, and the runner
    echoes a one-line "to rerun:" hint on stdout when sidecars are present.
    ``sidecar_files`` is intentionally unused at write time; it is preserved
    on the signature so the runner can keep its single
    "everything-needed-for-replay" call site.
    """
    del sidecar_files  # see docstring; kept for caller-stable signature

    inline_docker = {e.name: e.docker for e in recipe.inline_environments}

    resolved_cells: list[dict[str, Any]] = []
    for cell in recipe.cells:
        if cell.environment in inline_docker:
            cell_env: Any = {"docker": inline_docker[cell.environment]}
        else:
            cell_env = cell.environment
        cell_doc: dict[str, Any] = {
            "name": cell.name,
            "mitigations": list(cell.mitigations),
            "environment": cell_env,
        }
        if cell.extra_env:
            cell_doc["extra_env"] = dict(cell.extra_env)
        if cell.trials is not None:
            cell_doc["trials"] = cell.trials
        if cell.steps is not None:
            cell_doc["steps"] = cell.steps
        if cell.workload_config:
            # Emit cell-scope workload_config verbatim. The runner merges
            # recipe-scope under cell-scope at execution time; the resolved
            # YAML preserves both scopes (recipe-scope key below) so a
            # round-trip load+run produces the same effective config.
            cell_doc["workload_config"] = dict(cell.workload_config)
        resolved_cells.append(cell_doc)

    doc: dict[str, Any] = {
        "schema_version": recipe.schema_version,
        "workload": recipe.workload,
        "trials": recipe.trials,
        "steps": recipe.steps,
    }
    if recipe.ticket is not None:
        doc["ticket"] = recipe.ticket
    if recipe.workload_config:
        # Recipe-scope workload_config -- emitted only when non-empty so
        # recipes that never used the field round-trip to byte-equivalent
        # YAML. Cell-scope values are written per cell above.
        doc["workload_config"] = dict(recipe.workload_config)
    doc["confound"] = {
        "threshold": recipe.confound.threshold,
        "baseline_cell": recipe.confound.baseline_cell,
    }
    doc["cells"] = resolved_cells

    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")


def resolved_cell_environment(
    cell_environment: str,
    inline_environments: tuple,
    sidecar_files: tuple[Path, ...] | None = None,
) -> dict[str, Any]:
    """Return the resolved environment descriptor for a cell.

    Centralised so both ``write_matrix_json`` and any future audit / debug
    tooling agree on the shape: registered envs project their full
    :class:`aorta.registry.Environment` descriptor; inline-docker cells emit
    the shorthand needed to re-derive the same ``_inline_<hash>``.
    """
    extra = list(sidecar_files) if sidecar_files else None
    inline_docker = {e.name: e.docker for e in inline_environments}
    if cell_environment in inline_docker:
        return {
            "name": cell_environment,
            "docker": inline_docker[cell_environment],
            "venv": None,
            "source_package": "_inline_",
            "inline": True,
        }
    env_desc = get_environment(cell_environment, extra_files=extra)
    return {
        "name": env_desc.name,
        "docker": env_desc.docker,
        "venv": env_desc.venv,
        "source_package": env_desc.source_package,
        "inline": False,
    }


__all__ = [
    "NO_TICKET_SLUG",
    "format_timestamp",
    "resolve_run_dir",
    "resolved_cell_environment",
    "safe_slug",
    "write_matrix_json",
    "write_matrix_md",
    "write_resolved_recipe",
]
