"""``aorta run`` -- universal workload runner CLI shim.

Per B1 spec (issue #148, "Python API contract"):

    The Click handler in cli/run.py becomes a thin shell: parse CLI
    args -> build RunRequest -> call run_trials() -> derive exit
    code from results.  All orchestration logic lives in
    run_trials().  No business logic in the Click handler.

Anything more than that lives in:

* ``aorta.run.dispatcher`` -- ``run_trials`` and all validation /
  workload lifecycle / persistence,
* ``aorta.run.cli_helpers`` -- pure parsers (``parse_extra_env``,
  ``parse_mitigations``) and result aggregation
  (``summarize_results``) that B2 also reuses.

This file is exception-bridging + I/O only.
"""

from pathlib import Path

import click

from aorta.registry import RegistryError
from aorta.run.cli_helpers import (
    configure_verbose_logging,
    parse_csv,
    parse_extra_env,
    parse_mitigations,
    summarize_results,
)
from aorta.run.dispatcher import RunRequest, run_trials


@click.command()
@click.option("--workload", required=True, help="Workload name (aorta.workloads entry-point).")
@click.option("--trials", type=int, default=1, show_default=True, help="Number of trials.")
@click.option(
    "--environment", default="local", show_default=True, help="Registered environment name."
)
@click.option(
    "--image",
    type=str,
    default=None,
    help=(
        "OCI image reference (typically digest-pinned, e.g. "
        "'sha256:<64-hex>' or '<repo>@sha256:<digest>') to overlay "
        "onto the resolved environment's docker field. Other axes "
        "(buck_target, venv, source_package) of the named "
        "--environment are preserved. When omitted, the named "
        "environment's existing docker value (if any) is used as-is."
    ),
)
@click.option(
    "--buck-target",
    type=str,
    default=None,
    help=(
        "Buck2 target label (e.g. '//workloads/recom_repro:recom_repro') "
        "to overlay onto the resolved environment's buck_target field. "
        "Other axes (docker, venv, source_package) of the named "
        "--environment are preserved. When omitted, the named "
        "environment's existing buck_target (if any) is used as-is."
    ),
)
@click.option(
    "--mitigations", default="none", show_default=True, help="Comma-separated mitigation names."
)
@click.option(
    "--mitigations-file",
    "mitigation_files",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    multiple=True,
    help=(
        "JSON sidecar file with ad-hoc mitigations / environments "
        "(repeatable; merged with built-ins per B3.1)."
    ),
)
@click.option("--steps", type=int, default=None, help="Steps per trial (workload-specific).")
@click.option(
    "--results-dir",
    # NOTE: do NOT pass ``writable=True`` -- Click's writable check rejects
    # paths that don't exist yet (the default ``results`` on a fresh checkout)
    # and the dispatcher creates the directory itself.
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("results"),
    show_default=True,
    help="Directory to write per-trial JSON.",
)
@click.option(
    "--collect", default="", help="Comma-separated collector recipes (MVP: validated, no-op)."
)
@click.option(
    "--extra-env", default="", help="Comma-separated KEY=VAL pairs (applied after mitigations)."
)
@click.option(
    "-v",
    "--verbose",
    count=True,
    help=(
        "Stream per-trial progress (rank 0 only) to stderr. -v = INFO "
        "(trial start/finish, timings, exit_status); -vv = DEBUG "
        "(aorta platform internals). Scope is the aorta.* logger "
        "hierarchy; workload code in sibling packages is unaffected. "
        "Default is silent: only the final pass/fail summary prints."
    ),
)
def run(
    workload: str,
    trials: int,
    environment: str,
    image: str | None,
    buck_target: str | None,
    mitigations: str,
    mitigation_files: tuple[Path, ...],
    steps: int | None,
    results_dir: Path,
    collect: str,
    extra_env: str,
    verbose: int,
) -> None:
    """Run a workload across N trials with optional mitigations.

    Parses CLI args, builds a ``RunRequest``, hands off to
    ``run_trials``, and maps the outcome to an exit code.  No
    orchestration logic lives here -- see ``aorta.run.dispatcher``
    and ``aorta.run.cli_helpers``.
    """
    configure_verbose_logging(verbose)
    try:
        req = RunRequest(
            workload=workload,
            trials=trials,
            environment=environment,
            image=image,
            buck_target=buck_target,
            mitigations=parse_mitigations(mitigations),
            extra_env=parse_extra_env(extra_env),
            steps=steps,
            results_dir=results_dir,
            collect=parse_csv(collect),
            sidecar_files=tuple(mitigation_files),
        )
        results = run_trials(req)
    except (ValueError, LookupError, RegistryError, RuntimeError) as e:
        # ValueError    -- bad trials / unknown collector recipe / invalid extra_env key.
        # LookupError   -- UnknownEnvironmentError / UnknownMitigationError (KeyError subclasses).
        # RegistryError -- malformed sidecar / collision with built-ins or plugins.
        # RuntimeError  -- launch-mode validation failure.
        raise click.ClickException(str(e)) from e

    summary = summarize_results(results)
    if summary.failed:
        click.echo(f"Failed trials: {list(summary.failed_trial_ids)}")
        raise click.ClickException(f"{summary.failed}/{summary.total} trials failed")
    click.echo(f"All {summary.total} trial(s) passed. Results in: {req.results_dir / workload}")
