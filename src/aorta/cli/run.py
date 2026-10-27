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

import os
import re
import sys
from pathlib import Path

import click
from click.core import ParameterSource

from aorta.registry import RegistryError
from aorta.run.cli_helpers import (
    configure_verbose_logging,
    parse_csv,
    parse_extra_env,
    parse_mitigations,
    summarize_results,
)
from aorta.run.dispatcher import RunRequest, run_trials

# A workload name becomes a directory under --results-dir before the run (and
# its own validation) starts, so it has to be a plain name.
_WORKLOAD_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*")


def _forwarded_args(ctx: click.Context) -> list[str]:
    """This invocation's ``run`` options, for re-running it as the ASan child.

    Rebuilt from what Click parsed rather than sliced out of ``sys.argv``, which
    only holds the command line when ``aorta`` is the console script. Options
    left at their default are omitted; the child applies the same default.
    """

    args: list[str] = []
    for param in ctx.command.params:
        if not isinstance(param, click.Option) or param.name is None:
            continue
        if ctx.get_parameter_source(param.name) in (None, ParameterSource.DEFAULT):
            continue
        value = ctx.params[param.name]
        flag = max(param.opts, key=len)
        if param.count:
            args += [flag] * int(value)
        elif param.multiple:
            args += [f"{flag}={item}" for item in value]
        elif param.is_flag:
            if value:
                args.append(flag)
        elif value is not None:
            args.append(f"{flag}={value}")
    return args


def _child_command(run_args: list[str]) -> list[str]:
    """``aorta run <run_args>`` under this interpreter, as the console script runs it."""

    # aorta.cli is a package without __main__, so ``-m aorta.cli`` cannot run it.
    return [
        sys.executable,
        "-c",
        "import sys; from aorta.cli import main; sys.exit(main(prog_name='aorta'))",
        "run",
        *run_args,
    ]


def _run_under_asan(ctx: click.Context, workload: str, results_dir: Path) -> None:
    from aorta.instrumentation.rocjitsu_sanitizers.asan import resolve_asan_runtime
    from aorta.instrumentation.rocjitsu_sanitizers.asan_run import (
        EXIT_NOT_CHECKED,
        NodeHealthStatus,
        new_evidence_dir,
        run_under_asan,
    )
    from aorta.instrumentation.rocjitsu_sanitizers.models import Verdict

    runtime = resolve_asan_runtime()
    if runtime is None:
        raise click.ClickException(
            "--sanitizer asan needs the ASan-instrumented ROCm runtime and found none: "
            "install it (lib/asan beside the stock libraries), or point ROCM_ASAN_PATH "
            "at an ASan ROCm tree or AORTA_ASAN_ROCM_LIB at its library directory"
        )
    if not _WORKLOAD_NAME.fullmatch(workload):
        raise click.ClickException(f"--workload {workload!r} is not a workload name")
    command = _child_command(_forwarded_args(ctx))
    try:
        evidence_dir = new_evidence_dir(results_dir, (workload, "asan"))
        outcome = run_under_asan(command, evidence_dir=evidence_dir, runtime=runtime)
    except ValueError as error:
        raise click.ClickException(str(error)) from error

    for line in outcome.report.summary():
        click.echo(line, err=True)
    click.echo(f"ASan report: {outcome.report_path}", err=True)
    verdict = outcome.report.verdict
    health = outcome.report.node_health
    if verdict is Verdict.FAIL:
        raise click.ClickException("ASan found a memory error")
    if health is not None and health.status is NodeHealthStatus.FAILED:
        raise click.ClickException(
            "this node failed ASan's health check, so the workload was not run; "
            "run it on another node"
        )
    if verdict is Verdict.ERROR:
        raise click.ClickException("ASan could not check this run")
    if outcome.returncode:
        ctx.exit(outcome.returncode)
    if verdict is not Verdict.PASS:
        ctx.exit(EXIT_NOT_CHECKED)


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
    "--collect",
    default="",
    help=(
        "Comma-separated collector recipes. Validated and threaded into the "
        "workload config; the argv-wrapping profilers (rocprof, proton) attach "
        "on the subprocess flow ('aorta sweep run'), so on an in-process "
        "workload a collector only produces output if the workload consumes it."
    ),
)
@click.option(
    "--extra-env", default="", help="Comma-separated KEY=VAL pairs (applied after mitigations)."
)
@click.option(
    "--trial-isolation",
    type=click.Choice(["auto", "in_process", "process"]),
    default="auto",
    show_default=True,
    help="Trial execution policy; auto follows workload metadata.",
)
@click.option(
    "--sanitizer",
    type=click.Choice(["asan"]),
    default=None,
    help=(
        "Run the workload under a sanitizer. 'asan' first runs a small ASan "
        "canary on this node's GPUs (built with hipcc), because on a node whose "
        "GPU page migration corrupts memory no ASan result can be trusted. It "
        "then re-runs this command with the ASan-instrumented ROCm runtime "
        "(ROCM_ASAN_PATH / AORTA_ASAN_ROCM_LIB) and reports which dispatched "
        "kernels ASan actually checked. Exit status: 1 on a memory error, a lost "
        "report, a failed trial or a node that fails the canary (the workload is "
        "then not run); 3 when the run was clean but some kernels ran from device "
        "code built without ASan (PyTorch's own, for instance), or no canary could "
        "run; 0 only when every kernel was checked on a node the canary passed."
    ),
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
    trial_isolation: str,
    sanitizer: str | None,
    verbose: int,
) -> None:
    """Run a workload across N trials with optional mitigations.

    Parses CLI args, builds a ``RunRequest``, hands off to
    ``run_trials``, and maps the outcome to an exit code.  No
    orchestration logic lives here -- see ``aorta.run.dispatcher``
    and ``aorta.run.cli_helpers``.
    """
    if sanitizer == "asan":
        from aorta.instrumentation.rocjitsu_sanitizers.asan_run import ENV_ASAN_RUN

        if ENV_ASAN_RUN not in os.environ:
            _run_under_asan(click.get_current_context(), workload, results_dir)
            return
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
            trial_isolation=trial_isolation,  # type: ignore[arg-type]
            steps=steps,
            results_dir=results_dir,
            collect=parse_csv(collect),
            sidecar_files=tuple(mitigation_files),
            sanitizer=sanitizer,  # type: ignore[arg-type]
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
