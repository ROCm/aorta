"""`aorta run` - universal workload runner.

CLI entry point for running workloads across trials, environments, and mitigations.
This is a thin wrapper around the library API in aorta.run.dispatcher.
"""

import re
from pathlib import Path

import click

from aorta.registry import RegistryError
from aorta.run.collectors import KNOWN_RECIPES
from aorta.run.dispatcher import RunRequest, run_trials

# Module-level so the regex is compiled once and the function-local
# binding doesn't trip ruff's N806 (uppercase-name-in-function).
_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@click.command()
@click.option(
    "--workload",
    required=True,
    help="Workload name (from aorta.workloads entry-point group).",
)
@click.option(
    "--trials",
    type=int,
    default=1,
    show_default=True,
    help="Number of trials to run.",
)
@click.option(
    "--environment",
    default="local",
    show_default=True,
    help="Registered environment name.",
)
@click.option(
    "--mitigations",
    default="none",
    show_default=True,
    help="Comma-separated mitigation names.",
)
@click.option(
    "--mitigations-file",
    "mitigation_files",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    multiple=True,
    help=(
        "JSON sidecar file with ad-hoc mitigations and/or environments "
        "(repeatable).  Forwarded to the registry; sidecar entries are "
        "merged with built-ins and entry-point plugins, with the same "
        "name-collision rules (B3.1)."
    ),
)
@click.option(
    "--steps",
    type=int,
    default=None,
    help="Steps per trial (workload-specific; passes through to workload config).",
)
@click.option(
    "--results-dir",
    # NOTE: do NOT pass ``writable=True`` -- Click's writable check
    # rejects paths that don't exist yet (the default ``results`` on a
    # fresh checkout), and the dispatcher creates the directory itself.
    # Letting the dispatcher's ``mkdir`` surface real I/O errors keeps
    # the failure mode consistent with ``aorta env probe``.
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("results"),
    show_default=True,
    help="Directory to write per-trial JSON.",
)
@click.option(
    "--collect",
    default="",
    help="Comma-separated collector recipe names (rocprof, numerics, amd_log). MVP: no-op.",
)
@click.option(
    "--extra-env",
    default="",
    help="Comma-separated KEY=VAL pairs for one-off env overrides (applied after mitigations).",
)
def run(
    workload: str,
    trials: int,
    environment: str,
    mitigations: str,
    mitigation_files: tuple[Path, ...],
    steps: int | None,
    results_dir: Path,
    collect: str,
    extra_env: str,
) -> None:
    """Run a workload across trials with optional mitigations.

    Examples:

        # Simple run with default settings
        aorta run --workload fsdp --trials 1

        # Multiple trials with mitigation
        aorta run --workload fsdp --trials 3 --mitigations tf32_off

        # With collector recipes (MVP: validated but no-op)
        aorta run --workload fsdp --collect rocprof,numerics

        # With extra environment variables
        aorta run --workload fsdp --extra-env DEBUG=1,VERBOSE=true
    """
    # Parse comma-separated mitigations
    mitigation_list = tuple(m.strip() for m in mitigations.split(",") if m.strip())
    if not mitigation_list:
        mitigation_list = ("none",)

    # Parse comma-separated collectors
    collect_list = tuple(c.strip() for c in collect.split(",") if c.strip())

    # Validate collector names
    invalid = set(collect_list) - KNOWN_RECIPES
    if invalid:
        raise click.ClickException(
            f"Unknown collector recipes: {sorted(invalid)}. Valid: {sorted(KNOWN_RECIPES)}"
        )

    # Parse extra_env (format: KEY=VAL,KEY2=VAL2).  We validate that
    # the key is a plausible environment variable name -- otherwise the
    # actual ``os.environ.update`` would raise the much less friendly
    # ``ValueError: illegal environment variable name`` deep inside the
    # dispatcher.
    extra_env_dict: dict[str, str] = {}
    if extra_env:
        for pair in extra_env.split(","):
            pair = pair.strip()
            if not pair:
                continue
            if "=" not in pair:
                raise click.ClickException(
                    f"Invalid extra-env format: '{pair}'. Expected KEY=VALUE."
                )
            k, v = pair.split("=", 1)
            k = k.strip()
            if not k:
                raise click.ClickException(f"Invalid extra-env entry '{pair}': key is empty.")
            if not _ENV_KEY_RE.match(k):
                raise click.ClickException(
                    f"Invalid extra-env key '{k}': must match [A-Za-z_][A-Za-z0-9_]*."
                )
            extra_env_dict[k] = v.strip()

    # Build config overrides.  ``steps`` is carried by the dedicated
    # ``RunRequest.steps`` field; do NOT also stuff it into
    # ``config_overrides`` -- that would create two copies of the same
    # value, ambiguous if a future caller ever writes only one of them.
    config_overrides: dict = {}

    # Build request
    req = RunRequest(
        workload=workload,
        trials=trials,
        environment=environment,
        mitigations=mitigation_list,
        steps=steps,
        config_overrides=config_overrides,
        results_dir=results_dir,
        collect=collect_list,
        extra_env=extra_env_dict,
        sidecar_files=tuple(mitigation_files),
    )

    # Call dispatcher.  Bridge known library exceptions to ClickException
    # so the CLI prints a clean error instead of a Python traceback:
    #
    # * ``ValueError``    -- bad ``trials`` / unknown workload / unknown
    #                        collector recipe / invalid extra_env key.
    # * ``LookupError``   -- ``UnknownEnvironmentError`` /
    #                        ``UnknownMitigationError`` from the registry
    #                        (both subclass ``KeyError``).
    # * ``RegistryError`` -- malformed sidecar / collision between
    #                        sidecar entries and built-ins / plugins.
    # * ``RuntimeError``  -- launch-mode validation failure.
    try:
        results = run_trials(req)
    except (ValueError, LookupError, RegistryError) as e:
        raise click.ClickException(str(e)) from e
    except RuntimeError as e:
        raise click.ClickException(str(e)) from e

    # Report results
    total = len(results)
    passed = sum(1 for r in results if r.exit_status == "ok")
    failed = total - passed

    if failed > 0:
        # List failed trials
        failed_trials = [r.trial_id for r in results if r.exit_status != "ok"]
        click.echo(f"Failed trials: {failed_trials}")
        raise click.ClickException(f"{failed}/{total} trials failed")

    click.echo(f"All {total} trial(s) passed. Results in: {req.results_dir / workload}")
