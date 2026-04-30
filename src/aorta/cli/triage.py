"""`aorta triage` - mitigation matrix runner. (Optimize mode deferred to P1 per D11.)"""

import click


@click.group()
def triage() -> None:
    """Triage matrix runner for mitigation x docker x trials sweeps."""


@triage.command(name="run")
@click.option(
    "--mode",
    type=click.Choice(["matrix"]),
    default="matrix",
    show_default=True,
    help="matrix = full contingency table. 'optimize' deferred to P1 per D11.",
)
@click.option("--workload", required=True, help="Workload name.")
@click.option(
    "--mitigations",
    required=True,
    help="Comma-separated mitigation names. Include 'none' for baseline.",
)
@click.option(
    "--dockers",
    required=True,
    help="Comma-separated docker image names.",
)
@click.option(
    "--trials",
    type=int,
    default=8,
    show_default=True,
    help="Trials per cell.",
)
@click.option(
    "--steps",
    type=int,
    default=None,
    help="Steps per trial.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, writable=True),
    default="triage_results",
    show_default=True,
    help="Directory for matrix.json + matrix.md output.",
)
def triage_run(
    mode: str,
    workload: str,
    mitigations: str,
    dockers: str,
    trials: int,
    steps: int | None,
    output_dir: str,
) -> None:
    """Run the triage matrix: sweep mitigations x dockers x trials, output contingency table.

    Implemented by task B2 (triage/runner.py + triage/matrix.py).
    """
    raise click.ClickException("aorta triage run - not yet implemented (task B2)")
