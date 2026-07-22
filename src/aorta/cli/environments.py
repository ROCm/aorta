"""`aorta environments` — inspect the merged environments registry."""

from pathlib import Path

import click

from aorta.registry import load_environments


@click.group()
def environments() -> None:
    """Inspect the merged environments registry (built-ins + plugins)."""


@environments.command(name="list")
@click.option(
    "--file",
    "files",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    multiple=True,
    help="JSON file with extra environment entries to merge into the listing (repeatable).",
)
def list_(files: tuple[Path, ...]) -> None:
    """List every registered environment and its baseline recipe."""
    registry = load_environments(extra_files=list(files) or None)
    name_w = max(len("NAME"), *(len(n) for n in registry))
    src_w = max(len("SOURCE"), *(len(e.source_package) for e in registry.values()))
    docker_w = max(len("DOCKER"), *(len(e.docker or "-") for e in registry.values()))
    venv_w = max(len("VENV"), *(len(e.venv or "-") for e in registry.values()))

    click.echo(
        f"{'NAME'.ljust(name_w)}  {'SOURCE'.ljust(src_w)}  "
        f"{'DOCKER'.ljust(docker_w)}  {'VENV'.ljust(venv_w)}  ENV"
    )
    for name in sorted(registry):
        e = registry[name]
        env_str = " ".join(f"{k}={v}" for k, v in sorted(e.env.items())) or "(none)"
        click.echo(
            f"{name.ljust(name_w)}  {e.source_package.ljust(src_w)}  "
            f"{(e.docker or '-').ljust(docker_w)}  "
            f"{(e.venv or '-').ljust(venv_w)}  {env_str}"
        )
