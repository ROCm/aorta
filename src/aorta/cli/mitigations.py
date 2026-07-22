"""`aorta mitigations` — inspect the merged mitigations registry."""

from pathlib import Path

import click

from aorta.cli._env_display import format_env_bundle
from aorta.registry import load_mitigations


@click.group()
def mitigations() -> None:
    """Inspect the merged mitigations registry (built-ins + plugins)."""


@mitigations.command(name="list")
@click.option(
    "--file",
    "files",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    multiple=True,
    help="JSON file with extra mitigation entries to merge into the listing (repeatable).",
)
def list_(files: tuple[Path, ...]) -> None:
    """List every registered mitigation, its source package, and its env vars."""
    registry = load_mitigations(extra_files=list(files) or None)
    name_w = max(len("NAME"), *(len(n) for n in registry))
    src_w = max(len("SOURCE"), *(len(m.source_package) for m in registry.values()))

    click.echo(f"{'NAME'.ljust(name_w)}  {'SOURCE'.ljust(src_w)}  ENV")
    for name in sorted(registry):
        m = registry[name]
        env_str = format_env_bundle(m.env)
        click.echo(f"{name.ljust(name_w)}  {m.source_package.ljust(src_w)}  {env_str}")
