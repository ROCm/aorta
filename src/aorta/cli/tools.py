"""`aorta tools` -- discover and invoke registered analysis tools.

Tools register under the ``aorta.tools`` entry-point group (built-ins in
``aorta.tools.*``; private tools from downstream packages). ``aorta tools
list`` shows the merged registry; ``aorta tools run`` invokes one over
key=value inputs and prints its verdict.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click

from aorta.tools import get_tool, load_tools


@click.group()
def tools() -> None:
    """Discover and invoke AORTA analysis tools (emulators, sanitizers, ...)."""


@tools.command(name="list")
def list_() -> None:
    """List every registered tool and its source package."""
    registry = load_tools()
    if not registry:
        click.echo("no tools registered under the 'aorta.tools' entry-point group")
        return
    name_w = max(len("NAME"), *(len(n) for n in registry))
    click.echo(f"{'NAME'.ljust(name_w)}  CLASS")
    for name in sorted(registry):
        cls = registry[name]
        click.echo(f"{name.ljust(name_w)}  {cls.__module__}.{cls.__qualname__}")


@tools.command()
@click.argument("name")
@click.option("--input", "inputs", metavar="KEY=VALUE", multiple=True,
              help="Tool input (repeatable). Comma-joined values become a list "
                   "(e.g. --input checks=waitcheck,consan).")
@click.option("--output-dir", type=click.Path(file_okay=False, path_type=Path),
              default=None, help="Directory to write the tool's report into.")
def run(name: str, inputs: tuple[str, ...], output_dir: Path | None) -> None:
    """Invoke tool NAME over the given --input KEY=VALUE pairs."""
    tool = get_tool(name)()
    parsed = _parse_inputs(inputs)
    result = tool.invoke(inputs=parsed, output_dir=output_dir)
    verdict = result.get("overall_verdict")
    if verdict is not None:
        click.echo(f"overall_verdict: {verdict}")
    else:
        click.echo(json.dumps(result.get("report"), indent=2))
    # Non-zero exit only on an explicit fail verdict (mirrors the runner).
    if verdict == "fail":
        raise SystemExit(1)


def _parse_inputs(pairs: tuple[str, ...]) -> dict[str, Any]:
    """Parse ``KEY=VALUE`` pairs; comma-joined values become a list."""
    out: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise click.BadParameter(f"expected KEY=VALUE, got {pair!r}")
        key, _, value = pair.partition("=")
        out[key.strip()] = value.split(",") if "," in value else value
    return out
