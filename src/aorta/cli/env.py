"""``aorta env`` - environment capture for trial reproducibility.

The probe writes a versioned, schema-stable JSON snapshot of everything
needed to interpret a trial result without re-running the workload:

* ``system_health`` - verbatim output of ``rdhc --quick --json`` (or null
  when RDHC / sudo is unavailable).
* ``rocm`` - explicit reads of ``/opt/rocm/.info/version{,_dev}`` and
  ``/sys/module/amdgpu/version``.
* ``hip`` - ``hipconfig`` outputs (toolchain build state).
* ``hipblaslt`` - commit + library hash + Tensile YAML revision +
  applied-PR flags. Catches GEMM kernel library drift across docker
  images / conda envs / venvs.
* ``runtime_context`` - which container runtime + Python env we are in.
* ``docker`` - image + digest when running inside a container.
* ``env_vars`` - canonical list of HSA / RCCL / FBGEMM / PyTorch env vars.
* ``python_version``, ``pytorch_version``.

Implementation lives in :mod:`aorta.instrumentation.environment`. This
module just wires it into the Click CLI.
"""

from __future__ import annotations

import logging
from pathlib import Path

import click

log = logging.getLogger(__name__)


@click.group()
def env() -> None:
    """Capture and compare GPU/library environment for trial reproducibility."""


@env.command()
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    default=Path("env.json"),
    show_default=True,
    help="Path to write env.json.",
)
def probe(output: Path) -> None:
    """Capture trial-environment state to env.json (issue #147)."""
    from aorta.instrumentation.environment import capture_environment

    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    snapshot = capture_environment(output)
    _print_brief(output, snapshot)


def _print_brief(output: Path, snapshot: dict) -> None:
    """Print a short, human-friendly summary of the captured env."""
    click.echo(f"Wrote env probe to {output} (schema_version={snapshot.get('schema_version')})")
    rt = snapshot.get("runtime_context") or {}
    rocm = snapshot.get("rocm") or {}
    hip = snapshot.get("hip") or {}
    hipblaslt = snapshot.get("hipblaslt") or {}
    sysh = snapshot.get("system_health")
    click.echo(f"  runtime:  {rt.get('type', '?')} / python={rt.get('python_env', '?')}")
    click.echo(f"  rocm:     {rocm.get('version', '?')} (dev: {rocm.get('version_dev', '?')})")
    click.echo(f"  hip:      {hip.get('version', '?')} ({hip.get('platform', '?')})")
    click.echo(f"  hipblaslt: commit={hipblaslt.get('commit', '?')}")
    click.echo(f"  rdhc:     {'present' if sysh else 'unavailable (system_health=null)'}")
    click.echo(f"  python:   {snapshot.get('python_version', '?')}")
    click.echo(f"  pytorch:  {snapshot.get('pytorch_version', '?')}")
