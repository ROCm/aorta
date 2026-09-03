"""Entry point for the `aorta` console script."""

import click

from aorta.cli._lazy_group import LazyCommand, LazyGroup

# Command name -> where the click.Command lives, and its help text. Nothing
# here is imported until the command is actually invoked; see _lazy_group.py
# for why the help text is duplicated rather than read off the command.
_COMMANDS: dict[str, LazyCommand] = {
    "agent": LazyCommand(
        "aorta.cli.agent:agent",
        "Closed-loop mitigation search via the probe engine.",
    ),
    "bench": LazyCommand(
        "aorta.cli.bench:bench",
        "Micro-benchmarks for perf characterization (hw_queue_eval).",
    ),
    "bundle": LazyCommand(
        "aorta.cli.bundle:bundle",
        "Package an ``aorta probe`` run directory into a redacted tarball.",
    ),
    "chat": LazyCommand(
        "aorta.cli.chat:chat",
        "Ask questions about the AORTA codebase (interactive REPL).",
    ),
    "env": LazyCommand(
        "aorta.cli.env:env",
        "Capture and compare GPU/library environment for trial reproducibility.",
    ),
    "environments": LazyCommand(
        "aorta.cli.environments:environments",
        "Inspect the merged environments registry (built-ins + plugins).",
    ),
    "mitigations": LazyCommand(
        "aorta.cli.mitigations:mitigations",
        "Inspect the merged mitigations registry (built-ins + plugins).",
    ),
    "run": LazyCommand(
        "aorta.cli.run:run",
        "Run a workload across N trials with optional mitigations.",
    ),
    "sweep": LazyCommand(
        "aorta.cli.sweep:sweep",
        "Unified matrix runner: sweep mitigations x {environments|diagnostics} x trials.",
    ),
    # Deprecated aliases of `sweep` (issue #248): keep working, delegate to the
    # same engine. `run` is not one of them -- #248 reserves that name for the
    # distinct single-execution command above.
    "probe": LazyCommand(
        "aorta.cli.probe:probe",
        "Run an opaque user launch command across a mitigation x diagnostic matrix.",
    ),
    "triage": LazyCommand(
        "aorta.cli.triage:triage",
        "Triage matrix runner for mitigation x environment x trials sweeps.",
    ),
}


@click.group(cls=LazyGroup, lazy_commands=_COMMANDS)
# The distribution is ``amd-aorta``; only the import package is ``aorta``. Passing
# the import name makes Click fall back to ``packages_distributions()``, which can
# report ``amd-aorta`` twice -- and then ``--version`` raises instead of printing
# (issue #429). Two layouts do it: an editable install whose build left
# ``src/amd_aorta.egg-info`` beside the site-packages ``.dist-info`` (build
# isolation, the default for pip and uv, does; ``--no-build-isolation`` may not),
# and a venv that exposes site-packages through both ``lib`` and ``lib64``, as
# ``python -m venv`` does where ``sys.platlibdir`` is ``lib64``. Naming the
# distribution skips that fallback entirely, on every layout.
@click.version_option(package_name="amd-aorta")
def main() -> None:
    """AORTA - GPU debugging platform for ROCm."""


if __name__ == "__main__":
    main()
