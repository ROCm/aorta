"""Entry point for the `aorta` console script."""

import click

from aorta.cli import agent, bench, bundle, env, environments, mitigations, probe, run, sweep, triage


@click.group()
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


main.add_command(agent.agent)
main.add_command(bench.bench)
main.add_command(bundle.bundle)
main.add_command(env.env)
main.add_command(environments.environments)
main.add_command(mitigations.mitigations)
main.add_command(sweep.sweep)
# Deprecated aliases (issue #248): keep working, delegate to the same engine.
main.add_command(probe.probe)
main.add_command(run.run)
main.add_command(triage.triage)


if __name__ == "__main__":
    main()
