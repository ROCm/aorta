"""Entry point for the `aorta` console script."""

import click

from aorta.cli import agent, bench, bundle, env, environments, mitigations, probe, run, sweep, triage


@click.group()
# The distribution is ``amd-aorta``; only the import package is ``aorta``. Passing
# the import name makes Click fall back to ``packages_distributions()``, which
# reports ``amd-aorta`` twice under any editable install of this src layout -- the
# editable build leaves ``src/amd_aorta.egg-info`` on the path beside the
# site-packages ``.dist-info`` -- and twice again under a plain wheel install on a
# ``lib64`` distro, where ``lib64`` symlinks to ``lib`` and both land on
# ``sys.path``. ``--version`` then raises instead of printing (issue #429). Naming
# the distribution skips that fallback entirely.
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
