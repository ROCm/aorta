"""Entry point for the `aorta` console script."""

import click

from aorta.cli import env, run, triage


@click.group()
@click.version_option(package_name="aorta")
def main() -> None:
    """AORTA - GPU debugging platform for ROCm."""


main.add_command(env.env)
main.add_command(run.run)
main.add_command(triage.triage)


if __name__ == "__main__":
    main()
