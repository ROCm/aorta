"""jobs/aorlta-reward-standin.sbatch's cleanup, run by bash against a stub `docker` that records what it is asked and
matches a `--filter name=` as Docker does, by any part of a container's name."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

JOB = (Path(__file__).resolve().parents[2] / "jobs" / "aorlta-reward-standin.sbatch").read_text()
DOCKER = r"""#!/bin/bash
echo "$*" >> "$DOCKER_LOG"
case "$1" in
  ps) for n in $CONTAINERS; do [[ "$*" == *"name=$CNAME"* && "$n" == *"$CNAME"* ]] && echo "$n"; done ;;
  rm) shift; [ "$1" = -f ] && shift; printf 'removed %s\n' "$@" >> "$DOCKER_LOG" ;;
esac
exit 0
"""


def test_cleanup_removes_its_own_container_and_not_another_jobs_whose_name_extends_it(tmp_path):
    cleanup = re.search(r"^cleanup\(\) \{\n.*?^\}\n", JOB, re.M | re.S).group(0)
    stub = tmp_path / "bin" / "docker"
    stub.parent.mkdir()
    stub.write_text(DOCKER)
    stub.chmod(0o755)
    log = tmp_path / "docker.log"
    env = {"PATH": f"{stub.parent}:/usr/bin:/bin", "DOCKER_LOG": str(log), "CNAME": "aorlta-reward-standin-123",
           "CONTAINERS": "aorlta-reward-standin-123 aorlta-reward-standin-1234"}
    subprocess.run(["bash", "-c", f"{cleanup}PULLED=0\ncleanup"], env=env, check=True, timeout=60)
    removed = [line.split(" ", 1)[1] for line in log.read_text().splitlines() if line.startswith("removed ")]
    assert removed == ["aorlta-reward-standin-123"]
