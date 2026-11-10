"""jobs/aorlta-reward-standin.sbatch, run by bash against a stub `docker` that records what it is asked, matches a
`--filter name=` as Docker does, by any part of a container's name, and runs a `docker run`'s command in bash with only
the variables it passes (`-e`)."""

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
  run) shift; envs=()
    while [ $# -gt 2 ]; do
      if [ "$1" = -e ]; then shift; if [[ "$1" == *=* ]]; then envs+=("$1"); else envs+=("$1=${!1-}"); fi; fi
      shift
    done
    exec env -i PATH=/usr/bin:/bin "${envs[@]}" bash "$@" ;;
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


def test_the_job_removes_its_own_temporary_directory_and_nothing_else_when_the_path_has_a_space(tmp_path):
    wt, keep = tmp_path / "wt", tmp_path / "wt" / "scratch" / "keep"
    keep.parent.mkdir(parents=True)
    keep.write_text("x")
    (wt / "scratch space").mkdir()
    stub, py = tmp_path / "bin" / "docker", tmp_path / "venv" / "bin" / "python"
    for path, body in ((stub, DOCKER), (py, '#!/bin/bash\n[ -d "$TMPDIR" ] && echo "$TMPDIR" > "$AORLTA_T11_OUT/tmp"\n'
                                          'exit 0\n')):
        path.parent.mkdir(parents=True)
        path.write_text(body)
        path.chmod(0o755)
    for d in ("asan", "cwd"):
        (tmp_path / d).mkdir()
    env = {"PATH": f"{stub.parent}:/usr/bin:/bin", "HOME": str(tmp_path), "DOCKER_LOG": str(tmp_path / "docker.log"),
           "SLURM_JOB_ID": "123", "SLURM_SUBMIT_DIR": str(wt), "AORLTA_OUT": str(tmp_path / "out"),
           "AORLTA_VENV": str(tmp_path / "venv"), "AORLTA_ASAN_LIB": str(tmp_path / "asan"),
           "AORLTA_TMPDIR": str(wt / "scratch space")}
    r = subprocess.run(["bash", "-c", JOB], env=env, cwd=tmp_path / "cwd", capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert keep.exists()
    job_tmp = wt / "scratch space" / "aorlta-reward-standin-123"
    assert (tmp_path / "out" / "tmp").read_text() == f"{job_tmp}\n"  # pytest ran with the job's own TMPDIR in place
    assert not job_tmp.exists()
