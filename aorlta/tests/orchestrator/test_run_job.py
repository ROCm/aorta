"""jobs/aorlta-run.sbatch, read as text: the image it builds is the one the trainer image starts from, and the
containers that load a profile or the config get the site's tool roots and site file."""

import re
import subprocess
from pathlib import Path

PKG = Path(__file__).resolve().parents[2]
JOB = (PKG / "jobs" / "aorlta-run.sbatch").read_text()


def test_the_trainer_image_starts_from_the_tokenspeed_image_the_job_builds():
    base = re.search(r"^BASE=(\S+)$", JOB, re.M).group(1)
    assert base == "aorlta-tokenspeed:3b684074"
    assert f"FROM {base}\n" in (PKG / "docker" / "Dockerfile.trainer").read_text()


def test_the_orchestrator_and_the_checkers_get_the_tool_roots_and_the_site_file():
    checkers = JOB[JOB.index("CHK_DOCKER=("):JOB.index("GPUS=$(")]
    orchestrator = JOB[JOB.index("orchestrator() {"):JOB.index("ORCH_PID=$!")]
    for block in (checkers, orchestrator):
        for name in ("AORLTA_AORTA_ROOT", "AORLTA_ROCJITSU_ROOT", "AORLTA_LIBSTDCXX_SHIM", "AORLTA_SITE"):
            assert f"-e {name}=" in block, name
    copy = JOB.index('cp "$AORLTA_SITE" "$OUT/site.yaml"')
    assert copy < JOB.index("load_config(")  # every config the job reads has the site applied


def test_the_job_stops_before_the_run_if_the_docker_daemon_sees_a_mount_elsewhere():
    check = JOB[JOB.index("MARK=.aorlta-mount-"):JOB.index("vram_sampler >>")]
    assert JOB.index("add_mounts ORCH_MOUNTS rw") < JOB.index("MARK=.aorlta-mount-") < JOB.index("\norchestrator\n")
    assert '"${ORCH_MOUNTS[@]}"' in check and "*:ro) ;;" in check  # every read-write mount, as the run gets them
    assert "exit 95" in check and 'rm -f "$d/$MARK"' in check


DOCKER = r"""#!/bin/bash
echo "$*" >> "$DOCKER_LOG"
case "$1" in
  ps) printf '%s\n' $CONTAINERS ;;
  image) exit 1 ;;
  rm) shift; [ "$1" = -f ] && shift; printf 'removed %s\n' "$@" >> "$DOCKER_LOG" ;;
esac
exit 0
"""


def test_cleanup_removes_this_jobs_containers_and_not_another_jobs_whose_name_extends_it(tmp_path):
    funcs = "".join(re.search(rf"^{name}\(\) \{{.*?^\}}\n", JOB, re.M | re.S).group(0)
                    for name in ("add_mounts", "clean_img", "cleanup"))
    stub = tmp_path / "bin" / "docker"
    stub.parent.mkdir()
    stub.write_text(DOCKER)
    stub.chmod(0o755)
    log = tmp_path / "docker.log"
    names = "aorlta-run-123 aorlta-run-123-canary aorlta-run-1234 aorlta-run-1234-canary aorlta-run-12"
    env = {"PATH": f"{stub.parent}:/usr/bin:/bin", "DOCKER_LOG": str(log), "CONTAINERS": names}
    state = (f"CNAME=aorlta-run-123 VRAM_PID= OUT={tmp_path}/out RUN_ROOT={tmp_path}/runs SYNC_DIR={tmp_path}/shm "
             f"RUN_ID=r SOCK_DIR= RUN_DIR={tmp_path}/runs/r BUILT_TRAINER=0 BUILT_BASE=0 PULLED_CHK=0 BASE=b CHK_IMG=c")
    out = subprocess.run(["bash", "-c", f"{funcs}{state}\ncleanup"], env=env, capture_output=True, text=True,
                         check=True, timeout=60).stdout
    removed = [line.split(" ", 1)[1] for line in log.read_text().splitlines() if line.startswith("removed ")]
    assert sorted(" ".join(removed).split()) == ["aorlta-run-123", "aorlta-run-123-canary"]
    assert "containers left: 2" in out  # the stub still lists them: the count is this job's two, not five


def test_a_read_only_directory_inside_a_read_write_mount_gets_its_own_mount(tmp_path):
    funcs = re.search(r"^add_mounts\(\) \{.*?^\}\n", JOB, re.M | re.S).group(0)
    hf, pkg = tmp_path / "hf", tmp_path / "pkg"
    model = hf / "hub" / "models--org--name"
    for d in (model / "snapshots", pkg / "pools"):
        d.mkdir(parents=True)
    script = (f"{funcs}M=()\nadd_mounts M rw {hf} {pkg}\n"
              f"add_mounts M ro {model} {pkg}/pools {model}/snapshots {hf}\nprintf '%s\\n' \"${{M[@]}}\"")
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True, timeout=60)
    assert out.stdout.split() == ["-v", f"{hf}:{hf}", "-v", f"{pkg}:{pkg}", "-v", f"{model}:{model}:ro",
                                  "-v", f"{pkg}/pools:{pkg}/pools:ro"]
    assert f"{hf} is mounted rw already, not ro" in out.stderr  # docker refuses a destination twice
