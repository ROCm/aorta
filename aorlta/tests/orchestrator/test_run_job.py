"""jobs/aorlta-run.sbatch, read as text: the image it builds is the one the trainer image starts from, and the
containers that load a profile or the config get the site's tool roots and site file."""

import re
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
