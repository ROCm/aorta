"""jobs/aorlta-run.sbatch, read as text: the image it builds is the one the trainer image starts from, and the
containers that load a profile or the config get the site's tool roots and site file."""

import os
import re
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

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
                    for name in ("own_containers", "add_mounts", "clean_img", "cleanup"))
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


def test_an_adapter_inside_the_hugging_face_cache_is_mounted_read_only(tmp_path):
    funcs = re.search(r"^add_mounts\(\) \{.*?^\}\n", JOB, re.M | re.S).group(0)
    mounting = JOB[JOB.index("ORCH_MOUNTS=()\n"):JOB.index("# A directory the docker daemon sees")]
    hf, pkg, runs = tmp_path / "hf", tmp_path / "pkg", tmp_path / "runs"
    model, adapter = hf / "hub" / "models--org--base", hf / "hub" / "models--org--adapter"
    for d in (model, adapter, pkg, runs):
        d.mkdir(parents=True)
    state = (f"P={pkg} RUN_ROOT={runs} MODEL_DIR={model} AORLTA_HF_HOME={hf} AORLTA_TMPDIR= ADAPTER_DIR={adapter} "
             "AORLTA_AORTA_ROOT= AORLTA_ROCJITSU_ROOT= AORLTA_LIBSTDCXX_SHIM=\nPOOL_DIRS=()\nEXTRA_MOUNTS=()\n")
    out = subprocess.run(["bash", "-c", f"{funcs}{state}{mounting}printf '%s\\n' \"${{ORCH_MOUNTS[@]}}\""],
                         capture_output=True, text=True, check=True, timeout=60).stdout.split()
    assert f"{hf}:{hf}" in out and f"{adapter}:{adapter}:ro" in out


# --- two jobs on one node -------------------------------------------------------------------------------------

SQUEUE = r"""#!/bin/bash
# squeue -h -j ID -o %T: the state $JOBS names for ID ("ID:STATE ..."), Slurm's error for a job it does not know, or
# the controller's error when $SQUEUE_DOWN is set; squeue -h -w NODE -o %i: the jobs in $NODE_JOBS
[ -n "${SQUEUE_DOWN:-}" ] && { echo "slurm_load_jobs error: Unable to contact slurm controller" >&2; exit 1; }
case "$*" in
  *" -w "*) printf '%s\n' $NODE_JOBS; exit 0 ;;
esac
id=$3
for j in $JOBS; do
  [ "${j%%:*}" = "$id" ] && { echo "${j#*:}"; exit 0; }
done
echo "slurm_load_jobs error: Invalid job id specified" >&2
exit 1
"""
DOCKER2 = r"""#!/bin/bash
# docker ps -a [--filter ancestor=IMG] --format ...: $CONTAINERS, or the containers $USERS_OF_IMAGE names for IMG;
# docker rmi / rm: logged; docker run ... -c CMD: CMD run here (the cleanup's file work); docker image inspect: found,
# or with $IMAGES_DIR set, found while a file of the image's name is there (rmi removes it)
echo "$*" >> "$DOCKER_LOG"
case "$1" in
  ps) if [ "$3" = --filter ]; then printf '%s\n' $USERS_OF_IMAGE; else printf '%s\n' $CONTAINERS; fi ;;
  rm|rmi) shift; [ "$1" = -f ] && shift; printf '%s %s\n' "removed" "$*" >> "$DOCKER_LOG"
     [ -n "${IMAGES_DIR:-}" ] && rm -f "$IMAGES_DIR/$(printf %s "$*" | tr '/:' '__')" ;;
  image) [ -z "${IMAGES_DIR:-}" ] || [ -e "$IMAGES_DIR/$(printf %s "$3" | tr '/:' '__')" ]; exit ;;
  run) while [ "$#" -gt 0 ] && [ "$1" != -c ]; do shift; done; [ "$1" = -c ] && bash -c "$2" ;;
esac
exit 0
"""


def _functions(*names: str) -> str:
    return "".join(re.search(rf"^{name}\(\) \{{.*?^\}}\n", JOB, re.M | re.S).group(0) for name in names)


def _bash(tmp_path, script: str, **env) -> subprocess.CompletedProcess:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    for name, text in (("squeue", SQUEUE), ("docker", DOCKER2)):
        (bin_dir / name).write_text(text)
        (bin_dir / name).chmod(0o755)
    base = {"PATH": f"{bin_dir}:/usr/bin:/bin", "DOCKER_LOG": str(tmp_path / "docker.log"), "SLURM_JOB_ID": "123",
            "SLURMD_NODENAME": "node1", "JOBS": "", "NODE_JOBS": "", "CONTAINERS": "", "USERS_OF_IMAGE": ""}
    return subprocess.run([shutil.which("bash") or "/bin/bash", "-c", script], env={**base, **env}, capture_output=True,
                          text=True, timeout=60)


def _removed(tmp_path) -> list[str]:
    log = tmp_path / "docker.log"
    lines = log.read_text().splitlines() if log.exists() else []
    return sorted(" ".join(x.split(" ", 1)[1] for x in lines if x.startswith("removed ")).split())


@pytest.mark.parametrize("state, ended", [("", True), ("RUNNING", False), ("COMPLETING", False), ("PENDING", False),
                                          ("SUSPENDED", False), ("COMPLETED", True), ("CANCELLED", True),
                                          ("FAILED", True), ("TIMEOUT", True), ("NODE_FAIL", True),
                                          ("SOMETHING_NEW", False)])
def test_a_job_has_ended_only_when_slurm_says_so(tmp_path, state, ended):
    out = _bash(tmp_path, f"{_functions('job_ended')}job_ended 77 && echo ended || echo running",
                JOBS=f"77:{state}" if state else "77:")
    assert out.stdout.strip() == ("ended" if ended else "running"), out.stderr


def test_a_job_slurm_does_not_know_has_ended_and_no_answer_from_slurm_is_running(tmp_path):
    script = f"{_functions('job_ended')}job_ended 77 && echo ended || echo running"
    assert _bash(tmp_path, script).stdout.strip() == "ended"
    assert _bash(tmp_path, script, SQUEUE_DOWN="1").stdout.strip() == "running"
    tools = tmp_path / "tools"  # no squeue at all, the host's included: a PATH with grep alone
    tools.mkdir()
    (tools / "grep").symlink_to(shutil.which("grep"))
    assert _bash(tmp_path, script, PATH=str(tools)).stdout.strip() == "running"


def test_only_the_containers_of_jobs_slurm_no_longer_runs_are_stale(tmp_path):
    names = ("aorlta-run-100 aorlta-run-100-checkers aorlta-run-200 aorlta-run-200-canary aorlta-run-300 "
             "aorlta-run-400-mounts aorlta-run-123 aorlta-run-123-imports aorlta-run-12x other-1 aorlta-run-1000")
    out = _bash(tmp_path, f"{_functions('job_ended', 'stale_containers')}stale_containers", CONTAINERS=names,
                JOBS="100:RUNNING 300:COMPLETED 400:COMPLETING 1000:RUNNING")
    # 200 is unknown to Slurm (ended long ago), 300 completed; 100, 400 and 1000 run, 123 is this job
    assert sorted(out.stdout.split()) == ["aorlta-run-200", "aorlta-run-200-canary", "aorlta-run-300"], out.stderr
    down = _bash(tmp_path, f"{_functions('job_ended', 'stale_containers')}stale_containers", CONTAINERS=names,
                 SQUEUE_DOWN="1")
    assert down.stdout.split() == []  # no answer from Slurm: nothing is taken for a left-over
    calls = (tmp_path / "bin").exists()
    assert calls


def test_the_job_removes_stale_containers_after_its_claims_and_before_its_first_container():
    assert JOB.index('claim_dir "$SYNC_DIR/$RUN_ID"') < JOB.index("stale=$(stale_containers)") < JOB.index(
        '--name "$CNAME-canary"')
    assert JOB.index("trap cleanup EXIT") < JOB.index('claim_dir "$SYNC_DIR/$RUN_ID"')
    assert "aorlta-run-[0-9]+(-[a-z]+)?')\n" not in JOB[JOB.index("stale=$(stale_containers)"):]  # no blanket removal


@pytest.mark.parametrize("users, node_jobs, down, removed", [
    ("", "", False, True),
    ("aorlta-run-123 aorlta-run-123-checkers", "", False, True),  # this job's own containers do not count
    ("aorlta-run-456", "", False, False),  # another job's container was made from it
    ("", "456", False, False),  # another job runs on the node and may be about to use it
    ("", "", True, False),  # Slurm cannot say who else runs here
])
def test_an_image_this_job_built_is_removed_only_when_no_other_job_may_use_it(tmp_path, users, node_jobs, down,
                                                                               removed):
    env = {"USERS_OF_IMAGE": users, "NODE_JOBS": f"123 {node_jobs}".strip()}
    if down:
        env["SQUEUE_DOWN"] = "1"
    out = _bash(tmp_path, f"{_functions('other_jobs', 'remove_image')}CNAME=aorlta-run-123\nremove_image img:1", **env)
    assert _removed(tmp_path) == (["img:1"] if removed else []), out.stdout + out.stderr
    assert ("removed img:1" if removed else "kept img:1: another job may use it") in out.stdout


def _claim_script(tmp_path, body: str) -> str:
    return (f"{_functions('stamp', 'job_ended', 'claim', 'canon_dir', 'dir_key', 'claim_dir')}CLAIMS={tmp_path}/claims\n"
            f"CLAIMED=()\n{body}\n"
            "echo \"claimed ${#CLAIMED[@]}\"")


def test_a_claim_is_taken_kept_by_a_running_job_and_taken_over_from_an_ended_one(tmp_path):
    claims = tmp_path / "claims"
    out = _bash(tmp_path, _claim_script(tmp_path, 'claim "GPU 0" gpu0\nclaim "GPU 0" gpu0'))
    assert out.returncode == 0 and "claimed 2" in out.stdout and (claims / "gpu0").read_text() == "123 GPU 0\n"
    (claims / "gpu1").write_text("456 GPU 1\n")
    held = _bash(tmp_path, _claim_script(tmp_path, 'claim "GPU 1" gpu1'), SLURM_JOB_ID="789", JOBS="456:RUNNING")
    assert held.returncode == 89 and "job 456, which Slurm still runs" in held.stderr and "GPU 1" in held.stderr
    assert (claims / "gpu1").read_text() == "456 GPU 1\n"
    down = _bash(tmp_path, _claim_script(tmp_path, 'claim "GPU 1" gpu1'), SLURM_JOB_ID="789", SQUEUE_DOWN="1")
    assert down.returncode == 89 and (claims / "gpu1").read_text() == "456 GPU 1\n"
    over = _bash(tmp_path, _claim_script(tmp_path, 'claim "GPU 1" gpu1'), SLURM_JOB_ID="789", JOBS="456:CANCELLED")
    assert over.returncode == 0 and "taking over gpu1 from job 456" in over.stdout
    assert (claims / "gpu1").read_text() == "789 GPU 1\n"


def test_a_directory_claim_key_is_one_per_path(tmp_path):
    real = tmp_path / "shm" / "a"
    real.mkdir(parents=True)
    (tmp_path / "alias").symlink_to(tmp_path / "shm")
    spellings = [real, f"{real}/", f"{tmp_path}/shm/./a", f"{tmp_path}/shm/b/../a", tmp_path / "alias" / "a"]
    out = _bash(tmp_path, f"{_functions('canon_dir', 'dir_key')}" + "".join(f"dir_key '{p}'; " for p in spellings)
                + f"dir_key '{tmp_path}/shm/b'; dir_key '{tmp_path}/shm/not-yet/c/'; dir_key '{tmp_path}/shm/not-yet/c'")
    keys = out.stdout.split()
    assert len(set(keys[:5])) == 1 and keys[5] != keys[0] and keys[6] == keys[7], out.stderr
    assert keys[0].startswith("dir-") and len(keys[0]) == 20


@pytest.mark.parametrize("mine, clash", [("/x/shm/checkers", True), ("/x/shm/checkers/", True),
                                         ("/x/shm/checkers/run-b", True), ("/x/shm", True),
                                         ("/x/shm/checkers2", False), ("/x/shm/check", False)])
def test_a_directory_inside_or_around_a_live_jobs_directory_is_refused(tmp_path, mine, clash):
    claims = tmp_path / "claims"
    claims.mkdir()
    (claims / "dir-0123456789abcdef").write_text("456 directory /x/shm/checkers\n")
    out = _bash(tmp_path, _claim_script(tmp_path, f'claim_dir "{mine}"'), SLURM_JOB_ID="789", JOBS="456:RUNNING")
    assert out.returncode == (89 if clash else 0), out.stdout + out.stderr
    if clash:
        assert "job 456" in out.stderr and "overlaps" in out.stderr
    ended = _bash(tmp_path, _claim_script(tmp_path, f'claim_dir "{mine}"'), SLURM_JOB_ID="789", JOBS="456:COMPLETED")
    assert ended.returncode == 0, ended.stderr
    own = _bash(tmp_path, _claim_script(tmp_path, f'claim_dir "{mine}"'), SLURM_JOB_ID="456", JOBS="456:RUNNING")
    assert own.returncode == 0, own.stderr  # a job's own claims never stop it


@pytest.mark.parametrize("mine, refused", [("claims", True), ("claims/", True), ("claims/sockets", True),
                                           ("alias/claims/x", True), (".", True), ("claims2", False),
                                           ("shm/sockets", False)])
def test_a_directory_inside_or_around_the_claims_directory_is_refused(tmp_path, mine, refused):
    """The canary and the cleanup remove a claimed directory recursively: one that is the claims directory, inside
    it or around it (through a symlink too) would take every job's claims and the lock with it."""
    (tmp_path / "claims").mkdir()
    (tmp_path / "claims" / "gpu0").write_text("456 GPU 0\n")
    (tmp_path / "alias").symlink_to(tmp_path)
    out = _bash(tmp_path, _claim_script(tmp_path, f'claim_dir "{tmp_path}/{mine}"'), SLURM_JOB_ID="789",
                JOBS="456:RUNNING")
    assert out.returncode == (95 if refused else 0), out.stdout + out.stderr
    dirs = [p for p in (tmp_path / "claims").iterdir() if p.name.startswith("dir-")]
    assert len(dirs) == (0 if refused else 1) and (tmp_path / "claims" / "gpu0").read_text() == "456 GPU 0\n"
    if refused:
        assert "overlaps the claims directory" in out.stderr


def test_the_socket_directory_is_canonical_before_it_is_claimed_emptied_or_removed(tmp_path):
    """Through a symlink, the claim named the target while the canary removed the link and made a directory, nobody's
    claim, in its place: the job takes the canonical path once, and the claim, the canary and the cleanup use it."""
    assert JOB.count("REMOTE[0]") == 1
    take = 'SOCK_DIR=$(canon_dir "${REMOTE[0]}")'
    assert JOB.index(take) < JOB.index('claim_dir "$SOCK_DIR"') < JOB.index("rm -rf '$SOCK_DIR'; mkdir -p")
    (tmp_path / "shm" / "real").mkdir(parents=True)
    (tmp_path / "shm" / "link").symlink_to(tmp_path / "shm" / "real")
    out = _bash(tmp_path, f"{_functions('canon_dir')}REMOTE=({tmp_path}/shm/link/ x)\n"
                f'[ "${{#REMOTE[@]}}" -gt 0 ] && {take}\necho "$SOCK_DIR"')
    assert out.stdout.strip() == str(tmp_path / "shm" / "real"), out.stderr


def test_claims_and_kept_image_records_are_world_writable_so_any_users_later_job_takes_them_over(tmp_path):
    """The claims directory is sticky (only a file's owner removes it), so a claim another user's job left must be
    writable for a later job to take it over; one that is not is refused with who owns it."""
    out = _bash(tmp_path, "umask 022\n" + _claim_script(tmp_path, 'claim "GPU 0" gpu0\nclaim_dir /x/shm/r'))
    assert out.returncode == 0, out.stderr
    for f in (tmp_path / "claims").iterdir():
        if f.name != ".lock":
            assert stat.S_IMODE(f.stat().st_mode) == 0o666, f.name
    kept = _bash(tmp_path, f"umask 022\n{_functions('other_jobs', 'remove_image')}CNAME=aorlta-run-123\n"
                 f"CLAIMS={tmp_path}/claims\nremove_image img:1", NODE_JOBS="123 456")
    rec, = (tmp_path / "claims" / "images").iterdir()
    assert "kept img:1" in kept.stdout and stat.S_IMODE(rec.stat().st_mode) == 0o666
    (tmp_path / "claims" / "gpu1").write_text("456 GPU 1\n")
    (tmp_path / "claims" / "gpu1").chmod(0o444)  # as another user's 0644 claim is to this job
    if os.access(tmp_path / "claims" / "gpu1", os.W_OK):
        pytest.skip("running as a user who can write any file")
    over = _bash(tmp_path, _claim_script(tmp_path, 'claim "GPU 1" gpu1'), SLURM_JOB_ID="789", JOBS="456:CANCELLED")
    assert over.returncode == 89 and "cannot write" in over.stderr and "left by job 456" in over.stderr, over.stderr
    assert (tmp_path / "claims" / "gpu1").read_text() == "456 GPU 1\n"


def _cleanup(tmp_path, own: int, built: int = 1, calls: str = "cleanup", **env) -> subprocess.CompletedProcess:
    funcs = _functions("job_ended", "own_containers", "add_mounts", "clean_img", "other_jobs", "remove_image",
                       "retry_kept_images")
    funcs += re.search(r"^CLEANED=0\ncleanup\(\) \{.*?^\}\n", JOB, re.M | re.S).group(0)
    state = (f"CNAME=aorlta-run-{env.get('SLURM_JOB_ID', '123')} VRAM_PID= OUT={tmp_path}/out RUN_ROOT={tmp_path}/runs "
             f"SYNC_DIR={tmp_path}/shm RUN_ID=r SOCK_DIR={tmp_path}/shm/sockets RUN_DIR={tmp_path}/runs/r "
             f"BUILT_TRAINER={built} BUILT_BASE=0 PULLED_CHK=0 BASE=b CHK_IMG=c IMG=trainer:1 OWN_DIRS={own} "
             f"CLAIMS={tmp_path}/claims\nCLAIMED=({tmp_path}/claims/gpu0 {tmp_path}/claims/gpu1)\n")
    return _bash(tmp_path, f"{funcs}{state}{calls}", **env)


def test_cleanup_removes_the_runs_directories_only_when_this_job_claimed_them(tmp_path):
    for d in ("out", "runs", "shm/r/scratch", "shm/sockets", "claims"):
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
    (tmp_path / "claims" / "gpu0").write_text("123 GPU 0\n")
    (tmp_path / "claims" / "gpu1").write_text("456 GPU 1\n")  # taken over by another job since: not ours to drop
    out = _cleanup(tmp_path, 0, NODE_JOBS="123 456")
    assert (tmp_path / "shm" / "r").is_dir() and (tmp_path / "shm" / "sockets").is_dir(), out.stderr
    assert not (tmp_path / "claims" / "gpu0").exists() and (tmp_path / "claims" / "gpu1").exists()
    assert "kept trainer:1" in out.stdout and _removed(tmp_path) == []  # job 456 still runs on the node
    out = _cleanup(tmp_path, 1, NODE_JOBS="123")
    assert not (tmp_path / "shm" / "r").exists() and not (tmp_path / "shm" / "sockets").exists(), out.stderr
    assert _removed(tmp_path) == ["trainer:1"] and "removed trainer:1" in out.stdout


def test_the_job_claims_every_gpu_its_config_uses(tmp_path):
    start = 'read -ra RUN_GPUS < <(PYTHONPATH="$CODE/src$XPATH" "$PY" -c "'
    snippet = JOB[JOB.index(start) + len(start):JOB.index('") || { echo "[guard] $CONFIG does not load"',
                                                          JOB.index("RUN_GPUS"))]
    import sys

    def gpus(config: Path, **edits) -> list[int]:
        text = snippet.replace("$CODE/$CONFIG", str(config))
        prelude = "import sys; sys.path.insert(0, %r)\n" % str(PKG / "src")
        out = subprocess.run([sys.executable, "-c", prelude + text], capture_output=True, text=True, timeout=120,
                             cwd=PKG)
        assert out.returncode == 0, out.stderr
        return [int(g) for g in out.stdout.split()]

    assert gpus(PKG / "configs" / "standin.yaml") == list(range(8))  # 4 engines and their ranks, checkers on 4-7
    assert gpus(PKG / "configs" / "arith-score.yaml") == [0]
    devices = tmp_path / "devices.yaml"
    devices.write_text((PKG / "configs" / "arith-train.yaml").read_text().replace("  ranks: 1", "  ranks: 1\n  devices: [3]"))
    assert gpus(devices) == [0, 3]


def _aorta_modules_snippet() -> str:
    """The Python the job runs on the host to pick the aorta modules of the import pre-flight."""
    block = JOB[JOB.index("AORTA_MODULES=$("):]
    return block[block.index('-c "') + 4:block.index('") ||')]


def test_the_import_pre_flight_loads_aorta_only_for_a_profile_that_uses_it(tmp_path, monkeypatch):
    """A task whose profile names no aorta adapter starts without an aorta checkout; a stand-in profile still has
    its waitcheck backend imported in the run image, and a missing checkout still stops it there."""
    import json
    import sys

    sys.path.insert(0, str(PKG / "tests"))
    import test_plugins as TP

    from aorlta import plugins
    from aorlta.reward.profiles import aorta_preflight_modules, load_profile

    assert aorta_preflight_modules(load_profile("standin-static")) == ("models", "waitcheck")
    assert aorta_preflight_modules(load_profile("standin-gpu")) == ("models", "waitcheck")
    assert aorta_preflight_modules(load_profile("fake")) == ()
    saved = {id(t): dict(t) for t in plugins._TABLES}
    try:
        for t in plugins._TABLES:
            t.clear()
        plugins.LOADED.clear()
        d = tmp_path / "toyplugin"
        d.mkdir()
        (d / "toy_plugin.py").write_text(TP.PLUGIN)
        (d / "toy-profile.yaml").write_text(TP.PROFILE)
        (tmp_path / "pool").mkdir()
        toy = TP.toy_config(tmp_path, str(d / "toy_plugin.py"))
        standin = TP.toy_config(tmp_path, str(d / "toy_plugin.py"))
        standin["reward"] = {"profile": "standin-static", "reward_fn": "standin.v11"}
        picked = {}
        for name, data in (("toy", toy), ("standin", standin)):
            (tmp_path / f"{name}.yaml").write_text(json.dumps(data))
            code = _aorta_modules_snippet().replace("$CODE/$CONFIG", str(tmp_path / f"{name}.yaml"))
            out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                                 env={"PYTHONPATH": str(PKG / "src"), "PATH": "/usr/bin:/bin"})
            assert out.returncode == 0, out.stderr
            picked[name] = out.stdout.split()
        assert picked == {"toy": [], "standin": ["models", "waitcheck"]}
    finally:
        for t in plugins._TABLES:
            t.clear()
            t.update(saved[id(t)])
    check = [sys.executable, str(PKG / "jobs" / "aorlta-imports-check.py"), str(tmp_path / "no-aorta" / "src")]
    env = {"PYTHONPATH": str(PKG / "src"), "PATH": "/usr/bin:/bin"}
    for extra, aorta in (([], []), (["models", "waitcheck"], ["aorta.models", "aorta.waitcheck"])):
        report = json.loads(subprocess.run(check + extra, capture_output=True, text=True, env=env).stdout)
        assert report["aorta"] == extra
        assert sorted(k for k in report["failed"] if k.startswith("aorta.")) == aorta


@pytest.mark.parametrize("stuck", [True, False])
def test_a_container_that_will_not_go_keeps_the_jobs_images_claims_and_directories(tmp_path, stuck):
    """The leaving marker tells another job's cleanup this job uses no image, and a dropped claim frees its GPUs and
    directories: neither may happen while one of this job's containers is still there."""
    images = tmp_path / "images"
    images.mkdir()
    for name in ("trainer_1", "b"):  # b: the image the cleanup's helper runs from
        (images / name).write_text("")
    for d in ("out", "runs", "shm/r", "shm/sockets", "claims"):
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
    (tmp_path / "claims" / "gpu0").write_text("123 GPU 0\n")
    names = "aorlta-run-123-checkers aorlta-run-1234" if stuck else "aorlta-run-1234"  # the stub never removes one
    out = _cleanup(tmp_path, 1, NODE_JOBS="123", JOBS="123:RUNNING", CONTAINERS=names, IMAGES_DIR=str(images))
    kept = [(tmp_path / "claims" / "leaving-123").exists(), (tmp_path / "claims" / "gpu0").exists(),
            (images / "trainer_1").exists(), (tmp_path / "shm" / "r").exists(), (tmp_path / "shm" / "sockets").exists()]
    if stuck:
        assert kept == [False, True, True, True, True], out.stdout + out.stderr
        assert "[guard] cannot remove this job's containers: aorlta-run-123-checkers" in out.stderr
        assert "containers left: 1" in out.stdout
    else:
        assert kept == [True, False, False, False, False], out.stdout + out.stderr
        assert "[guard]" not in out.stderr and "containers left: 0" in out.stdout


def test_cleanup_runs_once_so_a_second_pass_never_removes_directories_claimed_since(tmp_path):
    for d in ("out", "runs", "shm/r", "shm/sockets", "claims"):
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
    # the TERM trap's cleanup, a job taking the released directories, then the EXIT trap's cleanup
    calls = (f"cleanup\nmkdir -p {tmp_path}/shm/r {tmp_path}/shm/sockets\n"
             f"echo '456 directory {tmp_path}/shm/r' > {tmp_path}/claims/dir-r\ncleanup")
    out = _cleanup(tmp_path, 1, calls=calls, NODE_JOBS="123")
    assert (tmp_path / "shm" / "r").is_dir() and (tmp_path / "shm" / "sockets").is_dir(), out.stdout + out.stderr
    assert out.stdout.count("containers left:") == 1
    assert JOB.index("CLEANED=0\ncleanup() {") < JOB.index("trap cleanup EXIT")
    assert "trap 'cleanup; exit 143' TERM INT" in JOB


def test_an_image_kept_for_another_job_is_removed_by_the_last_job_to_leave(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    for name in ("trainer_1", "older_1"):
        (images / name).write_text("")  # older:1 was on the node before either job: no job recorded it
    for d in ("out", "runs", "claims"):
        (tmp_path / d).mkdir(exist_ok=True)
    first = _cleanup(tmp_path, 0, NODE_JOBS="123 456", USERS_OF_IMAGE="aorlta-run-456", IMAGES_DIR=str(images))
    assert "kept trainer:1" in first.stdout and _removed(tmp_path) == [], first.stdout + first.stderr
    assert [p.read_text() for p in (tmp_path / "claims" / "images").iterdir()] == ["trainer:1\n"]
    # job 456 found the image and did not build it; it leaves last
    last = _cleanup(tmp_path, 0, built=0, SLURM_JOB_ID="456", NODE_JOBS="456", IMAGES_DIR=str(images))
    assert _removed(tmp_path) == ["trainer:1"] and "removed trainer:1" in last.stdout, last.stdout + last.stderr
    assert list((tmp_path / "claims" / "images").iterdir()) == [] and (images / "older_1").exists()
    # a record whose image is gone already is dropped, and nothing is removed for it
    (tmp_path / "claims" / "images" / "x").write_text("gone:1\n")
    _cleanup(tmp_path, 0, built=0, SLURM_JOB_ID="789", NODE_JOBS="789", IMAGES_DIR=str(images))
    assert list((tmp_path / "claims" / "images").iterdir()) == [] and _removed(tmp_path) == ["trainer:1"]


def test_of_two_jobs_cleaning_up_at_once_the_one_that_decides_last_removes_the_image(tmp_path):
    """Both jobs still run in Slurm's eyes while they clean up: each must not keep the image for the other, which is
    leaving too, or neither removes it. A job in its cleanup marks itself leaving before its first image decision."""
    images = tmp_path / "images"
    images.mkdir()
    (images / "trainer_1").write_text("")
    for d in ("out", "runs", "claims"):
        (tmp_path / d).mkdir(exist_ok=True)
    first = _cleanup(tmp_path, 0, NODE_JOBS="123 456", JOBS="123:RUNNING 456:RUNNING", IMAGES_DIR=str(images))
    assert "kept trainer:1" in first.stdout and _removed(tmp_path) == [], first.stdout + first.stderr
    assert (tmp_path / "claims" / "leaving-123").exists()
    last = _cleanup(tmp_path, 0, SLURM_JOB_ID="456", NODE_JOBS="123 456", JOBS="123:COMPLETING 456:RUNNING",
                    IMAGES_DIR=str(images))
    assert _removed(tmp_path) == ["trainer:1"] and "removed trainer:1" in last.stdout, last.stdout + last.stderr
    assert not (images / "trainer_1").exists() and list((tmp_path / "claims" / "images").iterdir()) == []
    # a job that is on the node and not leaving still keeps it
    (images / "trainer_1").write_text("")
    kept = _cleanup(tmp_path, 0, SLURM_JOB_ID="789", NODE_JOBS="789 999", JOBS="999:RUNNING",
                    IMAGES_DIR=str(images))
    assert "kept trainer:1" in kept.stdout and (images / "trainer_1").exists()


def test_a_job_is_leaving_only_after_its_last_container_and_never_from_an_earlier_attempt():
    """The marker says this job uses no image: it goes up after the cleanup's own helper container (whose image another
    leaving job could otherwise remove first) and before the first image decision; and a job that keeps its id across
    a requeue removes its old marker before it claims, builds or runs anything."""
    body = re.search(r"^cleanup\(\) \{.*?^\}\n", JOB, re.M | re.S).group(0)
    mark = ': > "$CLAIMS/leaving-$SLURM_JOB_ID"'
    assert body.count(mark) == 1 and JOB.count(mark) == 1
    assert body.index('--entrypoint bash "$img" -c') < body.index(mark) < body.index('remove_image "$IMG"') \
        < body.index("retry_kept_images")
    unmark = 'rm -f "$CLAIMS/leaving-$SLURM_JOB_ID"'
    assert JOB.index("trap cleanup EXIT") < JOB.index(unmark) < JOB.index('claim_dir "$SYNC_DIR/$RUN_ID"') \
        < JOB.index('stamp "BUILD $BASE"')


DOCKER_NAMES = r"""#!/bin/bash
# docker ps -a --format ...: the names in $NAMES_FILE; docker rm -f NAME...: each name leaves the file, except one
# named in $STUCK (a container docker cannot remove)
echo "$*" >> "$DOCKER_LOG"
case "$1" in
  ps) cat "$NAMES_FILE" ;;
  rm) shift; [ "$1" = -f ] && shift
      for n in "$@"; do
        echo "removed $n" >> "$DOCKER_LOG"
        case " ${STUCK:-} " in
          *" $n "*) ;;
          *) grep -vxF -- "$n" "$NAMES_FILE" > "$NAMES_FILE.new"; mv "$NAMES_FILE.new" "$NAMES_FILE" ;;
        esac
      done ;;
esac
exit 0
"""


def _earlier_attempt(tmp_path, names: str, **env) -> subprocess.CompletedProcess:
    (tmp_path / "bin").mkdir(exist_ok=True)
    (tmp_path / "bin" / "docker").write_text(DOCKER_NAMES)
    (tmp_path / "bin" / "docker").chmod(0o755)
    (tmp_path / "names").write_text("".join(f"{n}\n" for n in names.split()))
    script = (f"{_functions('stamp', 'own_containers', 'clear_earlier_attempt')}CNAME=aorlta-run-$SLURM_JOB_ID\n"
              "clear_earlier_attempt && echo cleared || echo refused")
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60,
                          env={"PATH": f"{tmp_path / 'bin'}:/usr/bin:/bin", "DOCKER_LOG": str(tmp_path / "docker.log"),
                               "NAMES_FILE": str(tmp_path / "names"), "SLURM_JOB_ID": "123", **env})


def test_a_requeued_job_removes_its_earlier_attempts_containers_and_no_other_jobs(tmp_path):
    """A requeue keeps the job id, and an attempt killed before its trap ran leaves aorlta-run-<id>[-part] behind, so
    the new attempt's first `docker run --name` would fail. Only this job's names go: not a job whose id extends
    this one's, and not another live job's."""
    names = ("aorlta-run-123 aorlta-run-123-checkers aorlta-run-123-canary aorlta-run-1234 aorlta-run-1234-checkers "
             "aorlta-run-12 aorlta-run-12-checkers aorlta-run-456")
    out = _earlier_attempt(tmp_path, names)
    assert out.stdout.strip().endswith("cleared"), out.stdout + out.stderr
    assert _removed(tmp_path) == ["aorlta-run-123", "aorlta-run-123-canary", "aorlta-run-123-checkers"]
    assert (tmp_path / "names").read_text().split() == ["aorlta-run-1234", "aorlta-run-1234-checkers",
                                                        "aorlta-run-12", "aorlta-run-12-checkers", "aorlta-run-456"]
    assert "earlier attempt: aorlta-run-123 aorlta-run-123-checkers aorlta-run-123-canary" in out.stdout


def test_a_first_attempt_removes_nothing_and_a_container_that_stays_refuses_the_job(tmp_path):
    first = _earlier_attempt(tmp_path, "aorlta-run-1234 aorlta-run-456")
    assert first.stdout.strip() == "cleared" and _removed(tmp_path) == [], first.stdout + first.stderr
    (tmp_path / "docker.log").unlink()
    stuck = _earlier_attempt(tmp_path, "aorlta-run-123 aorlta-run-123-checkers", STUCK="aorlta-run-123-checkers")
    assert stuck.stdout.strip().endswith("refused"), stuck.stdout + stuck.stderr
    assert "[guard] cannot remove this job's containers from an earlier attempt: aorlta-run-123-checkers" \
        in stuck.stderr


def test_the_earlier_attempts_containers_go_before_the_job_starts_any_container():
    """After the traps (a refusal still runs the cleanup), and before the first container or image build of this
    attempt; a refusal ends the job."""
    call = "clear_earlier_attempt || exit 97\n"
    assert JOB.count(call) == 1
    at = JOB.index(call)
    assert JOB.index("trap cleanup EXIT") < at < JOB.index('--name "$CNAME-canary"')
    assert at < JOB.index('stamp "BUILD $BASE"') and at < JOB.index('--name "$CNAME-imports"')
    assert JOB.index("docker run", JOB.index("trap 'cleanup; exit 143' TERM INT")) > at


def test_a_leaving_marker_is_dropped_once_its_job_has_ended_and_kept_while_it_runs(tmp_path):
    for d in ("out", "runs", "claims"):
        (tmp_path / d).mkdir(exist_ok=True)
    for j in ("111", "222"):
        (tmp_path / "claims" / f"leaving-{j}").write_text("")
    _cleanup(tmp_path, 0, built=0, SLURM_JOB_ID="333", NODE_JOBS="333", JOBS="111:COMPLETED 222:RUNNING 333:RUNNING")
    assert sorted(p.name for p in (tmp_path / "claims").glob("leaving-*")) == ["leaving-222", "leaving-333"]
