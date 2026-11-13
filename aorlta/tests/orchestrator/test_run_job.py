"""jobs/aorlta-run.sbatch, read as text: the image it builds is the one the trainer image starts from, and the
containers that load a profile or the config get the site's tool roots and site file."""

import re
import shutil
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
# docker rmi / rm: logged; docker run ... -c CMD: CMD run here (the cleanup's file work); docker image inspect: found
echo "$*" >> "$DOCKER_LOG"
case "$1" in
  ps) if [ "$3" = --filter ]; then printf '%s\n' $USERS_OF_IMAGE; else printf '%s\n' $CONTAINERS; fi ;;
  rm|rmi) shift; [ "$1" = -f ] && shift; printf '%s %s\n' "removed" "$*" >> "$DOCKER_LOG" ;;
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
    assert JOB.index('claim "directory $SYNC_DIR/$RUN_ID"') < JOB.index("stale=$(stale_containers)") < JOB.index(
        '--name "$CNAME-canary"')
    assert JOB.index("trap cleanup EXIT") < JOB.index('claim "directory $SYNC_DIR/$RUN_ID"')
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
    return (f"{_functions('stamp', 'job_ended', 'claim', 'dir_key')}CLAIMS={tmp_path}/claims\nCLAIMED=()\n{body}\n"
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
    out = _bash(tmp_path, f"{_functions('dir_key')}dir_key /dev/shm/aorlta/a; dir_key /dev/shm/aorlta/b; "
                          "dir_key /dev/shm/aorlta/a")
    a, b, a2 = out.stdout.split()
    assert a == a2 != b and a.startswith("dir-") and len(a) == 20


def _cleanup(tmp_path, own: int, **env) -> subprocess.CompletedProcess:
    funcs = _functions("add_mounts", "clean_img", "other_jobs", "remove_image", "cleanup")
    state = (f"CNAME=aorlta-run-123 VRAM_PID= OUT={tmp_path}/out RUN_ROOT={tmp_path}/runs SYNC_DIR={tmp_path}/shm "
             f"RUN_ID=r SOCK_DIR={tmp_path}/shm/sockets RUN_DIR={tmp_path}/runs/r BUILT_TRAINER=1 BUILT_BASE=0 "
             f"PULLED_CHK=0 BASE=b CHK_IMG=c IMG=trainer:1 OWN_DIRS={own}\n"
             f"CLAIMED=({tmp_path}/claims/gpu0 {tmp_path}/claims/gpu1)\n")
    return _bash(tmp_path, f"{funcs}{state}cleanup", **env)


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
