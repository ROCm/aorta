#!/bin/bash
# AORLTA: the reward service's GPU stand-ins (aorta's ConSan and ASan) end to end on aorta's four sanitizer fixtures,
# through the standin-gpu profile (tests/reward/test_c2_t11_gpu.py). The installed RocJITsu hook needs glibc >= 2.38,
# so the test runs in rocm/dev-ubuntu-24.04:7.0.2 (pulled with the registry credentials in the docker client's
# configuration if absent, and removed again if this job pulled it).
#
# Submit from the package root (the directory that holds src/aorlta):
#   mkdir -p runs && sbatch jobs/aorlta-reward-standin.sbatch
# Settings, from the environment (sbatch passes the submitting shell's environment on). A setting the test needs and
# does not have ends the job with exit 95.
#   WT                     default the submit directory. The package tree whose test runs
#   AORLTA_OUT             default <submit directory>/runs/reward-standin-<job id>. The test's evidence
#                          (t11-evidence.json), its reward cache and its work directories
#   AORLTA_VENV            default $HOME/venvs/aorlta. A Python 3.12 venv with requirements-dev.txt, which runs pytest
#                          in the container; it and the Python it was made from (pyvenv.cfg's `home`; one under /usr
#                          is left to the image's python3.12) are mounted read-only
#   AORLTA_AORTA_ROOT      default /opt/aorta. aorta at commit d22f6116235c05395a0241a83d6f7e952a988454: its src/ and
#                          its recipes/sanitizers/fixtures
#   AORLTA_ROCJITSU_ROOT   default /opt/rocjitsu. rocjitsu's bin/rj_waitcheck and lib/librocjitsu_dbi_hooks.so
#   AORLTA_LIBSTDCXX_SHIM  default /opt/libstdcxx-shim. A libstdc++ with GLIBCXX_3.4.31, for rj_waitcheck
#   AORLTA_ASAN_LIB        no default. The ROCm 7.0.2 ASan runtime libraries (lib/asan of ROCm's *-asan packages),
#                          which aorta gets as AORTA_ASAN_ROCM_LIB
#   AORLTA_TMPDIR          default $TMPDIR, else /tmp. Temporary files (TMPDIR); the test's go in a per-job directory
#                          under it, removed at the end
#   AORLTA_DOCKER_CONFIG   default $DOCKER_CONFIG, else $HOME/.docker. The docker client's configuration directory,
#                          which holds the registry credentials for the pull
#   AORLTA_MOUNTS          default none. More host directories, space-separated, mounted read-only into the container
#   AORLTA_EXCLUDE_NODES   default none. Nodes this job refuses to run on (exit 90), as a Slurm host list, such as nodes
#                          whose GPUs fail aorta's ASan node-health canary. #SBATCH lines cannot read the environment,
#                          so give Slurm the same list when submitting:
#                            sbatch --exclude="$AORLTA_EXCLUDE_NODES" jobs/aorlta-reward-standin.sbatch
# The container sees each directory at its host path; the paths in the standin-gpu profile's `adapters:` must lie
# inside these mounts.
#SBATCH -N 1
#SBATCH --no-requeue
#SBATCH -t 00:45:00
#SBATCH -J aorlta-reward-standin
#SBATCH -o runs/reward-standin-%j.out
#SBATCH -e runs/reward-standin-%j.out
set -uo pipefail
if [ -n "${AORLTA_EXCLUDE_NODES:-}" ]; then
  excluded=$(scontrol show hostnames "$AORLTA_EXCLUDE_NODES" 2>/dev/null || tr ',' '\n' <<< "$AORLTA_EXCLUDE_NODES")
  if grep -qxF -- "${SLURMD_NODENAME:-unknown}" <<< "$excluded"; then
    echo "[guard] refusing to run on ${SLURMD_NODENAME:-unknown}: it is in AORLTA_EXCLUDE_NODES" >&2; exit 90
  fi
fi
[ -n "${SLURM_JOB_ID:-}" ] || { echo "[guard] not inside a Slurm job" >&2; exit 91; }
AORLTA_VENV=${AORLTA_VENV:-$HOME/venvs/aorlta}
AORLTA_AORTA_ROOT=${AORLTA_AORTA_ROOT:-/opt/aorta}
AORLTA_ROCJITSU_ROOT=${AORLTA_ROCJITSU_ROOT:-/opt/rocjitsu}
AORLTA_LIBSTDCXX_SHIM=${AORLTA_LIBSTDCXX_SHIM:-/opt/libstdcxx-shim}
AORLTA_ASAN_LIB=${AORLTA_ASAN_LIB:-}
AORLTA_TMPDIR=${AORLTA_TMPDIR:-${TMPDIR:-/tmp}}
read -ra EXTRA_MOUNTS <<< "${AORLTA_MOUNTS:-}"
export TMPDIR=$AORLTA_TMPDIR
export DOCKER_CONFIG=${AORLTA_DOCKER_CONFIG:-${DOCKER_CONFIG:-$HOME/.docker}}
WT=${WT:-$SLURM_SUBMIT_DIR}
OUT=${AORLTA_OUT:-$SLURM_SUBMIT_DIR/runs/reward-standin-$SLURM_JOB_ID}
PY=$AORLTA_VENV/bin/python
[ -x "$PY" ] || { echo "[guard] no Python at $PY: set AORLTA_VENV to a Python 3.12 venv with requirements-dev.txt" >&2
  exit 95; }
[ -d "$AORLTA_ASAN_LIB" ] || {
  echo "[guard] set AORLTA_ASAN_LIB to the ROCm 7.0.2 ASan runtime directory (lib/asan)" >&2; exit 95; }
# The Python the venv was made from (pyvenv.cfg's `home`), which its bin/python links to: mounted beside the venv,
# unless it is under /usr, where the image's own python3.12 stands in for it.
PY_BASE=$(sed -n 's/^home *= *//p' "$AORLTA_VENV/pyvenv.cfg" 2>/dev/null)
PY_BASE=${PY_BASE%/bin}
case "$PY_BASE" in /usr|/usr/*) PY_BASE= ;; esac
IMG=rocm/dev-ubuntu-24.04:7.0.2
CNAME=aorlta-reward-standin-$SLURM_JOB_ID
mkdir -p "$OUT"
PULLED=0
stamp() { echo "=== [$(date -u +%FT%TZ)] $*"; }
add_mounts() {  # add_mounts ARRAY rw|ro DIR...: `-v DIR:DIR[:ro]` for each DIR that exists and is not inside a
  # directory ARRAY already mounts. A missing DIR is reported and left out, since docker would create it root-owned.
  local -n _mounts=$1
  local _mode=$2 _d _i
  shift 2
  for _d in "$@"; do
    _d=${_d%/}
    [ -n "$_d" ] || continue
    [ -d "$_d" ] || { echo "[mounts] not mounted, no such directory: $_d" >&2; continue; }
    for ((_i = 1; _i < ${#_mounts[@]}; _i += 2)); do
      case "$_d/" in "${_mounts[_i]%%:*}/"*) continue 2 ;; esac
    done
    if [ "$_mode" = ro ]; then _mounts+=(-v "$_d:$_d:ro"); else _mounts+=(-v "$_d:$_d"); fi
  done
}
cleanup() {
  docker rm -f "$CNAME" >/dev/null 2>&1
  if [ "$PULLED" = 1 ]; then docker rmi "$IMG" >/dev/null 2>&1 && echo "removed $IMG (pulled by this job)"; fi
}
trap cleanup EXIT
trap 'cleanup; exit 143' TERM INT

stamp "node $(hostname -s) job $SLURM_JOB_ID worktree $WT commit $(/usr/bin/git -C "$WT" rev-parse HEAD)"
if ! docker image inspect "$IMG" >/dev/null 2>&1; then
  timeout 900 docker pull "$IMG" >/dev/null 2>&1 || { echo "FATAL: cannot pull $IMG"; exit 92; }
  PULLED=1
fi
stamp "image $(docker image inspect --format '{{index .RepoDigests 0}}' "$IMG")"
MOUNTS=()
add_mounts MOUNTS rw "$WT" "$OUT" "$AORLTA_TMPDIR"
add_mounts MOUNTS ro "$AORLTA_VENV" "$PY_BASE" "$AORLTA_AORTA_ROOT" "$AORLTA_ROCJITSU_ROOT" "$AORLTA_LIBSTDCXX_SHIM" \
  "$AORLTA_ASAN_LIB" "${EXTRA_MOUNTS[@]}"
timeout 2400 docker run --rm --name "$CNAME" --network=host --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host --security-opt seccomp=unconfined "${MOUNTS[@]}" \
  -e SLURM_JOB_ID -e AORLTA_STANDIN_GPU=1 -e AORLTA_T11_OUT="$OUT" -e AORLTA_AORTA_ROOT="$AORLTA_AORTA_ROOT" \
  -e AORLTA_ROCJITSU_ROOT="$AORLTA_ROCJITSU_ROOT" -e AORLTA_RJ_WAITCHECK="$AORLTA_ROCJITSU_ROOT/bin/rj_waitcheck" \
  -e AORLTA_LIBSTDCXX_SHIM="$AORLTA_LIBSTDCXX_SHIM" \
  -e PYTHONDONTWRITEBYTECODE=1 -e TMPDIR="$AORLTA_TMPDIR/aorlta-reward-standin-$SLURM_JOB_ID" \
  -e AORTA_ASAN_ROCM_LIB="$AORLTA_ASAN_LIB" \
  --entrypoint bash "$IMG" -c "export PATH=/opt/rocm/bin:\$PATH; mkdir -p \$TMPDIR; cd '$WT'; \
    ldd --version | head -1; echo rocm \$(cat /opt/rocm/.info/version); \
    '$PY' -m pytest -q -rA tests/reward/test_c2_t11_gpu.py 2>&1 | tail -40; \
    rc=\${PIPESTATUS[0]}; chown -R $(id -u):$(id -g) '$OUT' \$TMPDIR; rm -rf \$TMPDIR; exit \$rc"
rc=$?
stamp "pytest rc=$rc"
[ -f "$OUT/t11-evidence.json" ] && "$PY" -c "
import json; e = json.load(open('$OUT/t11-evidence.json'))
for k, v in e['verdicts'].items():
    print(k, 'reward', v['reward'], v['reason'], {l: (x['state'], x['reason']) for l, x in v['legs'].items() if l.startswith('san.')})
"
exit $rc
