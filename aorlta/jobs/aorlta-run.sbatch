#!/bin/bash
# AORLTA: one run of the GRPO loop (`aorlta.orchestrator.run`) on one node, such as configs/standin.yaml's 3-iteration
# stand-in run. The TokenSpeed image is built in-job if the node lacks it, then the trainer image from
# docker/Dockerfile.trainer; an EXIT/TERM trap removes this job's containers, its /dev/shm slots and every image it
# built or pulled that no other job on the node may be using. docker runs in the background with `wait`, so a TERM
# (scancel) runs the trap at once: bash defers traps while a foreground command runs, and a killed docker client
# leaves its container running.
#
# Two runs on one node. The job asks for the whole node (the --exclusive directive below); a command-line option
# overrides it, so `sbatch --exclusive=user -w <node> ...` lets your own jobs share a node, on a partition that allows
# it (one with OverSubscribe=EXCLUSIVE never puts two jobs on a node). Their configs must name
# disjoint GPUs (rollout.engines, trainer.devices, reward.remote's GPUs, and any GPUs a task's own checks use), and
# each its own run id and reward.remote.socket_dir. The job claims the GPUs and its two /dev/shm directories
# (<sync.dir>/<run id> and the socket directory) in files under /dev/shm/aorlta-claims, and ends with exit 89 when a
# job Slurm still runs holds one (a GPU a task's own checks use is not claimed: the job cannot see it). Engine ports
# need nothing with rollout.port_base null (the default: each engine start picks a free window); two runs that both
# fix a port_base must give windows that do not overlap. A container or claim of a job Slurm no longer runs is what
# a cancelled or killed job left: the job removes or takes it over, and never touches another live job's.
#
# With `reward.remote` in the config (the standin-gpu profile's ConSan and ASan legs), the checkers run in
# rocm/dev-ubuntu-24.04:7.0.2 (glibc 2.39 for the ConSan hook, the ROCm 7.0.2 ASan runtime): first aorta's ASan
# node-health canary on every checker GPU (a node that fails it ends the job, exit 96), then one
# `aorlta.reward.remote serve` process per adapter and GPU (an adapter may name several GPUs), each seeing only its
# own GPU (ROCR_VISIBLE_DEVICES), on Unix sockets in /dev/shm that the orchestrator's container reaches.
#
# Submit from the package root (the directory that holds src/aorlta), with the config committed:
#   mkdir -p runs && CONFIG=configs/<run>.yaml sbatch jobs/aorlta-run.sbatch
#   RESUME=1          resume the run directory from its last checkpoint (a new job after a dead one)
#   COMPONENTS=fake   the loop with its fakes (a CPU smoke test of this script and the image)
#   KILL_AFTER_ITER=n [KILL_DELAY_S=s]   the resume drill on the real components: s seconds (default 120)
#                     after iteration n commits, SIGKILL the orchestrator's container, then resume it in this job
#   With remote checkers, a pre-flight drives them on aorta's fixtures (jobs/aorlta-checkers-smoke.py) before
#   anything is built; a failure ends the job (exit 98).
#   SMOKE=checkers    the canary, the checkers and their pre-flight only: no image builds, engines or trainer
#   SMOKE=skip        no checker pre-flight
#   While the orchestrator runs, every GPU's VRAM in use is sampled from sysfs every 10 s into vram.tsv.
# A config with `orchestrator.mode: score` is a score run (aorlta.orchestrator.score): the same checkers, images and
# engines, no trainer and no weight sync; the job ends by printing the run's score/summary.txt. It is never resumed.
#
# Site settings, from the environment (sbatch passes the submitting shell's environment on). A setting the run needs
# and does not have ends the job with exit 95.
#   AORLTA_VENV            default $HOME/venvs/aorlta. A Python 3.12 venv with requirements-dev.txt. It reads the
#                          config on the host and runs the checkers in their containers, which mount it and the Python
#                          it was made from (pyvenv.cfg's `home`; one under /usr is left to the image's python3.12)
#   TS_SRC                 no default. TokenSpeed's source tree at commit 3b684074, needed only on a node without the
#                          image aorlta-tokenspeed:3b684074, which is then built from its docker/Dockerfile.amd
#   AORLTA_AORTA_ROOT      default /opt/aorta. aorta at commit d22f6116235c05395a0241a83d6f7e952a988454: its src/ for
#                          the checkers and the import check, its recipes/sanitizers/fixtures for the pre-flight; needed
#                          only when the check profile uses an aorta adapter (reward.profiles.AORTA_ADAPTERS)
#   AORLTA_ROCJITSU_ROOT   default /opt/rocjitsu. rocjitsu's bin/rj_waitcheck and lib/librocjitsu_dbi_hooks.so
#   AORLTA_LIBSTDCXX_SHIM  default /opt/libstdcxx-shim. A libstdc++ with GLIBCXX_3.4.31, for rj_waitcheck
#                          (these three are also where the check profiles look for the tools; the job passes them into
#                          its containers, whose hipcc is the image's /opt/rocm)
#   AORLTA_SITE            default none. A site file of the run's paths, applied over the config (aorlta.config
#                          apply_site); the job keeps a copy in its output directory, which its containers read
#   AORLTA_ASAN_LIB        no default; remote checkers need it. The ROCm 7.0.2 ASan runtime libraries (lib/asan of
#                          ROCm's *-asan packages), which aorta gets as AORTA_ASAN_ROCM_LIB
#   AORLTA_HF_HOME         default $HF_HOME, else $HOME/.cache/huggingface. The Hugging Face cache (HF_HOME), read
#                          offline
#   AORLTA_TMPDIR          default $TMPDIR, else /tmp. Temporary files (TMPDIR), on the host and in the orchestrator's
#                          container
#   AORLTA_DOCKER_CONFIG   default $DOCKER_CONFIG, else $HOME/.docker. The docker client's configuration directory,
#                          which holds the registry credentials for pulling rocm/dev-ubuntu-24.04:7.0.2
#   AORLTA_MOUNTS          default none. More host directories, space-separated, mounted read-only into the
#                          orchestrator's and the checkers' containers
#   AORLTA_PYTHONPATH      default none. More import paths, colon-separated, after the package's own src/: where a
#                          plugin package of your own lives (`plugins:` in the config), on the host and in every
#                          container. Mount its directories too (AORLTA_MOUNTS)
#   AORLTA_EXCLUDE_NODES   default none. Nodes this job refuses to run on (exit 90), as a Slurm host list. #SBATCH
#                          lines cannot read the environment, so give Slurm the same list when submitting:
#                            sbatch --exclude="$AORLTA_EXCLUDE_NODES" jobs/aorlta-run.sbatch
#   AORLTA_CLAIMS_DIR      default /dev/shm/aorlta-claims. Where jobs on this node claim GPUs and directories: node-
#                          local, and the same for every job of the node, or two jobs cannot see each other's claims
# A container sees each directory at its host path. Mounted besides the directories above: the package root, /dev/shm
# (reward.remote.socket_dir must be under it, and sync.dir is by default), and from the config run.root,
# model.base_dir and model.adapter_dir (the adapter read-only; for a Hugging Face snapshot, the model's whole cache
# directory, so the snapshot's links resolve) and each prompt pool's dataset directory.
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH -t 05:00:00
#SBATCH -J aorlta-run
#SBATCH -o runs/job-%j.out
#SBATCH -e runs/job-%j.out
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
AORLTA_HF_HOME=${AORLTA_HF_HOME:-${HF_HOME:-$HOME/.cache/huggingface}}
AORLTA_TMPDIR=${AORLTA_TMPDIR:-${TMPDIR:-/tmp}}
read -ra EXTRA_MOUNTS <<< "${AORLTA_MOUNTS:-}"
XPATH=${AORLTA_PYTHONPATH:+:$AORLTA_PYTHONPATH}
export TMPDIR=$AORLTA_TMPDIR HF_HOME=$AORLTA_HF_HOME
export DOCKER_CONFIG=${AORLTA_DOCKER_CONFIG:-${DOCKER_CONFIG:-$HOME/.docker}}
P=$SLURM_SUBMIT_DIR
[ -f "$P/src/aorlta/orchestrator/loop.py" ] || {
  echo "[guard] submit from the package root (no orchestrator under $P)" >&2; exit 92; }
CONFIG=${CONFIG:?set CONFIG to the run config, a YAML file under the package root}
COMPONENTS=${COMPONENTS:-real}
RESUME=${RESUME:-0}
KILL_AFTER_ITER=${KILL_AFTER_ITER:-}
KILL_DELAY_S=${KILL_DELAY_S:-120}
SMOKE=${SMOKE:-}
SRC=${TS_SRC:-}
BASE=aorlta-tokenspeed:3b684074
CHK_IMG=rocm/dev-ubuntu-24.04:7.0.2
PY=$AORLTA_VENV/bin/python  # Python 3.12 (aorlta needs it), on the host and in the checker containers
[ -x "$PY" ] || { echo "[guard] no Python at $PY: set AORLTA_VENV to a Python 3.12 venv with requirements-dev.txt" >&2
  exit 95; }
# The Python the venv was made from (pyvenv.cfg's `home`), which its bin/python links to: the checker containers mount
# it beside the venv, unless it is under /usr, where the checker image's own python3.12 stands in for it.
PY_BASE=$(sed -n 's/^home *= *//p' "$AORLTA_VENV/pyvenv.cfg" 2>/dev/null)
PY_BASE=${PY_BASE%/bin}
case "$PY_BASE" in /usr|/usr/*) PY_BASE= ;; esac
OUT=$P/runs/$SLURM_JOB_ID
CNAME=aorlta-run-$SLURM_JOB_ID
BUILT_BASE=0
BUILT_TRAINER=0
PULLED_CHK=0
SOCK_DIR=
VRAM_PID=
CLAIMS=${AORLTA_CLAIMS_DIR:-/dev/shm/aorlta-claims}
CLAIMED=()
OWN_DIRS=0  # this job holds the claims on <sync.dir>/<run id> and the socket directory, so its trap may remove them
stamp() { echo "=== [$(date -u +%FT%TZ)] $*"; }
job_ended() {  # job_ended ID: Slurm no longer runs job ID: it does not know the job, or names a final state. Any other
  # answer, and no answer at all (squeue missing, the controller unreachable), counts as running, so a live job's
  # containers, images and claims are never taken for left-overs.
  local out
  if ! out=$(squeue -h -j "$1" -o %T 2>&1); then
    grep -qi 'invalid job id' <<< "$out"
    return
  fi
  case "$out" in
    ''|BOOT_FAIL|CANCELLED|COMPLETED|DEADLINE|FAILED|NODE_FAIL|OUT_OF_MEMORY|PREEMPTED|REVOKED|SPECIAL_EXIT|TIMEOUT)
      return 0 ;;
  esac
  return 1
}
stale_containers() {  # the aorlta-run-<job id>[-part] containers on this node whose job Slurm no longer runs
  local name id
  local -A ended=()
  while read -r name; do
    id=${name#aorlta-run-}
    id=${id%%-*}
    [ "$id" = "$SLURM_JOB_ID" ] && continue
    if [ -z "${ended[$id]:-}" ]; then
      if job_ended "$id"; then ended[$id]=1; else ended[$id]=0; fi
    fi
    [ "${ended[$id]}" = 1 ] && echo "$name"
  done < <(docker ps -a --format '{{.Names}}' | grep -xE 'aorlta-run-[0-9]+(-[a-z]+)?')
}
other_jobs() {  # the other Slurm jobs running on this node, one per line; "unknown" if Slurm cannot say
  local out
  out=$(squeue -h -w "${SLURMD_NODENAME:-$(hostname -s)}" -o %i 2>/dev/null) || { echo unknown; return; }
  grep -vxF -- "$SLURM_JOB_ID" <<< "$out" | grep .
}
remove_image() {  # remove_image IMAGE: remove an image this job built or pulled, unless another job may use it: a
  # container of another job was made from it (or from an image built on it), or another job runs on this node (it
  # may have found the image when it started and not have run it yet). A kept image stays for the last job to leave.
  local users jobs
  users=$(docker ps -a --filter "ancestor=$1" --format '{{.Names}}' 2>/dev/null | grep -vxE "$CNAME(-[a-z]+)?")
  jobs=$(other_jobs)
  if [ -n "$users$jobs" ]; then
    echo "kept $1: another job may use it (containers: ${users:-none}; jobs on this node: ${jobs:-none})" | tr '\n' ' '
    echo
    return
  fi
  docker rmi -f "$1" >/dev/null 2>&1 && echo "removed $1"
}
claim() {  # claim WHAT KEY...: hold each KEY for this job (a file under $CLAIMS naming its job id). A key held by a job
  # Slurm still runs ends this job, exit 89; one whose job ended is taken over. Under flock, so two jobs taking over
  # the same left-over claim cannot both get it.
  local what=$1 key f owner fd
  shift
  mkdir -p "$CLAIMS" && chmod 1777 "$CLAIMS" 2>/dev/null  # other users' jobs on a shared node claim here too
  touch "$CLAIMS/.lock" 2>/dev/null && chmod 666 "$CLAIMS/.lock" 2>/dev/null
  exec {fd}>>"$CLAIMS/.lock" && flock -w 60 "$fd" || { echo "[guard] cannot lock $CLAIMS/.lock" >&2; exit 89; }
  for key in "$@"; do
    f=$CLAIMS/$key
    owner=$(cut -d' ' -f1 "$f" 2>/dev/null)
    if [ -n "$owner" ] && [ "$owner" != "$SLURM_JOB_ID" ]; then
      if ! job_ended "$owner"; then
        echo "[guard] job $owner, which Slurm still runs (or Slurm cannot say), holds $what on this node ($f):" \
          "give this run other GPUs, or another run id and socket directory" >&2
        exec {fd}>&-
        exit 89
      fi
      stamp "taking over $key from job $owner, which has ended"
    fi
    echo "$SLURM_JOB_ID $what" > "$f" && CLAIMED+=("$f") || { echo "[guard] cannot write $f" >&2; exec {fd}>&-; exit 89; }
  done
  exec {fd}>&-
}
dir_key() { echo "dir-$(printf %s "$1" | sha256sum | cut -c1-16)"; }
vram_sampler() {  # every 10 s, from sysfs (no GPU context): UTC seconds, PCI address, VRAM used and total in bytes
  local c
  while :; do
    for c in /sys/class/drm/card*/device; do
      [ -r "$c/mem_info_vram_used" ] && echo "$(date -u +%s) $(basename "$(readlink -f "$c")") $(cat "$c/mem_info_vram_used") $(cat "$c/mem_info_vram_total")"
    done
    sleep 10
  done
}
add_mounts() {  # add_mounts ARRAY rw|ro DIR...: `-v DIR:DIR[:ro]` for each DIR that exists and is not inside a
  # directory ARRAY already mounts the same way, so a read-only DIR inside a read-write mount gets its own mount. A
  # missing DIR is reported and left out, since docker would create it root-owned.
  local -n _mounts=$1
  local _mode=$2 _d _i _dst _has
  shift 2
  for _d in "$@"; do
    _d=${_d%/}
    [ -n "$_d" ] || continue
    [ -d "$_d" ] || { echo "[mounts] not mounted, no such directory: $_d" >&2; continue; }
    for ((_i = 1; _i < ${#_mounts[@]}; _i += 2)); do
      _dst=${_mounts[_i]%%:*}
      case "${_mounts[_i]}" in *:ro) _has=ro ;; *) _has=rw ;; esac
      if [ "$_d" = "$_dst" ]; then  # docker takes a destination once: the first mount stands
        [ "$_has" = "$_mode" ] || echo "[mounts] $_d is mounted $_has already, not $_mode" >&2
        continue 2
      fi
      [ "$_has" = "$_mode" ] || continue
      case "$_d/" in "$_dst/"*) continue 2 ;; esac
    done
    if [ "$_mode" = ro ]; then _mounts+=(-v "$_d:$_d:ro"); else _mounts+=(-v "$_d:$_d"); fi
  done
}

# Everything runs from a snapshot of the committed tree, so the run belongs to one commit (run_plan.json records it).
CODE=$OUT/code
mkdir -p "$CODE"
COMMIT=$(/usr/bin/git -C "$P" rev-parse HEAD) || { echo "[guard] cannot read HEAD of $P" >&2; exit 93; }
if [ -n "${AORLTA_SITE:-}" ]; then
  cp "$AORLTA_SITE" "$OUT/site.yaml" || { echo "[guard] cannot read the site file $AORLTA_SITE" >&2; exit 94; }
  export AORLTA_SITE=$OUT/site.yaml
fi
/usr/bin/git -C "$P" archive HEAD | tar -x -C "$CODE" || { echo "[guard] cannot snapshot HEAD of $P" >&2; exit 93; }
[ -f "$CODE/$CONFIG" ] || { echo "[guard] $CONFIG is not committed at $COMMIT" >&2; exit 94; }
read -r RUN_ID SYNC_DIR RUN_DIR RUN_ROOT < <(PYTHONPATH="$CODE/src$XPATH" "$PY" -c "
from aorlta.config import load_config; c = load_config('$CODE/$CONFIG')
print(c.run.run_id, c.sync.dir, c.run_dir, c.run.root)") || { echo "[guard] $CONFIG does not load" >&2; exit 94; }
mkdir -p "$RUN_ROOT" || { echo "[guard] cannot create run.root ($RUN_ROOT)" >&2; exit 95; }
# The data the config names, for the orchestrator's container: model.base_dir, then model.adapter_dir or an empty line
# (for a Hugging Face snapshot, the model's whole cache directory, so the snapshot's links into blobs/ resolve), then
# each prompt pool's dataset directory (the pool's path, or its parent if that holds kernels.json and listings/, as the
# wiring reads it)
mapfile -t DATA_DIRS < <(PYTHONPATH="$CODE/src$XPATH" "$PY" -c "
from pathlib import Path
from aorlta.config import load_config
c = load_config('$CODE/$CONFIG')
def model(d):
    d = Path(d)
    return d.parent.parent if d.parent.name == 'snapshots' else d
print(model(c.model.base_dir))
print(model(c.model.adapter_dir) if c.model.adapter_dir else '')
for pool in c.data.prompt_pools:
    p = Path(pool.path)
    print(next((d for d in (p, p.parent) if (d / 'kernels.json').is_file() and (d / 'listings').is_dir()), p))")
MODEL_DIR=${DATA_DIRS[0]:-}
ADAPTER_DIR=${DATA_DIRS[1]:-}
POOL_DIRS=("${DATA_DIRS[@]:2}")
# reward.remote: the socket directory, then one 'adapter gpu socket' line per checker server
mapfile -t REMOTE < <(PYTHONPATH="$CODE/src$XPATH" "$PY" -c "
from aorlta.config import load_config; from aorlta.reward.remote import servers
r = load_config('$CODE/$CONFIG').reward.remote
if r is not None:
    print(r.socket_dir)
    for name, gpu, sock in servers(r):
        print(name, gpu, sock)")
if [ "$COMPONENTS" = real ]; then
  [ -f "$CODE/src/aorlta/orchestrator/wiring.py" ] && [ -f "$CODE/docker/Dockerfile.trainer" ] || {
    echo "[guard] COMPONENTS=real needs src/aorlta/orchestrator/wiring.py and docker/Dockerfile.trainer at $COMMIT" >&2
    exit 95; }
  IMG=aorlta-trainer:$(sha256sum "$CODE/docker/Dockerfile.trainer" | cut -c1-12)
else
  IMG=$BASE
  REMOTE=()  # the fakes check nothing
fi
[ "${#REMOTE[@]}" -gt 0 ] && SOCK_DIR=${REMOTE[0]}
if [ "$SMOKE" = checkers ] && [ -z "$SOCK_DIR" ]; then
  echo "[guard] SMOKE=checkers needs reward.remote in $CONFIG" >&2; exit 95
fi
if [ -n "$SOCK_DIR" ]; then
  case "$SOCK_DIR" in
    /dev/shm/?*) ;;
    *) echo "[guard] reward.remote.socket_dir ($SOCK_DIR) must be under /dev/shm, which the containers share" >&2
       exit 95 ;;
  esac
  [ -d "$AORLTA_ASAN_LIB" ] || {
    echo "[guard] remote checkers need AORLTA_ASAN_LIB, the ROCm 7.0.2 ASan runtime directory (lib/asan)" >&2
    exit 95; }
fi

clean_img() {  # an image this node has, for removing root-owned files
  local i
  for i in "$BASE" "$CHK_IMG"; do docker image inspect "$i" >/dev/null 2>&1 && { echo "$i"; return; }; done
}
cleanup() {
  [ -n "$VRAM_PID" ] && kill "$VRAM_PID" 2>/dev/null
  docker ps -a --format '{{.Names}}' | grep -xE "$CNAME(-[a-z]+)?" | xargs -r docker rm -f >/dev/null 2>&1
  find "$OUT" -maxdepth 3 -name 'gpucore.*' -type f -delete 2>/dev/null
  local img mounts=()
  img=$(clean_img)
  add_mounts mounts rw /dev/shm "$OUT" "$RUN_ROOT" 2>/dev/null
  # The GPU legs' logs, harnesses and the candidates' parsed sources (stand-in data) are kept, small, before the
  # node directory goes: they are how an unsettled ConSan or ASan leg is read afterwards.
  # (a job that did not get its claims leaves the directories alone: another live job holds them)
  [ -n "$img" ] && docker run --rm --network none --ulimit core=0 "${mounts[@]}" --entrypoint bash "$img" -c \
    "if [ $OWN_DIRS = 1 ] && cd '$SYNC_DIR/$RUN_ID/scratch/reward' 2>/dev/null; then
       find pkg \( -path '*/san.*' \( -name '*.log' -o -name aorlta_harness.hip -o -name '*.json' \) \
         -o -path '*/gate.parse-*' -name source.hip \) -size -2M -print0 | tar --null -czf '$OUT/gpu-leg-evidence.tgz' -T -;
       cd /; fi
     [ $OWN_DIRS = 1 ] && rm -rf '$SYNC_DIR/$RUN_ID' ${SOCK_DIR:+'$SOCK_DIR'}; chown -R $(id -u):$(id -g) '$OUT';
     [ -d '$RUN_DIR' ] && chown -R $(id -u):$(id -g) '$RUN_DIR'" >/dev/null 2>&1
  [ "$BUILT_TRAINER" = 1 ] && remove_image "$IMG"
  [ "$BUILT_BASE" = 1 ] && remove_image "$BASE"
  [ "$PULLED_CHK" = 1 ] && remove_image "$CHK_IMG"
  local f
  for f in "${CLAIMED[@]}"; do
    [ "$(cut -d' ' -f1 "$f" 2>/dev/null)" = "$SLURM_JOB_ID" ] && rm -f "$f"
  done
  echo "containers left: $(docker ps -a --format '{{.Names}}' | grep -cxE "$CNAME(-[a-z]+)?")"
}
trap cleanup EXIT
trap 'cleanup; exit 143' TERM INT

stamp "node $(hostname -s) job $SLURM_JOB_ID run $RUN_ID config $CONFIG components $COMPONENTS resume $RESUME commit $COMMIT"
# Before anything uses them (the checker canary empties the socket directory): this run's /dev/shm directories, and
# the GPUs it uses on this node, which are the engines', the trainer ranks' own (with trainer.devices null they share
# the engines') and the checker servers'. The fakes use no GPU.
claim "directory $SYNC_DIR/$RUN_ID" "$(dir_key "$SYNC_DIR/$RUN_ID")"
[ -n "$SOCK_DIR" ] && claim "directory $SOCK_DIR" "$(dir_key "$SOCK_DIR")"
OWN_DIRS=1
RUN_GPUS=()
if [ "$COMPONENTS" = real ]; then
  read -ra RUN_GPUS < <(PYTHONPATH="$CODE/src$XPATH" "$PY" -c "
from aorlta.config import load_config
c = load_config('$CODE/$CONFIG')
gpus = set(c.rollout.engines)
if c.orchestrator.mode != 'score':
    gpus |= set(c.trainer.devices or ())
if c.reward.remote is not None:
    for name in c.reward.remote.adapters:
        gpus |= set(c.reward.remote.gpus(name))
print(*sorted(gpus))") || { echo "[guard] $CONFIG does not load" >&2; exit 94; }
fi
for g in "${RUN_GPUS[@]}"; do claim "GPU $g" "gpu$g"; done
stamp "claimed GPUs ${RUN_GPUS[*]:-none} and the run's /dev/shm directories"
# A container left by a job Slurm no longer runs (cancelled, killed) is removed; another live job's are left alone.
stale=$(stale_containers)
[ -n "$stale" ] && { stamp "removing stale containers: $(echo $stale)"; echo "$stale" | xargs -r docker rm -f >/dev/null 2>&1; }
df -h /dev/shm | tail -1

# --- the remote checkers, first: a node whose canary or pre-flight fails ends the job before anything is built ---
if [ -n "$SOCK_DIR" ]; then
  if ! docker image inspect "$CHK_IMG" >/dev/null 2>&1; then
    stamp "PULL $CHK_IMG"
    timeout 900 docker pull "$CHK_IMG" >/dev/null 2>&1 || { echo "[guard] cannot pull $CHK_IMG" >&2; exit 97; }
    PULLED_CHK=1
  fi
  echo "$CHK_IMG $(docker image inspect --format '{{index .RepoDigests 0}}' "$CHK_IMG")" | tee -a "$OUT/image.txt"
  CHK_MOUNTS=()
  add_mounts CHK_MOUNTS rw "$P" /dev/shm
  add_mounts CHK_MOUNTS ro "$AORLTA_VENV" "$PY_BASE" "$AORLTA_AORTA_ROOT" "$AORLTA_ROCJITSU_ROOT" \
    "$AORLTA_LIBSTDCXX_SHIM" "$AORLTA_ASAN_LIB" "${EXTRA_MOUNTS[@]}"
  # --ulimit core=0: a candidate that faults the GPU would otherwise leave a 150-550 MB gpucore.<pid> in the working
  # directory
  CHK_DOCKER=(--network=host --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --ulimit core=0
              --security-opt seccomp=unconfined "${CHK_MOUNTS[@]}"
              -e SLURM_JOB_ID="$SLURM_JOB_ID" -e AORTA_ASAN_ROCM_LIB="$AORLTA_ASAN_LIB" -e TMPDIR="$SOCK_DIR/tmp"
              -e AORLTA_AORTA_ROOT="$AORLTA_AORTA_ROOT" -e AORLTA_ROCJITSU_ROOT="$AORLTA_ROCJITSU_ROOT"
              -e AORLTA_LIBSTDCXX_SHIM="$AORLTA_LIBSTDCXX_SHIM" -e AORLTA_SITE="${AORLTA_SITE:-}"
              -e PYTHONPATH="$CODE/src$XPATH" -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 -w "$CODE")
  GPUS=$(printf '%s\n' "${REMOTE[@]:1}" | awk '{print $2}' | sort -un | paste -sd, -)
  stamp "CHECKERS: aorta's ASan node-health canary on GPUs $GPUS"
  docker run --rm --name "$CNAME-canary" "${CHK_DOCKER[@]}" -e ROCR_VISIBLE_DEVICES="$GPUS" --entrypoint bash "$CHK_IMG" \
    -c "export PATH=/opt/rocm/bin:\$PATH; rm -rf '$SOCK_DIR'; mkdir -p \$TMPDIR; ldd --version | head -1; \
        echo rocm \$(cat /opt/rocm/.info/version); \
        exec '$PY' -m aorlta.reward.remote node-health --out '$OUT/node-health'" \
    > "$OUT/node-health.log" 2>&1 &
  wait $!
  rc=$?
  cat "$OUT/node-health.log"
  [ "$rc" -eq 0 ] || {
    echo "[guard] the ASan node-health canary did not pass (rc $rc): exclude $(hostname -s) (AORLTA_EXCLUDE_NODES)" \
      "and resubmit" >&2; exit 96; }
  serve="export PATH=/opt/rocm/bin:\$PATH; mkdir -p \$TMPDIR;"
  for line in "${REMOTE[@]:1}"; do
    read -r adapter gpu sock <<< "$line"
    serve+=" ROCR_VISIBLE_DEVICES=$gpu '$PY' -m aorlta.reward.remote serve --config '$CODE/$CONFIG' --adapter $adapter \
--socket '$sock' > '$OUT/checker-$adapter-gpu$gpu.log' 2>&1 &"
  done
  serve+=" wait"
  docker run -d --name "$CNAME-checkers" "${CHK_DOCKER[@]}" --entrypoint bash "$CHK_IMG" -c "$serve" >/dev/null \
    || { echo "[guard] the checker container did not start" >&2; exit 97; }
  for line in "${REMOTE[@]:1}"; do
    read -r adapter gpu sock <<< "$line"
    t0=$SECONDS
    until [ -S "$sock" ]; do
      [ "$(docker inspect -f '{{.State.Running}}' "$CNAME-checkers" 2>/dev/null)" = true ] || {
        echo "[guard] the checker container exited" >&2; tail -20 "$OUT"/checker-*.log; exit 97; }
      [ $((SECONDS - t0)) -lt 180 ] || { echo "[guard] no socket for $adapter on GPU $gpu after 180 s" >&2; exit 97; }
      sleep 1
    done
    stamp "CHECKER $adapter on GPU $gpu at $sock"
  done
  if [ "$SMOKE" != skip ]; then
    stamp "PRE-FLIGHT: the remote checkers on aorta's fixtures"
    docker run --rm --name "$CNAME-smoke" "${CHK_DOCKER[@]}" --entrypoint bash "$CHK_IMG" \
      -c "exec '$PY' '$CODE/jobs/aorlta-checkers-smoke.py' '$CODE/$CONFIG' '$OUT'" > "$OUT/smoke.log" 2>&1 &
    wait $!
    rc=$?
    cat "$OUT/smoke.log"
    for f in "$OUT"/checker-*.log; do echo "--- $f"; tail -5 "$f"; done
    stamp "PRE-FLIGHT rc=$rc"
    [ "$rc" -eq 0 ] || { echo "[guard] the checker pre-flight failed" >&2; exit 98; }
  fi
fi
[ "$SMOKE" = checkers ] && { stamp "SMOKE=checkers: done"; exit 0; }

# --- images -----------------------------------------------------------------------------------------------------
if docker image inspect "$BASE" >/dev/null 2>&1; then
  stamp "image $BASE already on node"
else
  [ -n "$SRC" ] && [ -f "$SRC/docker/Dockerfile.amd" ] || {
    echo "[guard] $BASE is not on this node: set TS_SRC to TokenSpeed's source tree at commit 3b684074 to build it" >&2
    exit 95; }
  stamp "BUILD $BASE"; t0=$SECONDS
  (cd "$SRC" && docker build --network host --progress plain --build-arg MAX_JOBS=64 \
     -f docker/Dockerfile.amd -t "$BASE" . > "$OUT/build-base.log" 2>&1)
  rc=$?; stamp "base build rc=$rc in $((SECONDS-t0)) s"; tail -3 "$OUT/build-base.log"
  [ "$rc" -ne 0 ] && exit "$rc"
  BUILT_BASE=1
fi
if [ "$IMG" != "$BASE" ]; then
  if docker image inspect "$IMG" >/dev/null 2>&1; then
    stamp "image $IMG already on node"
  else
    stamp "BUILD $IMG"; t0=$SECONDS
    docker build --network host --progress plain --build-arg MAX_JOBS=64 \
      -f "$CODE/docker/Dockerfile.trainer" -t "$IMG" "$CODE/docker" > "$OUT/build-trainer.log" 2>&1
    rc=$?; stamp "trainer build rc=$rc in $((SECONDS-t0)) s"; tail -3 "$OUT/build-trainer.log"
    [ "$rc" -ne 0 ] && exit "$rc"
    BUILT_TRAINER=1
  fi
fi
echo "$IMG $(docker image inspect --format '{{.Id}}' "$IMG")" | tee -a "$OUT/image.txt"
if [ "$COMPONENTS" = real ]; then  # every module the run imports, in its image, before any engine starts
  # aorta's backends only when the profile uses an aorta adapter: a task with checks of its own needs no checkout
  AORTA_MODULES=$(PYTHONPATH="$CODE/src$XPATH" "$PY" -c "
from aorlta.config import load_config; from aorlta.reward.profiles import aorta_preflight_modules, load_profile
print(' '.join(aorta_preflight_modules(load_profile(load_config('$CODE/$CONFIG').reward.profile))))") || {
    echo "[guard] cannot read the check profile of $CONFIG" >&2; exit 94; }
  IMPORT_MOUNTS=()
  add_mounts IMPORT_MOUNTS rw "$P"
  add_mounts IMPORT_MOUNTS ro "$AORLTA_AORTA_ROOT"
  docker run --rm --name "$CNAME-imports" --network none --ulimit core=0 "${IMPORT_MOUNTS[@]}" \
    -e PYTHONPATH="$CODE/src$XPATH" -e PYTHONDONTWRITEBYTECODE=1 -w "$CODE" --entrypoint python3 "$IMG" \
    jobs/aorlta-imports-check.py "$AORLTA_AORTA_ROOT/src" $AORTA_MODULES > "$OUT/imports.log" 2>&1 &
  wait $!
  rc=$?
  stamp "IMPORTS rc=$rc"; tail -3 "$OUT/imports.log" | cut -c1-2000
  [ "$rc" -eq 0 ] || { echo "[guard] the run image cannot import what the run needs" >&2; exit 99; }
fi

# --- the orchestrator -------------------------------------------------------------------------------------------
ARGS=(--config "$CODE/$CONFIG" --components "$COMPONENTS" --commit "$COMMIT")
[ "$RESUME" = 1 ] && ARGS+=(--resume)
ORCH_MOUNTS=()
add_mounts ORCH_MOUNTS rw "$P" "$RUN_ROOT" "$MODEL_DIR" "$AORLTA_HF_HOME" "$AORLTA_TMPDIR" /dev/shm
add_mounts ORCH_MOUNTS ro "${POOL_DIRS[@]}" "$ADAPTER_DIR" "$AORLTA_AORTA_ROOT" "$AORLTA_ROCJITSU_ROOT" \
  "$AORLTA_LIBSTDCXX_SHIM" "${EXTRA_MOUNTS[@]}"
# A directory the docker daemon sees at another place than this job does (a Slurm job's private /tmp, for one)
# would hide what the run writes there: each read-write mount must show the container this job's marker.
MARK=.aorlta-mount-$SLURM_JOB_ID
MARKED=()
for ((i = 1; i < ${#ORCH_MOUNTS[@]}; i += 2)); do
  case "${ORCH_MOUNTS[i]}" in *:ro) ;; *) d=${ORCH_MOUNTS[i]%%:*}; touch "$d/$MARK" 2>/dev/null && MARKED+=("$d") ;; esac
done
unseen=$(docker run --rm --name "$CNAME-mounts" --network none --ulimit core=0 "${ORCH_MOUNTS[@]}" --entrypoint bash \
  "$IMG" -c 'for d in "${@:2}"; do [ -e "$d/$1" ] || echo "$d"; done' _ "$MARK" "${MARKED[@]}")
for d in "${MARKED[@]}"; do rm -f "$d/$MARK"; done
[ -z "$unseen" ] || { echo "[guard] the docker daemon does not see these directories where this job does (a per-job" \
  "/tmp?): $(echo $unseen)" >&2; exit 95; }
vram_sampler >> "$OUT/vram.tsv" 2>/dev/null &
VRAM_PID=$!
ORCH_PID=
orchestrator() {  # extra arguments for aorlta.orchestrator.run; core=0, as every container here: the engines run in it
  stamp "RUN ${ARGS[*]} $*"
  docker run --rm --name "$CNAME" \
    --device=/dev/kfd --device=/dev/dri --group-add video --group-add render \
    --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --ipc=host --ulimit memlock=-1 --ulimit core=0 \
    --ulimit nofile=65535:65535 --network=host "${ORCH_MOUNTS[@]}" \
    -e SLURM_JOB_ID="$SLURM_JOB_ID" -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH="$CODE/src$XPATH" \
    -e HF_HOME="$AORLTA_HF_HOME" -e HF_HUB_OFFLINE=1 -e TMPDIR="$AORLTA_TMPDIR" -e AORLTA_SITE="${AORLTA_SITE:-}" \
    -e AORLTA_AORTA_ROOT="$AORLTA_AORTA_ROOT" -e AORLTA_ROCJITSU_ROOT="$AORLTA_ROCJITSU_ROOT" \
    -e AORLTA_LIBSTDCXX_SHIM="$AORLTA_LIBSTDCXX_SHIM" \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -w "$CODE" \
    --entrypoint python3 "$IMG" -m aorlta.orchestrator.run "${ARGS[@]}" "$@" >> "$OUT/run.log" 2>&1 &
  ORCH_PID=$!
}
orchestrator
if [ -n "$KILL_AFTER_ITER" ]; then
  marker="$RUN_DIR/iterations/$(printf 'iter-%06d' "$KILL_AFTER_ITER")"
  while kill -0 "$ORCH_PID" 2>/dev/null && [ ! -d "$marker" ]; do sleep 5; done
  if kill -0 "$ORCH_PID" 2>/dev/null; then
    sleep "$KILL_DELAY_S"
    if kill -0 "$ORCH_PID" 2>/dev/null; then
      stamp "DRILL: SIGKILL the orchestrator, $KILL_DELAY_S s after iteration $KILL_AFTER_ITER committed"
      docker kill --signal KILL "$CNAME" >/dev/null 2>&1
      wait "$ORCH_PID"
      stamp "DRILL: the killed run exited rc=$?"
      docker rm -f "$CNAME" >/dev/null 2>&1
      [[ " ${ARGS[*]} " == *" --resume "* ]] || ARGS+=(--resume)
      orchestrator
    fi
  fi
fi
wait "$ORCH_PID"
rc=$?
stamp "RUN rc=$rc (rc 0 is completed, rc 3 is stopped; see run_result.json)"
tail -5 "$OUT/run.log"

# Every committed iteration's summary names every span it must have; a score run prints its summary instead.
PYTHONPATH="$CODE/src$XPATH" "$PY" - "$CODE/$CONFIG" <<'EOF' | tee "$OUT/spans-check.txt"
import sys
from aorlta.config import load_config
from aorlta.episodes.writer import complete_iterations, read_summary
cfg = load_config(sys.argv[1])
if cfg.orchestrator.mode == "score":
    from aorlta.orchestrator.score import REPORT_DIR, SUMMARY_NAME
    path = cfg.run_dir / REPORT_DIR / SUMMARY_NAME
    print(path.read_text() if path.is_file() else f"no score summary at {path}\n", end="")
    sys.exit(0)
for n in complete_iterations(cfg.run_dir):
    s = read_summary(cfg.run_dir, n)
    print(f"iteration {n}: scored {s['samples']['scored']}/{s['samples']['total']} missing spans {s['spans']['missing']}"
          f" gap {s['spans']['gap_fraction']:.4f} landing {s['sync']['landing']} wall {s['timing']['wall_s']:.1f} s"
          f" warnings {s['warnings']}")
EOF
stamp "DONE"
exit "$rc"
