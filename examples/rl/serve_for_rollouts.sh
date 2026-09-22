#!/usr/bin/env bash
# Bring up one long-lived TokenSpeed server for a rollout-shaped driver.
#
# `tokenspeed_serve` (the workload) starts a server, benchmarks it and tears it
# down inside one container lifetime, which is right for a measurement and
# wrong for driving `aorta agent` against it: the agent needs an endpoint that
# outlives the process talking to it. So this reuses the workload's serving
# configuration rather than inventing one -- the same pinned image digest, the
# same device/group/shm/network flags from `_docker_argv`, the same
# `HF_HOME=/hf-cache` mount contract, and the same `tokenspeed serve` argv
# `ts_bench_serve.sh` builds -- and only changes the container's lifetime.
#
# Anything measured through this endpoint is therefore comparable with the
# recipes' numbers in configuration, but it is NOT a benchmark: nothing here
# audits served-request counts or guards against a silent pass, which is
# exactly what the workload adds on top. Use the recipes to measure serving,
# and this to drive a consumer.
#
#   examples/rl/serve_for_rollouts.sh up      # start, wait for health_generate
#   examples/rl/serve_for_rollouts.sh hold    # up, then block until signalled
#   examples/rl/serve_for_rollouts.sh models  # what the engine advertises
#   examples/rl/serve_for_rollouts.sh logs
#   examples/rl/serve_for_rollouts.sh down
#
# Under Slurm, use `hold` in its own step and drive from a separate `--overlap`
# step. A detached container started inside an `srun` step belongs to that
# step's cgroup, so Slurm SIGKILLs it when the step ends -- the server dies as
# soon as the bring-up command returns, and what the driver sees is a connection
# refused against an endpoint that was healthy a moment earlier. `hold` keeps
# the step alive for as long as the endpoint is meant to exist.
#
# Environment:
#   TS_MODEL      HF model id to serve   (default Qwen/Qwen3-8B)
#   TS_PORT       gateway port           (default 8000)
#   TS_CONTROL    control port           (default 8001)
#   TS_HF_HOME    host HF cache          (default $HF_HOME, else ~/.cache/huggingface)
#   TS_OUT_DIR    host scratch (/ts-out) (default $TMPDIR/aorta-rl-e2e/ts-out)
#   TS_LOG_DIR    host log directory     (default $TMPDIR/aorta-rl-e2e/logs)
#   TS_GPU        HIP_VISIBLE_DEVICES    (default 0)
#   TS_READY_SEC  readiness deadline     (default 2400)
#   TS_GRAMMAR    --grammar-backend      (default xgrammar; see below)
#   TS_SAMPLING   --sampling-backend     (default triton; see below)
set -euo pipefail

# The digest from recipes/tokenspeed/tokenspeed-serve-rollout.yaml, not the tag
# it resolved from: a moving tag would let the engine change underneath a result
# while this script still read as pinned.
IMAGE="${TS_IMAGE:-lightseekorg/tokenspeed-amd@sha256:60c12e37c01496891053b9c30c4204e5d1cf9b4b641859d3aadcbd95bccc7c78}"
MODEL="${TS_MODEL:-Qwen/Qwen3-8B}"
PORT="${TS_PORT:-8000}"
CONTROL="${TS_CONTROL:-8001}"
# Defaults are deliberately machine-neutral: the model download is large and
# slow, so an HF cache the caller already populated should be reused rather than
# refilled, and everything else is scratch. On a node whose root filesystem is
# small or full, point TMPDIR (or TS_OUT_DIR/TS_LOG_DIR) at the big mount --
# `/tmp` is the fallback, not a recommendation.
HF_HOME_HOST="${TS_HF_HOME:-${HF_HOME:-$HOME/.cache/huggingface}}"
OUT_DIR="${TS_OUT_DIR:-${TMPDIR:-/tmp}/aorta-rl-e2e/ts-out}"
LOG_DIR="${TS_LOG_DIR:-${TMPDIR:-/tmp}/aorta-rl-e2e/logs}"
GPU="${TS_GPU:-0}"
READY_SEC="${TS_READY_SEC:-2400}"
# Defaulted ON here, against the engine's own default of `none`, because
# `LiteLLMProposer.propose` sends `response_format={"type": "json_object"}` on
# every call and TokenSpeed answers that with a 500 --
#   "Grammar-based generation (json_schema, regex, ebnf, structural_tag) is not
#    supported when the server is launched with --grammar-backend none"
# -- so a default-configured server cannot serve `aorta agent` at all. The
# failure is total rather than degraded, and it does not appear in any existing
# `tokenspeed-serve-*` recipe because `tokenspeed bench serve` drives
# /v1/completions and never asks for a response format. The two choices the
# engine offers are `xgrammar` and `none`; there is no third.
GRAMMAR="${TS_GRAMMAR:-xgrammar}"
# Defaulted to `triton` here, against the engine's own default, for the same
# class of reason as the grammar backend above -- and this one is worse, because
# it fails silently.
#
#   tokenspeed/runtime/sampling/registry.py
#     def _get_default_backend_name() -> str:
#         if current_platform().is_nvidia:
#             return "flashinfer"
#         return "greedy"
#
# On this hardware `is_nvidia` is false, so the default resolves to `greedy`.
# The greedy backend ignores `temperature`, `top_p`, `top_k` and `seed`
# entirely: the request is accepted, HTTP 200 comes back, and every completion
# is the argmax. Asking for temperature 1.2 and asking for temperature 0 return
# the same tokens, and `n=8` in one request returns eight identical choices.
#
# Nothing warns. That makes it invisible to any check that reads status codes or
# eyeballs one completion, and it is fatal to a GRPO rollout specifically: the
# group's samples are all the same string, so the advantage is identically zero
# whatever the reward says. The first end-to-end run's "zero within-group
# spread on 9 of 9 groups" had this as a cause, not just a saturated reward.
#
# `--sampling-backend` accepts greedy | triton | triton_full | flashinfer |
# flashinfer_full. flashinfer is CUDA-only, so `triton` is the portable choice
# that honours sampling parameters on ROCm.
#
# Leave this at `greedy` only if you *want* argmax decoding -- which the serving
# recipes reasonably do, since a benchmark wants reproducible output.
SAMPLING="${TS_SAMPLING:-triton}"
NAME="${TS_NAME:-ts-rollout-serve}"

http_code() { curl -s -o /dev/null -w '%{http_code}' --max-time 5 "$1" 2>/dev/null || echo 000; }

# Whether this invocation currently owns the container -- created by its
# `docker run` and not yet handed over. One fact, read by every trap, because
# this leak class has now been found four times and each fix guarded one more
# code location:
#
#   1. `up` exited on a failure path with the container still named
#   2. `hold` armed its traps before `up` checked for a name collision, so a
#      collision removed *another* invocation's server
#   3. plain `up` had no interrupt guard during the readiness wait
#   4. the guard was cleared straight after `health_generate`, leaving
#      `docker logs` / `models` / `backends` unguarded
#
# Every one of those was a window, and guarding a window means the next line
# added outside it opens a fifth. So the guard is keyed on *ownership* instead:
# it is armed once, at the moment the container comes into existence, and
# released at exactly one point, where `up` decides to hand it over. Anything
# that exits in between -- a signal, `teardown_failed`, or an `exit` some later
# change adds -- goes through the same release.
_OWNED=0

# Set by `hold` *before* it calls `up`, to declare that this container is meant
# to outlive bring-up inside this same process rather than be handed to a later
# invocation. It can only cause ownership to be RETAINED, never asserted:
# `_release_owned` stays keyed on `_OWNED`, which is still set at exactly one
# place, just before `docker run -d` -- and it removes only the id in the
# cidfile, which is a path no other invocation can write. So the collision
# path -- where `up` exits because the name belongs to somebody else and this
# invocation created nothing -- is untouched by it, which is the property that
# matters, because removing another invocation's live server is the failure
# this whole guard was introduced to stop.
#
# The caller declares its intent rather than `up` inspecting who called it: the
# subcommand dispatch stays the only thing that knows what subcommand is
# running, and `up` keeps one rule for when ownership ends.
_RETAIN_OWNERSHIP=0

# How far `up` got. This decides only what a release *says* and what it exits
# with -- never whether it happens, which stays keyed on `_OWNED` alone.
_PHASE=bringup

# What this invocation owns, by container **id** rather than by name.
#
# The name was never a safe key and that is why ownership could not be armed
# before `docker run`. The collision check and `docker run` are two steps, so a
# racing invocation can take the name in between; a guard keyed on the name
# would then remove *their* container on the way out, which is the failure this
# whole mechanism exists to prevent and is worse than the leak it would fix.
#
# `--cidfile` removes the ambiguity rather than managing it. Measured against
# docker 29.1.3 rather than reasoned about, because the whole argument rests on
# when the file is written:
#
#   * a successful run writes the id, identical to `docker run`'s own stdout
#   * a **name collision** writes NOTHING (rc 125) -- so a racing invocation's
#     container is unreachable from here by construction, not by care
#   * a container that is created and then fails to *start* DOES get a cidfile,
#     and removing that id reclaims it -- which is the `docker run -d` window
#     this script has carried as a known leak
#   * an argument error writes nothing, so there is nothing to clean up
#
# So ownership can now be armed *before* `docker run`, closing the in-flight
# window too, and the guarantee is stronger than it was: an empty cidfile means
# we remove nothing at all, which is exactly right.
#
# **The path has to be per-invocation, and a shared one reopened the bug the
# cidfile was introduced to close.** `LOG_DIR` is keyed on `TS_LOG_DIR`, not on
# `TS_NAME`, so two runs with different names shared one `container.cid`. The
# reasoning above survives a *name* collision -- docker writes nothing -- but it
# says nothing about a *cidfile* collision, where docker refuses with rc 125
# ("container ID file found") and the file still holds the id the other
# invocation put there. Reproduced against a stub docker: the script ran
# `docker rm -f` on a peer's live server, which is precisely the outcome the
# whole guard exists to stop, arrived at from the other side.
#
# `$$` rather than `${NAME}`: the name is what a peer might be sharing, and two
# invocations of the same `TS_NAME` are exactly the case where one must not
# clear the other's file. A pid cannot collide with a live peer, and the stale
# `rm -f` below still covers a recycled pid from a dead one.
_CIDFILE=""
_CID=""

_release_owned() {
  [ "${_OWNED}" = "1" ] || return 0
  local cid="${_CID}"
  # The in-memory id first: it survives the file being removed, and the file
  # only exists to carry the id back from a `docker run` that may not have
  # returned yet.
  if [ -z "${cid}" ] && [ -n "${_CIDFILE}" ] && [ -s "${_CIDFILE}" ]; then
    cid="$(cat "${_CIDFILE}" 2>/dev/null || true)"
  fi
  if [ -z "${cid}" ]; then
    # Armed, but docker never created anything for us -- a collision or an
    # argument error. Deliberately *no* fallback to removing `${NAME}`: the
    # container wearing that name is then someone else's, and the fallback is
    # the bug.
    _OWNED=0
    return 0
  fi
  if [ "${_PHASE}" = "hold" ]; then
    echo "signalled; tearing down" >&2
  else
    echo "bring-up did not complete; removing ${NAME} (${cid:0:12})" >&2
  fi
  docker rm -f "${cid}" >/dev/null 2>&1 || true
  rm -f "${_CIDFILE}" 2>/dev/null || true
  _OWNED=0
}

# Give up on a container this invocation started, after saving what it said.
#
# Every failure path in `up` below is reached *after* `docker run -d` has
# succeeded, and each used to exit with the container still named. Two things
# followed. The next `up` died with "container already exists; run 'down'
# first", so the script needed a manual `down` before it would work again --
# and on the readiness and health_generate paths the process is still
# *running*, so an engine nobody is driving kept its GPU. On a shared node that
# is the expensive half: the allocation looks busy and the next job cannot have
# the device.
#
# Logs are captured before removal, because `docker rm -f` takes them with it
# and a bring-up failure with no logs is not diagnosable.
teardown_failed() {  # teardown_failed <exit-code> <message>
  local code="$1" msg="$2"
  # By id where we have one, for the same reason `_release_owned` is: every
  # call site here is downstream of a successful `docker run`, so the name is
  # ours in practice -- but "in practice" is what the cidfile exists to stop
  # this script relying on.
  local ref="${_CID:-${NAME}}"
  echo "FAIL: ${msg}" >&2
  docker logs --tail 60 "${ref}" > "${LOG_DIR}/server-failure.log" 2>&1 || true
  tail -n 60 "${LOG_DIR}/server-failure.log" >&2 2>/dev/null || true
  docker rm -f "${ref}" >/dev/null 2>&1 || true
  rm -f "${_CIDFILE}" 2>/dev/null || true
  # Ownership ends here, so the EXIT trap does not attempt a second removal.
  _OWNED=0
  exit "${code}"
}

up() {
  local rc
  mkdir -p "${LOG_DIR}" "${HF_HOME_HOST}/hub" "${OUT_DIR}/triton-cache" "${OUT_DIR}/home"
  if [ -n "$(docker ps -aq -f "name=^${NAME}$")" ]; then
    echo "container ${NAME} already exists; run 'down' first" >&2
    exit 1
  fi

  # The cache/identity env block is `tokenspeed_serve`'s, copied whole rather
  # than reconstructed. Every entry is load-bearing under `--user`, and each one
  # fails in a way that names something other than itself:
  #
  #   HOME, TRITON_CACHE_DIR, XDG_CACHE_HOME
  #       the image's HOME is `/`, unwritable by an ordinary uid, and Triton
  #       masks the EACCES as "Triton is not supported on the current
  #       platform" -- a broken-GPU-stack message on a node whose GPUs are fine
  #   TORCHINDUCTOR_CACHE_DIR, USER, LOGNAME
  #       torch's `cache_dir()` only computes a default when the first is
  #       unset, and that default calls `getpass.getuser()`, which has no
  #       passwd entry for the uid and raises `KeyError: getpwuid()` at import
  #       of `torch._dynamo`
  #
  # Rediscovering this list costs two failed bring-ups, which is what it cost
  # here before the workload's own copy was read.
  #
  # The engine's arguments are passed as positional parameters to `bash -c`,
  # not interpolated into its script text. Every one of them --  MODEL, PORT,
  # CONTROL, GRAMMAR, SAMPLING, READY_SEC -- comes from a caller-settable
  # environment variable, and interpolating them meant the container's shell
  # reparsed their contents: a quote or a `;` in any of them ran a different
  # command inside a `--network host` container with /dev/kfd and
  # seccomp=unconfined. As positional `$1`..`$6` the shell never reparses them,
  # so the values stay data whatever they contain. `_` fills `$0`.
  #
  # --gateway-startup-timeout for the reason ts_bench_serve.sh raises it: the
  # orchestrator's own default is 60s, and a cold start blows through it while
  # the engine is still loading weights and compiling kernels. Left at our
  # readiness deadline so a slow start reports as slow rather than as a dead
  # engine.
  # Ownership is armed *before* the container can exist, not after.
  #
  # It used to be armed on the line after `docker run -d` returned, which left
  # two windows: a signal while docker was creating the container, and a
  # `docker run` that fails *after* creating it. Neither could be closed while
  # the guard was keyed on `${NAME}`, because between the collision check above
  # and this call a racing invocation can take the name -- so an early guard
  # would have removed their container. Keyed on the cidfile it cannot: a name
  # collision writes no cidfile, so the release below finds nothing and removes
  # nothing. See the `_CIDFILE` comment for the measurements.
  #
  # The stale-file clear is required, not hygiene: docker refuses to run at all
  # when the cidfile already exists ("container ID file found"), so a leftover
  # file would break bring-up. It is safe because the path carries this
  # process's own pid, so the only file it can ever remove is one this pid
  # wrote -- which, since we are still before our own `docker run`, means a
  # recycled pid from a process that is gone. It used to be
  # `${LOG_DIR}/container.cid`, shared with every concurrent invocation, and
  # that made this line able to delete a peer's file; see `_CIDFILE`.
  _CIDFILE="${LOG_DIR}/${NAME}.$$.cid"
  rm -f "${_CIDFILE}"
  _OWNED=1
  # `if` rather than `&& exit 0`: a failing `[` as the last command of an
  # `&&` list is itself a non-zero status, which under `set -e` would exit the
  # trap with 1 and never reach the 130.
  trap '_release_owned; if [ "${_PHASE}" = "hold" ]; then exit 0; fi; exit 130' INT TERM
  trap '_release_owned' EXIT

  docker run -d \
    --cidfile "${_CIDFILE}" \
    --name "${NAME}" \
    --device /dev/kfd \
    --device /dev/dri \
    --group-add video \
    --group-add render \
    --security-opt seccomp=unconfined \
    --network host \
    --shm-size 16g \
    -v "${HF_HOME_HOST}:/hf-cache" \
    -v "${OUT_DIR}:/ts-out" \
    --user "$(id -u):$(id -g)" \
    -e HF_HOME=/hf-cache \
    -e HF_HUB_CACHE=/hf-cache/hub \
    -e TRITON_CACHE_DIR=/ts-out/triton-cache \
    -e HOME=/ts-out/home \
    -e XDG_CACHE_HOME=/ts-out/home/.cache \
    -e TORCHINDUCTOR_CACHE_DIR=/ts-out/torchinductor \
    -e USER="$(id -un)" \
    -e LOGNAME="$(id -un)" \
    -e HIP_VISIBLE_DEVICES="${GPU}" \
    --entrypoint bash \
    "${IMAGE}" -c 'exec tokenspeed serve "$1" \
      --host 127.0.0.1 --port "$2" --control-port "$3" \
      --grammar-backend "$4" \
      --sampling-backend "$5" \
      --gateway-startup-timeout "$6"' \
    _ "${MODEL}" "${PORT}" "${CONTROL}" "${GRAMMAR}" "${SAMPLING}" "${READY_SEC}" \
    > "${LOG_DIR}/container-id.txt"
  # Read back into memory so the id survives the file: a later `rm` of the
  # cidfile, or a concurrent invocation sharing this LOG_DIR, cannot then strand
  # a container we own with nothing able to name it.
  _CID="$(cat "${_CIDFILE}" 2>/dev/null || true)"
  echo "started ${NAME} ($(cut -c1-12 "${LOG_DIR}/container-id.txt"))"

  # Poll health, but re-check liveness every iteration: a crash during weight
  # load has to surface as its own failure instead of burning the whole
  # readiness budget and then being reported as a timeout.
  local t0 deadline
  t0=$(date +%s)
  deadline=$(( t0 + READY_SEC ))
  while [ "$(date +%s)" -lt "${deadline}" ]; do
    if [ -z "$(docker ps -q -f "name=^${NAME}$")" ]; then
      teardown_failed 50 "container exited during startup after $(( $(date +%s) - t0 ))s"
    fi
    if [ "$(http_code "http://127.0.0.1:${CONTROL}/health")" = "200" ]; then
      echo "OK: /health after $(( $(date +%s) - t0 ))s"
      break
    fi
    sleep 5
  done
  if [ "$(http_code "http://127.0.0.1:${CONTROL}/health")" != "200" ]; then
    # Still running, so this is the path where the leak cost a GPU rather than
    # just an awkward name collision.
    teardown_failed 51 "readiness timeout after $(( $(date +%s) - t0 ))s"
  fi

  # /health_generate pushes a real token through the engine. /health only proves
  # the port is bound, and driving a bound-but-unwired gateway would report a
  # wall of failed requests instead of a bring-up failure.
  for _ in $(seq 1 20); do
    if [ "$(http_code "http://127.0.0.1:${CONTROL}/health_generate")" = "200" ]; then
      echo "OK: /health_generate after $(( $(date +%s) - t0 ))s"
      # Bring-up is done, so the interrupt guard stops applying: a successful
      # `up` leaves the engine running on purpose, and `hold` installs its own
      docker logs "${NAME}" > "${LOG_DIR}/server.log" 2>&1 || true
      # Informational here, and explicitly so. `models` reports a failed or
      # unreadable fetch as 56, which is right for the subcommand and wrong as
      # a bring-up gate: the endpoint has already answered /health_generate,
      # and what decides whether this engine is usable is `backends` below.
      # Left bare it would be a third reading of the same defect -- an
      # unchecked non-zero under `set -e`, exiting through `_release_owned` and
      # losing the logs -- so the tolerance is stated rather than implied.
      #
      # 58 is not that. It means the list was read and `${MODEL}` is not on it,
      # which is a determinate misconfiguration rather than an unanswered
      # question: the prefix is stripped on the wire, so every rollout request
      # 404s against an engine that passes every health check. Handing that
      # container over is the expensive failure -- the driver reports a wall of
      # failed requests and the name is nowhere in the error -- so it is torn
      # down here, where the cause has a FAIL line. `|| rc=$?` for the reason
      # `backends` uses it below: a bare call would exit at this line.
      rc=0
      models || rc=$?
      if [ "${rc}" -eq 58 ]; then
        teardown_failed 58 "engine does not advertise ${MODEL}; see the FAIL line above"
      fi
      # An engine that ignores the sampling parameters, or that cannot answer a
      # `response_format` request at all, is useless for the one job this
      # server exists to do -- so bring-up has failed even though every health
      # check passed. Torn down rather than left running, for the reason the
      # other failure paths are: an engine nobody can use still holds a GPU.
      #
      # `backends`'s own status is propagated rather than a constant. It was
      # `teardown_failed 57 "...sampling_backend=greedy..."`, which meant a
      # grammar failure would have been reported as a greedy sampler under the
      # wrong exit code -- one verdict name for two defects, which is the
      # mistake this script has already been corrected for once. The specific
      # cause is on the FAIL line `backends` printed.
      #
      # `backends || rc=$?` rather than `backends; rc=$?`, and the difference
      # is the whole branch: `set -e` is in force, so a bare call to a function
      # that returns non-zero exits the shell *at that line*. `rc=$?` was never
      # reached and neither was the `teardown_failed` below -- the EXIT trap ran
      # `_release_owned` instead, which removes the container but does not
      # capture `docker logs`, so a bring-up that failed here left no
      # `server-failure.log` and no FAIL line naming what was wrong. The exit
      # code happened to survive, because bash propagates the failing command's
      # status through an EXIT trap, which is why this looked correct.
      #
      # A `||` list suppresses `set -e` for its left-hand side, so the status
      # can be read and `backends`'s own code -- 57 for sampling, 59 for
      # grammar -- still reaches the caller.
      rc=0
      backends || rc=$?
      if [ "${rc}" -ne 0 ]; then
        teardown_failed "${rc}" "engine is not usable for rollouts; see the FAIL line above"
      fi
      # Handover. Bring-up is complete, so `up` on its own releases the guard
      # here -- the single place it is released -- and the container is handed
      # to a later invocation.
      #
      # `hold` is the other case, and it used to go through this same release:
      # ownership was cleared and all three traps removed here, and `hold`
      # installed replacements on the command after `up` returned. A signal in
      # that interval exited with the container running and nothing registered
      # to remove it, which is precisely the leak this guard exists to prevent,
      # reintroduced by the handover itself.
      #
      # So ownership is *transferred* rather than cleared-then-reinstalled: the
      # traps armed at `docker run -d` stay armed, unbroken, for the life of the
      # process. There is no interval to hit because no trap is ever removed.
      # `hold` therefore installs none of its own, and `_PHASE` only changes
      # what a release prints and exits with.
      if [ "${_RETAIN_OWNERSHIP}" = "1" ]; then
        _PHASE=hold
      else
        _OWNED=0
        trap - INT TERM EXIT
        # The handover is the one exit from `up` that reaches neither
        # `_release_owned` nor `teardown_failed`, so it was the one path that
        # left its cidfile behind. Harmless while the path was shared and a
        # slow leak now that it carries a pid -- one file per successful
        # bring-up, in a directory nothing prunes. The id is already in `_CID`
        # (read back above for exactly this reason), so the file has no reader
        # left.
        rm -f "${_CIDFILE}" 2>/dev/null || true
      fi
      return 0
    fi
    sleep 5
  done
  teardown_failed 52 "health_generate unhealthy"
}

# The name the engine advertises, which is what has to match the litellm model
# id minus its `openai/` routing prefix. Checked rather than assumed: the prefix
# is stripped on the wire, so a mismatch here is a 404 on every request and
# reads as a broken gateway rather than a naming error.
#
# Reports rather than decides, for the reason `backends` does: `up` calls it
# for the record on a path where the gateway has already answered
# /health_generate, while `serve_for_rollouts.sh models` is a caller asking
# whether the endpoint is there. A failed fetch used to print "(no response)"
# and return 0 to both of them, so the subcommand answered "fine" for a gateway
# that said nothing -- absence of evidence as favourable evidence, the same
# shape as `backends`'s old `|| echo '{}'` and as the ownership guard's earlier
# windows.
#
# Two ways this can fail and both have to count. `curl` exits 0 for any HTTP
# response it manages to read, including a 500 -- so the transport check below
# catches a dead socket and used to let an error *page* through as the answer,
# printing it under "advertised models" and returning 0. That is the same
# absence-as-evidence shape one layer up from the `(no response)` bug this
# function was already corrected for: first the fetch failing was read as fine,
# now the fetch succeeding at fetching a failure was.
#
# `%{http_code}` rather than `--fail`/`--fail-with-body`: `--fail` suppresses
# the body, which is where the reason is, and `--fail-with-body` needs curl
# 7.76+ and this runs inside whatever the engine image ships.
#
# **And a 200 is not the answer this function's own docstring asks for.** The
# first paragraph says the advertised name "has to match the litellm model id"
# -- then the check stopped at the status line and never read the list, so an
# empty `data`, a malformed body, or a list advertising some *other* model all
# printed under "advertised models" and returned 0. That is the third turn of
# the same screw: first a failed fetch read as fine, then a fetched failure
# read as fine, and now a successful fetch of a list that does not contain what
# we asked for. Each time the check verified that something came back rather
# than that it said what it had to.
#
# It matters most on the path where it is cheapest to get wrong: `up` calls
# this after /health_generate has answered, so the engine is genuinely healthy
# and merely serving a different name -- and every rollout request then 404s
# with a gateway that looks fine at every other level.
#
# Ids are extracted rather than the body grepped for `${MODEL}`, because a
# substring match is satisfied by the wrong things: a model id that merely
# *contains* ours, or our name appearing in an error message. Same `tr`/`sed`
# shape as `_backend_field`, and whitespace-tolerant for the same reason.
_model_ids() {  # _model_ids <json>
  printf '%s' "$1" \
    | tr ',{}[]' '\n\n\n\n\n' \
    | sed -n 's/.*"id"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p'
}

models() {
  echo "--- advertised models (${PORT}) ---"
  # Declared apart from the assignment on purpose. `local body="$(cmd)"` takes
  # its status from `local`, not from `cmd`, so a failing fetch would look
  # successful -- the same `set -e` trap this file has now been corrected for
  # three times, and it would have been a fourth.
  local body status rc ids
  rc=0
  body="$(curl -s --max-time 10 -w '\n%{http_code}' \
    "http://127.0.0.1:${PORT}/v1/models")" || rc=$?
  if [ "${rc}" -ne 0 ]; then
    echo
    echo "FAIL: no response from ${PORT}/v1/models, so the advertised model id" \
         "is unverified" >&2
    return 56
  fi
  status="${body##*$'\n'}"
  body="${body%$'\n'*}"
  printf '%s\n' "${body}"
  if [ "${status}" != "200" ]; then
    echo "FAIL: ${PORT}/v1/models answered HTTP ${status}, so the advertised" \
         "model id is unverified" >&2
    return 56
  fi
  ids="$(_model_ids "${body}")"
  if [ -z "${ids}" ]; then
    echo "FAIL: ${PORT}/v1/models answered HTTP 200 with no model id in the" \
         "body, so the advertised model id is unverified" >&2
    return 56
  fi
  # Two codes, not one, and the difference is what `up` does with them. 56 is
  # *unverified* -- nothing came back, or what came back cannot be read -- and
  # `up` tolerates it because the gateway has already answered
  # /health_generate and `backends` is what decides usability. This is the
  # other thing: the list was read and it does not have us in it, which is not
  # an unanswered question but a determinate wrong answer. Every rollout
  # request against this engine 404s, so bring-up has failed.
  if ! printf '%s\n' "${ids}" | grep -qxF -- "${MODEL}"; then
    echo "FAIL: ${PORT}/v1/models advertises" \
         "[$(printf '%s' "${ids}" | tr '\n' ' ')] and not ${MODEL}, so every" \
         "rollout request would 404 on a healthy-looking gateway" >&2
    return 58
  fi
  echo "OK: ${MODEL} is advertised"
}

# The two backends this script overrides, read back from the engine rather than
# assumed to have applied. Both defaults are wrong for driving `aorta agent`,
# and `sampling_backend` is the one that cannot be caught any other way: a
# greedy engine answers every sampled request with HTTP 200 and the argmax, so
# there is no failure to notice downstream -- only completions that are all
# identical, which reads as a model property rather than a server setting.
# Returns non-zero when the engine reports greedy and greedy was not asked for.
# It only *reports*; the caller decides what that means, because the two callers
# want different things -- `up` owns a container and must tear it down, while
# the standalone `backends` command does not own one and must not touch it.
# One field out of /get_server_info, whitespace-tolerant.
#
# Factored out so the two backends are read the same way. The sampling check
# originally matched `'"sampling_backend":"greedy"'` with no space, which a
# pretty-printed response defeats -- `json.dumps` writes `": "` by default --
# and the grammar check added next would otherwise have been a second place to
# get that wrong.
_backend_field() {  # _backend_field <name> <json>
  printf '%s' "$2" \
    | tr ',{}' '\n\n\n' \
    | sed -n "s/.*\"$1\"[[:space:]]*:[[:space:]]*\"\\([^\"]*\\)\".*/\\1/p" \
    | head -n 1
}

backends() {
  local info reported grammar
  # No `|| echo '{}'`. Defaulting a failed fetch to an empty object made
  # `greedy` count 0, so a curl failure, a timeout or an empty body read
  # exactly like a healthy non-greedy engine -- the absence of evidence
  # encoded as favourable evidence, which is the one thing this check exists
  # to prevent. A fetch that did not work is now a fetch that did not work.
  if ! info=$(curl -s --max-time 10 "http://127.0.0.1:${CONTROL}/get_server_info"); then
    echo "FAIL: could not read ${CONTROL}/get_server_info, so the sampling" \
         "backend is unverified; refusing to report the engine as usable" >&2
    return 57
  fi
  echo "--- backends in effect (${CONTROL}) ---"
  echo "${info}" | tr ',' '\n' | grep -E '"(sampling_backend|grammar_backend)"' \
    || echo "(no backend fields in the response)"

  # Whitespace-tolerant, and extracting the value rather than testing for one
  # spelling of it. The old `grep -c '"sampling_backend":"greedy"'` required
  # compact JSON: `json.dumps` writes `": "` by default and any pretty-printed
  # response has a space, so a genuinely greedy engine could answer in a shape
  # this check read as non-greedy. The test stub happened to emit the compact
  # form, which is why the narrow match looked fine.
  reported=$(_backend_field sampling_backend "${info}")
  if [ -z "${reported}" ]; then
    # Present-but-unreadable is the same class as unfetchable: we asked and
    # cannot say, so we must not say "fine".
    echo "FAIL: ${CONTROL}/get_server_info reported no sampling_backend, so it" \
         "is unverified; refusing to report the engine as usable" >&2
    return 57
  fi
  if [ "${reported}" = "greedy" ] && [ "${SAMPLING}" != "greedy" ]; then
    # Fatal to the caller, not a warning, and the aorta side already treats it
    # that way: `tokenspeed_serve` fails the step with exit 57,
    # `rollout_sampling_ignored`, on exactly this reading. Warning here while
    # failing there meant two halves of one defect behaved differently, which
    # is invisible unless someone reads both -- and is how it survived.
    #
    # The severity is the point rather than the tidiness. A greedy engine
    # answers every sampled request with HTTP 200 and the argmax, so a rollout
    # driven against it yields a group of identical completions and an
    # identically zero advantage: the whole run is wasted and nothing in it
    # looks wrong. Leaving the server up to be driven anyway is the failure.
    echo "FAIL: asked for --sampling-backend ${SAMPLING} but the engine reports" \
         "'greedy'; sampling parameters are silently ignored, so every" \
         "completion in a group would be the argmax and the advantage zero" >&2
    echo
    return 57
  fi

  # The grammar backend, checked for the same reason and with a larger blast
  # radius than the sampling one. `LiteLLMProposer.propose` sends
  # `response_format={"type": "json_object"}` on *every* call, and an engine on
  # `--grammar-backend none` answers that with a 500 -- so the failure is
  # total rather than degraded, and `up` was reporting such a server as ready.
  # A greedy sampler at least returns text; this returns nothing usable at all.
  #
  # Exit 59 rather than reusing 57. 57 means "the engine ignored the sampling
  # parameters", which is a different statement, and this script has already
  # been corrected once for labelling two defects with one verdict name.
  # Deliberately not 58 either: that is `rollout_sampling_backend_mismatch` on
  # the aorta side, and a reader comparing the two should not meet one number
  # with two meanings.
  grammar=$(_backend_field grammar_backend "${info}")
  if [ -z "${grammar}" ]; then
    echo "FAIL: ${CONTROL}/get_server_info reported no grammar_backend, so it" \
         "is unverified; refusing to report the engine as usable" >&2
    return 59
  fi
  if [ "${grammar}" = "none" ] && [ "${GRAMMAR}" != "none" ]; then
    echo "FAIL: asked for --grammar-backend ${GRAMMAR} but the engine reports" \
         "'none'; every request carrying a response_format returns 500, so" \
         "aorta agent cannot be served at all" >&2
    echo
    return 59
  fi
  echo
}

hold() {
  # Tear the container down on the way out, so a cancelled step does not leave
  # a live engine holding a GPU that the next allocation cannot use.
  #
  # This used to install its own traps, after `up` returned, and the ordering
  # was the whole point: arming them *before* `up` was dangerous, because `up`
  # refuses a name that is already taken and refuses by exiting, so on a
  # collision with another invocation's live server the trap fired on the way
  # out and removed *their* container -- trading a leak for destroying someone
  # else's running engine, which is worse by a distance.
  #
  # Both orderings were wrong, for opposite reasons, and that is the tell that
  # the question was the wrong one. Arming late leaves the handover window a
  # signal can land in; arming early attributes a container this invocation did
  # not create. Neither is a question about *when*, because both answers are
  # about *whose*, and `_OWNED` already records whose.
  #
  # So `hold` installs no traps at all now. It declares that it wants the
  # container to outlive bring-up, and `up` keeps the ownership-keyed traps it
  # armed at `docker run -d` armed rather than clearing them. On the collision
  # path `_OWNED` is still 0, because `up` exits before it is ever set, so this
  # declaration removes nothing that belongs to anyone else.
  _RETAIN_OWNERSHIP=1
  up
  echo "holding ${NAME}; endpoint http://127.0.0.1:${PORT}/v1"
  while [ -n "$(docker ps -q -f "name=^${NAME}$")" ]; do
    sleep 10
  done
  echo "container ${NAME} is no longer running; releasing hold" >&2
  exit 1
}

case "${1:-up}" in
  up) up ;;
  hold) hold ;;
  models) models ;;
  # Exits non-zero on a greedy engine, so this is usable as a precondition
  # check -- but removes nothing: this invocation did not create the container
  # and must not delete one it does not own.
  backends) backends ;;
  logs) docker logs "${@:2}" "${NAME}" ;;
  # "no ${NAME}" used to be printed for two different things: there was nothing
  # to remove, and the removal failed. A `docker rm -f` that fails because the
  # daemon is unreachable reported success and exit 0 with the container still
  # running -- so the one command whose job is to release a GPU could fail to
  # release it and say so in words that mean it was already free. The probe
  # that distinguishes them has to be checked too, for the same reason: a
  # `docker ps` that cannot reach the daemon returns empty, and reading empty as
  # "nothing there" is how the first version got it wrong.
  down)
    if docker rm -f "${NAME}" >/dev/null 2>&1; then
      echo "removed ${NAME}"
    elif ! existing=$(docker ps -aq -f "name=^${NAME}$" 2>/dev/null); then
      echo "FAIL: cannot reach the docker daemon, so whether ${NAME} is still" \
           "running is unknown; not reporting it as removed" >&2
      exit 55
    elif [ -n "${existing}" ]; then
      echo "FAIL: ${NAME} still exists and could not be removed" >&2
      exit 55
    else
      echo "no ${NAME}"
    fi
    ;;
  *) echo "usage: $0 {up|hold|models|backends|logs|down}" >&2; exit 64 ;;
esac
