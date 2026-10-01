# Run AORTA Chat with a dedicated Qwen vLLM server

This runbook starts `Qwen/Qwen3.8-27B` on a Slurm GPU node and then starts the
AORTA Chat UI on port 8080. It is written for the Ruby cluster and the checkout
at `/apps/avsharma/aorta-cia-qwen38`.

The two ports have different jobs:

- **8080** serves the Chainlit web UI.
- **8001 or higher** serves the vLLM model API.

The commands below never probe or connect to `localhost:8000`. The Slurm job
starts its search at 8001, selects the first free port on the allocated GPU
node, starts vLLM on that port, and records the exact remote URL in a shared
environment file. The UI refuses to start unless that file exists, so omitting
the model-server steps cannot silently restore AORTA's built-in
`http://localhost:8000/v1` default.

The vLLM job reserves one whole GPU node for three hours. Cancel it when you
finish.

## Two-command quickstart

From `ruby-slurmlogin02`, start or reuse a healthy Qwen service:

```bash
cd /apps/avsharma/aorta-cia-qwen38
git switch cia_models
scripts/chat/start_qwen_vllm.sh
```

The command returns after `/health`, `/v1/models`, a real chat completion, and
a native function-call probe all pass. It enables Qwen's XML tool parser,
prints the selected compute-node URL, and never inspects port 8000.
Startup output is limited to Slurm state changes and one progress message every
30 seconds; add `--verbose` to print every five-second poll.
Shutdown follows the same policy instead of printing every `COMPLETING` poll.

Start the CIA-enabled UI:

```bash
scripts/chat/start_aorta_chat.sh
```

The launcher explicitly selects the `vllm` provider and native tool protocol,
overriding any different provider saved in the user's chat profile.
It prints a startup message immediately and uses a per-port lock, so retrying
the command cannot silently launch a second Chainlit server on the same port.

Open <http://127.0.0.1:8080>. Keep the second command running. `Ctrl-C` stops
the remote UI and SSH tunnel without leaving either process behind.
If the terminal is closed or killed before its cleanup trap runs, recover with:

```bash
scripts/chat/start_aorta_chat.sh --stop
```

When finished, release the Qwen GPU node:

```bash
scripts/chat/start_qwen_vllm.sh --stop
```

The remainder of this guide expands those two scripts into individual commands
for troubleshooting or changing their defaults. Run either launcher with
`--help` to see its supported overrides.

## 1. Prepare the AORTA checkout

Run these commands from `ruby-slurmlogin02`:

```bash
cd /apps/avsharma/aorta-cia-qwen38
git switch cia_models

# Create the environment only once.
if [ ! -x .venv-ui/bin/aorta ]; then
  python3.13 -m venv .venv-ui
  .venv-ui/bin/python -m pip install --upgrade pip
  .venv-ui/bin/python -m pip install -e ".[chat-ui,cia]"
fi

source .venv-ui/bin/activate
aorta chat --help
```

Create shared directories for the Hugging Face cache and runtime files:

```bash
mkdir -p /apps/avsharma/hf-cache
mkdir -p /apps/avsharma/aorta-chat-runtime
```

If you will enable CIA cluster submissions, install the current RocJITsu
sanitizer bundle. The downloader verifies the artifact's SHA-256 manifest and
uses `gh auth token`; run `gh auth login` first if it reports that no token is
available.

```bash
ROCJITSU_PREBUILT=/apps/avsharma/aorta-chat-runtime/rocjitsu-prebuilt
ROCJITSU_BUILD=/apps/avsharma/aorta-chat-runtime/rocjitsu-build

python scripts/sanitizers/download_sanitizer_artifacts.py \
  --run latest \
  --dest "${ROCJITSU_PREBUILT}" \
  --force

# Chat currently passes the raw-build layout to submitted CIA jobs. Present
# the verified prebuilt files under that layout without copying them.
mkdir -p "${ROCJITSU_BUILD}/lib/rocjitsu/src/rocjitsu/hooks"
mkdir -p "${ROCJITSU_BUILD}/tools"
ln -sfn \
  "${ROCJITSU_PREBUILT}/lib/librocjitsu_dbi_hooks.so" \
  "${ROCJITSU_BUILD}/lib/rocjitsu/src/rocjitsu/hooks/librocjitsu_dbi_hooks.so"
ln -sfn \
  "${ROCJITSU_PREBUILT}/bin/rj_waitcheck" \
  "${ROCJITSU_BUILD}/tools/rj_waitcheck"

test -f \
  "${ROCJITSU_BUILD}/lib/rocjitsu/src/rocjitsu/hooks/librocjitsu_dbi_hooks.so"
test -x "${ROCJITSU_BUILD}/tools/rj_waitcheck"
"${ROCJITSU_BUILD}/tools/rj_waitcheck" --help >/dev/null
ldd -r \
  "${ROCJITSU_BUILD}/lib/rocjitsu/src/rocjitsu/hooks/librocjitsu_dbi_hooks.so"
```

## 2. Create the vLLM Slurm script

Still on `ruby-slurmlogin02`, create the job script on the shared filesystem:

```bash
cat > /apps/avsharma/aorta-chat-runtime/qwen38-vllm.sbatch <<'SBATCH'
#!/usr/bin/env bash
#SBATCH --job-name=aorta-qwen38
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=03:00:00
#SBATCH --output=/apps/avsharma/aorta-chat-runtime/qwen38-vllm-%j.log

set -euo pipefail

RUNTIME_DIR=/apps/avsharma/aorta-chat-runtime
ENDPOINT_FILE="${RUNTIME_DIR}/qwen38-endpoint.env"
HF_CACHE=/apps/avsharma/hf-cache

mkdir -p "${RUNTIME_DIR}" "${HF_CACHE}"

# Check ports on the allocated GPU node without connecting to anything.
# Port 8000 is intentionally excluded because it belongs to another service.
PORT="$(
  python3 - <<'PY'
import socket

for port in range(8001, 8101):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("0.0.0.0", port))
    except OSError:
        sock.close()
        continue
    sock.close()
    print(port)
    break
else:
    raise SystemExit("no free model port in the range 8001-8100")
PY
)"

HOST="$(hostname -f)"
TMP_ENDPOINT="${ENDPOINT_FILE}.${SLURM_JOB_ID}.tmp"

umask 077
cat > "${TMP_ENDPOINT}" <<EOF
export QWEN_VLLM_JOB_ID="${SLURM_JOB_ID}"
export AORTA_CHAT_VLLM_BASE_URL="http://${HOST}:${PORT}/v1"
export AORTA_CHAT_VLLM_MODEL="Qwen/Qwen3.8-27B"
EOF
mv "${TMP_ENDPOINT}" "${ENDPOINT_FILE}"

cleanup() {
  if [ -f "${ENDPOINT_FILE}" ] &&
     grep -Fq "QWEN_VLLM_JOB_ID=\"${SLURM_JOB_ID}\"" "${ENDPOINT_FILE}"; then
    rm -f "${ENDPOINT_FILE}"
  fi
}
trap cleanup EXIT

echo "vLLM endpoint: http://${HOST}:${PORT}/v1"

docker pull rocm/vllm-dev:nightly
docker run --rm \
  --name "aorta-qwen38-${SLURM_JOB_ID}" \
  --network host \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --ipc=host \
  --shm-size 32g \
  -e HIP_VISIBLE_DEVICES=0 \
  -e HF_HOME=/root/.cache/huggingface \
  -v "${HF_CACHE}:/root/.cache/huggingface" \
  rocm/vllm-dev:nightly \
  vllm serve Qwen/Qwen3.8-27B \
    --served-model-name Qwen/Qwen3.8-27B \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 32768 \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    --default-chat-template-kwargs '{"enable_thinking":false}'
SBATCH

chmod 700 /apps/avsharma/aorta-chat-runtime/qwen38-vllm.sbatch
```

## 3. Submit the model server

First stop a model job left by an earlier run of this guide. This only cancels
the job ID recorded in AORTA's own endpoint file; it does not inspect or contact
port 8000.

```bash
ENDPOINT_FILE=/apps/avsharma/aorta-chat-runtime/qwen38-endpoint.env

if [ -s "${ENDPOINT_FILE}" ]; then
  source "${ENDPOINT_FILE}"
  if [ -n "$(
    ssh -o BatchMode=yes ruby-slurmlogin01.rckg.g03.cpe.ice.amd.com \
      "squeue -h -j ${QWEN_VLLM_JOB_ID} -o '%i'"
  )" ]; then
    echo "Cancelling previous vLLM job ${QWEN_VLLM_JOB_ID}"
    ssh -o BatchMode=yes ruby-slurmlogin01.rckg.g03.cpe.ice.amd.com \
      "scancel ${QWEN_VLLM_JOB_ID}"
  fi
  rm -f "${ENDPOINT_FILE}"
fi
```

Submit through the login node that can reach the Slurm controller:

```bash

QWEN_JOB_ID="$(
  ssh -o BatchMode=yes ruby-slurmlogin01.rckg.g03.cpe.ice.amd.com \
    "sbatch --parsable /apps/avsharma/aorta-chat-runtime/qwen38-vllm.sbatch"
)"

echo "Submitted vLLM job ${QWEN_JOB_ID}"
```

Wait for Slurm to allocate a node and for the job to publish its endpoint:

```bash
ENDPOINT_FILE=/apps/avsharma/aorta-chat-runtime/qwen38-endpoint.env

until [ -s "${ENDPOINT_FILE}" ]; do
  JOB_STATUS="$(
    ssh -o BatchMode=yes ruby-slurmlogin01.rckg.g03.cpe.ice.amd.com \
      "squeue -h -j ${QWEN_JOB_ID} -o '%.18i %.10T %.10M %.30R'"
  )"
  if [ -z "${JOB_STATUS}" ]; then
    echo "vLLM job ${QWEN_JOB_ID} exited before publishing an endpoint."
    ssh -o BatchMode=yes ruby-slurmlogin01.rckg.g03.cpe.ice.amd.com \
      "sacct -j ${QWEN_JOB_ID} -o JobID,State,ExitCode"
    break
  fi
  echo "${JOB_STATUS}"
  sleep 5
done

if [ -s "${ENDPOINT_FILE}" ]; then
  source "${ENDPOINT_FILE}"
  printf 'AORTA model endpoint: %s\n' "${AORTA_CHAT_VLLM_BASE_URL}"
else
  echo "No endpoint was created. Do not continue to the UI steps."
  false
fi
```

The printed URL must name a compute node and port 8001 or higher. It must not
say `localhost:8000`.

## 4. Wait for Qwen to become ready

Model loading can take a few minutes. Poll the endpoint recorded by the job:

```bash
HEALTH_URL="${AORTA_CHAT_VLLM_BASE_URL%/v1}/health"

until curl --fail --silent --show-error "${HEALTH_URL}" >/dev/null; do
  JOB_STATUS="$(
    ssh -o BatchMode=yes ruby-slurmlogin01.rckg.g03.cpe.ice.amd.com \
      "squeue -h -j ${QWEN_VLLM_JOB_ID} -o '%i %T %R'"
  )"
  if [ -z "${JOB_STATUS}" ]; then
    echo "vLLM job ${QWEN_VLLM_JOB_ID} exited before becoming ready."
    cat "/apps/avsharma/aorta-chat-runtime/qwen38-vllm-${QWEN_VLLM_JOB_ID}.log"
    break
  fi
  echo "Waiting for Qwen at ${HEALTH_URL} ..."
  sleep 5
done

if curl --fail --silent --show-error "${HEALTH_URL}" >/dev/null; then
  echo "Qwen is ready."
  curl --fail --silent --show-error \
    "${AORTA_CHAT_VLLM_BASE_URL}/models" | python3 -m json.tool
else
  echo "Qwen did not become ready. Do not continue to the UI steps."
  false
fi
```

If the Slurm job exits before the health check succeeds, inspect its log:

```bash
cat "/apps/avsharma/aorta-chat-runtime/qwen38-vllm-${QWEN_JOB_ID}.log"
```

Verify an actual model response, not only the health route:

```bash
curl --fail --silent --show-error \
  "${AORTA_CHAT_VLLM_BASE_URL}/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3.8-27B",
    "messages": [{"role": "user", "content": "Reply with READY only."}],
    "temperature": 0,
    "max_tokens": 16
  }' | python3 -m json.tool
```

The response should contain `"content": "READY"` and no reasoning trace.
The launcher also sends a function schema and requires one structured call with
`source = "READY"`; startup fails if vLLM returns XML as ordinary text instead.

## 5. Start the Chat UI

Start the UI from the same shell that sourced `qwen38-endpoint.env`. The
explicit guards are important: they prevent a missing endpoint file from
falling through to `localhost:8000`.

For ordinary chat:

```bash
cd /apps/avsharma/aorta-cia-qwen38
source .venv-ui/bin/activate
source /apps/avsharma/aorta-chat-runtime/qwen38-endpoint.env

: "${AORTA_CHAT_VLLM_BASE_URL:?Start the vLLM job and source its endpoint file first}"
: "${AORTA_CHAT_VLLM_MODEL:?The endpoint file did not specify a model}"

export AORTA_CHAT_LLM_PROVIDER=vllm
export AORTA_CHAT_LLM_TOOL_MODE=native
aorta chat ui --host 127.0.0.1 --port 8080
```

Open <http://127.0.0.1:8080>.

### Enable CIA cluster submissions

`ruby-slurmlogin02` cannot always reach the Slurm controller. To use
`triage_kernel_source`, `triage_assembly_source`, and `triage_workload`, run the
UI on `ruby-slurmlogin01` instead.

Check for a UI left by an earlier session before starting another one:

```bash
ssh -o BatchMode=yes ruby-slurmlogin01.rckg.g03.cpe.ice.amd.com '
  if [ -n "$(ss -H -ltn "sport = :8080")" ]; then
    echo "Port 8080 is already in use on ruby-slurmlogin01:"
    ss -ltnp "sport = :8080"
    pgrep -af "aorta chat ui|chainlit run" || true
    exit 1
  fi
'
```

If this reports an old AORTA UI, stop the terminal that owns it with `Ctrl-C`
before continuing. Do not kill an unfamiliar process merely to free the port.

Start the CIA-enabled UI:

```bash
ssh -t ruby-slurmlogin01.rckg.g03.cpe.ice.amd.com '
  cd /apps/avsharma/aorta-cia-qwen38
  source .venv-ui/bin/activate
  source /apps/avsharma/aorta-chat-runtime/qwen38-endpoint.env

  : "${AORTA_CHAT_VLLM_BASE_URL:?Start the vLLM job first}"
  : "${AORTA_CHAT_VLLM_MODEL:?The endpoint file did not specify a model}"

  export AORTA_CHAT_LLM_PROVIDER=vllm
  export AORTA_CHAT_LLM_TOOL_MODE=native
  export AORTA_CHAT_ALLOW_CLUSTER_JOBS=true
  export AORTA_CHAT_JOBS_PATH=/apps/avsharma/cia-chat-jobs
  export AORTA_CHAT_GPU_ARCH=gfx950
  export AORTA_CHAT_ROCJITSU_BUILD=/apps/avsharma/aorta-chat-runtime/rocjitsu-build
  export CIA_PARTITION=interactive
  export CIA_TIME_LIMIT=00:12:00
  export CIA_SEARCH_ROOTS=/apps/avsharma

  exec aorta chat ui --host 127.0.0.1 --port 8080
'
```

In a second terminal, forward that UI back to `ruby-slurmlogin02`:

```bash
ssh -N \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -L 8080:127.0.0.1:8080 \
  ruby-slurmlogin01.rckg.g03.cpe.ice.amd.com
```

If this says port 8080 is already in use, inspect the existing local listener
with `ss -ltnp 'sport = :8080'` and stop the terminal that owns the old tunnel.

Then open <http://127.0.0.1:8080>.

## 6. Verify the effective endpoint

In a separate terminal:

```bash
cd /apps/avsharma/aorta-cia-qwen38
source .venv-ui/bin/activate
source /apps/avsharma/aorta-chat-runtime/qwen38-endpoint.env

echo "${AORTA_CHAT_VLLM_BASE_URL}"
aorta chat config show
```

The environment variable overrides the profile. The UI log should report the
compute-node URL from `qwen38-endpoint.env`; it must not report
`http://localhost:8000`.

## 7. Stop everything

Stop the UI with `Ctrl-C`. If an SSH tunnel is running, stop that terminal with
`Ctrl-C` too.

If either process survives an interrupted terminal, stop only the UI and tunnel
managed by the launcher:

```bash
scripts/chat/start_aorta_chat.sh --stop
```

Cancel the model server while the endpoint file still exists:

```bash
source /apps/avsharma/aorta-chat-runtime/qwen38-endpoint.env

ssh -o BatchMode=yes ruby-slurmlogin01.rckg.g03.cpe.ice.amd.com \
  "scancel ${QWEN_VLLM_JOB_ID}"
```

Cancellation can leave the job in `COMPLETING` briefly while Docker exits.
Wait until Slurm removes it:

```bash
while JOB_STATUS="$(
  ssh -o BatchMode=yes ruby-slurmlogin01.rckg.g03.cpe.ice.amd.com \
    "squeue -h -j ${QWEN_VLLM_JOB_ID} -o '%i %T %M %R'"
)" && [ -n "${JOB_STATUS}" ]; do
  echo "${JOB_STATUS}"
  sleep 2
done

echo "vLLM job ${QWEN_VLLM_JOB_ID} has stopped."
```

The Slurm script removes `qwen38-endpoint.env` when the job exits, preventing a
later UI session from reusing a stale endpoint.
