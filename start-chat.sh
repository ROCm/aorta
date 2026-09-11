#!/bin/bash
# Serve the AORTA chat UI wired to the cluster agents.
#
# Every value here is a setting rather than a default because none of them is
# safe to guess: the proxy moves between nodes each time its job is resubmitted,
# and the sanitizer paths are this site's.
#
#   ./start-chat.sh          # serve on :8000
#   PORT=8010 ./start-chat.sh
set -uo pipefail

VENV="${VENV:-/apps/avsharma/aorta_llm/.venv}"
PORT="${PORT:-8000}"
PROXY_ENV="${PROXY_ENV:-/apps/avsharma/local_litellm/.env}"

# The proxy is a Slurm job, so its node changes on every resubmit. Find it
# rather than hardcoding where it was last time.
node="$(squeue --me -h -n litellm-proxy -o '%N' 2>/dev/null | head -1)"
if [ -z "$node" ]; then
    echo "No litellm-proxy job is running. Start one with:" >&2
    echo "  sbatch --partition=meta64 --time=5-00:00:00 \\" >&2
    echo "      /apps/avsharma/local_litellm/litellm-proxy.sbatch" >&2
    exit 1
fi
key="$(grep -E '^LITELLM_MASTER_KEY=' "$PROXY_ENV" | cut -d= -f2- | tr -d "\"'")"

# The model, for the chatbot and for Watch and Autopsy alike.
#
# Claude rather than Qwen, measured rather than assumed: on the same finished
# bundle, Autopsy took 44s on claude-haiku-4-5 and had produced nothing after
# 40 minutes on qwen3-35b. Qwen is a reasoning model and Autopsy's ReAct loop
# is six iterations with tools, so it spends its budget thinking at every step.
# In the chat path that compounds -- the tool ran so long without returning a
# verdict that the model called it again and submitted a second cluster job.
export AORTA_CHAT_VLLM_BASE_URL="http://${node}:4000/v1"
export AORTA_CHAT_VLLM_API_KEY="$key"
export AORTA_CHAT_VLLM_MODEL="${CHAT_MODEL:-claude-haiku-4-5}"

# Every model tried here answers with the native tool-call format rather than
# the ACTION: lines the text protocol parses. Left on "text" none of them calls
# a tool: they answer from their own knowledge, plausibly, and never touch the
# cluster.
export AORTA_CHAT_LLM_TOOL_MODE="${TOOL_MODE:-native}"

# What the sanitizers need, exported into each batch job rather than set here.
export AORTA_CHAT_ROCJITSU_BUILD="/apps/avsharma/rocjitsu/build"
export AORTA_CHAT_ROCJITSU_PRELOAD="/apps/avsharma/rocjitsu/toolchain/lib/libstdc++.so.6"
export AORTA_CHAT_JOBS_PATH="/apps/avsharma/jobs"
# The tools that submit work are off by default: they run pasted code on a GPU
# node and write outside the source root. This is the demo box, and turning them
# on is the point of it.
export AORTA_CHAT_ALLOW_CLUSTER_JOBS="${ALLOW_CLUSTER_JOBS:-true}"

# Watch and Autopsy read the same settings as the chatbot above, so there is
# one endpoint and one model here rather than two that can disagree. Autopsy
# still chooses its own token budget, which is a property of its reasoning
# rather than of the deployment.

echo "[chat] proxy=${node}:4000  model=${AORTA_CHAT_VLLM_MODEL} (chat and agents)  tool-mode=${AORTA_CHAT_LLM_TOOL_MODE}  cluster-jobs=${AORTA_CHAT_ALLOW_CLUSTER_JOBS}  port=${PORT}"
exec "$VENV/bin/aorta" chat ui --port "$PORT"
