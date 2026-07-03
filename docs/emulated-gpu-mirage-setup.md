# Emulated GPU setup (mirage + rocjitsu)

This guide reproduces the validated path for running AORTA under the [mirage](https://github.com/ROCm/rocm-systems/tree/develop/emulation/mirage)
GPU emulator on a Linux host with Docker. It matches what was exercised on PR
#227 (`users/vivekkhandelwal1/mirage-emulation`).

**Recommended:** build mirage via Docker, run torch workloads inside a vLLM/ROCm
container through `mirage run --image …`. Host-side `mirage run -- aorta …` with
a local venv is fine for unit tests but **impractically slow** for full torch
triages under rocjitsu CPU emulation.

No physical GPU is required for **rocjitsu** (software emulation). **rocjitsu-dbt**
requires a real GPU on the host.

---

## 1. Prerequisites

| Tool | Notes |
|------|-------|
| Docker 24+ | mirage build + containerized workloads |
| docker-buildx | Required by mirage Docker build (`sudo apt-get install -y docker-buildx`) |
| mirage | Built from [rocm-systems/emulation/mirage](https://github.com/ROCm/rocm-systems/tree/develop/emulation/mirage) |
| vLLM ROCm image | Default: `docker.io/vllm/vllm-openai-rocm:v0.23.0-patched-v2` |

---

## 2. Build mirage

```bash
git clone --depth 1 https://github.com/ROCm/rocm-systems.git
cd rocm-systems/emulation/mirage
sudo apt-get install -y docker-buildx   # once, if missing
./scripts/mirage-docker-build.sh

export MIRAGE_BIN="$PWD/build/manylinux/bin/mirage"
export PATH="$(dirname "$MIRAGE_BIN"):$PATH"
mirage emulators   # rocjitsu: installed + supported
```

---

## 3. Clone this branch

```bash
git clone --branch users/vivekkhandelwal1/mirage-emulation \
  --depth 1 https://github.com/ROCm/aorta.git aorta
cd aorta
pip install -e .   # optional: host unit tests only
python -m pytest tests/emulation/ -q   # 18 passed, no GPU
```

---

## 4. Environment variables

```bash
export MIRAGE_BIN=/path/to/rocm-systems/emulation/mirage/build/manylinux/bin/mirage
export MIRAGE_AORTA_IMAGE="${MIRAGE_AORTA_IMAGE:-docker.io/vllm/vllm-openai-rocm:v0.23.0-patched-v2}"
export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-/tmp/mirage-config}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/mirage-runtime}"
mkdir -p "$XDG_CONFIG_HOME" "$XDG_RUNTIME_DIR"
```

---

## 5. Quick smoke (helper scripts)

From the AORTA repo root:

```bash
chmod +x scripts/emulation/*.sh

# rocjitsu (software emulation, no physical GPU)
./scripts/emulation/run_mirage_container.sh gpu-smoke
./scripts/emulation/run_mirage_container.sh probe

# rocjitsu-dbt (real GPU on host, same-ISA passthrough)
EMULATOR=rocjitsu-dbt ./scripts/emulation/run_mirage_container.sh gpu-smoke
```

**Full CLI matrix** (12 commands × 2 emulators, ~9 min):

```bash
./scripts/emulation/run_mirage_matrix.sh
# Summary: /tmp/aorta-mirage-matrix-<timestamp>/summary.tsv
```

---

## 6. Container pattern (why)

| Approach | Result |
|----------|--------|
| `mirage run -- host-venv/aorta …` | Torch init under rocjitsu can take hours |
| `mirage run --image vllm… -- …` | **Works** — gpu_smoke ~3s (rocjitsu) or ~1.6s (dbt) |

Rules:

- Use the **container's** torch, not a host venv via `PYTHONPATH` (`Invalid device id`).
- Copy AORTA source to a **writable** path inside the container (`cp -a`); read-only mounts break `pip install -e`.
- Outer `mirage run` provides GPU emulation; inner AORTA uses `--environment local` for `aorta run` (avoids double mirage wrap).

---

## 7. Manual one-liner (gpu-smoke triage)

```bash
AORTA_ROOT=$PWD
OUT=/tmp/aorta-triage-out
mkdir -p "$OUT"

mirage profile show mi350x >/dev/null 2>&1 || \
  mirage profile create mi350x --emulator rocjitsu --agent MI350X \
    --num-nodes 1 --gpus-per-node 1 --no-input

mirage run --in-process --profile mi350x \
  --image "$MIRAGE_AORTA_IMAGE" \
  --env 'AORTA_CLI_JSON=["triage","run","--verbose","--recipe","/tmp/aorta-build/src/recipes/gpu-smoke-emulated.yaml","--output-dir","/out/triage_results"]' \
  --mount "$AORTA_ROOT:/aorta-src:ro" \
  --mount "$AORTA_ROOT/scripts/emulation/aorta_cli_runner.py:/runner.py:ro" \
  --mount "$OUT:/out" \
  --mount "$OUT:/tmp/aorta-build" \
  -- sh -c '
    rm -rf /tmp/aorta-build/src
    cp -a /aorta-src /tmp/aorta-build/src
    cp /runner.py /tmp/aorta_cli_runner.py
    pip install -q /tmp/aorta-build/src --no-deps click pyyaml
    cd /out && python3 /tmp/aorta_cli_runner.py
  '
```

Expected: `gpu_smoke PASS`, `Failure rate: 0%`.

---

## 8. Profile naming

| mirage builtin | AORTA `emulated-rocjitsu` env |
|----------------|-------------------------------|
| `mi350x` | `mirage_profile: mi350x` |

Scripts auto-create `mi350x` (rocjitsu) or `dbt-mi350x` (rocjitsu-dbt) if missing.

For host-side `aorta probe` with `environment: emulated-rocjitsu`, AORTA wraps
`mirage run --profile mi350x -- …` via `aorta.emulation.mirage_launch`.

---

## 9. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `BuildKit … buildx … missing` | `sudo apt-get install -y docker-buildx` |
| `Invalid device id` in container | Do not mount host torch; use container python |
| `Read-only file system` on pip | Copy source to `/tmp/aorta-build/src` inside container |
| `rdhc not on PATH` in env probe | Harmless for smoke tests |
| llm_determinism very slow | Use `EMULATOR=rocjitsu-dbt` on a machine with GPUs |

---

## 10. Related docs

- Design: [docs/plans/mirage-aorta-integration.md](plans/mirage-aorta-integration.md)
- Recipe: [recipes/gpu-smoke-emulated.yaml](../recipes/gpu-smoke-emulated.yaml)
