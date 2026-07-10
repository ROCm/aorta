# Mirage + AORTA manual demo guide (sharkmi300x-4)

Exhaustive command reference for manually testing and demoing [PR #227](https://github.com/ROCm/aorta/pull/227)
(`users/vivekkhandelwal1/mirage-emulation`) on this machine.

**Companion docs:**
- Setup summary: [emulated-gpu-mirage-setup.md](emulated-gpu-mirage-setup.md)
- Design: [plans/mirage-aorta-integration.md](plans/mirage-aorta-integration.md)

**What works on this host (validated):**
- **rocjitsu + mi350x** + vLLM container → full AORTA gpu_smoke / probe demo (~3–5 s)
- **Host unit tests** → no GPU required (emulation + gpu_smoke suites)

**Known limitations:**
- Default image tag `v0.23.0-patched-v2` is **not** on this registry — use `:latest`
- **rocjitsu-dbt** (MI350X guest → MI300X host) may fail with DBT translation errors
- **MI450X** (`mi450x` profile) — no end-to-end torch path yet (gfx1250 KMD gap)

---

## 0. Session setup (run once per shell)

```bash
export WORK_ROOT=/home/vikhande/work/13_04
export MIRAGE_ROOT="$WORK_ROOT/rocm-systems/emulation/mirage"
export AORTA_ROOT="$WORK_ROOT/aorta"
export MIRAGE_BIN="$MIRAGE_ROOT/build/manylinux/bin/mirage"
export PATH="$(dirname "$MIRAGE_BIN"):$PATH"

# Use :latest on sharkmi300x-4 (patched-v2 tag unavailable here)
export MIRAGE_AORTA_IMAGE=docker.io/vllm/vllm-openai-rocm:latest

export XDG_CONFIG_HOME=/tmp/mirage-demo-config
export XDG_RUNTIME_DIR=/tmp/mirage-demo-runtime
mkdir -p "$XDG_CONFIG_HOME" "$XDG_RUNTIME_DIR"

source "$WORK_ROOT/.venv-aorta/bin/activate"
cd "$AORTA_ROOT"
git pull   # ensure the emulation code is present (merged on the default branch)
pip install -e . --quiet
chmod +x scripts/emulation/*.sh
```

---

## 1. Prerequisites verification

### 1.1 Host

```bash
hostname                    # sharkmi300x-4
rocm-smi                    # 8× MI300X (gfx942) — only needed for rocjitsu-dbt demos
```

### 1.2 Docker

```bash
docker --version
docker info | head -20
docker images | grep vllm
docker pull docker.io/vllm/vllm-openai-rocm:latest   # if missing
```

### 1.3 Mirage build

```bash
ls -la "$MIRAGE_BIN"
mirage --version
mirage --help

# Rebuild only if binary missing:
cd "$MIRAGE_ROOT"
sudo apt-get install -y docker-buildx
./scripts/mirage-docker-build.sh
```

### 1.4 AORTA install

```bash
cd "$AORTA_ROOT"
pip install -e .
aorta --version
aorta --help
```

---

## 2. Mirage commands (complete demo reference)

### 2.1 Discovery

```bash
mirage --help
mirage --version
mirage about
mirage paths
mirage emulators
mirage emulators --long
mirage emulators --json
```

### 2.2 Profiles

```bash
mirage profile list
mirage profile list --long
mirage profile show mi300x
mirage profile show mi350x
mirage profile show mi450x

# Create custom profiles (scripts auto-create these if missing)
mirage profile create mi350x --emulator rocjitsu --agent MI350X \
  --num-nodes 1 --gpus-per-node 1 --no-input

mirage profile create dbt-mi350x --emulator rocjitsu-dbt --agent MI350X \
  --num-nodes 1 --gpus-per-node 1 --no-input

# Delete a custom profile (builtins cannot be deleted)
# mirage profile delete my-profile -f
```

### 2.3 Agents and topologies

```bash
mirage agent list
mirage agent show MI350X
mirage agent show MI300X
mirage agent show MI450X

mirage topology list
mirage topology show <name>    # if any custom topologies exist
```

### 2.4 Sessions (advanced — optional for demo)

```bash
mirage session list
mirage session start --profile mi350x
# mirage session show <id>
# mirage session stop <id>
# mirage session dir <id>
```

### 2.5 State

```bash
mirage state --help
# mirage state purge   # caution: removes sessions
```

### 2.6 `mirage run` — container GPU sanity (no AORTA)

```bash
# rocminfo inside emulated MI350X
mirage run --in-process --profile mi350x \
  --image "$MIRAGE_AORTA_IMAGE" \
  -- rocminfo

mirage run --in-process --profile mi350x \
  --image "$MIRAGE_AORTA_IMAGE" \
  -- rocminfo 2>&1 | grep -E "Agent|Name:|gfx|Device Type"

# Minimal torch kernel
FIXTURE="$MIRAGE_ROOT/tests/fixtures/ml/tiny_torch.py"
mirage run --in-process --profile mi350x \
  --image "$MIRAGE_AORTA_IMAGE" \
  --mount "$FIXTURE:/tiny_torch.py:ro" \
  -- python3 /tiny_torch.py
# expect: tiny_torch_ok

# torch.cuda check
mirage run --in-process --profile mi350x \
  --image "$MIRAGE_AORTA_IMAGE" \
  -- python3 -c "
import torch
print('torch', torch.__version__)
print('cuda_available', torch.cuda.is_available())
print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')
"

# Verbose logging
mirage run -vv --in-process --profile mi350x \
  --image "$MIRAGE_AORTA_IMAGE" \
  -- rocminfo 2>&1 | head -30

# Compare profiles
mirage run --in-process --profile mi300x --image "$MIRAGE_AORTA_IMAGE" -- rocminfo | grep gfx
mirage run --in-process --profile mi450x --image "$MIRAGE_AORTA_IMAGE" \
  -- python3 -c "import torch; print('mi450x cuda:', torch.cuda.is_available())"
```

### 2.7 `mirage run` flags reference (useful overrides)

```bash
# Override emulator for one run
mirage run --in-process --profile mi350x --emulator rocjitsu -- ...

# Multi-GPU topology override (advanced)
mirage run --in-process --profile mi350x --gpus-per-node 2 -- ...

# Keep session alive after command (debugging)
mirage run --keep-session --profile mi350x --image "$MIRAGE_AORTA_IMAGE" -- sleep 30
mirage session list
```

---

## 3. AORTA host commands (no mirage container)

These run on the host directly. **Torch triage under rocjitsu without a container is impractically slow** — use for registry/CLI inspection and dry-runs only.

### 3.1 Top-level

```bash
aorta --help
aorta --version
```

### 3.2 Registry inspection

```bash
aorta environments list
aorta environments list | grep emulated

aorta mitigations list

aorta triage list-environments
aorta triage list-mitigations

aorta sweep list-environments
aorta sweep list-mitigations
aorta sweep list-patterns
```

### 3.3 Emulation launch helper (Python)

```bash
python3 -c "
from aorta.emulation.mirage_launch import (
    is_emulated_environment,
    wrap_argv_for_environment,
    resolve_mirage_bin,
)
cfg = {'_aorta_environment': {
    'name': 'emulated-rocjitsu',
    'mirage_profile': 'mi350x',
    'emulator': 'rocjitsu',
}}
print('mirage_bin:', resolve_mirage_bin())
print('is_emulated:', is_emulated_environment(cfg))
print('wrapped:', wrap_argv_for_environment(cfg, ['python3', '-c', 'print(1)']))
"
```

### 3.4 Dry-runs (instant, no GPU execution)

```bash
cd "$AORTA_ROOT"

aorta triage run --dry-run --recipe recipes/gpu-smoke-emulated.yaml

aorta sweep run --dry-run --recipe recipes/gpu-smoke-emulated.yaml

aorta probe --dry-run \
  --recipe recipes/example-probe-smoke.yaml \
  --ticket PROBE-DRY -- \
  bash -c 'echo hi'

aorta sweep run --dry-run \
  --recipe recipes/example-probe-smoke.yaml \
  --ticket PROBE-DRY -- \
  bash -c 'echo hi'
```

### 3.5 Host env probe

```bash
aorta env probe --summary
aorta env probe -o /tmp/host-env.json
aorta env probe --field gpu_arch
aorta env probe --verbose
```

### 3.6 Host `aorta run` (real hardware — baseline comparison)

```bash
# Real MI300X GPU on this host (not emulated)
aorta run --workload gpu_smoke --environment local --trials 1 --steps 1

# Emulated path on host (wraps via mirage — slow for torch)
aorta run --workload gpu_smoke --environment emulated-rocjitsu --trials 1 --steps 1
```

### 3.7 Host triage flag mode (no recipe file)

```bash
aorta triage run --dry-run \
  --workload gpu_smoke \
  --mitigation-axis none \
  --environment-axis emulated-rocjitsu \
  --trials 1 --steps 1 \
  --ticket EMU-FLAG-MODE
```

### 3.8 Probe pattern catalogue

```bash
aorta probe --list-patterns
aorta probe --list-patterns --version
aorta sweep list-patterns
```

### 3.9 Agent (dry-run only for demo)

```bash
aorta agent --dry-run \
  --recipe recipes/example-probe-smoke.yaml \
  --ticket AGENT-DRY -- \
  bash -c 'echo hi'
```

---

## 4. Unit tests (no GPU)

```bash
cd "$AORTA_ROOT"

# Emulation module (mirage_launch, env schema, argv wrapping)
python -m pytest tests/emulation/ -v

# Container CLI runner JSON validation
python -m pytest tests/emulation/test_aorta_cli_runner.py -v

# gpu_smoke workload (fake torch)
python -m pytest tests/workloads/test_gpu_smoke.py -v

# Environment registry
python -m pytest tests/registry/test_environments.py -v

# Dispatcher _aorta_environment round-trip
python -m pytest tests/run/test_dispatcher.py -v -k environment

# All emulation-related together
python -m pytest tests/emulation/ tests/workloads/test_gpu_smoke.py -q
# expect: all pass
```

---

## 5. Helper scripts (recommended demo path)

From `$AORTA_ROOT`:

```bash
# ── rocjitsu software emulation (primary demo — no physical GPU required) ──

OUT=/tmp/demo-gpu-smoke      ./scripts/emulation/run_mirage_container.sh gpu-smoke
OUT=/tmp/demo-probe          ./scripts/emulation/run_mirage_container.sh probe
OUT=/tmp/demo-inference      ./scripts/emulation/run_mirage_container.sh inference-smoke
OUT=/tmp/demo-training-ddp   ./scripts/emulation/run_mirage_container.sh training-ddp-smoke
OUT=/tmp/demo-training-fsdp  ./scripts/emulation/run_mirage_container.sh training-fsdp-smoke

# ── rocjitsu-dbt (uses real MI300X GPUs — may fail on this host) ──

EMULATOR=rocjitsu-dbt OUT=/tmp/demo-dbt \
  ./scripts/emulation/run_mirage_container.sh gpu-smoke

# ── Profile override ──

PROFILE=mi350x EMULATOR=rocjitsu OUT=/tmp/demo-mi350x \
  ./scripts/emulation/run_mirage_container.sh gpu-smoke

# ── llm_determinism (slow under rocjitsu; faster with dbt if it works) ──
# Defaults to the in-repo recipes/llm-determinism-emulated.yaml singleton;
# set LLM_RECIPE to mount a full multi-cell recipe from the host instead.

OUT=/tmp/demo-llm \
  ./scripts/emulation/run_mirage_container.sh llm-determinism

EMULATOR=rocjitsu-dbt \
  LLM_RECIPE="$AORTA_ROOT/recipes/example-llm-determinism.yaml" \
  OUT=/tmp/demo-llm-dbt \
  ./scripts/emulation/run_mirage_container.sh llm-determinism

# ── Full matrix: single-process cases × 2 emulators (~10–15 min) ──
# Add the slow llm_determinism case with INCLUDE_LLM_DET=1 (raise TIMEOUT_SEC).

RESULT_ROOT=/tmp/demo-matrix-$(date +%Y%m%d-%H%M%S) \
  ./scripts/emulation/run_mirage_matrix.sh

cat "$RESULT_ROOT/summary.tsv"
column -t -s $'\t' "$RESULT_ROOT/summary.tsv"
awk -F'\t' 'NR>1 && $4 != "0" {print}' "$RESULT_ROOT/summary.tsv"
```

### Script environment knobs

| Variable | Default | Purpose |
|----------|---------|---------|
| `MIRAGE_BIN` | *(required)* | Path to mirage binary |
| `MIRAGE_AORTA_IMAGE` | `v0.23.0-patched-v2` | Override to `:latest` on this host |
| `AORTA_SRC` | repo root | AORTA checkout to mount |
| `EMULATOR` | `rocjitsu` | `rocjitsu` or `rocjitsu-dbt` |
| `PROFILE` | `mi350x` / `dbt-mi350x` | mirage profile name |
| `OUT` | `/tmp/aorta-*-out` | Host output directory |
| `XDG_CONFIG_HOME` | `/tmp/mirage-aorta-config` | mirage config dir |
| `XDG_RUNTIME_DIR` | `/tmp/mirage-aorta-runtime` | mirage runtime dir |
| `LLM_RECIPE` | *(optional; llm-determinism only)* | Host recipe to mount; defaults to in-repo `llm-determinism-emulated.yaml` |
| `INCLUDE_LLM_DET` | `0` | Matrix only: set `1` to add the slow llm_determinism case |

---

## 6. Manual container commands (every AORTA CLI used in the matrix)

These mirror `scripts/emulation/run_mirage_matrix.sh` and go beyond it.
Replace `AORTA_CLI_JSON` with the command you want to run inside the container.

**Shared boot snippet** (copy AORTA into writable path, install, run runner):

```bash
RUNNER="$AORTA_ROOT/scripts/emulation/aorta_cli_runner.py"
OUT=/tmp/demo-manual
mkdir -p "$OUT"

run_aorta_json() {
  local json="$1"
  mirage run --in-process --profile "${PROFILE:-mi350x}" \
    --image "$MIRAGE_AORTA_IMAGE" \
    --env "AORTA_CLI_JSON=$json" \
    --mount "$AORTA_ROOT:/aorta-src:ro" \
    --mount "$RUNNER:/runner.py:ro" \
    --mount "$OUT:/out" \
    --mount "$OUT:/tmp/aorta-build" \
    -- sh -c '
      rm -rf /tmp/aorta-build/src
      cp -a /aorta-src /tmp/aorta-build/src
      cp /runner.py /tmp/aorta_cli_runner.py
      python3 -m pip install -q /tmp/aorta-build/src --no-deps click pyyaml
      cd /out && python3 /tmp/aorta_cli_runner.py
    '
}
```

### 6.1 General / help

```bash
run_aorta_json '["--help"]'
run_aorta_json '["--version"]'
```

### 6.2 Registry

```bash
run_aorta_json '["environments","list"]'
run_aorta_json '["mitigations","list"]'
run_aorta_json '["triage","list-environments"]'
run_aorta_json '["triage","list-mitigations"]'
run_aorta_json '["sweep","list-environments"]'
run_aorta_json '["sweep","list-mitigations"]'
run_aorta_json '["sweep","list-patterns"]'
run_aorta_json '["probe","--list-patterns"]'
```

### 6.3 Environment probe

```bash
run_aorta_json '["env","probe","--summary"]'
run_aorta_json '["env","probe","-o","/out/env.json"]'
run_aorta_json '["env","probe","--field","gpu_arch"]'
run_aorta_json '["env","probe","--verbose"]'
```

### 6.4 Workload run (environment=local inside container)

Outer mirage provides emulation; inner AORTA uses `local` to avoid double-wrap:

```bash
run_aorta_json '["run","--workload","gpu_smoke","--environment","local","--trials","1"]'
run_aorta_json '["run","--workload","gpu_smoke","--environment","local","--trials","1","--steps","3"]'
run_aorta_json '["run","--workload","gpu_smoke","--environment","local","--trials","1","--results-dir","/out/results"]'
```

### 6.5 Triage (recipe-based)

```bash
# Dry-run
run_aorta_json '["triage","run","--dry-run","--recipe","/tmp/aorta-build/src/recipes/gpu-smoke-emulated.yaml"]'

# Full gpu_smoke triage (headline demo)
run_aorta_json '["triage","run","--verbose","--recipe","/tmp/aorta-build/src/recipes/gpu-smoke-emulated.yaml","--output-dir","/out/triage_results"]'

# sweep run equivalent (preferred modern CLI)
run_aorta_json '["sweep","run","--verbose","--recipe","/tmp/aorta-build/src/recipes/gpu-smoke-emulated.yaml","--output-dir","/out/triage_results"]'
```

### 6.6 Probe / sweep probe mode

```bash
run_aorta_json '["probe","--recipe","/tmp/aorta-build/src/recipes/example-probe-smoke.yaml","--output","/out/probe_results","--ticket","PROBE-MANUAL","--","bash","-c","echo hi from mirage probe"]'

run_aorta_json '["sweep","run","--recipe","/tmp/aorta-build/src/recipes/example-probe-smoke.yaml","--output","/out/probe_results","--ticket","SWEEP-MANUAL","--","bash","-c","echo hi from mirage sweep"]'

run_aorta_json '["probe","--dry-run","--recipe","/tmp/aorta-build/src/recipes/example-probe-smoke.yaml","--ticket","PROBE-DRY","--","bash","-c","echo hi"]'
```

### 6.7 Agent dry-run

```bash
run_aorta_json '["agent","--dry-run","--recipe","/tmp/aorta-build/src/recipes/example-probe-smoke.yaml","--ticket","AGENT-MANUAL","--","bash","-c","echo hi"]'
```

### 6.8 rocjitsu-dbt variant

```bash
PROFILE=dbt-mi350x mirage run --in-process --profile dbt-mi350x \
  --image "$MIRAGE_AORTA_IMAGE" \
  --env 'AORTA_CLI_JSON=["run","--workload","gpu_smoke","--environment","local","--trials","1"]' \
  --mount "$AORTA_ROOT:/aorta-src:ro" \
  --mount "$RUNNER:/runner.py:ro" \
  --mount "$OUT:/out" \
  --mount "$OUT:/tmp/aorta-build" \
  -- sh -c '...'   # same boot snippet as above
```

---

## 7. Inspect demo artifacts

### 7.1 gpu_smoke triage output

```bash
OUT=/tmp/demo-gpu-smoke   # or your OUT path

find "$OUT" -name "matrix.json" -o -name "matrix.md" -o -name "trial_*.json"

cat "$OUT/triage_results/EMU-SMOKE-001/gpu_smoke/"*/matrix.md

python3 -m json.tool "$OUT/triage_results/EMU-SMOKE-001/gpu_smoke/"*/matrix.json | head -60

python3 -m json.tool \
  "$OUT/triage_results/EMU-SMOKE-001/gpu_smoke/"*/cells/emulated-rocjitsu/gpu_smoke/trial_d0_m0_t0.json
```

**Expect:** `"passed": true`, `"exit_status": "ok"`, `gpu_smoke PASS` in logs.

### 7.2 Probe output

```bash
OUT=/tmp/demo-probe
find "$OUT/probe_results" -name "result.json" | head -5
ls -laR "$OUT/probe_results/PROBE-MIRAGE/"
```

### 7.3 Matrix summary

```bash
cat /tmp/demo-matrix-*/summary.tsv
less /tmp/demo-matrix-*/rocjitsu/triage-gpu-smoke/run.log
```

### 7.4 Bundle probe results (optional, host-side after probe demo)

```bash
aorta bundle /tmp/demo-probe/probe_results/PROBE-MIRAGE --ticket PROBE-MIRAGE
```

---

## 8. Recommended live demo order (~30 min)

| # | What | Command |
|---|------|---------|
| 1 | Show emulators | `mirage emulators` |
| 2 | Show builtin profiles | `mirage profile show mi350x` |
| 3 | Unit tests (no GPU) | `pytest tests/emulation/ tests/workloads/test_gpu_smoke.py -q` |
| 4 | Emulated GPU in container | `mirage run ... tiny_torch.py` |
| 5 | **Headline:** gpu_smoke triage | `./scripts/emulation/run_mirage_container.sh gpu-smoke` |
| 6 | Show matrix.json | `cat .../matrix.json` |
| 7 | Probe subprocess path | `./scripts/emulation/run_mirage_container.sh probe` |
| 8 | Show wrapped argv | Python `wrap_argv_for_environment` snippet (§3.3) |
| 9 | Full matrix (optional) | `./scripts/emulation/run_mirage_matrix.sh` |
| 10 | MI450 limitation (optional) | `mirage run --profile mi450x ... torch.cuda.is_available()` → False |

---

## 9. AORTA command quick-reference

| Command | In matrix? | Container? | Notes |
|---------|------------|------------|-------|
| `aorta --help` | yes | yes | |
| `aorta --version` | no | yes | §6.1 |
| `aorta environments list` | yes | yes | |
| `aorta mitigations list` | no | yes | §6.2 |
| `aorta triage list-environments` | yes | yes | |
| `aorta triage list-mitigations` | yes | yes | |
| `aorta triage run --dry-run` | yes | yes | |
| `aorta triage run --recipe gpu-smoke-emulated.yaml` | yes | yes | **headline demo** |
| `aorta sweep run --recipe ...` | no | yes | modern replacement for triage |
| `aorta sweep list-*` | no | yes | §6.2 |
| `aorta run --workload gpu_smoke --environment local` | yes | yes | inside container |
| `aorta run --environment emulated-rocjitsu` | no | host only | slow without container |
| `aorta env probe --summary` | yes | yes | |
| `aorta env probe -o /out/env.json` | yes | yes | |
| `aorta env probe --field gpu_arch` | yes | yes | |
| `aorta env probe --verbose` | no | yes | §6.3 |
| `aorta env recipe` | no | host | needs existing env.json |
| `aorta probe --recipe ... -- cmd` | yes | yes | deprecated → use sweep |
| `aorta probe --list-patterns` | yes | yes | |
| `aorta probe --dry-run` | no | yes | §6.6 |
| `aorta sweep run --recipe probe.yaml -- cmd` | no | yes | §6.6 |
| `aorta agent --dry-run` | no | yes | §6.7 |
| `aorta bundle <run_dir>` | no | host | after probe demo |
| `aorta bench hw_queue_eval` | no | — | not part of emulation PR |

---

## 10. Mirage command quick-reference

| Command | Demo use |
|---------|----------|
| `mirage emulators` | Show rocjitsu / rocjitsu-dbt available |
| `mirage profile list/show` | Show mi300x / mi350x / mi450x |
| `mirage profile create` | Create dbt or custom profiles |
| `mirage agent list/show` | Show MI350X / MI450X agent defs |
| `mirage topology list` | Optional topology inspection |
| `mirage run --profile mi350x --image ... -- cmd` | **Primary execution path** |
| `mirage run --in-process` | Faster single-node path (used by scripts) |
| `mirage run -vv` | Debug logging |
| `mirage session list` | Debug persistent sessions |
| `mirage paths` | Show XDG state locations |
| `mirage about` | Version / license info |

---

## 11. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `manifest for v0.23.0-patched-v2 not found` | `export MIRAGE_AORTA_IMAGE=docker.io/vllm/vllm-openai-rocm:latest` |
| `set MIRAGE_BIN` | Export path per §0 |
| `Invalid device id` | Use container torch via `--image`; don't mount host torch |
| `Read-only file system` on pip | Scripts copy to `/tmp/aorta-build/src` (writable) |
| `rdhc exited 1` in logs | Harmless for smoke tests |
| Second run tests stale code | Scripts `rm -rf /tmp/aorta-build/src` before copy |
| rocjitsu-dbt segfault / SGPR errors | Known on MI300X host; demo rocjitsu instead |
| MI450X `cuda_available False` | Known limitation; demo mi350x |
| Host torch triage hangs | Use container path (§5), not host venv |
| `aorta_cli_runner: invalid AORTA_CLI_JSON` | Pass a JSON list of strings, e.g. `'["--help"]'` |

---

## 12. Cleanup

```bash
rm -rf /tmp/mirage-demo-config /tmp/mirage-demo-runtime
rm -rf /tmp/demo-* /tmp/aorta-gpu-smoke-out /tmp/aorta-probe-out /tmp/aorta-mirage-matrix-*
```

---

## 13. Key files

| Path | Purpose |
|------|---------|
| `recipes/gpu-smoke-emulated.yaml` | Emulated triage recipe |
| `recipes/example-probe-smoke.yaml` | Probe-mode recipe |
| `recipes/example-llm-determinism.yaml` | Slow workload for dbt demo |
| `scripts/emulation/run_mirage_container.sh` | Main demo entrypoint |
| `scripts/emulation/run_mirage_matrix.sh` | 24-case regression matrix |
| `scripts/emulation/aorta_cli_runner.py` | `AORTA_CLI_JSON` bridge inside container |
| `src/aorta/emulation/mirage_launch.py` | argv wrapping for emulated environments |
| `src/aorta/registry/environments.py` | `emulated-rocjitsu` built-in |
