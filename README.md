# AORTA

A ROCm / PyTorch debugging, reproducibility, and workload-triage toolkit for AMD GPUs.

AORTA wraps opaque launch commands, runs recipe-driven mitigation sweeps, captures
versioned environment snapshots, evaluates GPU hardware-queue scheduling, and
reproduces workload-specific issues (numerics, races, nondeterminism) that
micro-benchmarks miss. It ships a single `aorta` CLI plus a plugin system so
downstream packages can register their own workloads.

## What It Does

- **Unified sweep.** Run a built-in workload — or your own opaque launch
  command — across a `mitigation × {environment | diagnostic} × trial` matrix from
  one command (`aorta sweep run`). Recipe-driven, with a confound detector for
  speed regressions and a five-tier failure classifier for subprocess runs.
- **Environment snapshot for reproducibility.** Capture a schema-stable snapshot
  of the trial environment — ROCm / HIP / hipBLASLt / rocBLAS / MIOpen / RCCL
  identities, GPU arch, PyTorch build flags, runtime SDPA backend state, ~30
  numerics-relevant env vars — so cross-environment regressions become a `jq` diff
  instead of a multi-day investigation (`aorta env probe`). Embedded automatically
  into every sweep result.
- **Hardware queue evaluation.** Stress-test GPU queue scheduling with 8–64+
  concurrent streams across distributed-training, inference, and latency-sensitive
  patterns.
- **Workload reproducers.** In-tree E2E workloads — LLM determinism probe, RCCL
  race reproducer, real training (DDP/FSDP) and inference loops — designed to catch
  timing races and silent corruption that pass on synthetic tests.

## Installation

PyTorch for ROCm is **installed separately** from the ROCm PyTorch index (it is
not bundled in the wheel). Install it first, then AORTA.

### From PyPI (recommended for users)

```bash
# 1. PyTorch for your ROCm version — see https://pytorch.org/get-started/locally/
#    for the index URL matching your ROCm release.
pip install --pre torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/nightly/rocm<YOUR_ROCM_VERSION>/

# 2. AORTA (distribution: amd-aorta; import package + CLI: aorta)
pip install amd-aorta
```

Optional extras:

```bash
pip install "amd-aorta[hw-queue]"   # hardware-queue evaluation (numpy/pandas/tabulate)
```

> `[hw-queue]` requires PyTorch — install torch from the ROCm index (step 1
> above) before installing this extra.

Other extras: `analysis` (matplotlib only), `hw-queue-profiling` (hw-queue +
profiling plots), `ebpf` (no extra Python deps; needs the `bpftrace` binary and
`CAP_BPF`/`CAP_PERFMON` or sudo at runtime).

### From source (for contributors)

We recommend [uv](https://github.com/astral-sh/uv) for fast environment management.

```bash
git clone https://github.com/ROCm/aorta.git
cd aorta

uv venv && source .venv/bin/activate

# PyTorch for your ROCm version — see https://pytorch.org/get-started/locally/
uv pip install --pre torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/nightly/rocm<YOUR_ROCM_VERSION>/

# Editable install with the extras you need
uv pip install -e ".[hw-queue]"
```

Plain `pip` works too: `pip install -e ".[hw-queue]"`.

## Quick Start

```bash
# --- Environment snapshot ---
aorta env probe -o env.json                      # full snapshot to disk
aorta env probe --summary                        # one-screen brief, no file write
aorta env probe --field pytorch_build.git_commit # one field, JSON-typed
diff <(jq -S . env_a.json) <(jq -S . env_b.json) # diff two snapshots

# --- Unified sweep (recipe-driven) ---
aorta sweep run --recipe recipes/example-llm-determinism.yaml --dry-run   # validate only
aorta sweep run --recipe recipes/example-llm-determinism.yaml             # run the matrix

# Distributed workloads launch under torchrun (llm_determinism, race, fsdp):
torchrun --standalone --nproc_per_node=2 $(which aorta) \
  sweep run --recipe recipes/example-fsdp-smoke.yaml
torchrun --standalone --nproc_per_node=1 $(which aorta) \
  run --workload llm_determinism --trials 1 --steps 50

# --- Unified sweep (wrap an opaque launch command) ---
aorta sweep run --recipe recipes/probe-template-bash.yaml \
  --ticket ROCM-1234 -- bash launch.sh

# --- Inspect the registries / pattern catalogue ---
aorta sweep list-mitigations
aorta sweep list-environments
aorta sweep list-patterns
aorta mitigations list                           # standalone registry view
aorta environments list

# --- Single workload trial (no matrix) — training/inference only ---
aorta run --workload training --trials 1 --steps 50
aorta run --workload inference --trials 1 --steps 50

# --- Hardware queue evaluation (requires amd-aorta[hw-queue] + torch) ---
aorta bench hw_queue_eval list
aorta bench hw_queue_eval run hetero_kernels --streams 8
aorta bench hw_queue_eval sweep hetero_kernels --streams 2,4,8,16
```

## Main Workflows

### Unified sweep (mitigation × diagnostic × trial)

`aorta sweep run` is the single front door for matrix runs. It auto-selects a flow:

- **Workload flow** — a recipe with `mode: triage` (or `--workload` flag mode) runs
  a registered in-process workload across mitigations × environments × trials.
- **Subprocess flow** — a recipe with `mode: probe`, or any trailing
  `-- <command>`, wraps an opaque launch command and classifies each trial with the
  five-tier failure detector.

Both flows write `matrix.md` + `matrix.json`, embed a per-environment `env.json`
snapshot, and emit a replayable `recipe.resolved.yaml`. See
[`docs/probe-188/usage.md`](docs/probe-188/usage.md).

### Environment snapshot / reproducibility

`aorta env probe` captures a versioned, schema-stable `env.json`. Diff two
snapshots with `jq` to localize cross-environment regressions. See
[`docs/env-probe.md`](docs/env-probe.md).

### Hardware queue evaluation

`aorta bench hw_queue_eval` stress-tests GPU queue scheduling across many
concurrent streams. See [`docs/hw-queue-eval.md`](docs/hw-queue-eval.md).

### Single-workload runs

`aorta run --workload <name>` runs one workload directly (trials/steps,
environment overlay, mitigations) without building a matrix — handy for iterating
on a single reproducer. Note: `llm_determinism` and `race` require a distributed
environment and must be launched under `torchrun`; `training` and `inference`
self-bootstrap a single process.

## Workloads

In-tree workloads, registered via the `aorta.workloads` entry-point group:

| Workload | Description |
| --- | --- |
| `training` | Real DDP / FSDP training loop. |
| `inference` | Offline / continuous-serving inference loop. |
| `llm_determinism` | Bit-exact double-run check of a transformer step (FSDP2-aware, RCCL-safe, optional MoE). See [`docs/llm-determinism.md`](docs/llm-determinism.md). |
| `race` | RCCL race / SDC reproducer (`mode: default \| ddp \| fsdp`). Distributed — launch under `torchrun`. |

`_subprocess` is a platform-internal workload that backs the subprocess flow; it is
not meant for direct `aorta run` use.

**Downstream / private workloads** register through the same `aorta.workloads`
entry-point group from their own `pyproject.toml`, so they appear in `aorta sweep`
without modifying this repo.

## Recipes

A **recipe** is the authoritative description of a sweep: which cells to run,
per-cell trial/step counts, the ticket, and confound-detection config. Recipes are
the primary interface; flag mode is an escape hatch.

- [`recipes/README.md`](recipes/README.md) — schema reference and field semantics.
- [`recipes/README-running-recipes.md`](recipes/README-running-recipes.md) — how to
  run them (including distributed launches under `torchrun`).
- Ready-made examples live in [`recipes/`](recipes/) (e.g.
  `example-llm-determinism.yaml`, `example-fsdp-smoke.yaml`,
  `probe-template-bash.yaml`).

Minimal workload recipe:

```yaml
schema_version: 1
ticket: EXAMPLE-001
workload: training
trials: 2
steps: 100
cells:
  - name: baseline-local
    mitigations: [none]
    environment: local
  - name: tf32_off-local
    mitigations: [tf32_off]
    environment: local
```

Run it:

```bash
aorta sweep run --recipe my-recipe.yaml
```

## CLI Migration (probe / triage → sweep)

`aorta probe` and `aorta triage` have merged into the unified `aorta sweep`
front door. The old commands **still work** as deprecated aliases — they delegate
to the same execution engine and print a one-line stderr notice — but new usage
should target `aorta sweep`.

| Deprecated command | Use instead |
| --- | --- |
| `aorta probe ... -- cmd` | `aorta sweep run ... -- cmd` |
| `aorta triage run ...` | `aorta sweep run ...` |
| `aorta triage list-mitigations` | `aorta sweep list-mitigations` |
| `aorta triage list-environments` | `aorta sweep list-environments` |
| `aorta probe --list-patterns` | `aorta sweep list-patterns` |

The standalone `aorta mitigations list` and `aorta environments list` groups are
**not** deprecated and remain available.

> Note: probe-only runtime knobs (`--stop-after-events`, `--max-trials`,
> `--disable-detector`) are not yet exposed on `aorta sweep run`. Until they land,
> keep using `aorta probe` for those specific flags.

## Documentation

| Guide | Description |
| --- | --- |
| [Getting Started](docs/getting-started.md) | Prerequisites, Docker setup, installation |
| [Hardware Queue Eval](docs/hw-queue-eval.md) | Workloads, CLI usage, metrics |
| [Environment Probe](docs/env-probe.md) | Capture / diff / query a versioned environment snapshot; jq cookbook |
| [`aorta sweep`](docs/probe/usage.md) | Unified matrix runner — built-in workloads **or** opaque launch commands |
| [LLM Determinism](docs/llm-determinism.md) | Bit-exact double-run nondeterminism probe |
| [`aorta agent`](docs/agent/agentic-testing-guide.md) | Closed-loop mitigation search (optional LLM proposer) |
| [`aorta bundle`](docs/probe-188/bundle.md) | Package sweep artifacts with recipe-driven redaction |
| [Recipes](recipes/README.md) | Recipe schema and running recipes |
| [Buck2](docs/buck2.md) | Build / run the AORTA CLI via Buck2 |


## Repository Layout

```
src/aorta/
├── cli/               # `aorta` CLI command groups (sweep, run, env, bundle, agent, ...)
├── workloads/         # In-tree workloads (training, inference, llm_determinism, race)
├── instrumentation/   # Environment probe (versioned env.json schema + capture)
├── registry/          # Mitigations + environments registry (extension points)
├── hw_queue_eval/     # Hardware queue evaluation framework
├── training/          # FSDP2 trainer with multi-stream overlap instrumentation
├── models/            # Synthetic ranking transformer
├── profiling/         # Stream profiler for overlap measurement
└── utils/             # Config loading, timing, device detection

recipes/               # Sweep recipes (examples + customer handout templates)
docs/                  # Guides and reference
scripts/               # Launch, profiling, analysis tooling
```

## Development

```bash
uv pip install -e ".[dev]"
pre-commit install
pytest tests/
```

---

*The FSDP2 overlap and hardware-queue workloads also run on NVIDIA CUDA for
side-by-side comparison with ROCm.*
