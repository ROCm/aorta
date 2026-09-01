# AORTA

A ROCm / PyTorch debugging, reproducibility, and workload-triage toolkit for AMD GPUs.

AORTA wraps opaque launch commands, runs recipe-driven mitigation sweeps, captures
versioned environment snapshots, evaluates GPU hardware-queue scheduling, and
reproduces workload-specific issues (numerics, races, nondeterminism) that
micro-benchmarks miss. It ships a single `aorta` CLI plus a plugin system so
downstream packages can register their own workloads.

Every night AORTA's own workload sweep runs on an MI350 runner; the results are
published to the [nightly CI dashboard](https://rocm.github.io/aorta/).

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

Install AORTA in the Python environment from which you will invoke `aorta`.
The core install provides the CLI, recipe support, and environment probe. It
does not install or require PyTorch. AORTA requires Python 3.10 or newer.

### From PyPI (recommended for users)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install amd-aorta
aorta --help
```

The distribution name is `amd-aorta`; the import package and command are
`aorta`.

### From source (for contributors)

Use [uv](https://github.com/astral-sh/uv) for a fast editable setup:

```bash
git clone https://github.com/ROCm/aorta.git
cd aorta
uv venv
source .venv/bin/activate
uv pip install -e .
aorta --help
```

Plain `pip` also works: `python -m pip install -e .`. Contributors who need
the test and lint tools can use `uv pip install -e ".[dev]"`.

### Optional dependencies

An **extra** adds dependencies for a specific feature; it is not part of the
minimal install. Install only the extras you need. For example:

```bash
# Published package:
python -m pip install "amd-aorta[hw-queue]"

# Editable source checkout:
uv pip install -e ".[hw-queue]"
```

Hardware-queue evaluation and most GPU training or inference workloads also
require PyTorch. AORTA does not bundle it. When the selected workload requires
PyTorch, install a build matching that environment's ROCm version — but note
that *where* those builds are hosted now depends on the ROCm major, so there is
no single URL pattern to fill in:

```bash
# Pick ONE index for your ROCm major:
#   ROCm 7.x  -> PyTorch's own index, one URL per ROCm minor
#   ROCm 10.x -> AMD's TheRock index (PyTorch's index has no rocm10.0)
PYTORCH_ROCM_INDEX=https://download.pytorch.org/whl/rocm7.2/
# PYTORCH_ROCM_INDEX=https://stable.repo.amd.com/rocm/whl-next/

python -m pip install torch --index-url "$PYTORCH_ROCM_INDEX"
```

The split is not cosmetic, and picking the wrong host fails with a bare
"no matching distribution" that does not hint at the reason:

- `download.pytorch.org/whl/rocm<minor>/` **stops at `rocm7.2`**. Both
  `whl/rocm7.14/` and `whl/rocm10.0/` return 403 — they were never published, so
  substituting your ROCm version into that path only works on the 7.x line.
- ROCm 10 torch is routed through TheRock at
  [`stable.repo.amd.com/rocm/whl-next/`](https://stable.repo.amd.com/rocm/whl-next/),
  which carries `torch` 2.11.0, 2.12.0 and 2.13.0 (all `+rocm10.0.0`, cp310
  through cp314) alongside `torchvision`, `torchaudio` and the `rocm-sdk-*`
  wheels. These are release builds, so no `--pre` is needed.
- The ROCm 10 **nightly** channel is a trap worth knowing about:
  `whl/nightly/rocm10.0/` responds 200, so it looks like the old pattern still
  works, but it serves only shared dependencies (numpy, sympy, filelock, …) and
  no `torch` at all. A 200 from the index root is not evidence that torch is
  there.

[pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/)
remains the authority on which channel is current; check it before pinning, since
the ROCm 10 line is new and the 7.2 stable index is the retiring one.

Other optional extras include `analysis`, `report`, `hw-queue-profiling`,
`agent`, and `ebpf`. The `ebpf` extra has no additional Python packages; its
runtime needs `bpftrace` and the required permissions.

## Where commands run

`aorta` runs in the active Python environment, and relative recipe, command,
and output paths are resolved from the current directory. The examples below
use paths from the AORTA repository root.

There is no global rule that AORTA must run on the host or inside a container:

- Run `aorta env probe` in the environment you want to inspect, including
  inside a container when appropriate. To inspect an image without installing
  AORTA in it, use the [package-mount workflow](docs/env-probe.md#probe-a-docker-image-without-aorta-installed).
- Some workloads run in the same environment as the CLI.
- Some workload plugins own a `docker run` launch. For those workloads, follow
  the plugin's instructions; the CLI normally runs on a Docker-capable host,
  while the workload image supplies its runtime dependencies.

The core dispatcher does not execute `docker run`; Docker-aware workload
plugins may do so and own the launch.

## Quick Start

```bash
# --- Environment snapshot ---
aorta env probe -o env.json                      # full snapshot to disk
aorta env probe --summary                        # one-screen brief, no file write
aorta env probe --field pytorch_build.git_commit # one field, JSON-typed
diff <(jq -S . env_a.json) <(jq -S . env_b.json) # diff two snapshots

# --- Unified sweep (recipe-driven) ---
aorta sweep run --recipe recipes/llm-determinism/example-llm-determinism.yaml --dry-run   # validate only
aorta sweep run --recipe recipes/llm-determinism/example-llm-determinism.yaml             # run the matrix

# Distributed workloads launch under torchrun (llm_determinism, race, fsdp):
torchrun --standalone --nproc_per_node=2 $(which aorta) \
  sweep run --recipe recipes/training/example-fsdp-smoke.yaml
torchrun --standalone --nproc_per_node=1 $(which aorta) \
  run --workload llm_determinism --trials 1 --steps 50

# --- Unified sweep (wrap an opaque launch command) ---
aorta sweep run --recipe recipes/probe/probe-template-bash.yaml \
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
[`docs/probe/usage.md`](docs/probe/usage.md).

At the end of every run `aorta sweep run` prints a concise summary to stdout —
which cells failed vs. errored, the workload's own failure hint, and the path to
each failing cell's artifact directory (logs + per-trial JSON) — so you don't
have to open `matrix.md` to find what broke. Pass `-v` (`-vv`) to also stream
live per-cell progress to stderr while a long matrix runs.

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
| [Getting Started](docs/getting-started.md) | Installation, command location, and workload-specific prerequisites |
| [Hardware Queue Eval](docs/hw-queue-eval.md) | Workloads, CLI usage, metrics |
| [Environment Probe](docs/env-probe.md) | Capture / diff / query a versioned environment snapshot; jq cookbook |
| [`aorta sweep`](docs/probe/usage.md) | Unified matrix runner — built-in workloads **or** opaque launch commands |
| [LLM Determinism](docs/llm-determinism.md) | Bit-exact double-run nondeterminism probe |
| [Layer Numerics](docs/layer-numerics.md) | Per-layer / per-stage NaN, magnitude, and out-of-range logger |
| [Profiling Collectors](docs/profiling-collectors.md) | `--collect rocprof` (wraps any subprocess command) / `--collect proton` (Python launches, or `mode: env`) — attach a GPU profiler without editing the payload |
| [TokenSpeed](docs/tokenspeed.md) | Third-party inference engine under `mode: probe` — kernel, operator-suite and serving triage, plus Waitcheck and ConSan over its JIT attention/gemm kernels (gfx950) |
| [TokenSpeed serving](docs/tokenspeed-serving.md) | `tokenspeed_serve` workload — TTFT / TPOT / ITL / throughput per cell, over model, concurrency and ISL/OSL sweeps (gfx950) |
| [TokenSpeed in `main`](docs/tokenspeed-in-main.md) | Orientation over the whole TokenSpeed integration — all five routes, the committed recipes, how to run and test them, and the measured numbers |
| [`aorta agent`](docs/agent/agentic-testing-guide.md) | Closed-loop mitigation search (optional LLM proposer) |
| [`aorta bundle`](docs/probe/bundle.md) | Package sweep artifacts with recipe-driven redaction |
| [Recipes](recipes/README.md) | Recipe schema and running recipes |
| [Buck2 Build Reference](docs/buck2-build-reference.md) | Build / run the AORTA CLI via Buck2 |


## Repository Layout

```
src/aorta/
├── cli/               # `aorta` CLI command groups (sweep, run, env, bundle, agent, ...)
├── workloads/         # In-tree workloads (training, inference, llm_determinism, race)
├── instrumentation/   # Environment probe (env.json) + layer_numerics NaN/OOB logger
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
