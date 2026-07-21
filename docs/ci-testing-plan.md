# CI testing plan

Tracking issue: [#264](https://github.com/ROCm/aorta/issues/264)

This document describes how automated tests gate pull requests in this repo.
Phase 1 (CPU) and Phase 2 (GPU on a self-hosted MI350 runner) together cover
the full `tests/` suite with no overlap.

## Goal

Every pull request must run the automated test suite, and a PR must not merge
while those tests are red. Phase 1 runs the CPU-runnable slice on GitHub-hosted
runners; Phase 2 runs the GPU complement on the self-hosted MI350 runner
(`smci350-rck-g03-f16-12.rck.dcgpu`).

## Current CI

| Workflow | Trigger | What it does |
| --- | --- | --- |
| `pre-commit.yml` | PR + push to `main` | Runs `pre-commit` hooks (whitespace, EOF, YAML) |
| `cpu-tests.yml` | PR + push to `main` | CPU pytest gate (`not gpu and not rocm`) on `ubuntu-latest` |
| `gpu-tests.yml` | PR (GPU paths) + nightly + dispatch | GPU pytest gate + nightly workload regression on `[self-hosted, gpu]` |
| `nightly.yml` | cron + dispatch | Builds/publishes rolling dev wheels |
| `release.yml` / `cleanup_releases.yml` | tags / cron | Release packaging + asset pruning |
| `gemm-sweep-analysis.yml`, `rccl-warp-speed-analysis.yml` | cron / dispatch | Scheduled analysis jobs |

## Historical context (before Phase 1)

Notably, **the `tests/` suite was not run in CI**. `pre-commit.yml` only lints;
it never executes pytest. A PR could break ~2,000 tests and still merge.

## Phase 1 - CPU test gate (implemented)

Workflow: [`.github/workflows/cpu-tests.yml`](../.github/workflows/cpu-tests.yml)

- **Triggers:** `pull_request` (every PR) and `push` to `main`.
- **Runner / matrix:** `ubuntu-latest`, Python `3.10`, `3.11`, `3.12` (the
  versions declared in `pyproject.toml`). `fail-fast: false` so one version's
  failure still reports the others.
- **Concurrency:** newer pushes to the same ref cancel the older run.
- **Selection:** `pytest -m "not gpu and not rocm"`. The `gpu` and `rocm`
  markers already exist (`pytest.ini`); GPU-only tests are deferred to Phase 2.

### Environment the CPU tests need

The "CPU" slice is not pure standard-library unit tests; parts of it exercise
real integration surfaces. The workflow provides exactly what those tests
assume, and nothing heavier:

| Dependency | Why it is needed |
| --- | --- |
| CPU-only PyTorch (`--index-url .../whl/cpu`) | Several test modules `import torch` at collection time. No CPU test needs a GPU, so the CPU wheel avoids pulling CUDA/ROCm runtimes. |
| `.[tests]` | pytest + `pytest-xdist` / `pytest-forked` / `pytest-timeout` / `pytest-cov` only. This slim extra deliberately excludes the lint/type tooling in `.[dev]` (black/isort/mypy/ruff/pre-commit), which the pytest-only job never uses. |
| `.[hw-queue]` | `numpy` / `pandas` / `tabulate` required by the `hw_queue_eval` tests (they error at collection without them). |
| `triton` | The env-probe contract tests (`tests/instrumentation/test_environment.py`) treat a "clean full probe" as non-partial only when `triton` is importable. |
| `bpftrace` (apt) | The `aorta.ebpf` runner tests set an explicit `bpftrace_path` and the runner validates a real file exists at `/usr/bin/bpftrace` before building the command. The subprocess itself is mocked (`FakePopen`), so only the binary's presence matters; apt installs it at exactly that path. |

The suite installs unpinned dependencies: it runs green on the current `click`
(>=8.2) as well as the `click>=8.0` floor declared in `pyproject.toml`. (An
earlier revision pinned `click<8.2` via a `.github/ci-constraints.txt`
constraints file because `tests/sweep/test_sweep_cli.py` used the
`CliRunner(mix_stderr=False)` argument that Click 8.2 removed; that test now
feature-detects the argument, so the pin and the constraints file are gone --
see [#269](https://github.com/ROCm/aorta/issues/269).) If a future
test/dependency incompatibility needs a CI-scoped pin, re-introduce a
`PIP_CONSTRAINT` constraints file (kept minimal, pinning only known-incompatible
pieces) rather than capping versions in the `tests` extra, which feeds `dev` /
`all` and would leak a compat bound into every dev/full install.

### Running the gate locally

Reproduce the CI environment -- no manual version juggling:

```bash
pip install -e ".[tests,hw-queue]"
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install triton
pytest -m "not gpu and not rocm" -n auto
```

(`bpftrace` still needs to be on the box for the `aorta.ebpf` tests; on Ubuntu
that's `sudo apt-get install -y bpftrace`.)

### Parallelism (`-n auto`, no `--forked`)

The command is:

```
pytest -m "not gpu and not rocm" -n auto
```

`-n auto` (`pytest-xdist`) parallelizes across cores to keep wall-clock time
down (~1-2 min). Per-test process isolation (`--forked`) is **not** used.

Historically the suite could not run pollution-free in a single interpreter:
running everything together produced ~100 failures whose membership shifted
with test ordering, while every affected file passed on its own -- the
signature of cross-file global-state pollution. The gate leaned on `--forked`
as a safety net while the leak was tracked down.

That pollution came from two culprits, both fixed in
[#270](https://github.com/ROCm/aorta/issues/270):

1. **Leaked `subprocess.Popen`.** The `aorta.ebpf` runner tests patched the
   shared stdlib `subprocess.Popen` with a hand-entered `mock.patch`
   (`ctx.__enter__()`), relying on the test body's `finally` to undo it. When
   `start()` raised before that `finally` was established (e.g. a missing
   `bpftrace` binary rejected in `_build_command`), the patch leaked, so every
   later `subprocess.run` in the interpreter hit the fake `Popen`. Fixed by
   restoring the patch via the `monkeypatch` fixture, which tears down
   regardless of whether `start()` raised.

2. **Corrupted `triton` in `sys.modules`.** Many dispatcher/discovery tests
   patch `importlib.metadata.entry_points` with a `MagicMock` to fake
   `aorta.workloads` discovery. Triton discovers its compiler backends lazily on
   first import via that same `entry_points` API, so if Triton's first import
   happened under the mock (e.g. `run_trials` -> `collect_env` probes the Triton
   version), backend discovery raised and left `sys.modules` half-initialized
   (`triton.backends.compiler` cached but `triton` gone). That then broke
   `torch.use_deterministic_algorithms` in every later workload test. Fixed by
   pre-importing Triton once in `tests/conftest.py`, against the real
   `entry_points`, so later mocked imports are harmless cache hits.

The suite is now deterministic in one interpreter (verified across randomized
orderings), so `--forked` is dropped and only `-n auto` is kept, for speed.

### Making it a required check

To actually block merges, add the `tests` jobs as **required status checks** on
the `main` branch:

`Settings -> Branches -> Branch protection rules -> main -> Require status
checks to pass before merging`, then select:

- `pytest (CPU, py3.10)`
- `pytest (CPU, py3.11)`
- `pytest (CPU, py3.12)`

(The workflow runs on PRs regardless; branch protection is what makes a red run
*block* the merge.)

## Phase 2 - GPU test gate (implemented)

Tracked in [#268](https://github.com/ROCm/aorta/issues/268).
Runner: `smci350-rck-g03-f16-12.rck.dcgpu` (MI350 / gfx950), labels
`self-hosted`, `gpu`.

Workflow: [`.github/workflows/gpu-tests.yml`](../.github/workflows/gpu-tests.yml)

### Marker reconciliation

The GPU gate selects tests with `pytest -m "gpu or rocm"`. Before Phase 2,
most GPU-only tests (`tests/hw_queue_eval/*`) gated on
`skipif(not torch.cuda.is_available())` without a `gpu` marker, so they were
incorrectly included in the CPU gate (where they skipped at runtime). Those
modules now carry `@pytest.mark.gpu` alongside the existing skip.

[`tests/test_marker_partition.py`](../tests/test_marker_partition.py) asserts
that the CPU and GPU marker selections partition the suite (no overlap, no
gaps). New GPU tests must carry `@pytest.mark.gpu` or `@pytest.mark.rocm`.

### Jobs

| Job | Triggers | What it runs |
| --- | --- | --- |
| `pytest (GPU, MI350)` | `pull_request` (GPU-touching paths), nightly cron, `workflow_dispatch` | `pytest -m "gpu or rocm" -n 4` inside a digest-pinned ROCm container (bounded workers so xdist doesn't oversubscribe the single GPU) |
| `workload regression (GPU, MI350)` | `pull_request` (GPU-touching paths), nightly cron, `workflow_dispatch` | Real-hardware workload smokes from [`config/ci/gpu_regression_smokes.yaml`](../config/ci/gpu_regression_smokes.yaml) via [`scripts/ci/run_gpu_regression_smokes.sh`](../scripts/ci/run_gpu_regression_smokes.sh). PRs run the fast single-GPU `pr` tier; nightly / dispatch run the full manifest |

### Execution environment (docker)

GPU CI runs inside a privileged ROCm PyTorch container (same device/capability
model as the existing analysis workflows):

- Compose file: [`docker/docker-compose.build.yaml`](../docker/docker-compose.build.yaml)
- CI env file: [`docker/.env.ci`](../docker/.env.ci) (committed; pins base image digest)
- CI Dockerfile: [`docker/Dockerfile.ci-gpu`](../docker/Dockerfile.ci-gpu)
- Container name: `aorta-ci-gpu`
- Workspace mount: repo root at `/workspace/aorta`

The base image is pinned by digest:

```
rocm/pytorch@sha256:376bfab5f4f680c8b4b843c6d0c5d1f0a04e5a84ec3e86728db8d11d79a9d1e3
```

(tag: `rocm7.2_ubuntu22.04_py3.10_pytorch_release_2.9.1`). Bump the digest in
`Dockerfile.ci-gpu` / `.env.ci` when intentionally upgrading the CI stack.

### Triggers and frequency

| Event | GPU pytest job | Workload regression job |
| --- | --- | --- |
| `pull_request` (GPU-touching paths) | yes | yes (`pr` tier: fast, single-GPU) |
| nightly cron (`0 8 * * *` UTC) | yes | yes (full manifest) |
| `workflow_dispatch` | yes | yes (full manifest) |

The regression tier is chosen by the workflow via `AORTA_CI_TIER` (`pr` on
pull requests, `full` otherwise). Mark a manifest entry with `pr: true` to
include it in the PR gate; keep heavier / multi-GPU smokes (e.g. the 2-GPU
`race_smoke`) out of the PR tier so PRs stay fast and never starve the single
runner.

PR path filter (GPU-touching changes only):

- `src/aorta/race/**`, `src/aorta/ebpf/**`, `src/aorta/hw_queue_eval/**`,
  `src/aorta/workloads/**`, `src/aorta/utils/gpu_control.py`
- matching tests, `config/ci/**`, `recipes/ci/**`, `scripts/ci/**`, `docker/**`,
  `.github/workflows/gpu-tests.yml`

**Concurrency:** one GPU workflow run per ref; newer pushes cancel superseded PR
runs so the single runner is not starved by stale jobs.

### Running the GPU gate locally

On a ROCm box with the repo mounted into the CI container:

```bash
cd docker
docker compose --env-file .env.ci -f docker-compose.build.yaml up -d --build
docker exec aorta-ci-gpu bash -lc '
  cd /workspace/aorta &&
  pip install -e ".[tests,hw-queue]" &&
  pytest -m "gpu or rocm" -n 4
'
docker compose --env-file .env.ci -f docker-compose.build.yaml down -v
```

Workload regression smokes (inside the same container):

```bash
docker exec aorta-ci-gpu bash -lc '
  cd /workspace/aorta &&
  pip install -e ".[tests,hw-queue]" &&
  bash scripts/ci/run_gpu_regression_smokes.sh
'
```

### Extending workload regression coverage

Add entries to [`config/ci/gpu_regression_smokes.yaml`](../config/ci/gpu_regression_smokes.yaml).
Each entry lists a command argv and optional `min_gpus` / `pr`. The runner
script skips entries when insufficient GPUs are present, and (on the PR gate)
skips entries not marked `pr: true`. No workflow edits are required when adding
new workloads -- register the workload via the `aorta.workloads` entry-point
group and add a smoke recipe/command to the manifest.

Current smokes: `gpu_smoke` (recipe + CLI, PR tier), `inference` smoke (PR
tier), `race` smoke (nightly only, requires 2 GPUs).

### Making the GPU gate a required check

After a stable soak on the runner, add the GPU jobs as **required status
checks** on `main`:

`Settings -> Branches -> Branch protection rules -> main -> Require status checks
to pass before merging`, then select:

- `pytest (GPU, MI350)`
- `workload regression (GPU, MI350)`

Because PR triggers are path-filtered, only PRs touching GPU-relevant paths will
report this check. Other PRs can merge without it (same pattern as optional
checks that did not run). Repo admins enable this once nightly + PR runs are
reliably green.

### Cost / capacity notes

- PR GPU runs are limited to GPU-touching paths so doc-only PRs do not consume
  the runner.
- Nightly runs catch drift without blocking every PR.
- The workload regression job runs on PRs too, but only the fast single-GPU
  `pr` tier; the heavier full manifest (e.g. the 2-GPU `race` smoke) is reserved
  for nightly / dispatch so PR latency stays low.

## Follow-up strategy

With the Phase 1 CPU gate in place, the remaining work is tracked as separate
issues so each can be picked up independently:

| Follow-up | Tracked in |
| --- | --- |
| ~~Phase 2 GPU test gate on a self-hosted runner~~ (done) | [#268](https://github.com/ROCm/aorta/issues/268) |
| Mirror GPU CI in `aorta-internal` (CPU + GPU gates, workload regression) | [#72](https://github.com/ROCm/aorta-internal/issues/72) |
| ~~Fix cross-file state pollution so the suite runs without `--forked`~~ (done) | [#270](https://github.com/ROCm/aorta/issues/270) |

Done: [#269](https://github.com/ROCm/aorta/issues/269) modernized
`test_sweep_cli.py` for `click>=8.2` and removed the `click<8.2` CI pin.

Not an issue: making the `pytest (CPU, py3.x)` and **`pytest (GPU, MI350)`**
jobs **required status checks** on `main` is a repo/admin setting (see
"Making it a required check" / "Making the GPU gate a required check" above),
not a code change.

## Summary

| Phase | Runner | Selection | Status |
| --- | --- | --- | --- |
| 1 - CPU gate | `ubuntu-latest` (3.10-3.12) | `not gpu and not rocm`, `-n auto` | Implemented (`cpu-tests.yml`) |
| 2 - GPU gate | `[self-hosted, gpu]` | `gpu or rocm`, `-n auto` + nightly workload regression | Implemented (`gpu-tests.yml`) |
