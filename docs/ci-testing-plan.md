# CI testing plan

Tracking issue: [#264](https://github.com/ROCm/aorta/issues/264)

This document describes how automated tests gate pull requests in this repo,
and the phased plan for growing that coverage from CPU-only today to GPU
hardware once a suitable runner exists.

## Goal

Every pull request must run the automated test suite, and a PR must not merge
while those tests are red. We start with the large CPU-runnable slice of the
existing `tests/` suite (Phase 1), and add GPU-hardware coverage later on a
self-hosted runner (Phase 2).

## Current CI (before this plan)

| Workflow | Trigger | What it does |
| --- | --- | --- |
| `pre-commit.yml` | PR + push to `main` | Runs `pre-commit` hooks (whitespace, EOF, YAML) |
| `nightly.yml` | cron + dispatch | Builds/publishes rolling dev wheels |
| `release.yml` / `cleanup_releases.yml` | tags / cron | Release packaging + asset pruning |
| `gemm-sweep-analysis.yml`, `rccl-warp-speed-analysis.yml` | cron / dispatch | Scheduled analysis jobs |

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
| `.[dev]` | pytest, `pytest-xdist`, `pytest-forked`, and lint/type tooling. |
| `.[hw-queue]` | `numpy` / `pandas` / `tabulate` required by the `hw_queue_eval` tests (they error at collection without them). |
| `triton` | The env-probe contract tests (`tests/instrumentation/test_environment.py`) treat a "clean full probe" as non-partial only when `triton` is importable. |
| `bpftrace` (apt) | The `aorta.ebpf` runner tests set an explicit `bpftrace_path` and the runner validates a real file exists at `/usr/bin/bpftrace` before building the command. The subprocess itself is mocked (`FakePopen`), so only the binary's presence matters; apt installs it at exactly that path. |

Installs go through a small pip constraints file,
[`.github/ci-constraints.txt`](../.github/ci-constraints.txt) (wired in via
`PIP_CONSTRAINT`). Today it pins only `click<8.2`: `tests/sweep/test_sweep_cli.py`
uses `CliRunner(mix_stderr=False)`, which Click 8.2 removed. The runtime floor is
`click>=8.0` and those tests depend on the pre-8.2 behaviour, so we pin the test
environment instead of editing the tests. The file is intentionally minimal --
pin only known-incompatible pieces so the gate stays reproducible without hiding
real breakage; drop each pin as the underlying test is modernized.

### Why per-test process isolation (`--forked`)

The command is:

```
pytest -m "not gpu and not rocm" -n auto --forked
```

Running the whole suite in a single interpreter produces ~100 failures whose
membership shifts with test ordering. Every affected file passes cleanly when
run on its own, which is the signature of **cross-file global-state pollution**
(leaked env vars, patched module globals, cached imports), not real breakage.

Rather than couple the gate to that latent pollution (which would make it flaky
and order-dependent), each test runs in its own forked subprocess
(`pytest-forked`), and `-n auto` (`pytest-xdist`) parallelizes across cores to
keep wall-clock time down (~1-2 min). This is deterministic today and stays
green as the offending fixtures are cleaned up over time.

> Follow-up (out of scope for the gate): track down and fix the specific
> fixtures that leak global state so the suite can eventually run pollution-free
> in a single process. Until then, `--forked` is the safety net, not a crutch to
> hide new pollution behind.

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

## Phase 2 - GPU test gate (planned, needs a GPU runner)

Blocked on: a GPU/self-hosted CI runner being wired up (e.g. MI300X/MI350X).

When that runner exists, add a `gpu-tests.yml` workflow:

- **Runner:** `runs-on: [self-hosted, gpu]` (label the runner accordingly).
- **Selection:** `pytest -m "gpu or rocm"` -- the complement of the Phase-1
  selection, so the two gates together cover the full suite with no overlap.
- **Real-hardware surfaces** currently only smoke-tested or mocked on CPU:
  - `aorta.ebpf` against a real `bpftrace` + BPF (needs `CAP_BPF`/`CAP_PERFMON`
    or root on the runner).
  - torch GPU workloads (`race`, `training`, `inference`, `llm_determinism`)
    on real devices.
  - `gpu_control` lock-clock / power-limit against real `rocm-smi`.
- **Triggers:** start on `workflow_dispatch` + nightly `schedule` to protect
  scarce GPU capacity; promote to a required `pull_request` check for
  GPU-touching paths once it is proven stable.
- **Regression gates:** fold in the nightly `aorta run` / `aorta triage`
  regression pattern (pin docker images by digest) once the runner and images
  are reachable from CI.

### Cost / capacity notes

- Keep GPU runs off the hot PR path initially (dispatch + nightly) so a queue of
  PRs cannot starve the single GPU runner.
- Consider a `paths:` filter so GPU CI only triggers for changes under GPU-
  relevant directories (`src/aorta/race/**`, `src/aorta/ebpf/**`,
  `src/aorta/utils/gpu_control.py`, workloads) once it becomes a PR check.

## Summary

| Phase | Runner | Selection | Status |
| --- | --- | --- | --- |
| 1 - CPU gate | `ubuntu-latest` (3.10-3.12) | `not gpu and not rocm`, `--forked` | Implemented (`cpu-tests.yml`) |
| 2 - GPU gate | `[self-hosted, gpu]` | `gpu or rocm` + real-hw + regression | Planned (needs runner) |
