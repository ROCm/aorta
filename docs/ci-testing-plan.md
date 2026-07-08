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
| `.[tests]` | pytest + `pytest-xdist` / `pytest-forked` / `pytest-timeout` / `pytest-cov` only. This slim extra deliberately excludes the lint/type tooling in `.[dev]` (black/isort/mypy/ruff/pre-commit), which the pytest-only job never uses. |
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
real breakage; drop each pin as the underlying test is modernized. Modernizing
`test_sweep_cli.py` and dropping this pin is tracked in
[#269](https://github.com/ROCm/aorta/issues/269).

The pin lives in the constraints file rather than the `tests` extra on purpose:
`tests` feeds `dev` and `all`, so a `click<8.2` cap there would leak a temporary
compat bound into every dev/full install (and none of the other extras carry an
upper bound). Keeping it in `PIP_CONSTRAINT` keeps the cap CI-scoped and trivial
to drop.

### Running the gate locally

Reproduce the exact CI environment (including the `click<8.2` pin) by pointing
pip at the same constraints file -- no manual version juggling:

```bash
export PIP_CONSTRAINT=.github/ci-constraints.txt
pip install -e ".[tests,hw-queue]"
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install triton
pytest -m "not gpu and not rocm" -n auto
```

(`bpftrace` still needs to be on the box for the `aorta.ebpf` tests; on Ubuntu
that's `sudo apt-get install -y bpftrace`. Or just `pip install "click<8.2"`
into the venv if you'd rather not set the env var -- the pin is temporary.)

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

## Phase 2 - GPU test gate (planned, needs a GPU runner)

Tracked in [#268](https://github.com/ROCm/aorta/issues/268).
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

## Follow-up strategy

With the Phase 1 CPU gate in place, the remaining work is tracked as separate
issues so each can be picked up independently:

| Follow-up | Tracked in |
| --- | --- |
| Phase 2 GPU test gate on a self-hosted runner | [#268](https://github.com/ROCm/aorta/issues/268) |
| Modernize `test_sweep_cli.py` for `click>=8.2` and drop the `click<8.2` pin | [#269](https://github.com/ROCm/aorta/issues/269) |
| ~~Fix cross-file state pollution so the suite runs without `--forked`~~ (done: [#270](https://github.com/ROCm/aorta/issues/270)) | [#270](https://github.com/ROCm/aorta/issues/270) |

Not an issue: making the `pytest (CPU, py3.x)` jobs **required status checks** on
`main` is a repo/admin setting (see "Making it a required check" above), not a
code change -- with `cpu-tests.yml` present on `main`, it is configured once in
the repository's branch-protection settings.

## Summary

| Phase | Runner | Selection | Status |
| --- | --- | --- | --- |
| 1 - CPU gate | `ubuntu-latest` (3.10-3.12) | `not gpu and not rocm`, `-n auto` | Implemented (`cpu-tests.yml`) |
| 2 - GPU gate | `[self-hosted, gpu]` | `gpu or rocm` + real-hw + regression | Planned (needs runner) |
