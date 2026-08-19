# CI testing plan

Tracking issue: [#264](https://github.com/ROCm/aorta/issues/264)

This document describes how automated tests gate pull requests in this repo.
Phase 1 (CPU) and Phase 2 (GPU on a self-hosted MI350 runner) together cover
the full `tests/` suite with no overlap.

## Goal

Every pull request must run the automated test suite, and a PR must not merge
while those tests are red. Phase 1 runs the CPU-runnable slice on GitHub-hosted
runners; Phase 2 runs the GPU complement on a self-hosted MI350 runner
(labels `self-hosted`, `gpu`).

## Current CI

| Workflow | Trigger | What it does |
| --- | --- | --- |
| `pre-commit.yml` | PR + push to `main` | Runs `pre-commit` hooks (whitespace, EOF, YAML) |
| `cpu-tests.yml` | PR (code-touching paths) + push to `main` | CPU pytest gate (`not gpu and not rocm`) on `ubuntu-latest` |
| `gpu-tests.yml` | PR (GPU paths) + nightly + dispatch | GPU pytest gate + nightly workload regression on `[self-hosted, gpu]` |
| `nightly.yml` | cron + dispatch | Builds/publishes rolling dev wheels |
| `release.yml` / `cleanup_releases.yml` | tags / cron | Release packaging + asset pruning |
| `gemm-sweep-analysis.yml`, `rccl-warp-speed-analysis.yml` | cron / dispatch | Scheduled analysis jobs |

## Historical context (before Phase 1)

Notably, **the `tests/` suite was not run in CI**. `pre-commit.yml` only lints;
it never executes pytest. A PR could break ~2,000 tests and still merge.

## Phase 1 - CPU test gate (implemented)

Workflow: [`.github/workflows/cpu-tests.yml`](../.github/workflows/cpu-tests.yml)

- **Triggers:** `pull_request` and `push` to `main`. On PRs a cheap `changes`
  job first decides whether the suite is relevant (see "Path gate" below); it
  always runs on pushes to `main`.
- **Runner / matrix:** `ubuntu-latest`, Python `3.10`, `3.11`, `3.12`, `3.13`,
  `3.14` (the versions declared in `pyproject.toml`). `fail-fast: false` so one
  version's failure still reports the others.
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

### Path gate (fail-open deny-list)

The CPU suite is a required status check, so the workflow can't use a
trigger-level `paths:` filter: GitHub leaves a path-skipped required check
Pending forever, making the PR unmergeable. Instead a cheap `changes` job runs
first and a job-level `if` decides whether the matrix runs -- the same pattern
[`gpu-tests.yml`](../.github/workflows/gpu-tests.yml) uses.

There is one crucial difference from the GPU gate, though. GPU's gated job is a
single, non-matrix job (`pytest (GPU, MI350)`): when skipped, its one static
check context still reports (as *skipped* == success), so it can be the required
check directly. The CPU job is a **matrix** (py3.10 through py3.14). GitHub evaluates
a job-level `if` *before* expanding the matrix, so a skipped `tests` job never
creates the `pytest (CPU, py3.x)` contexts at all -- and a required check pinned
to a context that never reports hangs the PR forever
([actions/runner#952](https://github.com/actions/runner/issues/952)). So the CPU
workflow adds a stable, non-matrix `required` job (check name **`CPU tests`**,
`if: always()`) that collapses the matrix result into one context that reports
on every PR: it passes when the matrix ran green or was legitimately skipped,
and fails if any leg failed. **That** aggregator is the required check, not the
per-version legs.

The relevance decision itself: the GPU gate allow-lists a small set of
GPU-relevant paths. The CPU gate is the opposite shape -- it's the catch-all
suite touched by nearly the whole tree (`src/**`, `tests/**`, packaging
metadata, and even a few docs -- e.g. `test_layer_numerics_docs.py` reads
`docs/layer-numerics.md`), so an allow-list would be huge and dangerous to get
wrong. It instead uses a small **deny-list** and **fails open**: the suite runs
unless *every* changed file matches a path that provably never feeds
`pytest -m "not gpu and not rocm"`. The safety is **asymmetric**: an *omitted*
deny entry only costs an unnecessary (fast) run, but an *overbroad or wrong*
entry misclassifies a relevant change as ignorable and silently skips real
tests. So new/unknown paths run the suite, and every deny entry must be proven
inert before it is added. On any error listing the PR's files -- including a
truncated listing past the 3000-file API cap -- it runs the suite. Renames are
evaluated on both sides, so moving a source file *into* an ignored path still
counts as a relevant (source-removing) change.

Currently the deny-list is just `.github/*` (CI workflows, composite actions,
templates, CODEOWNERS -- none of which any test imports), with a special case so
that a change to `cpu-tests.yml` itself still exercises the gate. This is why a
CI-only PR such as [#337](https://github.com/ROCm/aorta/pull/337) (which only
edited `sanitizers-nightly.yml`) no longer spins up the three-Python CPU matrix.

### Making it a required check

To actually block merges, add the aggregator as a **required status check** on
the `main` branch:

`Settings -> Branches -> Branch protection rules -> main -> Require status
checks to pass before merging`, then select:

- `CPU tests`

Do **not** require the per-version `pytest (CPU, py3.x)` legs: because
the matrix is skipped by a job-level `if` on irrelevant PRs, those contexts are
not emitted and a required check pinned to them would hang the PR forever
([actions/runner#952](https://github.com/actions/runner/issues/952)). The `CPU
tests` aggregator reports on every PR and already fails if any matrix leg fails,
so it is the correct single required check.

(The workflow runs on PRs regardless; branch protection is what makes a red run
*block* the merge.)

## Phase 2 - GPU test gate (implemented)

Tracked in [#268](https://github.com/ROCm/aorta/issues/268).
Runner: a self-hosted MI350 (gfx950) machine, labels `self-hosted`, `gpu`.

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
rocm/pytorch@sha256:4449f856653602317e4101a76fce599c7fcd58ccec2e539951fce5f73083179e
```

(tag: `rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0`). Bump the digest in
`Dockerfile.ci-gpu` / `.env.ci` when intentionally upgrading the CI stack.

That tag is the newest published combination on every axis at once — ROCm 7.2.4
is the newest classic-layout production release, and 24.04 / py3.12 / torch
2.10.0 are the newest Ubuntu, Python and PyTorch that line ships. Newer Python
(3.13, 3.14) and newer PyTorch (2.11, 2.12) exist **only** on the wheel-based
7.14 line, so they arrive with the #381 flip rather than separately.

Python's second, independent cap has now been lifted: `pyproject.toml` declares
through 3.14 and the CPU matrix tests 3.10–3.14 (issue #383). The GPU gate stays
on py3.12 only because that is the newest Python the classic ROCm line ships, so
the remaining move is the base-image flip, not a packaging change.

CI tracks the newest ROCm production release it can actually run, so the nightly
eval reports against the stack customers run. Two constraints bound "newest", and
a bump proposal — from a human or from automation — must clear both:

1. **Preview stream.** ROCm 7.9 through 7.13 are the *technology preview* stream,
   not newer production releases, so a higher number there is not an upgrade.
2. **Install layout.** 7.14 onward is production, but its `rocm/pytorch` images
   are wheel-based (see below), which this repo cannot read yet.

#### ROCm install layout: both are supported (issue #381)

TheRock publishes ROCm both as a system install rooted at `/opt/rocm` (DEB/RPM,
tarballs) and as a Python wheel rooted under `site-packages` with no `/opt/rocm`.
**Both layouts are readable.** Discovery goes through
`aorta.instrumentation.rocm_paths.resolve_rocm_roots`, which resolves three roots
(core / libraries / include) so the wheel layout's split across
`_rocm_sdk_core`, `_rocm_sdk_libraries` and `_rocm_sdk_devel` can be expressed.
On a classic install all three coincide at `/opt/rocm`, so nothing changed there.

What still has to fail closed is an install we cannot *read*. Everything we use
to read ROCm degrades to `null`/empty rather than erroring — the env probe's
version plus its hipBLASLt-commit and rocBLAS capture,
`scripts/audit_env_knobs.py`, and the sanitizer GEMM fixtures — so a bad base
image would gut the evidence while CI stayed green. `docker/rocm_layout_guard.py`
runs at build time in `Dockerfile.ci-gpu` and `Dockerfile.rocm-latest`, accepts
either layout, and fails the build when neither yields a readable version marker
and lib directory. It is pinned to the resolver by
`tests/docker/test_rocm_layout_guard.py`, which runs both implementations over
the same synthetic trees.

Neither image sets `ENV ROCM_HOME=/opt/rocm` any more. An explicit override
ranks above autodetection, so declaring it unconditionally lets a stale
`/opt/rocm` stub win over a correct wheel install; the classic base already
exports `/opt/rocm/bin` on `PATH` and `/opt/rocm/lib` on `LD_LIBRARY_PATH`
itself, so removing it changes nothing there.

Which images are which, and how to tell before pulling:

| Image | Layout | Tell |
|---|---|---|
| `rocm7.2.x` tags | classic | `/opt/rocm/bin` on `PATH`; no `npi.*` labels |
| `rocm7.14+` tags, `rocm/pytorch:latest` | wheel (TheRock) | `npi.*` labels present; no `/opt/rocm` on `PATH` |

Check before pulling — a classic image carries `/opt/rocm/bin` on `PATH`, a
wheel-based one does not:

```bash
docker buildx imagetools inspect rocm/pytorch:<tag> \
  --format '{{range .Image.Config.Env}}{{println .}}{{end}}' | grep ^PATH=
# classic 7.2.4 -> PATH=/opt/venv/bin:/opt/rocm/bin:...
# wheel   7.14  -> PATH=/opt/venv/bin:...            (no /opt/rocm/bin)

# The npi.* labels are the same signal from the other side:
docker buildx imagetools inspect rocm/pytorch:<tag> --format '{{json .Image.Config.Labels}}'
```

The wheel layout is *richer* in provenance, which is why reading it was worth
doing rather than merely tolerating: `share/therock/therock_manifest.json`
carries the full 40-char `rocm-libraries` commit — the classic header only
exposes a truncated `..._VERSION_TWEAK` (measured 10 chars on 7.2.4) — plus the
build's `the_rock_commit`, `github_run_id` and any patches applied on top of
each upstream pin. Schema 1.16 records all of it under `therock`, and the GEMM
libraries' `upstream_commit`.

The base image itself is still the classic 7.2.4 line; flipping it to 7.14 is
issue #383 and a deliberate separate step, since it moves ROCm, Python and
PyTorch at once.

#### The latest-ROCm canary lane (non-gating)

`.github/workflows/latest-rocm-canary.yml` follows `rocm/pytorch:latest` — a
wheel-layout image, hence dependent on the discovery work above — so a new ROCm
release is noticed early. Pointing the *gate* at a moving tag is what this
deliberately avoids: a red result would be ambiguous (our regression, or did the
base change?) and neither reproducible nor bisectable afterwards. So the lane
resolves `:latest` to a concrete digest at job start, records it with the
results, and stays out of the gate's way.

Non-gating is structural here, not just intended:

| Mechanism | Effect |
|---|---|
| its own workflow | cannot appear in `nightly-eval.yml`'s graph, cannot be added to branch protection by accident |
| eval exit code captured, not propagated | a regression on a brand-new ROCm records a row instead of a red X nobody can action |
| results published to `results/canary/` | `gen_dashboard.py` globs `results/*.json` **non-recursively**, so canary rows cannot enter the gated dashboard's history or trends (pinned by `test_load_results_ignores_the_canary_subdirectory`) |
| distinct container name (`aorta-rocm-canary`) | cannot tear down the gate's container on the shared MI350 runner |

Each row carries `lane` (`"gate"` / `"canary"`) and `base_image` (the resolved
digest, `null` in the gated lane where the Dockerfile pin already records it).
Cron is 15:00 UTC, clear of gpu-tests (08:00), the nightly eval (after the 11:00
wheels) and the sanitizer nightly (12:00); the lane is best-effort, so being
squeezed out is acceptable where starving the gate is not.

Publishing is a **separate `ubuntu-latest` job**, the same split
`nightly-eval.yml` uses and for the same two reasons: the GPU job installs and
executes a wheel, so it holds `contents: read` only and checks out with
`persist-credentials: false`; and because the workflow is
`workflow_dispatch`-triggerable, the publish job is gated to
`github.ref == 'refs/heads/main'` so a run from an unreviewed branch cannot write
to the shared `ci-results` branch (it still produces the artifact).

**Not done yet:** rendering these rows on the dashboard as observed-only is the
remaining half of #382. Until it lands the data accumulates on the `ci-results`
branch and each run's numbers are in its artifact and step summary.

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
  `src/aorta/workloads/**`, `src/aorta/instrumentation/rocjitsu_sanitizers/**`,
  `src/aorta/utils/gpu_control.py`
- matching tests (`tests/hw_queue_eval/**`,
  `tests/instrumentation/rocjitsu_sanitizers/**`, `tests/sanitizers/**`, ...),
  `config/ci/**`, `recipes/ci/**`, `recipes/sanitizers/**`, `scripts/ci/**`,
  `scripts/sanitizers/**`, `docker/**`, `.github/workflows/gpu-tests.yml`

The RocJITsu sanitizer engine (`rocjitsu_sanitizers`) is gfx-hardware code, so
its sources, recipes, dashboard scripts, and tests are GPU-relevant. The GPU
gate's `test_sanitizers_gpu.py` always runs a fail-closed guardrail on the
runner; its real clean/racy ConSan repro cases run when the DBI hook
(provisioned via `ROCJITSU_PREBUILT` or `ROCJITSU_BUILD`) and `hipcc` are
present (as in `sanitizers-nightly.yml`) and self-skip otherwise.

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

**Exit codes.** New `aorta sweep run` smokes must pass `--strict`, so the sweep
exits non-zero when a cell errors or never runs; the matrix flow otherwise
tolerates per-cell failures and exits 0, leaving this gate green on a broken
workload. `aorta run` needs no flag -- it already exits non-zero on a failed
trial. Note that `--strict` does not trip on a cell that ran but reported
`passed=False` (an expected A/B "bug reproduced" outcome); every smoke here is
expected to pass cleanly, so an errored workload is what this guards against.

Broader per-workload coverage (dtype and GPU-count axes, metric regressions
against blessed baselines) lives in the nightly eval matrix instead -- see
[`ci-nightly-eval.md`](ci-nightly-eval.md). Keep this manifest focused on fast
crash-level protection, with the PR tier the part that gates merges.

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

Not an issue: making the **`CPU tests`** aggregator and **`pytest (GPU, MI350)`**
jobs **required status checks** on `main` is a repo/admin setting (see
"Making it a required check" / "Making the GPU gate a required check" above),
not a code change. (For CPU, require the `CPU tests` aggregator, not the
per-version `pytest (CPU, py3.x)` legs -- see "Making it a required check".)

## Summary

| Phase | Runner | Selection | Status |
| --- | --- | --- | --- |
| 1 - CPU gate | `ubuntu-latest` (3.10-3.14) | `not gpu and not rocm`, `-n auto` | Implemented (`cpu-tests.yml`) |
| 2 - GPU gate | `[self-hosted, gpu]` | `gpu or rocm`, `-n 4` + nightly workload regression | Implemented (`gpu-tests.yml`) |
