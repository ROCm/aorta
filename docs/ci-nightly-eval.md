# Nightly evaluation + dashboard (implementation)

Implements Phases 1-5 of [`ci-plan.md`](ci-plan.md). This is the "how it works /
how to operate it" reference for the nightly evaluation, dashboard, alerting,
baselines, and automated bumps.

## Components

| Piece | File |
| --- | --- |
| Eval matrix (workload x config) | `config/ci/nightly_eval_matrix.yaml` |
| Expected-outcome baselines | `config/ci/regression_baselines.yaml` |
| CI dependency lock (exact pins) | `config/ci/requirements.lock` (generated) |
| Harvester + comparator (pure) | `scripts/ci/eval_lib.py` |
| Nightly harness | `scripts/ci/nightly_eval.py` |
| Baseline refresher | `scripts/ci/refresh_baselines.py` |
| Requirements locker | `scripts/ci/lock_requirements.sh` |
| Dashboard generator | `scripts/ci/gen_dashboard.py` |
| Regression alerter | `scripts/ci/alert_issue.py` |
| Nightly workflow | `.github/workflows/nightly-eval.yml` |
| Baseline refresh workflow | `.github/workflows/refresh-baselines.yml` |
| Lock refresh workflow | `.github/workflows/lock-requirements.yml` |
| Automated bumps | `.github/dependabot.yml` |

## Flow (nightly-eval.yml)

Triggered by `workflow_run` on **"Nightly wheels"** success (+ `workflow_dispatch`):

1. Build/start the pinned ROCm container (`rocm-ci-setup`).
2. Install the **released nightly wheel** `amd-aorta[hw-queue]` (constrained by
   `config/ci/requirements.lock` when present).
3. `nightly_eval.py` runs each matrix entry via `aorta sweep run --strict`,
   harvests `matrix.json`, and compares each cell to the baselines:
   - **record** — no baseline yet (metrics captured, treated as pass),
   - **pass / fail** — vs a blessed baseline,
   - **skip** — insufficient GPUs.
   The job fails only on a **fail** (blessed-baseline breach). Results go to
   `gpu-nightly-results.json` with build/ROCm metadata.
4. **Alert** (`alert_issue.py`): opens/updates one `nightly-regression` issue on
   failure; comments + closes it when green.
5. **Publish** (`gen_dashboard.py`): appends `results/<date>.json` to the
   `gh-pages` branch and regenerates the self-contained dashboard (latest status
   table + inline-SVG step-time trends + pass-rate trend). Served via GitHub
   Pages (source = `gh-pages` branch; enable Pages in Settings).

## Correctness vs performance

- **Correctness** is gated: a cell that should pass but errored/failed => `fail`.
- **Performance** is **trend-only by default**: baselines are correctness-only
  (`passed: true`); step-time/throughput are captured + charted but not gated.
  To turn on perf gating (Phase 5), regenerate baselines with
  `refresh_baselines.py --perf-gate` (adds `step_time_ms.max` / `throughput.min`
  bounds the comparator then enforces).

## Baselines

Baselines are ROCm/stack-specific. Generate them on the runner and bless via PR:

```
Actions -> Refresh baselines -> Run workflow
```

`refresh_baselines.py` runs the matrix, captures each passing cell, and opens a
PR updating `regression_baselines.yaml`. Review the diff and merge to bless. An
empty baseline file means **record-only** (nightly won't be red before blessing).

## Automated ROCm + dependency bumps

- **Dependabot** (`.github/dependabot.yml`) opens weekly PRs for pip / docker
  (ROCm base image digest) / github-actions updates.
- After a stack-moving bump, run **Refresh baselines** (numerics/perf move with
  ROCm/hipBLASLt — the baseline diff is the bump's impact) and **Lock
  requirements** to refresh `requirements.lock`. Both open PRs; a human blesses.
  No auto-merge.

## Results retention

The publish step keeps only the **most recent 180** `results/<date>.json` files on
the `gh-pages` branch (older ones are pruned), and the dashboard renders at most
the last 180 builds (`gen_dashboard.py --max-builds`). Files are tiny; adjust the
cap in `nightly-eval.yml` / the flag if a longer window is wanted.

## Operating checklist

1. Enable GitHub Pages (source: `gh-pages` branch) so the dashboard is served.
2. First nightly runs **record-only**; then run **Refresh baselines** to bless.
3. (Optional) Run **Lock requirements** to pin the CI dependency set.
4. (Later) Enable perf gating via `refresh_baselines.py --perf-gate`.
