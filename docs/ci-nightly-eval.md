# Nightly evaluation + dashboard (implementation)

Implements the CI plan (proposed in
[ROCm/aorta#300](https://github.com/ROCm/aorta/pull/300); `docs/ci-plan.md` lands
with that PR). This is the "how it works / how to operate it" reference for the
nightly evaluation, dashboard, alerting, baselines, and automated bumps.

## Components

| Piece | File |
| --- | --- |
| Eval matrix (workload x config) | `config/ci/nightly_eval_matrix.yaml` |
| Expected-outcome baselines | `config/ci/regression_baselines.yaml` |
| CI dependency lock (exact pins) | `config/ci/ci-constraints.txt` (generated) |
| Harvester + comparator (pure) | `scripts/ci/eval_lib.py` |
| Nightly harness | `scripts/ci/nightly_eval.py` |
| Baseline refresher | `scripts/ci/refresh_baselines.py` |
| Requirements locker | `scripts/ci/lock_requirements.sh` |
| Dashboard generator | `scripts/ci/gen_dashboard.py` |
| Regression alerter | `scripts/ci/alert_issue.py` |
| Nightly workflow | `.github/workflows/nightly-eval.yml` |
| Pages build + deploy (landing + dashboard) | `.github/workflows/pages.yml` |
| Baseline refresh workflow | `.github/workflows/refresh-baselines.yml` |
| Lock refresh workflow | `.github/workflows/lock-requirements.yml` |
| Automated bumps | `.github/dependabot.yml` |

## Flow (nightly-eval.yml)

Triggered by `workflow_run` on **"Nightly wheels"** success (+ `workflow_dispatch`):

1. Build/start the pinned ROCm container (`rocm-ci-setup`).
2. Install the **released nightly wheel** `amd-aorta[hw-queue]` (constrained by
   `config/ci/ci-constraints.txt` when present).
3. `nightly_eval.py` runs each matrix entry via `aorta sweep run --strict`,
   harvests `matrix.json`, and compares each cell to the baselines:
   - **record** — no baseline yet AND the cell passed (metrics captured),
   - **pass / fail** — vs a blessed baseline (honoring its expected `passed`),
   - **skip** — insufficient GPUs.
   **Fail-closed:** the job fails on any `fail` — a failed/errored cell (even
   with no baseline), a missing/empty `matrix.json`, a per-entry **timeout**, a
   blessed-baseline breach, or a run that did **zero work** (all skipped). An
   empty baseline file therefore protects against crashes/failures immediately;
   it just doesn't add perf/metric gates until blessed. Results go to
   `gpu-nightly-results.json` with build/ROCm metadata.
4. **Alert** (`alert_issue.py`): opens/updates one `nightly-regression` issue on
   failure; comments + closes it when green.
5. **Publish** (`publish` job on `ubuntu-latest`): appends
   `results/<date>.json` to the **`ci-results`** data branch (history only).
6. **Deploy** (`pages.yml`): a repo has a single Pages site, shared with the
   project docs, so one workflow owns the deploy. On main pushes, after each
   Nightly Evaluation completes, and on demand, `pages.yml` builds the Jekyll
   site into `_site/`, relocates the rendered README from `_site/index.html` to
   `_site/docs/index.html`, writes the self-contained dashboard
   (`gen_dashboard.py`, from the `ci-results` history) to `_site/index.html`, and
   deploys the combined site via `actions/upload-pages-artifact` +
   `actions/deploy-pages`. **Repo Pages source must be "GitHub Actions"**
   (Settings -> Pages).    Nightly dashboard: `https://rocm.github.io/aorta/`;
   project docs: `https://rocm.github.io/aorta/docs/`.
   Sanitizer nightly dashboard: `https://rocm.github.io/aorta/sanitizers/` (linked
   from the root nav). The route is always published so it never 404s: before the
   first successful sanitizer nightly it shows a "no sanitizer runs yet" empty
   state, and if the latest nightly failed it shows the last data under a red
   stale banner linking the failed run (rather than silently re-serving the
   previous green page). Only sanitizer data produced by a `main` run is
   published.

   The dashboard previously lived at `/ci/`, so that path is kept: `/ci/`
   redirects to the root and `/ci/data.json` is published alongside
   `/data.json` for anything already polling it. A verification step fails the
   deploy if any of those routes would be missing, since a Pages deploy
   replaces the whole site and a dropped route 404s immediately.

## What the dashboard shows

Three views, ordered so the page reads from "what should I do?" down to detail:

- **What changed since \<date\>** diffs the two most recent runs, which is the
  question a nightly actually answers. It reports, in this order: the toolchain
  that moved (PyTorch / ROCm / HIP -- deliberately *not* AORTA's own version,
  which is date-stamped and so changes every night), verdicts that flipped with
  regressions first and their reasons attached, cells the matrix gained or lost,
  and metrics that moved by more than `_MOVE_PCT` (10%). A quiet night says so
  in one line rather than rendering an empty list.
- **Run history** is one row per run (newest first) and one column per workload.
  Each cell carries the number of results and the colour of the worst verdict
  among them, so a workload that broke and stayed broken reads as a column that
  turns red and stays red -- something no single-run view can show. Rows link to
  their workflow run and are marked when the toolchain moved; a workload absent
  from that night's matrix is a muted dot, not a blank.
- **Workloads** is tonight's detail: cells grouped under their workload, with
  metrics and recipe provenance nested in collapsible rows.

Both of the first two withhold themselves until there are two runs to compare;
with a single run they would imply a trend from one sample. The page needs no
JavaScript -- the only script expands and collapses the metric rows, and its
controls stay hidden unless it runs.

## Correctness vs performance

- **Correctness** is gated: a cell that should pass but errored/failed => `fail`.
  Default baselines also bless **correctness metrics** (exact-equality checksums
  such as `logits_checksum`), so a finite-but-wrong output is caught even without
  perf gating. Baselines honor the expected `passed` outcome (an expected-failure
  baseline is supported).
- **Performance** is **trend-only by default**: step-time/throughput/latency are
  captured + charted but not gated.
  To turn on perf gating (Phase 5), regenerate baselines with
  `refresh_baselines.py --perf-gate` (adds `step_time_ms.max` plus per-metric
  `policy`/`value` bounds -- min for throughput, max for latency/step-time, equal
  for checksums -- that the comparator then enforces; a required metric that is
  absent is a failure).

## Baselines

Baselines are ROCm/stack-specific. Generate them on the runner and bless via PR:

```
Actions -> Refresh baselines -> Run workflow
```

`refresh_baselines.py` runs the matrix, captures each passing cell, and opens a
PR updating `regression_baselines.yaml`. Review the diff and merge to bless. An
empty baseline file means **record-only** (nightly won't be red before blessing).

The refresh **fails atomically** if any entry *ran* but couldn't be blessed
(timeout / missing or empty `matrix.json` / a cell that didn't pass) — this
prevents silently reverting live gates to record-only. Entries the runner
**can't physically exercise** (e.g. `min_gpus: 8` variants on a smaller box) are
**not** fatal: their existing baselines are carried over unchanged, so you can
still refresh single-GPU baselines on a small runner and only re-bless the
multi-GPU entries on an 8-GPU box.

## Automated ROCm + dependency bumps

- **pip (weekly, auto-merged).** Dependabot (`.github/dependabot.yml`, `pip`
  ecosystem) runs **weekly** and opens **one grouped PR** (`python-deps`,
  version updates, `patterns: ["*"]`) that raises the `==` pins in the
  requirements files.   `.github/workflows/dependabot-auto-merge.yml` first
  installs `requirements.txt` + `requirements-dev.txt` across Python
  3.10/3.11/3.12 (no other gate exercises those files -- the CPU gate installs
  `pyproject.toml` extras, not the requirements files), then turns on GitHub
  **auto-merge (squash)**, so a bump lands **with no human review** once that
  install job and the repo's other required checks (the aggregate `CPU tests`
  check and `pre-commit`) are green. **`Requirements validation` must be a
  required status check on `main`** (branch protection): it is the stable
  aggregate over the `validate-requirements` matrix (the per-version legs can't
  be required — a skipped matrix never emits their contexts, actions/runner#952),
  and requiring it is what enforces the install gate on the *current head SHA*.
  A bare `needs:` edge only gates enabling auto-merge; since Dependabot rebases
  the open PR weekly, without the required aggregate a later revision could merge
  as soon as the other required checks pass while its re-validation is still
  pending. The permissions are per-job: only the `auto-merge` job holds
  `contents`/`pull-requests: write`, while `validate-requirements` checks out and
  installs the newly published packages with a read-only token and
  `persist-credentials: false`. Only **one bump PR is open at a time**
  (`open-pull-requests-limit: 1`): a red or still-open bump PR blocks the next
  one instead of piling up, and Dependabot rebases the open PR upward each week.
  - **Security updates are the exception.** Dependabot opens security-update PRs
    regardless of `open-pull-requests-limit` and outside the version-update
    group, so more than one pip PR can be open at once; those also auto-merge on
    green via the same workflow. Only version updates obey the one-PR/grouped
    contract.
  - **`bump-validate.yml` must be absent for a clean pip merge.** If that
    workflow is present, a pip PR that edits `requirements*.txt` also triggers
    its self-hosted GPU eval, which cannot run on a Dependabot PR (no
    `ROCM_SHARED_KEY`) and would block the merge. It is removed in #345, so
    ensure that has landed. (`gpu-tests.yml` itself skips to Success, since
    `requirements*.txt` is not one of its GPU-relevant paths.)
  - A bump PR **edits exactly**: `requirements.txt` (`pyyaml`, `matplotlib`,
    `numpy`, `pandas`, `openpyxl`, `seaborn`, `beautifulsoup4`, `click`) and
    `requirements-dev.txt` (`pytest`, `pytest-cov`, `pre-commit`,
    `pytest-timeout`, `pytest-xdist`, `pytest-forked`).
  - It **never** edits: `pyproject.toml` (kept as `>=` library floors on
    purpose — a library shouldn't hard-pin), `docker/**`, or
    `config/ci/ci-constraints.txt` (that file is refreshed by the separate
    **Lock requirements** dispatch, not by a bump PR). The
    `git+https://github.com/AMD-AGI/TraceLens.git` VCS line is not
    Dependabot-managed and is left alone.
- **docker base image (manual, NOT auto-bumped).** The ROCm/Ubuntu OS/base
  image is intentionally **not** tracked by Dependabot, so an OS change never
  lands unreviewed. Bump it by hand: resolve the new digest with
  `docker buildx imagetools inspect rocm/pytorch:<tag>`, update the
  `FROM rocm/pytorch:...@sha256:...` line in `docker/Dockerfile.ci-gpu`, then run
  **Refresh baselines** (numerics/perf move with ROCm/hipBLASLt — the baseline
  diff is the bump's impact).
- **github-actions (weekly, human-merged).** Action version bumps stay on the
  **weekly** schedule and are reviewed/merged by a human — no auto-merge.
- After any stack-moving bump, run **Lock requirements** to refresh
  `config/ci/ci-constraints.txt` (partial pip constraints, not a hash-pinned
  lock). It opens a PR that a human blesses.

## Results retention

The publish step keeps only the **most recent 180** `results/<date>.json` files on
the `ci-results` data branch (older ones are pruned), and the dashboard renders at
most the last 180 builds (`gen_dashboard.py --max-builds`). Files are tiny; adjust
the cap in `nightly-eval.yml` / the flag if a longer window is wanted.

## Operating checklist

1. Set GitHub Pages **source = "GitHub Actions"** (Settings -> Pages). This
   switches the site from the legacy branch build to `pages.yml`, which serves
   the dashboard at `/` **and** the project docs under `/docs/` from one deploy.
   Run the **Pages (landing + nightly dashboard)** workflow once to publish
   immediately (the docs are served even before any nightly results, and the
   root shows the dashboard's empty state).
2. First nightly runs **record-only**; then run **Refresh baselines** to bless.
   Until then the dashboard reports `recording` rather than `passing`, because
   nothing has been graded against a baseline yet.
3. (Optional) Run **Lock requirements** to pin the CI dependency set.
4. (Later) Enable perf gating via `refresh_baselines.py --perf-gate`.
