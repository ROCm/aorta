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

   Each nightly's per-case results are co-located on the `sanitizer-results` data
   branch under `dashboard/runs/<YYYY-MM-DD>T<HHMMSS>-<run_id>/` (one directory per
   guardrail recipe, one under `survey/` per observed-only case, plus a
   `meta.json` with commit / date / gpu / run_url / gate, where `date` is an
   ISO-8601 instant -- runs published before the directory carried a time keep
   their date-only names and manifests, and are never renamed), and the rendered page
   links to them: the latest run table has a per-recipe **Report** link, each
   kernel-detail card carries a **run area** link to the case's directory, and
   each **Run** row in the history table links to its `runs/<id>/` area. A
   `runs/<id>/index.html` landing page lists that run's guardrail reports and its
   survey cases. Because `pages.yml` copies `dashboard/*` recursively into
   `_site/sanitizers/`, everything under `runs/` is served at
   `/sanitizers/runs/...` with relative links and no change to `pages.yml`. The
   publish job keeps a rolling window of the newest **30** run directories
   (`gen_sanitizer_dashboard.py --history-root <dir> --keep 30`); older ones are
   pruned. Previously these raw reports lived only in an expiring Actions
   artifact and were never linked; the rolling window makes them durable and
   reachable from the page.

   **Run areas (#384).** A case directory is not just its report -- it carries
   the recipe as it ran, the logs the verdict came from, the source-level inputs,
   and -- for the artifacts it deliberately does not publish -- a rebuild command
   wherever the module's tables know one (see below), so the copy-paste command
   the dashboard shows is actionable:

   | File | What it is |
   | --- | --- |
   | `index.html` | Landing page: the command, the env a reproduction needs, run identity, observed verdict, the recorded digests (including each `code_object_sha256`), the fixture rebuild steps, and every file below as a download. GitHub Pages does not auto-index a directory, so this is what makes the run-area link resolve. |
   | `sanitizer_report.json` | The full `aorta.sanitizer_report/0.1` document the dashboard renders from. |
   | `consan/consan.log.gz`, `waitcheck/waitcheck-*.log.gz` | The sanitizer output the verdict was derived from, gzipped. |
   | `recipe.yaml` | The recipe exactly as it ran, pinned to this run. |
   | `inputs/` | The recipe's source-level fixture inputs (the `.hip` repro sources, the GEMM shape CSV) -- a few KB in total. |
   | `REPRODUCE.md` | Commit, command, required env, fixture rebuild steps, and the digests of the artifacts that are not published. |
   | `env.json` | The same provenance, machine-readable (`aorta.sanitizer_run_area/0.1`), including `required_env` (each variable, who sets it, and why -- ConSan's four are listed only for a case that ran ConSan) and `rebuild` (the per-artifact commands, runnable from the repo root). `REPRODUCE.md` and the landing page are rendered *from* those two stored fields, so the prose cannot disagree with them and a retained area keeps describing what it actually recorded. |

   CI-built artifacts are deliberately **not** published: a GEMM `.hsaco` is
   ~16MB, and shipping one per retained run would bloat the data branch and
   Pages. `index.html` / `REPRODUCE.md` / `env.json` instead record each one's
   path, the SHA-256 the report carries for it (the waitcheck binary digest, the
   ConSan repro command and hook digests, every kernel's `code_object_sha256`)
   and the command that rebuilds it, so a local rebuild can be verified. Neither
   is universal: a digest is only recorded when the report has one under that
   artifact's basename, so a bare `isa_dir: fixtures/isa` reference has none, and
   a reference the module's rebuild tables do not recognise is named without a
   command rather than given a plausible-looking guess.

   Since that means the file list is *not* the whole recipe -- for a
   `source.kind: kernel` recipe the one input the recipe names is exactly the
   excluded one -- the landing page's Files caption says how many inputs are held
   back and links to the *Artifacts not published* table that lists them, naming
   their SHA-256s only when every entry has one. The rebuild commands are their
   own section higher up the page, so the caption does not promise them under
   that anchor. Either way a reader can tell a deliberate omission from a lost
   file.

   Those rebuild commands are per-artifact, because they are not
   interchangeable: a `--genco` code object is a raw ELF on some ROCm builds and
   a clang-offload bundle on others, so it must be unbundled conditionally (the
   recorded digest is of the unbundled object the loader opens); the GEMM objects
   are *extracted* from the shipped Tensile libraries by `prepare_gemm_isa.py`
   rather than compiled; and each `consan_load` / `lds_dispatch` binary needs the
   `-DOBJECT` / `-DLDS_HSACO` define naming the object it loads. A generic
   "`hipcc` it" hint would build a different file and fail the digest check.

   The commands are also written to be pasted as-is from the repo root, which is
   where the `git clone` in `REPRODUCE.md` leaves you: fixture paths are rooted at
   `recipes/sanitizers/fixtures/...` (there is no top-level `fixtures/`), each one
   starts by putting the ROCm LLVM bindir on `PATH` (the container exports only
   `/opt/rocm/bin`, while `clang-offload-bundler` -- which `hipcc` and
   `prepare_gemm_isa.py` both need -- lives beside the compilers) and creating the
   gitignored output directory, and the conditional unbundle is a single
   `if ... else cp ... fi` command rather than a prose aside, so an automated
   consumer of `rebuild` can execute the list without interpreting it.

   Logs, the recipe copy and its inputs are kept only for the newest **7** runs
   (`--keep-logs 7`) -- guardrail and survey areas alike -- while reports stay for
   the full 30. An older run keeps its report, manifests and landing page, and
   both are re-rendered when it is pruned so the page lists only the files still
   present and `env.json` stops naming the inputs it no longer carries.

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

- **Dependabot** (`.github/dependabot.yml`) opens weekly PRs for pip / docker
  (ROCm base image digest) / github-actions updates.
- After a stack-moving bump, run **Refresh baselines** (numerics/perf move with
  ROCm/hipBLASLt — the baseline diff is the bump's impact) and **Lock
  requirements** to refresh `config/ci/ci-constraints.txt` (partial pip
  constraints, not a hash-pinned lock). Both open PRs; a human blesses.
  No auto-merge.

## Results retention

The publish step keeps only the **most recent 180** `results/<date>.json` files on
the `ci-results` data branch (older ones are pruned), and the dashboard renders at
most the last 180 builds (`gen_dashboard.py --max-builds`). Files are tiny; adjust
the cap in `nightly-eval.yml` / the flag if a longer window is wanted.

The **sanitizer** nightly keeps its own rolling window on the `sanitizer-results`
data branch, and it has **two** bounds rather than one:

- **Reports: newest 30 runs.** `dashboard/runs/<id>/` holds one directory per
  guardrail recipe, one under `survey/` per observed-only case, and a `meta.json`.
  Each case directory is a *run area* (report, gzipped logs, `recipe.yaml`,
  `inputs/`, `REPRODUCE.md`, `env.json`, `index.html`) — see the run-area table
  earlier in this document. The publish job prunes older runs and re-renders with
  `gen_sanitizer_dashboard.py --history-root dashboard/runs --keep 30`.
- **Bulk: newest 7 runs.** `--keep-logs 7` bounds the logs, the recipe copy and
  the copied inputs, for guardrail and survey areas alike. An older run keeps its
  report, manifests and landing page; its bulk is pruned and the area is
  re-rendered so the page lists only what is still there. Pruning is one-way —
  raising `--keep-logs` later does not bring deleted logs back.

Adjust `keep` in `sanitizers-nightly.yml` (and the matching `--keep`) for the
report window, and `--keep-logs` for the bulk window.

Both windows bound the **checkout**, not the branch history. Pruning deletes a
file from the working tree, but the blob stays reachable from the commit that
added it, so every log ever published remains in `.git` even after its area is
pruned — `sanitizer-results` accumulates ordinary commits indefinitely. Day to
day that costs nothing (the publish job clones `--depth 1` and Pages deploys from
the workspace copy), so the exposure is remote repository size. It does change the
deadline for one decision, though: if real ConSan logs turn out large, capping
their size is a code change *before* the first publish and a history rewrite
afterwards. Measure them on the first nightly (`du -sh` the staged case dirs in
the publish job) rather than waiting for the branch to grow.

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
