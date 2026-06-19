# Triage recipes

> **Renamed (issue #248):** `aorta triage` and `aorta probe` are now the
> single command **`aorta sweep`**. Use `aorta sweep run` everywhere this
> guide says `aorta triage run`, and `aorta sweep run ... -- <cmd>` for the
> subprocess (former `aorta probe`) flow. The old commands still work as
> deprecated aliases that print a notice and delegate to the same engine.

A triage **recipe** is the authoritative description of a `aorta sweep run
--mode matrix` invocation: which `(mitigation x environment)` cells to run,
per-cell trial / step counts, the ticket the matrix belongs to, and the
speed-confound detection config.

Recipes are the primary interface. The `--mode matrix` flag shim is kept as
an escape hatch for ad-hoc one-shots; internally it constructs an in-memory
`Recipe` and reuses the same execution path.

## Quick reference

```yaml
schema_version: 1                    # required; loader rejects unknown versions
ticket: EXAMPLE-001                  # optional; drives output dir grouping
workload: fsdp                       # required; resolved via aorta.workloads entry-point group
trials: 8                            # required; per-cell trial count
steps: 5000                          # required; per-cell step count
save_logs: false                     # optional; when true, dispatcher writes per-trial stdout/stderr files

confound:
  threshold: 1.15                    # default; > 1.15 -> "speed (+N%)" flag
  baseline_cell: baseline-local      # optional; defaults to the first "baseline-*" cell
                                     # or the first cell with mitigations: [none]

cells:
  - name: baseline-local
    mitigations: [none]
    environment: local

  - name: tf32_off-local
    mitigations: [tf32_off]
    environment: local

  - name: stack-tf32-xnack-local     # mitigation stacking (env vars unioned in list order)
    mitigations: [tf32_off, xnack]
    environment: local
    trials: 16                       # optional per-cell override
    steps: 8000                      # optional per-cell override

  - name: try-nightly                # inline docker shorthand
    mitigations: [none]
    environment: { docker: "rocm/pytorch:nightly" }

  - name: custom-env-override        # one-off env var override for this cell only
    mitigations: [tf32_off]
    environment: local
    extra_env:
      MY_DEBUG_FLAG: "1"
```

## Schema rules (full detail)

- **`schema_version`** -- required `int`; currently `1`. Unknown values raise
  `RecipeSchemaError`.
- **`ticket`** -- optional string; format-free. Absent tickets route output
  to `triage_results/_no_ticket_/...`.
- **`workload`** -- required string; must resolve via `aorta.workloads`
  entry-point group at runtime. Unknown names surface as cell-level errors,
  not load-time errors, because workload discovery is B1's job.
- **`trials` / `steps`** -- required ints at top level; per-cell overrides
  allowed.
- **`confound.threshold`** -- optional float, default `1.15`.
- **`confound.baseline_cell`** -- optional string. Resolution order if
  absent: (1) first cell named `baseline-*`; (2) first cell with
  `mitigations == ["none"]`; (3) single-cell recipes default to that cell;
  (4) error.
- **`cells[*].name`** -- required string, unique within the recipe. Used as
  the `matrix.md` row label and the `cells/<name>/` directory name. Must
  match `^[A-Za-z0-9_][A-Za-z0-9_.\-]*$` (no path separators, no leading
  `-`, no `.` / `..`); reserved names like `matrix.md` / `matrix.json` are
  also rejected so a cell can't clobber a sibling artifact.
- **`cells[*].mitigations`** -- required `list[str]`. Each name resolved
  through `aorta.registry.get_mitigation()`. Empty list rejected (use
  `["none"]` for the explicit baseline). Multiple names union their
  env-var bundles in list order. **Stacked mitigations must agree on
  overlapping keys**: if two bundles set the same env var to different
  values the recipe is rejected at load time. Use `extra_env` to
  intentionally override.
- **`cells[*].environment`** -- required. Either:
  - a registered environment name (resolved via `aorta.registry.get_environment()`), OR
  - a mapping `{ docker: "<image-ref>" }` -- inline docker shorthand.
    Auto-named `_inline_<hash>` where `<hash>` is the first 8 hex chars of
    `blake2b(image-ref)`. Deterministic: two cells with the same ref share
    the same auto-name and the same per-environment env-probe. No other
    keys accepted.
- **`cells[*].extra_env`** -- optional `dict[str, str]`. Applied AFTER the
  mitigation bundle, so it can override a registered mitigation's env var
  for one-off experiments without polluting the registry. Recorded in
  `matrix.json` for audit.
- **`save_logs`** -- optional `bool`, default `false`. When `true`, the
  dispatcher captures the workload's in-process `stdout` / `stderr`
  writes into `trial_d{d}_m{m}_t{t}.{stdout,stderr}.log` files alongside
  the trial JSON. Both the file capture and the reserved-key injection
  described below are **rank-0 only** (matches the trial-JSON write
  guarantee); wrappers running on non-rank-0 won't see the keys and
  should treat capture as off there.
  `contextlib.redirect_*` only catches Python-level writes; workloads
  that spawn subprocesses (e.g. `recom_repro` invoking `docker run`)
  don't have their subprocess output captured automatically. Those
  wrappers can opt in by reading the platform-supplied
  `_aorta_save_logs` / `_aorta_log_prefix` config keys the dispatcher
  injects, and writing their own capture to a sibling path derived from
  the prefix (e.g. `<prefix>.subprocess.stdout.log`). The prefix is an
  absolute path-with-stem rooted in the per-workload results
  subdirectory (e.g. `<results_dir>/<workload>/trial_d0_m0_t0`,
  anchored via `Path.absolute()` so a relative `results_dir` still
  yields a usable prefix), so wrappers don't need to know
  `results_dir`. The dispatcher already holds the
  `<prefix>.{stdout,stderr}.log` paths open, so wrappers must NOT
  write to them directly.
- **`workload_config`** -- optional `dict[str, Any]`, allowed at both
  recipe scope (top level) and per cell. Forwarded to the workload
  constructor through the dispatcher's `Request.config_overrides`. Use
  this for workload-specific knobs that aren't env vars -- e.g.
  `shampoo_api: old` on the `recom_repro` workload to select the V1
  flat-kwarg SHAMPOO entry script (both `new` and `old` import from the
  OSS `distributed_shampoo` package; they differ in constructor shape).
  Cell-scope merges over recipe-scope on a per-key basis (cell wins on
  collision; non-collision keys union),
  so a recipe can set a workload-wide default and opt one cell out.
  Reserved keys: `"steps"` (first-class field; would be silently
  overwritten by the dispatcher) and any `_aorta_*` prefix
  (platform-supplied) are rejected at load time. Example:

  ```yaml
  workload: recom_repro
  workload_config:
    shampoo_api: new          # recipe default
  cells:
    - name: v3-baseline
      mitigations: [none]
      environment: nan-repro-v3
    - name: v2-baseline
      mitigations: [none]
      environment: nan-repro-v2-image
      workload_config:
        shampoo_api: old      # cell override
  ```

Every validation error reports a path like `cells[2].mitigations` so the
failure is localisable without reading the loader source.

## Workloads

The `workload:` name is resolved through the `aorta.workloads` entry-point
group at **cell-execution time, not recipe-load time**. A recipe naming an
unregistered workload loads and validates fine; it fails per-cell when the
cell runs. So "the recipe is valid" does not imply "the workload exists" --
confirm registration with a dry-run (or `pip install -e .` after adding a
new entry-point).

### `race`

Wraps the in-tree RCCL race reproducer (`aorta.race`). `launch_mode:
distributed`, `min_world_size: 2`. The communication pattern is selected by
the `mode` config key, not by a separate workload name:

```yaml
workload: race
workload_config:
  mode: fsdp            # default | ddp | fsdp
  warmup_iterations: 0
  verify_iterations: 50
  dtype: bfloat16       # bfloat16 | float16 | float32
```

`workload_config` accepts any field of
`aorta.race.config.ReproducerConfig` (e.g. `h2d_tensor_size`,
`fsdp_shard_size`, `gemm_layers`, `simulate_compute`, `h2d_prefetch`,
`same_stream_mode`, `stop_on_first_corruption`, `log_interval`).

Because `race` is `launch_mode: distributed`, it MUST be launched under
torchrun (a bare `aorta sweep run` starts one process, WORLD_SIZE=1, and is
refused by launch-mode validation). Use the `aorta` console script as
torchrun's target -- `-m aorta` is not a runnable module:

```bash
# validate only (no GPUs / no launcher):
aorta sweep run --recipe recipes/example-fsdp-smoke.yaml --dry-run

# single node, 2 ranks (bump --nproc_per_node to your GPU count):
torchrun --standalone --nproc_per_node=2 $(which aorta) sweep run \
  --recipe recipes/example-fsdp-smoke.yaml

# multi-node (1 rank/host x N hosts -- the AINIC topology):
torchrun --nnodes=N --nproc_per_node=1 --rdzv-backend=c10d \
  --rdzv-endpoint=$MASTER_ADDR:29500 $(which aorta) sweep run \
  --recipe recipes/example-fsdp-smoke.yaml
```

Each rank runs the full `sweep run`; the ranks find each other in
`dist.init_process_group()`. Cells run in-process per rank (sequentially),
and results are written by **rank 0 only** (the dispatcher gates writes on
`RANK`), so multi-rank launches don't clobber each other. This is the same
launch model as the `llm_determinism` workload.

> [!IMPORTANT]
> Unknown `workload_config` keys are **dropped with a `WARNING`**, not
> errors. A key that isn't a `ReproducerConfig` field is silently ignored
> at runtime -- which for a stress recipe means a lever you thought you set
> never applied (a false green). Watch the run log for
> `race: ignoring unknown workload_config key ...` and fix the recipe.
> Note `verify_iterations` defaults to `10000` and `simulate_compute` to
> `True`; cap these for smoke runs or a trial takes hours.

## Output layout

```
<output-dir>/
  <ticket or _no_ticket_>/
    <workload>/
      <timestamp>[-N]/                          # e.g. 2026-04-28T14-12-03; -2, -3 ... on same-second collisions
        matrix.md
        matrix.json
        recipe.resolved.yaml                    # post-resolution snapshot
        host_env.json                           # collect_env() once per run
        environments/<env-name>/env.json        # once per unique environment
        inline_environments.sidecar.json        # only when inline docker is used
        sidecars/<basename>                     # one copy per --mitigations-file
        cells/<cell-name>/<workload>/trial_*.json
```

`matrix.json` per-cell shape (the canonical machine-readable record):

- `failure_rate` -- fraction of trials with `exit_status != ok` OR
  `WorkloadResult.passed == False`. NOT a NaN-specific rate; see
  `exit_status_counts` to disambiguate failure modes.
- `exit_status_counts` -- histogram keyed by `TrialResult.exit_status`
  (`"ok"`, `"workload_failed"`, `"infrastructure_failed"`, ...). Total
  equals the cell's trial count.
- `min_step_time_ms`, `max_step_time_ms`, `p50_step_time_ms`,
  `p90_step_time_ms`, `p99_step_time_ms` -- summary stats over the
  concatenated per-trial step-time series.
- `mean_step_time_ms`, `std_step_time_ms`, `mean_wall_clock_sec` --
  unchanged; still the headline timing fields.
- `step_times_ms` -- raw concatenated series for downstream re-analysis.
- `step_time_source` -- which branch of the fallback ladder produced the
  cell's step-times: `"per_step"` (workload's own `step_times_ms`),
  `"elapsed_per_iter"` (`elapsed_sec / total_iterations`),
  `"wall_clock_total"` (`wall_clock_sec / steps`, folds in setup /
  teardown), or `"missing"` (no usable timing). Confound classification
  refuses to compute a ratio between cells whose sources differ -- those
  rows are marked `n/a` in `matrix.md` -- so a workload that only exposes
  wall-clock can't be silently compared against one that emits per-step
  timing.
- `resolved_env_vars` -- the env-var bundle as actually applied (mitigation
  union + `extra_env`).
- `resolved_environment` -- the resolved `Environment` descriptor.
- `workload_config` -- the merged per-cell `workload_config` dict (recipe
  scope union cell scope, cell wins on collision). Empty `{}` when neither
  scope sets it. `matrix.md` surfaces keys whose value varies across cells
  in a `Config` column (diffs only; the column is hidden when no cell has
  workload_config or when every cell agrees on every key).
- `trial_paths` -- per-trial JSON paths, sorted by trial index (NOT
  lexicographically, so `trial_2.json` precedes `trial_10.json`).

**Note on the trailing `<workload>/` directory inside each cell.** B1's
runner (`aorta.run.run_trials`) appends `/<workload>` to the output
directory it was given. B2 honours that contract: each cell is told to
write to `cells/<cell-name>/`, and B1 ends up writing
`cells/<cell-name>/<workload>/trial_N.json`. `matrix.json` records the
real paths; a future B1 follow-up can drop this level of nesting via a
`skip_workload_subdir` kwarg on `RunRequest`.

## Re-running a past matrix

Every run writes `recipe.resolved.yaml` alongside the matrix. The file is
**a strict, schema-valid recipe** -- you can pass it back to
`aorta sweep run --recipe ...` directly. Inline-docker cells are
re-emitted in the `{ docker: <ref> }` shorthand so the same
`_inline_<hash>` is re-derived without needing to ship a sidecar JSON
next to the file.

For runs that used `--mitigations-file`, the resolved YAML still
references those mitigation / environment names by name. The runner
snapshots each operator-supplied sidecar into `<run_dir>/sidecars/<basename>`
so the run directory is self-contained for replay. The runner also prints
the exact rerun command on stdout when sidecars are involved, e.g.:

```
cd <run_dir> && aorta sweep run --recipe recipe.resolved.yaml \
  --mitigations-file sidecars/foo.json
```

The per-cell mitigation env-var bundles AS APPLIED, plus the resolved
`Environment` descriptor for each cell, live in `matrix.json` (under
each cell's `resolved_env_vars` and `resolved_environment` keys) -- not
in `recipe.resolved.yaml` -- so the rerun artifact stays loadable while
audit data is still preserved next to the run.

> [!NOTE]
> "Reproducing the same matrix" is up to the registries available at
> rerun time. If a sidecar mitigation's env-var bundle drifts between
> runs the rerun will use the new bundle (snapshotting the *file* doesn't
> pin its *contents* once the operator edits it later). Inline-docker
> cells are immune (the docker ref is in the recipe text itself), but
> registry drift on named entries is currently not pinned. Compare each
> run's `matrix.json::cells[*].resolved_env_vars` to detect drift.

## Flag mode (escape hatch)

The equivalent of `recipes/example-fsdp-smoke.yaml` as flag-mode CLI:

```
aorta sweep run --mode matrix \
  --workload fsdp \
  --mitigation-axis none,tf32_off,xnack \
  --environment-axis local \
  --trials 2 --steps 100 \
  --ticket EXAMPLE-151
```

Inline docker still works in flag mode via the `image:` prefix on the
axis, e.g. `--environment-axis local,image:rocm/pytorch:nightly`. Each
comma-separated item is parsed independently; bare names go through the
registry, `image:<ref>` maps to the same `{ docker: <ref> }` shorthand as
recipe mode.

## Probe handout templates (issue #188 Phase 3)

Generic ``mode: probe`` recipes for customer handouts. Each template
includes a ``redaction:`` block consumed by ``aorta bundle`` when
packaging artifacts for sharing.

| Template | Typical trailing argv |
|---|---|
| `probe-template-torchrun.yaml` | `torchrun --nproc_per_node=N train.py ...` |
| `probe-template-buck2.yaml` | `buck2 run //path:target -- ...` |
| `probe-template-bash.yaml` | `bash launch.sh ...` |

Dry-run smoke:

```
aorta sweep run --recipe recipes/probe-template-bash.yaml --dry-run -- echo hi
```

See `docs/probe-188/handout-templates.md` for per-template walkthroughs.
