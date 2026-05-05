# Triage recipes

A triage **recipe** is the authoritative description of a `aorta triage run
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

  - name: stack-tf32+xnack-local     # mitigation stacking (env vars unioned in list order)
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
  the `matrix.md` row label and the `cells/<name>/` directory name.
- **`cells[*].mitigations`** -- required `list[str]`. Each name resolved
  through `aorta.registry.get_mitigation()`. Empty list rejected (use
  `["none"]` for the explicit baseline). Multiple names union their
  env-var bundles in list order.
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

Every validation error reports a path like `cells[2].mitigations` so the
failure is localisable without reading the loader source.

## Output layout

```
<output-dir>/
  <ticket or _no_ticket_>/
    <workload>/
      <timestamp>/                              # e.g. 2026-04-28T14-12-03
        matrix.md
        matrix.json
        recipe.resolved.yaml                    # post-resolution snapshot
        host_env.json                           # collect_env() once per run
        environments/<env-name>/env.json        # once per unique environment
        inline_environments.sidecar.json        # only when inline docker is used
        cells/<cell-name>/<workload>/trial_*.json
```

**Note on the trailing `<workload>/` directory inside each cell.** B1's
runner (`aorta.run.run_trials`) appends `/<workload>` to the output
directory it was given. B2 honours that contract: each cell is told to
write to `cells/<cell-name>/`, and B1 ends up writing
`cells/<cell-name>/<workload>/trial_N.json`. `matrix.json` records the
real paths; a future B1 follow-up can drop this level of nesting via a
`skip_workload_subdir` kwarg on `RunRequest`.

## Re-running a past matrix

Every run writes `recipe.resolved.yaml` alongside the matrix. Registry
names are expanded to env-var bundles and docker refs in that file, so
re-running it on a different machine reproduces the same matrix even if
the registries drift in the meantime.

## Flag mode (escape hatch)

The equivalent of `recipes/example-fsdp-smoke.yaml` as flag-mode CLI:

```
aorta triage run --mode matrix \
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
