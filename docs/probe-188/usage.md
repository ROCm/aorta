# `aorta probe` — Usage Walkthrough (issue #188, Phases 1–3)

> **Phase 1 (Tier 1 only)**: verdict is `exit_code == 0 ? "pass" : "fail"`.
> **Phase 2**: five-tier classifier + `custom_patterns` + `--list-patterns`.
> **Phase 3 (this section)**: `redaction:` recipe block, `aorta bundle`,
> handout templates — see [redaction.md](redaction.md) and [bundle.md](bundle.md).
>
> See [classifier.md](classifier.md) for detector IDs and [sandbox.md](sandbox.md)
> for the `condition` whitelist.

`aorta probe` runs an **opaque user launch command** across the
cartesian product of a **mitigation axis × diagnostic axis**, in an
**idempotent / resumable** output tree. It is the bring-your-own-script
equivalent of `aorta triage run`: aorta does not parse the user's argv,
it just executes it once per `(mitigation, diagnostic)` cell × `trials`
trials and records the exit code.

## 1. Quick start

```bash
aorta probe \
    --recipe my_probe.yaml \
    --output ./probe_results \
    --ticket ROCM-1234 \
    -- \
    python3 my_repro.py --steps 100
```

Artifact tree (the documented Phase-1 layout):

```
probe_results/
  ROCM-1234/                      # safe_slug(ticket)
    recipe.resolved.yaml          # one snapshot of the recipe
    host_env.json                 # one host-env capture
    matrix.md / matrix.json
    none-none/                    # safe_slug(cell.name)
      trial_0/
        stdout.log
        stderr.log
        result.json               # see §6 for the actual field set
        probe.env                 # only with --env-passthrough-mode file
    tf32_off-none/
      trial_0/
        ...
```

Re-running the **same command** is a no-op for already-complete cells.
A *cell* is "complete" iff every `trial_<n>/result.json` parses and has
a non-empty `verdict` field; in that case the runner skips the cell
entirely and re-uses the existing artifacts. If any trial under a cell
is missing or truncated, the runner re-runs the whole cell (per-trial
resume is tracked as a Phase 2+ enhancement).

## 2. Recipe shape (Phase 1)

```yaml
schema_version: 1
mode: probe                       # discriminator — required for probe-mode
ticket: ROCM-1234                 # optional; overridden by --ticket when that flag is passed
trials: 3                         # >= 1
mitigation_axis:
  - none
  - tf32_off
diagnostic_axis:
  - none
  - xnack                         # built-in (see note below on registered names)
step_time_regex: null             # phase-1 stub; ignored (rubric §F8)
collect_paths: []                 # phase-1 stub; ignored
timeout_per_trial: 1800           # seconds; null = no timeout
env_passthrough_mode: inherit     # 'inherit' | 'file' — overridden by --env-passthrough-mode when that flag is passed
```

Phase 2 keys accepted in `mode: probe` recipes (rubric §2.C):

```yaml
hang_window_sec: 30                # how long each hang signal must hold
hang_grace_period_at_start: 60     # wait before Tier 2 fires
custom_patterns:                   # Tier 5 user-defined patterns
  - id: oom_killer
    match:
      regex: "out of memory \\(oom_kill"
      condition: "exit_code == 137"   # optional; sandboxed at load
    on_match: fail                    # 'fail' | 'warn' | 'info'
    required_for_pass: false          # only valid with on_match: fail
```

Collect-until-N stopping rule (issue #232) replaces a fixed trial count
with an event-count target plus a hard cap -- the right primitive for an
intermittent bug:

```yaml
stop_after:
  events: 3            # stop this cell once 3 trials match the event verdict
  max_trials: 160      # hard cap -- always honored, required when events is set
  event_verdict: fail  # which verdict counts: fail (default) | pass | error
```

A cell runs until `events` qualifying verdicts are observed **or**
`max_trials` trials have run, whichever comes first. The matrix records
both the realised event count and the trials actually run, so the rate is
`events / trials_run`; `matrix.md` gains a **Stop after** column showing
"stopped early" vs "cap reached", and `matrix.json` carries the rule plus
each cell's `stop_after_note`. Resume is rule-aware: a cell whose on-disk
prefix already satisfies the rule is skipped. CLI escape hatch:
`--stop-after-events K --max-trials N` (unioned over the recipe's block).
`event_verdict: error` (issue #230) counts *error* trials -- handy to bail
out of a sweep that's mostly flaking on infrastructure rather than
reproducing the bug.

Detector-disable knobs (issue #229) silence detectors that fire on
benign, workload-specific behaviour:

```yaml
disable_detector_tiers:            # skip whole tiers (not evaluated)
  - tier3                          #   e.g. all kernel/GPU-counter checks
disable_detectors:                 # skip individual detectors
  - tier2:hang                     #   a repro that legitimately idles
  - tier3:vram_growth              #   opaque docker wrapper allocates VRAM
  - custom:my_pattern              #   a custom_patterns[*] id
```

A disabled detector is **not evaluated** and **never counts** toward the
verdict or the `failure_detectors_fired` / `warn_detectors_fired` lists;
the disabled set is echoed into `result.json::capture` for audit. The one
exception is a command that never launches (exec-time failure such as
command-not-found): Tier 1 is forced back on for that trial so a run that
did no real work can't resolve to a green verdict, even if the operator
disabled Tier 1. Tokens
are validated at load time (unknown tier, malformed `<tier>:<id>`, or a
built-in `tier1`-`tier4` id that isn't in that tier's catalogue ->
`RecipeSchemaError`); `custom:<id>` ids stay free-form. The
`--disable-detector TIER[:ID]` CLI flag
(repeatable) is **unioned onto** the recipe's set, so an operator can
silence one more detector without restating the recipe's list.

Verdict-keyed artifact retention (issue #231) keeps the **heavy** per-trial
output (profiler traces, per-layer dumps -- hundreds of MB each) only for
the trials where it has diagnostic value, so a big sweep doesn't fill the
disk with passing-run data:

```yaml
retain:
  on_fail:  full       # keep everything (log + summary + heavy artifacts)
  on_pass:  summary    # keep small summary artifacts; delete heavy ones
  on_error: log        # keep the trial log only; drop summary + heavy
```

Levels form a ladder, each keeping everything the one below it keeps:
`none` (the trial record only) < `log` (+ `stdout.log` / `stderr.log` /
`probe.env`) < `summary` (+ small collector roll-ups) < `full` (+ heavy
collector outputs; the default). Each of `on_fail` / `on_pass` / `on_error`
is optional and defaults to `full`, so **omitting `retain` keeps everything
exactly as before**. Deletion only ever touches *artifact files* -- the
per-trial `result.json` (the matrix / rate bookkeeping and the probe resume
marker) is **never** deleted, at any level. Pairs naturally with
`stop_after`: "collect N fails with full artifacts, summary-only for the
clean trials along the way."

When a `retain` policy runs, the applied level and the list of pruned
artifacts are written back into that trial's `result.json` under
`capture.retention` (`{"level": ..., "deleted": [...], "freed_bytes": N}`).
A reader of a bundled or resumed run can then tell a missing heavy artifact
was *pruned by policy* rather than never produced.

Collectors declare which of their outputs are heavy vs summary via an
optional `artifacts.json` manifest in the trial directory
(`{"artifacts": [{"path": "trace.pb", "class": "heavy"}, ...]}`); absent a
manifest, a file is classified by convention (`*.summary.*` is a summary,
the known logs are `log`, anything else is `heavy`).

**Reading `matrix.md` (probe runs).** The reproduction-summary table
splits the cell's two recipe axes into their own columns -- `Mitigation`
and `Diagnostic` -- instead of a fused `<mitigation>-<diagnostic>`
identifier, and ends with a `Directory` column giving the per-cell
artifact path relative to `matrix.md` (e.g. `tf32_off-none/`). The folder
name on disk is still `<mitigation>-<diagnostic>` (it stays the stable
join key for tooling and resume); only the table presentation changed, so
an unused diagnostic axis reads as `Diagnostic = none` rather than a
confusing trailing `-none` (issue #229). Triage-mode runs keep the
original `Cell` / `Mitigations` columns.

Phase 3 keys (`redaction`, top-level `condition`) are still **rejected
at load time** with a "deferred to Phase 3" error message.

**Registered mitigation / diagnostic names.** The built-in registry
(see `src/aorta/registry/mitigations.py`) currently ships only `none`,
`tf32_off`, and `xnack`. Any other name (e.g. `hsa_no_scratch_reclaim`,
`fa_prefer_ck`, `hip_launch_blocking`) must come from an
`aorta.mitigations` entry-point plugin or a `--mitigations-file`
sidecar JSON, otherwise the recipe fails to load with
`UnknownMitigationError`. Issue #195 tracks expanding the built-in
set; until that lands, swap any unregistered name for `none` (or one
of the three built-ins above) when copy-pasting this template.

## 3. Env-passthrough modes (`--env-passthrough-mode`)

`inherit` (default)
:   Each cell's mitigations are applied to `os.environ` for the duration
    of the trial, and the child `subprocess.Popen` inherits via
    `env=os.environ.copy()`. No env file is written.

`file`
:   Same as above, **plus** aorta writes `<trial_dir>/probe.env`
    (POSIX `KEY=VALUE\n`, one var per line) at `chmod 0600` and exports
    `AORTA_ENV_FILE=<absolute path>` into the child's environment.
    `AORTA_ENV_FILE` is the only variable aorta exports for this mode;
    the user's argv is **never modified**. Pick this mode if your
    launch command needs to forward the env file by hand:

    * `docker run --env-file "$AORTA_ENV_FILE" ...` — Docker reads the
      file directly.
    * `srun --export=ALL,AORTA_ENV_FILE bash -c 'set -a; . "$AORTA_ENV_FILE"; set +a; exec my_repro ...'`
      — Slurm forwards `AORTA_ENV_FILE` to the remote step, which then
      sources the file into the launched shell.

    Earlier drafts of this doc referenced `AORTA_ENV_FILE_VARS`; that
    variable is **not** exported and never was — use `AORTA_ENV_FILE`
    (a path to the env file) as shown above.

## 4. Resume semantics

`aorta probe` always passes `layout="flat_resume"` and
`resume_existing=True` to `aorta.triage.runner.run_recipe`. That means:

* `<output>/<ticket>/` is created with `mkdir(exist_ok=True)` — no
  timestamp segment, no workload segment, so re-invocations land in the
  same directory.
* Before each cell runs, the runner counts how many `trial_<n>/result.json`
  files under the cell already parse and carry a non-empty `verdict`. If
  that count reaches the cell's `trials` setting, the cell is skipped
  entirely and the existing per-trial JSONs are surfaced into matrix.json
  unchanged. If even one trial is missing or malformed, the **whole cell**
  re-runs from `trial_0` -- per-trial resume is a Phase 2+ enhancement
  (the per-trial coordinate would have to round-trip through the dispatcher
  and the dispatcher currently writes its own `trial_<N>.json` set
  unconditionally).
* When the recipe carries a `stop_after` rule the skip test is
  rule-aware instead of counting against `trials`: the runner walks the
  **contiguous** on-disk trial prefix from `trial_0` and skips the cell
  only when that prefix already satisfies the stopping rule -- i.e. it
  has accumulated `stop_after.events` qualifying verdicts **or** reached
  `stop_after.max_trials`. A cell that legitimately stopped early
  therefore resumes as a skip even though it has fewer than `max_trials`
  trials on disk. If the contiguous prefix has neither hit the event
  target nor reached the cap (or a trial in it is missing/malformed), the
  **whole cell** re-runs from `trial_0`; the re-run is bounded because the
  dispatcher applies the same early-stop.

## 5. CLI / recipe interaction

* `--ticket` overrides the recipe's `ticket` field when present. If the
  flag is omitted, the recipe value is used (falling back to
  `_no_ticket_` if the recipe also omits it, per
  `aorta.triage.output.NO_TICKET_SLUG`).
* `--env-passthrough-mode` overrides the recipe's
  `env_passthrough_mode` field when present. If the flag is omitted,
  the recipe value is used (falling back to `inherit` if the recipe
  also omits it).
* `--dry-run` validates the recipe, prints the planned cell list +
  argv, and exits without writing to disk.
* Any flag not in the table above must come from the recipe; the CLI is
  intentionally **thin** (see `tests/probe/test_cli_parsing.py`).

## 6. What the verdict means in Phase 2

Phase-2 `result.json` shape (exact keys written by
`SubprocessWorkload.run()`). The Phase-1 subset is annotated inline
so downstream tooling that only parsed the Phase-1 fields stays
forward-compatible:

```json
{
  "verdict": "fail",
  "exit_code": 137,
  "walltime_sec": 12.345,
  "peak_vram_mib": 71234,
  "argv": ["python3", "my_repro.py", "--steps", "100"],
  "cell_name": "none-none",
  "trial_index": 0,
  "failure_detectors_fired": ["tier1:exit_nonzero", "tier4:hip_error"],
  "error_detectors_fired": [],
  "warn_detectors_fired": ["custom:slow_iter"],
  "capture": {"loss": "nan"},
  "tier_durations_ms": {"tier1": 0.4, "tier2": 0.1, "tier3": 12.1, "tier4": 1.8, "tier5": 0.3},
  "env_passthrough_mode": "inherit",
  "timed_out": false
}
```

* **Phase-1 keys** (back-compat guaranteed by the
  `tests/probe/test_subprocess_workload.py` minimum-shape assertion):
  `verdict`, `exit_code`, `walltime_sec`, `peak_vram_mib`, `argv`,
  `cell_name`, `trial_index`, `env_passthrough_mode`, `timed_out`.
* **Phase-2 additions** (new in this PR, never replaced or removed):
  `failure_detectors_fired`, `warn_detectors_fired`, `capture`,
  `tier_durations_ms`.
* **Issue #230 addition**: `error_detectors_fired` — the infra-error
  signals (separate from genuine failures) that drive the `error` verdict.
* **Issue #231 addition**: `capture.retention` — present only when a
  `retain` policy ran; records the applied `level`, the `deleted` artifact
  list, and `freed_bytes` so pruning is auditable from the trial record.

`peak_vram_mib` is a coarse high-water mark sampled from two
`amd-smi` snapshots (pre- and post-Popen). It may be `null` when
`amd-smi` is missing or unparseable -- Tier-5 sandbox conditions
that reference it bind `null -> 0` inside `aorta.probe.sandbox.evaluate`
so they stay deterministic.

The verdict is three-way (`pass` / `fail` / `error`; issue #230) and comes
from `aorta.probe.classifier`, with precedence **fail > error > pass**:

1. Any `tier1:*` / `tier2:*` / `tier3:*` / `tier4:*` detector (other than
   the error detectors below) fires OR any `custom_patterns[*]` with
   `on_match: fail` fires → `verdict = "fail"`. `required_for_pass: true`
   patterns that don't fire add `meta:missing_pass_signal` (a fail).
2. Else, if only an error detector fired — `tier1:timeout` with no
   recognised hang, `tier1:exec_failed` (the command never launched), or a
   rejected `probe.env` — → `verdict = "error"`. An `error` trial produced
   no valid observation and is excluded from the matrix event-rate
   denominator.
3. Otherwise → `verdict = "pass"`.

Full ordering rules + per-tier detector IDs:
[classifier.md](classifier.md). `condition:` expression whitelist:
[sandbox.md](sandbox.md).

## 7. `--list-patterns` (Phase 2 deviation)

The rubric's `aorta probe list-patterns` subcommand is implemented as
a **flag** on the existing `aorta probe` command:

```bash
aorta probe --list-patterns          # full Tier 4 catalogue (one entry per pattern)
aorta probe --list-patterns --version  # 'aorta probe pattern library v1 (aorta <pkg>)'
```

This deviation preserves the Phase-1 CLI surface byte-equivalently —
`aorta probe -- <argv>` still works without a subcommand prefix. The
flag short-circuits before recipe loading.

## 8. Shared engine guarantee

`aorta probe` and `aorta triage run` both reach
`aorta.triage.runner.run_recipe` — there is no parallel runner. The
shared-engine contract is pinned by
`tests/probe/test_shared_engine.py`.

## 9. Redaction + bundle (Phase 3)

After a probe run completes, package the ticket leaf for sharing:

```bash
aorta bundle ./probe_results/ROCM-1234/ --review
```

* Redaction policy comes from the recipe's `redaction:` block.
* When `--redaction-from` is omitted, `aorta bundle` auto-loads
  `./probe_results/ROCM-1234/recipe.resolved.yaml` when present.
* Without a `redaction:` block, files are copied byte-for-byte
  (`IdentityRedactor`, zero counts in `manifest.json`).

### `result.json::env`

Every trial's `result.json` includes an `env` object with the cell's
resolved mitigation + diagnostic env bundle. The bundle scrubber removes
keys matching `scrub_env_keys` from that block (and from `probe.env` /
`host_env.json`).

Example minimal `redaction:` block:

```yaml
redaction:
  scrub_env_keys: ["AWS_*", "*_TOKEN", "USER", "HOME"]
  scrub_paths: true
  scrub_ip_addresses: true
```

Handout templates: [handout-templates.md](handout-templates.md).
Scrubber reference: [redaction.md](redaction.md).
Bundle CLI reference: [bundle.md](bundle.md).
