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

`peak_vram_mib` is a coarse high-water mark sampled from two
`amd-smi` snapshots (pre- and post-Popen). It may be `null` when
`amd-smi` is missing or unparseable -- Tier-5 sandbox conditions
that reference it bind `null -> 0` inside `aorta.probe.sandbox.evaluate`
so they stay deterministic.

The verdict comes from `aorta.probe.classifier`:

1. Any `tier1:*` / `tier2:*` / `tier3:*` / `tier4:*` detector fires
   OR any `custom_patterns[*]` with `on_match: fail` fires →
   `verdict = "fail"`.
2. `required_for_pass: true` patterns that don't fire add
   `meta:missing_pass_signal` and flip the verdict.
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
