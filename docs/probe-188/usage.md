# `aorta probe` — Phase 1 Usage Walkthrough (issue #188)

> Tier 1 only: verdict is `exit_code == 0 ? "pass" : "fail"`. No
> pattern-matching, hang detection, dmesg scraping, sandboxing, or
> bundling in this phase — see §2 / §3 of the rubric for what is
> coming next.

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
        result.json               # {verdict, exit_code, duration_s, ...}
        probe.env                 # only with --env-passthrough-mode file
    tf32_off-none/
      trial_0/
        ...
```

Re-running the **same command** is a no-op for completed trials: each
trial is "complete" iff its `result.json` parses and has a non-empty
`verdict` field. Truncated, missing, or malformed `result.json` files
trigger a re-run of just that trial.

## 2. Recipe shape (Phase 1)

```yaml
schema_version: 1
mode: probe                       # discriminator — required for probe-mode
ticket: ROCM-1234                 # optional; --ticket on CLI overrides
trials: 3                         # >= 1
mitigation_axis:
  - none
  - tf32_off
diagnostic_axis:
  - none
  - hsa_no_scratch_reclaim        # only if registered in the B3 registry
step_time_regex: null             # phase-1 stub; ignored (rubric §F8)
collect_paths: []                 # phase-1 stub; ignored
timeout_per_trial: 1800           # seconds; null = no timeout
env_passthrough_mode: inherit     # 'inherit' | 'file' — CLI flag overrides
```

Phase 2/3 keys (`custom_patterns`, `condition`, `redaction`) are
**rejected at load time** with a "deferred to Phase 2/3" error message.

## 3. Env-passthrough modes (`--env-passthrough-mode`)

`inherit` (default)
:   Each cell's mitigations are applied to `os.environ` for the duration
    of the trial, and the child `subprocess.Popen` inherits via
    `env=os.environ.copy()`. No env file is written.

`file`
:   Same as above, **plus** aorta writes `<trial_dir>/probe.env`
    (POSIX `KEY=VALUE\n`, one var per line) at `chmod 0600` and exports
    `AORTA_ENV_FILE=<absolute path>` into the child's environment. The
    user's argv is **never modified** — pick this mode if your launch
    command needs to forward the env file by hand
    (`docker run --env-file "$AORTA_ENV_FILE" ...`,
    `srun --export ALL,${AORTA_ENV_FILE_VARS} ...`, etc.).

## 4. Resume semantics

`aorta probe` always passes `layout="flat_resume"` and
`resume_existing=True` to `aorta.triage.runner.run_recipe`. That means:

* `<output>/<ticket>/` is created with `mkdir(exist_ok=True)` — no
  timestamp segment, no workload segment, so re-invocations land in the
  same directory.
* Before each cell runs, the runner checks every `trial_<n>/result.json`
  under that cell. If `trials` complete trials are already on disk, the
  cell is skipped entirely. Otherwise it picks up at the next missing
  index.

## 5. CLI / recipe interaction

* `--ticket` overrides the recipe's `ticket` field.
* `--env-passthrough-mode` overrides the recipe's `env_passthrough_mode`
  field.
* `--dry-run` validates the recipe, prints the planned cell list +
  argv, and exits without writing to disk.
* Any flag not in the table above must come from the recipe; the CLI is
  intentionally **thin** (see `tests/probe/test_cli_parsing.py`).

## 6. What the verdict means in Phase 1

```json
{
  "schema_version": 1,
  "verdict": "pass",
  "exit_code": 0,
  "duration_s": 12.345,
  "started_at": "2026-05-25T13:01:02Z",
  "finished_at": "2026-05-25T13:01:14Z",
  "argv": ["python3", "my_repro.py", "--steps", "100"],
  "mitigation": "none",
  "diagnostic": "none",
  "env_passthrough_mode": "inherit",
  "env_file": null
}
```

`verdict` is **exactly** `exit_code == 0 ? "pass" : "fail"`. Pattern
matching, hang detection, and per-step time analysis are deferred to
Phase 2.

## 7. Shared engine guarantee

`aorta probe` and `aorta triage run` both reach
`aorta.triage.runner.run_recipe` — there is no parallel runner. The
shared-engine contract is pinned by
`tests/probe/test_shared_engine.py`.
