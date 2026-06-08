---
name: verify-run-output
description: >-
  Verify the output of any aorta run — env probe, probe, triage run, run, or a
  single trial/matrix artifact — against its documented schema and internal
  consistency rules, then report a pass/warn/fail verdict. Use when the user
  asks to verify, validate, sanity-check, audit, or interpret an aorta run's
  results: env.json snapshots, probe result.json, triage/run trial JSONs,
  matrix.json / matrix.md, a results directory, or a recipe. Works for
  built-in workloads and for workloads, mitigations, and environments provided
  by external entry-point plugins or sidecar JSON files.
---

# Verify Run Output

Verify what an aorta command produced. The aorta platform writes a small set
of well-defined artifacts; this skill identifies which one you have, checks it
against its schema, checks it for internal consistency, and returns a verdict.

## Scope

The aorta CLI is a thin front-end over a shared engine. Every run type emits
artifacts from this fixed vocabulary:

| Run type | Primary artifacts |
|---|---|
| `aorta env probe` | `env.json` (a `collect_env` snapshot) |
| `aorta probe` | flat-resume tree: `matrix.json`/`matrix.md`, `recipe.resolved.yaml`, `host_env.json`, `<cell>/trial_<n>/result.json` |
| `aorta triage run` | timestamped tree: `matrix.json`/`matrix.md`, `host_env.json`, `environments/<env>/env.json`, `cells/<cell>/<workload>/trial_*.json` |
| `aorta run` | `<results-dir>/<workload>/trial_d<d>_m<m>_t<t>.json` (`TrialResult` JSONs) + embedded `env` snapshot |

A run type the user names that is not in this table (e.g. an "agent" or a new
subcommand) still produces JSON/markdown/tree artifacts in the same families,
so the same identify → schema-check → consistency-check → verdict workflow
applies. Treat an unrecognised artifact as "apply generic JSON/structure
checks and say so" — never invent schema rules.

## External workloads — important

Workloads, mitigations, diagnostics, and environments can be supplied by
**external entry-point plugin packages** (the `aorta.workloads` /
`aorta.mitigations` groups) or by `--mitigations-file` **sidecar JSON**, not
only by the public built-ins. Therefore:

- **Never flag a workload / mitigation / diagnostic / environment name as
  "unknown" or invalid** just because it is not a public built-in. Names like
  these are resolved at run time from plugins or sidecars and are legitimate.
- Verify the **shape and consistency** of the output, not the provenance of
  the names in it.
- Do not assume any particular private repository, customer, ticket, or
  reproducer is involved. Verify only what the artifacts state.

## Workflow

```
Verification progress:
- [ ] 1. Identify the artifact(s)
- [ ] 2. Run the structural + consistency validator
- [ ] 3. Interpret findings against the schema (reference.md)
- [ ] 4. Judge semantic plausibility (verdict justified by evidence?)
- [ ] 5. Report verdict + findings
```

### Step 1 — Identify

If given a directory, it is a run tree (probe / triage / run). If given a
file, classify by keys:

- `verdict` + `failure_detectors_fired` → **probe `result.json`** (one trial).
- `trial_id` + `exit_status` + `result` → **`TrialResult`** (`aorta run` /
  triage per-trial JSON).
- `cells` + `baseline_cell` → **`matrix.json`**.
- `partial` + (`rocm` | `captured_at`) → **`env.json`** snapshot.
- `.yaml`/`.yml` with `mode: probe` or a `workload:` key → a **recipe**.

### Step 2 — Run the validator

The bundled script does structural + consistency checks and exits non-zero on
any contract violation. Always run it first; it is faster and more reliable
than eyeballing JSON.

```bash
python .agents/skills/verify-run-output/scripts/verify_run.py <path> [<path> ...]
# show every passing check too:
python .agents/skills/verify-run-output/scripts/verify_run.py <path> --verbose
# machine-readable:
python .agents/skills/verify-run-output/scripts/verify_run.py <path> --json
```

It accepts a single artifact file, a run directory (it walks the tree), or
several paths at once. Findings are `ok` / `warn` / `fail`; the final line is a
`PASS` / `WARN` / `FAIL` verdict.

### Step 3 — Interpret against the schema

The script catches mechanical violations. For anything subtler, or to explain
a finding, consult [reference.md](reference.md) — it has the per-artifact key
sets, enums, and the cross-field consistency rules (verdict precedence,
`exit_status` ↔ `passed`, `failure_rate` math, `partial` ↔ `partial_reasons`,
matrix.md ↔ matrix.json agreement). The repo's own docs are the authoritative
source and reference.md cites them.

### Step 4 — Semantic plausibility

Beyond schema validity, judge whether the verdict is *earned*:

- A `pass` / `passed=true` trial that **never started its main work**
  (`main_work_started=false`, or `0/<N>` iters, or `did_not_run` outcome) is
  suspicious — the workload may not have tested anything. Call it out.
- A `fail` verdict should be explained by a fired detector or a
  `passed=false`; if nothing explains it, flag it.
- A `partial` env snapshot is **valid** (fail-soft by design) but its
  `partial_reasons` should be surfaced — they say what couldn't be captured.
- For a matrix, sanity-check that the baseline cell and confound tags tell a
  coherent story (e.g. a `speed (+N%)` confound means "verify before trusting
  this mitigation", not "fixed").

### Step 5 — Report

Use this structure:

```markdown
## Run output verification

**Artifact(s)**: <what was found, with the detected run type>
**Verdict**: PASS | PASS (with warnings) | FAIL

### Findings
- FAIL: <contract violation, with the offending value>
- WARN: <suspicious-but-not-wrong, with reasoning>

### Interpretation
<1–3 sentences: did the run do what it claims? Is the result trustworthy?>

### Suggested next step (only if warranted)
<e.g. re-run cell X, install rdhc for full system_health, inspect trial Y log>
```

Keep it tight. If everything is clean, say so in two lines — do not pad.

## Anti-patterns

- Do not rewrite or "fix" the artifacts. This skill is read-only verification.
- Do not validate names against the public built-in registry (see "External
  workloads" above).
- Do not treat a `partial` env snapshot or a documented absence (no docker on
  baremetal, an unset env var) as a failure.
- Do not paraphrase failure causes from logs. Surface the workload's own
  `failure_details[*].hint` / fired detector IDs verbatim.
