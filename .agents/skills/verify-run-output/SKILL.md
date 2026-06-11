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
- [ ] 5. Root-cause any failure (trace fail/warn back to its origin)
- [ ] 6. Report verdict + findings (+ root cause)
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

### Step 5 — Root-cause any failure

Verification is not done at "it failed" — if any cell/trial has a `fail`
verdict (or a suspicious `warn`), trace it back to its **origin** and report
*why*. A verdict without a cause is not actionable. This step is the difference
between "the run failed" and "the run failed because X, fix Y".

Work the evidence chain from detector → log → cause:

1. **Start from the fired detectors**, not a guess. Read the offending trial's
   `failure_detectors_fired` / `warn_detectors_fired` and
   `failure_details[*]`. Built-in IDs (`tier1:` … `tier4:`) tell you the
   *class* of failure (non-zero exit, signal, hang, dmesg signature, library
   error); `custom:` IDs come from the recipe/sidecar patterns and tell you
   what the **workload itself** flagged.
2. **Open the actual logs** the trial points at — `stdout.log` / `stderr.log`
   next to `result.json`, the `_subprocess/*.stdout.log`, and any
   `run.log` / log dir the workload prints. Pull the **verbatim** error line
   (the exception, the `Errors: N` summary, the dmesg signature). Do **not**
   paraphrase it.
3. **Separate the trigger from the root cause.** A non-zero exit
   (`tier1:exit_nonzero`) is the *trigger*; the ImportError / CUDA OOM /
   assertion in the log is the *root cause*. Likewise distinguish a
   **pre-run/setup crash** (workload never reached its main work — see the
   `main_work_started=false` / `0/<N>` iters signals from Step 4) from a
   **genuine in-workload failure** (the thing under test actually failed). The
   first means "the harness/repro/container is broken"; the second means "the
   bug reproduced". Saying which one it is is the most important output of this
   step.
4. **Explain a uniform matrix.** If *every* cell fails identically (same
   detectors, same log line), that is itself the finding: the failure is
   upstream of anything the matrix varies (mitigations/env), so no cell can
   discriminate and the run carries no mitigation signal. Name the shared
   cause.
5. **Check the cause against the environment / inputs.** Tie the root cause to
   concrete evidence already in the tree where you can: the failing `argv`, the
   docker image/tag, the `resolved_env_vars`, the `env.json` (e.g. a missing
   package, a version mismatch, an unset flag). Point at the file/field, don't
   speculate.
6. **Know when to stop.** Root-cause from the artifacts and the repo. If the
   true cause lives in an external container image, a third-party package, or
   code not in the tree, say so and report the most specific cause you *can*
   prove rather than inventing one. Don't fix anything — this skill is
   read-only.

If the verdict is a clean PASS with no warnings, there is nothing to
root-cause; skip to the report.

### Step 6 — Report

Use this structure:

```markdown
## Run output verification

**Artifact(s)**: <what was found, with the detected run type>
**Verdict**: PASS | PASS (with warnings) | FAIL

### Findings
- FAIL: <contract violation, with the offending value>
- WARN: <suspicious-but-not-wrong, with reasoning>

### Root cause (only when something failed or warned)
<the evidence chain: fired detector → verbatim log line → underlying cause.
State explicitly whether it is a pre-run/setup crash (harness/repro/container
broken) or a genuine in-workload failure (bug reproduced), and — for a uniform
matrix — that the cause is upstream of what the cells vary. Cite the file/field
the evidence came from. Omit this section entirely for a clean PASS.>

### Interpretation
<1–3 sentences: did the run do what it claims? Is the result trustworthy?>

### Suggested next step (only if warranted)
<e.g. fix the root cause (pin package X in image Y), re-run cell Z, install
rdhc for full system_health, inspect trial Y log>
```

Keep it tight. If everything is clean, say so in two lines — do not pad. When
something failed, the **Root cause** section is the part the user most needs —
make it specific and evidence-backed, never a guess.

## Anti-patterns

- Do not rewrite or "fix" the artifacts. This skill is read-only verification.
- Do not validate names against the public built-in registry (see "External
  workloads" above).
- Do not treat a `partial` env snapshot or a documented absence (no docker on
  baremetal, an unset env var) as a failure.
- Do not paraphrase failure causes from logs. Surface the workload's own
  `failure_details[*].hint` / fired detector IDs verbatim.
