# Run-output schemas + consistency rules

Per-artifact key sets, enums, and cross-field rules. The repo docs are
authoritative; this file is the verification-focused digest. When a check
needs the full schema, read the cited doc.

Authoritative sources (public repo):
- `docs/env-probe.md` — `env.json` schema + field sources + fail-soft contract.
- `docs/probe-188/usage.md` + `docs/probe-188/classifier.md` — probe
  `result.json` + verdict precedence + detector IDs.
- `src/aorta/run/results.py` — `TrialResult` dataclass.
- `src/aorta/workloads/_base.py` — `WorkloadResult` dataclass.
- `src/aorta/triage/output.py` + `src/aorta/triage/matrix.py` — `matrix.json` /
  `matrix.md` + `CellStats`.

---

## 1. env.json (env probe snapshot)

Produced by `aorta env probe`, and embedded as `TrialResult.env` in every run.

Always-present top-level keys (subset, stable across schema 1.x):
`schema_version` (str), `captured_at` (ISO-8601 UTC `…Z`), `partial` (bool),
`partial_reasons` (list[str]), `system_health` (dict|null), `rocm`, `hip`,
`hipblaslt`, `rocblas`, `miopen`, `rccl`, `gpu_arch`, `host`, `env_vars`
(dict[str, str|null]), `python_version`, `pytorch_version` (str|null),
`pytorch_build`, `runtime_context`, `docker` (dict|null), `build_system`.

Consistency rules:
- `partial == true`  ⟺ `partial_reasons` non-empty. Mismatch = WARN.
- A snapshot is **complete even when `partial`** — every key is present; it
  just records what fell back to `null`. `partial` is NOT a failure.
- Documented absences do NOT set `partial`: `docker == null` on baremetal,
  `env_vars[X] == null` for an unset var, `venv_path == null` outside a venv.
- `runtime_context.type` ∈ {`docker`, `podman`, `singularity`, `baremetal`}.
- `*.rocm_release_tweak` is a release identifier shared across hipblaslt /
  rocblas / miopen in one release — NOT a per-library commit. For real binary
  drift compare `<lib>.lib_hash` / `<lib>.kernel_db_revision`.

What to surface: if `partial`, echo `partial_reasons` (one per line) — they
name exactly what couldn't be captured (rdhc, headers, submodule SHAs, …).

---

## 2. probe result.json (one probe trial)

Produced by `aorta probe` at `<cell>/trial_<n>/result.json`.

Keys: `verdict` (`"pass"|"fail"|"error"`), `exit_code` (int),
`walltime_sec` (float), `peak_vram_mib` (int|null), `argv` (list[str]),
`cell_name` (str), `trial_index` (int), `failure_detectors_fired` (list[str]),
`error_detectors_fired` (list[str]; issue #230, optional on legacy files),
`warn_detectors_fired` (list[str]), `capture` (dict),
`tier_durations_ms` (dict), `env` (dict[str,str], cell env bundle),
`env_passthrough_mode` (`"inherit"|"file"`), `timed_out` (bool).

Verdict precedence (three-way, **fail > error > pass**; from `classifier.md`):
1. Any `tier1:`/`tier2:`/`tier3:`/`tier4:` detector (other than the error
   detectors below) OR any `on_match: fail` custom (`custom:<id>`) pattern
   fires → `verdict = "fail"`. A `required_for_pass: true` pattern that did
   NOT fire injects `meta:missing_pass_signal` and is a `fail`.
2. Else, if only an error detector fired
   (`tier1:timeout` with no co-firing `tier2:hang`, `tier1:exec_failed`, or
   `meta:env_file_validation_failed`) → `verdict = "error"` (no valid
   observation; excluded from the matrix event-rate denominator).
3. Otherwise → `verdict = "pass"`.

Consistency rules:
- `failure_detectors_fired` non-empty ⟹ `verdict == "fail"`. Else = FAIL.
- `error_detectors_fired` non-empty AND `failure_detectors_fired` empty ⟹
  `verdict == "error"`. Else = FAIL.
- `verdict == "fail"` with empty `failure_detectors_fired` = WARN (expect a
  detector or `meta:missing_pass_signal`); `verdict == "error"` with empty
  `error_detectors_fired` = FAIL (an error must name its reason).
- `timed_out == true` ⟺ a `…:timeout` detector present (in EITHER list —
  `tier1:timeout` is an error detector). Mismatch = WARN.
- `warn_detectors_fired` / `capture` never change the verdict.
- Detector IDs are stable strings; built-in (`tierN:`) and `custom:<id>` are
  peers. Do not treat a `custom:` ID as invalid — it is recipe-declared.

`peak_vram_mib` may legitimately be `null` (no/unparseable `amd-smi`).

---

## 3. TrialResult (aorta run / triage per-trial JSON)

Produced at `<results-dir>/<workload>/trial_d<d>_m<m>_t<t>.json` (run) or
`cells/<cell>/<workload>/trial_*.json` (triage).

Keys: `schema_version` (str, currently `"0.1"`), `trial_id` (str),
`workload` (str), `execution_env` (dict), `mitigations_applied` (list[str]),
`config` (dict), `env` (dict — embedded env.json snapshot),
`result` (dict — `WorkloadResult`), `wall_clock_sec` (float),
`exit_status` (enum).

`exit_status` ∈ {`ok`, `workload_failed`, `workload_setup_failed`,
`infrastructure_failed`}. A value outside this set = WARN (newer schema /
custom producer), not FAIL.

`result` (WorkloadResult) keys: `passed` (bool), `failure_count` (int),
`first_failure_iteration` (int|null), `failure_details` (list[dict]),
`total_iterations` (int), `step_times_ms` (list[float]), `elapsed_sec`
(float), `metrics` (dict), `main_work_started` (bool|null),
`executed_iterations` (int|null), `configured_iterations` (int|null).

Consistency rules:
- `exit_status == "ok"` ⟹ `result.passed == true`. Else = FAIL.
- `exit_status == "workload_failed"` with `result.passed == true` = WARN.
- `workload_setup_failed` means `setup()` raised — the workload never reached
  the measurement. A row of all-setup-failures is NOT "reproduces 100%".
- Semantic flag: `passed == true` AND (`main_work_started == false` OR
  `executed_iterations == 0`) ⟹ the trial passed without doing its work =
  WARN ("did the run actually test anything?").
- Embedded `env.partial == true` with empty `partial_reasons` = WARN.

---

## 4. matrix.json + matrix.md

Both `aorta probe` and `aorta triage run` write these via the shared engine.

`matrix.json`: `schema_version` (1), `workload`, `ticket`, `trials_per_cell`,
`steps_per_trial`, `run_timestamp`, `baseline_cell`, `confound`
(`{threshold, baseline_cell_configured}`), `warnings`, `recipe_source`,
`cells` (list).

Each cell (CellStats): `name`, `mitigations`, `environment`, `extra_env`,
`resolved_env_vars`, `trials`, `passed_count`, `failed_count`,
`error_count` (issue #230), `failure_rate`, `error_rate`, `unreliable`,
`mean_step_time_ms` (+ std/min/max/p50/p90/p99), `mean_wall_clock_sec`,
`exit_status_counts` (dict), `step_times_ms`, `trial_paths`, `error`
(str|null), `step_time_source`, `failure_hints`, `outcome_counts`,
`executed_iter_min/max`, `configured_iters`, `iters_display`,
`workload_config`, `confound`, `step_time_ratio`, `resolved_environment`,
`top_failure_detector_id`, `top_warn_detector_id`, `stop_after_note`.

Consistency rules (non-error cells):
- `passed_count + failed_count + error_count == trials`. Else = FAIL.
- `failure_rate == failed_count / (passed_count + failed_count)` — the event
  rate over *valid* trials; `error` trials are excluded from the denominator
  (issue #230). `0.0` when there are no valid trials. Else = FAIL.
- `error_rate == error_count / trials`. Else = FAIL.
- `unreliable == true` when `error_rate >= 0.10` (advisory; the event rate
  rests on too few valid trials).
- `sum(exit_status_counts.values()) == trials`. Else = WARN.
- `error != null` ⟹ row is preserved with zeroed numerics; `failure_rate`
  reported `n/a` in matrix.md. This is intentional completeness, not a bug.
- `baseline_cell` should be one of the cell names. Else = WARN.
- On-disk `trial_paths` should all exist; missing = WARN (incomplete run).

Confound column legend (matrix.md): `(baseline)`, `-` (works, no speed cost),
`speed (+N%)` (may suppress failure by running slower — verify before
trusting), `no effect`, `n/a` (uncomparable — check `step_time_source`),
`error`, `did_not_run` (cell never reached main work). A `speed` confound is a
caution, not a fix.

matrix.md ↔ matrix.json: `workload` and `ticket` should match; cell rows
correspond to `cells`. matrix.md shows only `mean step (ms)`; per-cell
percentiles, histograms, and trial paths live in matrix.json.

---

## 5. Recipe (yaml)

`recipe.resolved.yaml` is the reloadable snapshot. A `mode: probe` recipe has
`mitigation_axis` / `diagnostic_axis`; a triage recipe has `workload`,
`trials`, `steps`, `confound`, `cells`. Named mitigations/environments are
deliberately NOT expanded in the resolved recipe — to detect registry drift
between runs, compare `matrix.json::cells[*].resolved_env_vars`, not the
recipe.

---

## Probe artifact tree (flat-resume)

```
<output>/<ticket>/
  recipe.resolved.yaml
  host_env.json
  matrix.md / matrix.json
  <cell>/trial_<n>/{stdout.log, stderr.log, result.json, probe.env?}
```
A cell is "complete" iff every `trial_<n>/result.json` parses and has a
non-empty `verdict`. A missing/truncated trial means the whole cell re-runs.

## Triage artifact tree (timestamped)

```
<output>/<ticket>/<workload>/<timestamp>/
  matrix.md / matrix.json / recipe.resolved.yaml
  host_env.json
  environments/<env-name>/env.json
  cells/<cell-name>/<workload>/trial_*.json
  sidecars/<basename>            # copies of any --mitigations-file sidecars
```
