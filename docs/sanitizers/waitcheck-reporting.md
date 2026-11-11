# Waitcheck diagnostic evidence

Design rationale (11 November 2026, recorded before implementation): preserve
observations without treating a process exit or a numeric summary as proof of
exhaustive analysis. Add optional typed `diagnostic_summary` metadata to entry
and check results. Keep the report schema `aorta.sanitizer_report/0.1`: this is an
additive optional field, not a change to existing statuses or CLI exits. Old
constructors and reports omit it; absence conveys unknown evidence. Older strict
readers may reject the extension and should be upgraded, not bypassed.

The metadata separates `reported_count`, `count_meaning` (`exact`, `lower_bound`,
`unknown`), `emitted_records` before deduplication, `requested_limit` (null means
not recorded, not unlimited), `details` and `native_analysis` (each `complete`,
`incomplete`, `unknown`). Unique findings remain the existing findings list.
Detail completeness concerns the native diagnostic output; native completion
concerns the analysis, not AORTA execution or kernel correctness. A bare numeric
summary, including zero, has unknown count meaning and completion unless a
supported explicit signal establishes more. Matching record/count numbers and
reaching a requested cap alone make no exhaustive claims.

Keep `diagnostics_truncated` as the historical `>=` marker signal (OR at check
level). False never establishes complete details or native completion. Do not
rewrite dated evidence or infer this flag from unique findings. New explicit
omission signals belong in the new metadata.

Aggregate counts are summed only when every selected scan ran and has an exact
reported count. Otherwise the aggregate count is null/unknown, with per-entry
evidence retained. Unknown or failed entries cannot produce affirmative aggregate
completion. The requested limit remains per invocation; it is never summed.

The offline reader compares raw records before deduplication. A plain count/detail
difference remains a compatibility gap unless a supported explicit format explains
it. Warn/native4 and AORTA CLI0 remain separate facts; reader0 only means a summary
was written. No automatic rescan, cap change, or scope change is authorized.

## Source and validation

Primary source inspected at ROCm/rocm-systems revision
`d03c6fe410eed62b211d56bb7878dd1f494269fe`:

- [CLI](https://github.com/ROCm/rocm-systems/blob/d03c6fe410eed62b211d56bb7878dd1f494269fe/emulation/rocjitsu/tools/waitcheck_main.cpp),
  lines 1006–1017, 1123–1143, 1178–1195, 1231–1242, 1307–1335, 1768–1775:
  summary then diagnostic records then optional unprefixed terminal signal.
  `omitted N diagnostic(s) after limit` supports exact observed count, incomplete
  details and completed native analysis in this format. `omitted at least N
  diagnostic(s) after limit` supports a lower bound, incomplete details and early
  termination. `analysis stopped early; additional diagnostics may exist`
  supports a lower bound and early termination; detail completeness stays unknown.
  In the newer early-stop format all three summary counts use `>=`. Historical
  diagnostic-only `>=` stays readable with unknown native completion. Exit0 also
  permits upstream `--no-fail`; AORTA does not request it, so contradictory
  native exits are still rejected by the adapter/reader.
- [Analysis header](https://github.com/ROCm/rocm-systems/blob/d03c6fe410eed62b211d56bb7878dd1f494269fe/emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/code/analysis/waitcheck.h),
  lines 92–101 and 137–161: observed count, storage and early termination are
  separate. Storage limits alone do not stop analysis.
- [Analysis implementation](https://github.com/ROCm/rocm-systems/blob/d03c6fe410eed62b211d56bb7878dd1f494269fe/emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/code/analysis/waitcheck.cpp),
  lines 1332–1355, 5268–5327: native deduplication precedes observed-count
  increment and detail retention. Its keys differ from AORTA's finding keys.
- [Guide](https://github.com/ROCm/rocm-systems/blob/d03c6fe410eed62b211d56bb7878dd1f494269fe/emulation/rocjitsu/docs/waitcheck/README.md),
  lines 203–210, 268–275, 330–341; and
  [CLI regression source](https://github.com/ROCm/rocm-systems/blob/d03c6fe410eed62b211d56bb7878dd1f494269fe/emulation/rocjitsu/tests/tools/rj_waitcheck_smoke_test.cpp),
  lines 531–614. Inspected, not executed. Exhaustive JSONL mode is outside this
  adapter's supported text format.

These supported terminal formats establish interpretation of output, not binary
build provenance. Bare counts (including zero) lack such a completion signal.
The saved binary's revision is unknown. Local parser tests do not qualify a
native checker or promote the draft skill execution route.

Portable minimal derivatives and substitutions are documented in
`tests/instrumentation/rocjitsu_sanitizers/fixtures/README.md`. The saved cap64
case remains native4/warn, reported3/records3/unique3, reader0, with count meaning,
detail completeness and native completion unknown. Actual cap1 replay found the
supported terminal `omitted 2 diagnostic(s) after limit`, which the old reader's
count gate never reached. It remains native4/warn, reported3/records1/unique1,
but now reader0 explains an exact observed count, incomplete details and reported
completed native analysis in this explicit format. This is output interpretation,
not independent proof of exhaustive checker coverage or binary source revision.
Both producing AORTA CLI exits were0; both historical false truncation fields are
preserved. A bare count/record mismatch without that terminal signal still yields
reader2 and unknown semantics. Cap64 already supplies three details for this
control, so no rescan is needed to recover them. Original reports/logs and frozen
failure records remain unchanged.

## Local validation commands

Use Python from a local AORTA development environment with pytest installed.
From the repository root, run the portable reporting and consumer tests:

```bash
python -m pytest -q \
  tests/instrumentation/rocjitsu_sanitizers/test_waitcheck.py \
  tests/instrumentation/rocjitsu_sanitizers/test_report_pipeline.py \
  tests/instrumentation/rocjitsu_sanitizers/test_waitcheck_report.py \
  tests/instrumentation/rocjitsu_sanitizers/test_waitcheck_failures.py \
  tests/instrumentation/rocjitsu_sanitizers/test_diagnostic_summary.py \
  tests/instrumentation/rocjitsu_sanitizers/test_selection_models.py \
  tests/instrumentation/rocjitsu_sanitizers/test_recipe_daily.py \
  tests/instrumentation/rocjitsu_sanitizers/test_waitcheck_controls_example.py \
  tests/instrumentation/rocjitsu_sanitizers/test_waitcheck_skill_recipe.py \
  tests/instrumentation/rocjitsu_sanitizers/test_waitcheck_isolation_recipe.py \
  tests/instrumentation/rocjitsu_sanitizers/test_waitcheck_native_cases.py
git diff --check
```

The tests use packaged text fixtures, fake executors and inert objects. They need
no private development capture, native checker or GPU. The consumer cases cover
legacy reports, supported omission signals, unknown semantics, aggregate scope,
duplicate object aliases and report/count/native-exit contradictions. They do not
qualify native execution or establish source/build provenance.

For supplied real evidence, use the skill's saved reader as described in
[the procedure](../../.agents/skills/rocm-sanitizer-triage/references/waitcheck-procedure.md#4-read-the-saved-report-and-raw-log).
Preserve original inputs and use a fresh, writable output destination. Reader
exit0 means a summary was written; keep native and AORTA exits separate. The
[recorded examples](../../.agents/skills/rocm-sanitizer-triage/references/waitcheck-examples.md)
describe bounded real-tool evidence and remaining qualification limits.

## Changed implementation

- `src/aorta/instrumentation/rocjitsu_sanitizers/models.py`: optional typed metadata,
  conservative aggregation and validation of full selected scan scope.
- `src/aorta/instrumentation/rocjitsu_sanitizers/waitcheck.py`: raw records/counts,
  supported explicit signals, identity/format/count conflicts, requested limits
  and separate aggregate evidence. No cap/default or CLI exit change.
- `src/aorta/instrumentation/rocjitsu_sanitizers/__init__.py`: type exports.
- `.agents/skills/rocm-sanitizer-triage/scripts/waitcheck_report.py`: compatible
  old/new interpretation, report-only conflicts, raw records versus deduplication.
- `tests/instrumentation/rocjitsu_sanitizers/test_waitcheck.py`,
  `test_waitcheck_report.py`, new `test_diagnostic_summary.py` and the two fixture
  texts/README: synthetic/portable regressions.
- `src/aorta/instrumentation/rocjitsu_sanitizers/README.md`, `recipe.py` docstring,
  this note, skill references `waitcheck.md`, `waitcheck-procedure.md`,
  `waitcheck-examples.md`, and comments in `waitcheck-exact-entry.yaml`,
  `waitcheck-cap-high.yaml`, `waitcheck-cap-low.yaml`: reporting contract and limits.

Historical reports and capture helpers retain their original interpretation
records. This change concerns Waitcheck reporting; ASan, ConSan and FpSan remain
separate checker integrations.
