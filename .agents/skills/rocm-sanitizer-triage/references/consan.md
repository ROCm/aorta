# ConSan

**Readiness: draft.** Planning and conditional interpretation only; a qualified
skill launch procedure is pending. Follow the [common workflow](aorta-workflow.md).

## 1. Question and limits

Check supported shared-memory concurrency behavior through the RocJITsu runtime
hook. AORTA uses a caller-supplied targeted command and combined Waitcheck preflight.
RocJITsu components do not imply a full simulator is required. General automatic
whole-model top-K replay is outside this route. If the user supplies a load-only
command, ask for a valid targeted execution before promising dynamic race checking.

## 2. Tested environment

The source baseline is AORTA `12e75e694488cd29937f2ee83dccf530c1f0a92a`.
Tested hook/tool revisions, runtime, supported targets and launch environment for
this skill's procedure remain pending. Preserve bundle provenance when selecting
prebuilt RocJITsu components.

## 3. Required inputs

- Exact code object and selected kernel identity.
- Targeted caller command, argument layout, representative values and valid buffers.
- Intended execution and supported access scope, including any sampling limits.
- Environment, run bounds and artifact destination from the common workflow.

## 4. Prerequisites and setup

- Locate a compatible `librocjitsu_dbi_hooks.so` and establish the supported runtime
  hook setup for the selected environment.
- Establish the required combined Waitcheck preflight and its evidence.
- Confirm the command actually dispatches the intended kernel with valid arguments.
- Establish the checker's supported sites and required execution/coverage evidence.

A load-only helper may reach instrumentation, but it does not establish dynamic
race coverage. Do not invent launch arguments or treat zero-size inputs as a
representative execution. The specialist must supply a reviewed setup/launch procedure.

## 5. AORTA procedure

Use the `consan` recipe backend with a caller-supplied targeted command. A portable
recipe and verified runtime command are pending. Do not silently broaden a selected
kernel request to whole-application instrumentation when targeted replay is unavailable.

## 6. Evidence and interpretation

Preserve AORTA's status/reason, Waitcheck results, ConSan findings, execution evidence
and per-object coverage. Missing or contradictory coverage prevents a clean dynamic
checking claim. Apply the producing AORTA revision's rules for site-free/not-applicable
results; zero sites alone is not proof of a clean race check. Explain load-only
evidence as such. A finding can coexist with incomplete coverage.

## 7. Required examples and agent cases

Pending: known race and matched clean control; load-only command; invalid arguments;
missing/contradictory coverage; site-free input; Waitcheck preflight failure. Preserve
raw outputs and verify interpretation with the selected parser version.

## 8. Qualification record

Pending: procedure reviewer, tool/runtime revisions, commands, artifact locations,
control outcomes, access/sampling limits and agent evaluation results. Record version
restrictions rather than assuming pending parser changes have merged.
Advance readiness under the [common rules](aorta-workflow.md#route-readiness).

Source baseline: [AORTA ConSan setup, dispatch and coverage documentation](https://github.com/ROCm/aorta/blob/12e75e694488cd29937f2ee83dccf530c1f0a92a/src/aorta/instrumentation/rocjitsu_sanitizers/README.md).
