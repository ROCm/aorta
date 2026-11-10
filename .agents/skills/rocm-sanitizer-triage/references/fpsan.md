# Triton FpSan

**Readiness: draft.** Planning and conditional interpretation only; a qualified
skill launch procedure is pending. Follow the [common workflow](aorta-workflow.md).

## 1. Question and limits

Compare a Triton candidate with a reference under FpSan. AORTA runs the supplied
comparison command and evaluates its result; it does not construct or compile the
two kernels. RocJITsu is not a dependency of this backend. This is not a general
NaN/Inf detector or numerical-accuracy guarantee. HIP-FpSan is a separate integration.
If only a candidate is available, ask for the intended reference and common inputs.

## 2. Tested environment

The source baseline is AORTA `12e75e694488cd29937f2ee83dccf530c1f0a92a`.
Compatible Triton build, supported GPU target and exact runtime/toolchain for this
skill's reproducible procedure remain pending.

## 3. Required inputs

- Reference and candidate kernels with their identities and intended relationship.
- Common comparison inputs, shapes and configuration.
- A supplied comparison command and its build/runtime environment.
- Run bounds and artifact destination from the common workflow.

## 4. Prerequisites and setup

- Confirm a compatible FpSan-capable Triton environment and target.
- Establish instrumentation of both reference and candidate, with separate
  compilation caches and evidence that the intended builds execute.
- Confirm the command emits exactly one valid `AORTA_FPSAN_RESULT` record to stdout
  using the selected AORTA revision's contract.

The command owns cache separation and the reference/candidate construction. Merely
setting an instrumentation flag does not independently prove that the intended
instrumented kernels ran. A verified setup and command remain specialist deliverables.

## 5. AORTA procedure

Use the `fpsan` recipe backend with the supplied comparison command. A complete
portable recipe/command and its retained outputs are pending qualification. Do not
substitute ordinary numerical output for a missing FpSan reference or result record.

## 6. Evidence and interpretation

Retain AORTA's status/reason, command outcome, the single result record and supporting
instrumentation/build evidence. The contract includes reference/candidate SHA-256
digests and instrumentation booleans, with an optional target. Matching digests with
both flags true and a successful command can satisfy AORTA's comparison contract;
state that the flags are command-reported and identify any independent evidence.
Instrumented mismatch is a comparison finding. Missing/malformed/duplicate records,
unsupported or mismatched targets, and failed commands must not become a clean result.
Even a valid match concerns this FpSan comparison and inputs, not general correctness.

## 7. Required examples and agent cases

Pending: matching and intentionally mismatching pairs; missing reference; an
uninstrumented side; malformed/duplicate result; unsupported target; failed command.
Retain the public inputs, raw outputs and provenance needed to reproduce each case.

## 8. Qualification record

Pending: procedure reviewer, Triton/AORTA revisions, environment, commands, artifact
locations, control outcomes, limits and relevant agent evaluation results.
Advance readiness under the [common rules](aorta-workflow.md#route-readiness).

Source baseline: [AORTA FpSan adapter](https://github.com/ROCm/aorta/blob/12e75e694488cd29937f2ee83dccf530c1f0a92a/src/aorta/instrumentation/rocjitsu_sanitizers/fpsan.py)
and [result contract](https://github.com/ROCm/aorta/blob/12e75e694488cd29937f2ee83dccf530c1f0a92a/src/aorta/instrumentation/rocjitsu_sanitizers/fpsan_result.py).
