# ASan

**Readiness: draft.** Planning and conditional interpretation only; a qualified
skill launch procedure is pending. Follow the [common workflow](aorta-workflow.md).

## 1. Question and limits

Check supported memory accesses in suitably instrumented host/GPU code. Device
ASan does not require RocJITsu. It is not a race checker, and a clean result cannot
establish that every access form or allocation is covered. Ask whether the user can
provide or build an instrumented target when they supply an ordinary release binary.

## 2. Tested environment

The source baseline is AORTA `12e75e694488cd29937f2ee83dccf530c1f0a92a`.
Upstream documents measurements on gfx950, but this skill's reproducible procedure,
tested compiler/ROCm/runtime versions and supported target matrix remain pending.
Do not extrapolate one recorded target's behavior to every GPU or stack.

## 3. Required inputs

- Reproduction program, selected kernel/variant and representative arguments.
- Build command/options, compiler version and resulting device-code identity.
- Intended recipe route or explicitly selected compatible registered workload.
- AORTA/runtime environment and run bounds from the common workflow.

## 4. Prerequisites and setup

- Confirm device instrumentation in the selected code, not just host ASan symbols
  or the presence of a compiler flag.
- Establish a compatible instrumented ASan runtime and supported XNACK setup for
  the selected target. A stock runtime or an incompatible target can break reporting.
- Establish a way to prove the intended kernel dispatched and to check node health.
- Record unsupported access forms and allocation types for the tested environment.

An ordinary uninstrumented binary is not checked merely because ASan was requested.
The specialist must provide verified build/runtime setup and prechecks; do not
invent a universal compile line or apply instrumentation to unrelated builds.

## 5. AORTA procedure

The recipe backend is `asan`. AORTA also supports the interface
`aorta run --workload <name> --sanitizer asan` for compatible registered workloads.
This is not a complete command for an arbitrary workload. Pending qualification:
one portable user-program recipe/build and one existing compatible workload route.
An existing workload may validate setup; it does not substitute for checking the user's code.

## 6. Evidence and interpretation

Keep AORTA's report/reason, sanitizer reports, selected-code instrumentation and
dispatch evidence, runtime failures and node-health evidence. A memory finding and
broken checking are different outcomes. Exit zero and no diagnostic do not prove
the intended kernel executed or its memory accesses were checked. Explain unchecked
accesses and scope even when the reported run passes.

## 7. Required examples and agent cases

Pending: known invalid access and matched clean control; missing instrumentation or
runtime; selected kernel not dispatched; node/setup failure. Include both planned
routes and confirm the advertised behavior against the selected AORTA revision.
Use public, reproducible inputs and retain raw evidence.

## 8. Qualification record

Pending: procedure reviewer, exact build/environment, commands, artifact locations,
control outcomes, limits and agent evaluation results. Record any AORTA version
restrictions exposed by qualification rather than assuming pending fixes are present.
Advance readiness under the [common rules](aorta-workflow.md#route-readiness).

Source baseline: [AORTA device ASan capability measurements](https://github.com/ROCm/aorta/blob/12e75e694488cd29937f2ee83dccf530c1f0a92a/docs/sanitizers/asan-gpu-capability.md).
