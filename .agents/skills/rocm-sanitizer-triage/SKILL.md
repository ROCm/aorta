---
name: rocm-sanitizer-triage
description: Helps prepare AMD GPU kernels from source or existing binaries for offline Waitcheck through AORTA, select the intended gfx target and entry, and investigate reported wait hazards. Use for compilation preparation, authorized development validation, exact-entry recipes or report-only diagnosis. General execution qualification is pending. Excludes runtime memory/race checking, numerical comparisons and performance tuning.
---

# ROCm sanitizer triage

This shared skill provides the Waitcheck workflow: reuse or build final compiled
GPU code, prepare an exact-entry AORTA check, perform authorized development
validation and interpret the resulting evidence. The scan does not dispatch the
kernel. The user's code need not be a
registered AORTA workload. Other sanitizer integrations are not supplied here.

## Choose the question

Use [Waitcheck](references/waitcheck.md) when the question concerns supported
instruction-wait hazards in compiled AMD GPU code. A crash, hang or wrong answer
alone does not establish that question or its cause. Use available code and
evidence to clarify the symptom; do not promise a runtime memory/race check or a
numerical comparison from this integration.

Identify the producing checker before interpreting supplied evidence. If it is
unknown, ask for the report or checker and give only conclusions that do not
depend on choosing one. State the boundary when another checker produced it.

## Follow the common workflow

When the user simply asks to use this skill to investigate a kernel, guide the
work from the available source, compiled code or report. Explain the immediate
next step in plain language and ask only for information needed to choose it.
Follow [guided investigation](references/aorta-workflow.md#guide-an-investigation)
through the authorized steps and explain what the findings mean.

Apply the current user/workspace approval rules **before initial discovery**.
An existing session on the selected node still runs node commands. When approval
is required, propose the bounded inspection before running shell commands to
locate files or tools. Reuse authority already covering that batch. Follow
[machine and tool discovery](references/aorta-workflow.md#establish-the-machine-and-tools)
without asking again for an established machine or rediscovering known paths.

Read [the common workflow](references/aorta-workflow.md), then the
[Waitcheck guide](references/waitcheck.md). Read the
[exact-entry procedure](references/waitcheck-procedure.md) when preparing a recipe
or running the saved-report reader. For source without a usable object, read
[source preparation](references/waitcheck-build.md).

1. Establish whether the request is a plan, a run or saved-evidence interpretation.
   Keep an explicitly supplied input path as the selected artifact through the
   investigation. A matching file elsewhere does not change that selection.
   Reuse known program/artifact, target, build and intended scope; ask only for
   missing facts that affect the next step. Evidence-only requests do not require
   a new setup or run. When a user brings only a Waitcheck report, start with its
   findings and help investigate the reported hazards using the
   [report-first guidance](references/waitcheck-procedure.md#report-first-investigation).
2. Reuse a compiled artifact when its target/build/specialization matches the
   requested check. Otherwise help prepare a build from the user's source;
   source-only input is a preparation task, not an automatic refusal. Verify
   prerequisites from the guide. Compilation does not establish a native image
   or `.text` entry: inspect the object, without guessing or broadening scope.
3. Prepare a bounded recipe with explicit input identity, target, image/entry,
   timeout, diagnostic limit and fresh output destination. Keep unresolved values
   visibly incomplete. Do not substitute a control for the user's kernel.
4. Use the procedure for specifically authorized development validation, keeping
   pending qualification explicit. For routine supported execution, require the
   documented qualified profile. Follow workspace approval rules before **any**
   cluster command, including inspection, compilation and scanning. A planning
   or evidence-only request authorizes none of these actions.
5. Interpret saved evidence with the compatible reader and Waitcheck's rules.
   Preserve status/reason, findings, checked scope and incompleteness separately.
   Exit zero alone does not establish a clean check.

## Before an authorized development scan

1. Confirm permission for the specific machine and actions. A request to explain
   a report needs only the available evidence. On a cluster, inspection,
   compilation, dry run and scan each need permission covering that action.
2. Identify the pinned AORTA installation and the actual checker bundle or
   approved build. Verify its binary hash, target compatibility and dependencies.
3. Match the final object's digest, target, image, exact kernel name and entry
   offset to independent inventory evidence. Use the [preparation helper](references/waitcheck-prepare.md)
   when the object is available. A value of zero needs the same evidence as any other value.
4. Set the timeout, detail limit and fresh output locations. Review the dry-run
   report against that identity and scope. Stop on mismatch or missing evidence.
5. When these checks pass and the scan is explicitly authorized, perform the
   bounded scan in the [procedure](references/waitcheck-procedure.md), retain
   the report, native log and exits, and explain the result. If blocked, name
   the missing fact or permission and the smallest next action.

## Current readiness

**The expanded build-to-scan route is under development.** Source preparation and
explicitly authorized qualification runs are described; no general execution
profile is qualified yet. Do not mistake user authorization, a working compiler
or an AORTA-supported target for completed validation. Identify the missing
evidence and offer bounded development steps rather than stopping at "source is
insufficient." Advance supported execution claims under the
[readiness rules](references/aorta-workflow.md#route-readiness).

## Answer the user

Use the [common answer](references/aorta-workflow.md#common-answer): checked or
proposed scope, result, evidence, limitations and next action. For a plan, say
**not run**. Attribute supplied evidence to its producing run; do not imply that
you reproduced it. No supported missing-wait finding in a checked scope is not
proof of general correctness, memory safety or race freedom.
