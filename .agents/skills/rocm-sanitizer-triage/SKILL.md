---
name: rocm-sanitizer-triage
description: Plans offline Waitcheck checks through AORTA for compiled AMD GPU missing-wait hazards and explains supplied Waitcheck findings, incomplete checks and evidence limits. Use for exact-entry recipe preparation or saved Waitcheck results. Qualified execution is not yet provided. Excludes runtime memory/race checking, numerical comparisons, NVIDIA tools, general ROCm installation and performance tuning.
---

# ROCm sanitizer triage

This shared skill currently provides the Waitcheck integration: prepare an offline
check of final compiled GPU code and interpret saved missing-wait evidence through
AORTA. The scan does not dispatch the kernel. The user's code need not be a
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

Read [the common workflow](references/aorta-workflow.md), then the
[Waitcheck guide](references/waitcheck.md). Read the
[exact-entry procedure](references/waitcheck-procedure.md) when preparing a recipe
or using the bundled saved-report reader.

1. Establish whether the request is a plan, a run or saved-evidence interpretation.
   Reuse known program/artifact, target, build and intended scope; ask only for
   missing facts that affect the next step. Evidence-only requests do not require
   a new setup or run.
2. Verify Waitcheck prerequisites from its guide. Source alone cannot establish
   a final object, selected image or verified `.text` entry. Preserve unknowns;
   never invent selectors or silently fall back to whole-object scanning.
3. Prepare a bounded recipe with explicit input identity, target, image/entry,
   timeout, diagnostic limit and fresh output destination. Keep unresolved values
   visibly incomplete. Do not substitute a control for the user's kernel.
4. Run only when the documented route is qualified and the specific action is
   authorized. Follow workspace approval rules before **any** cluster command,
   including read-only inspection. A planning or evidence-only request does not
   authorize execution.
5. Interpret saved evidence with the compatible reader and Waitcheck's rules.
   Preserve status/reason, findings, checked scope and incompleteness separately.
   Exit zero alone does not establish a clean check.

## Current readiness

**Waitcheck remains draft: planning and supplied-evidence interpretation only.**
The template and procedure do not yet provide qualified skill-guided execution.
If asked to run, prepare the plan and identify missing qualification. Bounded
development controls do not qualify arbitrary kernels or environments. Advance
only under the [readiness rules](references/aorta-workflow.md#route-readiness).

## Answer the user

Use the [common answer](references/aorta-workflow.md#common-answer): checked or
proposed scope, result, evidence, limitations and next action. For a plan, say
**not run**. Attribute supplied evidence to its producing run; do not imply that
you reproduced it. No supported missing-wait finding in a checked scope is not
proof of general correctness, memory safety or race freedom.
