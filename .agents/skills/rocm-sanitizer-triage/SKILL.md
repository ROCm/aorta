---
name: rocm-sanitizer-triage
description: Plans ROCm GPU sanitizer checks through AORTA and explains supplied AORTA sanitizer results and their limits. Use for choosing Waitcheck, ASan, ConSan or Triton FpSan, preparing a targeted check, or understanding findings and unchecked scope. Qualified execution is not yet provided. Excludes NVIDIA checking, CPU-only sanitizers, general ROCm installation and performance tuning.
---

# ROCm sanitizer triage

Help the user answer a specific GPU correctness question using AORTA's existing
checker integrations. Their program need not be a registered AORTA workload.
AORTA prepares checks and records evidence; the selected checker detects its
supported class of problems. This skill guides the agent through those interfaces.

## Choose the question

Use this skill for AORTA sanitizer planning and evidence questions. A general
wrong-answer, crash or hang report needs clarification before assuming a sanitizer
is the right next step. General numerical-output validation or performance analysis
alone does not establish a sanitizer task.

| User's question | Read |
| --- | --- |
| Does compiled GPU code contain supported missing-wait hazards? | [Waitcheck](references/waitcheck.md) |
| Does an instrumented execution expose an invalid memory access? | [ASan](references/asan.md) |
| Does a targeted execution expose supported shared-memory races? | [ConSan](references/consan.md) |
| Do a Triton candidate and reference agree under FpSan? | [Triton FpSan](references/fpsan.md) |

Do not run all four by default. A crash, hang or wrong answer alone does not
identify the checker. Ask what was observed and whether the immediate question is
memory access, shared-memory concurrency, compiled instruction waits, or a
reference/candidate comparison. Use available code and evidence to narrow that
question; explain uncertainty instead of guessing a diagnosis. For a user-selected
checker that cannot answer the question, explain the mismatch before proposing another.

For supplied evidence, identify the producing checker from the report or user
context before applying its interpretation rules. If the checker is unknown, ask
for it and give only conclusions that do not depend on choosing one. If none of
the supported checkers fits, state that boundary; do not force a selection.

## Follow the common workflow

Read [the common workflow](references/aorta-workflow.md), then the selected
checker's reference. If selection is unresolved, clarify it before preparing a
checker-specific plan. Read additional references only when needed for an explicit
comparison or a dependency stated by the selected reference.

1. Establish whether the user wants a plan, a run, or interpretation of saved
   evidence. Collect the relevant program/kernel, inputs, build, environment and
   intended scope from available context; ask only for missing decisions or facts.
   For supplied-evidence interpretation, go to step 5. Assess prerequisites from
   the producing run's evidence as needed; do not prepare or reproduce a run merely
   to explain a report.
2. **verify the selected checker’s prerequisites.** Its reference owns the details.
   Record what is verified, missing or unknown; do not treat assumptions as checks.
3. Prepare the bounded AORTA invocation using that reference's supported route.
   State the selected target, inputs, limits and output location. Keep an incomplete
   plan visibly incomplete; never invent a working command to fill a missing procedure.
4. Run only when the route is qualified and the action is authorized under the
   user's instructions and workspace rules. Planning and evidence-only requests do
   not authorize a new run. Follow explicit cluster-approval requirements before
   **any** cluster command, including prerequisite inspection.
5. Read the saved evidence under the checker's interpretation rules. Preserve
   AORTA's status and reason, and distinguish findings, scoped no-finding results,
   failed checks and unchecked scope. Exit zero alone establishes none of these.

## Current readiness

All four references are **draft: planning and supplied-evidence interpretation only**.
This package does not yet provide qualified launch procedures. If asked to run,
prepare the plan and identify the missing qualification; do not present this draft
as execution support. A route can advance independently when its reference records
reviewed instructions, real checker qualification and agent evaluation evidence.
Unfinished routes remain draft. See the [readiness rules](references/aorta-workflow.md#route-readiness).

## Answer the user

Use the [common answer format](references/aorta-workflow.md#common-answer): what was
checked or proposed, the result, supporting evidence, limitations, and next action.
For a plan, say **not run**. For supplied evidence, state its provenance and do not
imply you reproduced the run. Never turn "no finding in this scope" into "the program
is correct."
