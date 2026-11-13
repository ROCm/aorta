# Maintaining the shared sanitizer skill and Waitcheck integration

This increment delivers the shared foundation and **Waitcheck only** in
[rocm-sanitizer-triage](../../.agents/skills/rocm-sanitizer-triage/SKILL.md).
It supports offline missing-wait planning and supplied-evidence interpretation;
qualified skill-guided execution remains draft. Other checker procedures arrive
in separate AORTA PRs. The completed shared package is submitted to AMD Skills
together after its integration, evaluation and product approvals.

## Shared requirement checklist

These are acceptance expectations, not evidence that evaluations passed.

| Piece | Maintained source | Observable check |
| --- | --- | --- |
| Selection | [SKILL.md](../../.agents/skills/rocm-sanitizer-triage/SKILL.md#choose-the-question) | Select for offline Waitcheck questions; leave runtime memory/race, numerical, installation and performance requests outside this increment. Clarify ambiguous symptoms. |
| Inputs and mode | [Common workflow](../../.agents/skills/rocm-sanitizer-triage/references/aorta-workflow.md#inputs-and-task-mode) | Separate plan, execution and saved evidence. Reuse facts, preserve unknowns, and do not require new setup just to explain supplied text. |
| Exact scope and prerequisites | [Waitcheck guide](../../.agents/skills/rocm-sanitizer-triage/references/waitcheck.md) and [procedure](../../.agents/skills/rocm-sanitizer-triage/references/waitcheck-procedure.md) | Establish final object/hash, target, image/entry and tool provenance. No source-only coverage claim or silent whole-object fallback. Preserve authorization and draft-readiness limits. |
| Interpretation | [Evidence rules](../../.agents/skills/rocm-sanitizer-triage/references/waitcheck.md#6-evidence-and-interpretation) | Preserve status/reason, findings, diagnostic counts/detail completeness and native completion separately. CLI zero is not a clean verdict. |
| Answer | [Common answer](../../.agents/skills/rocm-sanitizer-triage/references/aorta-workflow.md#common-answer) | State checked/proposed scope, result, evidence, limits and next action without inventing provenance or a reproduced run. |
| Checks | [Checks and CI](#checks-and-ci) | Distinguish structural/schema checks, mocked CPU tests, agent evaluations and real checker qualification. |
| Intake | [Maintenance and AMD intake](#maintenance-and-amd-intake) | Keep owner/license/product approvals and combined-package submission gates explicit. |

## Ownership and package boundaries

One person may own the foundation and all checker integrations. Shared ownership
covers the entrypoint, common workflow, card, single dataset and CI integration.
Each checker increment adds only its supported reference/procedure, useful helper
or template, and relevant cases. Independent review remains required.

The installed skill retains the three AMD-required files (`SKILL.md`,
`skill-card.md`, `evals/evals.json`), common workflow, Waitcheck guide/procedure,
[exact-entry template](../../.agents/skills/rocm-sanitizer-triage/assets/waitcheck-exact-entry.yaml)
and [saved-report reader](../../.agents/skills/rocm-sanitizer-triage/scripts/waitcheck_report.py).
The reader needs Python from a compatible AORTA installation, including the
matching report/model/parser APIs; copying the skill does not install AORTA.

Development resources remain in the repository under
[examples/sanitizers/waitcheck](../../examples/sanitizers/waitcheck/):

- [Three controls and retained limits](../../examples/sanitizers/waitcheck/docs/waitcheck-examples.md), control runner, adapted HIP sources and upstream MIT notice.
- [Two-kernel isolation](../../examples/sanitizers/waitcheck/docs/waitcheck-isolation.md) and cap/invalid/isolation recipe templates.
- Repository tests under `tests/instrumentation/rocjitsu_sanitizers/`.

These resources are not installed merely because they share a PR. Links from the
skill to repository-only examples use public ROCm/aorta URLs so catalog imports
do not depend on the source checkout's parent directories. Preserve
`LICENSE.upstream.txt` with the adapted sources. Do not ship internal project
state, review receipts, credentials or private kernel artifacts.

## Checker contribution

Preserve the shared workflow and result contract. Keep prerequisites in the
checker reference; there is no universal RocJITsu, instrumented-build or GPU
execution requirement. Do not add a second skill, result schema or runner merely
to extend the instructions.

For each later integration, provide its checking question, supported inputs,
tested revisions/environment, prerequisites, bounded AORTA route, interpretation
rules and finding/clean/incomplete examples. Retain exact commands, input/tool
identities, expected/observed results, artifacts and independent review. Advance
only the documented route under the
[readiness rules](../../.agents/skills/rocm-sanitizer-triage/references/aorta-workflow.md#route-readiness).
Backend availability, bounded controls, agent evaluation and release are separate.

Waitcheck's controls/reporting/isolation/attention evidence remains bounded by its
recorded inputs and tool versions. Native acquisition/build provenance, applicable
cleanup guarantees and complete loaded-skill activity/model evidence remain open.
Moving examples or passing local tests does not close those qualification gaps.

## Evaluation conventions

Maintain one [Skillscope dataset](../../.agents/skills/rocm-sanitizer-triage/evals/evals.json).
Current positive cases concern Waitcheck planning, source-only inputs, supplied
finding/corrected/refused-input evidence, diagnostic omission and cluster approval.
Runtime ASan is an exclusion, not a promised integration in this increment.
Use at least three positive and two negative routing cases and two positive
behavior cases. Preserve stable case IDs unless intentionally retiring a case.
Grade decisions rather than exact headings or wording; dataset presence is not a
passing model evaluation.

A negative means no skill in the **tested routing room** should activate. This
single-skill room contains only the sanitizer skill. Tasks owned by other skills
belong in those skills' positive datasets when testing a larger room.

Skillscope v0.1.2 copies the whole skill folder, including the required dataset,
into routing and behavioral workspaces. These are development evaluations, not
blind/held-out performance evidence. Keep teaching answers outside the installed
skill; removing a link does not hide a file. Record prompts, actual loaded skill,
model/settings, agent activity, outputs and the routing room. File-dependent cases
must establish their staged paths/working directory. No GPU case belongs in a
CPU-only workflow without separately authorized environment support.

For JSON, CSV and log analysis, use a saved reusable compatible reader. Review new
or modified helpers independently and retain reproducible commands and outputs.
Do not replace missing provenance or checker evidence with a successful reader.

## Checks and CI

The workflow `.github/workflows/sanitizer-skill-checks.yml` watches the skill,
Waitcheck examples, sanitizer docs/backend/tests and its own definition. It runs
the existing eleven-file CPU suite, with mocked subprocesses and inert data.
It does not dispatch GPU input programs or run a native checker.

Skillscope is pinned to v0.1.2 at
`68b69e2eff60c41f5d5482bb89f24e07ce4fbebb`. Structural checks validate formatting,
links, dataset shape and Description/Owner/License sections. Pending card fields
can pass these checks; that does not establish an owner or distribution approval.

Routing/behavior jobs remain explicit manual dispatch with
`run_agent_evals: true` and `ANTHROPIC_API_KEY`. They can spend model credits and
must use reviewed code/cases. Pull-request checks receive no model secret.
No self-hosted GPU or cluster job is configured. Workflow definitions do not
establish actual CI/model results. Preserve any user requirement to approve every
cluster command, including inspection.

From the AORTA root, with a separate pinned Skillscope checkout on PYTHONPATH:

```text
python -m skillscope structural --skills-dir .agents/skills/rocm-sanitizer-triage --skill-files skill-card.md --skill-sections Description,Owner,License --docs docs/sanitizers/skill-maintenance.md,docs/sanitizers/waitcheck-reporting.md,examples/sanitizers/waitcheck/docs/waitcheck-examples.md,examples/sanitizers/waitcheck/docs/waitcheck-isolation.md
```

Use supported Python (AORTA declares >=3.10) and the installed AORTA dependencies.
Retain source/helper versions, command, environment and output. The pinned model
workflow installs the current Claude CLI; the harness pin alone does not pin that
CLI. Review new/moved/modified scripts with their callers and tests.

### Local portability check

Copy the complete skill to a fresh directory outside AORTA. Run structural checks
with the card flags and a generic skill validator; omit repository-only `--docs`.
The external example links must remain public URLs. Resolve the separate AORTA
installation independently before using the reader.

A reduced instruction-only copy is useful for a bounded independent exercise,
but is not the distributable package or proof of actual catalog discovery.
Keep installed-copy/cross-skill checks, model evaluations and checker qualification
as separate gates.

## Maintenance and AMD intake

- Confirm maintenance/release ownership and the approved license in the [card](../../.agents/skills/rocm-sanitizer-triage/skill-card.md). AORTA declares MIT; the card still records pending skill approval.
- Obtain product/project approval for the public ROCm/aorta source and each integration PR. A public repository does not establish catalog intake approval.
- Keep instructions, helpers, templates, tests and cases consistent when AORTA contracts or supported versions change.
- Deliver the later checker integrations in separate AORTA PRs; review the completed shared package before one combined AMD Skills submission. No Waitcheck-only catalog release is proposed here.
- Before intake, close required qualification, relevant agent evaluations, installed-copy checks and supported-scope decisions. Preserve evidence limits.
- Recheck AMD federation requirements, register the approved source/alias, perform the targeted import and validate the generated copy. Do not directly author replacement imported files or marketplace manifests.

AMD imports a versioned product-repository copy. Fix upstream source, then follow
the catalog update process. Merge, intake approval and catalog publication remain
separate milestones controlled by the responsible owners/maintainers.

## Sources

- [AMD package requirements](https://github.com/amd/skills/blob/main/docs/skill-requirements.md)
- [AMD evaluations](https://github.com/amd/skills/blob/main/docs/evals.md)
- [AMD contribution process](https://github.com/amd/skills/blob/main/CONTRIBUTING.md)
- [Pinned Skillscope authoring](https://github.com/amd/skillscope/blob/v0.1.2/docs/authoring-evals.md)
- [Pinned Skillscope staging](https://github.com/amd/skillscope/blob/68b69e2eff60c41f5d5482bb89f24e07ce4fbebb/skillscope/agent.py)
