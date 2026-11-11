# Maintaining the AORTA sanitizer skill

This is the engineer handoff for the shared foundation in
[rocm-sanitizer-triage](../../.agents/skills/rocm-sanitizer-triage/SKILL.md).
The skill teaches an agent to choose a check, collect inputs, prepare an AORTA
invocation and explain evidence. AORTA invokes the checkers. The user's program
does not have to become a built-in AORTA workload.

## Shared requirement checklist

Use this map to review the base before specialist handoff. These are observable
acceptance expectations, not a record that evaluations have passed. Keep runtime
instructions in the linked skill files; keep test expectations and ownership here.

| Shared piece | Instruction location | Observable acceptance check |
| --- | --- | --- |
| Skill selection | [SKILL.md: Choose the question](../../.agents/skills/rocm-sanitizer-triage/SKILL.md#choose-the-question) and its description | Select for AORTA sanitizer planning/results; do not turn unrelated output validation, installation or tuning into a sanitizer task. |
| Checker selection | [SKILL.md: Choose the question](../../.agents/skills/rocm-sanitizer-triage/SKILL.md#choose-the-question) and the selected reference's question/limits | Explain why one checker fits; clarify ambiguous symptoms or an unknown report producer; leave unsupported questions unresolved rather than running all checkers. |
| Input collection | [Common workflow: Inputs and task mode](../../.agents/skills/rocm-sanitizer-triage/references/aorta-workflow.md#inputs-and-task-mode) | Reuse supplied facts and ask only for inputs that affect the next step. Distinguish a plan's unknowns from execution blockers and a past run from the current machine. |
| Common workflow | [SKILL.md: Follow the common workflow](../../.agents/skills/rocm-sanitizer-triage/SKILL.md#follow-the-common-workflow) and [invocation preparation](../../.agents/skills/rocm-sanitizer-triage/references/aorta-workflow.md#verify-prerequisites-and-prepare-the-invocation) | Verify the selected checker's prerequisites from its reference. For plans, preserve gaps; for run requests, enforce readiness and authorization; for evidence-only requests, interpret without unnecessary setup or reproduction. |
| Common answer and evidence | [Read evidence](../../.agents/skills/rocm-sanitizer-triage/references/aorta-workflow.md#read-evidence) and [Common answer](../../.agents/skills/rocm-sanitizer-triage/references/aorta-workflow.md#common-answer) | State checked/proposed scope, result, evidence, limitations and next action. Preserve status/reason, separate findings from coverage, and expose conflicting or mismatched artifacts. |
| Evaluation conventions | [Evaluation conventions](#evaluation-conventions) and [Checks and CI](#checks-and-ci) | Cases grade decisions rather than exact phrasing; retain case-level outputs and versions; separate structural passes, model behavior and real checker qualification. |
| Maintenance and intake | [Ownership](#ownership-and-file-boundaries), [Checker contribution](#checker-contribution) and [Maintenance and AMD intake](#maintenance-and-amd-intake) | Engineers have bounded file ownership and one shared integration path; pending owner/license/intake decisions remain explicit; readiness advances only with the required evidence. |

The shared-base milestone does not require filling checker-specific commands or
qualifying hardware routes. Those are specialist deliverables. Test the shared
rules using planning requests and supplied text; do not claim raw-report parsing
support while the required compatible reader is unavailable.

## Ownership and file boundaries

| Shared owner | Checker owner |
| --- | --- |
| Skill description, selection, common workflow and answer | One checker reference and its tested procedure |
| Single evaluation dataset, CI and review integration | Small public fixtures and finding/clean/incomplete cases |
| Card, license confirmation and AMD intake | Supported versions, prerequisites, limits and qualification evidence |

The shared owner integrates changes to `SKILL.md`, `references/aorta-workflow.md`,
`skill-card.md`, `evals/evals.json` and the CI workflow. Checker engineers work in
`references/<checker>.md` and `evals/fixtures/<checker>/`, where checker is
`waitcheck`, `asan`, `consan` or `fpsan`. They propose dataset additions in their
PR; the shared owner resolves concurrent edits. No second evaluation format or
custom runner is needed. Agree on a reviewed base commit before parallel branches.

## Checker contribution

One person may own both shared and checker work. Keep one checker stage active,
reuse the same contribution/validation gates and obtain independent review; a
handoff to additional engineers is not a prerequisite.

The Waitcheck draft now includes a portable
[exact-entry recipe asset](../../.agents/skills/rocm-sanitizer-triage/assets/waitcheck-exact-entry.yaml).
Its [three development examples](../../.agents/skills/rocm-sanitizer-triage/references/waitcheck-examples.md)
package two attributed HIP sources, a portable helper and generated recipes for
finding, corrected and digest-refusal outcomes. The helper's focused tests are
`tests/instrumentation/rocjitsu_sanitizers/test_waitcheck_controls_example.py`;
run these locally without a GPU or native checker. A dedicated skill CI workflow
exists at `.github/workflows/sanitizer-skill-checks.yml`; its definition is not
evidence that GitHub or model evaluations passed. Local tests are distinct from real node
validation. The reference's qualification table records remaining gaps.
It intentionally rejects unresolved index/offset values. Its local contract test is
`tests/instrumentation/rocjitsu_sanitizers/test_waitcheck_skill_recipe.py`:
`python -m pytest tests/instrumentation/rocjitsu_sanitizers/test_waitcheck_skill_recipe.py`.
That test only loads recipes with inert temporary data; it does not validate a
code object, launch a checker or qualify the route. The runtime reference records
the older AORTA/newer upstream diagnostic-count compatibility gap. Keep readiness
draft until the selected real procedure, controls and agent cases are validated.

Use the sections already present in the checker reference. Supply:

1. The checking question, supported input and limits.
2. Tested AORTA, checker, compiler/runtime, OS and target identities.
3. Required program/kernel, input and build information.
4. Checker-specific prerequisites and how to verify them.
5. A reviewed AORTA recipe or existing supported command, including output location
   and a bounded run. Identify example placeholders; never describe them as tested.
6. How to interpret the existing report, raw evidence, execution and coverage.
7. A known finding, matched clean control and incomplete/failed-check example.
8. A qualification record: exact revisions, commands, environment, expected and
   observed outcomes, artifact locations, date and reviewer.

Keep prerequisites in the checker reference. The common instruction delegates to
the selected checker. Do not introduce a universal RocJITsu, instrumented-build or
GPU-execution requirement. Do not add a new AORTA runner or report schema here.

Each route advances independently: **draft → procedure reviewed → checker
qualified → agent evaluated**. Record the evidence and supported environment in
the reference before updating the common readiness table. Procedure review alone
does not authorize an unqualified execution claim. All four routes remain in scope.

## Evaluation conventions

Maintain one [Skillscope dataset](../../.agents/skills/rocm-sanitizer-triage/evals/evals.json).
Use IDs `aorta-sanitizer-<checker>-<scenario>` for new checker cases and
`aorta-sanitizer-shared-<scenario>` for shared behavior. Preserve existing IDs
unless a case is intentionally retired. Prompts describe realistic user requests;
expected behavior describes observable decisions, not exact wording or headings.

```json
{
  "id": "aorta-sanitizer-asan-missing-instrumentation",
  "prompt": "Plan an AORTA memory check for my ordinary HIP release binary. Do not run commands.",
  "skill_should_trigger": true,
  "expected_behavior": [
    "Identifies ASan as the relevant route and asks about its required instrumentation and runtime",
    "Returns a conditional plan without claiming the binary was checked"
  ],
  "unexpected_behavior": ["Treats exit zero as proof of ASan coverage"]
}
```

Use at least three positive and two negative cases, with at least two positive
behavior cases. The two-behavior minimum follows our intake review and is stricter
than the current public minimum. This is a floor, not complete four-checker coverage.
The shared base exercises planning for all four, unclear requests, evidence limits,
authorization and close negatives. Checker owners add finding/clean/incomplete
cases backed by actual qualification records; the current cases do not replace them.

Engineers can use the [synthetic worked answer](#worked-answer-for-engineers)
to learn the common response format. Keep that teaching answer out of agent-facing
references and evaluation inputs. The corresponding prompt supplies only the
scenario; the expected answer is reviewer material. Use distinct held-out examples
when assessing generalization beyond these development cases.

Skillscope v0.1.2 copies the **whole skill folder** into both routing and behavioral
workspaces. The required `evals/evals.json` therefore remains readable by the agent;
these runs are development evaluations, not proof of blind or held-out performance.
Removing a link does not hide a file. Keep teaching answers outside the distributed
skill folder, and record this harness limitation when reporting results. See the
pinned [behavioral staging](https://github.com/amd/skillscope/blob/68b69e2eff60c41f5d5482bb89f24e07ce4fbebb/skillscope/agent.py)
and [routing staging](https://github.com/amd/skillscope/blob/68b69e2eff60c41f5d5482bb89f24e07ce4fbebb/skillscope/routing.py).

A negative means **no skill in the tested routing room should activate**. Our
single-skill CI room contains only this sanitizer skill. When evaluating alongside
other skills, move tasks owned by another skill to that skill's positive dataset
instead of treating them as universal negatives. Test competition with AORTA's
existing `verify-run-output` skill after it has a compatible evaluation dataset.

Store small, public fixtures under `evals/fixtures/<checker>/`. Include a provenance
note with origin, revisions, license, how to reproduce, expected result and limits.
Label synthetic teaching inputs explicitly. Do not ship proprietary kernels,
credentials or internal paths. Prefer small textual excerpts; catalog imports
exclude files over 100 KB. For raw JSON, CSV or log analysis, use an existing saved,
reusable reader or add a reviewed helper with a reproduction command. Do not infer
facts through disposable inline parsing.

Prompts currently carry all necessary evidence as supplied text. The pinned
harness copies fixtures with the skill, but file-dependent cases must establish
the path and working directory the agent should use. Verify that arrangement
before adding such a case. A future
hardware case must declare its environment and authorized setup separately; adding
it to today's CPU-only dataset without changing CI is not a valid handoff.

## Checks and CI

The shared workflow is included at `.github/workflows/sanitizer-skill-checks.yml`.
It pins Skillscope v0.1.2 at `68b69e2eff60c41f5d5482bb89f24e07ce4fbebb`, configures
CPU helper/recipe tests and structural checks, and provides explicit manual
planning model evaluations. The table describes the configured scope. Verify
actual GitHub results, evaluation credentials and model runs separately for the
revision being reviewed; workflow configuration alone establishes no passing run.

| Check | When and what it proves |
| --- | --- |
| Structural and governance-card sections | Matching PRs and manual dispatch; validates format, datasets and links, not owner approval or correctness |
| Routing and behavior | Explicit manual dispatch with `run_agent_evals: true` and configured `ANTHROPIC_API_KEY`; model-based, single-skill, planning-only evaluation |
| Checker qualification | Separately approved runs with real tools and recorded artifacts; not performed by this workflow |
| Cross-skill and installed-copy checks | Required integration work before intake; not covered by the single-skill workflow |

PR checks receive no model secret. Manual model checks may spend API credits and
must use reviewed code and cases. No self-hosted/GPU runner or cluster job is wired
here. Follow the user's authorization requirements for execution; in this workspace
every cluster command, including inspection, requires explicit approval.

To reproduce the structural check from a local checkout without installing a new
package globally (Python 3.10+ and PyYAML are required): clone Skillscope at the
pinned revision into a separate directory, set `PYTHONPATH` to that checkout, then
run from the AORTA root:

```text
python -m skillscope structural --skills-dir .agents/skills/rocm-sanitizer-triage --skill-files skill-card.md --skill-sections Description,Owner,License --docs docs/sanitizers/skill-maintenance.md,docs/sanitizers/waitcheck-reporting.md
```

Retain the source commit, harness version, command, output, tested case IDs, agent
CLI version, model and routing room where applicable. The pinned action installs
the current Claude CLI for model checks, so the harness pin alone does not pin the
whole evaluation environment. A format pass is not an agent behavior pass.
Review new or modified scripts independently, including how related scripts work
together, then fix and verify findings. No additional execution helper is needed
for the planning foundation.

### Local portability check

Use a fresh directory outside the AORTA checkout and record the source revision
plus uncommitted changes. Preserve separate copies for separate questions:

1. **Full package:** copy the skill directory unchanged; from the new parent
   directory run the same structural command with `--skills-dir
   rocm-sanitizer-triage` and the card flags. Omit `--docs`, which refers to
   repository-only engineer documentation. Also run the existing generic skill
   validator. Record the actual Python version; Skillscope v0.1.2 requires 3.10+.
2. **Instruction-only exercise:** copy `SKILL.md` and its runtime references to
   another fresh directory, excluding evaluations and engineer answers. Give an
   independent agent a new realistic request and only these files. Record its
   response and the files it used. This deliberately reduced copy is a test setup,
   not the distributable AMD package.

Both checks are local simulations. They do not establish catalog installation,
automatic discovery among installed skills, AORTA execution or checker coverage.
Keep full model routing/behavior and catalog validation as separate gates. Do not
replace a required evaluation dataset with the reduced exercise copy.

## Worked answer for engineers

This is a **synthetic teaching example** for the common answer format. No GPU or
AORTA run produced it. It is neither a raw report fixture nor qualification evidence.
It stays outside the shipped skill so ordinary skill loading does not expose it.

The inline development case `aorta-sanitizer-shared-incomplete-coverage` uses this
scenario: intended HIP kernel `store_index`, input `n=64`, Linux on an AMD GPU;
exact GPU/build/runtime versions unknown; exit `0` and no memory-error finding;
AORTA status `not_checked`, reason `asan_dispatch_unverified`; no dispatch log.
The user asks for interpretation without running commands.

1. **What was checked:** The supplied summary describes an intended ASan check of
   `store_index` with `n=64`. It does not establish that the kernel ran. I have not
   run or reproduced this check.
2. **Result:** Preserve `not_checked` and `asan_dispatch_unverified`. No memory-error
   finding is reported, but that is not a clean result for the selected kernel.
3. **Evidence:** Only the supplied synthetic text is available: exit `0`, the
   status/reason and missing dispatch log. No raw report or artifact path was supplied.
4. **Limitations:** Execution and checking coverage are unconfirmed. Build/runtime/
   GPU details are unknown. This cannot establish memory safety or cover other inputs.
5. **Next action:** Obtain the original report and target dispatch evidence and
   establish the selected checker's prerequisites. If a new run is needed, prepare
   a qualified procedure and follow the user's execution-approval rules first.

Do not provide this answer to the evaluated agent. Grade the decisions, not exact
words. The prompt is self-contained and requires no fixture staging or hardware.
Real finding/clean/incomplete-check controls remain later specialist deliverables.

## Maintenance and AMD intake

- Confirm accountable engineering/maintenance and product release owners in the
  [card](../../.agents/skills/rocm-sanitizer-triage/skill-card.md). Pending text can
  pass a section check; it is not an ownership decision.
- Confirm the approved distribution license. AORTA declares MIT, but the card's
  pending decision must be resolved by the owners before distribution.
- Obtain product/project approval for the public `ROCm/aorta` source. A public
  repository does not itself establish AMD intake approval.
- Review behavior whenever AORTA's recipe/report contract, checker prerequisites
  or supported versions change. Update the relevant reference and cases together.
- Test an installed copy outside the source checkout. Locate the AORTA executable
  or explicit checkout independently; skill-relative files must stay portable.
- Before submission, collect real qualification and passing agent evaluation
  records for every advertised execution route. Keep unfinished routes explicit.
- Recheck current AMD contribution guidance, register the agreed source and alias,
  perform a targeted import using `--only`, validate the generated copy and prepare
  the catalog PR. Do not hand-edit imported files or generated manifests. Catalog
  publication and product approval remain owner/maintainer decisions.

AMD imports a versioned copy from the public product repository; agents do not
read the latest AORTA branch on every request. Fix the source, then use the catalog's
update process. Do not include internal acceptance documents or this project's
private design folder in the shipped skill.

## Sources

- [AMD package requirements](https://github.com/amd/skills/blob/main/docs/skill-requirements.md)
- [AMD evaluations](https://github.com/amd/skills/blob/main/docs/evals.md)
- [AMD contribution process](https://github.com/amd/skills/blob/main/CONTRIBUTING.md)
- [Pinned Skillscope evaluation authoring](https://github.com/amd/skillscope/blob/v0.1.2/docs/authoring-evals.md)
- [Pinned Skillscope action](https://github.com/amd/skillscope/blob/v0.1.2/action.yml)
- [Pinned reusable workflow](https://github.com/amd/skillscope/blob/v0.1.2/.github/workflows/reusable.yml)
