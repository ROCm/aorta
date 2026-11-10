# Waitcheck

**Readiness: draft.** Prepare plans and interpret supplied evidence. The recipe
below is a preparation aid, not a qualified launch procedure. Follow the
[common workflow](aorta-workflow.md).

## 1. Question and limits

Inspect final compiled GPU code for supported instruction-wait hazards using
standalone RocJITsu `rj_waitcheck`. The offline scan does not execute the kernel or
require the full simulator. It does not check runtime memory safety, general races
or numerical correctness. A reported hazard does not establish a crash cause.

Choose one verified kernel entry or an explicitly requested whole-object scope.
Never silently substitute a whole-object scan, filename match, other build or demo
workload. Source alone is insufficient; ask for the final compiled object.

## 2. Tested environment

Source contract: AORTA `12e75e694488cd29937f2ee83dccf530c1f0a92a`.
Upstream reference: RocJITsu `d03c6fe410eed62b211d56bb7878dd1f494269fe`.
The adapter contract was also checked at AORTA
`ac11f84616a8bf7cc475e455bfb4a7ae250d4928`. Three development controls have now
run through that installation on Linux x86_64: a scalar missing-wait finding,
its corrected counterpart and a deliberately wrong input digest. See the
[reproducible examples and qualification gaps](waitcheck-examples.md).
This is bounded evidence for gfx1250 and the recorded tool, **not a qualified
general launch route**. Other targets, tool versions and hazard classes are not
established by these controls.

A compatible standalone tool can scan code without a matching physical GPU.
Compiler/ROCm SDK setup matters when creating an object, not inherently when
scanning an existing one. Do not impose ConSan's hook or full emulator setup.
Do not assume Windows binary support or treat an upstream target list as a tested
skill matrix.

## 3. Required inputs

| Input | Establish before preparing a complete recipe |
| --- | --- |
| Code | Final object path, SHA-256 and build/origin; preserve the original |
| Target | Exact architecture supported by the chosen tool/AORTA versions |
| Selection | Exact name, verified entry offset in `.text`, and selected image index |
| Scope | One entry or explicitly requested whole object |
| Tool/host | Separate AORTA installation/revision, native binary path/hash, build provenance and compatible host |
| Bounds | Working directory, fresh output directory, timeout and diagnostic policy |

Offline scanning does not require runtime buffers, tensor values or a GPU launch
command. Retain those only where relevant to build provenance or the wider
investigation. Names and filenames do not establish entry identity.

## 4. Prerequisites and setup

Verify the selected checker's prerequisites:

1. Establish the execution machine and AORTA installation. The skill installs
   neither AORTA nor the checker. This AORTA baseline declares Python >=3.10,
   PyYAML and Click.
2. Resolve `rj_waitcheck`: recipe lookup prefers
   `ROCJITSU_PREBUILT/bin/rj_waitcheck`, then
   `ROCJITSU_BUILD/tools/rj_waitcheck`, then PATH. A broken preferred file does
   not automatically fall back. Record the actual binary/hash and manifest/build
   provenance, not merely an environment-variable setting.
3. Verify object hash, architecture, image and entry. Pinned upstream inspection
   options include `--list-code-objects` and `--list-kernels`. Prepare the
   version-appropriate inspection and obtain required authorization before running
   it. AORTA accepts an entry offset but does not discover it. A symbol value or
   descriptor address must not be guessed to be the `.text` entry offset.
4. Establish output compatibility. This AORTA adapter reads text, not the separate
   experimental exhaustive-JSONL report. It recognizes older
   `diagnostics=>=N` truncation and does not fully interpret newer completion
   semantics. It validates an entry marker when present; absence of that marker
   is not independent proof of exact-entry coverage.
5. Resolve all recipe values. YAML values are literal; the loader does not expand
   environment variables. Syntax acceptance does not verify a real entry or host.

Do not inspect a cluster, acquire tools or run a scan merely to answer a planning
request. Honor applicable approval rules for each action. A saved manifest alone
does not establish that its executable is available today.

## 5. AORTA procedure

Copy the bundled [draft exact-entry recipe](../assets/waitcheck-exact-entry.yaml)
to the task directory. Replace every `REQUIRED_...` value with verified facts.
Index/offset placeholders deliberately fail parsing; do not delete them to bypass
that error. Set the object path relative to the copied recipe or use an absolute
path on the execution host. Retain an explicit digest to bind the plan to that
object; AORTA checks the file against it when scanning.

The recipe selects `source.kind: kernel`, one identity and only `waitcheck`.
Its scope/selection fields satisfy the schema; they do not prove entry coverage.
`consan_policy: strict` is required by this schema even for Waitcheck-only use.

Prepare these invocation shapes with concrete paths; they are not authorization
or instructions to execute this unqualified route:

```text
aorta sweep run --recipe <prepared.yaml> --output <new-plan-output> --dry-run
aorta sweep run --recipe <prepared.yaml> --output <new-scan-output>
```

A dry run reads inputs and writes a `not_checked/dry_run` report without scanning.
It cannot establish tool compatibility, real entry existence, target support or
checker coverage. Real invocation belongs to separately authorized development
qualification, or later skill execution after qualification and agent evaluation.

For explicitly requested whole-object scope, prepare a separately labeled recipe
without `entry_offset`; establish object/image identity and make no per-kernel
attribution. Do not use this as a fallback for an unresolved entry.

Expected output paths for the template:

- `sanitizer_report.json`: `aorta.sanitizer_report/0.1` worklist, checks, verdict
  and execution state.
- `waitcheck/waitcheck-0.log`: first selected scan. Prelaunch failures can have no
  log; multi-selection log numbering can have gaps after deduplication.
- Retain the recipe, original/build identity and tool provenance separately.
  AORTA records binary identity but does not fetch its upstream manifest.

The template proposes 900 seconds per scan and a 32-diagnostic cap. These are
starting bounds, not coverage evidence. This adapter accepts a positive cap, not
zero, and does not request exhaustive JSONL mode. Do not equate these counts with
an uncapped historical experiment.

## 6. Evidence and interpretation

Use a compatible saved reader for reports/logs. Preserve the producing AORTA
status, reason and execution state:

| Evidence | Explain |
| --- | --- |
| `warn` with supported diagnostics | Findings exist; retain their locations and supported hazard class |
| `pass`, verified identity and compatible complete scan, no findings | No supported hazard reported in that scope, not general correctness |
| `not_checked`, missing tool/identity or dry run | Intended scope was not established as checked |
| `error`, timeout, changed digest, launch/parse failure | Failed/incomplete checking; retain any observed findings |

The native tool can exit 4 for hazards; AORTA can exit zero for a complete warning.
Process exit alone is not a clean result. Expose conflicts between summaries and
raw output instead of turning inconsistent evidence into a pass.

Counts are version-dependent. Older `diagnostics=>=N` means a lower bound.
Newer pinned upstream can omit diagnostic details while completing the scan with
an exact observed total; early stopping makes the total a lower bound. The local
AORTA parser/model does not fully distinguish these cases. Preserve its report
and raw output, explain the compatibility gap, and do not infer full coverage from
parser success. Diagnostic records are not necessarily distinct bugs. Matching
warnings across builds do not prove identical code or a crash explanation.

## 7. Required examples and agent cases

The packaged [three-case example](waitcheck-examples.md) supplies two small source
files and generates finding, corrected and digest-mismatch recipes. It is for
explicitly authorized development validation; it does not check the user's kernel
or replace their requested scope with a demo.

Verified outcomes for the recorded inputs:

| Case | AORTA result | Interpretation |
| --- | --- | --- |
| Missing scalar wait | ran / warn; native exit 4; AORTA exit 0 | Supported hazard found despite exit zero |
| Corrected control | ran / pass; native exit 0; AORTA exit 0 | No supported finding in this completed scan |
| Wrong expected digest | error / error; AORTA exit 1; code_object_digest_mismatch | Input identity rejected; no completed scan, not a clean kernel |

In the third case, the check-level reason is `worklist_not_fully_checked`; retain
the more specific per-kernel reason as well. Backend identity in a report is not
proof that the checker launched. Empty findings on a rejected input do not mean pass.

For new builds, establish final-object/build identity again; old entries and hashes
are not defaults.
A realistic application with warnings complements that pair; it does not replace it.
Check exact-entry identity, missing tool/object, wrong target/image/entry, changed
digest, timeout, malformed output and version-specific cap/completion behavior.
Keep synthetic parser tests separate from real checker runs.

Agent cases must cover no-run planning, no silent scope change, findings despite
exit zero, incomplete evidence, and historical direct-tool results versus a newly
qualified AORTA route.

## 8. Qualification record

The [qualification table](waitcheck-examples.md#qualification-gaps) separates
the three verified controls from untested behavior. In particular, a single-kernel
object does not prove isolation from neighboring kernels; the low diagnostic count
does not exercise caps or early stopping. The native binary fingerprint is recorded,
but its full upstream build provenance remains unresolved. Applicable agent
evaluation and release acceptance are still required. Advance readiness only under the
[common rules](aorta-workflow.md#route-readiness).

Sources:
[AORTA recipe](https://github.com/ROCm/aorta/blob/12e75e694488cd29937f2ee83dccf530c1f0a92a/src/aorta/instrumentation/rocjitsu_sanitizers/recipe.py),
[AORTA adapter](https://github.com/ROCm/aorta/blob/12e75e694488cd29937f2ee83dccf530c1f0a92a/src/aorta/instrumentation/rocjitsu_sanitizers/waitcheck.py),
[RocJITsu guide](https://github.com/ROCm/rocm-systems/blob/d03c6fe410eed62b211d56bb7878dd1f494269fe/emulation/rocjitsu/docs/waitcheck/README.md).
