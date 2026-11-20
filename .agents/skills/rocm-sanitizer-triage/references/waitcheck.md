# Waitcheck

**Readiness: expanded route under development.** Help prepare builds, recipes and
explicitly authorized development checks; interpret saved evidence. No general
execution profile is qualified yet. Follow the
[common workflow](aorta-workflow.md).

## 1. Question and limits

Inspect final compiled GPU code for supported instruction-wait hazards using
standalone RocJITsu `rj_waitcheck`. The offline scan does not execute the kernel or
require the full simulator. It does not check runtime memory safety, general races
or numerical correctness. A reported hazard does not establish a crash cause.

Choose one verified kernel entry or an explicitly requested whole-object scope.
Never silently substitute a whole-object scan, filename match, other build or demo
workload. Source is a valid starting point: help build it using
[source preparation](waitcheck-build.md) if a usable final object is absent.

## 2. Tested environment

For the offline report reader, use AORTA commit
`90fe1d8a440f4c88dc8187e5a52a52dc4385c3bb` as the reference installation.
Its report models and native-log parser expose diagnostic summaries. Older
saved reports can still be read with that installation.

The historical recipe and adapter baseline was AORTA
`12e75e694488cd29937f2ee83dccf530c1f0a92a`. Its Python modules lack the
diagnostic summary support required by the current reader.
Upstream reference: RocJITsu `d03c6fe410eed62b211d56bb7878dd1f494269fe`.
The adapter contract was also checked at AORTA
`ac11f84616a8bf7cc475e455bfb4a7ae250d4928`. Three development controls have now
run through that installation on Linux x86_64: a scalar missing-wait finding,
its corrected counterpart and a deliberately wrong input digest. See the
[reproducible examples and qualification gaps](https://github.com/ROCm/aorta/blob/90fe1d8a440f4c88dc8187e5a52a52dc4385c3bb/examples/sanitizers/waitcheck/docs/waitcheck-examples.md).
This is bounded evidence for gfx1250 and the recorded tool, **not a qualified
general launch route**. Other targets, tool versions and hazard classes are not
established by these controls.

A compatible standalone tool can scan code without a matching physical GPU.
Compiler/ROCm SDK setup matters when creating an object, not inherently when
scanning an existing one. Do not impose ConSan's hook or full emulator setup.
Do not assume Windows binary support or treat an upstream target list as a tested
skill matrix.

### Architecture selection

Record three facts independently: host GPU (if present), actual object/build
target, and targets supported by the selected checker. On MI350 a new native
build is expected to use gfx950/wave64, subject to verification. This does not
convert an existing gfx1250 object or prohibit offline scanning it on that host.
Do not infer the code target from the machine name or silently use a compiler's
gfx1250/wave32 default.

Separate target policy from tested coverage. Later development checks scanned
selected gfx950 entries on an MI300 host. Those results apply to the recorded
objects and tool; MI350-host validation and general execution qualification
remain open. Verify each new object and tool combination.

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
2. [Obtain or verify the checker bundle](waitcheck-provision.md). Recipe lookup prefers
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
   experimental exhaustive-JSONL report. It preserves older `diagnostics=>=N`
   lower bounds and recognizes explicit omitted-detail/early-stop text signals
   from the pinned source. Bare counts have unknown native completion. It validates
   an entry marker when present; absence of that marker
   is not independent proof of exact-entry coverage.
5. Resolve all recipe values. YAML values are literal; the loader does not expand
   environment variables. Syntax acceptance does not verify a real entry or host.

Do not inspect a cluster, acquire tools or run a scan merely to answer a planning
request. Honor applicable approval rules for each action. A saved manifest alone
does not establish that its executable is available today.

## 5. AORTA procedure

For the complete user-input handoff, provenance record, bounded command shapes
and offline evidence reader, use the [single-entry procedure](waitcheck-procedure.md).
Its command shapes can support an explicitly authorized development/qualification
task. Recorded controls do not qualify arbitrary user execution.

The procedure owns recipe preparation, dry-run/run command shapes and retained
evidence. Its [exact-entry template](../assets/waitcheck-exact-entry.yaml) selects
one identity and only `waitcheck`; `consan_policy: strict` is a required schema
field, not a request to run ConSan. Unresolved image/entry placeholders fail
parsing intentionally. A dry-run `not_checked/dry_run` result does not establish
tool compatibility, real entry existence or checking coverage.

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

The template proposes 120 seconds per scan and a 64-diagnostic cap. These are
starting bounds, not coverage evidence. This adapter accepts a positive cap, not
zero, and does not request exhaustive JSONL mode. Do not equate these counts with
an uncapped historical experiment.

## 6. Evidence and interpretation

Use a compatible saved reader for reports/logs. Preserve the producing AORTA
status, reason and execution state:

The [single-entry report reader](waitcheck-procedure.md#4-read-the-saved-report-and-raw-log)
summarizes a user-supplied AORTA report and optional raw log without executing a
checker. Its successful exit means the summary was written, not that the kernel
passed. Identity and provenance still need the producing run's external evidence.

| Evidence | Explain |
| --- | --- |
| `warn` with supported diagnostics | Findings exist; retain their locations and supported hazard class |
| `pass`, verified identity and compatible complete scan, no findings | No supported hazard reported in that scope, not general correctness |
| `not_checked`, missing tool/identity or dry run | Intended scope was not established as checked |
| `error`, timeout, changed digest, launch/parse failure | Failed/incomplete checking; retain any observed findings |

The native tool can exit 4 for hazards; AORTA can exit zero for a complete warning.
Process exit alone is not a clean result. Expose conflicts between summaries and
raw output instead of turning inconsistent evidence into a pass.

When a user brings only a report, explain the reported dependency in plain
language. For a load/use warning, identify the instruction that starts the load,
the instruction that needs the value, and the wait named in the diagnostic.
Use the locations in this report. If it names only the use, leave the producer
unknown until matching disassembly or a fuller record identifies it.
The report alone does not give a source
line, show which path ran on a GPU, or establish a wrong numerical answer.
For a source-level investigation, request the matching build and its address
mapping. For a comparison, check whether the same load/use warning appears
in the other build, allowing for changed addresses.

If the original already warns, record each reported load and use before
comparing builds. Match the instructions and their dependency in the selected
entry of each build. A rebuild can reuse the old register for a different load.
Trace where the loaded value is used, even when its register name changes.
Say which warnings are present in the available records.
Identify findings shared by both builds, findings seen only in one, and any
whose correspondence is uncertain. Preserve unexpected outcomes.
Explain the outcome at the edited instruction even when the verdict stays the same.
If details were omitted, coverage is unclear or the instructions cannot be
matched, leave disappearance unresolved. Two reports can each contain one warning and
point to different operations. A separate kernel stage with zero diagnostics
leaves the first stage's warning open.

Counts are version-dependent. Optional `diagnostic_summary` separates reported
count/meaning, emitted records before deduplication, requested detail limit, detail
completeness and native analysis completion. Missing metadata stays unknown.
Older `diagnostics=>=N` is a lower bound with unknown native completion. The
historical `diagnostics_truncated` Boolean retains that marker; false is no proof
of complete details. Newer `omitted N diagnostic(s) after limit` supports an exact
observed count, incomplete details and completed native analysis in that format.
`omitted at least N diagnostic(s) after limit` or the supported early-stop sentence
establishes early termination. Bare numeric summaries, cap equality and exit0/4
alone establish neither exhaustive analysis nor complete details.

Compare the summary count with raw diagnostic records, then separately count
retained unique findings. Deduplication alone is not omitted output. Unexplained
raw count/record differences remain a compatibility gap (reader2), not proof of
corruption or a checker crash. Preserve the warning and all counts. Look for
fuller evidence for the same object digest, target, image and entry. If it comes
from another run, identify that run and verify comparable tool, settings and
scope before comparing. Keep each result attributed to its own run. Recommend
a separately authorized higher-limit run only if details are still needed.
Diagnostic records are not necessarily distinct bugs. Matching warnings across
builds do not prove identical code or a crash explanation. AORTA execution
`complete` describes adapter execution, separately from native analysis.

## 7. Required examples and agent cases

The repository [three-case example](https://github.com/ROCm/aorta/blob/90fe1d8a440f4c88dc8187e5a52a52dc4385c3bb/examples/sanitizers/waitcheck/docs/waitcheck-examples.md) supplies two small source
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

The [qualification table](https://github.com/ROCm/aorta/blob/90fe1d8a440f4c88dc8187e5a52a52dc4385c3bb/examples/sanitizers/waitcheck/docs/waitcheck-examples.md#qualification-gaps) separates
the verified controls from untested behavior. A separate
[two-kernel development pair](https://github.com/ROCm/aorta/blob/90fe1d8a440f4c88dc8187e5a52a52dc4385c3bb/examples/sanitizers/waitcheck/docs/waitcheck-isolation.md#recorded-development-result-10-november-2026)
now supports exact-entry isolation for its recorded build and tool: the faulty
entry warns while its corrected neighbor passes. This is not an arbitrary-kernel
or multi-entry-worklist qualification. A
[three-diagnostic limit comparison](https://github.com/ROCm/aorta/blob/90fe1d8a440f4c88dc8187e5a52a52dc4385c3bb/examples/sanitizers/waitcheck/docs/waitcheck-examples.md#recorded-diagnostic-limit-result)
originally stopped at a raw-count/retained-detail gate at limit1; the updated reader
now explains the saved explicit omission signal. Broader cap/completion semantics
and early stopping are not natively qualified. The native binary
fingerprint is recorded,
but its full upstream build provenance remains unresolved. Applicable agent
evaluation and release acceptance are still required. Advance readiness only under the
[common rules](aorta-workflow.md#route-readiness).

The [recorded attention result](https://github.com/ROCm/aorta/blob/90fe1d8a440f4c88dc8187e5a52a52dc4385c3bb/examples/sanitizers/waitcheck/docs/waitcheck-examples.md#recorded-attention-kernel-result)
adds one independently verified real-application AORTA warning with an explicit
omission signal, plus accepted agent interpretation content. It preserves the
same exact-entry and provenance boundaries; it does not qualify arbitrary kernels
or remove the remaining agent-activity and release gates.

Sources:
[AORTA recipe](https://github.com/ROCm/aorta/blob/12e75e694488cd29937f2ee83dccf530c1f0a92a/src/aorta/instrumentation/rocjitsu_sanitizers/recipe.py),
[AORTA adapter](https://github.com/ROCm/aorta/blob/12e75e694488cd29937f2ee83dccf530c1f0a92a/src/aorta/instrumentation/rocjitsu_sanitizers/waitcheck.py),
[RocJITsu guide](https://github.com/ROCm/rocm-systems/blob/d03c6fe410eed62b211d56bb7878dd1f494269fe/emulation/rocjitsu/docs/waitcheck/README.md).
