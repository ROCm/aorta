# Waitcheck: three development examples

Use this guide to reproduce the small controls or interpret their outcomes.
These are **development validation examples**; the skill route remains draft.
Obtain the authorization required by the user's environment before node commands.
Do not substitute these examples for checking the user's own kernel.

## What the examples establish

| Case | Construction | Expected result |
| --- | --- | --- |
| Finding | Scalar load consumed without the required wait | AORTA ran/warn; one supported hazard; native exit 4, AORTA exit 0 |
| No finding | Same kernel with `s_wait_kmcnt 0` before consumption | AORTA ran/pass; zero diagnostics; native and AORTA exits 0 |
| Refusal | Corrected binary, deliberately wrong expected fingerprint | AORTA error/error; CLI exit 1; kernel reason code_object_digest_mismatch |

The refusal reuses the corrected binary; it needs no third kernel. Empty findings
mean different things in the last two rows. Read execution state and reason before
explaining whether any checking completed.

Source files and attribution are in [assets/waitcheck-controls](../assets/waitcheck-controls/).
They adapt the HIP example in the
[pinned RocJITsu guide](https://github.com/ROCm/rocm-systems/blob/d03c6fe410eed62b211d56bb7878dd1f494269fe/emulation/rocjitsu/docs/waitcheck/README.md).
The corrected counterpart adds the scalar wait. The upstream MIT notice accompanies
the sources. Their host stub exists to build a HIP executable for inspection:
**never execute the compiled programs.**

## Requirements and build

- Linux with an installed AORTA sanitizer interface and Python 3.10 or newer.
- A compatible standalone `rj_waitcheck` and its recorded fingerprint/build origin.
- For rebuilding only: an AMD Clang/HIP SDK supporting gfx1250.
- Explicitly verified image indexes and kernel entry offsets for each compiled file.

This example targets **gfx1250**, independent of the host GPU. Offline Waitcheck
does not dispatch the kernels. Do not change the target to match the host GPU.
The helper does not build, install tools, discover entries or authorize execution.

Set paths for the local installation, using a new work directory:

```bash
EXAMPLES=/absolute/path/to/aorta/examples/sanitizers/waitcheck
WORK=/absolute/path/to/new-waitcheck-example
AMDCLANG=/absolute/path/to/amdclang++
WAITCHECK=/absolute/path/to/bundle/bin/rj_waitcheck
(
set -eu
mkdir "$WORK"
"$AMDCLANG" -O2 -x hip --offload-arch=gfx1250 "$EXAMPLES/assets/waitcheck-controls/missing_wait.hip" -o "$WORK/missing_wait"
"$AMDCLANG" -O2 -x hip --offload-arch=gfx1250 "$EXAMPLES/assets/waitcheck-controls/corrected_wait.hip" -o "$WORK/corrected_wait"
"$WAITCHECK" "$WORK/missing_wait" --list-code-objects
"$WAITCHECK" "$WORK/missing_wait" --target gfx1250 --list-kernels
"$WAITCHECK" "$WORK/corrected_wait" --list-code-objects
"$WAITCHECK" "$WORK/corrected_wait" --target gfx1250 --list-kernels
)
```

Save compiler identity, build commands and listing output with the run. The
expected kernel symbol for these sources is `_Z17waitcheck_controlv`. Establish
its actual image and `.text` entry separately for each file. An earlier build used
image 0 / entry 0; do not assume that for a new build. A descriptor address is not
the entry offset.

## Prepare, run and verify

The [portable helper](../waitcheck_controls.py) keeps generated recipes,
input snapshots, command records, reports and logs outside the installed skill.
It requires fresh output directories. It validates this small three-case example,
not arbitrary kernel results. Unexpected findings or output are failures of the
example's expectations and must be investigated, not silently ignored.

Run its `--help` from the Python environment containing AORTA. The prepare stage
requires both compiled files, kernel name, independently verified image/entry
selectors and a build identifier. JSON recipes are accepted by AORTA and preserve
literal paths; no shell variables are left inside them. The mismatch recipe changes
only the expected fingerprint and descriptive case metadata.

The run stage invokes only `aorta sweep run`, with the chosen prebuilt tool,
120-second per-scan timeout and a 64-diagnostic cap. It retains every result,
including the expected nonzero refusal. Verification checks report state, actual
findings, scope, tool/input identity and raw evidence; CLI exit alone is insufficient.
The verify stage reads retained evidence without rerunning AORTA or Waitcheck.

After recording the listings, set `MISSING_INDEX`, `MISSING_ENTRY`,
`CORRECTED_INDEX` and `CORRECTED_ENTRY` to their verified values (offsets may use
hexadecimal notation). Set `BUILD_ID` to the retained compiler/build record's
identifier, `AORTA` to the absolute CLI executable, `PREBUILT` to the bundle
root containing `bin/rj_waitcheck`, and `TOOL_SHA256` to its verified hash.
Use Python from the AORTA environment:

```bash
python "$EXAMPLES/waitcheck_controls.py" prepare \
  --missing-object "$WORK/missing_wait" --corrected-object "$WORK/corrected_wait" \
  --kernel-name _Z17waitcheck_controlv --build-id "$BUILD_ID" \
  --missing-index "$MISSING_INDEX" --missing-entry "$MISSING_ENTRY" \
  --corrected-index "$CORRECTED_INDEX" --corrected-entry "$CORRECTED_ENTRY" \
  --output "$WORK/prepared"
python "$EXAMPLES/waitcheck_controls.py" run \
  --prepared "$WORK/prepared" --aorta "$AORTA" --prebuilt "$PREBUILT" \
  --tool-sha256 "$TOOL_SHA256" --output "$WORK/run"
python "$EXAMPLES/waitcheck_controls.py" verify --run "$WORK/run"
```

The run helper's successful comparison includes an expected error in the
digest-mismatch case. Preserve the three individual AORTA exits and reports;
do not describe all three inputs as passing sanitizer checks.

## Recorded development baseline

On 10 November 2026 the original three controls were independently reviewed after
real AORTA execution. Linux x86_64, AORTA
`ac11f84616a8bf7cc475e455bfb4a7ae250d4928`
(`amd-aorta 0.3.1.dev336+gac11f846`), AMD Clang 22.0.0git from ROCm 7.2.4,
target gfx1250, image 0 / entry 0 for both inputs.

| Artifact | Recorded SHA-256 |
| --- | --- |
| Waitcheck binary | 2c406bcc3be32c48ae0ac46d0ca197a7b9a6173e0d25f5b5226fa972d8e31e77 |
| Missing-wait compiled input | 0783c0633fd74c3b9194bf04df8110a51a34dee56d6ec4080461fe0f92d689a4 |
| Corrected compiled input | 9629a61bd78d16d337b8f96ce0fabe57503251efd655aaefec2f59184dc92a1a |

Direct and AORTA scanner output agreed for the finding and corrected controls.
Raw counts were 3 instructions / 2 memory events / 1 diagnostic and 4 / 2 / 0.
The refusal returned `worklist_not_fully_checked` and
`code_object_digest_mismatch`, with no native return code or scanner log.
The adapter rejects that fingerprint before its process executor; no independent
process tracing was performed.

These fingerprints identify the tested artifacts; rebuilding can produce different
hashes. The tool's complete upstream build provenance is not established by its
fingerprint. Private machine paths and full development captures are intentionally
not shipped. Reproduction generates fresh evidence on the user's installation.

The packaged helper was subsequently copied outside the repository and exercised
on the same Linux installation using the retained, entry-verified binaries.
Prepare, run and verify succeeded for all three expected outcomes. Downloaded
records also passed offline verification on Windows. This validates the packaged
workflow and relocated evidence handling; it does not claim a fresh source rebuild
or native Windows scanner support.

## Recorded diagnostic-limit result

On 11 November 2026, the separate
[multi-hazard control](../assets/waitcheck-controls/multiple_missing_waits.hip)
was scanned through AORTA for gfx1250, image 0 / entry 0. Its recorded object
SHA-256 is `37beb79e69cb835178bce70666ca874bf50dce09f0bb5c7e2b19d2fd62ba4b75`.
The native tool fingerprint is the same as the baseline above. Captured
input/tool/helper/runtime fingerprints agreed before and after each call.

| Diagnostic limit | Raw summary count | Retained details | Report truncation | Native / AORTA exit | Original frozen reader |
| --- | --- | --- | --- | --- | --- |
| 64 | 3 | 3 | false | 4 / 0 | Compatible warning evidence for this control |
| 1 | 3 | 1 | false | 4 / 0 | Count/detail compatibility gap; reader exit 2 |

Both reports say `ran/warn`, execution `complete`. Neither raw summary uses the
older `>=` marker. At limit 1, a false truncation field therefore does not establish
that all diagnostic details were retained. Preserve the warning, raw count and
stored-detail count separately; exact native completion/count semantics need
version-appropriate qualification. These counts are not a count of distinct bugs.

The updated offline reader now recognizes the low-cap log's terminal
`omitted 2 diagnostic(s) after limit`. The old count gate stopped before considering
that line. In the pinned supported format it explains an exact observed count3,
one emitted detail, incomplete details and reported completed native analysis;
reader0 means that summary was written. The high-cap log has no explicit terminal
signal: its three records account for its reported3, while count meaning and native
completion remain unknown. A bare count/detail mismatch without a supported signal
still returns reader2. The original reports, frozen reader outputs and stopped
capture remain unchanged. This local interpretation does not prove binary build
provenance or general checker coverage. No rescan is needed: cap64 already has
the control's three details.

The bounded capture stopped at its strict raw-count/detail gate. Four dry runs
completed, but native invalid-image and invalid-entry scans were not reached.
A dry run does not establish native selector rejection. The capture helper's
exit 1 records the evidence-validation stop, not a demonstrated checker crash.
The overall execution route remains draft; this is bounded development evidence.

## Subsequent updated-report and selector results

On 11 November 2026, separate bounded development runs exercised the updated
AORTA producer and reader on the same Linux host and recorded native binary.
The original stopped capture above remains unchanged. Actual saved captures
were replayed independently with matching source/import/input/tool identities
and before/after receipts.

For the multi-hazard object above, cap64 and cap1 retained warning/native4/AORTA0.
New per-entry diagnostic_summary metadata records requested limits64/1, reported
counts3/3 and emitted records3/1. Cap64 leaves count meaning, detail completeness
and native completion unknown; its conservative check aggregate count is null.
Cap1 records exact observed count3, incomplete details and reported completed
native analysis from the explicit omitted2 signal. Both readers exit0; neither
means a kernel pass. These are newly produced report results, beyond replay of
the legacy reports alone. Broader formats and complete binary provenance remain
unestablished.

The separate selector check reused the shared-image isolation object,
SHA-256 `c81c3018816287dc08323aa12a70b1c0d27206f48a61c73136faea57d8d2bc1f`,
with accepted inventory image0/entries0 and256:

| Deliberate selection | Native message | Native / AORTA exit |
| --- | --- | --- |
| Image1, entry256 | failed to select gfx1250 code object 1 | 2 / 1 |
| Image0, entry20480 | kernel entry offset is not present in the code object | 2 / 1 |

Both scans retain error states, check reason worklist_not_fully_checked and
per-entry waitcheck_backend_exit_2 with the native message. Actual native logs
and validated invocation receipts support bounded native selection rejection;
exit2 alone would not establish the cause. This is distinct from an adapter
rejecting output after native exit0/4. Empty findings are failed checking, never
clean results. Dry runs remain not_checked/dry_run and cannot prove rejection.
The recorded source snapshot and tool bytes identify this evidence's scope;
arbitrary targets, entries and binary versions remain unqualified.

## Recorded attention-kernel result

On 11 November 2026, an authorized offline AORTA scan checked one catalog-verified
entry in a retained attention code object: gfx1250, image0, `.text` entry0,
SHA-256 `04220a091086a29960a3e7e4e6253bb35c7d0d98b22ad5ac697cb4bf7236d23d`.
The native binary fingerprint matches the baseline above. A reviewed dry capture
and before/after input/tool/source/runtime checks preserved the intended scope.
No kernel dispatch, workload registration or original-container recreation was
needed. The private input and development captures are not part of this package.

| Evidence | Recorded result |
| --- | --- |
| AORTA / native exit | 0 / 4 |
| Execution / check and entry verdict | complete / ran-warn, null reasons |
| Raw counters | 1670 instructions, 456 memory events, 769 diagnostic records |
| Requested limit / emitted and retained unique records | 64 / 64 and 64 |
| Explicit omission signal | omitted 705 diagnostic(s) after limit |
| Detail completeness / native analysis | incomplete / reported complete under the supported format semantics |

Independent offline verification agreed with the raw report and preserved command
and identity receipts. `diagnostics_truncated:false` retains its historical meaning;
the explicit omission shows that diagnostic details are incomplete. These769
records do not establish769 distinct bugs or a crash cause. A representative
finding reports missing `s_wait_asynccnt <= 0` before a memory operation at
`.text+0x1c34`, with producer `.text+0x1c20` and consumer `.text+0x1c34`.

A subsequent skill-guided offline interpretation correctly separated the warning,
native4, supplied AORTA0 and reader0, retained the omission and provenance limits,
and proposed no automatic rescan. Its answer content was independently accepted;
complete setup/model/command activity was not supplied. Full native build and
generating compiler provenance remain unknown. This is bounded application
evidence, not general execution qualification, memory safety or numerical validation.

## Qualification gaps

Further targeted inputs are available as **unqualified development drafts**:
the [multi-hazard source](../assets/waitcheck-controls/multiple_missing_waits.hip),
[high-cap](../assets/waitcheck-cap-high.yaml) and
[low-cap](../assets/waitcheck-cap-low.yaml) templates, and
[invalid-image](../assets/waitcheck-invalid-image.yaml) /
[invalid-entry](../assets/waitcheck-invalid-entry.yaml) templates. Preserve the
original controls. Establish each new build's actual identity before completing
its recipes. The recorded high-cap build now shows three diagnostic records;
source sites alone do not establish counts for other builds. The cap1 comparison
exposed the original count gate; subsequent updated-producer checks are recorded
above. Invalid selectors must be absent
from verified inventory; container size is not a general proof of `.text` extent.
Schema or missing-tool failures do not establish native selector rejection.
Templates themselves confer no qualification and do not check the user's kernel;
the bounded actual results above apply only to their recorded inputs and tool.

Focus on AORTA integration and the agent's decisions. This project does not need
to recreate the upstream checker's full instruction test suite. Add a control
when it covers an untested integration behavior or a new claim the skill will make.

| Claim | Current evidence | What is still needed |
| --- | --- | --- |
| Basic finding/no-finding/refusal | Three real, reviewed AORTA controls; packaged helper and relocated verification passed | These are not all checker behaviors; keep conclusions within tested scope |
| Exact-entry selection | [One shared-image missing/corrected pair](waitcheck-isolation.md#recorded-development-result-10-november-2026) produced scoped warn/pass with matching entry markers | Broader kernel collections, targets and multi-entry worklists are not established; reverify new builds |
| Failure handling | Real wrong-digest refusal; bounded native image/entry rejection above; local missing-input/tool/schema/refusal/timeout cases | Broader native failures, versions and applicable Linux process cleanup remain unestablished |
| Count/completion semantics | Real cap64/cap1 count3/details3/1; updated producer and reader separate counts, requested limits and completion while preserving unknowns | Full binary build provenance, broader format/version qualification and exhaustive checker coverage remain open |
| Other targets/hazard classes | One scalar-wait pair on gfx1250 | Add targeted controls only when expanding the claimed scope |
| Realistic application | Authorized exact-entry attention AORTA scan above; report/raw/command identity independently verified | Generating compiler/native build provenance and broader application coverage remain unknown |
| Agent behavior | Shared-base manual checks and supplied Waitcheck evidence-answer content accepted; real-attention interpretation content accepted | Complete model/settings and activity evidence; answer content alone is not a full command audit |
| Release | Examples and retained local evidence | Owner/license/intake, CI execution and applicable evaluation gates |

**Do we need more samples?** These three are enough to explain and regress the
basic workflow. For exact-entry isolation, a two-kernel sample was added.
Its [source, draft recipes and recorded bounded result](waitcheck-isolation.md)
are provided. This pair now has a verified development result; do not collect
more random kernels to replace the remaining specific qualification checks.
Other gaps mostly need controlled failure/output fixtures or agent tests.
Keep the route draft, or explicitly narrow its supported claims, until the
relevant gaps are closed under the [readiness rules](../../../../.agents/skills/rocm-sanitizer-triage/references/aorta-workflow.md#route-readiness).
