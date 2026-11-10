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
SKILL=/absolute/path/to/rocm-sanitizer-triage
WORK=/absolute/path/to/new-waitcheck-example
AMDCLANG=/absolute/path/to/amdclang++
WAITCHECK=/absolute/path/to/bundle/bin/rj_waitcheck
(
set -eu
mkdir "$WORK"
"$AMDCLANG" -O2 -x hip --offload-arch=gfx1250 "$SKILL/assets/waitcheck-controls/missing_wait.hip" -o "$WORK/missing_wait"
"$AMDCLANG" -O2 -x hip --offload-arch=gfx1250 "$SKILL/assets/waitcheck-controls/corrected_wait.hip" -o "$WORK/corrected_wait"
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

The [portable helper](../scripts/waitcheck_controls.py) keeps generated recipes,
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
python "$SKILL/scripts/waitcheck_controls.py" prepare \
  --missing-object "$WORK/missing_wait" --corrected-object "$WORK/corrected_wait" \
  --kernel-name _Z17waitcheck_controlv --build-id "$BUILD_ID" \
  --missing-index "$MISSING_INDEX" --missing-entry "$MISSING_ENTRY" \
  --corrected-index "$CORRECTED_INDEX" --corrected-entry "$CORRECTED_ENTRY" \
  --output "$WORK/prepared"
python "$SKILL/scripts/waitcheck_controls.py" run \
  --prepared "$WORK/prepared" --aorta "$AORTA" --prebuilt "$PREBUILT" \
  --tool-sha256 "$TOOL_SHA256" --output "$WORK/run"
python "$SKILL/scripts/waitcheck_controls.py" verify --run "$WORK/run"
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

## Qualification gaps

Focus on AORTA integration and the agent's decisions. This project does not need
to recreate the upstream checker's full instruction test suite. Add a control
when it covers an untested integration behavior or a new claim the skill will make.

| Claim | Current evidence | What is still needed |
| --- | --- | --- |
| Basic finding/no-finding/refusal | Three real, reviewed AORTA controls; packaged helper and relocated verification passed | These are not all checker behaviors; keep conclusions within tested scope |
| Exact-entry selection | Matching identity in one-kernel inputs | [Two-kernel preparation package](waitcheck-isolation.md); build/list/run and independent acceptance still required |
| Failure handling | Wrong expected digest rejected | Focused missing-input/tool, wrong selector and timeout cases as applicable; no new kernel needed for most |
| Count/completion semantics | Low-count output below cap | Targeted many-diagnostic/cap or early-stop evidence for each supported tool format; keep unknown formats unqualified |
| Other targets/hazard classes | One scalar-wait pair on gfx1250 | Add targeted controls only when expanding the claimed scope |
| Realistic application | Historical attention results were direct-tool evidence | Fresh approved AORTA comparison if that application scope is claimed |
| Agent behavior | Shared-base manual checks exist | Scoped instruction-following cases for these outcomes, no silent scope fallback and no pass from empty/error results |
| Release | Examples and retained local evidence | Owner/license/intake, CI execution and applicable evaluation gates |

**Do we need more samples?** These three are enough to explain and regress the
basic workflow. Do not collect more random kernels. For broader exact-entry
qualification, the next useful additional sample is a two-kernel isolation case.
Its [source, draft recipes and expected results](waitcheck-isolation.md) are now
provided for preparation; no isolation result is claimed.
Other gaps mostly need controlled failure/output fixtures or agent tests.
Keep the route draft, or explicitly narrow its supported claims, until the
relevant gaps are closed under the [readiness rules](aorta-workflow.md#route-readiness).
