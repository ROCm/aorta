# Waitcheck: two kernels in one image

**Draft development test; not run or qualified.** This tests whether selecting
one kernel keeps a neighboring kernel out of the result. Follow the
[common workflow](aorta-workflow.md) and applicable approval rules. Every cluster
command, including listing and hashing, needs explicit approval where required
by the user's environment. This plan does not grant execution approval.

The [earlier controls](waitcheck-examples.md) use separate files. This
[sample](../assets/waitcheck-controls/two_kernel_isolation.hip) puts a missing-wait
kernel and its corrected counterpart in the **same device image**. Both recipes
must use the same file, SHA-256, target and image index; only their kernel names
and verified entry offsets differ, apart from descriptive case labels.

| Proposed selection | Expected completed result | Purpose |
| --- | --- | --- |
| `waitcheck_missing` | `ran/warn`, native exit 4, AORTA exit 0; intended scalar-wait finding | The hazardous entry is analyzed |
| `waitcheck_corrected` | `ran/pass`, native exit 0, AORTA exit 0; zero diagnostics | Its hazardous neighbor does not contaminate the selection |

These are expectations for a new build, not observed results. Investigate any
unexpected result rather than changing expectations to pass. Neither outcome
proves general correctness. **Never execute the compiled program or dispatch
either kernel.** The unsafe assembly exists only as scanner input; the host stub
retains both kernels during compilation. The source's upstream license is included.

## 1. Build and list, after approval

Use Linux, a compiler/SDK supporting gfx1250, and a compatible standalone
`rj_waitcheck`. Record the compiler and tool identity/provenance; previous hashes
in the examples guide are evidence, not proof of today's installation. Keep
gfx1250 even on a different host GPU: this test scans offline. No install,
upgrade or container change is included.

Substitute actual absolute paths below. `WORK` must not exist; this block refuses
reuse and preserves evidence on failure. GNU `timeout` is required. Save the
filled command block and its outer exit status with the returned evidence.

```bash
SKILL=/absolute/path/to/rocm-sanitizer-triage
WORK=/absolute/path/to/new-waitcheck-isolation
AMDCLANG=/absolute/path/to/amdclang++
PREBUILT=/absolute/path/to/rocjitsu/bundle
WAITCHECK="$PREBUILT/bin/rj_waitcheck"
(
set -eu
mkdir "$WORK"
cp "$SKILL/assets/waitcheck-controls/two_kernel_isolation.hip" "$WORK/"
cp "$SKILL/assets/waitcheck-controls/LICENSE.upstream.txt" "$WORK/"
sha256sum "$WORK/two_kernel_isolation.hip" "$AMDCLANG" "$WAITCHECK" > "$WORK/setup.sha256"
timeout --kill-after=10s 30s "$AMDCLANG" --version > "$WORK/compiler.txt" 2>&1
timeout --kill-after=10s 300s "$AMDCLANG" -O2 -x hip --offload-arch=gfx1250 \
  "$WORK/two_kernel_isolation.hip" -o "$WORK/two_kernel_isolation" > "$WORK/build.log" 2>&1
sha256sum "$WORK/two_kernel_isolation" > "$WORK/object.sha256"
timeout --kill-after=10s 60s "$WAITCHECK" "$WORK/two_kernel_isolation" \
  --list-code-objects > "$WORK/images.log" 2>&1
timeout --kill-after=10s 60s "$WAITCHECK" "$WORK/two_kernel_isolation" \
  --target gfx1250 --list-kernels > "$WORK/kernels.log" 2>&1
)
```

**Stop for identity review.** Do not proceed from partial listings or a failed
command. Verify both named kernels occur in the same gfx1250 image with distinct
`.text` entry offsets. Do not assume image 0, source order, symbol/descriptor
addresses or old-control offsets. If the listing cannot establish this, prepare
version-appropriate inspection and obtain approval for extra commands.

Analyze returned listings/logs using a saved reusable reader, extending an
existing one if suitable. Review new/changed readers independently. This package
does not yet provide a validated analysis reader for the new pair.

## 2. Prepare the recipes

Copy the [missing-entry template](../assets/waitcheck-isolation-missing.yaml) and
[corrected-entry template](../assets/waitcheck-isolation-corrected.yaml) into
`WORK`. Fill every `REQUIRED_...` value from verified evidence. Use
`two_kernel_isolation` as the relative input path when recipes sit beside it.
Keep the same actual file/hash/image in both, with each entry's verified name and
offset. Values are literal; YAML does not expand shell variables. Unresolved
index/offset placeholders intentionally fail; never delete `entry_offset`.

Each recipe selects one Waitcheck check, a 120-second timeout and 64 diagnostic
details. The required `consan_policy` field does not enable ConSan. Record the
AORTA executable and installed package version/location separately from the
checkout revision: a branch alone does not establish what the CLI imports.
Pin native tool path/hash/build origin, working directory and fresh output paths.

## 3. Proposed dry runs and scans

After recipe/identity review and applicable approval, use these invocation shapes.
Replace angle-bracket placeholders; these are not copy-ready commands. Capture
command, cwd, relevant environment, stdout/stderr and each individual exit code.
Use fresh output directories and an outer timeout of 180 seconds with a 10-second
termination grace. Stop on unexpected failure and retain evidence. Timeout or
missing output means failed/incomplete checking, not a pass.

```text
<absolute-aorta> sweep run --recipe <missing.yaml> --output <new-dry-missing> --dry-run
<absolute-aorta> sweep run --recipe <corrected.yaml> --output <new-dry-corrected> --dry-run
```

Dry reports should show `not_checked/dry_run` and the requested identities.
They do not establish real entry existence or tool compatibility. After reviewing
them, separately approved scans use the same recipes without `--dry-run`:

```text
<absolute-aorta> sweep run --recipe <missing.yaml> --output <new-scan-missing>
<absolute-aorta> sweep run --recipe <corrected.yaml> --output <new-scan-corrected>
```

Set `ROCJITSU_PREBUILT` to the verified bundle for both scans. Verify actual tool
and input hashes before and after; the environment variable alone is insufficient.
Do not use `waitcheck_controls.py` for this pair: its shared-name, three-case
contract describes the earlier experiment. Prepare and review concrete capture
and analysis commands/helpers before delegating execution; the shapes above are
preparation instructions, not a claim that an execution runner is ready.

## 4. Accept only evidence of isolation

Retain source/license, compiler/build record, compiled binary, image/kernel
listings, filled recipes, AORTA/tool provenance, exits and all reports/raw logs.
Use a saved, reviewed reader for returned evidence. Require:

1. Identical input bytes, target, image and native tool in both runs. Kernel
   names and distinct entries agree with the retained listing evidence.
2. Each report/worklist contains only the requested entry. Native argv uses its
   `--code-object-index` and `--kernel-entry`; raw entry markers agree. If markers
   are absent, obtain other independent scope evidence before accepting isolation.
3. Both scans complete with compatible output and no early stopping, truncated
   findings, timeout or identity conflict. Missing entry reports the intended
   scalar-load/use hazard; investigate extra diagnostics. Corrected entry has
   none, including none attributed to its hazardous neighbor.
4. Saved analysis and independent review agree with raw outputs and identities.
   Two passes, two warnings or any error leave isolation unresolved. Do not
   relabel them as the expected result.

Success would establish isolation only for this build, target, tool and AORTA
combination. It would not qualify other targets, arbitrary multi-entry worklists,
timeouts, caps or the whole route. Those gates remain separate.

Source basis: existing packaged controls, the
[pinned RocJITsu guide](https://github.com/ROCm/rocm-systems/blob/d03c6fe410eed62b211d56bb7878dd1f494269fe/emulation/rocjitsu/docs/waitcheck/README.md),
and AORTA's `waitcheck.py` argv builder at baseline `113858c4`.
