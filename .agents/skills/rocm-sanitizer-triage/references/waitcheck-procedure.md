# Prepare a user's offline Waitcheck check

**Draft procedure.** Use this to prepare a plan or to interpret saved evidence.
Real scans are currently development qualification under separate authorization;
this document does not qualify skill-guided execution. Read
[Waitcheck prerequisites and limits](waitcheck.md) first.

The first proposed route is Linux x86_64, the recorded compatible standalone
`rj_waitcheck`, gfx1250, and one verified entry in one selected image. The host
does not dispatch the compiled input; a matching physical GPU is unnecessary.
Other targets, formats and whole-object scans need their own support evidence.

## 1. Collect facts and preserve the input

Use the user's final compiled object. A demo can test setup but cannot answer
whether the user's kernel has a missing-wait hazard. Source alone is insufficient.

Before completing the recipe, establish the following from the final build and
version-appropriate object inspection. Retain the inspection output and command.

| Fact | Evidence needed |
| --- | --- |
| Compiled input | Path on the execution host, SHA-256, build/source revision, compiler and options or supplied-artifact origin |
| Selection | Exact kernel name, target architecture, selected image index and verified `.text` entry offset |
| AORTA | Chosen executable/interpreter, installed package version, actual imported source/checkout revision and local modifications where available |
| Native tool | Actual resolved executable path/hash, upstream source revision, build options and manifest/acquisition origin where available |
| Host and limits | OS/CPU, relevant loader dependencies, working directory, fresh output paths, timeout and diagnostic policy |

Record unknowns as unknown. A tool hash identifies bytes; it does not identify
their source/build provenance. An installed package version can differ from the
checkout imported by an editable install. The current control binary's full
build provenance is unresolved; do not describe it as reproducibly sourced.
Preserve this gap in any proposed qualification record.

For a remote machine, obtain approval for the specific inspection before any
cluster command. For a text-only request, prepare these checks without running
them. Do not compile, install or dispatch a program merely to explain a report.

## 2. Prepare one recipe and its evidence record

Copy [waitcheck-exact-entry.yaml](../assets/waitcheck-exact-entry.yaml) into a new
task directory and resolve every placeholder. Keep explicit object SHA-256,
image index and entry offset. Literal paths refer to the execution machine;
relative object paths resolve from the copied recipe. YAML does not expand shell
variables. A descriptor/symbol address is not automatically a `.text` entry.

The starting bounds are 120 seconds for the backend and 64 diagnostic details,
matching the low-count development controls. Adjust bounds only with a stated
reason; raising the cap does not by itself establish complete analysis. Keep
the original recipe and record its hash. Never delete `entry_offset` to make an
unresolved selection parse: that changes the scan scope.

Keep a small task record beside the recipe. The following are evidence to retain,
not a new machine-readable result schema:

- `recipe.yaml` and its SHA-256; intended identity and inspection receipts.
- `provenance.md`: the facts/unknowns above and where each came from.
- `commands.txt`: exact commands, working directory, relevant environment changes,
  requested limits, authorization scope and producing timestamps. Record only
  relevant variables; do not dump credentials or the entire environment.
- `plan/` and `scan/`: separate fresh AORTA output directories.
- Per invocation, stdout, stderr and process exit, including failures.
- Input/tool hashes before and after scanning; original files remain unchanged.

Resolve the actual native tool selected by this AORTA version. Its lookup uses
prebuilt bundle, build tree and PATH; a configured variable alone does not prove
which file ran. Retain the report's backend identity and compare it with the
recorded intended binary. The reader below does not collect this provenance.

## 3. Review the plan, then run only within authorization

The following are command shapes to fill with absolute paths. `AORTA` is the
chosen executable, `WORK` a new task directory, and `PYTHON` the corresponding
AORTA interpreter. `SKILL` is this skill's directory, discovered separately from
the executable; activating a venv alone does not prove which skill was loaded.

Prepare the dry run with separate stdout/stderr/exit receipts:

```bash
"$AORTA" sweep run --recipe "$WORK/recipe.yaml" --output "$WORK/plan" --dry-run
```

Read its report before planning a real scan. Confirm one Waitcheck check, intended
target/object/hash/image/entry, and `not_checked/dry_run`. A dry run does not
launch the checker or establish that an image/entry exists or is supported.
Stop if identity or scope differs; keep the failed plan and do not broaden scope.

For an approved development scan, use a fresh `scan/` destination and bound the
outer CLI as well as the recipe's backend. Example using GNU `timeout`:

```bash
timeout --signal=TERM --kill-after=10s 180s \
  "$AORTA" sweep run --recipe "$WORK/recipe.yaml" --output "$WORK/scan" \
  > "$WORK/scan.stdout.txt" 2> "$WORK/scan.stderr.txt"
scan_exit=$?
printf '%s\n' "$scan_exit" > "$WORK/scan.exit.txt"
```

Capture the exit immediately in a shell that continues after a nonzero command;
an outer `set -e` must not discard the failure receipt. A timeout or interrupted
invocation may leave partial/no report. Preserve it; no report is an evidence
gap, never a pass. Timeout normalization tests do not establish Linux descendant
cleanup. Before claiming a process-tree cleanup guarantee, validate the chosen
launcher separately with a harmless controlled process.

After any outcome, preserve logs/reports and record post-run input/tool hashes.
If bytes changed, attribute no result to a stable intended build until reconciled.
Do not retry automatically with different selectors, weaker checks or another
tool. This stage does not supply a general remote execution/capture runner.

## 4. Read the saved report and raw log

Use [waitcheck_report.py](../scripts/waitcheck_report.py) with the Python environment
containing the compatible AORTA report/parser APIs. It reads local evidence and
writes a new Markdown summary; it runs no checker or input program.

```bash
"$PYTHON" "$SKILL/scripts/waitcheck_report.py" \
  --report "$WORK/scan/sanitizer_report.json" \
  --log "$WORK/scan/waitcheck/waitcheck-0.log" \
  --output "$WORK/summary.md"
```

Pass the log path explicitly; the reader does not follow report paths. With
copied remote evidence, supply its local paths and keep recorded execution paths
unchanged. A prelaunch refusal or dry run can legitimately have no native log;
omit `--log` and retain the reason. For a reported completed scan, missing raw
output limits confirmation. Preserve both check-level and kernel-level reasons.

Reader exits describe reading, never the kernel: 0 means a summary was written,
including any missing-evidence limits; 2 means a summary records conflicting
report/raw evidence or unsupported raw output; 1 means invalid/unsupported report
scope or I/O failure.
Invalid command-line arguments also use the standard usage-error exit 2, with
stderr and no summary; distinguish that from a written evidence summary.
It accepts one Waitcheck check and one exact-entry identity, preserves the original
report verdict and compares supplied raw output where supported. It does not
verify the original object/tool bytes, recipe, intended kernel or full provenance.
Each input is limited to 4 MiB; oversized or unsupported evidence is refused,
leaving the originals available for a version-appropriate reader.
Use the external receipts in steps 1–3 for those facts. A reader exit of 0 alone
is not a clean sanitizer result.

The summary separates raw count/meaning, emitted diagnostic records, unique
retained findings, requested limit, detail completeness and native completion.
Missing report fields remain unknown. Deduplication alone does not imply omitted
details. Only supported explicit omission/early-stop messages establish newer
semantics; an unexplained bare count/record mismatch remains a compatibility gap.
The historical truncation Boolean and AORTA execution status are preserved,
without upgrading false/complete into native completeness claims.

Use [the interpretation rules](waitcheck.md#6-evidence-and-interpretation) and
the common answer: checked/proposed scope, result, supporting evidence, limits,
next action. A supplied prose summary supports a conditional explanation without
requiring this helper or a new run. Never infer general correctness, memory safety
or race freedom from Waitcheck.

## Remaining qualification

Finding/corrected/digest-refusal controls and one shared-image isolation pair have
retained real evidence. Local failure tests also cover report handling with
simulated responses. A real cap64/cap1 comparison now records three summary
diagnostics with three/one retained details; the low-cap report leaves truncation
false. The original reader rejected its count mismatch; the updated reader now
explains its supported explicit omission signal. See the
[recorded result](waitcheck-examples.md#recorded-diagnostic-limit-result).
Native invalid selection was not reached in that original run. Subsequent bounded
runs independently validated updated-producer cap64/cap1 metadata and native
rejection of deliberately absent image/entry selectors; see the
[subsequent results](waitcheck-examples.md#subsequent-updated-report-and-selector-results).
This closes those recorded integration cases, not broader format/target support.
Applicable process cleanup, complete tool provenance, full agent activity/model
evidence and the real-attention skill walkthrough remain open. Preserve warnings
and count evidence; unknown completion formats remain unqualified.
See [qualification gaps](waitcheck-examples.md#qualification-gaps).
