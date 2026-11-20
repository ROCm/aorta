# Prepare Waitcheck recipes from compiled objects

Use the Python environment for the AORTA development procedure. Set `SKILL` to the absolute directory containing this skill's `SKILL.md`. Set `ROCJITSU_PREBUILT` to the verified native bundle and use the checker fingerprint from its trusted acquisition record. The helper reads the compiled object twice with the native checker's inventory options. Obtain permission for those commands when working on a node. [Tool and run prerequisites](waitcheck-procedure.md)

## One compiled object and one named kernel

Substitute your absolute paths, actual target, exact kernel name, expected checker digest and a fresh destination whose parent exists. You can start in any working directory.
Pass the user's selected file to `--object`. A same-named or equal-hash file
under the current directory is a separate choice. The helper uses the selected
regular file's resolved absolute path and rejects a final-component symlink.
Verify the argument and retain any path normalization, including a resolved
parent directory, so its relationship to the user's selection is explicit.

```sh
python "$SKILL/scripts/waitcheck_prepare.py" prepare \
  --waitcheck "$ROCJITSU_PREBUILT/bin/rj_waitcheck" --waitcheck-sha256 "$CHECKER_SHA256" \
  --object /path/to/final/kernel.hsaco --target gfx950 --kernel my_kernel \
  --output /path/to/work/my-plan
```

Read `my-plan/entry-0001/recipe.yaml`. It records the absolute object path, its SHA-256, the architecture, the kernel name, the verified image index and the byte offset in the image’s `.text` instruction section. The helper uses the installed exact-entry template and AORTA's recipe loader. It pins one Waitcheck entry, a 120-second checker timeout and a 64-detail limit.

For example, if the inventory lists image `1` and entry `0x120`, the recipe uses image `1` and decimal offset `288`. An explicit `0` is equally valid. The helper refuses a missing selector, repeated name for a single-kernel request, conflicting inventory counts, changed object or failed listing. The `descriptor_vaddr` field locates the descriptor and cannot supply an entry offset.

`inventory-0001/` retains the arguments, stdout, stderr and status for each attempted listing. Each call has a 15-second timeout. The helper accepts the text image counts and version-1 kernel inventory JSONL, up to 4 MiB per listing. Other formats require checking the installed tool's contract. `--waitcheck-sha256` is optional if you only want to record the tool's current bytes. Pin the expected digest when it was already verified. A digest alone does not establish source or build provenance.

A successfully prepared plan has `manifest.json`. If preparation failed, keep the partial directory and inspection receipts. Recipes in an unfinished plan are unsuitable for launch. Preparation ends before the AORTA dry run. Review the planned identity, follow the procedure for permission and dry-run verification, and preserve its separate `plan/` and `scan/` outputs.

## Directory of compiled objects

Choose the architecture and file patterns explicitly to describe the scope you intend. Patterns match filenames in every subdirectory, including the directory root. The default is `*.hsaco`. You can repeat `--pattern` for other compiled formats the installed native tool supports. The helper inventories every matched object and selects every listed entry on the requested target. An object with no such target stops preparation and identifies the gap. A directory containing an unbuilt model needs its compiled artifacts first.

```sh
python "$SKILL/scripts/waitcheck_prepare.py" prepare \
  --waitcheck "$ROCJITSU_PREBUILT/bin/rj_waitcheck" --waitcheck-sha256 "$CHECKER_SHA256" \
  --directory /path/to/model-objects --pattern '*.hsaco' --pattern '*.co' \
  --target gfx950 --output /path/to/work/model-plan
```

The output directory must be fresh and outside the input directory. Review the object paths, patterns and entry count in `manifest.json`. Entries with the same kernel or file name in different artifacts keep separate IDs and recipes. A model's runtime dispatches, unseen specializations and other targets are outside this inventory.

For each authorized run, use that entry's own recipe and output locations. For example, the documented AORTA dry run for the first entry uses `--recipe /path/to/work/model-plan/entry-0001/recipe.yaml --output /path/to/work/model-plan/entry-0001/plan --dry-run`. After accepting the dry-run result and obtaining the required scan permission, the scan destination for that entry is `/path/to/work/model-plan/entry-0001/scan`. Keep each report there as `scan/sanitizer_report.json` and its native log, when available, as `scan/waitcheck/waitcheck-0.log`. Follow the same procedure for the authorized entries, preserving incomplete checks.

## Summarize saved directory coverage

```sh
python "$SKILL/scripts/waitcheck_prepare.py" aggregate \
  --plan /path/to/work/model-plan --output /path/to/work/coverage.json
```

Read `outcome_counts` and the individual `entries`. Every prepared entry has a row, even when the report is missing. Each usable report is compared with its expected identity and interpreted with the offline reader. A `reported_pass` or `reported_warn` describes that entry's saved result. `raw_log: null` means the reader could only use the report. Inspect the linked report and use the [single-entry reader](waitcheck-procedure.md) for diagnostic details. An identity mismatch, reader conflict, incomplete check or missing report leaves coverage unresolved for that entry.

Exit 0 means the helper wrote the requested plan or coverage file. A coverage file can contain failed or missing checks. Existing output paths are preserved. Aggregation runs no native commands and makes no overall kernel-correctness claim.
