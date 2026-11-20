# Obtain the standalone Waitcheck executable

Reuse a verified existing bundle where possible. Keep the acquisition run ID,
source revision, manifest and binary SHA-256 with the case. If an existing tool
has a hash but no source/build record, keep that gap explicit.

**Pinned example.** The RocJITsu workflow built revision
`d03c6fe410eed62b211d56bb7878dd1f494269fe` in
[run 43418385944](https://github.com/ROCm/rocm-systems/actions/runs/43418385944).
Use that numeric ID with the downloader. The display number is 68. The workflow
defaults to 30-day artifact retention; availability must be checked when used.
Another build of the same source can have different binary bytes.

### Download on Linux

Use the downloader from the selected AORTA source checkout. It needs GitHub
Actions read permission on `ROCm/rocm-systems`; an authenticated `gh` session
or the downloader's supported token environment can provide it. Keep secrets
out of the command line, saved receipts and the checker environment.

```bash
# AORTA_SRC is the checkout. BUNDLE is a new absolute destination.
RUN_ID=43418385944
ARTIFACT=rocjitsu-sanitizers-Release-d03c6fe410ee
python3 "$AORTA_SRC/scripts/sanitizers/download_sanitizer_artifacts.py" --run "$RUN_ID" --list
```

Require a successful matching run and that exact artifact listed as available.
A listing can exit zero with no usable artifacts. For an expired artifact,
reuse a verified retained copy, or locate another successful run at the intended
revision and explicitly record its new numeric ID. For access denied, resolve
the repository's Actions read permission. Avoid substituting `latest`.

```bash
python3 "$AORTA_SRC/scripts/sanitizers/download_sanitizer_artifacts.py" \
  --run "$RUN_ID" --artifact "$ARTIFACT" --dest "$BUNDLE"
```

For the pinned Release artifact, the verified SHA-256 of `bin/rj_waitcheck` is
`2c406bcc3be32c48ae0ac46d0ca197a7b9a6173e0d25f5b5226fa972d8e31e77`.
The manifest specifies glibc 2.26 minimum. A later run needs its own receipts.

### Verify before use

On a successful download, require `MANIFEST.json`, `sha256sums.txt`, and
`bin/rj_waitcheck`. From this bundle check the manifest's source commit and
run ID against the selected run. Compare the binary's hash to its own checksum
entry; downloader success by itself can mean that the checksum list was missing.
For a retained copy, use its recorded run and artifact identity.

```bash
# Example for the pinned run. Requires jq and GNU sha256sum.
(
  cd "$BUNDLE" &&
  jq -e --arg run "$RUN_ID" --arg commit d03c6fe410eed62b211d56bb7878dd1f494269fe \
    '(.run_id | tostring) == $run and .commit == $commit' MANIFEST.json >/dev/null &&
  test -x bin/rj_waitcheck &&
  grep -Eq '^[0-9a-fA-F]{64}  (\./)?bin/rj_waitcheck$' sha256sums.txt &&
  sha256sum --check sha256sums.txt &&
  printf '%s  %s\n' 2c406bcc3be32c48ae0ac46d0ca197a7b9a6173e0d25f5b5226fa972d8e31e77 bin/rj_waitcheck | sha256sum --check -
) && export ROCJITSU_PREBUILT="$BUNDLE"
```

Stop on failure. Retain the manifest, checksums and resolved checker path with
the run receipts. Later confirm AORTA actually resolves that binary, including
any explicit backend override. A bundle verified on one host still needs a
compatible loader and tool/target combination for the intended check.
