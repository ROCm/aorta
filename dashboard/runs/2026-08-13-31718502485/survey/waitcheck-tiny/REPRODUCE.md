# Reproduce `waitcheck-tiny`

Sanitizer case `waitcheck-tiny` from run `2026-08-13-31718502485` of the AORTA sanitizer nightly. This directory is the run area for that one case: its report, the sanitizer output the verdict came from, and the provenance below. Its logs, recipe copy and inputs were pruned for falling outside the nightly's log-retention window. See `index.html` for every file actually published here.

## Run

- Commit: `68c980aa449f656a870923831f63a8aa34c145a3`
- Date: 2026-08-13
- Target: `gfx950`
- Class: survey (observed-only, non-gating)
- Workflow run: https://github.com/ROCm/aorta/actions/runs/31718502485

## Observed

- Verdict: `pass`
- Execution: `complete`
- Findings: 0

## Run it yourself

First, the checkout this run used:

```
git clone https://github.com/ROCm/aorta && cd aorta
git checkout 68c980aa449f656a870923831f63a8aa34c145a3
pip install -e .
```

Then rebuild the CI-built fixtures, which are not published here (see 'Artifacts not published' below). Run these from the repo root, the directory the clone above leaves you in:

- fixtures/isa/tiny.hsaco -- a gfx code object built from fixtures/kernels/tiny_vecadd.hip: `export PATH="${ROCM_PATH:-${ROCM_HOME:-/opt/rocm}}/lib/llvm/bin:${PATH}"`; `mkdir -p recipes/sanitizers/fixtures/isa`; `hipcc --genco --offload-arch=gfx950 recipes/sanitizers/fixtures/kernels/tiny_vecadd.hip -o tmp.o`; `if head -c 24 tmp.o | grep -qF __CLANG_OFFLOAD_BUNDLE__; then clang-offload-bundler --type=o --unbundle --input=tmp.o --targets=hipv4-amdgcn-amd-amdhsa--gfx950 --output=recipes/sanitizers/fixtures/isa/tiny.hsaco; else cp tmp.o recipes/sanitizers/fixtures/isa/tiny.hsaco; fi && rm -f tmp.o`. The conditional matters: --genco emits a raw code object on some ROCm builds and a clang-offload bundle on others, and the recorded digest is of the unbundled object.

The exact invocations the nightly used are in .github/workflows/sanitizers-nightly.yml.

Set `ROCJITSU_PREBUILT` (unpacked rocjitsu bundle supplying rj_waitcheck and the ConSan hook) before running.

Finally, the sweep itself:

```
aorta sweep run --recipe recipes/sanitizers/daily-waitcheck-tiny.yaml
```

## Artifacts not published

These are CI-built and too large to publish for every retained run. Rebuild them as above and check the digest matches.

| Path | SHA-256 |
|---|---|
| `fixtures/isa/tiny.hsaco` | `ba783710eebb0ee38fe48bb8c64ec2d1cf4d535491a81956065bc7aada1b2fa2` |

## Recorded digests

| Key | Value |
|---|---|
| `code_object:tiny.hsaco` | `ba783710eebb0ee38fe48bb8c64ec2d1cf4d535491a81956065bc7aada1b2fa2` |
| `path` | `/workspace/aorta/.sanitizer-nightly/rocjitsu-prebuilt/bin/rj_waitcheck` |
| `sha256` | `5fb63fdb255d40a5e860b1ed390eb487231c99bc12df74867fdbbb4a68ae8d64` |

## Files here

See `index.html` for the browsable list. `sanitizer_report.json` is the full `aorta.sanitizer_report/0.1` document the dashboard renders from.
