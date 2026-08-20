# Reproduce `waitcheck-tiny`

Sanitizer case `waitcheck-tiny` from run `2026-08-12-31615014466` of the AORTA sanitizer nightly. This directory is the run area for that one case: its report, the sanitizer output the verdict came from, and the provenance below. Its logs, recipe copy and inputs were pruned for falling outside the nightly's log-retention window. See `index.html` for every file actually published here.

## Run

- Commit: `f1660c492df6c68bf6750582652688ebbea181c6`
- Date: 2026-08-12
- Target: `gfx950`
- Class: survey (observed-only, non-gating)
- Workflow run: https://github.com/ROCm/aorta/actions/runs/31615014466

## Observed

- Verdict: `pass`
- Execution: `complete`
- Findings: 0

## Run it yourself

First, the checkout this run used:

```
git clone https://github.com/ROCm/aorta && cd aorta
git checkout f1660c492df6c68bf6750582652688ebbea181c6
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
| `fixtures/isa/tiny.hsaco` | `87c4da610ce6cd0d88858586eb43883ec5795b9221a79a2c8568832043c57d4e` |

## Recorded digests

| Key | Value |
|---|---|
| `code_object:tiny.hsaco` | `87c4da610ce6cd0d88858586eb43883ec5795b9221a79a2c8568832043c57d4e` |
| `path` | `/workspace/aorta/.sanitizer-nightly/rocjitsu-prebuilt/bin/rj_waitcheck` |
| `sha256` | `5c02eee0055cfa17524ceebece462936da2fcccbc629dad82cd1c388ec0909e3` |

## Files here

See `index.html` for the browsable list. `sanitizer_report.json` is the full `aorta.sanitizer_report/0.1` document the dashboard renders from.
