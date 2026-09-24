# Reproduce `waitcheck-lds-dispatch`

Sanitizer case `waitcheck-lds-dispatch` from run `2026-09-24T142227-35998938187` of the AORTA sanitizer nightly. This directory is the run area for that one case: its report, the sanitizer output the verdict came from, and the provenance below. See `index.html` for every file actually published here.

## Run

- Commit: `fe7f92211b6254aa3f6da5b634931a41ca921413`
- Date: 2026-09-24 14:22:27 UTC
- Target: `gfx950`
- Class: survey (observed-only, non-gating)
- Workflow run: https://github.com/ROCm/aorta/actions/runs/35998938187
- Container image: `rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0@sha256:3174cb7061d94c427da96c0edef4adea28046fa3f3b2ff3948dc4e995665ff8c`
- rocjitsu bundle: `58136f1944ecd7786a925454f76b7739c072f092` (https://github.com/ROCm/rocm-systems/actions/runs/35883377508)

## Observed

- Verdict: `pass`
- Execution: `complete`
- Findings: 0

## Run it yourself

First, the checkout this run used:

```
git clone https://github.com/ROCm/aorta && cd aorta
git checkout fe7f92211b6254aa3f6da5b634931a41ca921413
pip install -e .
```

Then rebuild the CI-built fixtures, which are not published here (see 'Artifacts not published' below). Run these from the repo root, the directory the clone above leaves you in:

- fixtures/isa/lds.hsaco -- a gfx code object built from fixtures/kernels/lds_reduce.hip: `export PATH="$(python -c "from aorta.instrumentation.rocm_paths import resolve_rocm_roots; print(resolve_rocm_roots().llvm_bin_dir)"):${PATH}"`; `export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}$(python -c "import os; from aorta.instrumentation.rocm_paths import resolve_rocm_roots as r; x = r(); print(os.pathsep.join(dict.fromkeys([str(x.core_lib_dir), str(x.lib_dir)])))")"`; `mkdir -p recipes/sanitizers/fixtures/isa`; `hipcc --genco --offload-arch=gfx950 recipes/sanitizers/fixtures/kernels/lds_reduce.hip -o tmp.o`; `if head -c 24 tmp.o | grep -qF __CLANG_OFFLOAD_BUNDLE__; then clang-offload-bundler --type=o --unbundle --input=tmp.o --targets=hipv4-amdgcn-amd-amdhsa--gfx950 --output=recipes/sanitizers/fixtures/isa/lds.hsaco; else cp tmp.o recipes/sanitizers/fixtures/isa/lds.hsaco; fi && rm -f tmp.o`. The conditional matters: --genco emits a raw code object on some ROCm builds and a clang-offload bundle on others, and the recorded digest is of the unbundled object.

The exact invocations the nightly used are in .github/workflows/sanitizers-nightly.yml.

Set `ROCJITSU_PREBUILT` (unpacked rocjitsu bundle supplying rj_waitcheck and the ConSan hook) before running.

Finally, the sweep itself:

```
aorta sweep run --recipe recipes/sanitizers/daily-waitcheck-lds-dispatch.yaml
```

## Artifacts not published

These are CI-built and too large to publish for every retained run. Rebuild them as above and check the digest matches.

| Path | SHA-256 |
|---|---|
| `fixtures/isa/lds.hsaco` | `7cde715c4d9be08fb2d2e7c61485484a0b12d7f6ce59635f90e02f6d6cf5a56a` |

## Recorded digests

| Key | Value |
|---|---|
| `code_object:lds.hsaco` | `7cde715c4d9be08fb2d2e7c61485484a0b12d7f6ce59635f90e02f6d6cf5a56a` |
| `path` | `/workspace/aorta/.sanitizer-nightly/rocjitsu-prebuilt/bin/rj_waitcheck` |
| `sha256` | `a0f34c2eeda0b6bed443f3784e63e23544b388d244d754bcec1f2ea7dadb1cb0` |

## Files here

See `index.html` for the browsable list. Logs are gzipped; `sanitizer_report.json` is the full `aorta.sanitizer_report/0.1` document the dashboard renders from.
