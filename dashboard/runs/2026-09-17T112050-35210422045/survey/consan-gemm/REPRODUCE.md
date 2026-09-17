# Reproduce `consan-gemm`

Sanitizer case `consan-gemm` from run `2026-09-17T112050-35210422045` of the AORTA sanitizer nightly. This directory is the run area for that one case: its report, the sanitizer output the verdict came from, and the provenance below. See `index.html` for every file actually published here.

## Run

- Commit: `73409f79a56ac10ae5374b419e5737221daadc34`
- Date: 2026-09-17 11:20:50 UTC
- Target: `gfx950`
- Class: survey (observed-only, non-gating)
- Workflow run: https://github.com/ROCm/aorta/actions/runs/35210422045
- Container image: `rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0@sha256:3174cb7061d94c427da96c0edef4adea28046fa3f3b2ff3948dc4e995665ff8c`
- rocjitsu bundle: `6e6574b59b8a1283c95e85cb93dd0e4eeab378f2` (https://github.com/ROCm/rocm-systems/actions/runs/35209663700)

## Observed

- Verdict: `error`
- Execution: `error`
- Findings: 0
- Reason: `consan_output_parse_error: reader 109123484335968 atomic site count mismatch`

## Run it yourself

First, the checkout this run used:

```
git clone https://github.com/ROCm/aorta && cd aorta
git checkout 73409f79a56ac10ae5374b419e5737221daadc34
pip install -e .
```

Then rebuild the CI-built fixtures, which are not published here (see 'Artifacts not published' below). Run these from the repo root, the directory the clone above leaves you in:

- fixtures/isa/consan_gemm_f32.hsaco -- one representative heavy f32 SS GEMM code object: `export PATH="$(python -c "from aorta.instrumentation.rocm_paths import resolve_rocm_roots; print(resolve_rocm_roots().llvm_bin_dir)"):${PATH}"`; `export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}$(python -c "import os; from aorta.instrumentation.rocm_paths import resolve_rocm_roots as r; x = r(); print(os.pathsep.join(dict.fromkeys([str(x.core_lib_dir), str(x.lib_dir)])))")"`; `mkdir -p recipes/sanitizers/fixtures/isa`; `python scripts/sanitizers/prepare_gemm_isa.py --csv recipes/sanitizers/fixtures/gemm_shapes_unique.csv --out recipes/sanitizers/fixtures/isa --top-n 0 --consan-object recipes/sanitizers/fixtures/isa/consan_gemm_f32.hsaco`. Extracted from the shipped Tensile libraries, never compiled from a .hip source.
- fixtures/bin/consan_gemm_load -- a host repro binary built from fixtures/kernels/consan_load.hip: `export PATH="$(python -c "from aorta.instrumentation.rocm_paths import resolve_rocm_roots; print(resolve_rocm_roots().llvm_bin_dir)"):${PATH}"`; `export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}$(python -c "import os; from aorta.instrumentation.rocm_paths import resolve_rocm_roots as r; x = r(); print(os.pathsep.join(dict.fromkeys([str(x.core_lib_dir), str(x.lib_dir)])))")"`; `mkdir -p recipes/sanitizers/fixtures/bin`; `hipcc --offload-arch=gfx950 -DOBJECT="\"$(pwd)/recipes/sanitizers/fixtures/isa/consan_gemm_f32.hsaco\"" recipes/sanitizers/fixtures/kernels/consan_load.hip -o recipes/sanitizers/fixtures/bin/consan_gemm_load`. The -DOBJECT define is required: it is how the binary learns which code object to load.

The exact invocations the nightly used are in .github/workflows/sanitizers-nightly.yml.

Set `ROCJITSU_PREBUILT` (unpacked rocjitsu bundle supplying rj_waitcheck and the ConSan hook) before running. `aorta` sets `HSA_TOOLS_LIB`, `HSA_TOOLS_DISABLE_REGISTER`, `RJ_CONSAN_MODE`, `RJ_CONSAN_POLICY` for you from the recipe's policy block.

Finally, the sweep itself:

```
aorta sweep run --recipe recipes/sanitizers/daily-consan-gemm.yaml
```

## Artifacts not published

These are CI-built and too large to publish for every retained run. Rebuild them as above and check the digest matches.

| Path | SHA-256 |
|---|---|
| `fixtures/isa/consan_gemm_f32.hsaco` | `57c5d8efa448315ddc5f49170757ebeec1eb397468676ff158d19c71572b1c55` |
| `fixtures/bin/consan_gemm_load` | `6e4ff7c53fcf007e3c6604ec6093510cbff56ae83337fbb1a6057aa245be39ad` |

## Recorded digests

| Key | Value |
|---|---|
| `code_object:consan_gemm_f32.hsaco` | `57c5d8efa448315ddc5f49170757ebeec1eb397468676ff158d19c71572b1c55` |
| `command` | `/workspace/aorta/recipes/sanitizers/fixtures/bin/consan_gemm_load` |
| `command_sha256` | `6e4ff7c53fcf007e3c6604ec6093510cbff56ae83337fbb1a6057aa245be39ad` |
| `hook` | `/workspace/aorta/.sanitizer-nightly/rocjitsu-prebuilt/lib/librocjitsu_dbi_hooks.so` |
| `hook_sha256` | `35e793caddd1966afc7071334ba7484b5c28c73a9fd9203249267337373b92da` |
| `selected_identity_sha256` | `acd50ca2bb450f4fb8144de0b24d11a502038583bf296d9e36f32091c5fbe079` |
| `selected_kernel` | `gemm_f32_ss` |

## Files here

See `index.html` for the browsable list. Logs are gzipped; `sanitizer_report.json` is the full `aorta.sanitizer_report/0.1` document the dashboard renders from.
