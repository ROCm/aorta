# Reproduce `consan-gemm`

Sanitizer case `consan-gemm` from run `2026-09-01T125503-33507543444` of the AORTA sanitizer nightly. This directory is the run area for that one case: its report, the sanitizer output the verdict came from, and the provenance below. See `index.html` for every file actually published here.

## Run

- Commit: `d11092a603d9fe07f12046d144540be1fa80c264`
- Date: 2026-09-01 12:55:03 UTC
- Target: `gfx950`
- Class: survey (observed-only, non-gating)
- Workflow run: https://github.com/ROCm/aorta/actions/runs/33507543444
- Container image: `rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0@sha256:4449f856653602317e4101a76fce599c7fcd58ccec2e539951fce5f73083179e`
- rocjitsu bundle: `bed84edd689a56b7443f03a69cbe619d5fc2c1a4` (https://github.com/ROCm/rocm-systems/actions/runs/33428269891)

## Observed

- Verdict: `error`
- Execution: `error`
- Findings: 0
- Reason: `consan_strict_load_rejection`

## Run it yourself

First, the checkout this run used:

```
git clone https://github.com/ROCm/aorta && cd aorta
git checkout d11092a603d9fe07f12046d144540be1fa80c264
pip install -e .
```

Then rebuild the CI-built fixtures, which are not published here (see 'Artifacts not published' below). Run these from the repo root, the directory the clone above leaves you in:

- fixtures/isa/consan_gemm_f32.hsaco -- one representative heavy f32 SS GEMM code object: `export PATH="$(python -c "from aorta.instrumentation.rocm_paths import resolve_rocm_roots; print(resolve_rocm_roots().llvm_bin_dir)"):${PATH}"`; `mkdir -p recipes/sanitizers/fixtures/isa`; `python scripts/sanitizers/prepare_gemm_isa.py --csv recipes/sanitizers/fixtures/gemm_shapes_unique.csv --out recipes/sanitizers/fixtures/isa --top-n 0 --consan-object recipes/sanitizers/fixtures/isa/consan_gemm_f32.hsaco`. Extracted from the shipped Tensile libraries, never compiled from a .hip source.
- fixtures/bin/consan_gemm_load -- a host repro binary built from fixtures/kernels/consan_load.hip: `export PATH="$(python -c "from aorta.instrumentation.rocm_paths import resolve_rocm_roots; print(resolve_rocm_roots().llvm_bin_dir)"):${PATH}"`; `mkdir -p recipes/sanitizers/fixtures/bin`; `hipcc --offload-arch=gfx950 -DOBJECT="\"$(pwd)/recipes/sanitizers/fixtures/isa/consan_gemm_f32.hsaco\"" recipes/sanitizers/fixtures/kernels/consan_load.hip -o recipes/sanitizers/fixtures/bin/consan_gemm_load`. The -DOBJECT define is required: it is how the binary learns which code object to load.

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
| `fixtures/isa/consan_gemm_f32.hsaco` | `5bd40b78d1eae1f6cbbea6fe09851b9676d72845fa9082fb49379514be647805` |
| `fixtures/bin/consan_gemm_load` | `2e9e496e458823e17cbee41a48b3fdb848279682b44ebdd3096d9ce5bd442d65` |

## Recorded digests

| Key | Value |
|---|---|
| `code_object:consan_gemm_f32.hsaco` | `5bd40b78d1eae1f6cbbea6fe09851b9676d72845fa9082fb49379514be647805` |
| `command` | `/workspace/aorta/recipes/sanitizers/fixtures/bin/consan_gemm_load` |
| `command_sha256` | `2e9e496e458823e17cbee41a48b3fdb848279682b44ebdd3096d9ce5bd442d65` |
| `hook` | `/workspace/aorta/.sanitizer-nightly/rocjitsu-prebuilt/lib/librocjitsu_dbi_hooks.so` |
| `hook_sha256` | `ec09f19e38fbedb0a1bcaeb4e88bae50df2da8b9c4ceb27a7c067a88364a6b02` |
| `selected_identity_sha256` | `990933656b83ac601425c2160ab393032a6c521021b196195ae0dd04673edb19` |
| `selected_kernel` | `gemm_f32_ss` |

## Files here

See `index.html` for the browsable list. Logs are gzipped; `sanitizer_report.json` is the full `aorta.sanitizer_report/0.1` document the dashboard renders from.
