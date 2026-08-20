# Reproduce `consan-gemm`

Sanitizer case `consan-gemm` from run `2026-08-13-31714026084` of the AORTA sanitizer nightly. This directory is the run area for that one case: its report, the sanitizer output the verdict came from, and the provenance below. Its logs, recipe copy and inputs were pruned for falling outside the nightly's log-retention window. See `index.html` for every file actually published here.

## Run

- Commit: `68c980aa449f656a870923831f63a8aa34c145a3`
- Date: 2026-08-13
- Target: `gfx950`
- Class: survey (observed-only, non-gating)
- Workflow run: https://github.com/ROCm/aorta/actions/runs/31714026084

## Observed

- Verdict: `error`
- Execution: `error`
- Findings: 0
- Reason: `combined_hook_timeout`

## Run it yourself

First, the checkout this run used:

```
git clone https://github.com/ROCm/aorta && cd aorta
git checkout 68c980aa449f656a870923831f63a8aa34c145a3
pip install -e .
```

Then rebuild the CI-built fixtures, which are not published here (see 'Artifacts not published' below). Run these from the repo root, the directory the clone above leaves you in:

- fixtures/isa/consan_gemm_f32.hsaco -- one representative heavy f32 SS GEMM code object: `export PATH="${ROCM_PATH:-${ROCM_HOME:-/opt/rocm}}/lib/llvm/bin:${PATH}"`; `mkdir -p recipes/sanitizers/fixtures/isa`; `python scripts/sanitizers/prepare_gemm_isa.py --csv recipes/sanitizers/fixtures/gemm_shapes_unique.csv --out recipes/sanitizers/fixtures/isa --top-n 0 --consan-object recipes/sanitizers/fixtures/isa/consan_gemm_f32.hsaco`. Extracted from the shipped Tensile libraries, never compiled from a .hip source.
- fixtures/bin/consan_gemm_load -- a host repro binary built from fixtures/kernels/consan_load.hip: `export PATH="${ROCM_PATH:-${ROCM_HOME:-/opt/rocm}}/lib/llvm/bin:${PATH}"`; `mkdir -p recipes/sanitizers/fixtures/bin`; `hipcc --offload-arch=gfx950 -DOBJECT="\"$(pwd)/recipes/sanitizers/fixtures/isa/consan_gemm_f32.hsaco\"" recipes/sanitizers/fixtures/kernels/consan_load.hip -o recipes/sanitizers/fixtures/bin/consan_gemm_load`. The -DOBJECT define is required: it is how the binary learns which code object to load.

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
| `fixtures/isa/consan_gemm_f32.hsaco` | `c1190d2005185bbf797b913022a681d52fb8ea90748901495dde56cc06f359b3` |
| `fixtures/bin/consan_gemm_load` | `b8ff5ea7642e28771add88aea42a049f86db3260cb674ecea03300f7c296abd8` |

## Recorded digests

| Key | Value |
|---|---|
| `code_object:consan_gemm_f32.hsaco` | `c1190d2005185bbf797b913022a681d52fb8ea90748901495dde56cc06f359b3` |
| `command` | `/workspace/aorta/recipes/sanitizers/fixtures/bin/consan_gemm_load` |
| `command_sha256` | `b8ff5ea7642e28771add88aea42a049f86db3260cb674ecea03300f7c296abd8` |
| `hook` | `/workspace/aorta/.sanitizer-nightly/rocjitsu-prebuilt/lib/librocjitsu_dbi_hooks.so` |
| `hook_sha256` | `bb799f4bacf3938be04a8eda7766d410e8a4d3d4f2051d2b151a3ec24051a14b` |
| `selected_identity_sha256` | `505ae1eff9c51feb670a16eb7ad5347c7cb416a451e93e53db0d7f16b67a8075` |
| `selected_kernel` | `gemm_f32_ss` |

## Files here

See `index.html` for the browsable list. `sanitizer_report.json` is the full `aorta.sanitizer_report/0.1` document the dashboard renders from.
