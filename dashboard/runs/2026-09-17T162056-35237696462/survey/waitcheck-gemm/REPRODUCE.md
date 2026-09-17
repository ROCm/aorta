# Reproduce `waitcheck-gemm`

Sanitizer case `waitcheck-gemm` from run `2026-09-17T162056-35237696462` of the AORTA sanitizer nightly. This directory is the run area for that one case: its report, the sanitizer output the verdict came from, and the provenance below. See `index.html` for every file actually published here.

## Run

- Commit: `d2fa5f1645c23b78bcf822570e76fd1e7d38c18e`
- Date: 2026-09-17 16:20:56 UTC
- Target: `gfx950`
- Class: survey (observed-only, non-gating)
- Workflow run: https://github.com/ROCm/aorta/actions/runs/35237696462
- Container image: `rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0@sha256:3174cb7061d94c427da96c0edef4adea28046fa3f3b2ff3948dc4e995665ff8c`
- rocjitsu bundle: `6e6574b59b8a1283c95e85cb93dd0e4eeab378f2` (https://github.com/ROCm/rocm-systems/actions/runs/35209663700)

## Observed

- Verdict: `warn`
- Execution: `complete`
- Findings: 32

## Run it yourself

First, the checkout this run used:

```
git clone https://github.com/ROCm/aorta && cd aorta
git checkout d2fa5f1645c23b78bcf822570e76fd1e7d38c18e
pip install -e .
```

Then rebuild the CI-built fixtures, which are not published here (see 'Artifacts not published' below). Run these from the repo root, the directory the clone above leaves you in:

- fixtures/isa/consan_gemm_f32.hsaco -- one representative heavy f32 SS GEMM code object: `export PATH="$(python -c "from aorta.instrumentation.rocm_paths import resolve_rocm_roots; print(resolve_rocm_roots().llvm_bin_dir)"):${PATH}"`; `export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}$(python -c "import os; from aorta.instrumentation.rocm_paths import resolve_rocm_roots as r; x = r(); print(os.pathsep.join(dict.fromkeys([str(x.core_lib_dir), str(x.lib_dir)])))")"`; `mkdir -p recipes/sanitizers/fixtures/isa`; `python scripts/sanitizers/prepare_gemm_isa.py --csv recipes/sanitizers/fixtures/gemm_shapes_unique.csv --out recipes/sanitizers/fixtures/isa --top-n 0 --consan-object recipes/sanitizers/fixtures/isa/consan_gemm_f32.hsaco`. Extracted from the shipped Tensile libraries, never compiled from a .hip source.

The exact invocations the nightly used are in .github/workflows/sanitizers-nightly.yml.

Set `ROCJITSU_PREBUILT` (unpacked rocjitsu bundle supplying rj_waitcheck and the ConSan hook) before running.

Finally, the sweep itself:

```
aorta sweep run --recipe recipes/sanitizers/daily-waitcheck-gemm-object.yaml
```

## Artifacts not published

These are CI-built and too large to publish for every retained run. Rebuild them as above and check the digest matches.

| Path | SHA-256 |
|---|---|
| `fixtures/isa/consan_gemm_f32.hsaco` | `57c5d8efa448315ddc5f49170757ebeec1eb397468676ff158d19c71572b1c55` |

## Recorded digests

| Key | Value |
|---|---|
| `code_object:consan_gemm_f32.hsaco` | `57c5d8efa448315ddc5f49170757ebeec1eb397468676ff158d19c71572b1c55` |
| `path` | `/workspace/aorta/.sanitizer-nightly/rocjitsu-prebuilt/bin/rj_waitcheck` |
| `sha256` | `6c7c2baa6a99ebf73d39d4c0719a2590fcdeecd1d12b96ed5a04ed36e2ce6c2b` |

## Files here

See `index.html` for the browsable list. Logs are gzipped; `sanitizer_report.json` is the full `aorta.sanitizer_report/0.1` document the dashboard renders from.
