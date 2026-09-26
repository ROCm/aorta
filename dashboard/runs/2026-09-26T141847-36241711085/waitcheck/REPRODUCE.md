# Reproduce `waitcheck`

Sanitizer case `waitcheck` from run `2026-09-26T141847-36241711085` of the AORTA sanitizer nightly. This directory is the run area for that one case: its report, the sanitizer output the verdict came from, and the provenance below. See `index.html` for every file actually published here.

## Run

- Commit: `f32e64cf324fdae8fb6f49d3135888c4646b5104`
- Date: 2026-09-26 14:18:47 UTC
- Target: `gfx950`
- Class: guardrail (gated guardrail)
- Workflow run: https://github.com/ROCm/aorta/actions/runs/36241711085
- Container image: `rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0@sha256:3174cb7061d94c427da96c0edef4adea28046fa3f3b2ff3948dc4e995665ff8c`
- rocjitsu bundle: `11f077a2c928aa2096903e8d793b1c487a479c15` (https://github.com/ROCm/rocm-systems/actions/runs/36089010895)

## Observed

- Verdict: `warn`
- Execution: `complete`
- Findings: 64

## Run it yourself

First, the checkout this run used:

```
git clone https://github.com/ROCm/aorta && cd aorta
git checkout f32e64cf324fdae8fb6f49d3135888c4646b5104
pip install -e .
```

Then rebuild the CI-built fixtures, which are not published here (see 'Artifacts not published' below). Run these from the repo root, the directory the clone above leaves you in:

- fixtures/isa -- the per-transpose f32 GEMM code objects (sol_<n>.hsaco): `export PATH="$(python -c "from aorta.instrumentation.rocm_paths import resolve_rocm_roots; print(resolve_rocm_roots().llvm_bin_dir)"):${PATH}"`; `export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}$(python -c "import os; from aorta.instrumentation.rocm_paths import resolve_rocm_roots as r; x = r(); print(os.pathsep.join(dict.fromkeys([str(x.core_lib_dir), str(x.lib_dir)])))")"`; `mkdir -p recipes/sanitizers/fixtures/isa`; `python scripts/sanitizers/prepare_gemm_isa.py --csv recipes/sanitizers/fixtures/gemm_shapes_unique.csv --out recipes/sanitizers/fixtures/isa --top-n 3`. Extracted from the shipped Tensile libraries, never compiled from a .hip source.

The exact invocations the nightly used are in .github/workflows/sanitizers-nightly.yml.

Set `ROCJITSU_PREBUILT` (unpacked rocjitsu bundle supplying rj_waitcheck and the ConSan hook) before running.

Finally, the sweep itself:

```
aorta sweep run --recipe recipes/sanitizers/daily-waitcheck-gemm.yaml
```

## Artifacts not published

These are CI-built and too large to publish for every retained run. Rebuild them as above and check the digest matches.

| Path | SHA-256 |
|---|---|
| `fixtures/isa` | `—` |

## Recorded digests

| Key | Value |
|---|---|
| `code_object:sol_126578.hsaco` | `57c5d8efa448315ddc5f49170757ebeec1eb397468676ff158d19c71572b1c55` |
| `code_object:sol_137678.hsaco` | `aeb46fded14817c99369dad49e083efac47a71ce14921549f29027d37fb9a091` |
| `code_object:sol_175415.hsaco` | `57c5d8efa448315ddc5f49170757ebeec1eb397468676ff158d19c71572b1c55` |
| `path` | `/workspace/aorta/.sanitizer-nightly/rocjitsu-prebuilt/bin/rj_waitcheck` |
| `sha256` | `921330d92bb4645e2879a7725f6dc5dc0e7428c4cae22f990435e2bef9f37cf9` |

## Files here

See `index.html` for the browsable list. Logs are gzipped; `sanitizer_report.json` is the full `aorta.sanitizer_report/0.1` document the dashboard renders from.
