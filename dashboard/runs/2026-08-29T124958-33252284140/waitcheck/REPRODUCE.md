# Reproduce `waitcheck`

Sanitizer case `waitcheck` from run `2026-08-29T124958-33252284140` of the AORTA sanitizer nightly. This directory is the run area for that one case: its report, the sanitizer output the verdict came from, and the provenance below. Its logs, recipe copy and inputs were pruned for falling outside the nightly's log-retention window. See `index.html` for every file actually published here.

## Run

- Commit: `e61f455a130bf266c83730ac055cd208efe0af87`
- Date: 2026-08-29 12:49:58 UTC
- Target: `gfx950`
- Class: guardrail (gated guardrail)
- Workflow run: https://github.com/ROCm/aorta/actions/runs/33252284140
- Container image: `rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0@sha256:4449f856653602317e4101a76fce599c7fcd58ccec2e539951fce5f73083179e`
- rocjitsu bundle: `97c1640b2f6529ef59a6dcf1068243107e190d09` (https://github.com/ROCm/rocm-systems/actions/runs/32857249576)

## Observed

- Verdict: `warn`
- Execution: `complete`
- Findings: 64

## Run it yourself

First, the checkout this run used:

```
git clone https://github.com/ROCm/aorta && cd aorta
git checkout e61f455a130bf266c83730ac055cd208efe0af87
pip install -e .
```

Then rebuild the CI-built fixtures, which are not published here (see 'Artifacts not published' below). Run these from the repo root, the directory the clone above leaves you in:

- fixtures/isa -- the per-transpose f32 GEMM code objects (sol_<n>.hsaco): `export PATH="$(python -c "from aorta.instrumentation.rocm_paths import resolve_rocm_roots; print(resolve_rocm_roots().llvm_bin_dir)"):${PATH}"`; `mkdir -p recipes/sanitizers/fixtures/isa`; `python scripts/sanitizers/prepare_gemm_isa.py --csv recipes/sanitizers/fixtures/gemm_shapes_unique.csv --out recipes/sanitizers/fixtures/isa --top-n 3`. Extracted from the shipped Tensile libraries, never compiled from a .hip source.

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
| `code_object:sol_126578.hsaco` | `5bd40b78d1eae1f6cbbea6fe09851b9676d72845fa9082fb49379514be647805` |
| `code_object:sol_137678.hsaco` | `7ea836fe29106e8548b5e145340994e3e2a25c872f63db486b37703f7c646ac2` |
| `code_object:sol_175415.hsaco` | `5bd40b78d1eae1f6cbbea6fe09851b9676d72845fa9082fb49379514be647805` |
| `path` | `/workspace/aorta/.sanitizer-nightly/rocjitsu-prebuilt/bin/rj_waitcheck` |
| `sha256` | `a70945fb1135a436c13c83ed96a6cff3655784138f4db3d20ef7315ade0a8968` |

## Files here

See `index.html` for the browsable list. `sanitizer_report.json` is the full `aorta.sanitizer_report/0.1` document the dashboard renders from.
