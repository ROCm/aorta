# Reproduce `waitcheck`

Sanitizer case `waitcheck` from run `2026-08-10-31386669577` of the AORTA sanitizer nightly. This directory is the run area for that one case: its report, the sanitizer output the verdict came from, and the provenance below. Its logs, recipe copy and inputs were pruned for falling outside the nightly's log-retention window. See `index.html` for every file actually published here.

## Run

- Commit: `7f3ba90830b9478b8d01b8c307bc252a45118449`
- Date: 2026-08-10
- Target: `gfx950`
- Class: guardrail (gated guardrail)
- Workflow run: https://github.com/ROCm/aorta/actions/runs/31386669577

## Observed

- Verdict: `warn`
- Execution: `complete`
- Findings: 32

## Run it yourself

First, the checkout this run used:

```
git clone https://github.com/ROCm/aorta && cd aorta
git checkout 7f3ba90830b9478b8d01b8c307bc252a45118449
pip install -e .
```

Then rebuild the CI-built fixtures, which are not published here (see 'Artifacts not published' below). Run these from the repo root, the directory the clone above leaves you in:

- fixtures/isa -- the per-transpose f32 GEMM code objects (sol_<n>.hsaco): `export PATH="${ROCM_PATH:-${ROCM_HOME:-/opt/rocm}}/lib/llvm/bin:${PATH}"`; `mkdir -p recipes/sanitizers/fixtures/isa`; `python scripts/sanitizers/prepare_gemm_isa.py --csv recipes/sanitizers/fixtures/gemm_shapes_unique.csv --out recipes/sanitizers/fixtures/isa --top-n 3`. Extracted from the shipped Tensile libraries, never compiled from a .hip source.

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
| `code_object:sol_126578.hsaco` | `80c7c264ad7c7a7156b575dc1732947d808fcc2a70d4d4ed8e336d174748934f` |
| `code_object:sol_137678.hsaco` | `80c7c264ad7c7a7156b575dc1732947d808fcc2a70d4d4ed8e336d174748934f` |
| `code_object:sol_175415.hsaco` | `80c7c264ad7c7a7156b575dc1732947d808fcc2a70d4d4ed8e336d174748934f` |
| `path` | `.sanitizer-nightly/rocjitsu-build/tools/rj_waitcheck` |
| `sha256` | `8431040601e79204efb642cff6bb71c59a0c510fed1de0514a518b84defab0e0` |

## Files here

See `index.html` for the browsable list. `sanitizer_report.json` is the full `aorta.sanitizer_report/0.1` document the dashboard renders from.
