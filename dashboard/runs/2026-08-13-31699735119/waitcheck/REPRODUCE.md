# Reproduce `waitcheck`

Sanitizer case `waitcheck` from run `2026-08-13-31699735119` of the AORTA sanitizer nightly. This directory is the run area for that one case: its report, the sanitizer output the verdict came from, and the provenance below. Its logs, recipe copy and inputs were pruned for falling outside the nightly's log-retention window. See `index.html` for every file actually published here.

## Run

- Commit: `f1660c492df6c68bf6750582652688ebbea181c6`
- Date: 2026-08-13
- Target: `gfx950`
- Class: guardrail (gated guardrail)
- Workflow run: https://github.com/ROCm/aorta/actions/runs/31699735119

## Observed

- Verdict: `warn`
- Execution: `complete`
- Findings: 64

## Run it yourself

First, the checkout this run used:

```
git clone https://github.com/ROCm/aorta && cd aorta
git checkout f1660c492df6c68bf6750582652688ebbea181c6
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
| `code_object:sol_126578.hsaco` | `c1190d2005185bbf797b913022a681d52fb8ea90748901495dde56cc06f359b3` |
| `code_object:sol_137678.hsaco` | `f5935e693790333b8add9346dec9844364d10b6cc144a79e1798a619979f950a` |
| `code_object:sol_175415.hsaco` | `c1190d2005185bbf797b913022a681d52fb8ea90748901495dde56cc06f359b3` |
| `path` | `/workspace/aorta/.sanitizer-nightly/rocjitsu-prebuilt/bin/rj_waitcheck` |
| `sha256` | `5c02eee0055cfa17524ceebece462936da2fcccbc629dad82cd1c388ec0909e3` |

## Files here

See `index.html` for the browsable list. `sanitizer_report.json` is the full `aorta.sanitizer_report/0.1` document the dashboard renders from.
