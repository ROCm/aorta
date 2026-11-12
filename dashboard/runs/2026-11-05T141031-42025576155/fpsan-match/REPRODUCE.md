# Reproduce `fpsan-match`

Sanitizer case `fpsan-match` from run `2026-11-05T141031-42025576155` of the AORTA sanitizer nightly. This directory is the run area for that one case: its report, the sanitizer output the verdict came from, and the provenance below. Its logs, recipe copy and inputs were pruned for falling outside the nightly's log-retention window. See `index.html` for every file actually published here.

## Run

- Commit: `6ad9b1aefedddab05f565936ce30af95c57d8291`
- Date: 2026-11-05 14:10:31 UTC
- Target: `gfx950`
- Class: guardrail (gated guardrail)
- Workflow run: https://github.com/ROCm/aorta/actions/runs/42025576155
- Container image: `rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0@sha256:3174cb7061d94c427da96c0edef4adea28046fa3f3b2ff3948dc4e995665ff8c`
- rocjitsu bundle: `d03c6fe410eed62b211d56bb7878dd1f494269fe` (https://github.com/ROCm/rocm-systems/actions/runs/37841600188)

## Observed

- Verdict: `pass`
- Execution: `complete`
- Findings: 0

## Run it yourself

First, the checkout this run used:

```
git clone https://github.com/ROCm/aorta && cd aorta
git checkout 6ad9b1aefedddab05f565936ce30af95c57d8291
pip install -e .
```

`aorta` sets `TRITON_INSTRUMENTATION_MODE=fpsan` for you from the recipe.

Finally, the sweep itself:

```
aorta sweep run --recipe recipes/sanitizers/daily-fpsan-match.yaml
```

## Recorded digests

| Key | Value |
|---|---|
| `candidate_digest` | `53b41a29b1abfb54fc74ec44947454297a4ea4aee00cb832b55331f894b42455` |
| `candidate_instrumented` | `true` |
| `command` | `/workspace/aorta/recipes/sanitizers/fixtures/fpsan/fpsan_match.py` |
| `command_sha256` | `9c4b643874004a469230e7c78510ee6915be31ba429d0cb74b6ef8eb16a83e8a` |
| `reference_digest` | `53b41a29b1abfb54fc74ec44947454297a4ea4aee00cb832b55331f894b42455` |
| `reference_instrumented` | `true` |
| `reported_target` | `gfx950` |
| `result_marker` | `AORTA_FPSAN_RESULT` |
| `selected_kernels` | `row_dot` |
| `triton_instrumentation_mode` | `fpsan` |

## Files here

See `index.html` for the browsable list. `sanitizer_report.json` is the full `aorta.sanitizer_report/0.1` document the dashboard renders from.
