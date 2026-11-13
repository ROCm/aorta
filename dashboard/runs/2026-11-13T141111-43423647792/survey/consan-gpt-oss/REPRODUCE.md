# Reproduce `consan-gpt-oss`

Sanitizer case `consan-gpt-oss` from run `2026-11-13T141111-43423647792` of the AORTA sanitizer nightly. This directory is the run area for that one case: its report, the sanitizer output the verdict came from, and the provenance below. See `index.html` for every file actually published here.

## Run

- Commit: `21d7befb8a8462c9de98c1d2c8ddafc2df0e1ee3`
- Date: 2026-11-13 14:11:11 UTC
- Target: `gfx950`
- Class: survey (observed-only, non-gating)
- Workflow run: https://github.com/ROCm/aorta/actions/runs/43423647792
- Container image: `rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0@sha256:3174cb7061d94c427da96c0edef4adea28046fa3f3b2ff3948dc4e995665ff8c`
- rocjitsu bundle: `d03c6fe410eed62b211d56bb7878dd1f494269fe` (https://github.com/ROCm/rocm-systems/actions/runs/43418385944)

## Observed

- Verdict: `fail`
- Execution: `complete`
- Findings: 49

## Run it yourself

First, the checkout this run used:

```
git clone https://github.com/ROCm/aorta && cd aorta
git checkout 21d7befb8a8462c9de98c1d2c8ddafc2df0e1ee3
pip install -e .
```

Set `ROCJITSU_PREBUILT` (unpacked rocjitsu bundle supplying rj_waitcheck and the ConSan hook) before running. `aorta` sets `HSA_TOOLS_LIB`, `HSA_TOOLS_DISABLE_REGISTER`, `RJ_CONSAN_MODE`, `RJ_CONSAN_PRESET`, `RJ_CONSAN_POLICY` for you from the recipe.

Finally, the sweep itself:

```
python scripts/sanitizers/model_survey.py run --inventory recipes/sanitizers/models/gpt-oss.yaml --out results
```

## Recorded digests

| Key | Value |
|---|---|
| `code_object:dispatched.hsaco` | `44f4a73c86466325840d1e902309b84ccfe320a3cf0d8560482ce7385ade7d97` |
| `command` | `/workspace/aorta/.sanitizer-nightly/model-survey-work/gpt-oss/cases/_combine_sampled_and_draft_tokens_kernel__P1/launch.sh` |
| `command_sha256` | `1e966c3fd5e102d8e7a81bdf6e554ef5bee3c839acd9bf25496f7b3223f9de39` |
| `hook` | `/workspace/aorta/.sanitizer-nightly/rocjitsu-prebuilt/lib/librocjitsu_dbi_hooks.so` |
| `hook_sha256` | `e933343155d0e049b683c5f589987b01c2f2ac44f725f3fa814fb2a62bcf11fe` |
| `selected_identity_sha256` | `0b70d7997fb530070e01391454ecae90b89d86e403d33a376987c296031552ac` |
| `selected_kernel` | `_combine_sampled_and_draft_tokens_kernel` |

## Files here

See `index.html` for the browsable list. Logs are gzipped; `sanitizer_report.json` is the full `aorta.sanitizer_report/0.1` document the dashboard renders from.
