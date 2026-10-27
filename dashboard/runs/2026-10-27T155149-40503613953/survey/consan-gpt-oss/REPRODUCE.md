# Reproduce `consan-gpt-oss`

Sanitizer case `consan-gpt-oss` from run `2026-10-27T155149-40503613953` of the AORTA sanitizer nightly. This directory is the run area for that one case: its report, the sanitizer output the verdict came from, and the provenance below. See `index.html` for every file actually published here.

## Run

- Commit: `4fb536db06a8e0d6dc6e21cc2658505e8a1360f7`
- Date: 2026-10-27 15:51:49 UTC
- Target: `gfx950`
- Class: survey (observed-only, non-gating)
- Workflow run: https://github.com/ROCm/aorta/actions/runs/40503613953
- Container image: `rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0@sha256:3174cb7061d94c427da96c0edef4adea28046fa3f3b2ff3948dc4e995665ff8c`
- rocjitsu bundle: `d03c6fe410eed62b211d56bb7878dd1f494269fe` (https://github.com/ROCm/rocm-systems/actions/runs/37841600188)

## Observed

- Verdict: `fail`
- Execution: `complete`
- Findings: 51

## Run it yourself

First, the checkout this run used:

```
git clone https://github.com/ROCm/aorta && cd aorta
git checkout 4fb536db06a8e0d6dc6e21cc2658505e8a1360f7
pip install -e .
```

Set `ROCJITSU_PREBUILT` (unpacked rocjitsu bundle supplying rj_waitcheck and the ConSan hook) before running. `aorta` sets `HSA_TOOLS_LIB`, `HSA_TOOLS_DISABLE_REGISTER`, `RJ_CONSAN_MODE`, `RJ_CONSAN_PRESET`, `RJ_CONSAN_POLICY` for you from the recipe's policy block.

Finally, the sweep itself:

```
python scripts/sanitizers/model_survey.py run --inventory recipes/sanitizers/models/gpt-oss.yaml --out results
```

## Recorded digests

| Key | Value |
|---|---|
| `code_object:dispatched.hsaco` | `a3e32aef6c272336bc5c3a17f7712dfec6da7fdd0564da68c3dcc1164dcd0b18` |
| `command` | `/workspace/aorta/.sanitizer-nightly/model-survey-work/gpt-oss/cases/_masked_compaction__P1/launch.sh` |
| `command_sha256` | `dea22f598f276ea56d373117ef8c046a73c768e4e132cf9c774b7af8e4c3d4cb` |
| `hook` | `/workspace/aorta/.sanitizer-nightly/rocjitsu-prebuilt/lib/librocjitsu_dbi_hooks.so` |
| `hook_sha256` | `e933343155d0e049b683c5f589987b01c2f2ac44f725f3fa814fb2a62bcf11fe` |
| `selected_identity_sha256` | `3a4caf79bff13f6d18ad59bedc19747a000cb0b4660c4c660907b74d5384ac48` |
| `selected_kernel` | `_masked_compaction` |

## Files here

See `index.html` for the browsable list. Logs are gzipped; `sanitizer_report.json` is the full `aorta.sanitizer_report/0.1` document the dashboard renders from.
