# Reproduce `consan-qwen3`

Sanitizer case `consan-qwen3` from run `2026-10-30T141145-41054343941` of the AORTA sanitizer nightly. This directory is the run area for that one case: its report, the sanitizer output the verdict came from, and the provenance below. See `index.html` for every file actually published here.

## Run

- Commit: `b789671019359a96a8ccaed21ab8747c4fc74fc0`
- Date: 2026-10-30 14:11:45 UTC
- Target: `gfx950`
- Class: survey (observed-only, non-gating)
- Workflow run: https://github.com/ROCm/aorta/actions/runs/41054343941
- Container image: `rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0@sha256:3174cb7061d94c427da96c0edef4adea28046fa3f3b2ff3948dc4e995665ff8c`
- rocjitsu bundle: `d03c6fe410eed62b211d56bb7878dd1f494269fe` (https://github.com/ROCm/rocm-systems/actions/runs/37841600188)

## Observed

- Verdict: `not_checked`
- Execution: `partial`
- Findings: 0
- Reason: `1 of 2 kernels not checked (1x consan_not_applicable: no LDS access, barrier, atomic or fence sites in any loaded code object)`

## Run it yourself

First, the checkout this run used:

```
git clone https://github.com/ROCm/aorta && cd aorta
git checkout b789671019359a96a8ccaed21ab8747c4fc74fc0
pip install -e .
```

Set `ROCJITSU_PREBUILT` (unpacked rocjitsu bundle supplying rj_waitcheck and the ConSan hook) before running. `aorta` sets `HSA_TOOLS_LIB`, `HSA_TOOLS_DISABLE_REGISTER`, `RJ_CONSAN_MODE`, `RJ_CONSAN_PRESET`, `RJ_CONSAN_POLICY` for you from the recipe's policy block.

Finally, the sweep itself:

```
python scripts/sanitizers/model_survey.py run --inventory recipes/sanitizers/models/qwen3.yaml --out results
```

## Recorded digests

| Key | Value |
|---|---|
| `code_object:dispatched.hsaco` | `336ef011aa1ac84be3ecafe614eb3cbe9a3022995b0c31da6113dbec7d38473b` |
| `command` | `/workspace/aorta/.sanitizer-nightly/model-survey-work/qwen3/cases/_fwd_kernel__C0/launch.sh` |
| `command_sha256` | `2fc3569b85cd33450f416882377d16dd14d8d3bfd60c4e9606501bfd361f7bee` |
| `hook` | `/workspace/aorta/.sanitizer-nightly/rocjitsu-prebuilt/lib/librocjitsu_dbi_hooks.so` |
| `hook_sha256` | `e933343155d0e049b683c5f589987b01c2f2ac44f725f3fa814fb2a62bcf11fe` |
| `selected_identity_sha256` | `a11ba5da82235039869f3199c28eb19dd834c71916c301ede01cc75f3a36ea19` |
| `selected_kernel` | `_fwd_kernel` |

## Files here

See `index.html` for the browsable list. Logs are gzipped; `sanitizer_report.json` is the full `aorta.sanitizer_report/0.1` document the dashboard renders from.
