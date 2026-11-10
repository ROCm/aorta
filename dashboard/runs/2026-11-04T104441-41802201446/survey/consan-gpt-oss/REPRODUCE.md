# Reproduce `consan-gpt-oss`

Sanitizer case `consan-gpt-oss` from run `2026-11-04T104441-41802201446` of the AORTA sanitizer nightly. This directory is the run area for that one case: its report, the sanitizer output the verdict came from, and the provenance below. Its logs, recipe copy and inputs were pruned for falling outside the nightly's log-retention window. See `index.html` for every file actually published here.

## Run

- Commit: `6ad9b1aefedddab05f565936ce30af95c57d8291`
- Date: 2026-11-04 10:44:41 UTC
- Target: `gfx950`
- Class: survey (observed-only, non-gating)
- Workflow run: https://github.com/ROCm/aorta/actions/runs/41802201446
- Container image: `rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0@sha256:3174cb7061d94c427da96c0edef4adea28046fa3f3b2ff3948dc4e995665ff8c`
- rocjitsu bundle: `d03c6fe410eed62b211d56bb7878dd1f494269fe` (https://github.com/ROCm/rocm-systems/actions/runs/37841600188)

## Observed

- Verdict: `fail`
- Execution: `complete`
- Findings: 49

## Run it yourself

First, the checkout this run used:

```
git clone https://github.com/ROCm/aorta && cd aorta
git checkout 6ad9b1aefedddab05f565936ce30af95c57d8291
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
| `code_object:dispatched.hsaco` | `d5458dc446e25aab98bdd40cfb51d76bb19d8e5d5682fbae69397a5468482ab4` |
| `command` | `/workspace/aorta/.sanitizer-nightly/model-survey-work/gpt-oss/cases/_rope_kernel_cached_thd_2c_gqa_onehead_fwd__P1/launch.sh` |
| `command_sha256` | `e9113a38e9136643150a7fc7a7a7d8175bb0de748f925391b72f44af5f24e09d` |
| `hook` | `/workspace/aorta/.sanitizer-nightly/rocjitsu-prebuilt/lib/librocjitsu_dbi_hooks.so` |
| `hook_sha256` | `e933343155d0e049b683c5f589987b01c2f2ac44f725f3fa814fb2a62bcf11fe` |
| `selected_identity_sha256` | `10a061331a7b4beb24989c0333b87d07d319d2f18b76294cc3446e4530c1e907` |
| `selected_kernel` | `_rope_kernel_cached_thd_2c_gqa_onehead_fwd` |

## Files here

See `index.html` for the browsable list. `sanitizer_report.json` is the full `aorta.sanitizer_report/0.1` document the dashboard renders from.
