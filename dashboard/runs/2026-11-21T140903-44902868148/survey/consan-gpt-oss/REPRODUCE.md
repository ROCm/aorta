# Reproduce `consan-gpt-oss`

Sanitizer case `consan-gpt-oss` from run `2026-11-21T140903-44902868148` of the AORTA sanitizer nightly. This directory is the run area for that one case: its report, the sanitizer output the verdict came from, and the provenance below. See `index.html` for every file actually published here.

## Run

- Commit: `576ff2facfbbbbf5a5b7e61e4ef8fc5fe79d709c`
- Date: 2026-11-21 14:09:03 UTC
- Target: `gfx950`
- Class: survey (observed-only, non-gating)
- Workflow run: https://github.com/ROCm/aorta/actions/runs/44902868148
- Container image: `rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0@sha256:3174cb7061d94c427da96c0edef4adea28046fa3f3b2ff3948dc4e995665ff8c`
- rocjitsu bundle: `d03c6fe410eed62b211d56bb7878dd1f494269fe` (https://github.com/ROCm/rocm-systems/actions/runs/43418385944)

## Observed

- Verdict: `fail`
- Execution: `complete`
- Findings: 48

## Run it yourself

First, the checkout this run used:

```
git clone https://github.com/ROCm/aorta && cd aorta
git checkout 576ff2facfbbbbf5a5b7e61e4ef8fc5fe79d709c
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
| `code_object:dispatched.hsaco` | `8b64fa68f36e5fa05a1480165b0531dfcc3ec45c8a23e08e866fce25b4483c7f` |
| `command` | `/workspace/aorta/.sanitizer-nightly/model-survey-work/gpt-oss/cases/_matmul_ogs_w2__P1/launch.sh` |
| `command_sha256` | `4b09edfd65f8a11a6499313739f19a04dfeebefa1e208634120cc96bb89f4132` |
| `hook` | `/workspace/aorta/.sanitizer-nightly/rocjitsu-prebuilt/lib/librocjitsu_dbi_hooks.so` |
| `hook_sha256` | `e933343155d0e049b683c5f589987b01c2f2ac44f725f3fa814fb2a62bcf11fe` |
| `selected_identity_sha256` | `82535bbcd4e7af3e1e10a858d9ff65ed8a52b9aad563df40c76eb10c6b83bb25` |
| `selected_kernel` | `_matmul_ogs_NNT_bf16xbf16xmxfp4_32x256x128x1` |

## Files here

See `index.html` for the browsable list. Logs are gzipped; `sanitizer_report.json` is the full `aorta.sanitizer_report/0.1` document the dashboard renders from.
