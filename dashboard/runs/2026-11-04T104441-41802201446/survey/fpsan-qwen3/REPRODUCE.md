# Reproduce `fpsan-qwen3`

Sanitizer case `fpsan-qwen3` from run `2026-11-04T104441-41802201446` of the AORTA sanitizer nightly. This directory is the run area for that one case: its report, the sanitizer output the verdict came from, and the provenance below. See `index.html` for every file actually published here.

## Run

- Commit: `6ad9b1aefedddab05f565936ce30af95c57d8291`
- Date: 2026-11-04 10:44:41 UTC
- Target: `gfx950`
- Class: survey (observed-only, non-gating)
- Workflow run: https://github.com/ROCm/aorta/actions/runs/41802201446
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
python3 scripts/sanitizers/model_survey.py run --sanitizers fpsan --inventory recipes/sanitizers/models/qwen3.yaml --out results
```

## Recorded digests

| Key | Value |
|---|---|
| `_fwd_kernel/C0.candidate_digest` | `dba12a952bdea343f9944e2680594bb8c18a78f5d7f4de2d691e73c45cb7c2a8` |
| `_fwd_kernel/C0.reference_digest` | `dba12a952bdea343f9944e2680594bb8c18a78f5d7f4de2d691e73c45cb7c2a8` |
| `_fwd_kernel/C0.verdict` | `pass` |
| `_fwd_kernel/C1.candidate_digest` | `3929fc52b083d4e078c140e3772c5ce1ccf865f6ad974f36c2ad1804f01cd0a5` |
| `_fwd_kernel/C1.reference_digest` | `3929fc52b083d4e078c140e3772c5ce1ccf865f6ad974f36c2ad1804f01cd0a5` |
| `_fwd_kernel/C1.verdict` | `pass` |
| `_topk_topp_kernel/C0.candidate_digest` | `a7f866a7262c7c7eac4d92dc9d4c4ccd0adcd766bafad910a740344a8efe6d7c` |
| `_topk_topp_kernel/C0.reference_digest` | `a7f866a7262c7c7eac4d92dc9d4c4ccd0adcd766bafad910a740344a8efe6d7c` |
| `_topk_topp_kernel/C0.verdict` | `pass` |
| `code_object:dispatched.hsaco` | `336ef011aa1ac84be3ecafe614eb3cbe9a3022995b0c31da6113dbec7d38473b` |
| `result_marker` | `AORTA_FPSAN_RESULT` |
| `triton_instrumentation_mode` | `fpsan` |

## Files here

See `index.html` for the browsable list. Logs are gzipped; `sanitizer_report.json` is the full `aorta.sanitizer_report/0.1` document the dashboard renders from.
