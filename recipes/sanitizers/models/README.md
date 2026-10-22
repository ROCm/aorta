# Open-source model kernels on the Workload survey tab

These inventories put Triton kernels from open-source models on the Workload
survey tab (Tab 2) of the sanitizer dashboard. The `sanitizers-model-survey`
nightly job runs `scripts/sanitizers/model_survey.py` over each one and
publishes two rows per
model, `waitcheck-<model>` and `consan-<model>`, beside the hipBLASLt GEMM.
Like the rest of Tab 2 they are observed-only and never gate the nightly.

| Inventory | Model | Kernels | (kernel, scenario) pairs |
|---|---|--:|--:|
| `gpt-oss.yaml` | gpt-oss | 23 | 182 |
| `qwen3.yaml` | Qwen3 | 3 | 7 |

## What a nightly does

1. **Launch every pair once.** `launch.py --mode run` builds the scenario's real
   arguments and dispatches the kernel, with a private `TRITON_CACHE_DIR`.
   Triton specialises on runtime values, so the object it compiles here is the
   one the kernel actually dispatches. Identical objects (two scenarios that
   compile to the same bytes) are scanned once.
2. **Waitcheck** scans every launched object in one `kernel_list` sweep.
3. **ConSan** dispatches one scenario per kernel, the first the spec lists, which
   is its most frequent launch shape, against the same cache, so it
   instruments exactly the bytes waitcheck scanned. Kernels are started in a
   nightly rotation, and only while the case timeout still fits in the budget.
   Kernels in `consan.exclude` are never started; each carries its reason.

`model_survey.json` in each published case records what was launched, any
launch failure, which ConSan cases ran, and which were excluded or did not fit
that night.

A kernel with no LDS, barrier, atomic or fence site has nothing for ConSan to
instrument and reads `not_checked` (`consan_not_applicable`). A model where
that is the worst ConSan outcome shows `not_checked` on the summary row, with
every kernel's own verdict in its detail table.

## Kernel sources

The specs, flattened kernel files and the launch harness are not in this repo.
They ship as the `aorta-model-kernels` release asset, which `model_survey.py`
downloads and refuses unless its SHA-256 matches `ASSET_SHA256` in that script.
Its `THIRD_PARTY_NOTICES.md` lists each origin project (vLLM, AITER, Triton's
`triton_kernels`) and its license. Only kernels the model launches when serving
(spec `role` of `launched` or `conditional`) are included.

To build a new asset version, run `scripts/sanitizers/build_model_kernel_asset.py`
and upload the tarball to a `model-kernels-v<N>` release. In the same change, bump
`ASSET_VERSION` and `ASSET_SHA256` in `model_survey.py`, and set `asset_version`
and `expected_pairs` in each inventory.

## Reproducing a row

On a gfx950 host with ROCm, PyTorch and Triton installed, and `ROCJITSU_PREBUILT`
pointing at a rocjitsu sanitizer bundle:

```bash
python scripts/sanitizers/model_survey.py run \
  --inventory recipes/sanitizers/models/gpt-oss.yaml --out results
```
