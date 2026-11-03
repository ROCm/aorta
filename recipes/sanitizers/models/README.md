# Open-source model kernels on the sanitizer dashboard

These inventories put Triton kernels from open-source models on the sanitizer
dashboard. The `sanitizers-model-survey` nightly job runs
`scripts/sanitizers/model_survey.py` over each one and publishes two rows per
model on the Workload survey tab (Tab 2), `waitcheck-<model>` and
`consan-<model>`, beside the hipBLASLt GEMM. The `sanitizers-fpsan-survey` job
runs it again with `--sanitizers fpsan` and publishes a third row,
`fpsan-<model>`, on the FpSan tab. All of them are observed-only and
never gate the nightly.

| Inventory | Model | Kernels | (kernel, scenario) pairs | Rows |
|---|---|--:|--:|---|
| `gpt-oss.yaml` | gpt-oss | 23 | 182 | `waitcheck-`, `consan-`, `fpsan-gpt-oss` |
| `qwen3.yaml` | Qwen3 | 3 | 7 | `waitcheck-`, `consan-`, `fpsan-qwen3` |
| `triton-tutorials.yaml` | Triton tutorials | 4 | 5 | `fpsan-triton-tutorials`, pending asset v2 |

`triton-tutorials.yaml` names asset version 2, which is built but not yet
published, so the pinned v1 asset refuses it and neither nightly job lists it.
It goes live once `model-kernels-v2` is released and pinned (see below).

`--sanitizers` takes a comma-separated subset of `waitcheck,consan,fpsan`
(default `waitcheck,consan`). Selecting `fpsan` requires an `fpsan:` block in
every inventory passed.

## What a nightly does

1. **Launch every pair once.** `launch.py --mode run` builds the scenario's real
   arguments and dispatches the kernel, with a private `TRITON_CACHE_DIR`.
   Triton specialises on runtime values, so the object it compiles here is the
   one the kernel actually dispatches. Identical objects (two scenarios that
   compile to the same bytes) are scanned once. A launch starts only while its
   timeout fits in the budget; after the first wave it must also leave time for
   the model's sanitizer waves.
2. **Waitcheck** scans every launched object in one `kernel_list` sweep.
3. **ConSan** dispatches one scenario per kernel, the first the spec lists, which
   is its most frequent launch shape, against the same cache, so it
   instruments exactly the bytes waitcheck scanned. Kernels are started in a
   nightly rotation, and only while the case timeout still fits in the budget.
   Kernels in `consan.exclude` are never started; each carries its reason.
4. **FpSan** (`--sanitizers fpsan` only) compares every launched pair with its
   reference: one `aorta sweep` per pair, whose `fpsan_command` is
   `scripts/sanitizers/fpsan_compare.py`. Pairs start in a nightly rotation
   while the case timeout fits in the budget. Kernels in `fpsan.exclude` (the
   integer-only ones, which FpSan has nothing to rewrite in) are never started.

`model_survey.json` in each published case records what was launched, any
launch failure, which ConSan or FpSan cases ran, and which were excluded or did
not fit that night.

A kernel with no LDS, barrier, atomic or fence site has nothing for ConSan to
instrument and reads `not_checked` (`consan_not_applicable`). A model where
that is the worst ConSan outcome shows `not_checked` on the summary row, with
every kernel's own verdict in its detail table.

## How FpSan pairs are defined

Triton FpSan rewrites floating-point operations into a payload algebra in which
add and multiply are exactly associative, so two kernels that compute the same
function produce bit-identical sanitized outputs whatever their tiling or
reduction order. `fpsan_compare.py` runs the reference and then the candidate
through the asset's own `launch.py`, unchanged, each in its own empty
`TRITON_CACHE_DIR` with the same seed, SHA-256s every tensor argument of the one
launch, and prints one `AORTA_FPSAN_RESULT` line. A digest mismatch is a `fail`.
Anything that makes the comparison untrustworthy (no launch, more than one, an
uninstrumented side, a reference identical to the candidate) prints no line, so
the pair reads `error`, never `pass`.

- **gpt-oss and Qwen3**: the reference is the same kernel and scenario at a
  conservative launch config, `num_warps=4`, `num_stages=1` and no AMD launch
  kwargs. An `fpsan.reference` entry, keyed by kernel or by
  `<kernel>/<scenario>`, changes that: smaller reduction tiles (`constexprs`),
  another warp count, or a grid. Tile sizes that change how many masked lanes
  are computed stay as captured, because FpSan does not zero masked lanes.
- **Triton tutorials**: each candidate is compared with another spec entry at
  the same scenario (`reference.<kernel>.kernel`), such as the persistent
  matmul against the plain one.

The merged `fpsan-<model>` report keeps one row per pair. Its FpSan check
backend carries `<kernel>/<scenario>.verdict`, `.reason`, and both digests, and
the FpSan tab renders those rows.

Before the survey the job runs `scripts/sanitizers/fpsan_probe.py`, and fails
if this image's Triton does not instrument a probe kernel under FpSan (it needs
Triton 3.8 or later) or if a re-tiled reduction disagrees with itself under it.

## Kernel sources

The specs, flattened kernel files and the launch harness are not in this repo.
They ship as the `aorta-model-kernels` release asset, which `model_survey.py`
downloads and refuses unless its SHA-256 matches `ASSET_SHA256` in that script.
Its `THIRD_PARTY_NOTICES.md` lists each origin project (vLLM, AITER, Triton's
`triton_kernels`, Triton's tutorials) and its license. Only kernels the model
launches when serving (spec `role` of `launched` or `conditional`) are included,
plus the tutorial references.

To build a new asset version, run `scripts/sanitizers/build_model_kernel_asset.py`
and upload the tarball to a `model-kernels-v<N>` release. In the same change, bump
`ASSET_VERSION` and `ASSET_SHA256` in `model_survey.py`, and set `asset_version`
and `expected_pairs` in each inventory. For v2, also add `triton-tutorials` to
`MODELS` in the `sanitizers-fpsan-survey` job.

## Reproducing a row

On a gfx950 host with ROCm, PyTorch and Triton installed, and `ROCJITSU_PREBUILT`
pointing at a rocjitsu sanitizer bundle:

```bash
python scripts/sanitizers/model_survey.py run \
  --inventory recipes/sanitizers/models/gpt-oss.yaml --out results
```

An `fpsan-<model>` row needs Triton 3.8 or later and no rocjitsu bundle:

```bash
python scripts/sanitizers/fpsan_probe.py --out fpsan-probe.json
python scripts/sanitizers/model_survey.py run --sanitizers fpsan \
  --inventory recipes/sanitizers/models/gpt-oss.yaml --out results
```
