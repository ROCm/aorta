# amd-roctracer — kernel-level attribution on the portable AMD backend

Three distinct Triton kernels — an elementwise scale, a fused bias+GELU, and a
row-sum reduction — launched in sequence, captured by Proton's `roctracer`
backend. This is the example to reach for when you want to know *which kernel*
in a launch sequence owns the GPU time, on the *whole-kernel* AMD backend that
exists in every released Triton — including the releases that predate
`rocprofiler`, which arrived in 3.8.0. (`instrumentation` ships in every release
too, but it measures *inside* one kernel — see
[`../amd-instrumentation`](../amd-instrumentation/README.md).)

It differs from [`triton-vecadd`](../triton-vecadd/README.md) and
[`triton-softmax`](../triton-softmax/README.md) in two ways that matter: the
payload is a real multi-kernel sequence rather than a single launch, and the
recipe pins the backend explicitly through `mode: env` instead of leaving
`backend: "auto"` on `mode: cli`. The second is not a style choice — see
[Why `mode: env`](#why-mode-env).

## Requirements

| | |
|---|---|
| Runtime | Triton + PyTorch built for ROCm, one AMD GPU |
| Profiler | Proton, which ships inside Triton — no separate install |
| Triton version | **Any released Triton.** `roctracer` has been in Proton's backend list since it was introduced, so this example does not care which release you have. `rocprofiler` needs 3.8.0 or newer |
| Python deps | `torch`, `triton` |

## Run it in a container

```bash
docker run --rm -it \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host \
  --security-opt seccomp=unconfined \
  -v "$PWD:/work" -w /work \
  rocm/pytorch:latest \
  bash -lc '
    pip install -e . &&
    aorta sweep run \
      --recipe examples/profiling/proton/amd-roctracer/recipe.yaml \
      --output ./profiling_results \
      -- python examples/profiling/proton/amd-roctracer/pipeline.py \
           --rows 4096 --cols 1024 --iters 20
  '
```

Pin the image by digest for anything you intend to compare over time. A ROCm
venv with `torch` + `triton` on the host works the same way.

## Run standalone

```bash
python examples/profiling/proton/amd-roctracer/pipeline.py --rows 4096 --cols 1024 --iters 20
```

Options: `--rows`, `--cols`, `--iters`, `--backend`. Output:

```
pipeline: device=...
pipeline: rows=4096 cols=1024 iters=20
pipeline: max_rel_err=...
pipeline: PASS
```

With no `--backend` and no `AORTA_PROTON_*` in the environment the payload
takes **no capture at all** — it just runs the three kernels and checks them.
That is the fast way to separate "my payload is broken" from "my profiler is
broken".

The check is a bounded *relative* error (ceiling `1e-5`) against
`torch.nn.functional.gelu(x * scale + bias).sum(dim=-1)`, not exact equality:
the reduction kernel sums `--cols` float32 values in a different order than
`Tensor.sum`, so the low bits differ legitimately. Measured error on the
defaults is around `1.9e-07`, four orders of magnitude inside the ceiling, so a
failure here means a real bug rather than drift. The comparison is written as
`not (err <= tol)` so a NaN fails instead of passing.

The activation and the reduction each use one program per row, with the block
extent rounded up to the next power of two, so a very large `--cols` asks
Triton for a block it may not find registers for. Keep `--cols` at or below a
few thousand.

## Run standalone under Proton

This payload drives Proton itself, so there is no `python -m
triton.profiler.proton` wrapper to add — name the backend and it profiles:

```bash
python examples/profiling/proton/amd-roctracer/pipeline.py --backend roctracer
proton-viewer -m time/s pipeline.hatchet
```

To rehearse exactly what aorta does, export the bundle by hand instead:

```bash
mkdir -p ./proton_out
env AORTA_PROTON_DIR=./proton_out \
    AORTA_PROTON_NAME=./proton_out/proton \
    AORTA_PROTON_CONTEXT=shadow \
    AORTA_PROTON_DATA=tree \
    AORTA_PROTON_BACKEND=roctracer \
  python examples/profiling/proton/amd-roctracer/pipeline.py --iters 20
```

`AORTA_PROTON_NAME` present is what the payload treats as "aorta asked for a
capture"; the rest of the variables fall back to the collector's own defaults
when absent, so a partial bundle still behaves.

## What you get

A `.hatchet` JSON tree in the trial's `proton/` directory, reported as
`proton_artifact_dir`, plus the collector's numeric metrics —
`proton_kernel_count`, `proton_gpu_time_ms` and `proton_top_kernel_ms` — in
`perf.md` and `matrix.json::cells[*].metrics_summary`.

The point of this example is what the tree's shape tells you. With
`context: "shadow"` each Triton kernel gets its own leaf under `ROOT`, so the
capture names all three:

- `proton_top_kernels` (per-trial dispatcher JSON, `.result.metrics` — not
  `perf.md`, which carries numeric scalars only) lists `scale_kernel`,
  `bias_gelu_kernel` and `row_sum_kernel`, ranked by exclusive GPU time.
- `proton_kernel_count` is non-zero and **predictable**: the correctness pass
  runs before `proton.start()`, so the capture holds exactly `--iters` launches
  of each of the three kernels. At the default `--iters 20` that is 60.
- `proton_gpu_time_ms` is non-zero and is the sum of the three leaves;
  `proton_top_kernel_ms` is whichever kernel dominates.

Expect the fused activation to be the most expensive of the three and the
reduction the cheapest — it reads the same bytes but writes one value per row
instead of a full matrix. Read the raw tree with `proton-viewer -m time/s
<file>.hatchet`, in the same environment as the capture.

## Why `mode: env`

Pinning `roctracer` through `mode: cli` **silently captures nothing**, and the
mechanism is worth knowing because it is not a bug you would guess from the
output. Triton's Proton CLI front-end resolves the backend like this:

```python
backend = args.backend if args.backend else _select_backend()
```

`_select_backend()` initialises the Triton HIP driver as a side effect — and it
is only called when `-b` is *absent*. roctracer records nothing unless it
starts after the HIP runtime is up, so with `-b roctracer` the profiler
attaches to a runtime that does not exist yet. The run still exits 0 and still
writes a hatchet: a ~160-byte one, holding a bare `ROOT` frame with empty
metrics. aorta's parser finds no `time (<unit>)` leaves, degrades to
`proton_artifact_dir`, and the trial carries no Proton metrics while looking
like a success.

Upgrading Triton does not help. That line is line 73 of
`third_party/proton/proton/proton.py` at the `v3.8.0` tag, unchanged, so 3.8.0
skips the driver-initialising call on the `-b` path exactly as 3.7.1 does.

`mode: env` avoids this entirely. aorta exports `AORTA_PROTON_*` and leaves
argv alone; the payload imports `torch` at module scope — which brings the HIP
runtime up — and only then calls `proton.start(backend=...)` itself. The
collector enforces this: `mode: cli` with an explicit `roctracer` backend
raises `ProtonWrapError` and names `mode: env` as the route, so a `mode: cli`
version of this recipe would not run at all.

The guard stops at `roctracer`. `rocprofiler` looks like the same case and is
the opposite one: it is configured by an `__attribute__((constructor))` when
`libproton.so` loads, so it wants to be set up *before* HSA rather than after,
and its CLI pin is allowed for that reason. The import order in this example's
payload — torch first, `proton.start()` after — is therefore the reverse of
[`../amd-rocprofiler/gelu.py`](../amd-rocprofiler/gelu.py)'s, deliberately.
See [Pinning an explicit AMD
backend](../../../../docs/profiling-collectors.md#pinning-an-explicit-amd-backend)
for both contracts side by side, including which one is measured and which is
taken from upstream's source.

## Notes

- **Why `roctracer` and not `rocprofiler`.** `roctracer` is deprecated
  upstream in favour of the rocprofiler-sdk backend, and as of Triton 3.8.0 it
  is no longer the only whole-kernel AMD backend you can name on a release. It
  is still the one present in *every* release: 3.6.0, 3.7.0 and 3.7.1 offer
  `cupti` / `roctracer` / `instrumentation` and nothing else — of those `cupti`
  is NVIDIA's and `instrumentation` measures inside a kernel rather than timing
  it — while 3.8.0 adds `rocprofiler` alongside them. So this example is the one
  that runs on whatever image you have, and
  [`../amd-rocprofiler`](../amd-rocprofiler/README.md) is the same idea on the
  newer backend; that one still needs a post-3.8 `main` build, for its
  `pcsampling` mode rather than for the backend.
- **Device selection.** Proton on AMD reads `ROCR_VISIBLE_DEVICES` and
  *rejects* `HIP_VISIBLE_DEVICES` and `CUDA_VISIBLE_DEVICES` outright for
  `roctracer` — `proton.start()` raises `ValueError` before any kernel runs.
  aorta's collector translates the rejected spellings automatically and logs a
  warning; a standalone run has to unset them yourself.
- **Do not stack this with `rocprof`.** `roctracer` installs an HSA queue
  interceptor and so does `rocprofv3`; the pairing is rejected at recipe load.
  Only [`../amd-instrumentation`](../amd-instrumentation/README.md) coexists
  with `rocprof`.
- **`Could not load libroctracer64.so`.** Some container images get ROCm from
  Python wheels, which ship only `libroctracer64.so.4` while Proton `dlopen`s
  the unversioned name. Add a directory of unversioned symlinks to
  `$LD_LIBRARY_PATH`, or use an image with a system ROCm install. This example
  pins `roctracer`, so that is the name you will see here; the sibling
  `rocprofiler` example (and `backend: auto` from Triton 3.8.0) fails on
  `librocprofiler-sdk.so` instead, with the same remedy.
- Triton compiles on first launch. The unprofiled correctness pass absorbs that
  compile, which is why the capture's launch count is exact.

## Provenance

The three kernels are original to this repository — not adapted from a Triton
tutorial. They are deliberately ordinary shapes (elementwise, per-row
activation, per-row reduction) so the tree reads like the front of a
transformer block without needing one.
