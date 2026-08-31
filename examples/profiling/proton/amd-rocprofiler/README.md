# amd-rocprofiler — the rocprofiler-sdk backend, with PC sampling

> **This example needs Triton built from upstream `main`. No released Triton
> has the `rocprofiler` backend**, so unlike every other example in this tree
> **its capture has never been verified** — no obtainable wheel or container
> image provides the backend. What *was* exercised on a released Triton
> (3.7.1): the payload itself, its self-check, and the availability guard
> below, which fails the trial with the message it promises rather than a
> traceback. See [Availability](#availability) for the evidence and the check
> to run first.

One transcendental-heavy Triton GELU launched in a loop, captured by Proton's
`rocprofiler` backend with `backend_mode: "pcsampling"`. Where
[`../amd-roctracer`](../amd-roctracer/README.md) gives you whole-kernel spans —
this kernel took *N* microseconds — PC sampling gives you statistical
instruction-level attribution from the rocprofiler-sdk: *where inside* the
kernel the samples landed, sampled periodically rather than instrumented. The
payload is a loop of erf-based GELU launches precisely so there is a steady
stream of instructions to sample.

## Check this first

```bash
python -c "from triton._C.libproton import proton as p; print(p.get_available_profilers())"
```

- A list containing `rocprofiler` — you can run this example.
- A list without it — your Triton has the registry but not the backend.
- **`AttributeError`** — the installed Triton predates the feature entirely.
  Upstream builds the CLI's `-b` choices from this very function; a Triton
  without it has no backend registry at all. This is what every released Triton
  does.

The payload runs the same check itself and exits **2** — the "this environment
cannot run this" code it also uses for a missing GPU, so a failed trial reads
as an environment problem rather than a bad result — with a message naming the
requirement:

```
gelu: Triton 3.7.1 has no libproton.get_available_profilers, so its Proton
predates the backend registry and cannot offer 'rocprofiler' (it has only
['cupti', 'instrumentation', 'roctracer']). This example needs Triton built
from upstream main; no released wheel or container image ships the rocprofiler
backend.
```

The check is answered from the pre-registry backend set when
`get_available_profilers` is missing, rather than refusing every explicit
backend: `--backend roctracer` still profiles fine on Triton 3.7.1, and the
message exists to name the fix, not to be conservative.

## Availability

The evidence, from querying the images and indexes directly:

| Source | Triton | Proton `-b` choices |
|---|---|---|
| `rocm/pytorch:rocm7.14_...pytorch_release_2.12.0` | 3.7.1 | `cupti`, `roctracer`, `instrumentation` |
| Triton 3.7.0 | 3.7.0 | same three |
| Triton 3.6.0+rocm7.2.4 | 3.6.0 | same three |
| PyPI `triton` (latest) | 3.7.1 | same three |
| PyTorch ROCm nightly | `pytorch-triton-rocm` 3.6.0 | same three |
| Upstream `main` | — | from `libproton.get_available_profilers()`; **includes `rocprofiler`**, modes `[None, "pcsampling", "periodic_flushing"]` |

So `roctracer` — deprecated upstream — remains the only AMD backend you can
name on a released Triton, which is why
[`../amd-roctracer`](../amd-roctracer/README.md) is the example to start from
and this one is the forward-looking sibling.

## Requirements

| | |
|---|---|
| Runtime | Triton + PyTorch built for ROCm, one AMD GPU |
| Triton version | **Upstream `main`, built from source.** Released Triton does not have the backend |
| Profiler | Proton, which ships inside Triton — no separate install |
| ROCm | rocprofiler-sdk available to the Triton build |
| Python deps | `torch`, `triton` |

## Run it in a container

The image has to carry a `main`-built Triton, so the usual `rocm/pytorch:latest`
will not do — build Triton from source inside it, or start from an image that
already did:

```bash
docker run --rm -it \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host \
  --security-opt seccomp=unconfined \
  -v "$PWD:/work" -w /work \
  <image-with-triton-main> \
  bash -lc '
    python -c "from triton._C.libproton import proton as p; print(p.get_available_profilers())" &&
    pip install -e . &&
    aorta sweep run \
      --recipe examples/profiling/proton/amd-rocprofiler/recipe.yaml \
      --output ./profiling_results \
      -- python examples/profiling/proton/amd-rocprofiler/gelu.py \
           --size 4194304 --iters 50
  '
```

Pin the image by digest for anything you intend to compare over time.

## Run standalone

```bash
python examples/profiling/proton/amd-rocprofiler/gelu.py --size 4194304 --iters 50
```

Options: `--size`, `--iters`, `--backend`, `--backend-mode`. Output:

```
gelu: device=...
gelu: size=4194304 iters=50 triton=...
gelu: max_abs_err=...
gelu: PASS
```

**This path works on any released Triton.** With no `--backend` and no
`AORTA_PROTON_*` in the environment the payload takes no capture at all — it
just runs the kernel and checks it. That is deliberate: you can confirm the
payload itself is sound on the Triton you actually have, and only the capture
is gated on the backend.

Tolerance is `1e-6` absolute against `torch.nn.functional.gelu`, not exact
equality: the device `erf` and torch's fused activation differ in the last
bits. The op is elementwise, so nothing accumulates and the ceiling stays
tight — measured error is around `2.4e-07`. The comparison is written as
`not (err <= tol)` so a NaN fails instead of passing.

## Run standalone under Proton

This payload drives Proton itself, so there is no `python -m
triton.profiler.proton` wrapper to add:

```bash
python examples/profiling/proton/amd-rocprofiler/gelu.py \
  --backend rocprofiler --backend-mode pcsampling
proton-viewer -m time/s gelu.hatchet
```

To rehearse exactly what aorta does, export the bundle by hand:

```bash
mkdir -p ./proton_out
env AORTA_PROTON_DIR=./proton_out \
    AORTA_PROTON_NAME=./proton_out/proton \
    AORTA_PROTON_CONTEXT=shadow \
    AORTA_PROTON_DATA=tree \
    AORTA_PROTON_BACKEND=rocprofiler \
    AORTA_PROTON_MODE=pcsampling \
  python examples/profiling/proton/amd-rocprofiler/gelu.py
```

Substituting `--backend roctracer` is a useful control on a released Triton:
same payload, same env-mode plumbing, whole-kernel spans instead of samples.

## What you get

A `.hatchet` JSON tree in the trial's `proton/` directory, reported as
`proton_artifact_dir`. Whether the collector's numeric metrics
(`proton_kernel_count`, `proton_gpu_time_ms`, `proton_top_kernel_ms`) also
appear depends on whether a PC-sampling tree carries `time (<unit>)` leaves —
aorta's parser keys on that metric and nothing else. **We have not been able to
check**, having no Triton with the backend, so treat the artifact as the
deliverable and read it with `proton-viewer` in the same environment as the
capture. The comparable `roctracer` capture of the same payload does publish
all three.

## Notes

- **`backend_mode`, not `instrumentation_mode`.** Both render Proton's single
  `--mode`, so the schema makes them mutually exclusive, and `backend_mode`
  requires an explicit (non-`auto`) backend. Valid values here are
  `pcsampling` and `periodic_flushing`; `roctracer` accepts only
  `periodic_flushing`.
- **PC sampling is statistical.** It reports where samples landed, not an exact
  per-instruction cost, so short runs are noisy. That is why the payload
  defaults to 50 launches rather than one. For deterministic intra-kernel
  attribution on a backend that exists today, use
  [`../amd-instrumentation`](../amd-instrumentation/README.md) instead — it
  counts cycles per source-level scope rather than sampling.
- **Why `mode: env`.** Two reasons, both mechanical. Triton's Proton CLI calls
  `_select_backend()` — which initialises the HIP driver as a side effect —
  only when `-b` is absent, so pinning a queue-tracing backend on the CLI
  attaches the profiler before the runtime exists and writes a hatchet with a
  bare `ROOT` frame while exiting 0. And the shipped CLI parses `-m/--mode`
  without forwarding it to `start()`, so `backend_mode` would be a silent
  no-op there. In `mode: env` the payload imports `torch` first and then passes
  both values to `proton.start()` itself. The collector rejects `mode: cli`
  with an explicit `roctracer` / `rocprofiler` backend outright.
- **Device selection.** Proton on AMD reads `ROCR_VISIBLE_DEVICES` and rejects
  `HIP_VISIBLE_DEVICES` / `CUDA_VISIBLE_DEVICES` for the queue-intercepting
  backends. aorta's collector translates the rejected spellings automatically
  and logs a warning; a standalone run has to unset them yourself.
- **Do not stack this with `rocprof`.** `rocprofiler` installs an HSA queue
  interceptor and so does `rocprofv3`; the pairing is rejected at recipe load.
  Only `backend: instrumentation` coexists.

## Provenance

The kernel is original to this repository — not adapted from a Triton
tutorial. It is the exact-`erf` GELU, chosen because its expansion is many
instructions per element, which is what makes a sampling profiler's output
interesting rather than uniformly a load.
