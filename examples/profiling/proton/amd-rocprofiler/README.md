# amd-rocprofiler — the rocprofiler-sdk backend, with PC sampling

> **This example needs Triton built from upstream `main`, past the `v3.8.0`
> tag — for the mode, not for the backend.** `rocprofiler` itself is released:
> Triton 3.8.0 (2026-08-28) ships it, and
> `libproton.get_available_profilers()` there returns
> `['cupti', 'rocprofiler', 'roctracer', 'instrumentation']`. Its `pcsampling`
> mode is not. On 3.8.0,
> `proton.start(backend="rocprofiler", mode="pcsampling")` raises
> `ValueError: [PROTON] RocprofSDKProfiler: unsupported mode: pcsampling`,
> because AMD PC sampling landed upstream after that tag. So unlike every other
> example in this tree **its PC-sampling capture has never been verified** — no
> obtainable wheel or container image takes the measurement this recipe asks
> for. What *was* exercised: the payload and its self-check on a released
> Triton (3.7.1), and both guards below, which fail the trial with the message
> they promise rather than a traceback. See [Availability](#availability) for
> the evidence and the checks to run first.

One transcendental-heavy Triton GELU launched in a loop, captured by Proton's
`rocprofiler` backend with `backend_mode: "pcsampling"`. Where
[`../amd-roctracer`](../amd-roctracer/README.md) gives you whole-kernel spans —
this kernel took *N* microseconds — PC sampling gives you statistical
instruction-level attribution from the rocprofiler-sdk: *where inside* the
kernel the samples landed, sampled periodically rather than instrumented. The
payload is a loop of erf-based GELU launches precisely so there is a steady
stream of instructions to sample.

## Check this first

Two independent questions — is the backend present, and does this build's
`rocprofiler` support `pcsampling`? Since 3.8.0 the answers can differ, so ask
both.

```bash
python -c "from triton._C.libproton import proton as p; print(p.get_available_profilers())"
```

- A list containing `rocprofiler` — your Triton has the backend. Triton 3.8.0
  returns `['cupti', 'rocprofiler', 'roctracer', 'instrumentation']`.
- A list without it — your Triton has the registry but not the backend.
- **`AttributeError`** — the installed Triton predates the registry entirely.
  Upstream builds the CLI's `-b` choices from this very function; a Triton
  without it has no backend registry at all. Triton 3.7.1 and earlier are in
  this case.

Then the mode, which that list does not answer:

```bash
python -c "import triton.profiler as proton; proton.start('probe', backend='rocprofiler', mode='pcsampling')"
```

On 3.8.0 this raises
`ValueError: [PROTON] RocprofSDKProfiler: unsupported mode: pcsampling`, and a
post-3.8 `main` build is what the recipe as written needs.
`mode='periodic_flushing'` is accepted on 3.8.0, so it is the way to exercise
the rest of the plumbing on a released Triton.

There is a third question, and on aorta's own CI base it is the one that
answers first. A ROCm that came from Python wheels ships only versioned
sonames, so Proton's `dlopen` of the unversioned `librocprofiler-sdk.so` fails
before any mode is considered. Measured in the ROCm 10 base image that
[#411](https://github.com/ROCm/aorta/pull/411) moves CI to
(`rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0`, Triton
3.8.0, MI350X):

```
gelu: Proton could not start: [PROTON] Could not load `librocprofiler-sdk.so`
```

Both `pcsampling` *and* the `periodic_flushing` that 3.8.0 does support fail
that way there, identically — which is why the payload does not guess. So the
PC-sampling capture stays unverified on that image too, but for a packaging
reason rather than a backend or mode one.

The payload covers all three itself — the availability question up front, the
mode and library questions by classifying what `proton.start()` raises — and
exits **2** for each. That is the "this environment cannot run this" code it
also uses for a missing GPU, so a failed trial reads as an environment problem
rather than a bad result. Anything it does *not* recognise is re-raised as a
traceback rather than explained away, because a confident wrong diagnosis costs
more than an unhandled one. The exits carry different messages, because they
have different fixes. On Triton 3.7.1 the backend is missing:

```
gelu: Triton 3.7.1 has no libproton.get_available_profilers, so its Proton
predates the backend registry and cannot offer 'rocprofiler' (it has only
['cupti', 'instrumentation', 'roctracer']). ...
```

and on 3.8.0 the backend check passes, `proton.start()` raises, and the payload
reports the mode instead:

```
gelu: Proton refused backend='rocprofiler' with mode='pcsampling': [PROTON]
RocprofSDKProfiler: unsupported mode: pcsampling
gelu: the backend exists on this Triton but does not support that mode. AMD
pcsampling landed after the 3.8.0 tag, so it needs an upstream `main` build;
`periodic_flushing` works on 3.8.0.
```

The availability check is answered from the pre-registry backend set when
`get_available_profilers` is missing, rather than refusing every explicit
backend: `--backend roctracer` still profiles fine on Triton 3.7.1, and the
message exists to name the fix, not to be conservative.

## Availability

The evidence, from querying the images and indexes directly and from an
isolated `triton==3.8.0` install:

| Source | Triton | Proton `-b` choices | `rocprofiler` modes |
|---|---|---|---|
| `rocm/pytorch:rocm7.14_...pytorch_release_2.12.0` | 3.7.1 | `cupti`, `roctracer`, `instrumentation` | backend absent |
| Triton 3.7.0 | 3.7.0 | same three | backend absent |
| Triton 3.6.0+rocm7.2.4 | 3.6.0 | same three | backend absent |
| PyTorch ROCm nightly | `pytorch-triton-rocm` 3.6.0 | same three | backend absent |
| PyPI `triton` (latest, released 2026-08-28) | 3.8.0 | `cupti`, **`rocprofiler`**, `roctracer`, `instrumentation` | `periodic_flushing` only — `pcsampling` raises `unsupported mode` |
| Upstream `main`, past `v3.8.0` | — | same four | `pcsampling` and `periodic_flushing` |

Two consequences worth keeping apart. The backend became nameable in 3.8.0, so
a `backend: rocprofiler` recipe now runs on a released Triton — but this
example's `backend_mode: pcsampling` does not, which is why the capture is
still unverified. And `roctracer`, deprecated upstream, is no longer the only
whole-kernel AMD backend on the newest release; it is the one present in
*every* release, including those predating 3.8.0
(`instrumentation` is in every release too, but measures inside a kernel).
That is why [`../amd-roctracer`](../amd-roctracer/README.md) remains the
example to start from on an arbitrary image and this one is the
forward-looking sibling.

## Requirements

| | |
|---|---|
| Runtime | Triton + PyTorch built for ROCm, one AMD GPU |
| Triton version | **Upstream `main` past `v3.8.0`, built from source**, for `backend_mode: pcsampling`. The `rocprofiler` backend alone needs only 3.8.0; `backend_mode: periodic_flushing` runs there |
| Profiler | Proton, which ships inside Triton — no separate install |
| ROCm | rocprofiler-sdk available to the Triton build |
| Python deps | `torch`, `triton` |

## Run it in a container

The image has to carry a `main`-built Triton past `v3.8.0` for the recipe's
`pcsampling` mode, so the usual `rocm/pytorch:latest` will not do — build Triton
from source inside it, or start from an image that already did. An image with a
plain 3.8.0 gets you as far as the backend and then fails the mode check, which
is what the `get_available_profilers()` line below will *not* tell you:

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
is gated on the backend and its mode.

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

On Triton 3.8.0 that command reaches `proton.start()` and stops there with the
unsupported-mode error. Two controls run to completion on a released Triton:
`--backend rocprofiler --backend-mode periodic_flushing` on 3.8.0, which
exercises this backend without asking for PC sampling, and `--backend
roctracer` on any release — same payload, same env-mode plumbing, whole-kernel
spans instead of samples.

## What you get

A `.hatchet` JSON tree in the trial's `proton/` directory, reported as
`proton_artifact_dir`. Whether the collector's numeric metrics
(`proton_kernel_count`, `proton_gpu_time_ms`, `proton_top_kernel_ms`) also
appear depends on whether a PC-sampling tree carries `time (<unit>)` leaves —
aorta's parser keys on that metric and nothing else. **We have not been able to
check**: the newest obtainable Triton (3.8.0) has the backend but refuses
`pcsampling`, so no PC-sampling tree has been produced to look at. Treat the
artifact as the deliverable and read it with `proton-viewer` in the same
environment as the capture. The comparable `roctracer` capture of the same
payload does publish all three.

## Notes

- **Is this the backend you want?** It gives statistical instruction-level
  attribution and needs a post-3.8 build. For the alternatives that run on a
  released Triton, see
  [Choosing a Proton backend](../../../../docs/proton-backends.md).

- **`backend_mode`, not `instrumentation_mode`.** Both render Proton's single
  `--mode`, so the schema makes them mutually exclusive, and `backend_mode`
  requires an explicit (non-`auto`) backend. The backend's documented domain is
  `pcsampling` and `periodic_flushing`, and that is what aorta's schema
  accepts; what a given build *implements* is the separate question above.
  `roctracer` accepts only `periodic_flushing`.
- **PC sampling is statistical.** It reports where samples landed, not an exact
  per-instruction cost, so short runs are noisy. That is why the payload
  defaults to 50 launches rather than one. For deterministic intra-kernel
  attribution on a backend whose mode is available on a released Triton, use
  [`../amd-instrumentation`](../amd-instrumentation/README.md) instead — it
  counts cycles per source-level scope rather than sampling.
- **Why `mode: env`.** One reason, and it is not the one that forces the
  sibling example's hand. `--mode` reaches Proton through the CLI only on 3.8.0
  and newer — 3.7.1 and earlier parse `-m/--mode` and then call `start()`
  without it — so `mode: env` is what makes `backend_mode` version-independent.
  Unlike [`../amd-roctracer`](../amd-roctracer/README.md), this backend is
  *not* refused under `mode: cli`: `mode: cli` would be correct here too, and
  on ordering grounds it is the shape upstream prefers, since it loads
  `libproton` before the payload runs. Env mode is chosen for the `--mode`
  portability and is safe here only because of the next bullet.
- **The payload's import order is load-bearing.** `libproton.so` calls
  `rocprofiler_force_configure` from an `__attribute__((constructor))`, so
  importing Proton is what configures rocprofiler-sdk. Triton v3.8.0's
  `third_party/proton/csrc/lib/Profiler/RocprofSDK/RocprofSDKProfiler.cpp`
  warns, in the comment above that call, that "any code that fully initializes
  HSA beforehand (e.g. triton's HIP driver query at pytest collection time, or
  a torch import chain) causes rocprofiler-sdk 1.2.0 to silently skip
  kernel-dispatch buffer tracing installation on already-existing queues,
  producing an empty dispatch buffer and no per-kernel timing data." So
  [`gelu.py`](gelu.py) imports `triton.profiler` and `libproton` **before**
  `torch`, and says so at the import site. Note this particular ordering claim
  comes from upstream's source comment and was not measured here — nothing in
  reach can run this backend (see [Availability](#availability)) — whereas the
  `roctracer` claim next door is measured.

  **Every sibling payload now shares this import order**, for a second reason
  that was measured: on Triton 3.8.0, registering Proton after HSA is up makes
  the atexit `rocprofiler::registration::finalize()` re-enter its own
  non-recursive registration mutex through Proton's `protonToolFini` and
  deadlock, so the process hangs forever after writing a complete capture
  ([ROCm/aorta#434](https://github.com/ROCm/aorta/issues/434)).

  With that, the payloads no longer differ in ordering at all: each imports
  Proton first and calls `proton.start()` from `main()`, after torch. That is
  not a compromise between the two contracts — the contracts constrain
  *different events*. `rocprofiler` constrains when it is **configured**, and
  the `libproton.so` constructor does that at import; `roctracer` constrains
  when the **session starts**. One ordering satisfies both, which is why the
  "two backends, two import orders" framing this bullet used to carry was
  wrong rather than merely out of date. Do not let a linter sort these imports.
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
