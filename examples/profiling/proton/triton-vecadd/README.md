# triton-vecadd — Proton capture of a Triton elementwise kernel

The smallest Proton example: one Triton kernel, launched in a short loop,
checked against `torch` elementwise add. Use it to confirm the `proton`
collector attaches and produces a hatchet tree before pointing it at a real
model.

## Requirements

| | |
|---|---|
| Runtime | Triton + PyTorch built for ROCm, one AMD GPU |
| Profiler | Proton, which ships inside Triton — no separate install |
| Python deps | `torch`, `triton` |

**Not runnable against a bare host interpreter.** Proton attaches as
`python -m triton.profiler.proton` using the *workload's own* interpreter,
so Triton must be importable there. A host that has the `proton` console
script on `PATH` but no `triton` in its Python will fail — which is exactly
why the collector uses `-m` rather than the console script.

## Run it in a container

Install aorta inside the container and run the sweep there:

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
      --recipe examples/profiling/proton/triton-vecadd/recipe.yaml \
      --output ./profiling_results \
      -- python examples/profiling/proton/triton-vecadd/vecadd.py \
           --size 1048576 --iters 20
  '
```

Pin the image by digest for anything you intend to compare over time. A
ROCm venv with `torch` + `triton` on the host works the same way.

## Run standalone

```bash
python examples/profiling/proton/triton-vecadd/vecadd.py --size 1048576 --iters 20
```

Options: `--size`, `--iters`, `--block-size` (must be a power of two, since
`tl.arange` requires one — a non-power-of-two is rejected by argparse rather
than failing later inside Triton's compiler). Output:

```
vecadd: device=...
vecadd: size=1048576 iters=20 block_size=1024
vecadd: max_abs_err=0.000e+00
vecadd: PASS
```

The check is exact equality against `x + y` — elementwise float32 addition
has no reassociation freedom, so anything non-zero means a real indexing or
masking bug.

## Run standalone under Proton

```bash
python -m triton.profiler.proton -n vecadd \
  examples/profiling/proton/triton-vecadd/vecadd.py --size 1048576 --iters 20
proton-viewer -m time/s vecadd.hatchet
```

## What you get

A `.hatchet` JSON tree in the trial's `proton/` directory; the absolute path
is reported as `proton_artifact_dir`. The collector walks the tree and emits
`proton_kernel_count`, `proton_gpu_time_ms` and `proton_top_kernel_ms`, which
reach `perf.md` and `matrix.json::cells[*].metrics_summary` because those
reports aggregate numeric scalars only. The non-numeric `proton_top_kernels`
list and `proton_artifact_dir` ride the same metrics channel but appear only
in the per-trial dispatcher JSON (`.result.metrics`) — look there, not in
`perf.md`, for the kernel names.

Read the raw tree yourself with `proton-viewer -m time/s <file>.hatchet`
(run it in the same environment as the capture).

## Notes

- **Device selection.** Proton on AMD does not honor
  `HIP_VISIBLE_DEVICES`; use `ROCR_VISIBLE_DEVICES` to pin a GPU or the
  device ids in the tree will not match the ones you expect.
- **Backend.** The recipe leaves `backend: "auto"`, which omits Proton's
  `-b` so Proton picks the backend matching the active runtime. On AMD that
  resolves to `rocprofiler` from Triton 3.8.0 onward and to `roctracer` on
  3.7.1 and earlier. Naming one is a version commitment: `rocprofiler` is
  the preferred AMD backend upstream and released as of 3.8.0, but 3.7.x and
  earlier accept only `cupti`/`roctracer`/`instrumentation` and exit with an
  argparse `invalid choice: 'rocprofiler'` before the payload runs. For
  `roctracer` it is an attach-mode commitment too: Proton's CLI front-end
  initialises the HIP runtime only on the path where `-b` is absent, and
  `roctracer` records nothing unless it starts after that, so on Triton 3.7.x
  and earlier pinning it under this recipe's `mode: "cli"` captures an empty
  `ROOT`-only tree and still exits 0. Measured on 3.8.0 the same pin captures
  normally, but the collector refuses that pairing on every version — 3.7.x is
  still shipped by images in use and the failure is silent — and names
  `mode: "env"`, where the payload drives Proton itself;
  [`../amd-roctracer/`](../amd-roctracer/README.md) is the worked case and
  [ROCm/aorta#439](https://github.com/ROCm/aorta/issues/439) tracks the
  refusal's removal.
  `rocprofiler` carries no such commitment — it is configured when
  `libproton.so` loads, so `mode: "cli"` suits it and is allowed.
- **What `auto` resolved to is not in the artifact.** The `.hatchet`
  metadata carries the device type and never the backend name, and the
  collector publishes no `proton_backend` metric. Two captures of this
  payload taken on a 3.7.1 image and a 3.8.0 image look alike and come from
  different backends, so read the Triton version out of the run's env
  snapshot before comparing their numbers.
- **Do not stack this with `rocprof`.** Proton's AMD backends intercept HSA
  queues, and so does `rocprofv3`; running both fights over the same
  interception point, and the pairing is rejected at recipe load. Only the
  `instrumentation` backend coexists with `rocprof`.
- **``Could not load `lib<something>.so` ``.** Some container images get
  ROCm from Python wheels, which ship only the versioned soname while Proton
  `dlopen`s the unversioned name. Add a directory of unversioned symlinks to
  `$LD_LIBRARY_PATH`, or use an image with a system ROCm install. The library
  named follows whichever backend ran: `libroctracer64.so` below Triton 3.8.0,
  and `librocprofiler-sdk.so` from 3.8.0 on, since that is where `auto` starts
  resolving to `rocprofiler` — measured on a ROCm 10 image. Same remedy
  either way.
- Triton compiles on first launch, so a cold kernel cache dominates wall
  time. That compile does not dispatch a kernel and so does not appear in
  the tree.

## Provenance

The kernel is adapted from the Triton tutorial `01-vector-add.py` in
[triton-lang/triton](https://github.com/triton-lang/triton), MIT License.
