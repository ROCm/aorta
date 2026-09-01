# triton-softmax — Proton capture attributed to Python frames

A fused row-softmax: a reduction kernel with a real numerical-stability
concern, checked against `torch.softmax`. Same attach mechanism as
[`triton-vecadd`](../triton-vecadd/README.md), but this example's options
(documented in [`recipe.yaml`](recipe.yaml)) set `context: "python"` so the
hatchet tree is keyed by Python call stack instead of launch site — the
configuration you want once a script launches more than one kernel.

## Requirements

| | |
|---|---|
| Runtime | Triton + PyTorch built for ROCm, one AMD GPU |
| Profiler | Proton, which ships inside Triton — no separate install |
| Python deps | `torch`, `triton` |

**Not runnable against a bare host interpreter** — see the vecadd README for
why Proton must attach via the workload's own interpreter.

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
      --recipe examples/profiling/proton/triton-softmax/recipe.yaml \
      --output ./profiling_results \
      -- python examples/profiling/proton/triton-softmax/softmax.py \
           --rows 4096 --cols 1024 --iters 20
  '
```

## Run standalone

```bash
python examples/profiling/proton/triton-softmax/softmax.py --rows 4096 --cols 1024 --iters 20
```

Options: `--rows`, `--cols`, `--iters`. Output:

```
softmax: device=...
softmax: rows=4096 cols=1024 iters=20
softmax: max_abs_err=...
softmax: PASS
```

Tolerance is `1e-6` against `torch.softmax`, not exact equality: the fused
kernel reduces in a different order than PyTorch, so the last bits differ
legitimately.

The kernel uses one program per row, with `block_size` rounded up to the
next power of two so a whole row fits in one program. That is what makes the
softmax "fused", and it is also the limit: a very large `--cols` asks Triton
for a block it may not be able to allocate registers for. Keep `--cols` at
or below a few thousand.

## Run standalone under Proton

```bash
python -m triton.profiler.proton -n softmax -c python \
  examples/profiling/proton/triton-softmax/softmax.py --rows 4096 --cols 1024
proton-viewer -m time/s softmax.hatchet
```

## What you get

The same artifact and metric set as `triton-vecadd`
(`proton_kernel_count`, `proton_gpu_time_ms`, `proton_top_kernel_ms`,
`proton_top_kernels`, `proton_artifact_dir`), but with the tree keyed by
Python frame, so `softmax()` and the correctness-check `torch.softmax` call
appear as separate nodes.

### Scoped measurement instead

`context: "python"` still measures the whole process. When you want time for
one region only — a single training step, one layer — switch the recipe to
`mode: "env"`. Proton then leaves argv untouched, aorta exports the
`AORTA_PROTON_*` variables, and the payload is expected to call
`proton.start()` / `proton.finalize()` itself around the region of interest.
The payload in this directory deliberately does *not* do that: it stays a
plain Triton script so it can be run and understood without Proton
installed.

## Notes

- **Device selection.** Use `ROCR_VISIBLE_DEVICES`, not
  `HIP_VISIBLE_DEVICES` — Proton on AMD does not honor the latter.
- **Backend.** Left on `backend: "auto"`; see
  [`../triton-vecadd/README.md`](../triton-vecadd/README.md) for why, for
  why pinning a queue-intercepting AMD backend means `mode: "env"`, and for the
  backend-library (`Could not load lib...`) note — which names
  `libroctracer64.so` or `librocprofiler-sdk.so` depending on which backend
  `auto` selected.
- **Do not stack this with `rocprof`**; both intercept HSA queues and the
  pairing is rejected at recipe load. Only the `instrumentation` backend
  coexists.

## Provenance

The kernel is adapted from the Triton tutorial `02-fused-softmax.py` in
[triton-lang/triton](https://github.com/triton-lang/triton), MIT License.
