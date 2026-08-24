# torch-matmul — rocprof trace of PyTorch library GEMM kernels

Where [`hip-gemm`](../hip-gemm/README.md) profiles a kernel you can read,
this one profiles the kernels a framework picks for you. The payload issues
repeated `a @ b` on the GPU and lets PyTorch route it to hipBLASLt /
rocBLAS, so the kernel trace contains real Tensile kernel names
(`Cijk_Alik_Bljk_...`) rather than a hand-written symbol.

Use it to check that the collector's summary parsing survives a framework
stack: many distinct kernels per step, autotuning on the first call, and
kernel names long enough to stress a CSV reader.

## Requirements

| | |
|---|---|
| Runtime | PyTorch built for ROCm, one AMD GPU |
| Profiler | `rocprofv3` on `PATH`, **inside the same interpreter's container** |
| Python deps | `torch` |

**This example does not run against a bare host interpreter unless you have
installed PyTorch-for-ROCm there.** The reference path is a ROCm PyTorch
container.

## Run it in a container

Install aorta *inside* the container and run the sweep there. Wrapping
`docker run` from the host would attach `rocprofv3` to the docker client
process, which dispatches no kernels and therefore produces no profile.

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
      --recipe examples/profiling/rocprof/torch-matmul/recipe.yaml \
      --output ./profiling_results \
      -- python examples/profiling/rocprof/torch-matmul/matmul.py \
           --size 2048 --iters 20 --dtype float16
  '
```

Pin the image by digest rather than `:latest` for anything you intend to
compare over time.

A ROCm PyTorch virtualenv on the host works identically — the container is
just the reproducible way to get one.

## Run standalone

Inside the same container or venv:

```bash
python examples/profiling/rocprof/torch-matmul/matmul.py \
  --size 2048 --iters 20 --dtype float16
```

Options: `--size`, `--iters`, `--warmup`, `--dtype`
(`float32` / `float16` / `bfloat16`), `--device`. Output:

```
matmul: device=... dtype=float16
matmul: size=2048 iters=20 warmup=3
matmul: mean_kernel_ms=...
matmul: gflops=...
matmul: PASS
```

The payload exits `2` if torch sees no GPU rather than falling back to CPU:
a silent CPU run would yield an empty profile and read as a collector bug.

## What you get

The same artifact set as `hip-gemm` — `<stem>_kernel_stats.csv`,
`<stem>_kernel_trace.csv`, `<stem>_domain_stats.csv`,
`<stem>_agent_info.csv` under the trial's `rocprof/` directory — plus HIP
API rows, because this example's options (documented in
[`recipe.yaml`](recipe.yaml)) set `trace: "kernel,hip"`.

`kernel_include_regex: "Cijk|gemm|matmul"` narrows the capture to GEMM
kernels so `rocprof_top_kernel_ms` reports the matmul rather than a
`fill`/`copy` kernel. It is a **collection** filter, not a post-filter: the
collector passes it to `rocprofv3` as `--kernel-include-regex`, so an excluded
kernel never reaches `_kernel_trace.csv` either. If a kernel you expected is
missing, widen or drop the regex and re-run — it is not waiting in the raw
trace.

## Notes

- The first matmul on a cold hipBLASLt cache can take seconds while the
  library picks a kernel. That autotuning dispatch lands in the trace; the
  `--warmup` iterations exist to keep it out of the timed loop, not out of
  the profile.
- `float16` is the default because it is what the Tensile fast paths are
  tuned for. Switch to `--dtype float32` to see a different kernel family
  selected for the same shapes.
