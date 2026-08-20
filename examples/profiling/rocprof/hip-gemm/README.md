# hip-gemm — rocprof kernel trace of a tiled HIP SGEMM

The quickstart profiling example. A single self-contained HIP source file
compiled by `hipcc`, launching one named kernel (`sgemm_tiled`) per
iteration, with a built-in correctness check. No Python, no PyTorch, no
Triton — if you have a ROCm toolchain you can run this.

Use it to answer "does the `rocprof` collector work on this machine at all?"
before pointing it at a real workload.

## Requirements

| | |
|---|---|
| Toolchain | `hipcc` (ships with ROCm) |
| Runtime | One AMD GPU, `/dev/kfd` + `/dev/dri` visible |
| Profiler | `rocprofv3` on `PATH` (ROCm 6.2+), or `$ROCPROF_BIN` |
| Python deps | none |

Verified against ROCm 7.0.2.2 / `rocprofv3` 1.0.1 on gfx950.

## Build

```bash
hipcc -O3 -o /tmp/hip_gemm examples/profiling/rocprof/hip-gemm/gemm.hip
```

Build artifacts are intentionally not checked in; `/tmp` keeps the repo
clean. Add `--offload-arch=<gfx>` if you are cross-building for a GPU that
is not in the build host.

## Run standalone

```bash
/tmp/hip_gemm 512 20
```

Arguments are `[matrix_size] [iterations]`, defaulting to `512 20`; the
`AORTA_GEMM_N` / `AORTA_GEMM_ITERS` environment variables are the fallback
when arguments are omitted. Expected output:

```
gemm: device=0 arch=gfx950:sramecc+:xnack-
gemm: n=512 iters=20 tile=16
gemm: mean_kernel_ms=0.0230
gemm: gflops=11680.22
gemm: max_rel_err=3.304e-06
gemm: PASS
```

The process exits non-zero if the sampled dot products disagree with the
kernel's output, so a wrong answer is a failed trial rather than a fast one.

## Run standalone under rocprofv3

Useful for comparing what aorta captured against a hand-rolled capture:

```bash
rocprofv3 --kernel-trace --stats --output-format csv \
  -d /tmp/rocprof_out -o gemm -- /tmp/hip_gemm 512 20
```

## Run under aorta

```bash
aorta sweep run \
  --recipe examples/profiling/rocprof/hip-gemm/recipe.yaml \
  --output ./profiling_results \
  -- /tmp/hip_gemm 512 20
```

Everything after `--` is the opaque command; [`recipe.yaml`](recipe.yaml)'s
`collect:` block is what turns on profiling and carries this example's
rocprof options. Swap the command for any other GPU binary and the same
recipe still applies — that is the point of the argv-wrapping design.

Passing `--collect <name>` on the command line replaces the recipe's
collector list, so use it to run this payload under a different collector.

## What you get

Per-trial rocprof artifacts land in the trial's collector directory under
`rocprof/`; the absolute path is reported as the `rocprof_artifact_dir`
metric so you never have to guess it.

`rocprofv3` names its CSVs by output stem: with `-o <stem>` they are flat
files named `<stem>_kernel_stats.csv`, `<stem>_kernel_trace.csv`,
`<stem>_domain_stats.csv`, and `<stem>_agent_info.csv`. With no `-o` it
nests them one level deeper as `<hostname>/<pid>_kernel_stats.csv` and so
on.

`<stem>_kernel_stats.csv` is the summary the collector parses:

```
"Name","Calls","TotalDurationNs","AverageNs","Percentage","MinNs","MaxNs","StdDev"
"sgemm_tiled(float const*, float const*, float*, int)",23,538601,23417.434783,100.00,22160,37280,3033.067227
```

23 calls, not 20 — the payload does three warmup launches before the timed
loop, and rocprof counts every dispatch.

The parsed numbers reach `perf.md` and `matrix.json` as
`rocprof_kernel_count`, `rocprof_gpu_time_ms`, and `rocprof_top_kernel_ms`.

## Notes

- `rocprofv3` writes progress and HSA-init lines to **stderr**, so a
  rocprof-collected trial's stderr is not byte-identical to an unprofiled
  one. Profiling is a measurement mode, not a transparent passthrough.
- With no GPU activity `rocprofv3` writes no files at all. A trial whose
  payload never dispatched a kernel yields no rocprof metrics rather than
  zeros.
- Keep `iterations` small. The example exists to produce a clean profile,
  not a benchmark number; a long run just makes a bigger trace.
