# Examples

Runnable examples and copy-paste config templates for aorta. Everything here
is open source, cheap to run, and free of customer or host specifics — point
it at your own machine, confirm the machinery works, then swap in your own
workload.

Two kinds of thing live in this directory:

- **Runnable examples** — a payload plus the recipe that runs it, in a
  category subdirectory. Each one has its own README with requirements, a
  standalone command, and the aorta command.
- **Config templates** — single files you pass to a flag. Not runnable on
  their own.

## Index

| | What it is | Start at |
|---|---|---|
| [`profiling/`](profiling/README.md) | Four GPU payloads captured with the `rocprof` and `proton` collectors: a HIP SGEMM, a torch matmul, and two Triton kernels | [`profiling/README.md`](profiling/README.md) |
| [`mitigations-sidecar.json`](mitigations-sidecar.json) | Template for a sidecar file that adds named mitigations *and* environments without installing a plugin. Pass with `--mitigations-file` | [`src/aorta/registry/README.md`](../src/aorta/registry/README.md) |
| [`probe-flag-sidecar.json`](probe-flag-sidecar.json) | Ready-made sidecar of workload-internal flags (`FBGEMM_*`, `TORCHINDUCTOR_*`, `EVAL_DISABLE_PIPELINING`) to sweep as mitigations. Pairs with [`recipes/probe/probe-flag-sweep.yaml`](../recipes/probe/probe-flag-sweep.yaml) | [`src/aorta/registry/README.md`](../src/aorta/registry/README.md) |

## Quickstart

The fastest end-to-end check that aorta can run and measure a GPU payload on
your host. No Python dependencies beyond aorta itself — `hipcc` compiles one
file:

```bash
hipcc -O3 -o /tmp/hip_gemm examples/profiling/rocprof/hip-gemm/gemm.hip
aorta sweep run \
  --recipe examples/profiling/rocprof/hip-gemm/recipe.yaml \
  --collect rocprof \
  --output ./profiling_results \
  -- /tmp/hip_gemm 512 20
```

You should get a run directory with `matrix.md` / `perf.md`, and a `rocprof/`
directory of CSVs under the trial. See
[`profiling/rocprof/hip-gemm/README.md`](profiling/rocprof/hip-gemm/README.md)
for what the numbers mean.

A sidecar template needs no GPU at all, so it is a good smoke test of the
CLI on any host:

```bash
aorta sweep list-mitigations --mitigations-file examples/mitigations-sidecar.json
aorta sweep list-environments --mitigations-file examples/mitigations-sidecar.json
```

Sidecar entries list with `SOURCE = sidecar:<filename>` so it is obvious which
file shipped which entry.

## Prerequisites

| Example | Needs |
|---|---|
| `profiling/rocprof/hip-gemm` | `hipcc`, one AMD GPU, `rocprofv3` |
| `profiling/rocprof/torch-matmul` | PyTorch for ROCm, `rocprofv3` |
| `profiling/proton/*` | PyTorch **and** Triton for ROCm |
| the sidecar templates | nothing — no GPU, no container |

The PyTorch and Triton examples are meant to run **inside** a ROCm container
with aorta installed there, not from a host interpreter that shells out to
`docker run`. Each example's README shows the container invocation and
explains why: a profiler attached to the `docker` client sees no kernels.

## Conventions

Every runnable example in this tree follows the same five rules, and a new
one should too:

1. **Standalone-runnable.** Reproducible without aorta; the README shows the
   bare command first.
2. **Self-checking.** The payload verifies its own output and exits non-zero
   when wrong, so a bad result is a failed trial rather than a fast one.
3. **Cheap by default.** Defaults finish in seconds; size and iteration
   counts come from arguments, never a hard constant.
4. **Open source, no host specifics.** Original or permissively licensed,
   with the upstream and its license named in the README's Provenance
   section. No customer content, no internal repository content, no absolute
   host paths, no environment dumps.
5. **One README per example**, covering requirements, standalone run, aorta
   run, and the artifacts produced.

## Adding an example

Add it under an existing category directory, or create a new category
alongside them and give it a `README.md` that indexes its examples and a row
in the [Index](#index) table above. A category README owns its own
per-example table and conventions — see
[`profiling/README.md`](profiling/README.md) for the shape to copy.

Keep this file an index: one row per category or template, no duplicated
run instructions.

## See also

- [`docs/getting-started.md`](../docs/getting-started.md) — installing aorta
  and a first run.
- [`docs/profiling-collectors.md`](../docs/profiling-collectors.md) —
  `rocprof` / `proton` reference: options, artifacts, analysis,
  troubleshooting.
- [`recipes/README.md`](../recipes/README.md) — the recipe schema, including
  the `collect:` block.
- [`recipes/`](../recipes/) — ready-made recipes, including probe-mode
  smoke tests.
- [`src/aorta/registry/README.md`](../src/aorta/registry/README.md) — the
  sidecar file schema and how mitigations / environments resolve.
