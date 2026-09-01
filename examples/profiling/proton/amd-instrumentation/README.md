# amd-instrumentation — attribution *inside* a single Triton kernel

One deliberately unbalanced Triton kernel with two named intra-kernel scopes —
`cheap` (a single multiply) and `expensive` (a short `erf` loop) — captured by
Proton's `instrumentation` backend. Every other Proton backend tells you how
long a kernel took; this one tells you *where inside the kernel* the cycles
went.

Two things make it worth a separate example:

- **Intra-kernel attribution**, which no queue-tracing backend can give. A
  `roctracer` capture of this payload reports a single number for
  `unbalanced_kernel` (verified); this one reports a number per scope.
- **It installs no HSA queue interceptor**, so it is the one Proton backend
  that can share a process with `rocprofv3`. That is exactly why aorta's
  `rocprof` + `proton` conflict guard accepts `backend: instrumentation` and
  rejects `auto` / `roctracer` / `rocprofiler`.

Read [What you get](#what-you-get) before you point this at anything: the
capture is real and readable, but it publishes **no numeric aorta metrics**
today.

## Requirements

| | |
|---|---|
| Runtime | Triton + PyTorch built for ROCm, one AMD GPU |
| Profiler | Proton, which ships inside Triton — no separate install |
| Triton version | Any released Triton has the backend. Verified on 3.7.1 |
| Python deps | `torch`, `triton` |
| Opt-in | `pl.enable_semantic("triton")` in the payload — Triton-DSL instrumentation is off by default |

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
      --recipe examples/profiling/proton/amd-instrumentation/recipe.yaml \
      --output ./profiling_results \
      -- python examples/profiling/proton/amd-instrumentation/hotspot.py \
           --size 65536 --steps 8
  '
```

Pin the image by digest for anything you intend to compare over time. A ROCm
venv with `torch` + `triton` on the host works the same way.

## Run standalone

```bash
python examples/profiling/proton/amd-instrumentation/hotspot.py --size 65536 --steps 8
```

Options: `--size`, `--steps`, `--iters`, `--backend`. Output:

```
hotspot: device=...
hotspot: size=65536 steps=8 iters=5
hotspot: max_abs_err=...
hotspot: PASS
```

With no `--backend` and no `AORTA_PROTON_*` in the environment the payload
takes no capture at all — the `pl.enter_scope` / `pl.exit_scope` calls compile
away to nothing when instrumentation is off, so this is also the way to check
that the kernel itself is sound.

Tolerance is `1e-6` absolute against `x * 2` followed by `--steps` iterations
of `torch.erf`, not exact equality: the device `erf` and torch's differ in the
last bits. `erf` is contractive toward zero, so the discrepancy shrinks rather
than compounds across iterations. `--steps` is capped at 64 because
`loop_steps` is a `tl.constexpr` and the loop is unrolled at compile time — a
large value buys a long compile and a big kernel, not a slow one.

## Run standalone under Proton

This payload drives Proton itself, so there is no `python -m
triton.profiler.proton` wrapper to add:

```bash
python examples/profiling/proton/amd-instrumentation/hotspot.py --backend instrumentation
proton-viewer -m cycles hotspot.hatchet
```

To rehearse exactly what aorta does, export the bundle by hand:

```bash
mkdir -p ./proton_out
env AORTA_PROTON_DIR=./proton_out \
    AORTA_PROTON_NAME=./proton_out/proton \
    AORTA_PROTON_CONTEXT=shadow \
    AORTA_PROTON_DATA=tree \
    AORTA_PROTON_BACKEND=instrumentation \
    AORTA_PROTON_MODE=default \
  python examples/profiling/proton/amd-instrumentation/hotspot.py
```

## What you get

A `.hatchet` JSON tree in the trial's `proton/` directory, reported as
`proton_artifact_dir` — **and nothing else**. No `proton_kernel_count`, no
`proton_gpu_time_ms`, no `proton_top_kernel_ms`, no `proton_top_kernels`.

That is a real gap and not a misconfiguration. An instrumentation capture's
leaves carry `cycles` and `normalized_cycles`; they carry no `time (<unit>)`
metric and no `count`. aorta's parser (`src/aorta/instrumentation/proton/_parse.py`)
keys exclusively on `time (<unit>)`, finds nothing to aggregate, and degrades
to the artifact directory. So an instrumentation trial contributes nothing to
`perf.md` or `matrix.json::cells[*].metrics_summary`; the measurement lives
entirely in the artifact, and you read it yourself.

What the artifact holds is the thing worth having — a tree with the scopes
nested under the kernel:

```json
[{"children": [{"children": [{"children": [], "frame": {"name": "cheap", "type": "function"},
  "metrics": {"cycles": 152040, "normalized_cycles": 593.9, "device_id": "0", "device_type": "0"}},
  {"children": [], "frame": {"name": "expensive", "type": "function"},
  "metrics": {"cycles": 8463028, "normalized_cycles": 33058.7, "device_id": "0", "device_type": "0"}}],
  "frame": {"name": "unbalanced_kernel", "type": "function"}, "metrics": {}}],
  "frame": {"name": "ROOT", "type": "function"}, "metrics": {"cycles": 0, "normalized_cycles": 0}},
  {"HIP": {"0": {"arch": "gfx950", "num_sms": 256}}}]
```

Note the shape: `unbalanced_kernel` carries empty `metrics` and the numbers sit
on its scope children. That capture is the payload's defaults on one MI355X
(gfx950) — `expensive` at roughly 55× the cycles of `cheap`, which is the whole
point of the imbalance. The same payload under `--backend roctracer` reports
one leaf, `unbalanced_kernel`, and one number; the scopes are invisible to it.
Read the instrumentation tree with:

```bash
proton-viewer -m cycles <file>.hatchet
proton-viewer -m normalized_cycles <file>.hatchet
```

in the same environment as the capture. `normalized_cycles` divides by the
number of warps that recorded the scope, so it is the figure to compare across
runs with different grids.

## Why `mode: env`

**Whether the CLI forwards `--mode` depends on the Triton version, and
`mode: env` is the spelling that does not.** Triton 3.7.1's
`triton/profiler/proton.py` parses `-m/--mode` and then never forwards it:

```python
backend = args.backend if args.backend else _select_backend()
start(args.name, context=args.context, data=args.data, backend=backend, hook=args.hook)
```

There is no `mode=` in that call, so on 3.7.1 and earlier
`instrumentation_mode` (and `granularity`, and `backend_mode`) is rendered into
the wrap and dropped on the floor. Triton 3.8.0 fixed it — line 75 of
`third_party/proton/proton/proton.py` at the `v3.8.0` tag is
`start(args.name, context=args.context, data=args.data, backend=backend,
mode=args.mode, hook=args.hook)` — so a `mode: cli` recipe does carry the knob
there. The collector renders `--mode` on every version rather than refusing the
combination, because it validates in aorta's own interpreter and cannot know
which Triton will run the wrap.

This example keeps `mode: env` because that route is version-independent: it
exports the value as `AORTA_PROTON_MODE` and the payload hands it to
`proton.start()` itself, so `instrumentation_mode: default` takes effect on
3.7.1 and 3.8.0 alike. An example that only worked on the newest release would
be a poor thing to point someone at an arbitrary container image with.

That is the *only* reason this example uses `mode: env`, and it is worth being
precise about, because the sibling [`../amd-roctracer`](../amd-roctracer/README.md)
needs it for a different and much harder one. There, the same snippet's
`_select_backend()` — called only when `-b` is absent — is what initialises the
Triton HIP driver, and a queue-intercepting backend pinned ahead of the runtime
records nothing; that line is unchanged in 3.8.0, so no version fixes it. This
backend installs no queue interceptor: a `-b instrumentation` CLI wrap of this
payload captures both scopes correctly (verified on 3.7.1: 1738 bytes, cycle
counts intact). So the collector's refusal to pin a backend under `mode: cli`
covers `roctracer` and `rocprofiler` and not this one — what `mode: cli` costs
here is a version guarantee on the mode knob, not the capture.

## Notes

- **`granularity` is deliberately not set in the recipe.** The option pair
  `instrumentation_mode: default` + `granularity: warp` renders Proton's single
  `--mode` as the string `default:granularity=warp`, and Triton 3.7.1 rejects
  that at kernel exit:

  ```
  RuntimeError: Only warp granularity is supported for now
  ```

  raised from `libproton.exit_instrumented_op`. Only the *string* spelling of
  the `granularity=` key is broken — the bare `mode="default"` works, and so
  does the equivalent typed object
  `triton.profiler.mode.Default(granularity="warp")`. Other mode knobs pass
  through as strings fine (`mode="default:buffer_size=4096"` works). Warp
  granularity is this backend's own default on this version anyway — the
  working captures are per-warp — so omitting the key costs nothing. Whether a
  later release accepts the string spelling was not checked on 3.8.0; spell the
  key explicitly only on a build where you have confirmed it.
- **Instrumenting Triton-DSL kernels is opt-in, and the opt-in is a
  trade-off.** `triton/profiler/language.py` enables only the Gluon semantic by
  default, because Triton's higher-level IR undergoes aggressive rewrites —
  loop pipelining, instruction re-ordering, IR duplication — that can
  invalidate naive instrumentation and produce misleading results. The payload
  calls `pl.enable_semantic("triton")` to accept that risk. Treat an
  instrumented Triton-DSL kernel's scope boundaries as *subject to those
  rewrites*: a scope that the compiler hoisted out of, split, or duplicated
  will report cycles that do not correspond to the source region you drew. The
  wide, one-sided imbalance in this payload is deliberate for that reason — a
  result that survives the rewrites is one you can still reason about.
- **Every launch happens inside the session.** The backend rewrites the
  kernel's IR to insert the scope records, so a warm-up launch taken before
  `proton.start()` would put an *uninstrumented* binary in Triton's cache and
  the profiled launches would reuse it. This payload therefore keeps its first
  launch inside the capture, unlike
  [`../amd-roctracer`](../amd-roctracer/README.md).
- **This is the backend to combine with `rocprof`.** No queue interception
  means no fight over the interception point. `rocprof` + `proton` on
  `backend: auto` / `roctracer` / `rocprofiler` is rejected at recipe load;
  `backend: instrumentation` is accepted.
- **Device selection.** The collector performs no `HIP_VISIBLE_DEVICES`
  translation for this backend, because Proton does not reject the variable
  here — there is no queue interceptor to confuse. Pin devices the way you
  normally would.

## Provenance

The kernel is original to this repository — not adapted from a Triton tutorial.
The two scopes exist only to be lopsided, so the capture demonstrates
intra-kernel attribution on a result you can predict by reading the source.
