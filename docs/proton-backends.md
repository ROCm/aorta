# Choosing a Proton backend

Proton is one profiler with five backends, and they do not measure the same
thing. Picking the wrong one gives you a healthy-looking capture that cannot
answer your question — or, on some version combinations, a capture that is
silently empty.

This is the decision guide. For the option schema, artifact layout and
troubleshooting, see [Profiling Collectors](profiling-collectors.md); this page
does not restate them.

## Start from the question, not the backend

| Your question | Backend | Why |
| --- | --- | --- |
| Which kernel owns the GPU time? | `auto` | Whole-kernel spans, attributed to the launching frame. The default, and the only spelling correct on every Triton. |
| Same, but the run record must name the backend | `roctracer` | Pins the whole-kernel AMD path explicitly. Costs you `mode: env` — see [Attach mode](#attach-mode-is-part-of-the-choice). |
| *Where inside* one kernel do the cycles go? | `instrumentation` | Deterministic cycle counts against scopes you name in the kernel. |
| Where do samples land across a kernel, without naming scopes? | `rocprofiler` + `backend_mode: pcsampling` | Statistical instruction-level attribution. Needs a post-3.8 build — see [What is available where](#what-is-available-where). |
| I need Proton *and* `rocprofv3` in one process | `instrumentation` | The only backend that installs no HSA queue interceptor, so it is the only one the conflict guard accepts alongside `rocprof`. |
| I am on NVIDIA | `cupti` | Accepted so a recipe stays portable. Untested here; aorta's examples are AMD. |

If you are not sure, use `auto`. It is the default for a reason: it omits
Proton's `-b` and lets Proton pick the backend matching the runtime, which is
the only choice that is correct across versions.

## What each backend actually gives you

Numbers below are from the runnable examples in
[`examples/profiling/proton/`](../examples/profiling/proton/), measured on
gfx950 / ROCm 7.0.2 / Triton 3.7.1.

**Whole-kernel timing** (`auto`, `roctracer`, `rocprofiler`). One duration per
kernel, summed across launches.
[`amd-roctracer`](../examples/profiling/proton/amd-roctracer/) captures three
Triton kernels over 20 iterations and reports `proton_kernel_count` 60,
`proton_gpu_time_ms` 0.398, and `bias_gelu_kernel` as the top kernel. This is
the shape that populates `perf.md` and `matrix.json`, so it is the one to reach
for when you want numbers you can compare across cells of a sweep.

**Intra-kernel cycles** (`instrumentation`). One cycle count per scope you
declared inside the kernel.
[`amd-instrumentation`](../examples/profiling/proton/amd-instrumentation/) puts
two scopes in one deliberately unbalanced kernel and reports `expensive` at
8,608,876 cycles against `cheap` at 153,368 — a 56x split that whole-kernel
timing cannot see at all, because both live in the same kernel.

**Statistical instruction attribution** (`rocprofiler` + `pcsampling`). Samples
the program counter, so you get a distribution over instructions rather than
scopes you chose in advance.
[`amd-rocprofiler`](../examples/profiling/proton/amd-rocprofiler/) is that
example; its capture is not verified, because no obtainable build runs it yet.

## The cost nobody expects: `instrumentation` publishes no numeric metrics

A healthy `instrumentation` capture yields **only** `proton_artifact_dir` in the
trial metrics. Its leaves carry `cycles` and `normalized_cycles`, never
`time (<unit>)` or `count`, and the collector's summary keys exclusively on
wall-clock time. So the capture is real and useful, and `perf.md` will show
nothing from it.

Read it directly instead:

```bash
proton-viewer -m normalized_cycles <trial>/proton/proton.hatchet
```

If you need numbers that land in the sweep reports, you need a whole-kernel
backend. If you need to know which region of a kernel is slow, you need this
one and you read the artifact yourself. That trade is the reason both examples
exist.

## What is available where

`rocprofiler` is not universally available, and `auto` does not mean the same
thing across versions. Verified against real installs:

| | Triton 3.7.1 and earlier | Triton 3.8.0 |
| --- | --- | --- |
| `-b` choices | `cupti`, `roctracer`, `instrumentation` | adds `rocprofiler` |
| `auto` on AMD resolves to | `roctracer` | **`rocprofiler`** |
| CLI forwards `--mode` | no — parsed, then dropped | yes |
| `rocprofiler` + `pcsampling` | backend absent | `ValueError: unsupported mode` |

Two consequences worth internalising:

**`auto` silently changes which backend measured when Triton crosses 3.8.0.**
Nothing in the `.hatchet` records the resolved backend — it carries the device
type (`{"HIP": {...}}`) and no backend name. What disambiguates a run is the
Triton version in its environment snapshot. If you are comparing
`proton_gpu_time_ms` across runs from different images, check that first.

**`pcsampling` still needs a post-3.8 `main` build.** AMD PC sampling landed
upstream after the 3.8.0 tag, so 3.8.0 registers the backend and then rejects
the mode.

## Attach mode is part of the choice

Pinning a backend can commit you to an attach mode, because the two AMD tracing
backends are configured at **different moments**, and each needs something
different to be true at its own moment. This is the part that bites.

| Backend | Configured when | Needs at that moment | Attach mode |
| --- | --- | --- | --- |
| `roctracer` | session start, at `proton.start()` | HIP runtime **already up** | `mode: env`, payload calls `proton.start()` *after* importing torch |
| `rocprofiler` | `libproton.so` load, at *import* | HSA **not yet up** | either; `mode: cli`, or `mode: env` with Proton imported *before* torch |
| `instrumentation` | session start | no requirement | either |

**Those two requirements look opposed and are not.** They constrain different
events: `roctracer` constrains when the *session starts*, `rocprofiler`
constrains when Proton is *imported*. One payload shape satisfies both — import
`triton.profiler` first, call `proton.start()` after torch — and all three
`amd-*` examples use it. This page previously said the two wanted opposite
import orders, which was wrong when written rather than merely stale; see
[ROCm/aorta#434](https://github.com/ROCm/aorta/issues/434).

So `mode: env` with an explicit `roctracer` is required — the collector refuses
a `mode: cli` pin of it, because on Triton 3.7.1 that captures a 160-byte tree
holding a bare `ROOT` from a run that exits 0. That refusal is a 3.7.x-and-
earlier workaround: the same pin captured 17 kernels on 3.8.0. It is kept on
every version anyway, because 3.7.x is what several images in use ship and the
failure it prevents is silent; removing it once the floor reaches 3.8 is tracked
in [#439](https://github.com/ROCm/aorta/issues/439).

And `rocprofiler`'s requirement lands on the import instead. Triton 3.8.0 calls
`rocprofiler_force_configure` from a constructor in `libproton.so`, and its own
source warns that a torch import chain first makes the SDK skip dispatch-buffer
tracing on existing queues. So a payload driving it must import Proton before
torch, as
[`amd-rocprofiler/gelu.py`](../examples/profiling/proton/amd-rocprofiler/gelu.py)
does and says at the import site.

That import order is now the rule for **every** Proton payload, whatever
backend it targets, for a second and independent reason: on Triton 3.8.0 the
reverse order deadlocks the process at exit, after a complete capture has
already been written, so it reads as a hang with no error
([ROCm/aorta#434](https://github.com/ROCm/aorta/issues/434)). All three `amd-*`
payloads mark the import `# isort: skip  # noqa: I001` so the linter cannot
reorder it, and `test_proton_payloads_import_proton_before_torch` fails the CPU
suite if one is reversed.

The practical rule: **let the example you copied choose the attach mode, and
import Proton before torch whatever you copied.** The attach mode still varies
by backend; the import order no longer does.

## Combining with `rocprof`

`rocprof` and `proton` both install HSA queue interceptors, so the pairing is
rejected at recipe load for `auto`, `roctracer` and `rocprofiler`. Only
`instrumentation` coexists, because it interposes on nothing.

If you need whole-kernel data from `rocprofv3` *and* intra-kernel data from
Proton, either use `backend: instrumentation`, or run them as two cells and
compare.

## Known sharp edges

Each of these is a real, measured defect rather than a caveat, and each has an
actionable error or a workaround.

- **`backend_mode: periodic_flushing` on `roctracer` segfaults** on Triton
  3.7.1 / ROCm 7.0.2 — exit 139, no traceback — once kernels dispatch.
  Reproduced on two payloads. Leave it unset there.
- **`granularity` is unusable on 3.7.1** in either attach mode: the CLI drops
  `--mode`, and under `mode: env` the rendered `default:granularity=warp` is
  rejected at kernel exit with `RuntimeError: Only warp granularity is
  supported for now` — for warp, which is that backend's own default. Set
  `instrumentation_mode` alone.
- **`Could not load` lib...``** means a packaging problem, not an unsupported
  backend or mode. A wheel-provided ROCm ships only versioned sonames while
  Proton opens the unversioned name. Which library it names follows the
  backend: `libroctracer64.so` below 3.8.0, `librocprofiler-sdk.so` from 3.8.0
  on, since that is where `auto` starts resolving to `rocprofiler`.
- **`mode: env` captures wedged at exit on Triton 3.8.0** — diagnosed and fixed
  by import order, so this one is a defect you can now avoid rather than work
  around. Proton and rocprofiler-sdk self-deadlock in `protonToolFini` when
  Proton is imported *after* torch: the capture completes, the `.hatchet` is
  written, and the process then sits in `futex_wait` forever. Import
  `triton.profiler` before torch and it does not happen; the GPU suite runs
  those captures rather than skipping them. See
  [#434](https://github.com/ROCm/aorta/issues/434) and
  [Attach mode](#attach-mode-is-part-of-the-choice).

## Worked decisions

**"My sweep shows one cell 30% slower and I want to know which kernel."**
Leave `backend: auto`. Compare `proton_top_kernels` and `proton_top_kernel_ms`
across cells in `matrix.json`. No backend choice needed.

**"I know it is `fused_attention` and I want to know which part of it."**
`backend: instrumentation` with scopes around the regions you suspect, copying
[`amd-instrumentation`](../examples/profiling/proton/amd-instrumentation/).
Accept that you will read the `.hatchet` yourself.

**"I need to prove which backend produced last week's numbers."**
Pin it explicitly rather than relying on `auto`, and take the attach mode the
table above requires. Record the Triton version — the artifact does not.

**"Both a rocprofv3 trace and Proton attribution, one run."**
`backend: instrumentation`. Anything else is refused at recipe load.

## See also

- [Profiling Collectors](profiling-collectors.md) — option schema, attach-mode
  mechanics, artifact layout, analysis recipes, troubleshooting
- [`examples/profiling/proton/`](../examples/profiling/proton/) — one runnable
  example per backend, each with the attach mode its backend needs, and all of
  them importing Proton before torch
- [Profiling Guide](profiling.md) — capturing and interpreting profiling data
  more generally
