# Profiling examples

Runnable, open-source payloads for aorta's profiling collectors. Each one is
a small GPU workload plus the recipe that captures it, so you can verify a
collector works on your machine before pointing it at something expensive.

Collectors attach by wrapping the command's argv, so none of these payloads
knows it is being profiled. Any command you can run, you can profile —
these examples are just cheap, license-clean things to point the collectors
at.

## The examples

| Example | What it profiles | Requires | Run it |
|---|---|---|---|
| [`rocprof/hip-gemm`](rocprof/hip-gemm/) | One hand-written tiled SGEMM kernel (`sgemm_tiled`) | `hipcc`, a GPU. **No Python deps.** | `aorta sweep run --recipe examples/profiling/rocprof/hip-gemm/recipe.yaml --output ./profiling_results -- /tmp/hip_gemm 512 20` |
| [`rocprof/torch-matmul`](rocprof/torch-matmul/) | hipBLASLt / rocBLAS GEMM kernels PyTorch picks for `a @ b` | `torch` for ROCm. Container. | `aorta sweep run --recipe examples/profiling/rocprof/torch-matmul/recipe.yaml --output ./profiling_results -- python examples/profiling/rocprof/torch-matmul/matmul.py` |
| [`proton/triton-vecadd`](proton/triton-vecadd/) | One Triton elementwise kernel, launch-site attribution | `torch` + `triton` for ROCm. Container. | `aorta sweep run --recipe examples/profiling/proton/triton-vecadd/recipe.yaml --output ./profiling_results -- python examples/profiling/proton/triton-vecadd/vecadd.py` |
| [`proton/triton-softmax`](proton/triton-softmax/) | A Triton fused reduction, Python-frame attribution | `torch` + `triton` for ROCm. Container. | `aorta sweep run --recipe examples/profiling/proton/triton-softmax/recipe.yaml --output ./profiling_results -- python examples/profiling/proton/triton-softmax/softmax.py` |
| [`proton/amd-roctracer`](proton/amd-roctracer/) | Three Triton kernels in sequence, whole-kernel attribution on a pinned `roctracer` backend | `torch` + `triton` for ROCm. Container. `roctracer` is the *whole-kernel* AMD backend present in every released Triton, including the ones predating `rocprofiler` (added in 3.8.0). `instrumentation` is in every release too, but measures inside a kernel. | `aorta sweep run --recipe examples/profiling/proton/amd-roctracer/recipe.yaml --output ./profiling_results -- python examples/profiling/proton/amd-roctracer/pipeline.py` |
| [`proton/amd-instrumentation`](proton/amd-instrumentation/) | Two named scopes *inside* one unbalanced Triton kernel — intra-kernel attribution | `torch` + `triton` for ROCm. Container. The only Proton backend that coexists with `rocprofv3`; publishes no numeric metrics, only `proton_artifact_dir`. | `aorta sweep run --recipe examples/profiling/proton/amd-instrumentation/recipe.yaml --output ./profiling_results -- python examples/profiling/proton/amd-instrumentation/hotspot.py` |
| [`proton/amd-rocprofiler`](proton/amd-rocprofiler/) | Statistical instruction-level attribution (`backend_mode: pcsampling`) instead of whole-kernel spans | `torch` + `triton` for ROCm. Container. **A post-3.8 Triton `main` build**: the `rocprofiler` backend is released as of 3.8.0, but AMD `pcsampling` landed after that tag, so this example's capture ships documented but unverified. | `aorta sweep run --recipe examples/profiling/proton/amd-rocprofiler/recipe.yaml --output ./profiling_results -- python examples/profiling/proton/amd-rocprofiler/gelu.py` |

## Start here

`rocprof/hip-gemm` is the quickstart. It is the only example with zero
Python dependencies — `hipcc` compiles one file and you have a GPU payload —
so it is the fastest way to find out whether profiling works at all on a
given host:

```bash
hipcc -O3 -o /tmp/hip_gemm examples/profiling/rocprof/hip-gemm/gemm.hip
aorta sweep run \
  --recipe examples/profiling/rocprof/hip-gemm/recipe.yaml \
  --output ./profiling_results \
  -- /tmp/hip_gemm 512 20
```

The other six need a PyTorch-for-ROCm interpreter (and Triton, for the five
Proton ones). The reference way to get one is a ROCm PyTorch container with
aorta installed **inside** it — see each example's README. Running
`aorta sweep run -- docker run ...` from the host instead would attach the
profiler to the docker client, which dispatches no kernels and produces an
empty capture.

## Categories

| Directory | Collector | Notes |
|---|---|---|
| `rocprof/` | `rocprof` (`rocprofv3`) | Whole-process kernel and API tracing. Writes CSV / JSON / pftrace. |
| `proton/` | `proton` (Triton's profiler) | Runs inside the workload's interpreter; writes a hatchet tree. |

`rocprof` and `proton` both intercept HSA queues and will fight if enabled
together, so the pairing is rejected at recipe load. Only Proton's
`instrumentation` backend coexists with `rocprof`.

The two Triton examples leave `backend: "auto"`, which omits Proton's `-b`
and lets Proton pick the backend matching the active runtime. That is what
makes them run out of the box across Triton versions: `rocprofiler` is the
preferred AMD backend upstream and released as of Triton 3.8.0, but 3.7.x and
earlier accept only `cupti`/`roctracer`/`instrumentation` and exit with an
argparse `invalid choice: 'rocprofiler'` before the payload runs. The
convenience has a cost worth knowing: what `auto` resolves to on AMD is
`rocprofiler` from 3.8.0 onward and `roctracer` below it, and nothing in the
`.hatchet` records which one ran — the run's env snapshot is where you read the
Triton version that settles it.

The three `amd-*` examples pin a backend instead, and all use `mode: "env"`
with a payload that calls `proton.start()` itself — but for three different
reasons, which are worth keeping apart. Only one of the three is forced.

For `amd-roctracer` it is forced. Proton's CLI front-end initialises the HIP
runtime only on the path where `-b` is absent, and `roctracer` records nothing
unless it starts after that runtime is up, so a `roctracer` pin under the
default `mode: "cli"` captures an empty tree; the collector refuses that one
combination rather than letting a trial pass with nothing in it.

For `amd-rocprofiler` it is a choice, and the backend's contract is the
*opposite* of its sibling's rather than the same. `libproton.so` configures
rocprofiler-sdk from an `__attribute__((constructor))` at load time, so it
needs to be set up *before* HSA comes up — which makes `mode: "cli"` legal for
it, and on ordering grounds the shape upstream prefers. That example uses
`mode: "env"` so `backend_mode` reaches Proton on every Triton version, and it
is safe there only because the payload imports Proton before `torch`. This
ordering is taken from upstream's source comments, not measured here, unlike
the `roctracer` behaviour above.

For `amd-instrumentation` it is a choice too. That backend installs no queue
interceptor, so a CLI pin of it captures correctly and the collector allows it.
It uses `mode: "env"` only so that `instrumentation_mode` reaches Proton on
Triton 3.7.1 and earlier, whose CLI parses `--mode` and then drops it.

## Where the collector options live

Each example's `recipe.yaml` carries its collector configuration as a
`collect:` block, so the commands above need no `--collect` flag — the
recipe is the whole story, and the option values are reviewable,
copy-pasteable text sitting next to the payload they profile.

`--collect <name>` still works and takes precedence: passing it REPLACES the
recipe's collector list, keeping only the option blocks of the collectors
that survive. That is the way to run one example under a different
collector, or to compare a profiled and an unprofiled run of the same
payload:

```bash
# recipe's own collector, with its options
aorta sweep run --recipe .../torch-matmul/recipe.yaml \
  -- python .../torch-matmul/matmul.py

# override to a different collector (drops rocprof's option block)
aorta sweep run --recipe .../torch-matmul/recipe.yaml --collect proton \
  -- python .../torch-matmul/matmul.py
```

The override is `torch-matmul` rather than `hip-gemm` because the two
collectors do not accept the same commands. `rocprof` wraps any command, but
Proton's `mode: cli` takes over a *script*, so it attaches only to `python
... <script>.py`, a bare `pytest`, or `python -m pytest`. Pointing
`--collect proton` at a native binary such as `/tmp/hip_gemm` fails at setup
with a `ProtonWrapError` naming `mode: env` as the escape hatch — deliberately,
rather than running the payload unprofiled.

To run a payload with no capture at all, comment out the recipe's `collect:`
block — a useful way to separate "my payload is broken" from "my profiler is
broken".

## Adding an example

One directory per example, nested under its collector's category. Three
files, no more:

```
examples/profiling/<collector>/<example-name>/
  <payload>.{hip,py,sh}   the workload: standalone-runnable, self-checking,
                          open-source, cheap by default
  recipe.yaml             mode: probe, plus the collector's option block
  README.md               requirements, standalone run, aorta run, artifacts
```

The conventions the existing seven follow, and that a new one should too:

1. **Standalone-runnable.** A user must be able to reproduce the payload
   without aorta. Every README shows the bare command first.
2. **Self-checking.** The payload verifies its own output and exits
   non-zero when wrong, so a bad result is a failed trial rather than a
   suspiciously fast one.
3. **Cheap by default.** Defaults finish in seconds. Size and iteration
   count come from CLI arguments (or environment variables), never a hard
   constant.
4. **Open source, no host specifics.** Payloads are original or adapted
   from permissively licensed upstreams, with the upstream and its license
   named in the README's Provenance section. No customer content, no
   internal repository content, no absolute host paths, no environment
   dumps.
5. **One collector per example.** Two collectors in one recipe makes a
   failure ambiguous, and some combinations conflict outright.

Adding a category is the same move one level up: a new directory beside
`rocprof/` and `proton/`, plus a row in the Categories table above.

## See also

- `docs/profiling-collectors.md` — collector reference: options, artifact
  layout, analysis recipes, troubleshooting.
- `recipes/README.md` — recipe schema.
- `recipes/README-running-recipes.md` — running recipes on a cluster.
