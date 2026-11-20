# Build or reuse the input for Waitcheck

Read this when the user has source or an uncertain compiled artifact. Prefer the
project's existing build/export path; the user does not need to register an AORTA
workload or use a packaged demonstration.

## Choose the next operation

| Available input | Next step |
| --- | --- |
| Matching final object with known build/target/specialization | Preserve it and inspect exact selection; no rebuild just to use this skill |
| Object with unknown identity/target | Inspect it and reconcile provenance before deciding to reuse or rebuild |
| HIP/C++ or assembly source | Identify the project's build target and final device artifact. Use the project's compiler/assembler and link or export steps for the intended variant |
| Triton source | Use the project's compiled-kernel export, or the bounded helper below when its supported inputs fit |
| Saved report only | Interpret that evidence; no new compiler, image or scan is needed |

For an existing object, the requested target may differ from the physical host:
Waitcheck is offline. For a new MI350-native build verify gfx950 and wave64.
Always pass target and wave size explicitly to a compiler that otherwise defaults
to something else. A compiler argument is build evidence, not independent proof
of the emitted object's architecture.

For HIP or assembly, start with the build command used for this kernel. If it
produces a standalone device object, use that output. If device code is embedded
in a host executable or library, identify the selected device image with the
installed tool's inventory, or use the project's extraction/export path. Keep
the parent artifact and the relationship to the extracted bytes. If the build
command or packaging is unknown, ask for the build target or output listing.

### HIP source and embedded device code

For a single HIP translation unit with no separate device-link step, this
Linux example uses the project's target, include paths, definitions and optimization
settings. Use a new output directory. Record the compiler version and actual flags.

```bash
"$HIPCC" --genco --offload-arch="$BUILD_TARGET" \
  "${PROJECT_FLAGS[@]}" "$SOURCE" -o "$FRESH_BUILD_DIR/kernel.hsaco"
```

Here `HIPCC` is the ROCm hipcc driver and `PROJECT_FLAGS` is a Bash array of
the selected project flags. [HIP kernel compilation](https://rocm.docs.amd.com/projects/HIP/en/docs-7.0.1/how-to/kernel_language_cpp_support.html#kernel-compilation).
For a HIP program with a host main, a recorded alternative is
`amdclang++ -x hip --offload-arch="$BUILD_TARGET" "$SOURCE" -o "$WORK/program"`
with the project's normal flags. Inventory that linked host artifact or extract
the selected device image below. For a multi-file build, use the project's final device-link target. Verify the
emitted artifact contains a final device image and the intended kernel entry.
For a newer target such as gfx1250, some installed compilers reject the processor.
Keep that failure and locate a compiler supporting the requested target.

If the final host ELF contains a Clang offload bundle in `.hip_fatbin`, prefer
direct inventory of the host artifact when supported. For extraction, set
`HOST_ELF` to the original, work in a fresh directory and stop on any failure.

```bash
llvm-objcopy --dump-section ".hip_fatbin=$WORK/device.bundle" "$HOST_ELF" "$WORK/host-copy"
clang-offload-bundler --type=o --input="$WORK/device.bundle" --list
# Select the exact listed bundle ID for the intended target and feature settings.
clang-offload-bundler --type=o --input="$WORK/device.bundle" --targets="$BUNDLE_ID" \
  --unbundle --output="$WORK/kernel.hsaco"
```

Preserve the parent and extracted hashes and bundle ID. Inventory the extracted
file independently: its image index can differ from the parent's. Multiple
matching images or a different packaging format require the project's exporter
or version-specific inspection. These are illustrative commands; the installed
tool's options and output format need verification. The standalone HIP and
extraction commands above were not exercised in this review. See
[Clang HIP device compilation](https://clang.llvm.org/docs/HIPSupport.html#device-code-compilation)
and [Clang offload bundle formats and extraction](https://clang.llvm.org/docs/ClangOffloadBundler.html).

For Triton, use the project's exported specialization or the
[bounded Triton helper](#bounded-triton-helper) below.

## Establish the build specialization

Read the selected entry and relevant helpers before preparing its build. Explain
briefly what it computes, which buffers it reads/writes, and which shapes, strides,
types and compile-time choices select the requested variant. Use facts already in
the source/project configuration; ask only for unresolved choices that affect the
build or interpretation. A model name or kernel filename alone is insufficient.

For example, a slot-mapping kernel's index/layout choices, a rotary kernel's
in-place or separate-output variant, and an attention kernel's tile/masking
choices can describe different specializations. Establish the actual choices
from that case; do not apply these examples as defaults or claim numerical
correctness from understanding the operation.

Preserve optional-argument forms: an absent argument, an integer offset and a
pointer to offsets are different specializations. Do not replace an explicit
`None` with a dummy tensor/pointer, or toggle a branch, to satisfy an exporter.
If faithful preparation is unsupported, keep that case blocked and explain the
missing capability while using the project's supported export path where available.

Reuse known source revision, dependencies and project build commands. Collect only
the missing facts needed to build the intended variant:

- Exact file and entry function, signature/pointer types, constexpr values and
  selected configuration, compiler/Triton/ROCm versions and compiler options.
- Source/dependency paths, wrapper/heuristic behavior, architecture-sensitive
  imports, and the intended object target and wave size.
- Whether the scenario was captured from a real launch or synthesized only for
  compilation. A field saying `role: launched` does not override a known fallback.

Distinguish the selected implementation from the surrounding model. A Triton
alternative or simulator stand-in does not check a different HIP/C++ operation
used on the reported model path. Preserve that scope even when both operations
have similar names or shapes. Scenario names also do not establish different
compiler specializations: compare the actual signature, constants and options;
keep runtime-case identities without claiming extra static coverage.

Check the compiler's numeric types against storage notes. An eight-bit buffer or
a range constant does not by itself establish FP8 conversion. For example, a
numeric cast to a uint8 pointer's element type differs from a cast to an FP8 type.
Packed integer storage can be valid when an explicit encoding/bitcast or operation
format contract supplies that meaning. If those facts conflict or are missing,
resolve the intended signature/format before building that variant; preserve the
original evidence rather than silently changing the dtype, range or arithmetic.

Separate numeric compiler settings from prose annotations. For example, a replay
record may supply `num_stages: 2` and a documentation field saying `default`.
Preserve the note as metadata and the explicit numeric setting as the build option;
do not send prose to the compiler or infer a missing number from that word. This
records the replay configuration, not proof of the original compiler's implicit
default. Resolve substantive conflicts or missing settings before building.

For Triton, a file may contain helpers as well as launch entries. Do not guess a
symbol from a filename or choose the first suffix match. Preserve explicit aliases,
selected tuning/configuration constants and function/tuple constants. A module
that chooses types or code using the active runtime target needs target-consistent
preparation; passing a different `--arch` later does not undo those decisions.

When preparing from named capture records, bind types/constants to the exact
source parameter names rather than assuming the records follow declaration order.
A name-keyed AOT signature is not a positional runtime launch. Reject missing,
duplicate or unexpected names and retain the original capture order and metadata;
do not rearrange a real program's launch arguments based on that AOT mapping.
An argument can itself be an ordered tuple of pointers. Preserve its grouping,
member count, order and individual element types; a one-member tuple is not a
scalar pointer. Do not flatten peer outputs into new function parameters.

Keep runtime dimensions/strides separate from compile-time choices. For example,
a persistent normalization kernel can take row counts at runtime while fixing
`NUM_PRGMS` at compilation. Preserve the selected configuration; do not recompute
that constant from the compiler host's GPU. Target-dependent padding can also
change output shapes or strides supplied at runtime. Record those layout choices
without turning runtime arguments into constexprs or claiming they were exercised
by an offline build.

A default in the Python signature is not an established value for an AOT case.
Supply every parameter explicitly from the selected project configuration,
including defaulted parameters and optional None values. Static preparation must
not execute or infer default expressions. Python still evaluates those expressions
when the module is actually imported. Preserve scalar-width annotations and all
explicit compiler settings, including known AMD options nested in project metadata;
do not silently drop settings or choose the last conflicting value.

Resolve actual dependencies, such as AITER, in the selected build environment.
Do not invent replacement modules, erase decorators or change arithmetic/types to
make compilation pass. Inspect the source and authorize the build scope: importing
a module and evaluating its heuristics executes Python code. Compilation helpers
do not provide a sandbox.

Check platform-specific intrinsics in the selected branches. For example, CUDA
programmatic dependent-launch operations such as `gdc_wait` are not interchangeable
with AMD instruction-counter waits. Their presence calls for specialization and
compiler-support checks, not automatic missing-wait attribution or source edits.

## Find a usable compiler

Start with the project's known interpreter, build command or compiler image.
The compiler's Python can differ from AORTA's Python; missing Triton in the
AORTA environment is not a reason to install it there automatically. Verify a
known compiler candidate before searching more broadly. Ask for its location
or image only when it is missing from the available context.

Distinguish image inspection, starting a temporary container, compiler imports
and source compilation when proposing permissions. A version/API import check
does not establish that the selected target compiles. A previously used image
still needs current availability and access checks; record its actual identity.

If a container-access attempt is denied, preserve the error and offer exact
commands for the user's own node terminal instead of repeating the denied operation or
changing machine permissions. Use the user's established access method. A sudo
denial establishes failure of that method, not whether normal Docker access
works. Keep untested access methods unknown and propose only the needed bounded
check before ruling out the container route. A password-required `sudo -n`
failure cannot prompt; a permitted interactive sudo command belongs in the
user's terminal, never a request to share a password.

The handoff should keep the selected absolute source/configuration paths,
kernel, target/wave size and fresh output location. Include bounded commands and
how to retain stdout, stderr, exits and build receipts. State which actions
remain unapproved. Retain the outer command actually run, relevant nonsecret
environment settings, host/container path mappings and actual image identity in
the [procedure's evidence record](waitcheck-procedure.md#2-prepare-one-recipe-and-its-evidence-record).
The helper's argv alone does not record the container launch. Ask for the
resulting outputs, then resume object inspection and AORTA preparation from
those artifacts. Attribute manually run steps to the
user; do not claim the agent executed or independently verified them. If access
remains unavailable, identify that prerequisite and keep later stages pending.

## Bounded Triton helper

[waitcheck_compile.py](../scripts/waitcheck_compile.py) prepares a single explicitly
selected Triton specialization. It requires Triton in the selected interpreter;
installing the skill does not install the compiler. Its CPU fixture tests are
not a native compiler qualification. Real MI350 compilation is still pending.

The helper takes the source, exact symbol, a JSON build configuration, explicit
target/wave size and a fresh output directory. Use the project's exporter when
the helper cannot faithfully express its wrappers or constants. Do not substitute
dummy values merely because a real specialization is unavailable.

Example command shape, after source/configuration review and required
authorization. Set `BUILD_TARGET` and `WAVE_SIZE` from the intended build
specialization. `PYTHON` is the interpreter containing its Triton compiler.

```bash
timeout --signal=TERM --kill-after=10s 180s \
  "$PYTHON" "$SKILL/scripts/waitcheck_compile.py" \
  --source "$SOURCE" --kernel "$EXACT_PYTHON_SYMBOL" --config "$BUILD_CONFIG" \
  --arch "$BUILD_TARGET" --warp-size "$WAVE_SIZE" --output "$FRESH_BUILD_DIR"
```

Add `--python-path` for each explicitly selected dependency/source directory.
The helper accepts `JITFunction`, `staticmethod` and heuristics resolvable from
concrete supplied arguments. It refuses unresolved autotuners rather than running
autotuning or stripping configuration. Use the project exporter or prepare a
faithfully selected configuration for unsupported wrappers/function constants.
Keep named `repr` and `launch_metadata` callbacks on a JIT decorator when present;
they are different from an autotuner. Do not strip them to pass static preparation.
Their compatibility and possible compile-time effects still need source review
and actual compiler evidence; importing source remains unsandboxed.
Record the source callable and emitted kernel symbol separately: a naming callback
can produce a different binary entry name. Use the exact inspected native name
in the recipe, tied to the matching source/build/object; do not guess a suffix
match, force the Python name or remove the callback to make the names agree.

For a reviewed source that queries Triton's active target during import, the
opt-in `--source-target compile` mode supplies the same explicit `--arch`/wave
target before import and keeps it through heuristic evaluation and AOT compilation.
For example, a module may choose FP8 formats while loading even when the selected
kernel uses BF16 tensors. This answers only the target query; it does not assert
that a GPU of that architecture is present. Default `--source-target runtime`
leaves normal runtime target selection unchanged.

Use compile mode only when the source's target queries are suitable for this
limited context. Triton device, stream, launcher and other unsupported runtime
requests fail the build, even if source catches the initial refusal. Direct
Torch/JAX device calls are outside this guard; importing Python remains unsandboxed.
The helper checks a version-sensitive driver interface, refuses unsupported state,
restores the previous active provider and records the mode, target-query count,
refusals and restoration result in build.json. A failed restoration is a failed
build. Check these fields and native compiler behavior before accepting this route.
A zero target-query count can be valid when the source does not query that target;
do not impose another kernel's query count as a universal requirement. This counter
does not measure Waitcheck coverage or detect every hardware API call.

The configuration contains `signature`, `constexprs`, `options` and optional
`heuristic_args` objects. For a function whose actual arguments match this
illustration, a configuration could be:

```json
{
  "signature": {"X": "*fp32", "Y": "*fp32", "N": "i32", "BLOCK": "constexpr"},
  "constexprs": {"BLOCK": 128},
  "options": {"num_warps": 4, "num_stages": 2},
  "heuristic_args": {"N": 1024}
}
```

These values are illustrative, not a default specialization for an unknown
kernel. Preserve legitimate non-finite constants from the project's input format;
do not replace them to satisfy a stricter analysis parser.
The helper's JSON-safe build receipt can represent such a constant as
`{"nonfinite": "-Infinity"}`. That records the value; it is not a replacement
dictionary to copy into the compiler configuration's scalar constexpr field.

For a flat pointer tuple, a signature entry such as
`"PeerYvs": ["*bf16", "*bf16"]` represents one two-member tuple argument.
The helper converts that nonempty array to a Triton tuple type while preserving
order. Nested tuples, mixed pointer/scalar arrays and tuple constexprs are outside
this narrow signature extension; use a faithful project exporter for those cases.

The fresh build directory retains `build.json`, `build.log` and, on successful
compilation, `kernel.hsaco`. Exit0 means the helper produced a build artifact;
it is not a sanitizer verdict. The receipt leaves independent object-target and
image/entry inspection unresolved. Failure exits retain build diagnostics where
the output directory could be created; do not infer a completed build from a
partially written artifact.

For each build, preserve the input configuration, source hash, compiler identity,
actual argv, stdout/stderr/exit and emitted object/hash. Missing dependencies,
unsupported wrappers and compiler errors remain failed preparation; no Waitcheck
result exists yet. Keep failed output rather than retrying with weaker settings.
Bound the build command externally using the environment's reviewed process
controls; do not claim descendant cleanup from an outer timeout alone.

## Inspect, scan, and account for cases

After building or reusing an object, verify its actual target, selected image,
exact kernel entry and `.text` offset with version-appropriate inspection.
Compiler metadata or a Python function name alone is not that verification.
Continue with the [AORTA procedure](waitcheck-procedure.md). Keep compilation,
dry run, native scan and report interpretation as separate statuses.

For a multi-kernel operation, record the individual entries and their relationship.
For example, split attention writes partial results in one kernel and combines
them in a reduction kernel. Each static check needs its own verified object/entry
and result. A check of one stage, or even separate checks of both, does not test
the runtime handoff, launch ordering, buffer contents or numerical result.

For a corpus, retain model/specification, source, specialization, scenario, target
and object identity in each case record. Duplicate names and existing result files
must not collapse distinct requests. Classify helper/unconfigured sources explicitly;
unresolved cases remain work, not implicit passes or exclusions.

Start a corpus with a small representative set of unchanged source kernels,
resolve demonstrated build/selection/workflow gaps, then expand the requested
list. Report preparation failures separately from warnings and scoped no-finding
results. An unmodified kernel is not a known-clean control merely because it
compiled or ran successfully.

When the user requests deliberate-fault validation, keep originals unchanged and
use separately identified copies of selected cases. Specify the intended wait
hazard and review its instruction dependency independently of the checker result.
Use an unchanged baseline, an intended-fault copy and a matching wait-restored
copy with the same target/build choices. Keep their actual outcomes separate.
An ordinary wrong-answer edit or out-of-bounds access is not by itself a
Waitcheck fault. A restored wait concerns that dependency, not general correctness.
Verify the change in the final compiled instructions: a source edit may be
repaired by the compiler. Preserve target, specialization and entry mapping,
record changed fingerprints, and compare the modified and original artifacts
through the same AORTA route. Do not assume the original is free of other hazards
or dispatch an intentionally faulty artifact without separate authorization.

For a requested development control, an opaque inline-assembly load/use block
can preserve an intended dependency for inspection. Treat it as an unverified
candidate until the selected compiler/target actually emits the intended
instructions, operands and reachable use. A compiler memory clobber or side-effect
annotation does not insert a GPU counter wait; a NOP does not guarantee a load
completed. Check that the restored variant's wait protects the same dependency.
Do not byte-search/patch apparent wait opcodes without verified instruction
boundaries, target and selected-entry scope. Preserve unrelated findings; a
clean-looking faulty candidate leaves detection/coverage unresolved rather than
qualifying the checker. Compilation or object hashes alone cannot establish this.

Known-fault/corrected controls help qualify detection on a target; they are not
a prerequisite for helping a user compile or scan their own kernel. A no-finding
case alone does not qualify the checker, and this skill's actual loaded-agent
behavior needs its own evidence. Keep those validation claims separate.

## Implementation basis

The helper uses Triton's [ASTSource compiler interface](https://github.com/triton-lang/triton/blob/v3.6.0/python/triton/compiler/compiler.py)
and the [wrapper behavior](https://github.com/triton-lang/triton/blob/v3.6.0/python/triton/runtime/autotuner.py)
inspected at v3.6.0. The optional import context uses the inspected
[DriverConfig interface](https://github.com/triton-lang/triton/blob/v3.6.0/python/triton/runtime/driver.py)
and [DriverBase contract](https://github.com/triton-lang/triton/blob/v3.6.0/python/triton/backends/driver.py).
Version changes and real project builds still need compatibility
validation. Support is determined by actual behavior, not these links alone.
