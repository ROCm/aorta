# `rocjitsu_sanitizers` tool

A built-in `aorta.tools` plugin that runs the two rocjitsu sanitizers over the
**top kernels a workload launches** and folds the findings into one
`pass` / `warn` / `fail` guardrail verdict:

- **waitcheck** (static) — `rj_waitcheck` inspects a *saved* code object
  (`.hsaco`, HIP fat binary, executable, shared lib, directory corpus) and
  reports missing AMDGPU waits. No GPU required.
- **ConSan** (dynamic) — the combined HSA-tools hook
  (`librocjitsu_dbi_hooks.so`, loaded via `HSA_TOOLS_LIB`) instruments the code
  object during a real run and reports races. Needs hardware or the RocJITsu
  simulator.

Reconciled against `rocm-systems/emulation/rocjitsu/docs/sanitizers.md` (branch
`shared/rocjitsu/sanitizers`).

## The rocjitsu build (not vendored)

The tool locates the two artifacts at runtime; it never builds or ships them:

| Env var | Points at |
|---|---|
| `ROCJITSU_BUILD` | a rocjitsu build dir (resolves both artifacts) |
| `RJ_WAITCHECK_BIN` | just `rj_waitcheck` |
| `ROCJITSU_SANITIZER_HOOK` | just `librocjitsu_dbi_hooks.so` |

Build only the two targets (not the whole repo):

```sh
cmake -S emulation/rocjitsu -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --target rocjitsu_dbi_hooks rj_waitcheck
export ROCJITSU_BUILD="$PWD/build"
```

When the artifacts are absent, every check records `skipped` and the verdict is
`not_checked` — so the tool is safe to discover/import on any machine.

## Target support (from the doc's table)

| Target | waitcheck | ConSan |
|---|---|---|
| `gfx942`, `gfx950` (MI350X/MI355X), `gfx1100`, `gfx1201`, `gfx1250` | full | full |
| `gfx1150`, `gfx1151`, `gfx1200` | full | unsupported (waitcheck-only) |

ConSan on `gfx950` needs real hardware (no gfx950 simulator). The per-conflict
diagnostic lines that make a race **fail** the gate are only emitted when
`RJ_CONSAN_LOG=1` — pass `consan_log` when you want races caught.

## Bring your own kernels

The tool consumes a **kernel worklist** (`rocjitsu_sanitizers.kernels/1`). Any
workload can produce one; two built-in producers are provided:

- a **Magpie** `benchmark_report.json` (`kernel_summary` / `top_bottlenecks`), and
- a **hipBLASLt GEMM dispatch CSV** (ranked by launch count).

Or hand it a pre-built worklist JSON directly (`kernels=...`), so a workload in
any package can feed its own top kernels in.

## CLI

```sh
# discover tools
aorta tools list

# static waitcheck over saved code objects (no GPU)
aorta tools run rocjitsu_sanitizers \
    --input gemm_csv=gemm_shapes.csv --input isa_dir=./kernel_isa \
    --input target=gfx950 --input checks=waitcheck \
    --output-dir ./san-out

# dynamic ConSan around a real app (on gfx950 hardware)
aorta tools run rocjitsu_sanitizers \
    --input kernels=kernels.json --input target=gfx950 \
    --input checks=consan --input consan_command="python my_repro.py" \
    --input consan_log=1 --output-dir ./san-out
```

## Python

```python
from pathlib import Path
from aorta.tools import get_tool

tool = get_tool("rocjitsu_sanitizers")()
result = tool.invoke(
    inputs={
        "gemm_csv": Path("gemm_shapes.csv"),
        "isa_dir": Path("./kernel_isa"),
        "target": "gfx950",
        "checks": ["waitcheck", "consan"],
        "consan_command": "python my_repro.py",
        "consan_log": True,
    },
    output_dir=Path("./san-out"),
)
print(result["overall_verdict"])   # pass | warn | fail | not_checked
```

## Verdicts

| Verdict | Meaning | Exit code (`aorta tools run`) |
|---|---|---|
| `pass` | check ran, no findings | 0 |
| `warn` | waitcheck found advisory missing-wait(s) | 0 |
| `fail` | ConSan found a race/hazard | 1 |
| `not_checked` | no check ran (all skipped) | 0 |

A **skip is not a pass**: each skipped check records a `reason` and its
`support` block.
