# aorta-report Detailed Architecture Plan

**Version:** 2.0  
**Date:** January 2026  
**Status:** Implemented

---

## 1. Overview

`aorta-report` is a unified CLI tool for TraceLens analysis and report generation. This document describes the modular architecture design with **colocated CLI commands**.

### Design Philosophy

- **High Cohesion**: CLI commands live next to their implementation
- **Single Responsibility**: Each package owns its complete interface (API + CLI)
- **Lazy Loading**: Dependencies imported only when commands are invoked
- **Maintainability**: Small, focused files (<300 lines each)

---

## 2. Directory Structure

```
src/aorta/report/
├── __init__.py                 # Package version and exports
├── __main__.py                 # python -m aorta.report support
├── cli.py                      # Main CLI orchestrator (~80 lines)
│
├── analysis/                   # TraceLens analysis modules
│   ├── __init__.py            # Package exports
│   ├── cli.py                 # 'analyze' command group (~150 lines)
│   ├── analyze_gemm.py        # GEMM kernel analysis
│   ├── analyze_single.py      # Single config analysis
│   ├── analyze_sweep.py       # Sweep analysis
│   └── tracelens_wrapper.py   # TraceLens integration
│
├── comparison/                 # Report comparison modules
│   ├── __init__.py            # Package exports
│   ├── cli.py                 # 'compare' command group (~220 lines)
│   ├── combine.py             # Excel file combining
│   ├── gpu_timeline_comparison.py
│   ├── collective_comparison.py
│   └── formatting.py          # Excel formatting utilities
│
├── generators/                 # Report generation modules
│   ├── __init__.py            # Package exports
│   ├── cli.py                 # 'generate' command group (~270 lines)
│   ├── html_generator.py      # HTML report generation
│   ├── excel_report.py        # Final Excel report
│   ├── plot_generator.py      # Plot orchestration
│   └── plot_helper/           # Individual plot functions
│       ├── __init__.py
│       ├── common.py
│       ├── summary_dashboard.py
│       ├── gpu_by_rank.py
│       ├── gpu_percent_change.py
│       ├── gpu_heatmap.py
│       ├── nccl_charts.py
│       ├── gemm_data.py
│       ├── gemm_boxplots.py
│       ├── gemm_violin.py
│       └── gemm_interaction.py
│
├── processing/                 # Data processing modules
│   ├── __init__.py            # Package exports
│   ├── cli.py                 # 'process' command group (~170 lines)
│   ├── gpu_timeline_single.py
│   ├── gpu_timeline_sweep.py
│   ├── process_comms.py
│   └── process_gemm_variance.py
│
├── pipelines/                  # Pipeline orchestrators
│   ├── __init__.py            # Package exports
│   ├── cli.py                 # 'pipeline' command group (~200 lines)
│   ├── summary_pipeline.py    # Full analysis pipeline
│   └── gemm_pipeline.py       # GEMM analysis pipeline
│
└── templates/                  # HTML templates
    ├── __init__.py
    ├── performance_report_template.py
    └── sweep_comparison_template.py
```

---

## 3. CLI Architecture

### 3.1 Main Orchestrator (`cli.py`)

The main `cli.py` is a **thin orchestrator** (~80 lines) that:

1. Defines the root `@click.group()` with global options (`--verbose`, `--quiet`)
2. Imports command groups from each package
3. Registers them with `cli.add_command()`

```python
# cli.py - Main orchestrator
import click
from . import __version__

@click.group()
@click.version_option(version=__version__, prog_name="aorta-report")
@click.option("-v", "--verbose", is_flag=True)
@click.option("--quiet", is_flag=True)
@click.pass_context
def cli(ctx, verbose, quiet):
    """aorta-report: Unified CLI for TraceLens analysis."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["quiet"] = quiet

# Register command groups from subpackages
from .analysis.cli import analyze
from .comparison.cli import compare
from .generators.cli import generate
from .processing.cli import process
from .pipelines.cli import pipeline

cli.add_command(analyze)
cli.add_command(compare)
cli.add_command(generate)
cli.add_command(process)
cli.add_command(pipeline)

def main():
    cli(obj={})
```

### 3.2 Package CLI Modules

Each package has its own `cli.py` that defines:

1. A `@click.group()` for the command group
2. All subcommands using `@group.command()`
3. Imports from the same package (relative imports)

**Example: `analysis/cli.py`**

```python
# analysis/cli.py
import click
from pathlib import Path

@click.group()
@click.pass_context
def analyze(ctx):
    """Run TraceLens analysis on traces."""
    pass

@analyze.command("single")
@click.argument("trace_dir", type=click.Path(exists=True))
@click.option("--geo-mean", is_flag=True)
@click.pass_context
def analyze_single(ctx, trace_dir, geo_mean):
    """Analyze a single configuration."""
    from . import analyze_single_config  # Relative import from same package
    
    result = analyze_single_config(
        input_dir=Path(trace_dir),
        use_geo_mean=geo_mean,
        verbose=ctx.obj.get("verbose", False),
    )
    click.echo(f"Complete: {result}")
```

---

## 4. Command Reference

### 4.1 Command Groups Summary

| Group | File | Commands | Lines |
|-------|------|----------|-------|
| `analyze` | `analysis/cli.py` | `single`, `sweep`, `gemm` | ~150 |
| `compare` | `comparison/cli.py` | `gpu_timeline`, `collective` | ~220 |
| `generate` | `generators/cli.py` | `html`, `excel`, `plots` | ~270 |
| `process` | `processing/cli.py` | `gpu-timeline`, `comms`, `gemm-variance` | ~170 |
| `pipeline` | `pipelines/cli.py` | `summary`, `gemm` | ~200 |

### 4.2 Full Command Tree

```
aorta-report
├── --version
├── --verbose / -v
├── --quiet
│
├── analyze
│   ├── single <TRACE_DIR>
│   │   ├── --individual-only
│   │   ├── --collective-only
│   │   ├── --geo-mean
│   │   ├── --short-kernel-threshold INT
│   │   ├── --topk-ops INT
│   │   └── -o, --output PATH
│   │
│   ├── sweep <SWEEP_DIR>
│   │   ├── --geo-mean
│   │   └── -o, --output PATH
│   │
│   └── gemm <REPORTS_DIR>
│       ├── -t, --threads INT (multiple)
│       ├── -c, --channels INT (multiple)
│       ├── -r, --ranks INT (multiple)
│       ├── --top-k INT
│       └── -o, --output PATH
│
├── compare
│   ├── gpu_timeline
│   │   ├── -b, --baseline PATH (required)
│   │   ├── -t, --test PATH (required)
│   │   ├── --baseline-label TEXT
│   │   ├── --test-label TEXT
│   │   └── -o, --output PATH (required)
│   │
│   └── collective
│       ├── -b, --baseline PATH (required)
│       ├── -t, --test PATH (required)
│       ├── --baseline-label TEXT
│       ├── --test-label TEXT
│       └── -o, --output PATH (required)
│
├── generate
│   ├── html
│   │   ├── --mode [sweep|performance] (required)
│   │   ├── --sweep1 PATH
│   │   ├── --sweep2 PATH
│   │   ├── --label1 TEXT
│   │   ├── --label2 TEXT
│   │   ├── --plots-dir PATH
│   │   └── -o, --output PATH (required)
│   │
│   ├── excel
│   │   ├── --gpu-combined PATH (required)
│   │   ├── --gpu-comparison PATH (required)
│   │   ├── --coll-combined PATH (required)
│   │   ├── --coll-comparison PATH (required)
│   │   ├── --baseline-label TEXT
│   │   ├── --test-label TEXT
│   │   └── -o, --output PATH (required)
│   │
│   └── plots
│       ├── -i, --input PATH
│       ├── --excel-input PATH
│       ├── --gemm-csv PATH
│       ├── --type [all|summary|gemm]
│       ├── --dpi INT
│       └── -o, --output PATH (required)
│
├── process
│   ├── gpu-timeline <INPUT_DIR>
│   │   ├── --mode [auto|single|sweep]
│   │   ├── --geo-mean
│   │   └── -o, --output PATH
│   │
│   ├── comms <SWEEP_DIR>
│   │   └── -o, --output PATH
│   │
│   └── gemm-variance <INPUT_CSV>
│       ├── --base-path PATH (required)
│       ├── --tolerance FLOAT
│       └── -o, --output PATH
│
└── pipeline
    ├── summary
    │   ├── -b, --baseline PATH (required)
    │   ├── -t, --test PATH (required)
    │   ├── -o, --output PATH (required)
    │   ├── --baseline-label TEXT
    │   ├── --test-label TEXT
    │   ├── --skip-tracelens
    │   ├── --gpu-timeline / --no-gpu-timeline
    │   ├── --collective / --no-collective
    │   ├── --final-report / --no-final-report
    │   ├── --plots / --no-plots
    │   └── --html / --no-html
    │
    └── gemm
        ├── --sweep-dir PATH (required)
        ├── -o, --output PATH (required)
        ├── --top-k INT
        ├── -t, --threads INT (multiple)
        ├── -c, --channels INT (multiple)
        ├── --timestamps / --no-timestamps
        └── --plots / --no-plots
```

---

## 5. Data Flow

### 5.1 Summary Pipeline Flow

```
┌─────────────────┐     ┌─────────────────┐
│ Baseline Traces │     │   Test Traces   │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌────────────────────────────────────────┐
│        analyze single (TraceLens)       │
│   analysis/analyze_single.py            │
└────────────────────┬───────────────────┘
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│  GPU Timeline   │     │  Collective     │
│    Reports      │     │    Reports      │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│ compare         │     │ compare         │
│ gpu_timeline    │     │ collective      │
└────────┬────────┘     └────────┬────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
         ┌───────────────────────┐
         │   generate excel      │
         │   (Final Report)      │
         └───────────┬───────────┘
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│ generate plots  │     │ generate html   │
└─────────────────┘     └─────────────────┘
```

### 5.2 GEMM Pipeline Flow

```
┌─────────────────┐
│   Sweep Dir     │
│ tracelens_      │
│ analysis/       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  analyze gemm   │
│  (Top-K Kernels)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ process         │
│ gemm-variance   │
│ (Add timestamps)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ generate plots  │
│ --type gemm     │
└─────────────────┘
```

---

## 6. Benefits of Colocated CLI Design

### 6.1 Comparison with Alternatives

| Approach | Pros | Cons |
|----------|------|------|
| **Single cli.py** | All in one place | 1000+ lines, hard to maintain |
| **Separate cli/ folder** | Clear separation | Jumps between directories |
| **Colocated** ✓ | High cohesion, easy to find | Need to look at multiple files |

### 6.2 Key Benefits

1. **Discoverability**: Working on `analysis`? CLI is right there in `analysis/cli.py`

2. **Ownership**: Each package owns its complete interface:
   - `analysis/` → business logic + CLI
   - No cross-cutting changes needed

3. **Testing**: Test `analysis/cli.py` with `analysis/` fixtures:
   ```python
   # tests/test_analysis_cli.py
   from aorta.report.analysis.cli import analyze
   ```

4. **Lazy Loading**: Commands import dependencies only when invoked:
   ```python
   @analyze.command("gemm")
   def analyze_gemm(ctx, ...):
       from . import analyze_gemm_reports  # Imported only when command runs
   ```

5. **Scalability**: Adding new functionality:
   - Add new module to package
   - Add command to package's `cli.py`
   - No changes to main `cli.py`

---

## 7. Implementation Status

### 7.1 Completed Commands

| Command | Package | Status |
|---------|---------|--------|
| `analyze single` | `analysis/` | ✅ |
| `analyze sweep` | `analysis/` | ✅ |
| `analyze gemm` | `analysis/` | ✅ |
| `compare gpu_timeline` | `comparison/` | ✅ |
| `compare collective` | `comparison/` | ✅ |
| `generate html` | `generators/` | ✅ |
| `generate excel` | `generators/` | ✅ |
| `generate plots` | `generators/` | ✅ |
| `process gpu-timeline` | `processing/` | ✅ |
| `process comms` | `processing/` | ✅ |
| `process gemm-variance` | `processing/` | ✅ |
| `pipeline summary` | `pipelines/` | ✅ |
| `pipeline gemm` | `pipelines/` | ✅ |

### 7.2 Planned Commands

| Command | Package | Status |
|---------|---------|--------|
| `compare runs` | `comparison/` | ⏸️ Deferred |

---

## 8. File Size Summary

| File | Lines | Description |
|------|-------|-------------|
| `cli.py` | ~80 | Main orchestrator |
| `analysis/cli.py` | ~150 | analyze commands |
| `comparison/cli.py` | ~220 | compare commands |
| `generators/cli.py` | ~270 | generate commands |
| `processing/cli.py` | ~170 | process commands |
| `pipelines/cli.py` | ~200 | pipeline commands |
| **Total CLI** | **~1,090** | Split across 6 files |

**Before refactoring**: 1,182 lines in single file  
**After refactoring**: ~80 lines main + ~200 avg per package CLI

---

## 9. Adding New Commands

### 9.1 Adding a Command to Existing Group

1. Open the package's `cli.py` (e.g., `analysis/cli.py`)
2. Add the command:

```python
@analyze.command("new_command")
@click.argument("input_path", type=click.Path(exists=True))
@click.option("--option", help="Some option")
@click.pass_context
def analyze_new_command(ctx, input_path, option):
    """New command description."""
    from . import new_function  # Import from same package
    
    result = new_function(input_path, option)
    click.echo(f"Done: {result}")
```

3. No changes needed to main `cli.py`

### 9.2 Adding a New Command Group

1. Create new package: `new_package/`
2. Create `new_package/__init__.py` with exports
3. Create `new_package/cli.py`:

```python
import click

@click.group()
@click.pass_context
def new_group(ctx):
    """New command group description."""
    pass

@new_group.command("subcommand")
def new_subcommand():
    """Subcommand description."""
    pass
```

4. Register in main `cli.py`:

```python
from .new_package.cli import new_group
cli.add_command(new_group)
```

---

## 10. Related Documentation

- [USER_GUIDE.md](./USER_GUIDE.md) - End-user documentation
- [ANALYZE_CMD_DEV_DOCS.md](./ANALYZE_CMD_DEV_DOCS.md) - Analyze implementation details
- [COMPARE_CMD_DEV_DOCS.md](./COMPARE_CMD_DEV_DOCS.md) - Compare implementation details
- [GENERATE_EXCEL_DEV_DOCS.md](./GENERATE_EXCEL_DEV_DOCS.md) - Excel generation details
- [GENERATE_PLOTS_DEV_DOCS.md](./GENERATE_PLOTS_DEV_DOCS.md) - Plot generation details
- [PIPELINE_DEV_DOCS.md](./PIPELINE_DEV_DOCS.md) - Pipeline implementation details
