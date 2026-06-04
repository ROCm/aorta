# Buck2

Build and run the AORTA CLI with [Buck2](https://buck2.build/) -- a
fast, hermetic, multi-language build system. The repo already ships a
working Buck2 setup (added in [#187]); this doc is the user-facing
walkthrough.

> The repo's current Buck2 scaffold defines `python_library` /
> `python_binary` targets only -- no `python_test` targets yet, so
> `buck2 test` is not configured. Tests still run via `pytest` (see
> the project's `pyproject.toml`). Wiring up `python_test` targets is
> tracked separately.

[#187]: https://github.com/ROCm/aorta/pull/187

## Why Buck2 for AORTA?

Three concrete wins over `pip install -e . && aorta ...`:

* **Hermetic Python**. The toolchain downloads a pinned CPython 3.13.6
  standalone from python-build-standalone -- *not* your `.venv` Python and
  *not* the host's `/usr/bin/python3`. Every Buck-built `aorta` binary
  runs under the same interpreter ABI on every machine, so the
  Python-interpreter portion of an `env probe` snapshot is identical
  across machines; the rest of the snapshot (ROCm versions, hostname,
  GPU, paths) is host-specific by design and diffs cleanly on the
  fields that actually changed.
* **`aorta env probe` becomes self-documenting**. The snapshot's
  `build_system` field auto-populates with
  `{"kind": "buck2", "buck2_version": ..., "repo_root": ..., "revision": ...}`
  whenever `env probe` is run inside a Buck2 workspace (whether via
  `buck2 run` or pip-installed; detection is workspace-based, not
  provenance-based). Combined with the hermetic
  Python above, two snapshots from different Buck checkouts diff cleanly
  on `revision` -- no bespoke "what built this" plumbing needed.
* **Fast incrementals**. The Buck daemon keeps the parsed target
  graph + Watchman state in memory; subsequent `buck2 run` invocations
  are sub-second when nothing changed.

You can keep using `pip install -e .` -- nothing about Buck2 is
required. This doc is for users who want the determinism and speed
benefits.

## Quick start

```bash
# One-time: ensure buck2 is on PATH. Releases at https://github.com/facebook/buck2/releases
buck2 --version

# From the aorta repo root
buck2 build aorta                     # build the CLI binary
buck2 run aorta -- --help             # build + execute; args after `--` go to aorta
buck2 run aorta -- env probe --summary

# Inspect what Buck2 sees
buck2 targets //:                     # list targets in the root BUCK
buck2 cquery 'deps(aorta)'            # full configured dep graph
buck2 audit config build              # effective config
```

`aorta` here is a `.buckconfig` alias for `//:aorta` -- the canonical
target label. The shorter form is purely command-line sugar; everything
in `BUCK` files uses the full `//:aorta` form.

## The setup at a glance

```text
aorta/
├── .buckconfig                  # cells, toolchains, target aliases
├── .buckroot                    # workspace marker (empty file; what `buck2 root` finds)
├── BUCK                         # //:aorta_lib + //:aorta targets
├── toolchains/BUCK              # system_demo_toolchains() -> CPython 3.13.6
└── third-party/python/BUCK      # click + pyyaml wheels pinned by sha256
```

The `prelude` cell (declared in `.buckconfig`) has no on-disk
directory in this repo. `external_cells.prelude = bundled` tells
Buck2 to use the copy of the OSS prelude shipped inside the `buck2`
binary itself, so nothing is vendored.

Two targets exist today:

* `//:aorta_lib` -- `python_library` over `src/aorta/**/*.py`, with
  ebpf scripts as resources. Deps: `click`, `pyyaml`. Source prefix
  `src/` is stripped so files import as `aorta.foo`, not `src.aorta.foo`.
* `//:aorta` -- `python_binary` with `main_function = "aorta.cli.main"`.
  Depends on `:aorta_lib`. This is the executable.

## Concepts (just enough)

* **Target** -- a named thing you can build / run / test, addressed as
  `//path/from/repo/root:target_name`.
* **Rule** -- a function (built-in or third-party) that turns sources
  into a target. We use `python_library`, `python_binary`,
  `prebuilt_python_library`, `http_file`.
* **Provider** -- the typed output a rule emits. `python_binary` emits
  a `RunInfo` provider; `buck2 run` knows how to execute it.
* **Cell** -- a Buck2-aware sub-repository, defined in the `[cells]`
  section of `.buckconfig`. We have `root` (this repo), `prelude`
  (Buck2's bundled rules), and `toolchains`.
* **Daemon (`buckd`)** -- per-project; holds the parsed graph and
  Watchman state in memory. First invocation is slower (~3s), then
  near-instant for unchanged inputs. Kill with `buck2 kill`.

For a deeper walk-through, read [The Buck2 Book](https://buck2.build/docs/about/why/).

## `aorta env probe` under Buck2

The CLI itself doesn't care which build system launched it. What
*does* change is the `build_system` field, which `env probe`
auto-populates when run **inside a Buck2 workspace** (regardless of
whether the binary was built by Buck2 or pip-installed -- detection is
workspace-based via `buck2 root`, not binary-provenance-based; see
[`src/aorta/instrumentation/build_system.py`]).

[`src/aorta/instrumentation/build_system.py`]: ../src/aorta/instrumentation/build_system.py

```bash
# From inside the aorta repo, run via Buck2
buck2 run aorta -- env probe --field build_system
# -> {"kind": "buck2", "buck2_version": "buck2 c920dd03...", "repo_root": "/path/to/aorta", "revision": "<git sha>"}

# Same binary, but run from outside any Buck2 workspace
(cd /tmp && buck2 root)                     # exits non-zero -> kind=none
# To run aorta from /tmp you'd typically use a pip-installed aorta:
cd /tmp && aorta env probe --field build_system
# -> {"kind": "none"}

# One-screen brief (no file written)
buck2 run aorta -- env probe --summary

# Full snapshot to disk
buck2 run aorta -- env probe -o /tmp/env.json
jq '.build_system, .python_version, (.partial_reasons | length)' /tmp/env.json
```

The two interesting cross-checks:

```bash
# `python_version` is the Buck-pinned CPython (3.13.6) regardless of your venv
buck2 run aorta -- env probe --field python_version

# `--buck-target` makes the probe shell back out to `buck2 cquery` for
# library-identity introspection (issue #163). Empty on pure-Python targets
# like aorta_lib (no ROCm .so deps); populated when run against a
# Buck-buildable C++/HIP target.
buck2 run aorta -- env probe --buck-target aorta_lib --field library_introspection
# -> []
```

`build_system.repo_root` is the *Buck-root* path -- meaningful even
when you run inside a container that mounts the repo at a different
path. If Buck2 isn't on PATH, or your `cwd` is outside any Buck
workspace, `build_system` cleanly degrades to `{"kind": "none"}` --
the field is always present and never `null`, and no entry is
appended to the snapshot's `partial_reasons` list (a non-Buck
environment isn't an error condition, it's just one of the two
documented shapes; see
[`src/aorta/instrumentation/build_system.py`]).

## `aorta triage` under Buck2

`aorta triage` runs a mitigation x environment matrix. Three discovery
subcommands are read-only and demo-friendly; `triage run` actually
executes cells.

Built-in mitigations are defined in `src/aorta/registry/mitigations.py`
(currently ~22 entries spanning core / hardware-queue / RCCL / PyTorch
/ HIP / SDPA-backend groups). The transcripts below show only the
"core" three so the doc doesn't drift as built-ins are added; running
the command on your machine prints the full set.

```bash
# Inspect the registries (built-ins + any installed plugins)
buck2 run aorta -- triage list-mitigations
# NAME      SOURCE  ENV
# none      aorta   (none)
# tf32_off  aorta   DISABLE_TF32=1
# xnack     aorta   HSA_XNACK=1
# ... (~19 more ROCm / HIP / RCCL / PyTorch / SDPA entries; truncated)

buck2 run aorta -- triage list-environments
# NAME     SOURCE  DOCKER  VENV
# default  aorta   -       -
# local    aorta   -       -

# Validate a recipe without executing it
buck2 run aorta -- triage run --recipe recipes/example-fsdp-smoke.yaml --dry-run
# Dry run: fsdp / ticket=EXAMPLE-151
# Cells (3):
#   - baseline-local: mitigations=['none'] environment=local trials=2 steps=100
#   - tf32_off-local: mitigations=['tf32_off'] environment=local trials=2 steps=100
#   - xnack-local:    mitigations=['xnack']    environment=local trials=2 steps=100
```

### Running a recipe end-to-end (graceful degradation)

The public `aorta` build ships no `aorta.workloads` plugins (workloads
register via a separate package's entry-points; see
[Extending: adding workloads via plugins](#extending-adding-workloads-via-plugins)
below).
The example recipe targets `workload: fsdp`, so each cell will error
with `Workload 'fsdp' not found. Available: []`. That's intentional --
the recipe's own header documents it -- and `aorta triage` is built to
keep going:

```bash
buck2 run aorta -- triage run --recipe recipes/example-fsdp-smoke.yaml -v
```

What you get back (under `triage_results/EXAMPLE-151/fsdp/<timestamp>/`):

| Artifact | What's in it |
|---|---|
| `matrix.md` | Human-readable summary table; per-cell row shows `error` in the Confound column |
| `matrix.json` | Schema-stable JSON; each cell carries the captured exception as `cells[*].error` |
| `recipe.resolved.yaml` | The recipe after registry resolution -- reloadable replay artifact |
| `host_env.json` | One `env probe` snapshot for the matrix host scope |
| `environments/<name>/env.json` | One snapshot per environment-axis value |

Exit code policy:

* **Matrix written, baseline anchored** -> exit 0, even if individual
  cells failed pre-setup (`confound: error`). This is what the example
  recipe above produces against the public build: every cell errors
  because the workload is unknown, but the matrix.md is written and
  baseline classification "completed" with all-error rows -- the
  runner treats that as a successful artifact emission.
* **Baseline unrescuable** -> exit 1 via `MatrixIncompleteError`.
  Triggered when every trial in the baseline cell ended in
  `did_not_run` (workload exists, `setup()` ran, but the subprocess
  crashed before reaching the measurement loop) AND no other cell
  qualifies as a fallback baseline. Artifacts are still written; only
  the post-execution classification fails.

For the demo, the first branch is the one you see -- a cleanly emitted
matrix you can inspect with `jq '.cells[].confound' triage_results/.../matrix.json`.

### Sidecar mitigations / environments

For ad-hoc experiments without installing a plugin:

```bash
buck2 run aorta -- triage list-mitigations \
    --mitigations-file examples/mitigations-sidecar.json
buck2 run aorta -- triage list-environments \
    --mitigations-file examples/mitigations-sidecar.json
```

The sidecar entries appear with `SOURCE = sidecar:<filename>`. See
[`src/aorta/registry/README.md`] for the file schema.

[`src/aorta/registry/README.md`]: ../src/aorta/registry/README.md

## Extending: adding workloads via plugins

Workloads are discovered via the `aorta.workloads` entry-point group
(see [`src/aorta/run/discovery.py`]). The same pattern works for the
`aorta.mitigations` and `aorta.environments` groups.

[`src/aorta/run/discovery.py`]: ../src/aorta/run/discovery.py

In a separate Python package that depends on `aorta`:

```toml
# my_plugin/pyproject.toml
[project.entry-points."aorta.workloads"]
my_workload = "my_plugin.workloads:MyWorkload"
```

Wiring that into a Buck2 build requires the plugin to be **shipped as a
wheel**, not as a `python_library`. The reason is mechanical:
`importlib.metadata.entry_points()` reads `.dist-info/entry_points.txt`,
which only exists in built wheels -- a `python_library` ships only `.py`
files and entry-point discovery returns an empty set.

The plugin-side BUCK pattern, given a pre-built
`my_plugin-0.1.0-py3-none-any.whl`:

```python
# third-party/python/BUCK in YOUR repo (next to aorta)
prebuilt_python_library(
    name = "my_plugin",
    binary_src = "wheels/my_plugin-0.1.0-py3-none-any.whl",
    visibility = ["PUBLIC"],
)
```

Then add `"//third-party/python:my_plugin"` to your fork of `:aorta`'s
`deps`. After rebuild, `buck2 run aorta -- triage list-mitigations` and
`buck2 run aorta -- triage run --workload my_workload ...` see the
plugin.

`discover_workloads()` catches any import-time exception per
entry-point and logs a warning -- one broken plugin does not break the
rest, so cross-team plugin packs are safe to layer.

## Common patterns

```bash
# Force a clean build (rare; the daemon's incremental engine is usually right)
buck2 clean
buck2 kill                              # also restart the daemon

# Build under a separate buck-out (useful for docker / CI isolation)
buck2 --isolation-dir docker build aorta
#                     ^^^^^^ -- writes to buck-out/docker/ with its own daemon socket

# Print the resolved run command (the path to the built binary) instead of executing it
buck2 run --emit-shell aorta -- env probe --summary
# -> /path/to/buck-out/.../aorta.par env probe --summary
#   useful when you want to capture the binary path for a separate launcher

# Inspect the most recent build's event log (JSON, one event per line)
buck2 log show | jq -c 'select(.Event.data.SpanStart != null) | .Event.data.SpanStart' | head
```

## Troubleshooting

### "Workload 'foo' not found. Available: []"

You're seeing entry-point discovery return empty. Two checks:

1. Did you install the plugin's wheel (not just source)? `pip show`
   does not report entry points, so check directly via
   `importlib.metadata`:

   ```bash
   python -c "from importlib.metadata import entry_points; \
              print(list(entry_points(group='aorta.workloads')))"
   ```

   Empty list means the wheel's `dist-info/entry_points.txt` either
   wasn't installed or didn't include the `aorta.workloads` group.
   You can also inspect the wheel directly:
   `unzip -p my_plugin-*.whl '*/entry_points.txt'`.
2. If running under Buck2, the plugin's wheel must be a
   `prebuilt_python_library` in the dep graph -- a plain
   `python_library` of the plugin's `.py` files won't carry
   `dist-info`.

### `buck2` is slow on the first command after a config change

`.buckconfig` edits force a daemon restart **and** invalidate the
cached parse state, so the next invocation re-parses the graph from
scratch -- noticeably slower than a normal cold start (see "Daemon
(`buckd`)" in the glossary above) since nothing can be reused. After
that single re-parse, you're back to sub-second incrementals.

### `partial_reasons` complains about `rdhc`, `/opt/rocm/...` etc.

These come from the env probe, not Buck2. The probe runs on whatever
host you're on; CPU-only / no-ROCm hosts cleanly report each missing
field by appending an entry to the snapshot's `partial_reasons` list
rather than crashing. The Buck-built `aorta` binary doesn't paper
over a missing GPU stack -- it surfaces it.

### `Network: Up: 0B  Down: 68MiB` on the first build

Buck2 is fetching the pinned CPython standalone (~50 MB) and the pinned
`click` / `pyyaml` wheels (~5 MB total) from their `http_file` URLs.
Cached under `buck-out/` after the first hit. Subsequent builds are
network-silent.

### `buck2 run aorta -- --version` fails with a missing-metadata error

Click's `--version` decorator queries `importlib.metadata` for the
package version, which only works for pip-installed distributions.
The Buck-built binary's link tree doesn't carry pip dist-info for
`aorta` itself, so the lookup fails. The exact exception text varies
by Click / Python version -- recent Click surfaces it as
`RuntimeError: 'aorta' is not installed`, older versions raise
`PackageNotFoundError` from `importlib.metadata`. Use
`buck2 run aorta -- --help` instead, or query
`buck2 run aorta -- env probe --field python_version` for the
toolchain Python version.

## See also

* [Env Probe](env-probe.md) -- the snapshot schema and `jq` cookbook.
* [`src/aorta/registry/README.md`](../src/aorta/registry/README.md) --
  mitigation / environment plugin authoring; same entry-point pattern
  as workloads.
* [Buck2 docs](https://buck2.build/docs/about/why/) -- upstream
  concept docs.
