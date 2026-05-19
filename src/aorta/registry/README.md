# `aorta.registry` — mitigations + environments

Two small registries that ship with aorta:

- **Mitigations** (`name → env vars`) — process-level flags applied just before
  the workload subprocess launches. Examples: `tf32_off`, `xnack`.
- **Environments** (`name → docker / venv / buck_target recipe`) — baseline
  state of the process, container, or Buck-built binary the workload runs in.
  Examples: `local`, `default`.

Both follow the same shape: built-ins ship from this package; external
contributions arrive via Python entry-points and are merged at runtime.

## Adding a mitigation

There are two paths. **For most contributions, Path 2 (plugin) is the right
answer** — Path 1 is reserved for runtime-level flags that benefit every
workload.

### Path 1 — built-in (runtime-level flags only)

Built-ins are for env vars read by a **runtime or library** — ROCm, hipBLASLt,
PyTorch, NCCL, OpenMP, the Linux kernel — **not** by workload Python code.

The defining property: setting the var produces the same effect regardless of
which workload runs. If your workload's training script has to call
`os.environ.get(...)` for the var to take effect, it is NOT a built-in
candidate — it's a plugin candidate (Path 2).

| Qualifies as built-in | Does NOT qualify (use Path 2) |
|---|---|
| `DISABLE_TF32` (hipBLASLt reads it) | `AMP_DTYPE` (only the workload's Python reads it) |
| `HSA_XNACK` (ROCm runtime reads it) | `SHAMPOO_PRECONDITIONER_DTYPE` (only meta-recom reads it) |
| `CUDA_LAUNCH_BLOCKING` (PyTorch reads it) | Any custom flag your workload introspects |
| `NCCL_DEBUG`, `OMP_NUM_THREADS`, `LD_PRELOAD` | Anything that's a silent no-op on workloads that don't read it |

If a Path 1 candidate qualifies, edit `src/aorta/registry/mitigations.py` and
add one entry to the dict:

```python
BUILTIN_MITIGATIONS = {
    "none":     {},
    "tf32_off": {"DISABLE_TF32": "1"},
    "xnack":    {"HSA_XNACK": "1"},
    "no_sdma":  {"HSA_ENABLE_SDMA": "0"},   # <-- your addition
}
```

Open a PR. Adding a built-in requires sign-off from an aorta core maintainer —
the bar is "would this benefit any aorta user, on any workload?".

> **No need to clone:** you can do this entirely in the GitHub web UI. Navigate
> to `src/aorta/registry/mitigations.py`, click the pencil icon, edit the dict,
> scroll down, fill in a PR title, click "Propose changes". Done.

### Path 2 — plugin (everything else)

For workload-coupled flags, customer-specific recipes, experimental things, or
anything where you maintain your own Python package: ship the mitigation from
your package via the `aorta.mitigations` entry-point group.

In your package's `pyproject.toml` — **one entry-point per mitigation**, the
EP name IS the mitigation name (mirrors how `aorta.workloads` registers
workloads):

```toml
[project.entry-points."aorta.mitigations"]
my_flag    = "my_package.mitigations:MY_FLAG"
other_flag = "my_package.mitigations:OTHER_FLAG"
```

In your package's source — each entry is a plain `dict[str, str]` of env vars:

```python
# my_package/mitigations.py
MY_FLAG: dict[str, str]    = {"MY_ENV_VAR": "1"}
OTHER_FLAG: dict[str, str] = {"OTHER_FLAG": "verbose"}
```

After `pip install` (or `pip install -e .`), each mitigation appears in
`aorta mitigations list` tagged with your package name. `pip show -f
my_package` lists every entry-point individually, so you can see at a glance
which names your dist contributes.

> Plugin authors: re-run `pip install` after editing `pyproject.toml` —
> entry-points are read at install time, not at import time.

### Path 3 — JSON sidecar (ad-hoc, throwaway, shareable)

For "I want to try these named bundles on my box, share the file with a
colleague over Slack, run a triage sweep against them, and throw it away when
I'm done" — write a JSON file and pass it on the command line.

**Wired today** (B3.1):

```bash
aorta mitigations list --file ./my-experiments.json
aorta environments list --file ./my-experiments.json
```

**Lands with B1 / B2** — the flags below are accepted by the CLI (Click
checks the file exists), but `aorta run` and `aorta triage run` themselves
are not yet implemented: the JSON is not parsed and the merged registries
are not yet consumed. Today the commands raise "not yet implemented" before
reaching the loader:

```bash
aorta run    --workload fsdp --mitigations-file ./my-experiments.json --mitigations my_flag
aorta triage run --workload fsdp --mitigations-file ./my-experiments.json ...
```

`--mitigations-file` is repeatable. Each file may declare mitigations,
environments, or both — both registries are populated from the same file.
Files are merged left-to-right and combine with built-ins + plugins under
the same collision rule.

`my-experiments.json`:

```json
{
  "version": 1,
  "mitigations": {
    "my_flag":  { "MY_ENV_VAR": "1" },
    "amp_bf16": { "AMP_DTYPE": "bfloat16" }
  },
  "environments": {
    "my_local_image": { "docker": "myorg/private:test@sha256:..." },
    "host_venv_3_12": { "venv":   "/home/me/.venvs/aorta-3.12" }
  }
}
```

Both top-level keys are optional (a file can ship only mitigations, or only
environments, or both). `version: 1` is required — it's a forward-compat
lever; an aorta build that doesn't understand a future `version: 2` rejects
the file cleanly instead of misinterpreting it.

Sidecar entries list with `source_package = "sidecar:<filename>"` so it's
obvious which file shipped which entry. Starting template:
[`examples/mitigations-sidecar.json`](../../../examples/mitigations-sidecar.json).

## Trying things locally without upstreaming

Sometimes you want to test a mitigation on your own machine without sending a
PR or publishing a package. Two options:

### One-off — CLI override

For "is this even worth pursuing" experiments:

```bash
aorta run --workload fsdp --env HSA_ENABLE_SDMA=0
```

No registry involvement. The flag isn't named, isn't tracked, doesn't appear
in `aorta mitigations list`. Doesn't work for triage sweeps (which need
named entries).

### Repeated local use — throwaway plugin package

Make a tiny folder anywhere:

```
my-local-mits/
├── pyproject.toml
└── my_mits.py
```

`pyproject.toml`:

```toml
[project]
name = "my-local-mits"
version = "0.1"

[project.entry-points."aorta.mitigations"]
no_sdma = "my_mits:NO_SDMA"
```

`my_mits.py`:

```python
NO_SDMA = {"HSA_ENABLE_SDMA": "0"}
```

Then once:

```bash
pip install -e my-local-mits/
```

After that, your mitigations work everywhere — `--mitigation` flags, triage
sweeps, the lot. Nothing leaves your machine. To remove: `pip uninstall
my-local-mits`.

This is just Path 2 with a throwaway local package. Same workflow applies to
workloads via the `aorta.workloads` entry-point group.

## Adding an environment

Same two paths, with `aorta.environments` as the entry-point group. Plugin
payloads must use only the keys `docker`, `venv`, or `buck_target`; anything
else (e.g. `rocm`) raises `RegistryError` at load time. ROCm version is
implicit in the docker image digest, the host the venv runs on, or the Buck
checkout's captured revision — capture it from `aorta env probe` at runtime,
not via static declaration.

### Tier hints: how the workload consumes the resolved environment

The dispatcher threads the resolved `Environment` dataclass into each
workload's config under the reserved key `_aorta_environment`:

```python
config["_aorta_environment"] = {
    "name": "...",
    "docker": "...",         # or None
    "venv": "...",           # or None
    "buck_target": "...",    # or None
    "source_package": "...",
}
```

Workloads that can isolate themselves read this dict and branch their
subprocess invocation accordingly. The platform itself launches no docker
images, activates no venvs, and invokes no `buck2 run` — it threads the
metadata; the wrapper decides. Today's `recom_repro` wrapper consumes
`docker`; a Buck-aware wrapper consumes `buck_target` with the same
pattern:

```python
# Inside a workload's run() method:
env = self.config.get("_aorta_environment") or {}
buck_target = env.get("buck_target")
image = env.get("docker")

if buck_target:
    argv = ["buck2", "run", buck_target, "--", *script_args]
elif image:
    argv = ["docker", "run", ..., image, "python", ...]
else:
    argv = [sys.executable, str(entry)]
```

Adding a fourth tier later (e.g. Bazel) follows the same pattern: extend
`Environment`, extend `_VALID_ENV_KEYS`, document the read pattern here.
The dispatcher round-trips the field for free (`asdict(env_descriptor)`).

## Collisions

If two contributors register the same name (built-in vs plugin, or plugin vs
plugin), `load_mitigations()` / `load_environments()` raises
`RegistryCollisionError` naming both packages. There is no winner — the human
resolves it by renaming or removing one of the entries. This is intentional:
silent overrides cause hours of "why does my mitigation behave wrong"
debugging.

## Verifying what's registered

```bash
aorta mitigations list
aorta environments list
```

Each shows every registered entry plus its source package — useful when
debugging "did my plugin actually load?".

## Hard rule: no logic in registries

These modules contain only **data + lookup**. No environment manipulation, no
docker invocation, no validation of whether `DISABLE_TF32=1` is a "good"
value. Logic that consumes the registry data lives in the dispatchers (the
mitigation harness, the workload runtime). Mixing the two would make the
registry untestable in isolation and impossible to mock.
