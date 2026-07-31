# AORTA Env Probe for Buck2 Workloads

Canonical source: `docs/buck2-env-probe.md` in the selected AORTA revision.
Refresh exported copies from this file rather than editing two copies.

This guide is only for workloads built or launched with Buck2. The probe
records the Buck dependency selection and the software/hardware environment
seen by the packaged Python workload. It launches no GPU kernels and runs no
training steps. On a multi-GPU host, collection can take tens of seconds.

## What to send

Send two files:

1. `env.buck-client.json` — the dependencies Buck selected for the target and
   settings you used.
2. `env.workload.rank<RANK>.json` — the Python, torch, ROCm, and GPU
   environment seen by one node-local rank 0 process.

These files answer different questions. A host-side Buck query cannot prove
what a workload process saw, and a workload process may not be able to run a
nested Buck query.

## Before you start

Use one pinned AORTA checkout, branch, release, or source archive for both
captures. AORTA requires Python 3.10 or newer. For a source checkout:

```bash
python -m pip install -e <AORTA_CHECKOUT>
```

If torch imports from a normal Python environment, verify that first:

```bash
python -c "import torch; print(torch.__version__, torch.version.hip)"
```

If torch exists only as a Buck target, do not inject AORTA into an unrelated
Python process with `PYTHONPATH`; that process will not contain the workload's
torch target.

## Buck workload — file 1: Buck dependency information

Run this from the root of the real Buck checkout. Use the workload target and
copy every mode file, `-c` override, and `-m` modifier from the workload's
actual invocation.

### When the workload uses Buck's default context

```bash
aorta env probe \
  --execution-context direct \
  --buck-target <WORKLOAD_TARGET> \
  --buck-default-context \
  -o env.buck-client.json
```

### When the workload has explicit configuration

```bash
aorta env probe \
  --execution-context direct \
  --buck-target <WORKLOAD_TARGET> \
  --buck-option mode=<MODE_FILE_1> \
  --buck-option config=<SECTION.KEY=VALUE> \
  --buck-option mode=<MODE_FILE_2> \
  --buck-option modifier=<MODIFIER> \
  -o env.buck-client.json
```

Repeat `--buck-option` in the exact order used by the workload command. The
three accepted types are `mode=...`, `config=KEY=VALUE`, and `modifier=...`.
Do not paste the complete Buck command into one quoted string.

This file should report:

- `build_system.kind: "buck2"`
- `buck_invocation.status: "success"`
- `buck_invocation.context_source: "default_confirmed"` or `"explicit"`
- a non-null `buck_invocation.context_fingerprint`
- a non-empty `library_introspection` for a target that depends on recognized
  PyTorch/ROCm/RCCL/hipBLASLt targets

Only config key names are written to the JSON. Config values influence the
fingerprint but are not stored.

## Buck workload — file 2: workload process

A Buck `.par` is the packaged Python executable for the application. It owns
the application's startup code, so the AORTA CLI cannot wrap it.

First, ask the repository owner to expose the provided AORTA revision as a
Buck cell or vendored source target. Confirm this command lists `aorta_lib`:

```bash
buck2 targets <AORTA_CELL>//:
```

If that target is unavailable, stop and contact support. Do not substitute a
different AORTA revision.

Add only AORTA's library to the existing application rule. Keep its current
rule macro, name, `main`/`main_function`, torch target, and all other
dependencies unchanged:

```python
# Inside the existing python_binary(...) or repository-specific Python macro:
deps = [
    # Keep every existing dependency.
    "<AORTA_CELL>//:aorta_lib",
]
```

At the beginning of the application's `main()`, after launcher environment
variables are set but before model, optimizer, or pipeline construction:

```python
import os

from aorta.instrumentation.environment import capture_to


def main():
    output = os.environ.get("AORTA_ENV_OUTPUT")
    if output:
        # One local-rank-0 process writes per node. {rank} makes filenames
        # unique across nodes; on a single process it resolves to rank 0.
        rank = os.environ.get("RANK", "0")
        if os.environ.get("LOCAL_RANK", "0") == "0":
            capture_to(
                output.format(rank=rank),
                probe_invocation=os.environ.get(
                    "AORTA_PROBE_INVOCATION",
                    "buck2_run",
                ),
            )
        return

    # Existing application setup follows.
```

### Local workload process launched by `buck2 run`

```bash
export AORTA_ENV_OUTPUT='<SHARED_OUTPUT_DIR>/env.workload.rank{rank}.json'
export AORTA_PROBE_INVOCATION=buck2_run

buck2 run <APPLICATION_WITH_AORTA_PROBE> -- <EXISTING_APPLICATION_ARGS>
```

Treat this as a capture-only run. Importing packages and running diagnostic
commands changes process startup timing; do not use the same run to measure a
timing-sensitive failure rate. Unset `AORTA_ENV_OUTPUT` for normal workload
runs. On a single node, validate `env.workload.rank0.json`. On a multi-node
job, validate and send one file per node-local rank 0.

`buck2 run` builds the target and then launches its `RunInfo` command. Its
build actions may run locally, remotely, or from cache, but this workload
capture does not by itself prove it ran inside a remote build/test action.

### Remote build/test action

To capture a remote worker, the repository owner must run the same
capture-enabled binary as a declared build/test action, pass a declared output
path as `AORTA_ENV_OUTPUT`, and set:

```bash
export AORTA_PROBE_INVOCATION=buck2_action
```

Do not use `buck2_action` for a normal `buck2 run`. Executor placement must be
confirmed separately using the repository's Buck execution evidence.

Do not pass `buck_target` to `capture_to()` inside the workload. The
client-side file already records the dependency information, while the Buck
command and checkout are often unavailable inside an action.

If the application source cannot be changed, ask the repository owner for a
temporary capture-only entrypoint with the same AORTA, torch, and application
dependencies. It should call `capture_to()` and exit without starting training.

## Validate both files before sending

Run the validator included with the same AORTA checkout:

```bash
python3 <AORTA_CHECKOUT>/scripts/validate_buck_env_pair.py \
  --require-library rccl \
  --require-library hipblaslt \
  env.buck-client.json \
  env.workload.rank0.json
```

The validator always requires the PyTorch Buck identity. Keep only the
additional `--require-library` lines for libraries the workload is expected to
use. For a multi-node capture, rerun the validator for every workload file.

A file labeled `buck2_action` must contain detected isolation evidence. If
your repository proves placement through separate `what-ran` evidence, an
AORTA maintainer may explicitly add `--allow-unisolated-action`; do not use
that override merely to silence an error.

Do not use the files until the validator prints `PASS`. If it fails, give the
error lines to the AORTA maintainer.

## What to send

Send:

1. `env.buck-client.json`
2. Each `env.workload.rank<RANK>.json` produced by a node-local rank 0.
3. The exact workload target.
4. The ordered mode-file and modifier names.
5. Config key names, `buck_invocation.option_order`, and
   `buck_invocation.context_fingerprint`.

If config values are sensitive, do not send them. The fingerprint lets two
captures be compared without recording those values.

## Expected limitations

- `execution_context.likely_execution_platform` may be `null`. AORTA does not
  guess the selected remote platform.
- `probe_namespace` inequality is useful evidence; equality does not prove two
  processes shared a namespace.
- `system_health` may be unavailable when RDHC is not installed.
- Optional Python packages may remain `null` when the workload does not use
  them.

## Troubleshooting

### `pytorch_version` is null

The probe did not run in the workload's Python process. Confirm the Buck
`python_binary` depends on the real torch target and that `capture_to()` runs
inside its `main()`.

### `buck_invocation.context_source` is `unspecified`

Rerun the client probe with either `--buck-default-context` or the exact
mode/config/modifier options.

### `library_introspection` is empty

Confirm `--buck-target` names the real top-level workload target and that all
configuration options match its normal invocation. If the graph uses library
labels AORTA does not yet recognize, give the AORTA maintainer a scrubbed list
of the relevant label names.

### Warning says this is a client-host snapshot

That is expected for a local `buck2 run`. Keep it as the client/runtime
snapshot and do not claim it represents a remote worker.

## Buck2 references

- [Python binary dependencies and entrypoints](https://buck2.build/docs/prelude/rules/python/python_binary/)
- [Command-line config and flag-file precedence](https://buck2.build/docs/concepts/buckconfig/)
- [`buck2 run` and execution policy flags](https://buck2.build/docs/users/commands/run/)
- [Configured queries and exact attributes](https://buck2.build/docs/users/commands/cquery/)
