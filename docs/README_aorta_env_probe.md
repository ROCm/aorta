# AORTA Environment Probe — Customer Instructions

Source of truth: `docs/README_aorta_env_probe.md` in the AORTA revision
provided by your support contact. If this file is exported separately, update
it from that source rather than editing two copies.

Use this probe to record the software and hardware environment around a GPU
workload. It launches no GPU kernels and runs no training steps. It reads
version files, imports Python packages when available, and runs bounded
diagnostic commands. On a multi-GPU host, collection can take tens of seconds.

## What to send

- Workload not built with Buck2: send one `env.host.json`.
- Workload built with Buck2: send two files:
  1. `env.buck-client.json` — the dependencies Buck selected for the target
     and settings you used.
  2. `env.workload.json` — the Python, torch, ROCm, and GPU environment seen by
     the workload process.

These files answer different questions. A host-side Buck query cannot prove
what a workload process saw, and a workload process may not be able to run a
nested Buck query.

## Before you start

Use the exact AORTA checkout, branch, release, or source archive provided by
your support contact. AORTA requires Python 3.10 or newer. For a source
checkout:

```bash
python -m pip install -e <AORTA_CHECKOUT>
```

If torch imports from a normal Python environment, verify that first:

```bash
python -c "import torch; print(torch.__version__, torch.version.hip)"
```

If torch exists only as a Buck target, use the Buck instructions below. Do not
inject AORTA into an unrelated Python process with `PYTHONPATH`; that process
will not contain the workload's torch target.

## Non-Buck workload

Activate the same environment and container used by the workload, then run:

```bash
aorta env probe \
  --execution-context direct \
  -o env.host.json
```

Send `env.host.json`. Missing optional tools appear in `partial_reasons`; they
do not prevent the rest of the snapshot from being useful.

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
the application's startup code, so the AORTA CLI cannot wrap it. Add AORTA's
`aorta_lib` to the existing `python_binary` dependencies. Keep its current
application entrypoint, torch target, and other dependencies unchanged:

```python
python_binary(
    name = "application_with_aorta_probe",
    main_function = "<EXISTING_APPLICATION_MAIN_FUNCTION>",
    deps = [
        "<AORTA_CELL>//:aorta_lib",
        # Keep the existing torch and application dependencies here.
    ],
)
```

At the beginning of the application's `main()`, after launcher environment
variables are set but before model, optimizer, or pipeline construction:

```python
import os

from aorta.instrumentation.environment import capture_to


def main():
    output = os.environ.get("AORTA_ENV_OUTPUT")
    if output:
        # Rank 0 writes the shared artifact. Every rank exits after capture so
        # one delayed rank cannot disturb a timing-sensitive distributed run.
        if os.environ.get("RANK", "0") == "0":
            capture_to(
                output,
                probe_invocation=os.environ.get(
                    "AORTA_PROBE_INVOCATION",
                    "buck2_run",
                ),
            )
        return

    # Existing application setup follows.
```

Launch the application with:

```bash
export AORTA_ENV_OUTPUT=<SHARED_OUTPUT_DIR>/env.workload.json
export AORTA_PROBE_INVOCATION=buck2_run

buck2 run <APPLICATION_WITH_AORTA_PROBE> -- <EXISTING_APPLICATION_ARGS>
```

Treat this as a capture-only run. Importing packages and running diagnostic
commands changes process startup timing; do not use the same run to measure a
timing-sensitive failure rate. Unset `AORTA_ENV_OUTPUT` for normal workload
runs.

Use `buck2_run` when `buck2 run` launches the `.par` on the client host. Use
`buck2_action` only when the probe itself runs as a declared build/test action
and executor placement is independently known. Do not label a normal
`buck2 run` process as a remote action.

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
  env.buck-client.json \
  env.workload.json
```

The validator requires the PyTorch Buck identity by default. If the workload
also requires specific libraries, add repeatable checks such as
`--require-library rccl` or `--require-library hipblaslt`.

Do not send the files until the validator prints `PASS`. If it fails, send the
error lines to your support contact.

## What to send

Send:

1. `env.buck-client.json`
2. `env.workload.json`
3. The exact workload target.
4. The ordered mode-file and modifier names.
5. Config key names and `buck_invocation.context_fingerprint`.

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
labels AORTA does not yet recognize, send a scrubbed list of the relevant label
names to your support contact.

### Warning says this is a client-host snapshot

That is expected for a local `buck2 run`. Keep it as the client/runtime
snapshot and do not claim it represents a remote worker.
