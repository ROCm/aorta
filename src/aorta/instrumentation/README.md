# aorta.instrumentation

Platform-level introspection for AORTA. The MVP is the env probe (issue
#147); future submodules (NaN/numerics detector, drift watcher, etc.)
will live here too.

## What's here

| Submodule | Purpose | Public API |
| --- | --- | --- |
| [`environment`](environment.py) | Snapshot the trial environment (rdhc + ROCm version files + hipconfig + hipBLASLt + container detection + canonical env vars + Python/PyTorch versions). | `collect_env() -> EnvSnapshot` |

## env probe quick reference

### Library API (the primary deliverable; B1 / B2 call this in-process)

```python
from aorta.instrumentation.environment import collect_env, EnvSnapshot

snapshot: EnvSnapshot = collect_env()   # NEVER raises

# Embed in a trial result and serialise
trial_result["env"] = snapshot.to_dict()

# Reconstruct from a previously persisted env.json blob
rebuilt = EnvSnapshot.from_dict(loaded_json["env"])

# Quick human summary (six lines, no external dependencies)
print(snapshot.summary())
```

`EnvSnapshot` is a `@dataclass(frozen=True)` mirroring the env.json
schema 1-to-1. `frozen=True` prevents attribute rebinding on the
snapshot itself (`snap.rocm = ...` raises), but does NOT deep-freeze
the nested `dict` / `list` containers -- treat embedded snapshots as
read-only and `deepcopy(snap.to_dict())` before mutating if you need
to.

### CLI (thin wrapper over `collect_env()`)

```bash
# Default path: ./env.json
aorta env probe

# Custom path; parent dirs are created if missing
aorta env probe -o runs/exp1/env.json
```

After the run, `cat env.json` reveals the same dict that
`snapshot.to_dict()` returns. The CLI also prints a six-line summary to
stdout.

### Installing RDHC for full `system_health` coverage

`rdhc` (ROCm Deployment Health Check) is a system tool from the
[`rocm-systems`](https://github.com/ROCm/rocm-systems/tree/main/projects/rocm-core/rdhc)
repo, NOT a Python package. Aorta wraps it but does not vendor it -- if
absent, the env probe still produces a complete snapshot with
`system_health: null` and a `partial_reasons` entry pointing at the
install path.

It is not in `requirements.txt` because:

* `rdhc` is not on PyPI.
* Hard-pinning would break aorta on non-ROCm hosts, stripped ROCm
  docker images, and CPU-only CI runners.
* The fail-soft contract is the design: every snapshot is honest about
  what it could and could not capture, the run continues either way.

For the install commands (Ubuntu / RHEL / SLES / source), the
passwordless-sudo recipe, and verification steps, see the user-facing
guide: [`docs/env-probe.md`](../../../docs/env-probe.md#installing-rdhc).

### Fail-soft contract (`partial` / `partial_reasons`)

`collect_env()` is documented as **never raises**. Two layers enforce
that:

1. Every probe is individually fail-soft. If a probe falls back to
   `None` (rdhc not installed, /opt/rocm absent, hipconfig missing,
   torch absent, ...), it appends a human-readable line to a shared
   `partial_reasons: list[str]`. The snapshot's top-level `partial: bool`
   is then `True`.
2. The orchestrator body is wrapped in a top-level `try / except
   Exception`. If anything truly unexpected raises (a stdlib call
   misbehaves, a future probe is buggy, ...), the disaster-recovery
   helper `_disaster_snapshot` constructs a fully-shaped `EnvSnapshot`
   from defaults with the exception captured in `partial_reasons`.
   Even the helper guards its own `_utc_now_iso` and
   `platform.python_version` calls.

Documented absences DO NOT trigger `partial`:

* `docker == None` on baremetal (no container, nothing to record).
* `env_vars[X] == None` for an unset env var (the documented contract).
* `runtime_context.venv_path == None` outside a venv.

### env.json schema (v1.0)

See `EnvSnapshot` in [`environment.py`](environment.py) for the
authoritative shape and field-by-field docstrings. Top-level keys:

```
schema_version    captured_at       partial         partial_reasons
system_health     rocm              hip             hipblaslt
runtime_context   docker            env_vars
python_version    pytorch_version
```

Schema is **stable + versioned**. Add fields freely; never rename or
remove without bumping `schema_version`. Empty `applied_prs: {}` may
gain `pr_<id>_applied` keys later -- that is additive and does not bump
the version.

## How to add a new env var to `CANONICAL_ENV_VARS`

The env var list in `environment.py` is **explicit, not prefix
matching**. Adding a new variable is a deliberate three-step change so
that what we capture stays auditable and reviewable.

1. **Edit `CANONICAL_ENV_VARS`** in `src/aorta/instrumentation/environment.py`.
   Group it with related vars under the matching `# HSA / runtime` /
   `# GPU queue / codegen` / `# RCCL` / `# FBGEMM` / `# PyTorch / inductor`
   comment. If it doesn't fit any group, add a new group with a short
   header comment naming it.

2. **Update the stability-guard test**:
   `tests/instrumentation/test_environment.py::TestEnvVars::test_canonical_var_names_stable`
   This is a literal `assert set(...) == {...}` over the canonical
   list. Without updating it, your PR fails. The test exists to force
   you to acknowledge that adding a variable is a schema-shape change
   reviewers care about.

3. **Document the addition** in your PR description: which workload /
   library uses it, why it materially affects trial results, and a
   pointer to the upstream documentation. The point of `CANONICAL_ENV_VARS`
   is to capture variables that *change behaviour observably* -- if it
   doesn't, it doesn't belong here.

What does **not** belong here:

* **Workload config** like `AMP_DTYPE`, `MODEL_DTYPE`,
  `SHAMPOO_PRECONDITIONER_DTYPE`, model precision, optimizer
  hyperparameters. Those are training-script arguments
  (Hydra/argparse), captured by `aorta run` in the trial result
  (Task B1). Some workloads forward them as env vars to subprocesses --
  the env probe still does NOT capture them, since they are workload
  state, not environment state.
* **Anything you can capture by reading a file or running a tool**
  (rocm version, hipconfig output, etc.). Those go in their own block.

## How to add a new probe block

Roughly the same pattern as the existing blocks (`_run_rdhc`,
`_capture_rocm_version_files`, `_capture_hip_toolchain`, ...):

1. Define probe constants (filesystem paths, command names) at module
   scope, paired with provenance comments. Add to the
   `TestPathConstants` parametrize list and the
   `test_known_constant_set_is_stable` set.
2. Write the probe as `_capture_<block>(reasons: list[str]) -> dict | None`.
   It MUST NOT raise; catch every error path and return `None` plus a
   `reasons.append(f"<block>.<field>: <reason>")` line.
3. Add a field to `EnvSnapshot` and update `_disaster_snapshot` to
   give it a sane default. The
   `test_disaster_snapshot_populates_every_envsnapshot_field` test
   forces this -- it enumerates `dataclasses.fields(EnvSnapshot)` and
   fails if any are missing from the disaster path.
4. Wire into `collect_env()`'s body inside the `try` block.
5. Bump `SCHEMA_VERSION` if the addition isn't strictly additive
   (renaming / removing a field never bumps; adding always-present
   keys is additive).

## Tests

`tests/instrumentation/test_environment.py` has ~80 tests covering:

* Schema completeness (every top-level key present in every snapshot)
* Round-trip via `to_dict()` / `from_dict()` (incl. through JSON,
  forward-compat with extra keys)
* `collect_env()` never raises (probe-fail and unexpected-exception paths)
* Per-probe behaviour (RDHC happy + four failure modes, ROCm files,
  hipconfig, hipblaslt header parsing + lib hashing + tensile
  fingerprint, runtime context detection, Docker metadata, env vars,
  PyTorch version)
* B1/B2-style integration: snapshot embeds in fake trial result and
  round-trips through JSON
* Idempotency: two calls produce equivalent snapshots
* CLI thin-wrapper invariant (line count + no probing imports)
* No-GPU-compute guard via fake torch in `sys.modules`
* Disaster-snapshot completeness via `dataclasses.fields()`

Run them with:

```bash
pytest tests/instrumentation/test_environment.py -v
```

## See also

* User-facing how-to: [`docs/env-probe.md`](../../../docs/env-probe.md)
* Issue with the full schema + acceptance criteria:
  [#147](https://github.com/ROCm/aorta/issues/147)
