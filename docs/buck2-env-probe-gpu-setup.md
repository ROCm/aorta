# Buck2 env-probe setup on a Linux GPU host

This guide starts from a machine with no Buck2 installation. It has two
separate goals:

1. Run a local Buck2-built AORTA probe.
2. Measure Buck2 remote-execution behavior in an existing RE-enabled
   repository.

The first goal is self-contained. It does **not** answer whether a remote
worker exposes a native marker or which execution platform a real workload
resolved to. A local Buck2 installation does not create an RE backend, CAS,
credentials, worker image, or execution-platform registration.

## 1. Machine prerequisites

Confirm the host is Linux and that the GPU is visible:

```bash
uname -a
python3 --version
test -e /dev/kfd && echo "/dev/kfd: present" || echo "/dev/kfd: missing"
command -v rocminfo || true
command -v rocm_agent_enumerator || true
```

A representative GPU snapshot needs `/dev/kfd` and a working ROCm
installation. Buck2 itself does not require a GPU.

## 2. Install Buck2

Download the release binary matching the host architecture from the official
[Buck2 releases](https://github.com/facebook/buck2/releases). Keep the
downloaded version visible in the test report; do not silently substitute a
different repository-provided binary.

If login and compute nodes use different home directories, choose a filesystem
visible to both and set it as `SHARED_ROOT`:

```bash
SHARED_ROOT="/shared/aorta-support"
mkdir -p "$SHARED_ROOT/bin"
install -m 0755 /path/to/downloaded/buck2 "$SHARED_ROOT/bin/buck2"
export PATH="$SHARED_ROOT/bin:$PATH"

buck2 --version
```

Use the same shared root for the base clone, worktree, Buck binary, and result
artifacts that must be visible on both host and compute nodes.

## 3. Fetch the branch into a dedicated worktree

```bash
AORTA_BASE="$SHARED_ROOT/aorta"
AORTA_WORKTREE="$SHARED_ROOT/aorta-worktrees/buck-invocation-context"
AORTA_BRANCH="<branch-provided-by-aorta-support>"

mkdir -p "$SHARED_ROOT/aorta-worktrees"
if [ ! -d "$AORTA_BASE/.git" ]; then
  git clone https://github.com/ROCm/aorta.git "$AORTA_BASE"
fi

git -C "$AORTA_BASE" fetch origin "$AORTA_BRANCH"
if [ -e "$AORTA_WORKTREE" ]; then
  echo "STOP: worktree path already exists: $AORTA_WORKTREE"
  exit 1
fi
git -C "$AORTA_BASE" worktree add --track \
  -b aorta-env-probe-customer-test \
  "$AORTA_WORKTREE" \
  "origin/$AORTA_BRANCH"

cd "$AORTA_WORKTREE"
```

The public repository already contains `.buckroot`, `.buckconfig`, a bundled
prelude declaration, toolchains, and the `//:aorta`/`//:aorta_lib` targets.
Do not run `buck2 init` over this checkout. Keep the base clone unchanged; all
commands and edits belong in the worktree.

## 4. Local Buck2 smoke test

```bash
cd "$AORTA_WORKTREE"

buck2 root
buck2 build //:aorta --show-output
buck2 run //:aorta -- --help

buck2 run //:aorta -- env probe \
  --execution-context buck2_run \
  --buck-target //:aorta_lib \
  --buck-default-context \
  -o "$PWD/env.local-buck.json"

python3 -m json.tool "$PWD/env.local-buck.json" >/dev/null
```

Check these fields:

```bash
python3 - "$PWD/env.local-buck.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as fh:
    doc = json.load(fh)

print("schema_version:", doc.get("schema_version"))
print("build_system:", doc.get("build_system"))
print("buck_invocation:", doc.get("buck_invocation"))
print("execution_context:", doc.get("execution_context"))
print("probe_namespace:", doc.get("probe_namespace"))
print("partial_reasons:", doc.get("partial_reasons"))
PY
```

This proves that AORTA builds and that target introspection runs under the
explicitly confirmed default Buck context. It does not prove remote execution.

## 5. Reproduce a real workload invocation

Run the probe from the real Buck repository root. Copy every configuration
input that precedes the workload target:

```bash
aorta env probe \
  --buck-target //example/package:training \
  --buck-mode-file @mode/opt \
  --buck-mode-file @mode/inplace \
  --buck-config example.gpu_backend=amd \
  --buck-modifier rocm \
  -o env.host.json
```

Mode files, config overrides, and modifiers are passed to `cquery` as atomic
arguments. Never paste the complete workload command into one string.

If torch exists only as a Buck target, define a `python_binary` that depends
on both AORTA's `aorta_lib` and the repository's torch target, then run the
same `env probe` arguments through that binary. A host `PYTHONPATH` is not a
declared remote-action input and must not be used as an RE workaround.

## 6. Remote-execution prerequisite

The Q1/Q2 measurement requires an existing repository with:

- configured execution platforms and remote workers;
- CAS/RE credentials;
- permission to use both `--local-only` and `--remote-only`;
- a repository-native action that writes its environment to a declared output;
- `buck2 log what-ran` access to prove where the action executed;
- a representative workload target;
- an AORTA probe target declared as a Buck dependency.

If `--remote-only` cannot be proven from `what-ran`, report Q1 as **untested**.
Do not use `--prefer-remote` or a cache hit as remote-execution evidence.

## Copy-paste prompt for an agent on the GPU machine

Replace every angle-bracket placeholder before sending this prompt:

```text
You are working on an authorized Linux GPU machine in a real Buck2 checkout.
Help me set up and run AORTA's Buck-aware env probe. Do not guess remote-
execution semantics and do not expose credentials, hostnames, internal target
labels, environment values, or private paths in your final response.

Inputs:
- Shared filesystem root: <SHARED_ROOT>
- AORTA base clone: <SHARED_ROOT>/aorta
- AORTA worktree: <SHARED_ROOT>/aorta-worktrees/buck-invocation-context
- AORTA branch: <AORTA_BRANCH>
- Buck repository root: <REPO_ROOT>
- Buck2 binary (or "not installed"): <BUCK2>
- Workload target: <WORKLOAD_TARGET>
- Repository-native env-dump target: <ENVDUMP_TARGET>
- Buck-declared AORTA probe target: <AORTA_PROBE_TARGET>
- Ordered mode files: <MODE_FILES_OR_NONE>
- Ordered -c overrides: <CONFIG_OVERRIDES_OR_NONE>
- Ordered -m modifiers: <MODIFIERS_OR_NONE>
- Private output directory: <PRIVATE_OUT_DIR>

Tasks:
1. Read the repository instructions and make no source edits until you have
   reported the setup state.
2. Check Linux, Python, /dev/kfd, ROCm visibility, Buck2 version, `buck2 root`,
   configured execution platforms, and whether the supplied targets exist.
3. If Buck2 is absent, install an approved official release binary at
   `<SHARED_ROOT>/bin/buck2`, prepend `<SHARED_ROOT>/bin` to PATH, and report
   its exact version. Do not run `buck2 init` inside an existing configured
   repository.
4. Fetch the AORTA branch into the dedicated worktree under
   `<SHARED_ROOT>/aorta-worktrees` exactly as documented above. Do not switch
   the base clone or overwrite an existing worktree. Then run the local
   `//:aorta` smoke commands from that worktree.
5. Build the exact AORTA probe argv using repeatable --buck-mode-file,
   --buck-config, and --buck-modifier options. Preserve order and do not use
   shell=True, eval, or one opaque command string.
6. Capture a host-side env.json and an in-action env.json. The in-action probe
   must be a declared Buck dependency; do not inject a host PYTHONPATH.
7. If and only if strict local and remote execution are available, run
   scripts/buck2_execution_context_probe.sh with cache-busted actions. Prove
   LocalExecute versus remote execution using `buck2 log what-ran`.
8. Query the real workload target with `buck2 audit
   execution-platform-resolution` and cquery attributes
   `buck.execution_platform` and `buck.target_configuration`, carrying the same
   mode/config/modifier context.
9. Compare schema_version, build_system, buck_invocation, execution_context,
   probe_namespace, ROCm/PyTorch/library identities, and partial_reasons.
10. Keep all raw env/config/log artifacts in <PRIVATE_OUT_DIR>. Return only:
    command success/failure, Buck2/AORTA versions, schema version, status
    enums, redacted config keys, fingerprints, candidate marker NAMES, and
    whether executor placement was proven.

Stop conditions:
- If remote-only execution is unavailable or `what-ran` does not prove remote
  execution, mark Q1 untested.
- If measured output does not name a resolved execution platform, leave
  likely_execution_platform unset.
- Do not invent an AORTA_RE_IMAGE convention or infer a platform from candidate
  lists.
```
