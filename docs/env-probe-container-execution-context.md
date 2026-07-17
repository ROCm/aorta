# env probe — container & Buck2 execution context (design note)

Status: partially implemented · Scope: `src/aorta/instrumentation/environment.py`, `src/aorta/cli/env.py`, `src/aorta/instrumentation/_probe_main.py`, `docs/env-probe.md`

## Implementation status

**Landed (phase 1 — container-context, no Buck2 dependency; schema 1.11):**

- `container_detected` (bool) — the generic-isolation smoke detector that
  fixes the RE/k8s-sandbox-as-`baremetal` false negative. Additive schema bump.
- `execution_context.probe_invocation` — a self-declared label
  (`direct` / `buck2_run` / `buck2_action`) stamped from the new
  `--execution-context` flag.
- `--execution-context` flag on both `aorta env probe` and
  `python -m aorta.instrumentation._probe_main` — labels the snapshot and
  **warns loudly to stderr** when a container/RE capture is claimed but
  `container_detected` is `false` and neither `$AORTA_RE_IMAGE` nor
  `$AORTA_DOCKER_IMAGE` is set.

Phase 1 captures **nothing new about a remote worker** — it only makes the
"probed the wrong place" failure visible. That is the whole point: it is a
guardrail, not a data source.

**Still REMAINING (phase 2 — Buck2 / remote-execution, needs MI350 empirics):**

- `execution_context.likely_execution_platform` — resolved RE platform label.
  **Blocked on Open Q2** (does `buck2 audit configurations`/cquery report the
  *selected* platform, or only candidates?). The key exists today and is
  always `null`.
- `$AORTA_RE_IMAGE` launcher convention — **blocked on Open Q1**: a native RE
  marker (`BUCK2_RE_*` / `RE_PLATFORM` / sandbox rootfs path) may already
  exist and be preferable; do not invent the convention until Q1 is answered.
  (The env var is already read by the `--execution-context` warning so the
  convention can be adopted without further code once Q1 is settled.)
- `probe_namespace` (mount-ns / cgroup hash diff key) — deferred; the phase-1
  `container_detected` boolean covers the immediate false-negative bug, and
  the namespace hash is only useful once two-probe RE diffing exists.
- The Buck2 `genrule`/`sh_test` rule wrapper — documentation-only per the
  external-tool policy; deferred until phase 2.
- Open Q1, Q2, Q3 below all remain open (Q1/Q2 need a real Buck2 RE cluster;
  Q3 is "where does the NaN-repro runbook live").

The **field classification, risks, and runbook** sections below are the
enduring rationale and apply to both phases.

## Problem

`aorta env probe` captures state from **the process that runs it**. There is no
host-vs-container distinction beyond a small `runtime_context` / `docker` block.
Nearly every field (ROCm, HIP, hipBLASLt/rocBLAS/MIOpen/RCCL, GPU arch, PyTorch
build, env vars) is read from whatever filesystem, `/proc`, and Python
interpreter the probe process sees — trustworthy **only if the probe ran inside
the same execution context as the workload**.

For plain Docker this mostly holds (the playbook is "run the probe inside the
container"). For **open-source Buck2 (Meta)** it breaks in a way today's schema
cannot see:

- Buck2 is **remote-execution first**. Per action, Buck2 decides to run it
  **locally** (invoking host, *unsandboxed* — no container at all) or
  **remotely** on an RE worker inside whatever image the chosen execution
  platform's `remote_execution_properties` specifies (BuildBarn / BuildBuddy /
  EngFlow).
- The local-vs-remote decision is **per action** and can **race both** (hybrid
  execution). There is no single "the build ran in container X" answer to
  capture.
- `aorta env probe --buck-target …` runs `buck2 cquery`, `buck2 root`,
  `buck2 --version`, `hg id -i` **on the invoking host**, not the RE worker. It
  returns configured-target-**graph labels**, not the worker's filesystem /
  driver / ROCm state.
- `_detect_container_type()` recognizes only `/.dockerenv`,
  `/run/.containerenv`, and `SINGULARITY_NAME`/cgroup tokens. An RE worker's
  bare `runc`/containerd sandbox has none of these and silently falls through to
  `"baremetal"` — actively misleading. **Phase 1's `container_detected`
  addresses exactly this false negative.**

**Bottom line:** "did I probe the container the workload ran in?" is not
answerable from the artifact today, because Buck2 has no fixed "the container" —
it has a per-action execution-platform decision the probe has no visibility into.

Sources: Buck2 Remote Execution docs (buck2.build/docs/users/remote_execution/),
Buck2 Architectural Model (buck2.build/docs/developers/architecture/buck2/),
Buck2 Glossary, Tweag "A Tour Around Buck2."

## Field classification

Source → classification → Buck2 caveat.

| Field(s) | Source | Class | Buck2 caveat |
|---|---|---|---|
| `rocm`, `hip`, `hipblaslt`, `rocblas`, `miopen`, `rccl`, `composable_kernel`, `tensile`, catalogs | `/opt/rocm` reads, `nm`/`c++filt`, `hipconfig` | **process** | If actions run remotely while the probe runs on the host, these describe the **wrong machine** — not host/container drift, an unrelated box. |
| `amdgpu_driver.kfd_device_present` / `.kfd_sysfs_present` / `.kmd_version` | `/dev/kfd`, `/sys/class/kfd`, `/sys/module/amdgpu/version` | **host-kernel** | Valid only if the probe runs on the **same physical/VM host** as the workload. An RE worker on another machine has its own driver. |
| `amdgpu_driver.package_*` / `.module_*` | `dpkg`/`rpm`, `modinfo` | **filesystem/process** | Same, one level down (see schema 1.10 scope narrowing). |
| `gpu_arch` | `rocm_agent_enumerator` | **host/device** | Needs `/dev/kfd` passthrough; a passthrough gap reads as "no GPU." |
| `env_vars` | `os.environ` of probe | **process** | Buck2 actions set env per-rule; the invoking shell's env is often not what the action saw. |
| `runtime_context.type` | `/.dockerenv`, `/run/.containerenv`, cgroup | **process, narrow** | RE sandbox → `"baremetal"` false negative. Use `container_detected` for the runtime-agnostic answer. |
| `container_detected` | named-runtime match, `/proc/self/ns/mnt` vs `/proc/1/ns/mnt`, container/k8s cgroup tokens | **process, runtime-agnostic** | `true` when *any* isolation signal fires even if the runtime can't be named. Still describes the probe's OWN process — it does not prove the probe ran where the workload ran. |
| `execution_context.probe_invocation` | `--execution-context` flag (self-declared) | **caller-asserted** | Honest record of the caller's claim, not an auto-detected fact (phase 1). |
| `runtime_context.python_env` / `.venv_path` / `.conda_env_name` | `sys.prefix`, `$CONDA_DEFAULT_ENV` | **process** | May not match the interpreter Buck2 packages into a `python_binary`. |
| `docker.image` / `.digest` | `$AORTA_DOCKER_IMAGE` / `$AORTA_DOCKER_DIGEST` | **externally asserted** | No Buck2 equivalent exists today (see `$AORTA_RE_IMAGE`, phase 2). |
| `docker.container_id` | `/proc/self/cgroup` | **process** | Same host-kernel dependency; irrelevant to a remote worker. |
| `build_system` (`kind`, `buck2_version`, `repo_root`, `revision`) | `buck2 --version`/`root`, `hg id`/`git rev-parse` on invoking host | **client-host** | Describes the developer's checkout, not the RE worker's toolchain. Two engineers get identical blocks while builds ran on different images. |
| `library_introspection` (`buck2 cquery 'deps(target)'`) | configured target graph (labels) | **ambiguous — graph-accurate, execution-silent** | Reliable for declared deps; not for what code actually executed remotely (hybrid/racing/RE-cache). |
| `pytorch_build.*` | `import torch`, `torch.__config__` | **process** | If the shell's torch ≠ the `python_binary`'s torch target, this silently describes the wrong build (the "torch is a Buck target" trap documented in `docs/env-probe.md`, schema 1.8 Buck/monorepo native-lib recovery). |
| `host` (`kernel_release`, `glibc_version`, `machine`) | `os.uname()`, `os.confstr()` | **host-kernel** | Same wrong-machine risk. |
| `nics` | `lspci`/`ethtool`/`ibv_devices`/`rdma link` | **host/device** | Misleading if probed off the collectives host. |
| `system_health` | `rdhc --quick --json` | **host** | Same. |

## Risks for Buck2

1. **False confidence from a clean snapshot.** Every field degrades gracefully,
   so a probe run on the wrong machine still produces a full, `partial:false`
   artifact. Nothing says "captured somewhere unrelated to where your job ran."
2. **`runtime_context.type` misclassifies RE sandboxes as `"baremetal"`** — reads
   as "no isolation," the opposite of the truth, worse than "unknown." *(Phase 1
   `container_detected` mitigates this.)*
3. **`buck2 cquery` answers "what's declared," not "what ran."** Deterministic
   for the label graph; blind to local/remote/RE-cache execution.
4. **`build_system.revision`/`buck2_version` describe the client, not the
   executor** — false reassurance that two probes "match."
5. **No signal at all for local-vs-remote execution** of a given action.
6. **The "torch is a Buck target" workaround already exists for exactly this
   gap** (documented in `docs/env-probe.md`, schema 1.8) — it generalizes to
   `env_vars`, `gpu_arch`, ROCm/HIP, and `amdgpu_driver` whenever an RE
   action's env differs from the invoking shell.

## Design — additive schema bump

New fields (same additive pattern as the amdgpu_driver 1.10 change). The
**Phase** column tracks what landed in phase 1 vs. what remains for phase 2
(see Implementation status above):

| Field | Type | Meaning | Phase |
|---|---|---|---|
| `container_detected` | `bool` | **Single boolean.** `true` on *any* isolation signal: a named-runtime match (`_detect_container_type() != "baremetal"`), a private mount namespace (`/proc/self/ns/mnt` != `/proc/1/ns/mnt`), or a container/k8s token in `/proc/self/cgroup` (`docker`/`containerd`/`kubepods`/`libpod`/`lxc`/`crio`). Fixes the RE-sandbox-as-baremetal false negative. No k8s-vs-containerd distinction. | **1 (done)** |
| `execution_context.probe_invocation` | `"direct" \| "buck2_run" \| "buck2_action"` | How the probe was launched. Phase 1: **self-declared** via `--execution-context` (defaults to `direct`). Phase 2 may auto-detect via native RE env vars (**see Open Q1**). | **1 (done, self-declared)** |
| `execution_context.likely_execution_platform` | `str \| null` | Best-effort: from `buck2 audit configurations` / cquery on the target, the resolved platform label. Advisory, not guaranteed (**Open Q2**). Key present today, always `null`. | 2 (remaining) |
| `probe_namespace` | `str \| null` | Hash of the mount namespace / cgroup path (e.g. `/proc/self/ns/mnt` inode). A coarse "same isolation boundary?" diff key, nothing more. | 2 (remaining) |
| `$AORTA_RE_IMAGE` convention | env var | Extend the existing `$AORTA_DOCKER_IMAGE` launcher pattern to Buck2: customers set this in their `remote_execution_properties` / action env (**pending Open Q1** — a native marker may exist and be preferable). Already read by the phase-1 warning. | 2 (remaining) |

### CLI — a labeling+validation flag, not a behavior change

```
aorta env probe --execution-context buck2_action
python -m aorta.instrumentation._probe_main --execution-context buck2_action /out/env.json
```

Does not change what is captured (all still process-derived). It:
1. stamps `execution_context.probe_invocation`, and
2. **warns loudly to stderr** (fail-soft, never a hard error) if
   `container_detected` is `false` **and** neither `$AORTA_RE_IMAGE` nor
   `$AORTA_DOCKER_IMAGE` is set — i.e. "you claimed a container/RE capture but I
   see zero isolation signal." This single check catches the core misdiagnosis.
   Implemented in **both** the Click CLI (`aorta env probe`) and the
   dependency-free `_probe_main` entry point, since the latter is the one most
   likely used inside a Buck2 action / container.

### Durable fix — a Buck2 rule wrapper (documented, not vendored)

A `genrule`/`sh_test` that runs `aorta env probe` **as part of the action**, so
it executes wherever Buck2 actually placed that action. This is the direct RE
analogue of the README's existing "put AORTA and torch in one `python_binary`"
pattern. Documented in `docs/`, not shipped as vendored Buck code (external-tool
policy). *(Phase 2.)*

## Recipes / runbook guidance

1. Never accept a bare host-shell probe as sufficient for a Buck2-driven job
   without first asking: **local or remote execution?**
2. If remote, the probe must run as a Buck2 action/genrule dependency of the
   failing target, not standalone.
3. Always capture **both** a host-side probe and an in-action probe; diff
   `runtime_context`, `container_detected`, and `amdgpu_driver.kfd_device_present`
   as the first triage step.

### Proposed runbook wording (Residual NaN / repro)

> **⚠️ If your job runs under Buck2, a host-shell `aorta env probe` is not
> enough.** Buck2 may execute your build/test locally or on a remote executor,
> and the two can have completely different ROCm/driver/GPU environments. A
> host-shell probe only tells us about your shell — not what your job saw if it
> executed remotely.
>
> Before filing a repro, tell us: **did this job run locally or via remote
> execution?** If unsure, assume remote for anything routed through a shared
> Buck2 RE cluster.
>
> - **Local** Buck2 build (`--local-only`, or RE disabled in `.buckconfig`): a
>   host-shell probe is representative — `aorta env probe -o env.json`.
> - **Can run remotely:** add
>   `aorta env probe --execution-context buck2_action -o env_workload.json` as a
>   dependency/step of the failing target so it captures the RE worker's actual
>   environment; submit **that** file.
> - **When in doubt, submit both.** If they disagree on `runtime_context`,
>   `container_detected`, `amdgpu_driver`, or ROCm versions, that mismatch is
>   itself diagnostic — send it as-is.

## Open questions (empirically testable on MI350)

1. **Is there already a native "I'm on an RE worker" env var?** Test: a trivial
   `genrule`/`sh_test` that runs `env | sort > $OUT` both locally and
   forced-remote, then diff. If Buck2/the RE backend injects a stable marker
   (`BUCK2_RE_*`, `RE_PLATFORM`, a sandbox rootfs path), read *that* instead of
   inventing `$AORTA_RE_IMAGE`. If not, the customer-set convention is confirmed
   necessary.
2. **Can `buck2 audit configurations` / cquery name the *resolved* execution
   platform, not just the available ones?** Test on a real target. If it reports
   the selected platform, `likely_execution_platform` becomes confident; if only
   candidates, it stays advisory.
3. Where does the "Residual NaN repro" runbook actually live? Not found in the
   `aorta` repo — likely customer-facing docs, or a doc to be created under
   `docs/`.
