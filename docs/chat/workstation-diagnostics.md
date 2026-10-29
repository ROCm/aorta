# Run CIA diagnostics on one workstation

AORTA Chat can run the Launch → Watch → Autopsy pipeline on the same Linux
workstation as the chat server. Slurm is optional: when `sbatch` is not
installed, the default `auto` backend starts a detached local process instead.

## What the workstation needs

- A ROCm-supported AMD GPU and a working ROCm installation.
- The AORTA chat and CIA extras in one Python environment.
- Any sanitizer backend required by the diagnostic you want to run. WaitCheck
  is static; ConSan requires ROCjitsu and its preload library.
- An LLM endpoint configured for Chat and CIA. That endpoint may be remote. If
  you also host the model with vLLM on this workstation, leave enough GPU
  memory for the workload being diagnosed.

You do **not** need Slurm, SSH access to another machine, or a shared
filesystem.

## Install and configure

From a source checkout:

```bash
cd /path/to/aorta
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[chat-ui,cia]'
```

Add the following to `~/.config/aorta/chat.toml`:

```toml
allow_cluster_jobs = true
cia_job_backend = "local"
gpu_arch = "gfx950"  # change this to your GPU architecture
```

The setting retains its historical `allow_cluster_jobs` name, but it gates
both local and Slurm diagnostic execution. It remains off by default because a
diagnostic tool runs pasted code with the permissions of the account serving
Chat.

Configure the model provider in the same file. For ConSan, also set
`rocjitsu_build` and `rocjitsu_preload`; see
[configuration](configuration.md#the-cluster-diagnostic-tools).

## Verify and start

```bash
source /path/to/aorta/.venv/bin/activate
aorta chat doctor
aorta chat tools
aorta chat ui
```

`aorta chat tools` should include `triage_kernel_source`,
`triage_assembly_source`, and `triage_workload`. Diagnostic job records,
combined logs, and bundles are written under `~/cia-jobs` unless `jobs_path`
is configured.

## Backend selection

`cia_job_backend` accepts:

- `auto` (default): use Slurm when `sbatch` is on `PATH`; otherwise run locally.
- `local`: always run on this workstation. Use this when Slurm client commands
  happen to be installed but no usable cluster is configured.
- `slurm`: require Slurm and fail clearly if `sbatch` cannot submit the job.

The same value can be set for one session:

```bash
export CIA_JOB_BACKEND=local
aorta chat ui
```

A local launch creates its own process group, writes stdout and stderr to the
normal CIA job log, and records its exit status beside `job.json`. Cancelling
the Chat turn terminates that process group. The persisted process identity
includes more than a PID, so a later cleanup cannot accidentally signal an
unrelated process after Linux reuses the number.

Launch, Watch, and Autopsy all work in local mode. If Autopsy recommends an
additional production matrix sweep, AORTA leaves that recommendation in the
report instead of using the cluster-only SSH escalation path.

## Troubleshooting

- **A local run says it cannot pin a node:** remove `cia_demo_node`. Node names
  apply to Slurm; local execution intentionally refuses to pretend it ran on a
  requested remote machine.
- **`hipcc` is missing:** install ROCm or add its binaries to `PATH` before
  starting Chat.
- **A sanitizer says `DID NOT RUN`:** configure that sanitizer's backend. This
  is not a clean result.
- **The GPU is out of memory:** a local vLLM server and the diagnostic workload
  share the same GPU. Move inference to another endpoint or reduce its GPU
  allocation.
