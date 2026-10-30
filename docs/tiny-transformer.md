# Tiny transformer

`tiny_transformer` trains a small transformer and records whether the run passed or failed. It uses aorta's existing training loop. You choose how the processes start by setting `launcher` inside the recipe:

| `launcher` | What it does | When to use it |
|---|---|---|
| `inprocess` (default) | Trains in the process that is running aorta. One process. | Check that the recipe runs. No Spur, no Monarch, and no GPU required. |
| `spur` | Starts the training processes with `spur srun`. | The same training job, launched by Spur. Start Spur yourself first. |
| `monarch` | Starts the training processes with Monarch. | The same training job, launched by Monarch, on a Slurm allocation or, with `scheduler: spur`, on a Spur allocation. |

Aorta does not install Spur or Monarch. Install them with [Install and run Spur](spur-install.md) and [Install and run Monarch](monarch-install.md). Continuous integration runs only the one-process recipe.

The model is a 4-layer transformer: hidden size 256, 4 attention heads, feed-forward size 512, vocabulary 256. A recipe can override any training field. A field you leave out keeps that default.

## How to tell that it passed

Read the result written by rank 0, the first process. The run passed when all of these are true:

- `passed` is `true`.
- `final_loss` is an ordinary number. `NaN` and infinity are failures.
- `world_size` is the number of processes that trained together.

The one-process recipe runs as a single process, so `world_size` is 1. It does not compare that count with a configured value.

The three 16-GPU recipes, for Spur, Monarch, and Monarch on Spur, run on 2 nodes with 8 GPUs each, which is 16 processes. They set `expected_world_size: 16`. The run fails if any other number of processes joins.

All four recipes also fail the run when:

- the loss, a gradient, or a model output is `NaN` or infinity
- the processes finish with different losses

The loss comparison is `checks.fail_on_rank_loss_mismatch`. It stays off unless a recipe sets it to `true`. All four recipes here set it to `true`. While it is on, every process trains on the same data, so the losses can be compared. The recipes use `float32` so that comparison stays stable.

## Which GPU a process uses

This does not decide pass or fail. It only picks the device.

If `CUDA_VISIBLE_DEVICES` names one GPU, that is the only GPU the process can see, so it uses device 0. If several GPUs are visible, or the variable is unset, the process uses `LOCAL_RANK` as the GPU index on its machine. Under Spur, `LOCAL_RANK` is that process's index on its node (`SLURM_LOCALID`). Two processes on the same machine then use two different GPUs, including when tasks are assigned to nodes round-robin. If `SLURM_LOCALID` is missing, the index falls back to the global rank modulo the number of GPUs per node.

## Run one process

Install aorta in the active environment, then run these commands from the repository root:

```bash
aorta triage run --recipe recipes/training/tiny-transformer.yaml --dry-run
aorta triage run --recipe recipes/training/tiny-transformer.yaml
```

`--dry-run` checks the recipe and does not train. The real run is one process, for 20 steps. `device: auto` uses a GPU when one is visible, and the CPU otherwise.

`aorta run --workload tiny_transformer` is available after you install this version of the package. That command does not read a recipe file. It uses the built-in defaults: one process, batch size 2, sequence length 32. It does not use the settings in `recipes/training/tiny-transformer.yaml`.

## Run 16 GPUs with Spur

Install Spur and start it on two machines with 8 GPUs each, as in [Install and run Spur](spur-install.md). Check it with that page's `spur sinfo -N` and `spur srun` steps first. This recipe does not start Spur, and it does not request an allocation.

Run aorta from a machine that can execute the `spur` binary. Set `spur_bin` in the recipe to that binary, for example `$SPUR_HOME/bin/spur`. The default is `spur` on `PATH`. Leave Slurm's own `srun` as the `srun` on `PATH`.

```bash
export SPUR_CONTROLLER_ADDR=http://<controller IPv4>:6817
export AORTA_MASTER_PORT=29501
export AORTA_SHARED_DIR=/path/shared/by/every/node
export AORTA_RANK0_IP=10.x.x.x    # optional; otherwise aorta asks Spur

aorta triage run --recipe recipes/training/tiny-transformer-spur.yaml --dry-run
aorta triage run --recipe recipes/training/tiny-transformer-spur.yaml
```

Aorta starts one `spur srun` of the training job. Each GPU runs that job directly. The workers do not run the `aorta triage run` command again.

| Variable | Required | What to set |
|---|---|---|
| `SPUR_CONTROLLER_ADDR` | yes, unless the controller is on `localhost:6817` | Address of the Spur controller. Every `spur` command uses it. |
| `AORTA_MASTER_PORT` | yes | A free TCP port. The other processes connect to rank 0 on this port. Aorta does not pick a port for you. |
| `AORTA_SHARED_DIR` | yes | A directory that every node can read and write. Rank 0 writes `rank0.json` there, and the aorta process reads it back. A directory that exists only on one machine, such as that machine's `/tmp`, is not visible to the others. |
| `AORTA_RANK0_IP` | no | IPv4 address of the machine that will run rank 0. That machine is where the others connect. If you leave this unset, aorta asks Spur for the address. |
| `spur_bin` in the recipe | no | Path to the Spur binary. The default is `spur`. |

If one process fails during training, it exits instead of waiting for the others. Spur then stops the rest of the step, so the remaining processes do not hold the GPUs until a timeout.

## Run 16 GPUs with Monarch

Install Monarch into the Python environment that runs aorta, as in [Install and run Monarch](monarch-install.md). Every machine must see that environment at the same path. Workers load `aorta.workloads.monarch_trainer`, so aorta has to be installed in that environment too. Run that page's one-machine check first. If Python cannot import Monarch, the run fails and the message says Monarch is not installed.

Use a Slurm allocation with 2 machines, and run aorta from a shell on one of those machines. The recipe reads the job id from `SLURM_JOB_ID`, which Slurm sets in that shell. To use a different allocation, set `MONARCH_SLURM_JOB` to its job id. The recipe does not call `salloc` or `sbatch` itself.

```bash
salloc -N 2 --exclusive -t 02:00:00              # if you do not already have an allocation
srun --jobid "$SLURM_JOB_ID" --overlap -N1 --pty bash    # a shell on the first machine

source /shared/path/monarch/monarch-env.sh       # ROCm only; see the Monarch install page
export AORTA_CONTROLLER_IP=10.x.x.x              # optional; see below

aorta triage run --recipe recipes/training/tiny-transformer-monarch.yaml --dry-run
aorta triage run --recipe recipes/training/tiny-transformer-monarch.yaml
```

The workers connect to the machine running aorta. Aorta looks up that machine's IPv4 address from its hostname. If the lookup fails or gives an address the workers cannot reach, set `AORTA_CONTROLLER_IP` to the right IPv4 address.

Aorta then starts one Monarch worker process on each machine, as one `srun --overlap` step of the allocation. Monarch starts one training process per GPU on each machine: 2 machines and 8 GPUs each for this recipe. If training fails, aorta stops those worker processes. Multi-machine settings such as the RCCL network variables are listed in [Settings for runs across machines](monarch-install.md#settings-for-runs-across-machines).

## Run 16 GPUs with Monarch on Spur

Use this when Spur, not Slurm, gives you the machines. Monarch still runs the training. Spur starts Monarch's worker processes.

Before you start:

1. Install and start Spur on 2 machines with 8 GPUs each, as in [Install and run Spur](spur-install.md).
2. Install Monarch, as in [Install and run Monarch](monarch-install.md).

Then, on one of the two Spur machines, request a Spur allocation and run aorta from the shell it opens. `spur salloc` opens that shell on the machine where you run it, and the Monarch workers connect back to that machine.

```bash
export SPUR_CONTROLLER_ADDR=http://<controller IPv4>:6817
"$SPUR_HOME/bin/spur" salloc -N2 --gpus-per-node=8 --exclusive -t 02:00:00

# In the new shell. SPUR_JOB_ID is now set.
source /shared/path/monarch/monarch-env.sh       # ROCm only
export AORTA_CONTROLLER_IP=10.x.x.x              # optional, same as above

aorta triage run --recipe recipes/training/tiny-transformer-monarch-spur.yaml --dry-run
aorta triage run --recipe recipes/training/tiny-transformer-monarch-spur.yaml
exit                                              # releases the Spur allocation
```

The recipe sets `scheduler: spur`. Aorta then runs `spur srun -N2 --ntasks=2 --ntasks-per-node=1` inside your Spur allocation: once to read each machine's hostname, and once to start one Monarch worker on each machine. Set `spur_bin` in the recipe to `$SPUR_HOME/bin/spur`, or to wherever your `spur` binary is. The default is `spur` on `PATH`.

- **Run aorta from `spur salloc`, not from `spur sbatch`.** Spur runs a multi-machine batch script on every machine, so `spur sbatch` would start one aorta controller on each machine. `spur salloc` opens one shell on the machine where you run it.
- **Aorta refuses to start without `SPUR_JOB_ID`.** That variable is how aorta knows it is inside a Spur allocation. It does not request one itself.
- **Use the Slurm recipe inside a Slurm allocation, and this recipe inside a Spur allocation.** Inside a Spur allocation, `SLURM_JOB_ID` is Spur's job id. The Slurm recipe stops with an error there instead of passing that id to Slurm's `srun`.
- **Worker output appears when the worker step ends.** `spur srun` delivers a step's output at the end. Monarch's own per-machine log is `/tmp/$USER/monarch_log.log`.

## If RANK is already set in your shell

Run aorta once. That process starts the workers. It does not train on its own.

The process should train, instead of starting another set of workers, only when it is already a worker:

- Spur sets `AORTA_INNER=1` on each worker.
- torchrun sets `TORCHELASTIC_RUN_ID` together with `RANK` and `WORLD_SIZE`.

A `RANK` left over from an earlier job, with neither of those set, still starts the workers. The result is still saved.

## The four recipes

| Recipe | How it starts | Processes |
|---|---|---|
| `recipes/training/tiny-transformer.yaml` | in this process | 1 |
| `recipes/training/tiny-transformer-spur.yaml` | Spur | 16 |
| `recipes/training/tiny-transformer-monarch.yaml` | Monarch on a Slurm allocation | 16 |
| `recipes/training/tiny-transformer-monarch-spur.yaml` | Monarch on a Spur allocation | 16 |

Each recipe trains for 20 steps. The 16-process recipes use batch size 8 and sequence length 128. The one-process recipe uses batch size 2 and sequence length 32 so it can run without a GPU. All four use the model above and the pass checks in "How to tell that it passed".

## What you do yourself

Request the machines, start Spur, and install Monarch before the 16-GPU recipes. This workload does not do those steps. Continuous integration runs the one-process recipe only. Run the 16-GPU recipes yourself on an allocation you already have.
