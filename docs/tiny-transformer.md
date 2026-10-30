# Tiny transformer

`tiny_transformer` trains a small transformer and records whether the run passed or failed. It uses aorta's existing training loop. You choose how the processes start by setting `launcher` inside the recipe:

| `launcher` | What it does | When to use it |
|---|---|---|
| `inprocess` (default) | Trains in the process that is running aorta. One process. | Check that the recipe runs. No Spur, no Monarch, and no GPU required. |
| `spur` | Starts the training processes with `spur srun`. | The same training job, launched by Spur. Start Spur yourself first. |
| `monarch` | Starts the training processes with Monarch. | The same training job, launched by Monarch. Have a Slurm allocation first. |

Aorta does not install Spur or Monarch. Continuous integration runs only the one-process recipe.

The model is a 4-layer transformer: hidden size 256, 4 attention heads, feed-forward size 512, vocabulary 256. A recipe can override any training field. A field you leave out keeps that default.

## How to tell that it passed

Read the result written by rank 0, the first process. The run passed when all of these are true:

- `passed` is `true`.
- `final_loss` is an ordinary number. `NaN` and infinity are failures.
- `world_size` is the number of processes that trained together.

The one-process recipe runs as a single process, so `world_size` is 1. It does not compare that count with a configured value.

The Spur and Monarch recipes run on 2 nodes with 8 GPUs each, which is 16 processes. They set `expected_world_size: 16`. The run fails if any other number of processes joins.

Those three recipes also fail the run when:

- the loss, a gradient, or a model output is `NaN` or infinity
- the processes finish with different losses

The loss comparison is `checks.fail_on_rank_loss_mismatch`. It stays off unless a recipe sets it to `true`. All three recipes here set it to `true`. While it is on, every process trains on the same data, so the losses can be compared. The recipes use `float32` so that comparison stays stable.

## Which GPU a process uses

This does not decide pass or fail. It only picks the device.

If `CUDA_VISIBLE_DEVICES` names one GPU, that is the only GPU the process can see, so it uses device 0. If several GPUs are visible, or the variable is unset, the process uses `LOCAL_RANK` as the GPU index on its machine. Under Spur, `LOCAL_RANK` is that process's index on its node (`SLURM_LOCALID`). Two processes on the same machine then use two different GPUs, including when Slurm assigns tasks round-robin across nodes. If `SLURM_LOCALID` is missing, the index falls back to the global rank modulo the number of GPUs per node.

## Run one process

Install aorta in the active environment, then run these commands from the repository root:

```bash
aorta triage run --recipe recipes/training/tiny-transformer.yaml --dry-run
aorta triage run --recipe recipes/training/tiny-transformer.yaml
```

`--dry-run` checks the recipe and does not train. The real run is one process, for 20 steps. `device: auto` uses a GPU when one is visible, and the CPU otherwise.

`aorta run --workload tiny_transformer` is available after you install this version of the package. That command does not read a recipe file. It uses the built-in defaults: one process, batch size 2, sequence length 32. It does not use the settings in `recipes/training/tiny-transformer.yaml`.

## Run 16 GPUs with Spur

Start Spur before this recipe. The recipe does not start Spur, and it does not request a Slurm allocation.

Run aorta from a machine that can execute the `spur` binary. If `spur` is not on `PATH`, set `spur_bin` in the recipe to that binary. Leave Slurm's own `srun` as the `srun` on `PATH`.

```bash
export AORTA_MASTER_PORT=29501
export AORTA_SHARED_DIR=/path/shared/by/every/node
export AORTA_RANK0_IP=10.x.x.x    # optional; otherwise aorta asks Spur

aorta triage run --recipe recipes/training/tiny-transformer-spur.yaml --dry-run
aorta triage run --recipe recipes/training/tiny-transformer-spur.yaml
```

Aorta starts one `spur srun` of the training job. Each GPU runs that job directly. The workers do not run the `aorta triage run` command again.

| Variable | Required | What to set |
|---|---|---|
| `AORTA_MASTER_PORT` | yes | A free TCP port. The other processes connect to rank 0 on this port. Aorta does not pick a port for you. |
| `AORTA_SHARED_DIR` | yes | A directory that every node can read and write. Rank 0 writes `rank0.json` there, and the aorta process reads it back. A directory that exists only on one machine, such as that machine's `/tmp`, is not visible to the others. |
| `AORTA_RANK0_IP` | no | IPv4 address of the machine that will run rank 0. That machine is where the others connect. If you leave this unset, aorta asks Spur for the address. |
| `spur_bin` in the recipe | no | Path to the Spur binary. The default is `spur`. |

If one process fails during training, it exits instead of waiting for the others. Spur then stops the rest of the step, so the remaining processes do not hold the GPUs until a timeout.

## Run 16 GPUs with Monarch

Install a ROCm build of Monarch yourself, on the machine that runs aorta and on the workers. Aorta does not install it. The `torchmonarch` packages published for pip are CUDA builds, so they do not work on a ROCm host. Workers load `aorta.workloads.monarch_trainer`, so that module has to be importable there too. If Python cannot import Monarch, the run fails and the message says Monarch is not installed.

Use a Slurm allocation you already have. Set `SLURM_JOB_ID`, or set `MONARCH_SLURM_JOB` to that job id. This recipe does not call `salloc` or `sbatch`.

```bash
export SLURM_JOB_ID=<existing allocation>
export AORTA_CONTROLLER_IP=10.x.x.x    # the machine running aorta, when its hostname is not an IPv4 address

aorta triage run --recipe recipes/training/tiny-transformer-monarch.yaml --dry-run
aorta triage run --recipe recipes/training/tiny-transformer-monarch.yaml
```

Set `AORTA_CONTROLLER_IP` to the IPv4 address of the machine running aorta when that machine's hostname resolves to an IPv6 address. The workers connect to this address. When the hostname already resolves to IPv4, you can leave the variable unset.

Aorta then starts one Monarch worker process on each node, attached to the existing allocation. Monarch starts one training process per GPU on that node. For this recipe that is 2 nodes and 8 GPUs on each node. If training fails, aorta stops those worker processes.

## If RANK is already set in your shell

Run aorta once. That process starts the workers. It does not train on its own.

The process should train, instead of starting another set of workers, only when it is already a worker:

- Spur sets `AORTA_INNER=1` on each worker.
- torchrun sets `TORCHELASTIC_RUN_ID` together with `RANK` and `WORLD_SIZE`.

A `RANK` left over from an earlier job, with neither of those set, still starts the workers. The result is still saved.

## The three recipes

| Recipe | How it starts | Processes |
|---|---|---|
| `recipes/training/tiny-transformer.yaml` | in this process | 1 |
| `recipes/training/tiny-transformer-spur.yaml` | Spur | 16 |
| `recipes/training/tiny-transformer-monarch.yaml` | Monarch | 16 |

Each recipe trains for 20 steps. The 16-process recipes use batch size 8 and sequence length 128. The one-process recipe uses batch size 2 and sequence length 32 so it can run without a GPU. All three use the model above and the pass checks in "How to tell that it passed".

## What you do yourself

Request the machines, start Spur, and install Monarch before the 16-GPU recipes. This workload does not do those steps. Continuous integration runs the one-process recipe only. Run the 16-GPU recipes yourself on an allocation you already have.
