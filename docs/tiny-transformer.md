# Tiny transformer workload

`tiny_transformer` trains the runs/04 model through the existing `training`
loop and records pass or fail. The launcher is a `workload_config` field, so
the same recipe shape runs in-process, under Spur, or under Monarch. Spur and
Monarch are not install dependencies. CI uses the in-process recipe.

The model defaults are a 4-layer transformer: hidden size 256, 4 heads, FFN
512, vocab 256. Recipes can override any training field. Omitted fields keep
those defaults.

## What a pass is

Rank 0 reports `passed: true`, one finite `final_loss`, and `world_size`
equal to the process group. The in-process smoke is world size 1. The Spur
and Monarch recipes set `expected_world_size: 16` (`nodes * gpus_per_node`)
and fail the trial when the group is a different size.

`checks.fail_on_rank_loss_mismatch` is off unless the recipe turns it on.
With one id in `CUDA_VISIBLE_DEVICES`, the rank calls `set_device(0)`.
Otherwise it uses `LOCAL_RANK`.

## Launchers

| `launcher` | Where it runs | When to use it |
|---|---|---|
| `inprocess` (default) | This process, world size 1 | Smoke test. No Spur, Monarch, or GPU required |
| `spur` | `spur srun` on an overlay that is already up | Same job through Spur. Does not submit a Slurm allocation |
| `monarch` | A Monarch proc mesh on an allocation that already exists | Same job through Monarch. Does not submit a Slurm allocation |

A process with `AORTA_INNER=1`, or a torchrun rank (`TORCHELASTIC_RUN_ID`
plus `RANK` and `WORLD_SIZE`), trains instead of spawning again. `RANK` alone
does not make the process a worker. A controller that inherited `RANK` from
an earlier shell still saves the trial: the rank-0 write gate does not treat
that process as a peer.

## In-process smoke

From the repository root, after `aorta` is installed in the active environment:

```bash
aorta triage run --recipe recipes/training/tiny-transformer.yaml --dry-run
aorta triage run --recipe recipes/training/tiny-transformer.yaml
```

`--dry-run` validates the recipe and does not train. The real run is one rank.
`device: auto` uses a GPU when one is visible and CPU otherwise.

`aorta run --workload tiny_transformer` lists the workload only after the
package is reinstalled. It does not take a recipe, so it uses the defaults
above rather than `recipes/training/tiny-transformer.yaml`.

## Spur

The controller runs one `spur srun` of the resolved training config. It does
not replay the parent `aorta` command. Workers map `SLURM_PROCID` to `RANK`.
`LOCAL_RANK` is `SLURM_LOCALID` when Spur sets it, so cyclic placement does
not point two tasks on the same node at one GPU. `RANK % gpus_per_node` is
only the fallback when `SLURM_LOCALID` is unset. `MASTER_ADDR` is the IPv4
address of the node that will be rank 0, because that rank hosts the
rendezvous store.

Bring the Spur overlay up first. This recipe does not start it and does not
submit Slurm. Point `workload_config.spur_bin` at the overlay binary. On this
cluster that binary is not on `PATH`; do not install Spur onto `PATH` and do
not symlink `srun`.

```bash
export AORTA_MASTER_PORT=29501
export AORTA_SHARED_DIR=/path/shared/by/every/node
export AORTA_RANK0_IP=10.x.x.x          # optional; otherwise the controller probes
export AORTA_CONTROLLER_IP=10.x.x.x     # optional; only if you also want the controller address logged

aorta triage run --recipe recipes/training/tiny-transformer-spur.yaml --dry-run
aorta triage run --recipe recipes/training/tiny-transformer-spur.yaml
```

| Variable | Required | Meaning |
|---|---|---|
| `AORTA_MASTER_PORT` | yes | TCP port for the rendezvous. The controller does not pick one |
| `AORTA_SHARED_DIR` | yes | Directory every node can read and write. Rank 0 writes `rank0.json` here. Not node-local `/tmp` |
| `AORTA_RANK0_IP` | no | IPv4 of the node that runs `RANK=0`. When unset, one `spur srun` probe prints it |
| `AORTA_CONTROLLER_IP` | no | Controller IPv4. Not used as `MASTER_ADDR`. Set it when the hostname resolves to IPv6 link-local first and you want that lookup to succeed |
| `workload_config.spur_bin` | no | Spur binary. Default `spur` |

Run the controller from a host that can execute `spur_bin`. A rank that fails
while the others are still in a collective does not enter `dist.barrier`. It
exits, and Spur's step supervisor terminates the rest of the step.

## Monarch

Monarch is imported only for `launcher: monarch`. The wheels published for
`torchmonarch` are CUDA builds. This cluster uses the pytorch_monarch source
build against ROCm (`venvs/rocm10.1` in the Monarch tree). The actor class is
`aorta.workloads.monarch_trainer.Trainer`. Workers import that module, so the
same environment has to be on the worker nodes.

The controller attaches to an allocation that already exists. Set
`SLURM_JOB_ID` or `MONARCH_SLURM_JOB`. It does not call `salloc` or `sbatch`.

```bash
export SLURM_JOB_ID=<existing allocation>
export AORTA_CONTROLLER_IP=10.x.x.x     # when the hostname resolves to IPv6 link-local first

aorta triage run --recipe recipes/training/tiny-transformer-monarch.yaml --dry-run
aorta triage run --recipe recipes/training/tiny-transformer-monarch.yaml
```

The controller binds its own transport to an IPv4 address, then `srun
--overlap` starts one Monarch worker process per node. The proc mesh is
`gpus_per_node` GPUs on each of `nodes` hosts (2×8 for the recipe). A missing
`monarch` import fails the trial with a message that names the source build.

## Recipes

| Recipe | Launcher | World size |
|---|---|---|
| `recipes/training/tiny-transformer.yaml` | in-process | 1 |
| `recipes/training/tiny-transformer-spur.yaml` | Spur | 16 |
| `recipes/training/tiny-transformer-monarch.yaml` | Monarch | 16 |

The two distributed recipes use batch size 8 and sequence length 128. The
in-process smoke uses batch size 2 and sequence length 32 so it can run
without a GPU. All three keep the same model shape and the same numeric
checks, including cross-rank loss agreement.

## What this does not do

The workload does not submit a Slurm job, start the Spur overlay, or install
Monarch. It does not measure RDMA, WeightBridge, or a second mesh shape.
Those stay outside this recipe. A 16-GPU Spur or Monarch run is a manual
check on an existing allocation. It is not a CI gate.
