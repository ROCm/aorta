# Running a recipe on a multi-node GPU cluster

How to install, smoke-test, and run **any** triage recipe on a distributed GPU
cluster. This is the operational guide; for recipe *authoring* (schema, cells,
fields) see `README.md`.

Throughout, `<recipe>` is the path to any recipe YAML, e.g.
`recipes/<your-recipe>.yaml`.

## 1. Install

In an environment that already has a working PyTorch (ROCm or CUDA) build:

```bash
pip install -e .
aorta --version
```

Workloads register via the `aorta.workloads` entry point — no extra wiring.

## 2. CPU smoke test (no GPU needed)

Confirm the code is intact on any machine, including a laptop:

```bash
python -m pytest tests/ -q
```

## 3. Validate a recipe without running it

```bash
aorta sweep run --recipe <recipe> --dry-run
```

## 4. Run on the cluster

A recipe is run with a single command, executed identically on **every rank**
(one rank per GPU) under any launcher that provides the standard distributed
env:

```bash
torchrun --nnodes=<N> --nproc-per-node=<GPUS_PER_NODE> \
  --rdzv-backend=c10d --rdzv-endpoint=<HEAD_HOST>:<PORT> \
  $(command -v aorta) sweep run --recipe <recipe>
```

Launch contract (read once):

- Distributed workloads call `dist.init_process_group(backend="nccl")` with no
  args, reading the standard env: `RANK`, `WORLD_SIZE`, `MASTER_ADDR`,
  `MASTER_PORT`, `LOCAL_RANK` (used to bind the GPU). Any launcher that sets
  these works (torchrun shown; under Slurm drive the same `aorta sweep run`
  line via `srun` / `torchrun`).
- Run the **same** command on every rank. Only rank 0 writes result artifacts;
  other ranks participate in the collectives.
- A recipe's per-cell `extra_env` is applied by the runner (via
  `os.environ.update`) *before* the workload's `setup()` calls
  `init_process_group()`. That is in time for most vars — **but not for ones a
  library caches at load time.** Some `NCCL_*` / `RCCL_*` vars are read once when
  the library is first loaded (at `import torch`, which happens before the runner
  applies `extra_env`), so setting them via `extra_env` is too late and they are
  silently ignored. Set those in the **launcher environment** (export them in the
  shell/script that runs `torchrun`/`srun`, before Python starts) and verify they
  took effect in the run log. A second caveat: workloads like `race` do not
  destroy/re-init the process group between cells, so a `NCCL_*` change in a later
  cell will not take effect mid-run and can cause a confusing false-green.
- Topology (rank count, ranks-per-host) is the launcher's responsibility.

## 5. Read results

```bash
cat triage_results/<TICKET>/<workload>/<timestamp>/matrix.md
```

The run directory `triage_results/<TICKET>/<workload>/<timestamp>/` contains
`matrix.md` (summary table), `matrix.json` (full per-cell stats),
`recipe.resolved.yaml`, and `cells/<cell>/.../trial_*.json` (per-trial detail).
The `<TICKET>` comes from the recipe's `ticket:` field; the CLI prints
`Wrote matrix to <run_dir>` on rank 0 when finished.

## Tip: smoke-test a recipe before the full matrix

A full matrix (many cells × trials × steps) can take hours. If a recipe has a
small companion variant (fewer cells/trials/iters), run that first to confirm
the end-to-end path works on your cluster in seconds, then launch the full one.
