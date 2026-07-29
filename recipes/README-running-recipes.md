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
- Fresh-process distributed trials treat any worker bootstrap/crash as fatal so
  peers cannot continue into a different trial generation. Use
  `srun --kill-on-bad-exit=1` under Slurm. Any launcher without an elastic agent
  store (including srun and older/static torchrun) must export a job-unique
  `AORTA_TRIAL_MASTER_PORT_BASE` whose next 1000 ports are reserved for isolated
  trial rendezvous (for example, `30000`).
- A recipe's per-cell `extra_env` is applied by the runner (via
  `os.environ.update`) before workload `setup()`. In legacy `in_process` mode,
  this is still too late for values cached during Python/native-library import.
  In `process` mode, AORTA puts the controlled overlay in the worker environment
  before the fresh interpreter starts, so import-time HIP/RCCL values can vary
  safely by trial and cell.
- Workloads declare their default/required isolation policy. `race` requires a
  fresh process per trial; its AINIC recipe env is therefore active before
  torch/HIP/RCCL initialization. Other workloads remain in-process unless a
  recipe requests `trial_isolation: process`.
- Launcher identity (`RANK`, `WORLD_SIZE`, `LOCAL_RANK`, `MASTER_ADDR`,
  `MASTER_PORT`, etc.) always belongs to torchrun/srun and cannot be overridden
  by an isolated cell.
- Topology (rank count, ranks-per-host) is the launcher's responsibility.

## 5. Read results

```bash
cat triage_results/<TICKET>/<workload>/<timestamp>/matrix.md
```

The run directory `triage_results/<TICKET>/<workload>/<timestamp>/` contains
`matrix.md` (summary table), `matrix.json` (full per-cell stats),
`recipe.resolved.yaml`, and `cells/<cell>/.../trial_*.json` (per-trial detail).
The `<TICKET>` comes from the recipe's `ticket:` field. On rank 0 when
finished the CLI prints a concise pass/fail/error summary of the cells
(pointing at each non-clean cell's artifacts) followed by the
`Wrote matrix to <run_dir>` line; pass `-v` to also stream per-trial
progress to stderr while the matrix runs.

## Tip: smoke-test a recipe before the full matrix

A full matrix (many cells × trials × steps) can take hours. If a recipe has a
small companion variant (fewer cells/trials/iters), run that first to confirm
the end-to-end path works on your cluster in seconds, then launch the full one.
