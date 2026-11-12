# Getting started with AORLTA

AORLTA trains a model with GRPO on tasks whose answers a program can check: the model answers, a set of checks
scores the answer, and a LoRA on the model is trained on the difference between the answers to one prompt. It runs
on one 8-GPU node: TokenSpeed engines generate, a reward service runs the checks, and the trainer sits beside the
engines. It fits a task where you can write a prompt, read an answer out of the reply and score it automatically,
such as "write or fix a kernel and run it", "answer and check against a reference", or "propose a change and
measure it". [ADAPTING.md](ADAPTING.md) says how to plug in your own task.

This guide is the order to do things in. Paths are relative to `aorlta/` in a checkout of this repository.

## What you need

| | tested | notes |
|---|---|---|
| GPUs | one node with 8 MI350X (gfx950) | The images and the shipped profiles build for gfx950 only. MI355X is gfx950 too, but untested. MI300X (gfx942) needs TokenSpeed's image built for it, `docker/Dockerfile.trainer`'s causal-conv1d built for it (`HIP_ARCHITECTURES`, `PYTORCH_ROCM_ARCH`), and `offload_arch`/`target` changed in the profiles: untested. |
| fewer GPUs | no | The layout is configuration: `rollout.engines` (one engine per GPU listed), `trainer.devices` (null: rank r beside engine r), and `reward.remote.adapters` (the GPU sanitizer checks). The engine start-up check needs `rollout.P >= 4`, not 4 engines, so one engine with one colocated rank and CPU-only checks should run, more slowly; it has not been run. |
| node software | Docker, and Slurm | `jobs/aorlta-run.sbatch` refuses to run outside a Slurm job (exit 91). The docker daemon must be able to read the checkout and every data directory at the paths the job sees them: not a root-squashed network home, and not a Slurm job's private `/tmp`. |
| Python | 3.12 | For the CPU tests and for the job, which reads the config on the host. Many cluster hosts have an older `python3`; `uv venv --seed -p 3.12 <dir>` (uv fetches a 3.12 if it has to), or any CPython 3.12, will do. |
| images | built by the job | TokenSpeed at commit `3b684074c0ef79bc8bc00b31cda8dc2850e4b71f` (`git clone https://github.com/lightseekorg/tokenspeed && git -C tokenspeed checkout 3b684074c0ef79bc8bc00b31cda8dc2850e4b71f`, then `TS_SRC=$PWD/tokenspeed`), built by the job from its `docker/Dockerfile.amd`; and the trainer image on top of it from `docker/Dockerfile.trainer`. |
| model | `Qwen/Qwen3.5-4B` at revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | The job runs offline, so download it first: `HF_HOME=<cache> hf download Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. `model.base_dir` is the snapshot directory under `<cache>/hub/`. Only the Qwen3.5 family works as it stands: the turn templates use Qwen3.5's control-token ids (`rollout/templates.py`), and the trainer needs transformers' Qwen3.5 linear-attention fast path (`trainer.require_fast_path`). Another Qwen3.5 size, or a fine-tune of it, is the supported change. |
| prompts | a pool of your own | The stand-in configs read a kernel dataset (`kernels.json` and `listings/`) that is not published. `examples/standin-pool/` shows its format but is too small for a run: the stand-in landing check alone picks 40 kernels of 60 to 400 instructions. |
| stand-in check tools | only for the stand-in profiles | `standin-static` runs rocjitsu's `rj_waitcheck` (ROCm/rocm-systems branch `shared/rocjitsu/sanitizers`, built from source, with a libstdc++ that has GLIBCXX_3.4.31); `standin-gpu` adds ConSan (`librocjitsu_dbi_hooks.so` from the same build) and ASan (the ROCm 7.0.2 `*-asan` packages' `lib/asan`), through aorta's sanitizer backends at commit `d22f6116235c05395a0241a83d6f7e952a988454`. None of these is packaged; a task with checks of its own needs none of them. |

## 1. Install and run the CPU tests

```bash
cd aorlta
uv venv --seed -p 3.12 ~/venvs/aorlta && . ~/venvs/aorlta/bin/activate   # or any Python 3.12 venv
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-dev.txt
pip install -e . --no-deps
```

The suite in three parts, as CI runs it. On a machine with many cores, do not use `-n auto`: a worker can grow by
about 1.5 GB for each wired test it runs, which is why CI caps the workers and sets the allocator variables below.

```bash
export OMP_NUM_THREADS=1 MALLOC_MMAP_THRESHOLD_=65536 MALLOC_ARENA_MAX=2
python -m pytest -q -n 8 -m "not heavy and not cpu_run"            # about 2 min; CI uses 4 workers
OMP_NUM_THREADS=2 python -m pytest -q -n 2 -m "heavy and not cpu_run"   # about 6 min
OMP_NUM_THREADS=4 python -m pytest -q -m cpu_run                    # about 6 min
```

A test that needs something this machine does not have skips and says what: hipcc, the base model's tokenizer
(read from `HF_HOME`), the stand-in dataset (`AORLTA_STANDIN_DATASET`), aorta's fixtures and rocjitsu's
`rj_waitcheck` (`AORLTA_AORTA_ROOT`, `AORLTA_RJ_WAITCHECK`), or a GPU (only inside `jobs/aorlta-reward-standin.sbatch`).
Run them in a git checkout: a run records its code commit.

## 2. Run the loop on CPU

`--components fake` runs the real loop with fake engines, checks, trainer and weight sync (README, "The loop on CPU,
with fake components"). It checks that a config loads and the loop runs; the fakes use prompts and checks of their
own, so it says nothing yet about your data.

## 3. Set up the node

Put your paths in a site file, so the configs stay committed without them:

```yaml
run: {root: /scratch/me/aorlta-runs}            # the docker daemon must be able to write here
model: {base_dir: /data/hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a}
data: {prompt_pools: [/data/my-pool]}
```

and set the job's environment (the full list is in the header of `jobs/aorlta-run.sbatch`):

```bash
export AORLTA_SITE=/path/to/site.yaml AORLTA_VENV=~/venvs/aorlta TS_SRC=/path/to/tokenspeed
export AORLTA_HF_HOME=/path/to/hf-cache AORLTA_TMPDIR=/path/to/tmp   # both visible to the docker daemon
# the stand-in profiles only:
export AORLTA_ROCJITSU_ROOT=... AORLTA_LIBSTDCXX_SHIM=... AORLTA_AORTA_ROOT=... AORLTA_ASAN_LIB=...
```

`rocm/dev-ubuntu-24.04:7.0.2`, which the job pulls for the GPU sanitizer checks, comes from Docker Hub: set
`AORLTA_DOCKER_CONFIG` to a docker client config with credentials if anonymous pulls are rate-limited.

## 4. The first GPU run: score, then train

Score the base model first (`orchestrator.mode: score`): the same rollouts and checks with no training, so it shows
whether the checks settle and whether the reward separates answers, before any training time is spent. Then train.
Both are one job, submitted from `aorlta/` with the config committed:

```bash
mkdir -p runs && CONFIG=configs/score-standin.yaml sbatch jobs/aorlta-run.sbatch
```

What the job does, in order, is listed in the README ("A stand-in run on GPUs"). Its log is `runs/job-<id>.out`;
the run itself is under `<run.root>/standin/<run.run_id>/`.

Time, measured on MI350X nodes from `main`:

| step | time |
|---|---|
| TokenSpeed image build, on a node that has never built it | about 12 min (2 to 3 min for the trainer image after it); about 1 min where docker's build cache still holds the layers |
| the GPU sanitizer checkers' start and pre-flight (`standin-gpu` only) | about 1 min |
| a score run of `configs/score-standin.yaml` (16 prompts x 4 samples) | about 7 min, setup included |
| a training run of `configs/dev-standin.yaml` (32 prompts x 8 samples, `standin-static`) | about 8 min to the end of iteration 1, then about 4 min per iteration |

The job removes the images it built when it ends, so the next job on that node builds them again (from the build
cache, if it is still there).

## 5. Read the outputs

- `score/summary.txt` (a score run) or `iterations/iter-<n>/summary.json` (a training run): the counts, the reward,
  each check's results and the landing verdicts. The README's "Run outputs" has a snippet that prints one line per
  iteration.
- `episodes.jsonl` holds ids, not text. To read an answer, decode it with the model's tokenizer:

  ```python
  import json
  from transformers import AutoTokenizer
  tok = AutoTokenizer.from_pretrained(BASE_DIR)
  ep = json.loads(open(f"{RUN_DIR}/iterations/iter-000001/episodes.jsonl").readline())
  print(ep["prompt_id"], ep["end_reason"], ep["reward"])
  print(tok.decode(ep["turns"][-1]["response_ids"]))
  ```
- TensorBoard: `pip install tensorboard` (not a dependency), then `tensorboard --logdir <run_dir>/tb`.
- Checkpoints (`ckpt/iter-<n>/trainer/trainer.pt`) hold the fp32 LoRA, the optimizer state and the RNG states; the
  last 2 are kept. The README's "Run outputs" says how to use a trained LoRA outside the run.

## 6. The settings you will change

Every key, with its default and range, is in `src/aorlta/config.py`; an unknown key is an error.

| key | what |
|---|---|
| `rollout.P`, `rollout.K` | prompts per iteration, samples per prompt (one GRPO group per prompt) |
| `rollout.temperature`, `top_p`, `top_k` | sampling |
| `rollout.max_new_tokens`, `rollout.max_repair_turns` (0 to 3) | the reply budget, and how many compile-feedback turns a chain gets |
| `rollout.engines`, `rollout.gpu_memory_utilization` | the engine GPUs, and each engine's share of its GPU (a colocated trainer rank gets the rest) |
| `trainer.ranks`, `trainer.devices`, `trainer.lr`, `trainer.micro_batch_tokens` | the LoRA trainer; the LoRA itself is r 16, alpha 32 (`LORA_R`, `LORA_ALPHA` in `orchestrator/wiring.py`) |
| `trainer.max_generation_gap`, `trainer.gap_statistic` | how far the trainer's log-probs may be from the engine's before a sample is left out |
| `orchestrator.iterations`, `orchestrator.run_seed` | the run's length and seed |
| `orchestrator.min_scored_fraction` | the share of an iteration's samples that must be scored before it trains (0.9 by default); after one retry of the unsettled checks, a shortfall stops the run |
| `orchestrator.deadlines.generation_s`, `scoring_s` | floors of the per-iteration deadlines |
| `orchestrator.score.prompts`, `passes` | what a score run scores |
| `reward.profile`, `reward.timeouts`, `reward.gpu_check_s` | which checks run, and their time limits |

Code comments and some config comments cite labels of the internal design notes (Section 6.x, C1-C9, D7, T10); the
ones you meet in configs are D7 (the minimum scored fraction), D9 (`rollout.thinking`), D10 (the GPU layout) and
D11 (`sync.partial`).

## How many GPUs

AORLTA runs on one node; it does not span nodes. On that node, the config says which GPU does what:

| key | what it places |
|---|---|
| `rollout.engines` | one TokenSpeed engine per GPU index listed |
| `trainer.ranks`, `trainer.devices` | LoRA trainer ranks: with `devices` null, rank r shares engine r's GPU (it gets what the engine leaves, `1 - gpu_memory_utilization`); a list gives the ranks GPUs of their own |
| the check profile, `reward.remote.adapters` | where GPU checks run (the stand-in sanitizers: their own GPUs; a plugin's check: its own settings) |

| GPUs | layout | runs |
|---|---|---|
| 1 | `engines: [0]`, `ranks: 1` (beside the engine), CPU checks | the Q&A example (`configs/qa-*.yaml`, `configs/arith-*.yaml`); measured below |
| 2 | `engines: [0, 1]`, `ranks: 1` or 2 | any task without GPU checks; untested |
| 4 | `engines: [0, 1, 2, 3]`, `ranks: 1` to 4 | `configs/dev-standin.yaml`; with GPU checks, put them on GPUs the engines leave (`hipk.run.gpus`) |
| 8 | 4 engines and ranks, the other 4 for GPU checks | `configs/standin.yaml`, `configs/hipk-*.yaml` |

Memory, measured from the job's `vram.tsv` (peak per GPU, MI350X, 288 GiB), with Qwen3.5-4B:

| what | peak |
|---|---|
| a TokenSpeed engine at `gpu_memory_utilization: 0.45` | 131 GiB: an engine reserves its share for its KV cache, whatever the model's size, so lower the share on a GPU someone else uses |
| an engine plus a colocated rank, 512-token replies (`configs/arith-train.yaml`) | 187 GiB |
| an engine plus a colocated rank, 4096-token replies, `micro_batch_tokens: 20480` (`configs/dev-standin.yaml`) | 266 GiB |

Estimates, not measured: a rank holds the bf16 base (2 bytes per parameter), a bf16 copy of the merged target
weights it publishes, the small fp32 LoRA and its optimizer state, and activations that grow with
`micro_batch_tokens`. So a 4B model needs roughly 20 GB before activations, and a 27B model roughly 110 GB, which
still fits beside an engine only with `gpu_memory_utilization` lowered. An engine runs the whole model on one GPU
(no tensor parallelism here), so a model whose bf16 weights do not fit in an engine's share (about 60B parameters
at 0.45 of 288 GiB) does not fit at all.

On a node other people use: give the run the GPU indices that are free (`rocm-smi` shows them), lower
`rollout.gpu_memory_utilization`, and keep to one AORLTA job per node. Engines listen on fixed host ports (20000 +
2000 x engine index), and the job removes any `aorlta-run-*` container it finds when it starts, taking it for one a
cancelled job left behind. The job asks Slurm for the whole node (`--exclusive`).

## Models, checkpoints and adapters

- **The model** is `model.base_dir`: a Hugging Face snapshot directory (weights, config and tokenizer), read
  offline. To switch, download the model (`hf download <repo> --revision <rev>`) and point `model.base_dir` at its
  snapshot. Today that must be a Qwen3.5-family model (another size, or a fine-tune of one): the turn templates use
  Qwen3.5's control-token ids, the trainer checks for Qwen3.5's linear-attention kernels
  (`trainer.require_fast_path`), and the weight sync and the merge tool expect Qwen3.5's checkpoint layout
  (`model.language_model.layers.N...`). A Qwen3 model such as Qwen3-0.6B needs those three generalised first; it is
  the most requested change.
- **Checkpoints** are written to `<run.root>/<run.source>/<run.run_id>/ckpt/iter-<n>/` after every iteration, and the
  last `orchestrator.keep_ckpts` (2) are kept; older ones are deleted by the run. Each holds the fp32 LoRA, the Adam
  state and the RNG states (a few hundred MB for Qwen3.5-4B, an estimate from its 130 MB fp32 LoRA). Delete a
  finished run's directory, or its `ckpt/`, when you no longer need to resume it.
- **Adapters and merged models**: the README's "Run outputs" says how to turn a checkpoint into a PEFT adapter, how
  to merge an adapter into its base (`tools/merge_base_adapter.py`), and how to score it: a score run with
  `model.adapter_dir` set to the adapter serves the base plus the adapter.

## When things go wrong

The job's exit status says where it stopped; its log says why.

| exit | meaning | what to do |
|---|---|---|
| 0 / 3 | the run completed / stopped | a stop's reason is in `run_result.json` |
| 90 | this node is in `AORLTA_EXCLUDE_NODES` | give `sbatch --exclude` the same list |
| 91 | not inside a Slurm job | submit with `sbatch` |
| 92 | not submitted from the package root | `cd aorlta` first |
| 93, 94 | the checkout, the config or the site file could not be read; the config is not committed, or does not load | commit the config; `python -c "from aorlta.config import load_config; load_config('configs/X.yaml')"` shows the error |
| 95 | a setting the job needs is missing, or the docker daemon does not see a directory where the job does | the message names it; a private `/tmp` or a root-squashed home is the usual cause |
| 96 | aorta's ASan node-health canary failed on a checker GPU | ASan cannot be trusted on that node; exclude it and resubmit |
| 97, 98 | the checker image could not be pulled, a checker did not start, or the checkers' pre-flight failed | `runs/<job id>/checker-*.log` and `smoke.log` |
| 99 | the run image cannot import what the run needs | `runs/<job id>/imports.log` |
| 2 (from the orchestrator) | the config asks for something the wiring cannot run | the message lists every problem at once |

Stops that a run reports in `run_result.json`, and what they usually mean:

- `engine_start_failed`: an engine's settings read back wrong, or its self-test drew the same sample for different
  seeds. TokenSpeed on AMD defaults to a greedy sampler that ignores temperature and seed; the engines are started
  with `sampling_backend: triton`, and this check is what catches it if that ever stops taking effect.
- the round trip is not PROVEN: a weight update did not reach the engines. Nothing is trained or scored.
- the scored fraction is short: too many checks did not settle (look at the counts by reason in the summary). A
  check that cannot settle leaves a sample unscored, never clean; lower `min_scored_fraction` only once you know why.
- a gap or landing check failed: trainer and engine disagree on the model. Stop and report it, with the run
  directory; it is not a setting to tune away.

Good to know: a run id is used once (copy the config under a new `run.run_id` for a new run); a score run is never
resumed; `RESUME=1` resumes a training run from its last checkpoint; and `KILL_AFTER_ITER=n` is the kill-and-resume
drill. The weight slots live in `/dev/shm` and are removed when the run ends.
