# Tutorial: your own task, end to end

This walks through `examples/hip-kernels/`, a small task built as a plugin: the model is asked to write a HIP kernel
from its description (vector add, SAXPY, ReLU, scale and shift, clamp, reverse, a per-block sum, a transpose), and
the reward builds its answer, runs it on a GPU and compares every output element with a host reference. Nothing in
`src/aorlta/` names this task. Copy the directory, change the four parts below, and you have your own.

Paths are relative to `aorlta/`. The commands assume the environment from the README (a Python 3.12 venv with
`requirements-dev.txt` and `pip install -e . --no-deps`).

## What you write

| part | in the example | what it does |
|---|---|---|
| the prompts | `examples/hip-kernels/tasks.json`, and `records()` in `hip_kernels.py` | one prompt record per task: the chat, its prompt id, and a hash of what the checks read for it |
| how an answer is read | `HipkParse` and `HipkFeedback` | the last ```cpp block is the answer; no block, or no definition of the kernel asked for, gets a repair turn that says so; a compile error gets one with the compiler's errors |
| the checks | `HipkRun` and `examples/hip-kernels/hipk-gpu.yaml` | the stand-in parse and compile gates, then `ref.native`: build the answer with a generated host program, run it on a GPU, compare |
| the reward | `HipkV1` | -1.0 no parsable answer, -0.9 compile error, -0.8 wrong signature, -0.6 crash or hang, -0.5 wrong output, +1.0 correct, minus 0.05 x the share of the token budget used |

The module registers each part under a name (`register()` at its end):

```python
plugins.register_task(plugins.Task("hipk", records))
plugins.register_parse_adapter("hipk.parse", HipkParse)
plugins.register_feedback_template("hipk.feedback", HipkFeedback)
plugins.register_profile("hipk-gpu", HERE / "hipk-gpu.yaml")
plugins.register_check_adapter("hipk.run", HipkRun)
plugins.register_reward_fn("hipk.v1", HipkV1)
```

and the run config lists the module and uses the names (`configs/hipk-score.yaml`):

```yaml
plugins: [examples/hip-kernels/hip_kernels.py]
data:
  prompt_pools:
    - {path: /path/to/aorta/aorlta/examples/hip-kernels, source: standin, task: hipk}
rollout:
  compile_adapter: standin.hipcc     # a built-in part, mixed in by name
  parse_adapter: hipk.parse
  feedback_template: hipk.feedback
reward:
  profile: hipk-gpu
  reward_fn: hipk.v1
```

[ADAPTING.md](ADAPTING.md) has the full contract of each part. The rules that matter most:

- **Prompts.** `records(pool_dir, tokenizer, thinking)` returns `PromptRecord`s with `source="standin"` (a plugin's
  data is handled as public stand-in data: its text may appear in logs). `task.task_sha256` must change whenever what
  a check reads for that prompt changes: it is part of every cache key, and `hipk.run` finds the task by it.
  Training takes `rollout.P` different prompts per iteration, so the pool needs at least `P` of them; the landing
  check uses the first `sync.probes` (4) prompts' chats, which must share one system message.
- **A check never guesses.** Anything it cannot settle (no GPU, a tool missing, output it cannot read) is
  `unsettled`, never `clean`; only the check may blame the candidate for a timeout. `verdict()` in the example is
  the whole mapping, and `tests/examples/test_hip_kernels.py` tests every row of it on CPU.
- **The reward's parts.** `compute` returns `(reward, {"base", "broke_healthy", "length"})`, which add up to the
  reward, within [-1.55, 1.0]; an unscored sample is `(None, {"reason": ...})` and is never counted clean. `decide`
  answers early only when no pending leg could change the answer. The service recomputes and checks all of this,
  and a reward that breaks it is unscored and raises an alarm.
- **The check's GPUs** are its own setting (`hipk.run.gpus` in the profile): keep them off `rollout.engines`.
- **Precise tasks.** An iteration stops when more than `rollout.identical_groups_stop` (5% by default) of its groups
  have all `K` answers identical, which is how a broken sampler shows. A short kernel can have one obvious answer,
  and with 8 prompts one such group is 12.5%, so the example's configs set it to 0.5; the engines' start-up
  self-test still checks the sampler, on open-ended chats of its own.

## 1. Check the tasks

Each task in `tasks.json` comes with a correct reference kernel in `reference/<id>.hip`. Build every task's harness
with its reference kernel, on CPU:

```bash
python examples/hip-kernels/hip_kernels.py selfcheck --hipcc /opt/rocm/bin/hipcc --build-only
```

and run them on one GPU (one line per task, all `"pass": true`; on a cluster, inside a job):

```bash
python examples/hip-kernels/hip_kernels.py selfcheck --hipcc /opt/rocm/bin/hipcc --gpu 0
```

The harness is the candidate's source followed by a host `main` that fills the inputs from a fixed seed, runs the
kernel once with the launch the prompt states, and compares every output element with the task's `reference` loop
(host C++), within `atol + rtol * |want|`. Output elements start as a sentinel, so an element the kernel never
writes is a mismatch, and every array has a guard region past its end, so a write past the end is caught too.

## 2. Run the tests

```bash
OMP_NUM_THREADS=1 python -m pytest -q tests/examples/test_hip_kernels.py tests/test_plugins.py
```

They load the configs with the plugin, build the prompts and probes, run the parse and feedback, run `hipk.run`
against a stand-in `hipcc` (no GPU), and check the reward against the service's invariants for every combination of
leg outcomes. With hipcc installed, they also build every harness.

## 3. The loop on CPU, with fake components

`--components fake` loads the config and the plugin exactly as a real run does, then runs the loop with fake engines,
checks and trainer: it tells you the config is consistent and every name resolves, not that your task works (the
fakes use prompts and checks of their own).

```bash
export SCRATCH=/path/to/scratch
cat > "$SCRATCH/site.yaml" <<EOF
run: {root: $SCRATCH/runs}
model: {base_dir: $SCRATCH/base}
data: {prompt_pools: [$PWD/examples/hip-kernels]}
EOF
AORLTA_SITE=$SCRATCH/site.yaml python -m aorlta.orchestrator.run --config configs/hipk-score.yaml --components fake
```

`tests/orchestrator/test_plugin_wired.py` is the stronger CPU test of a plugin: the real loop, with a tiny model and
in-process engines, trains one iteration on a plugin's task.

## 4. Score the base model on a GPU node

A score run (`orchestrator.mode: score`) generates and checks with no training, so it is the first real test of a
task: does the reward come out, and does it separate good answers from bad ones? Set the site paths (the base
model's snapshot directory, a run root the docker daemon can write, this checkout's `examples/hip-kernels`) in a
site file and submit from `aorlta/`, with the config committed:

```bash
export AORLTA_SITE=/path/to/site.yaml AORLTA_VENV=/path/to/venv TS_SRC=/path/to/tokenspeed-3b684074
mkdir -p runs && CONFIG=configs/hipk-score.yaml sbatch jobs/aorlta-run.sbatch
```

This task needs none of the stand-in sanitizer tools (rocjitsu, aorta's sanitizer backends, the ASan runtime): only
the TokenSpeed image the job builds, whose hipcc builds and runs the kernels. The job prints `score/summary.txt` at
the end; `score/report.jsonl` has one row per prompt.

## 5. Train

`configs/hipk-train.yaml` is the same task as a 3-iteration training run (one LoRA rank beside engine 0). Submit it
the same way. Eight prompts are far too few to learn anything from: it is a template to copy, with your own pool.

## A second example: questions and answers, nothing compiled

`examples/qa/` is the shape of a task whose answer is not code: a question with a reference answer (a domain "expert"
model), or a tool's verdict. Its parts, in `qa_task.py`:

| part | what it does |
|---|---|
| prompts | `<pool>/qa.jsonl`, one `{"id", "question", "answer", "aliases"?}` per line; the model is told to end with `Answer: <answer>` |
| parse and feedback | a reply with an `Answer:` line is the answer; one without gets a repair turn that says so |
| compile | none: `rollout.compile_adapter: none`, so a parsed answer ends its chain |
| the check | one leg, `ref.answer`, on the CPU: the last `Answer:` line, normalised (case, spaces, quotes, a final full stop, digit commas), against the answer and its aliases; with a judge command in the profile's `qa.match.judge`, an answer that does not match is the judge's to decide, and no verdict leaves it unscored |
| the reward | `qa.v1`: +1.0 correct, -0.5 wrong, -1.0 no answer line, minus the length term |

It comes with two pools, each split into `train/` and `heldout/`: `gpu-facts/`, 24 short questions on GPU
programming, and `arith/` and `arith4/`, generated multiplications (`make_arith.py`), which VALIDATION.md trains on
to show that the loop learns. Its configs run on **one GPU**: one engine on GPU 0 and one trainer rank beside it.

```bash
OMP_NUM_THREADS=1 python -m pytest -q tests/examples/test_qa.py           # on CPU, including a wired training run
CONFIG=configs/qa-score.yaml sbatch jobs/aorlta-run.sbatch                # score the base model on the held-out set
CONFIG=configs/qa-train.yaml sbatch jobs/aorlta-run.sbatch                # train
```

Then export the trained checkpoint, score it on the held-out set, and compare the two with
`tools/eval_compare.py` (VALIDATION.md, "Does it learn?"). Facts the base model does not know cannot be taught this
way; what training on such a pool changes is how the model answers.

## Making it yours

- **Your prompts.** Change `tasks.json` and `USER`, or write `records()` for your own data format. Keep the prompt
  ids stable and unique, and hash into `task_sha256` everything your check reads.
- **Your answer format.** `HipkParse.check` returns `edits` (code to compile), `no_change`, or `rejected` with a
  message for a repair turn. A task whose answers are not compiled can register a compile adapter that accepts
  everything: each chain then ends after its first parsed answer (`rollout.max_repair_turns` still bounds the
  repair turns a rejected answer gets).
- **Your check.** Keep `run` short and total: build, run, read, map to a state, and turn every exception into
  `unsettled`. A slow check needs a pool timeout to match (`reward.timeouts`, named by the pool's `timeout`).
- **Your reward.** Keep the three parts and the range, so the service can check it; give every failure you can
  name its own value, so that a group of answers to one prompt gets different rewards (GRPO learns only from the
  difference).
- **The settings** are the run config's ([README](../README.md) and `src/aorlta/config.py`, which documents every
  key): `rollout.K` samples per prompt, `rollout.P` prompts per iteration, `rollout.temperature`,
  `rollout.max_repair_turns` (0 to 3), `rollout.max_new_tokens`, `orchestrator.min_scored_fraction` (the share of an
  iteration's samples that must be scored before it trains), `orchestrator.deadlines`, `trainer.lr`. The LoRA is
  r 16, alpha 32 on fixed target modules (`LORA_R`, `LORA_ALPHA` in `orchestrator/wiring.py`).
