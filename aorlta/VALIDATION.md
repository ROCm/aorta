# Is it working? What AORLTA's validation shows, and what it does not

Each item says what is checked, the evidence, and how to reproduce it. Commands run from `aorlta/` in the
environment of [docs/GETTING-STARTED.md](docs/GETTING-STARTED.md); `<...>` are your paths. GPU runs were on one node
of 8 MI350X (gfx950) with Qwen/Qwen3.5-4B at revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.

## 1. The CPU test suite

About 1,900 tests over every component, run by CI on every pull request that touches `aorlta/` (the three
`pytest (AORLTA ..., py3.12)` jobs). What they prove, by directory of `tests/`:

| tests | what they prove |
|---|---|
| `test_config.py`, `test_schemas.py` | every config key's type and range; unknown or duplicate keys refused; the run plan is written first and once |
| `rollout/` | turn-0 and repair-turn token sequences match the chat template exactly; compile errors are parsed from real clang output (`data/clang/`); chains end for the documented reasons; a sampler that ignores seeds is caught |
| `reward/` | the reward service under a fake checker with injected latency, stalls, crashes and dirty controls: an unsettled check never yields a clean reward, controls bracket results, the watchdog restarts a stalled worker, the cache keys on everything a result depends on; `standin.v11` by property tests (decide never contradicts compute) |
| `trainer/` | the GRPO loss and its aggregations, packing equivalence with unpacked samples, the merged-weight forward against PEFT's, the per-step checks (non-finite or zero gradients, frozen weights that moved, ranks that disagree) |
| `sync/` | the LoRA merge and its rounding, weight-slot writing and manifests, reload, the landing check's verdicts, the start-up round trip |
| `orchestrator/` | the loop end to end with fakes; kill-and-resume restores the trained state bit for bit; the wired loop on CPU with a tiny Qwen3.5 and in-process engines trains, publishes and lands (`test_wiring_e2e.py`); score runs; plugin tasks through the real wiring (`test_plugin_wired.py`) |
| `test_plugins*.py`, `examples/` | the plugin registry, three task shapes registered from a package outside `aorlta/`, and both worked examples (their checks, rewards, and one CPU training iteration through the real wiring each) |
| `tools/` | the adapter merge against PEFT's merge, the checkpoint export (merged, it is exactly what the engines were served), the before/after comparison |

Reproduce (about 15 minutes on 8 cores):

```bash
export OMP_NUM_THREADS=1 MALLOC_MMAP_THRESHOLD_=65536 MALLOC_ARENA_MAX=2
python -m pytest -q -n 8 -m "not heavy and not cpu_run"
OMP_NUM_THREADS=2 python -m pytest -q -n 2 -m "heavy and not cpu_run"
OMP_NUM_THREADS=4 python -m pytest -q -m cpu_run
```

Expected: every test passes; a few skip and say what this machine lacks (hipcc, the tokenizer in `HF_HOME`, the
stand-in dataset, the stand-in check tools, a GPU).

## 2. Mutation checks

A test that cannot fail proves nothing, so the new code was mutated by hand and the tests rerun: each mutant must
make a test fail. For the plugin registry: dropping the registry lookup in `component_source`, in the check-adapter
factories, in the parse-adapter lookup, in the wiring's prompt pool, and the run plan's plugin record (5 of 5
caught); the plugin self-test chats (1 of 1); the checkpoint export's tensors and its manifest check (2 of 2). The
components merged before them recorded their own mutation runs in their pull requests. Reproduce one: make the edit,
run the named test file, see it fail, revert.

## 3. The checks every run makes on itself

These run inside every real run, so a run that completes has passed them; the run stops otherwise.

| check | what it refuses | evidence |
|---|---|---|
| engine start | an engine whose sampler, log-prob output, prefix cache or dtype read back wrong, or whose samples do not vary with the seed | every GPU run below started 4 engines (or 1) and passed |
| start-up round trip | weights that do not reach the engines: a synthetic perturbation must change every probe, restoring the base must bring it back bit for bit | PROVEN in every GPU run below |
| landing check | a weight update that did not land on every engine after each training step | LANDED on every engine, every iteration, in every training run below (66 updates) |
| trainer–engine gap gate | a sample whose trainer log-probs differ from the engine's by more than `trainer.max_generation_gap` is left out of the step; too many and the step is refused | p99 of the per-sample max \|Δ log-prob\|: 0.68–0.73 nats (stand-in, 4096-token replies), at most 0.19 (kernel example), 0.25–0.53 (arithmetic), against the 2.0 limit; no sample over it |
| step checks | a non-finite or zero gradient, a non-finite loss, a group trained twice, a LoRA tensor that did not move, a frozen weight that did | passed every step |
| scored fraction | training on an iteration whose checks did not settle for enough samples; an unscored sample is never counted clean | 100% scored in the example runs, 72–82% in the stand-in run (where ConSan often has nothing to watch, as documented) |

## 4. GPU runs

| run | config | commit | GPUs | result |
|---|---|---|---|---|
| stand-in training | `configs/dev-standin.yaml` | `21d7befb` | 4 engines, 1 rank | 3 iterations; 207, 184, 209 of 256 scored; every update landed |
| stand-in score | `configs/score-standin.yaml` | `21d7befb` | 4 engines, 4 checker GPUs | round trip PROVEN; 50 of 64 scored |
| kernel example, self-check | `examples/hip-kernels/hip_kernels.py selfcheck` | this branch | 1 at a time (GPUs 0, 4, 7) | all 8 reference kernels pass on each |
| kernel example, score | `configs/hipk-score.yaml` | this branch | 4 engines, 4 check GPUs | 64 of 64 scored, 56 correct; by task 8/8 on the six elementwise kernels, 5/8 transpose, 3/8 block sum |
| kernel example, train, export, score | `configs/hipk-train.yaml`, `tools/export_adapter.py`, a score run with `model.adapter_dir` | this branch | 4 engines, 1 rank | 3 iterations landed; the exported adapter merged, served and PROVEN; 59 of 64 correct on the same 8 prompts it trained on (base 56): that shows the pipeline, not learning |
| arithmetic, train | `configs/arith-train.yaml` | this branch | 1 GPU (engine and rank) | 20 iterations of 16 prompts x 8 samples, 13–49 s each; every update landed; correct on the training prompts 93.3% (mean of iterations 1–5) to 97.5% (16–20); mean reward 0.87 to 0.93 |
| arithmetic, held-out before/after | `configs/arith-score.yaml`, base and trained | this branch | 1 GPU | 64 held-out prompts x 4 samples: pass rate 94.9% (243/256) base, 98.4% (252/256) trained; mean reward 0.895 to 0.948; on the same prompt and seed, 12 samples went from wrong to right and 3 from right to wrong (sign test p = 0.035) |
| harder arithmetic, train | `configs/arith4-train.yaml` | this branch | 1 GPU | 40 iterations of 16 prompts x 8 samples, about 21 s each; every update landed; correct on the training prompts 74.8% (iterations 1–5) to 98.9% (36–40); mean reward 0.58 to 0.96 |
| harder arithmetic, held-out before/after | `configs/arith4-score.yaml`, base and trained | this branch | 1 GPU | 64 held-out prompts x 4 samples: pass rate **83.2% (213/256) base, 98.4% (252/256) trained**; mean reward 0.724 to 0.950; 30 prompts better, 1 worse; on the same prompt and seed, 42 samples went from wrong to right and 3 from right to wrong (sign test p < 1e-9) |

Reproduce a run: `CONFIG=configs/<run>.yaml sbatch jobs/aorlta-run.sbatch` with a site file (GETTING-STARTED.md,
"Set up the node"). The job's log ends with one line per iteration, or a score run's summary.

## 5. Does it learn? The before/after evaluation

The arithmetic example (`examples/qa/`, `arith4/` and `arith/`) asks for a product of two random 4-digit (or
3-digit) numbers and rewards an exact final answer: +1.0 correct, -0.5 wrong, -1.0 no `Answer:` line. Training
prompts and held-out prompts share no pair. To reproduce, train, export the last checkpoint, score the base and the
trained adapter on the held-out set under two run roots, and compare:

```bash
CONFIG=configs/arith4-train.yaml sbatch jobs/aorlta-run.sbatch
python tools/export_adapter.py --ckpt <run_dir>/ckpt/iter-000040 --out <adapter>
CONFIG=configs/arith4-score.yaml AORLTA_SITE=<site-base.yaml> sbatch jobs/aorlta-run.sbatch      # run.root: <base>
CONFIG=configs/arith4-score.yaml AORLTA_SITE=<site-trained.yaml> sbatch jobs/aorlta-run.sbatch   # adds model.adapter_dir
python tools/eval_compare.py --base <base>/standin/arith4-score --trained <trained>/standin/arith4-score \
    --tokenizer <model.base_dir> --out <report-dir>
```

Result (commit of this branch, Qwen3.5-4B, one MI350X GPU for the engine and the LoRA rank together, 40 iterations
of 16 prompts x 8 samples at temperature 1.0, about 21 s an iteration, 20 minutes of run time with setup):

| iterations | mean reward | correct (training prompts) | replies with no `Answer:` line |
|---|---|---|---|
| 1–5 | 0.58 | 74.8% | 29 of 640 |
| 6–10 | 0.73 | 84.1% | 12 of 640 |
| 11–20 | 0.89 | 94.5% | 16 of 1280 |
| 21–30 | 0.93 | 97.3% | 6 of 1280 |
| 31–40 | 0.95 | 98.4% | 2 of 1280 |

Held out (64 multiplications never trained on, 4 samples each at temperature 0.7), from `tools/eval_compare.py`:

| | base | trained |
|---|---|---|
| pass rate | 83.2% (213/256) | 98.4% (252/256) |
| mean reward | 0.724 | 0.950 |
| mean reply length | 412 tokens | 551 tokens |
| replies needing the repair turn for a missing `Answer:` line | 47 | 28 |

What changed is visible in the rollouts: on `3961 × 2885` the base model answered `Answer: 11429685` with no working
(wrong), and the trained one wrote out the partial products and their sum before answering 11,427,485 (right). The
policy learned to work the product out rather than guess it. The 3-digit set shows the same direction with less
room to move (94.9% to 98.4% held out; 12 samples wrong to right, 3 right to wrong).

What this shows and does not: the loop learns, the gain holds on prompts it never saw, and the checks that guard
each step (landing, gap, scored fraction) passed throughout. It is one run, one seed, on a toy task the base model
was already fair at; section 6 makes the same measurement on a real kernel task.

## 6. Learning on a real kernel task

The kernel example of section 4 has 8 tasks, far too few to learn from. This run used 93 HIP kernel-writing tasks in
eight families (elementwise, reduction, transpose, softmax, layernorm, scan, GEMM, gather/scatter), in float, double,
int and `__half`, with launches that include one thread per element, grid-stride loops with fewer threads than
elements, one block per row, 2-D tiles and a batch on `gridDim.z`. Each task gives a kernel's name, signature, launch
and what it computes, in the format of `examples/hip-kernels/tasks.json`. The model writes the kernel; the check
builds it, runs it on a GPU and compares every output element with a host reference loop, and also fails a write
past the end of an array or into an input. The reward is the `hipk.v1` table (`examples/hip-kernels/hip_kernels.py`):
+1.0 when the output matches, -0.5 wrong output, -0.6 crash or timeout, -0.8 bad signature, -0.9 compile error, -1.0
no code block, minus a small length penalty. Each task has one correct reference kernel, used only to prove the check
(93 of 93 pass); no kernel carries a planted bug.

The task set is not in this repository. It was generated by a script that writes each family's ops at several
shapes, with a reference kernel per task, and splits them by op and by shape: within each family the ops are divided
between 54 training and 39 held-out tasks, a held-out task uses no shape a training task of its family uses, and the
whole scan family is held out; the script refuses a split that breaks either rule. The check in
`examples/hip-kernels` handles float tasks only, so running this set needs that check extended to the other dtypes.
These runs used the check as it was before it compiled the candidate alone into a device code object, when the
candidate's host code was built into the harness. So every reply was searched afterwards: none of the trained model's
312 held-out replies prints the verdict line, defines `main` or runs host code of its own (`printf`, a constructor,
`exit`), and in the 5,120 training replies the only matches are in prose or a commented-out `main`.

To reproduce, put the tasks in `<pool>/train/tasks.json` and `<pool>/heldout/tasks.json`, copy `configs/hipk-train.yaml`
and `configs/hipk-score.yaml` with `data.prompt_pools[0].path` set to those directories (and, for training,
`rollout.P: 16`, `rollout.temperature: 1.0`, `orchestrator.iterations: 40`, `trainer.lr: 2.0e-5`), then as in section 5:

```bash
CONFIG=<kernel-train.yaml> sbatch jobs/aorlta-run.sbatch
python tools/export_adapter.py --ckpt <run_dir>/ckpt/iter-000040 --out <adapter>
CONFIG=<kernel-score.yaml> AORLTA_SITE=<site-base.yaml> sbatch jobs/aorlta-run.sbatch      # run.root: <base>
CONFIG=<kernel-score.yaml> AORLTA_SITE=<site-trained.yaml> sbatch jobs/aorlta-run.sbatch   # adds model.adapter_dir
python tools/eval_compare.py --base <base-run-dir> --trained <trained-run-dir> --tokenizer <model.base_dir> \
    --out <report-dir>
```

Result (Qwen3.5-4B, 8 MI350X GPUs: 4 engines, one LoRA rank, kernels run on GPUs 4–7; a fresh LoRA, 40 iterations of
16 prompts × 8 samples at temperature 1.0, lr 2e-5, up to 2 compile-repair turns; 52 minutes). Every update LANDED
on all four engines, no sample came near the gap bound (largest 1.24 against 2.0, off-policy fraction 0), and the
scored fraction never fell below 0.984. On the training prompts the share of correct kernels rose from 59% in
iterations 1–5 to 88% in iterations 31–40, and the mean reward from 0.30 to 0.78.

Held out (39 tasks never trained on, 4 samples × 2 passes each at temperature 0.7, up to 2048 new tokens), from
`tools/eval_compare.py`:

| | base | trained |
|---|---|---|
| pass rate | 47.4% (148/312) | 76.6% (239/312) |
| mean reward | 0.130 | 0.633 |
| compiled | 84.0% | 97.1% |
| mean reply length | 966 tokens | 223 tokens |

On the same task, pass and sample slot: 108 samples went from wrong to right and 17 from right to wrong (exact sign
test p = 2e-17). By task, 23 improved and 5 got worse (p = 0.0009). Part of the gain is shorter replies (the base hit
the 2048-token cap on 45 samples), but leaving out every slot where either model hit the cap, the pass rate still
rises from 56% to 77%, with 75 samples wrong to right against 17. The largest gains are on families the model saw
least or not at all:

- scan, never trained on: 2% → 73%;
- transpose: 31% → 97%;
- softmax: 25% → 63%;
- layernorm: 54% → 92%.

GEMM (43% → 43%) and gather/scatter (75% → 69%) did not improve. On a held-out running max, the base model spent its
budget in comments and then wrote a kernel that calls `hipMalloc` from device code; the trained model wrote a short
correct running-max loop.

What this shows and does not: the loop teaches a real, checkable task, and the gain holds on ops, shapes and a
family it never trained on. **But the reward checks correctness only, not speed**, and the model learned exactly
that. On tasks launched with one block per row, none of the trained model's 101 correct kernels has the block's
threads share the row: one thread does the whole row. The base model already did this (45 of 48), and the reward
cannot tell the difference. It also learned one wrong habit: it drops the grid-stride loop where a launch has fewer
threads than elements (one task, 5/8 → 0/8). It is one run with one seed. Writing *fast* kernels needs a speed term
in the reward, which is not done yet.

## Not yet shown

- **Fast kernels.** Section 6 shows a model learning to write *correct* kernels on held-out tasks; its reward has no
  speed term, the trained kernels leave most of a block idle, and GEMM and gather/scatter did not improve. Each
  learning result here is one run with one seed.
- **The 93-task set of section 6 is not in the repository**, and the kernel example's check handles float tasks only.
- **The examples' checks are cheap stand-ins** for the checks a real vertical would run (a full test suite, a
  sanitizer on a real workload, a simulator); section 6's check compares outputs and does not time them.
- **Only Qwen3.5-4B, only gfx950, only one node.** Other Qwen3.5 sizes should work and have not been run; Qwen3
  models (such as Qwen3-0.6B) need the changes listed under "Models" in GETTING-STARTED.md; fewer than 8 GPUs has run
  only in the 1-GPU arithmetic and Q&A layouts; MI300X and MI355X have not been run.
- **Long runs**: the longest training run here is 40 iterations. Stability over hundreds of iterations, and resuming
  one on real hardware beyond the kill-and-resume drill, are not shown.
- **Multi-turn episodes beyond compile-repair**, an evaluation exporter during a run, and runs outside Slurm.
