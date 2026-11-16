# isa-fuzzer: a starter for an ISA-fuzzer-style task

The model is given a goal and writes a short test program in a declared format; a check validates the program and, in
a real task, runs it through your own tool. Everything here is a PLACEHOLDER: `toy-asm` is a made-up format and the
goals are toy ones. It is a plugin ([docs/ADAPTING.md](../../../docs/ADAPTING.md)) and imports nothing from the other
examples, so the directory can be copied on its own.

| file | what | what to replace |
|---|---|---|
| `isa_fuzzer.py` | the plugin: prompts, parse and feedback, the `ref.program` check, the `isa-fuzzer.v1` reward | `USER` and `LINE_RE` (your format), `program_problem` (your check), `IsaV1.BASE` (your reward: its TODO lists the rules) |
| `isa-fuzzer.yaml` | the check profile: one CPU leg | `isa-fuzzer.check: {tool: [...]}` sends the programs that pass to your tool |
| `pool/{train,heldout}/tasks.jsonl` | 4 + 4 toy goals | your goals: many more than `rollout.P` |
| `score.yaml`, `train.yaml` | score the held-out set, train on the rest; one GPU | the site paths |

The tool reads `{"goal", "program"}` as JSON on stdin and prints `{"verdict": "pass"}` or `{"verdict": "fail"}`;
anything else leaves the sample unscored, never clean.

From `aorlta/`, with the site paths set (`AORLTA_SITE`, docs/TUTORIAL.md section 4) and the config committed:

```bash
OMP_NUM_THREADS=1 python -m pytest -q tests/examples/test_verticals.py -k isa   # on CPU, a score and a training run
CONFIG=examples/verticals/isa-fuzzer/score.yaml sbatch jobs/aorlta-run.sbatch    # score the base model
CONFIG=examples/verticals/isa-fuzzer/train.yaml sbatch jobs/aorlta-run.sbatch    # train, then score the adapter
```
