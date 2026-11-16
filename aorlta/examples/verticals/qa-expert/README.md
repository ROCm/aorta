# qa-expert: a starter for a Q&A expert model

The model answers questions in your domain and is rewarded for matching a reference answer. Everything here is a
PLACEHOLDER: the questions are toy ones and the check is an exact match after normalising. It is a plugin
([docs/ADAPTING.md](../../../docs/ADAPTING.md)) and imports nothing from the other examples, so the directory can be
copied on its own; `examples/qa/` is a fuller example of the same shape, with a judge command.

| file | what | what to replace |
|---|---|---|
| `qa_expert.py` | the plugin: prompts, parse and feedback, the `ref.answer` check, the `qa-expert.v1` reward | `SYSTEM` (your domain), `normalise` and `ExpertMatch.run` (your match or judge), `ExpertV1.BASE` (your reward: its TODO lists the rules) |
| `qa-expert.yaml` | the check profile: one CPU leg | your check's settings, if it has any |
| `pool/{train,heldout}/questions.jsonl` | 4 + 4 toy questions, `{"id", "question", "answer", "aliases"?}` | your questions: many more than `rollout.P` |
| `score.yaml`, `train.yaml` | score the held-out set, train on the rest; one GPU | the site paths |

Score the base model first: if it already answers nearly everything, or nothing, there is nothing to learn.

From `aorlta/`, with the site paths set (`AORLTA_SITE`, docs/TUTORIAL.md section 4) and the config committed:

```bash
OMP_NUM_THREADS=1 python -m pytest -q tests/examples/test_verticals.py -k qa   # on CPU, a score and a training run
CONFIG=examples/verticals/qa-expert/score.yaml sbatch jobs/aorlta-run.sbatch    # score the base model
CONFIG=examples/verticals/qa-expert/train.yaml sbatch jobs/aorlta-run.sbatch    # train, then score the adapter
```
