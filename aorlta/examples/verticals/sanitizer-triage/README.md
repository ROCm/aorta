# sanitizer-triage: a starter for a sanitizer or NaN triage task

The model reads a tool's report and answers with a structured verdict: a JSON object with a category from a closed
list and a short rationale. Everything here is a PLACEHOLDER: the reports are toy text written for this starter, not
any tool's output, and the check looks only at the verdict's form. It is a plugin
([docs/ADAPTING.md](../../../docs/ADAPTING.md)) and imports nothing from the other examples, so the directory can be
copied on its own.

| file | what | what to replace |
|---|---|---|
| `sanitizer_triage.py` | the plugin: prompts, parse and feedback, the `ref.verdict` check, the `sanitizer-triage.v1` reward | `CATEGORIES` and `USER` (your verdicts), `TriageCheck.run` (compare with the report's `label`, or run the real tool: its TODO), `TriageV1.BASE` (your reward: its TODO lists the rules) |
| `sanitizer-triage.yaml` | the check profile: one CPU leg | your check's settings, if it has any |
| `pool/{train,heldout}/reports.jsonl` | 4 + 4 toy reports, `{"id", "report", "label"}` | your reports: many more than `rollout.P` |
| `score.yaml`, `train.yaml` | score the held-out set, train on the rest; one GPU | the site paths |

From `aorlta/`, with the site paths set (`AORLTA_SITE`, docs/TUTORIAL.md section 4) and the config committed:

```bash
OMP_NUM_THREADS=1 python -m pytest -q tests/examples/test_verticals.py -k triage   # on CPU, a score and a training run
CONFIG=examples/verticals/sanitizer-triage/score.yaml sbatch jobs/aorlta-run.sbatch  # score the base model
CONFIG=examples/verticals/sanitizer-triage/train.yaml sbatch jobs/aorlta-run.sbatch  # train, then score the adapter
```
