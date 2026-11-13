# qa: questions with reference answers, nothing compiled

The second worked example of a task as a plugin ([docs/TUTORIAL.md](../../docs/TUTORIAL.md)): a question, a reference
answer, and a reward for matching it, exactly or after normalising, with an optional judge command. It runs on one
GPU and needs no compile step and no sanitizer.

| file | what |
|---|---|
| `qa_task.py` | the plugin: prompts, parse and feedback, the `ref.answer` check, the `qa.v1` reward |
| `qa-match.yaml` | the check profile: one CPU leg |
| `gpu-facts/{train,heldout}/qa.jsonl` | 24 questions on GPU programming |
| `arith/`, `arith4/` | 3-digit and 4-digit multiplications, written by `make_arith.py` |

Configs: `configs/qa-{train,score}.yaml`, `configs/arith-{train,score}.yaml`, `configs/arith4-{train,score}.yaml`.
Tests: `tests/examples/test_qa.py`.
