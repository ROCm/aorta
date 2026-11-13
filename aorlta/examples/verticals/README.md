# verticals: starter plugins, one per kind of task

Three skeletons to copy and fill in, each a plugin package on its own ([docs/ADAPTING.md](../../docs/ADAPTING.md)):
its module, check profile, a toy prompt pool split into `train/` and `heldout/`, and a score and a train config.
Nothing in one imports another, so copy a directory, rename its prefix, and replace what is marked PLACEHOLDER and
TODO. The pools, formats and reports are toy placeholders written for these starters; none of them is a real task.

| directory | the model gets | it produces | the placeholder check |
|---|---|---|---|
| `isa-fuzzer/` | a goal | a short test program in a declared toy format | the format's grammar, line count and required instructions; an optional external tool after that |
| `qa-expert/` | a question in your domain | an answer on an `Answer:` line | an exact match against the reference, after normalising |
| `sanitizer-triage/` | a tool's report | a JSON verdict: a category from a closed list and a rationale | the verdict's keys, category and rationale; not yet the reference label |

Each runs on one GPU, with `rollout.compile_adapter: none` and one CPU leg, and its reward keeps the service's rules:
in [-1.55, 1.0], parts exactly `base`, `broke_healthy` and `length`, a value of its own for each failure, and an
unsettled check unscored, never clean. `tests/examples/test_verticals.py` loads each config, tests each placeholder,
and scores and trains each on CPU through the real wiring.
