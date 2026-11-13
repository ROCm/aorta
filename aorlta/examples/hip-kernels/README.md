# hip-kernels: a worked example of a task of your own

Write a HIP kernel from its description; the reward builds the answer, runs it on a GPU and compares every output
element with a host reference. It is a plugin (`hip_kernels.py`): nothing in `src/aorlta/` names it.
[docs/TUTORIAL.md](../../docs/TUTORIAL.md) walks through it.

| file | what |
|---|---|
| `hip_kernels.py` | the plugin: prompts, parse and feedback, the `ref.native` check, the `hipk.v1` reward, and a `selfcheck` command |
| `tasks.json` | the 8 tasks: signature, what to compute, the launch, the inputs, a host reference loop and a tolerance |
| `reference/<id>.hip` | a correct kernel for each task, used only by `selfcheck` |
| `hipk-gpu.yaml` | the check profile: parse gate, compile gate, `ref.native` on GPUs 4-7 |

The run configs are `configs/hipk-score.yaml` and `configs/hipk-train.yaml`; the tests are
`tests/examples/test_hip_kernels.py`.
