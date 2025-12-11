Goal:
Run the gemm train scripts on two nodes.
This is the script to run on single node
`aorta/scripts/gemm_analysis/run_train_various_channels.sh`


Code Structure:
- No emoji in comments, docstrings, logs, or print statements.
- When updating README or instructions, use concise, structured notes instead of chatty prose.
- Write professional, clean, and readable code.
- When updating code, do not create duplicate files; modify existing ones when appropriate.

Scope:
- Repository: `aorta`
- Directory under test: `scripts/gemm_analysis` and `scripts/multi_node`
- Entrypoints to exercise:
    - `master_launch.sh`


Tasks:
- Create `master_launch.sh`, `local_launch.sh` and if necessary `config_node.sh` and `set_env_variables.sh` to run `run_train_various_channels.sh` over multi nodes
    - Hint: 
        - use the examples from DLRM including DLRM_XXX files in `multi_node` folder
        - Assume we run `aorta/scripts/gemm_analysis/run_train_various_channels.sh` with one set of thread and channels. If it does not have this functionality add it.

- Tell me why you are making any changes; it should be an interative coding.

Implementation Log
- Append all implementation entries to  `scripts/multi_node/implementation_logs.logs`
Instructions for assistant:
- Every time you make meaningful progress on this task (design decision, code change, bug found/fixed, test run), append a new entry below.
- Never delete or rewrite earlier entries.
- Keep entries concise but specific enough that a human reviewer can follow the reasoning and verification steps.

Entry template:
- **Date**: YYYY-MM-DD
- **User / Agent**: <who made the change>
- **Branch / Commit**: <branch and/or short SHA, if applicable>
- **Summary**: 1–3 bullets of what you did.
- **Files touched**: key files/paths.
- **Design / Debugging notes**: decisions, hypotheses tested, failures.
- **Verification**: what you ran (commands/tests) and the outcome.
- **Open questions / TODOs**: remaining work or uncertainties.
