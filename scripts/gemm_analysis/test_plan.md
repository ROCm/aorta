Goal:
Develop an automated regression test suite for `scripts/gemm_analysis` so that script changes do not produce wrong reports.

Code Structure:
- No emoji in comments, docstrings, logs, or print statements.
- When updating README or instructions, use concise, structured notes instead of chatty prose.
- Write professional, clean, and readable code.
- When updating code, do not create duplicate files; modify existing ones when appropriate.

Scope:
- Repository: `aorta`
- Directory under test: `scripts/gemm_analysis`
- Tests do not run `run_train_various_channels.sh`; they use pre-generated traces stored under tests/gemm_analysis/testdata/....
- Entrypoints to exercise:
  - `run_tracelens_analysis.sh`
  - `analyze_gemm_reports.py`
  - `plot_gemm_variance.py`
  - `enhance_gemm_variance_with_timestamps.py`
  - `gemm_report_with_collective_overlap.py`
  - `process_gpu_timeline.py`
  - `create_embeded_html_report.py`

Data:
- Use fixed synthetic data for two channel configurations, two threads, and 7 ranks.
- Synthetic inputs are assumed to be output of `run_train_various_channels.sh` which generates pytorch traces
sweep_YYYYMMDD_HHMMSS/
├── YYthread/
│   └── XXchannels/
│       ├── torch_profiler/rank*/trace.json
- Synthetic inputs live in `tests/gemm_analysis/testdata` (or: add scripts to generate them).

Baseline:
- Baseline branch: `origin/main` .
- Generate expected outputs once and store under `tests/gemm_analysis/expected_outputs/...`.

Task:
- Implement a test harness (pytest-based with supporting Python scripts) that:
  - Runs the above pipeline on the synthetic data.
  - Compares the generated outputs on the current branch to `expected_outputs`.
  - Compares outputs on the current branch to `expected_outputs`, enforcing:
    - Core numeric metrics and per-GEMM aggregates match within a small tolerance.
    - Required columns/fields are present and have the same semantics.
    - Cosmetic or additive changes (extra columns, reordered columns, HTML styling/layout, timestamps, build IDs) are allowed.

How to run:
- From repo root:
  - First time setup:
    - `python tests/gemm_analysis/generate_synthetic_data.py --output-dir tests/gemm_analysis/testdata`
    - `python tests/gemm_analysis/generate_baseline.py`
  - Run tests:
    - `pytest tests/gemm_analysis/ -v`

Environment:
- ./venvs/aorta/bin/activate

README.md updates
- When script usage or interfaces change, update `scripts/gemm_analysis/README.md` so users know how to run them.

Implementation Log
- Append all implementation entries to  `scripts/gemm_analysis/implementation_logs.md`
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
