---
title: "aorta probe — feature rubric for ROCm/aorta#188"
issue: https://github.com/ROCm/aorta/issues/188
branch: users/vivekag/aorta-probe-188
base_commit: b79da71
base_pr: "#187 (build: make the AORTA CLI Buck2-buildable + audit deps -> cquery)"
implementer: amd-vivekag
author_of_issue: oyazdanb
planner: feature-rubric-planner v1 (codebase-anchored)
planner_run_date: 2026-05-25
---

# `aorta probe` — Implementation Rubric (Issue #188)

> Wrap-and-collect command for opaque user launch commands. Three sequential phases, each a separate PR off `main`.

---

## 0. Findings That May Revise the Issue

Read before scoring. These are gaps / contradictions between the issue text and the current codebase that the implementer (or issue author) must resolve in writing before merge of the Phase 1 PR.

### F1 — Output-tree layout conflicts with `run_recipe`'s contract

The issue specifies `Y/<ticket>/<cell>/trial_<n>/{stdout.log, stderr.log, result.json}` and "re-running with same `--output` skips completed cells". The current `aorta.triage.runner.run_recipe` calls `aorta.triage.output.resolve_run_dir(...)` (`src/aorta/triage/output.py:85-124`), which **always creates a fresh, timestamped, non-overwriteable leaf**: `<output_dir>/<ticket>/<workload>/<timestamp>[-N]/`, and explodes per-cell artifacts under `cells/<safe_slug(cell.name)>/<workload>/trial_<N>.json`. That layout (a) has no resume affordance and (b) names a `<workload>` and `<timestamp>` segment the issue's layout elides.

Two conformant resolutions:

1. **(Recommended)** Extend `resolve_run_dir` with a `layout: Literal["timestamped", "flat_resume"]` selector that probe-mode passes as `"flat_resume"`. In `flat_resume`, the leaf is `<output_dir>/<safe_slug(ticket)>/` (idempotent `mkdir(exist_ok=True)`), per-cell artifacts live at `<run_dir>/<safe_slug(cell.name)>/trial_<n>/`, and `SubprocessWorkload` writes its own `stdout.log` / `stderr.log` / `result.json` inside each `trial_<n>/`. This keeps `run_recipe` as the single matrix engine.
2. Fork the output writer entirely for probe-mode. **Rejected.** Violates the issue's "MUST call the same `run_recipe`" rule by introducing parallel I/O layer divergence.

→ Phase 1 must include a `layout=` parameter on `resolve_run_dir` and `run_recipe`, defaulting to `"timestamped"` so `aorta triage run` behaviour is byte-equivalent.

### F2 — Recipe schema explicitly forbids the probe-mode keys today

`src/aorta/triage/recipe.py:39-51` defines `_VALID_TOP_LEVEL` as a closed frozenset that **rejects** every probe-mode key in the issue (`mode`, `mitigation_axis`, `diagnostic_axis`, `step_time_regex`, `collect_paths`, `redaction`, `custom_patterns`). Loading the example probe-mode recipe from the issue today raises `RecipeSchemaError`. Phase 1 must extend `_VALID_TOP_LEVEL` and the parser to accept these keys when `mode == "probe"`, and reject them when `mode == "triage"` (default). It must also relax three triage-mode required keys (`workload`, `cells`, `steps`) to optional when `mode == "probe"` because probe-mode synthesises cells from `mitigation_axis × diagnostic_axis` and fixes `workload` internally to a reserved name.

### F3 — No `SubprocessWorkload` exists, and the `_aorta_` prefix is closed to user input

The issue mandates `SubprocessWorkload` in `aorta/workloads/_subprocess.py`, taking `argv` from `setup()`. `src/aorta/run/dispatcher.py:193` explicitly rejects any `config_overrides` key starting with `_aorta_` ("platform-supplied; not a user override"). Therefore argv cannot be smuggled in via `config_overrides`. The clean fix: add a typed field `subprocess_argv: tuple[str, ...] | None = None` to `RunRequest` (`src/aorta/run/dispatcher.py:39-122`), and have the dispatcher inject it as `config["_aorta_subprocess_argv"]` after merging `config_overrides` (same pattern as `config["_aorta_environment"]` at `dispatcher.py:346`). `SubprocessWorkload.setup()` reads the reserved key.

### F4 — `_subprocess` is not registered as an entry-point workload

`pyproject.toml:42-48` shows `[project.entry-points."aorta.workloads"]` is commented out. `aorta.run.discovery.get_workload_class` (`src/aorta/run/discovery.py:65-87`) raises `ValueError` for any name not found via `importlib.metadata.entry_points`. Phase 1 must register the workload:

```toml
[project.entry-points."aorta.workloads"]
_subprocess = "aorta.workloads._subprocess:SubprocessWorkload"
```

`_subprocess` is leading-underscored so it cannot collide with user-facing workload names and so it is visibly platform-internal in `aorta triage list-workloads` (when that command lands).

### F5 — `aorta bundle` truly does not exist

`gh pr list --state all --search 'bundle'` returns only unrelated PRs. No `src/aorta/cli/bundle.py`, no `src/aorta/bundle/` module, no design doc beyond an empty `docs/probe-188/` directory. **Phase 3 is genuinely blocked on a separate ticket landing `aorta bundle`.** Phase 1 + 2 must ship standalone without Phase 3 dependencies. The rubric reflects this by gating every Phase 3 criterion on the upstream bundle PR.

### F6 — `--env-passthrough-mode {inherit, file}` and the no-parse invariant collide

The issue says "everything after `--` is forwarded byte-for-byte; aorta never parses it" AND "the one real boundary is `docker run`, addressed with a two-mode passthrough (inherit | file)". These two together imply:

- `inherit` mode: aorta sets per-cell env vars in its own process and `exec`s the user's argv. Works for `torchrun`, `bash`, `python`, `buck2 run`. For `docker run`, **the user is responsible** for `-e KEY` lines in their own argv — aorta does not inject them.
- `file` mode: aorta writes a `KEY=VALUE\n` file at `${AORTA_ARTIFACT_DIR}/probe.env` and exports `AORTA_ENV_FILE=<path>` into the child process so the user can reference it themselves in their argv (`docker run --env-file $AORTA_ENV_FILE ...`).

This reading is the only one consistent with the no-parse invariant. The implementer should propose it in writing in the Phase 1 PR description and get the issue author to confirm before merge.

### F7 — `mitigation_axis` and `diagnostic_axis` resolve through B3 today; no schema change to the registry needed

`tf32_off`, `hsa_no_scratch_reclaim`, `fa_prefer_ck`, `hip_launch_blocking`, `none` — every name in the issue's example axes looks like a B3 `Mitigation`. The probe-mode recipe-builder can synthesise cells with `mitigations=(mitigation_axis_value, diagnostic_axis_value)`, name them `<m>-<d>`, and rely on the existing `_validate_no_mitigation_collisions` check (`src/aorta/triage/recipe.py:531-572`) to catch overlapping env-var keys between the two axes. The only registry-side work is **verifying that every name in the issue's example axes is registered** (Phase 1 docs deliverable).

---

## 1. Branching and PR Convention

The base branch `users/vivekag/aorta-probe-188` (currently at `b79da71`) is the **integration branch**. Each phase ships as its own PR off `main`, **not** stacked off the integration branch, so reviewers see one self-contained diff per phase and one phase can land independently of later phases stalling.

| Phase | Branch | PR Title |
|---|---|---|
| 1 | `users/vivekag/aorta-probe-188-phase-1` | `probe: MVP — aorta probe command + recipe schema + engine reuse (#188 phase 1)` |
| 2 | `users/vivekag/aorta-probe-188-phase-2` | `probe: built-in 5-tier classifier + sandboxed custom_patterns (#188 phase 2)` |
| 3 | `users/vivekag/aorta-probe-188-phase-3` | `probe: bundle integration + redaction + handout templates (#188 phase 3)` |

Each PR body references issue #188 and the prior phase PR (so the dependency chain is auditable in the GitHub UI).

---

# PHASE 1 — MVP

**Goal.** Ship `aorta probe` as a CLI command that loads a probe-mode recipe, synthesises mitigation × diagnostic cells, and runs them through `aorta.triage.runner.run_recipe` (no parallel runner). Per-trial artifacts land in `Y/<ticket>/<cell>/trial_<n>/{stdout.log, stderr.log, result.json}`. Re-runs with the same `--output` skip completed cells.

## 1.A Scope

**In:**
- New CLI subcommand `aorta probe` with all flags from the issue (`--recipe`, `--output`, `--ticket`, `--dry-run`, `--env-passthrough-mode`, trailing `--` argv).
- Recipe schema extension: `mode: probe` discriminator; new probe-mode keys (`mitigation_axis`, `diagnostic_axis`, `step_time_regex`, `collect_paths`, optional `confound`, `timeout_per_trial`). Strict unknown-key rejection retained.
- `SubprocessWorkload` (`src/aorta/workloads/_subprocess.py`) — invoked via the existing dispatcher, takes `argv` from a new `RunRequest.subprocess_argv` field.
- `RunRequest.subprocess_argv` field; dispatcher injection of `_aorta_subprocess_argv` into workload config (mirrors `_aorta_environment`).
- `aorta.triage.runner.run_recipe` extended with `layout: Literal["timestamped", "flat_resume"]` (default `"timestamped"`); `aorta.triage.output.resolve_run_dir` honours it.
- Probe-mode resume: existing `<cell>/trial_<n>/result.json` (well-formed JSON with non-empty `verdict` field) marks the trial complete and skipped.
- `--dry-run`: prints each cell's planned env + argv with no execution.
- Shared-engine unit test asserting `aorta probe` and `aorta triage run` both reach `run_recipe`.

**Out (Phase 1):**
- Any failure-tier classification beyond exit code (Tier 1 only, no patterns).
- `custom_patterns`, `redaction`, `condition` sandbox.
- `aorta bundle` integration.
- `aorta probe list-patterns`.
- `py-spy` integration.
- Hang detection (Tier 2), GPU/kernel detection (Tier 3), pattern library (Tier 4).
- Recipe templates for torchrun / buck2 (Phase 3 deliverable).
- Generic `--collect-path` glob/copy semantics beyond `${AORTA_ARTIFACT_DIR}/{stdout,stderr}.log` (Phase 2 wires the rest).

**Assumptions:**
- Every name in the issue's example axes is already registered in B3 (verify per F7).
- Resolution F1 (extend `resolve_run_dir`) is accepted by the issue author.
- Resolution F6 (env-file handoff) is accepted by the issue author.

## 1.B Functional Requirements (Weighted)

| # | Requirement | Weight | Verification |
|---|---|---|---|
| 1.1 | `aorta probe --help` exits 0 and shows `--recipe`, `--output`, `--ticket`, `--dry-run`, `--env-passthrough-mode` flags, plus the trailing-argv usage line. | 3 | Unit test `tests/probe/test_cli_parsing.py::test_help_lists_documented_flags` (Click `--help` invocation). |
| 1.2 | `aorta probe --recipe X --dry-run -- echo hi` exits 0, prints one line per (mitigation × diagnostic) cell with the planned env-var bundle and the literal argv `['echo', 'hi']`, and writes nothing to disk. | 5 | Unit test `tests/probe/test_dry_run.py` (capture stdout, snapshot-compare cell list, assert `tmp_path` empty post-run). |
| 1.3 | `aorta probe --recipe X --output Y --ticket T -- bash -c 'echo hi; exit 0'` runs every cell × trial, writes `Y/<safe_slug(T)>/<safe_slug(cell)>/trial_<n>/{stdout.log,stderr.log,result.json}`, exits 0. | 5 | Integration test `tests/probe/test_end_to_end.py::test_smoke_passes` with `tmp_path` as output. |
| 1.4 | Re-invoking the same command with the same `--output` and `--ticket` does NOT re-execute cells whose `result.json` exists, is valid JSON, and contains a non-empty `verdict` key. Partial / truncated `result.json` (empty file, syntactically invalid, missing `verdict`) IS re-executed. | 5 | Unit tests `tests/probe/test_resume.py::test_skips_completed_cell` and `::test_reruns_truncated_result_json` (write half a `{`, verify next invocation re-runs). |
| 1.5 | `aorta probe` and `aorta triage run` both reach `aorta.triage.runner.run_recipe` with a validated `Recipe`. Verified by a unit test that monkeypatches `run_recipe` to a `MagicMock` and invokes both CLIs. | 5 | Unit test `tests/probe/test_shared_engine.py::test_probe_and_triage_share_run_recipe` (mock pattern mirrors `tests/triage/test_runner_b1_api.py`). |
| 1.6 | Recipe loader rejects unknown top-level keys with `RecipeSchemaError` listing the unknown keys and the allowed set. Probe-mode keys (`mode`, `mitigation_axis`, `diagnostic_axis`, `step_time_regex`, `collect_paths`, `timeout_per_trial`) are rejected when `mode != "probe"` or absent. | 4 | Unit tests `tests/probe/test_recipe.py::test_rejects_unknown_top_level`, `::test_rejects_probe_keys_in_triage_mode`. |
| 1.7 | Probe-mode recipe loader synthesises cells as the cartesian product of `mitigation_axis × diagnostic_axis`, naming each `<m>-<d>`. Cell-name collisions (axis values that slug-collide) are rejected at load time. | 4 | Unit test `tests/probe/test_recipe.py::test_cell_synthesis_and_collision`. |
| 1.8 | `mitigation_axis` and `diagnostic_axis` items resolve through `aorta.registry.get_mitigation` at load time; unknown names raise `UnknownMitigationError` with the name. | 4 | Unit test `tests/probe/test_recipe.py::test_axis_unknown_name_rejected`. |
| 1.9 | `--env-passthrough-mode inherit` (default): the dispatcher sets per-cell mitigation + diagnostic env vars in `os.environ` before invoking the workload's `run()`; `SubprocessWorkload.run()` `exec`s the user argv via `subprocess.Popen(..., env=os.environ.copy())`. No `--env-file` is written. | 4 | Unit test `tests/probe/test_env_passthrough.py::test_inherit_mode_does_not_write_env_file` (monkeypatch `subprocess.Popen` with a recording fake, assert no extra file under `tmp_path`). |
| 1.10 | `--env-passthrough-mode file`: in addition to setting env vars in-process, write `<trial_dir>/probe.env` with `KEY=VALUE\n` lines (POSIX env-file format) and export `AORTA_ENV_FILE=<absolute path>` into the child process env. **Never modifies the user's argv.** File mode is `chmod 0600` at write time (defence against R5 leakage). | 4 | Unit test `tests/probe/test_env_passthrough.py::test_file_mode_writes_env_file_and_exports_pointer` AND `::test_env_file_is_0600`. |
| 1.11 | `RunRequest.subprocess_argv: tuple[str, ...] \| None = None` is the only legal channel for passing argv to `SubprocessWorkload`. Setting `config_overrides["_aorta_subprocess_argv"]` is rejected by the existing reserved-prefix check (`dispatcher.py:193`). The dispatcher copies `request.subprocess_argv` into `config["_aorta_subprocess_argv"]` after `config_overrides` is merged. | 4 | Unit tests `tests/run/test_dispatcher.py::test_subprocess_argv_injected_into_config` and `::test_user_supplied_aorta_subprocess_argv_rejected`. |
| 1.12 | `SubprocessWorkload.run()` populates a per-trial `result.json` containing at minimum: `verdict: "pass" \| "fail"`, `exit_code: int`, `walltime_sec: float`, `argv: list[str]`, `cell_name: str`, `trial_index: int`. Tier-1 verdict logic: `exit_code == 0` → pass, else → fail. | 4 | Unit test `tests/probe/test_subprocess_workload.py::test_pass_and_fail_minimum_result_shape`. |
| 1.13 | `aorta.triage.output.resolve_run_dir(layout="flat_resume")` creates `<output_dir>/<safe_slug(ticket)>/` (or `<output_dir>/_no_ticket_/` when `ticket` is `None`) with `mkdir(parents=True, exist_ok=True)`, no timestamp suffix, no `<workload>` segment. Default `layout="timestamped"` preserves today's behaviour byte-equivalently. | 5 | Unit tests `tests/triage/test_output_layout.py::test_flat_resume_layout` AND existing `test_resolve_run_dir_*` tests continue to pass unchanged. |
| 1.14 | When `--ticket` is omitted, the output directory uses `_no_ticket_` (existing constant `NO_TICKET_SLUG` from `aorta.triage.output`); Phase 3 will tighten this for `aorta bundle`, but Phase 1 accepts both. | 2 | Unit test `tests/probe/test_end_to_end.py::test_no_ticket_routed_to_no_ticket_slug`. |
| 1.15 | The Click handler in `src/aorta/cli/probe.py` is a thin shim (≤ 60 lines body, matches the `aorta run` discipline tested at `tests/run/test_cli_parsing.py::TestCliHandlerIsThinShell`). All orchestration lives in `aorta.probe.cli_helpers` or `aorta.triage.runner.run_recipe`. | 3 | Unit test `tests/probe/test_cli_parsing.py::test_handler_is_thin_shim`. |
| 1.16 | Recipe `schema_version: 1` is preserved (per open-question recommendation #4). Probe-mode is differentiated by `mode: probe`. A `mode: triage` recipe is byte-equivalent to today's recipes (back-compat: omitting `mode` defaults to `"triage"`). | 3 | Unit test `tests/probe/test_recipe.py::test_mode_defaults_to_triage_and_existing_recipes_load`. |
| 1.17 | Probe-mode entries in `aorta.workloads` entry-point group: `_subprocess = "aorta.workloads._subprocess:SubprocessWorkload"` is registered in `pyproject.toml` and resolves via `get_workload_class("_subprocess")`. | 3 | Unit test `tests/probe/test_subprocess_workload.py::test_resolved_via_entry_point`. |
| 1.18 | `aorta probe` exits non-zero (Click `ClickException`) when the recipe is invalid, when no axis items resolve, when `--env-passthrough-mode` is invalid, or when the trailing argv is empty (`--` followed by nothing). | 3 | Unit tests `tests/probe/test_cli_parsing.py::test_invalid_recipe_nonzero_exit`, `::test_empty_argv_nonzero_exit`. |
| 1.19 | Probe-mode rejects `condition`, `custom_patterns`, `redaction` keys at load time with a "deferred to Phase 2/3" error message that points to the phase. (Prevents users writing recipes that will silently no-op until later phases.) | 2 | Unit test `tests/probe/test_recipe.py::test_phase_2_3_keys_rejected_with_pointer`. |
| 1.20 | `tests/probe/` contains an `__init__.py` and is collected by `pytest` per the existing `[tool.pytest.ini_options]` (`testpaths = ["tests"]`). | 1 | `pytest tests/probe` discovers ≥ 1 test. |
|   | **Total weight** | **77** |  |

## 1.C File-by-File Deliverables

### New files

| Path | Purpose |
|---|---|
| `src/aorta/cli/probe.py` | Thin Click shim. Parses CLI, builds probe-mode `Recipe`, calls `run_recipe(..., layout="flat_resume")`. |
| `src/aorta/probe/__init__.py` | Package marker for probe helpers. |
| `src/aorta/probe/cli_helpers.py` | Pure parsers: `parse_env_passthrough_mode`, `split_argv_at_double_dash` (Click's own `--` handling is enough; helper is for the dry-run formatter). |
| `src/aorta/probe/recipe_builder.py` | `build_probe_recipe_from_dict(data, sidecar_files=None) -> Recipe`. Synthesises cells from `mitigation_axis × diagnostic_axis`, sets `workload="_subprocess"`, threads `step_time_regex` / `collect_paths` / `timeout_per_trial` onto a typed `ProbeExtras` block attached to the `Recipe` (via `workload_config`). |
| `src/aorta/probe/resume.py` | `is_trial_complete(trial_dir: Path) -> bool` — checks for valid `result.json` with non-empty `verdict`. Called from the runner per-cell. |
| `src/aorta/workloads/_subprocess.py` | `SubprocessWorkload(Workload)`. `setup()` reads `config["_aorta_subprocess_argv"]`; `run()` `subprocess.Popen`s, captures stdout/stderr to files in the trial dir, writes `result.json` with Tier-1 verdict. |
| `tests/probe/__init__.py` | Package marker. |
| `tests/probe/test_cli_parsing.py` | Click `--help`, invalid inputs, thin-shim assertion. |
| `tests/probe/test_dry_run.py` | `--dry-run` semantics. |
| `tests/probe/test_recipe.py` | Schema validation: probe-mode keys, axis-name resolution, cell synthesis, Phase-2/3 key rejection. |
| `tests/probe/test_subprocess_workload.py` | `SubprocessWorkload` unit tests (entry-point resolution, result.json shape, signal-as-fail). |
| `tests/probe/test_env_passthrough.py` | `inherit` vs `file` mode (no docker daemon needed; assert side effects on `os.environ` and `tmp_path`). |
| `tests/probe/test_resume.py` | Resume semantics: completed → skip, truncated → re-run. |
| `tests/probe/test_shared_engine.py` | Mocks `aorta.triage.runner.run_recipe`, invokes both `aorta probe` and `aorta triage run`, asserts both reach the same call site. |
| `tests/probe/test_end_to_end.py` | End-to-end smoke: synthetic axis values that resolve to no-op mitigations, real subprocess (`bash -c 'echo hi'`), assert artifact tree. |
| `tests/probe/fixtures/probe_minimal.yaml` | A 2-cell probe-mode recipe with `none` on both axes (minimum valid). |
| `tests/probe/fixtures/probe_with_phase_2_keys.yaml` | Has `custom_patterns:` block to assert Phase 1 rejection with pointer. |

### Modified files

| Path | Change |
|---|---|
| `src/aorta/cli/__init__.py` | Add `from aorta.cli import probe` and `main.add_command(probe.probe)`. |
| `src/aorta/triage/recipe.py` | Extend `_VALID_TOP_LEVEL` with probe-mode keys + `mode`. Branch `_build_recipe` on `data.get("mode", "triage")`. New helper `_build_probe_recipe(data, sidecar_files)` lives in `aorta.probe.recipe_builder` and is called from here. `Recipe` gains an optional `probe_extras: ProbeExtras \| None = None` frozen-dataclass field for `step_time_regex`, `collect_paths`, `timeout_per_trial`, `env_passthrough_mode` (set on the Recipe so the runner sees it without re-parsing). |
| `src/aorta/triage/output.py` | `resolve_run_dir(..., layout: Literal["timestamped", "flat_resume"] = "timestamped") -> Path`. `flat_resume` branch produces `<output_dir>/<safe_slug(ticket)>/` with `mkdir(parents=True, exist_ok=True)`. |
| `src/aorta/triage/runner.py` | `run_recipe(..., layout="timestamped", resume_existing=False)`. When `layout="flat_resume"`, per-cell artifacts land at `<run_dir>/<safe_slug(cell.name)>/trial_<n>/` (matching the dispatcher's `_aorta_log_prefix` writes). When `resume_existing=True`, `_run_one_cell` consults `aorta.probe.resume.is_trial_complete` before invoking `run_trials` for each trial; completed trials are loaded from disk and reported as skipped in the per-cell log line. |
| `src/aorta/run/dispatcher.py` | Add `subprocess_argv: tuple[str, ...] \| None = None` to `RunRequest`. After `config_overrides` is merged (line ~320), if `request.subprocess_argv is not None`, `config["_aorta_subprocess_argv"] = list(request.subprocess_argv)`. (`_aorta_*` prefix block at line 193 already prevents user-side injection of the same key.) |
| `pyproject.toml` | Uncomment and populate `[project.entry-points."aorta.workloads"]` with `_subprocess = "aorta.workloads._subprocess:SubprocessWorkload"`. |
| `BUCK` | No change required — top-level glob `src/aorta/**/*.py` already picks up new files. **Verify** with `buck2 build //:aorta_lib` post-change. |
| `recipes/README.md` | Add a section documenting probe-mode recipe shape and the `mode:` discriminator. |
| `recipes/example-probe-smoke.yaml` | New minimal probe-mode recipe ('echo hi'-friendly). Phase 3 ships the full handout templates; Phase 1 ships one smoke. |
| `README.md` | New "Probe-mode" subsection under the CLI overview, linking the recipe README and `docs/probe-188/usage.md`. |
| `docs/probe-188/usage.md` | New: command-line walkthrough including `--dry-run`, the two env-passthrough modes (with the F6 design rationale documented), the resume model, and the artifact tree. |

## 1.D Test Plan

- **Unit tests:** every row in §1.B with a `tests/probe/...` reference, plus the dispatcher additions in `tests/run/test_dispatcher.py` and the output-layout test in `tests/triage/test_output_layout.py`.
- **Integration test:** `tests/probe/test_end_to_end.py` invokes the real `aorta probe` Click command via `CliRunner`, with `tmp_path` as `--output`, a 2-cell recipe whose axes both contain `none`, and the trailing argv `['bash', '-c', 'echo hi']`. Asserts:
  - exit 0;
  - `<tmp_path>/<ticket>/<cell>/trial_0/{stdout.log,stderr.log,result.json}` exist for both cells;
  - `result.json::verdict == "pass"`.
- **Regression gates touched:** `tests/triage/test_runner_b1_api.py` (must still pass — runner contract unchanged for triage-mode), `tests/triage/test_recipe.py` (existing triage recipes still load), `tests/triage/test_output_layout.py` (default layout still timestamped).
- **Manual verification:** on a workstation with `aorta` installed editable, run:
  ```
  aorta probe --recipe recipes/example-probe-smoke.yaml --output /tmp/probe-188 --ticket SMOKE-1 -- bash -c 'echo hello'
  ```
  Verify the artifact tree manually; re-invoke and verify the second run is a no-op (look for "skipped" log lines).

## 1.E Documentation Deliverables

- `README.md` — new "Probe-mode" subsection.
- `docs/probe-188/usage.md` — full walkthrough (replaces the currently empty directory).
- `recipes/README.md` — probe-mode schema reference.
- `aorta probe --help` and `aorta probe --recipe <X> --help` Click strings.

## 1.F CI / Lint / Build Gates

| Gate | Command | Notes |
|---|---|---|
| Pytest | `pytest tests/` | New `tests/probe/` must collect and pass. |
| ruff | `ruff check src/aorta tests/probe` | Per `pyproject.toml:[tool.ruff]` — `E, F, W, I, N, UP, B, C4`. |
| black | `black --check src/aorta/cli/probe.py src/aorta/probe/ src/aorta/workloads/_subprocess.py tests/probe/` | Per `pyproject.toml:[tool.black]`. |
| isort | `isort --check src/aorta tests/probe` | Per `pyproject.toml:[tool.isort]`. |
| mypy | `mypy src/aorta/cli/probe.py src/aorta/probe src/aorta/workloads/_subprocess.py` | Per `pyproject.toml:[tool.mypy]`. |
| Pre-commit | `pre-commit run --all-files` | Three current hooks (trailing-whitespace, end-of-file-fixer, check-yaml). |
| Buck2 build | `buck2 build //:aorta_lib && buck2 build //:aorta` | The top-level `BUCK` globs `src/aorta/**/*.py`; verify new files do not introduce a missing third-party dep beyond `click`/`pyyaml`. **`subprocess` is stdlib — no BUCK dep change expected.** |

## 1.G Definition of Done (Phase 1)

- [ ] All 20 functional requirements in §1.B pass their named tests.
- [ ] `pytest`, `ruff`, `black`, `isort`, `mypy`, `pre-commit`, and both `buck2 build` targets succeed locally and in CI.
- [ ] No file under `src/aorta/triage/runner.py` or `src/aorta/triage/recipe.py` regressed against `tests/triage/`.
- [ ] `aorta probe --help` shows the documented flag set; `aorta triage run --help` is unchanged.
- [ ] `README.md`, `recipes/README.md`, `docs/probe-188/usage.md` updated.
- [ ] PR description quotes Findings F1, F3, F6 from this rubric and gets explicit `oyazdanb` approval on each before merge.
- [ ] PR title is `probe: MVP — aorta probe command + recipe schema + engine reuse (#188 phase 1)`.
- [ ] No new top-level dependencies in `pyproject.toml` or `requirements.txt`.
- [ ] `git diff main...HEAD` touches **zero files** under `src/aorta/ebpf/`, `src/aorta/hw_queue_eval/`, `src/aorta/profiling/`, `src/aorta/race/`, `src/aorta/report/`, `src/aorta/training/`, `notebooks/`, `experiments/`, `analysis/`, `misc/` (out-of-scope guardrail).
- [ ] No `subprocess.run` / `subprocess.Popen` import under `src/aorta/triage/` (preserves the #151 grep-test rule asserted by `tests/triage/test_runner_b1_api.py`).

## 1.H Do-Not-Do List (Phase 1)

- Do **not** add Tier 2–5 failure detection. `result.json::verdict` is set by Tier 1 only (exit code 0/non-zero).
- Do **not** implement `custom_patterns`, `condition`, or the sandbox. Recipes carrying these keys are rejected at load time with a "deferred to Phase 2" message (#1.B row 1.19).
- Do **not** ship `aorta probe list-patterns` or `aorta probe attach*` (out-of-scope per the issue).
- Do **not** add `py-spy`, `amd-smi`, or dmesg integration.
- Do **not** modify `aorta triage run`'s CLI surface or default behaviour. Every existing recipe must load and run byte-equivalently.
- Do **not** invent a parallel runner. Every cell goes through `aorta.triage.runner.run_recipe` (the shared-engine test enforces this).
- Do **not** rewrite the `<workload>` segment in the timestamped layout — only the `flat_resume` branch elides it.
- Do **not** redact anything (Phase 3 owns redaction).

---

# PHASE 2 — Built-in Five-Tier Classifier + `custom_patterns`

**Goal.** Make `SubprocessWorkload.run()` (and a per-trial post-process hook) populate `verdict`, `failure_detectors_fired[]`, `capture{}` based on the full 5-tier classifier. Land the sandboxed `condition` evaluator. Render `top failure` / `top warn` columns in `matrix.md`.

## 2.A Scope

**In:**
- New module `src/aorta/probe/classifier/` with one submodule per tier:
  - `tier1_process.py` — exit code, signal, timeout, coredump.
  - `tier2_hang.py` — stdout-silent + GPU-idle + /proc/<pid>/io two-of-three predicate, with `hang_grace_period_at_start` from the recipe.
  - `tier3_kernel.py` — `dmesg --since=<probe_start>` scan for amdgpu reset / SDMA timeout / VM_L2 fault / XGMI / PCIe AER; `amd-smi` VRAM growth and thermal-throttle counters. Fail-soft: missing `dmesg` permission / missing `amd-smi` binary → log once per run, continue with Tiers 1+2+4.
  - `tier4_patterns.py` — built-in pattern library (Python tracebacks, HIP/CUDA/ROCm errors, NCCL/RCCL errors, collective timeouts, NaN signatures). Versioned: module-level `BUILTIN_PATTERN_VERSION = "1"`.
  - `tier5_custom.py` — user `custom_patterns` runner.
- New module `src/aorta/probe/sandbox.py` — the `condition` evaluator (AST-walk whitelist, see §2.E for the exact rule set).
- `aorta probe list-patterns` subcommand: prints every built-in pattern ID, source, sample regex.
- `aorta probe list-patterns --version` prints `BUILTIN_PATTERN_VERSION` and the `aorta` package version on separate lines.
- `result.json` extended with `verdict`, `failure_detectors_fired: list[str]`, `capture: dict[str, str | float | int]`, `tier_durations_ms: dict[str, float]`.
- `matrix.md` gains `Top failure` and `Top warn` columns (detector IDs, with built-in and custom listed as peers).
- Recipe-load validation: every `custom_patterns[*].match.regex` is compile-validated; every `custom_patterns[*].match.condition` is sandbox-validated.

**Out (Phase 2):**
- `aorta bundle`.
- Redaction (Phase 3).
- Recipe templates (Phase 3).
- Network upload.

## 2.B Functional Requirements (Weighted)

| # | Requirement | Weight | Verification |
|---|---|---|---|
| 2.1 | Tier 1 fires for `exit_code != 0`, SIGSEGV (`-signal.SIGSEGV`), SIGABRT, SIGBUS, timeout (`Popen.communicate(timeout=...)` `TimeoutExpired` after `recipe.timeout_per_trial`), and the presence of a `core.*` file in the trial dir post-exit. Detector IDs: `tier1:exit_nonzero`, `tier1:sigsegv`, `tier1:sigabrt`, `tier1:sigbus`, `tier1:timeout`, `tier1:coredump`. | 5 | `tests/probe/classifier/test_tier1.py` covers each, with parameterised synthetic commands (`exit 1`, `bash -c 'kill -SEGV $$'`, etc.). |
| 2.2 | Tier 2 hang detector fires only when at least two of (stdout silent for `hang_window_sec`, GPU idle per `amd-smi`, `/proc/<pid>/io` rchar+wchar delta = 0) hold simultaneously, AND only after `hang_grace_period_at_start` (default 60s) has elapsed since trial start. Detector ID: `tier2:hang`. | 5 | `tests/probe/classifier/test_tier2.py` with a fake `amd-smi` shim (env var `AORTA_PROBE_AMDSMI_FAKE=idle`) and a synthetic child writing to stdout periodically. Includes a regression test asserting hang does NOT fire during grace. |
| 2.3 | Tier 3 dmesg detector fires for each documented amdgpu signature. When `dmesg` is missing/permission-denied, the runner logs `tier3 disabled: <reason>` exactly once for the whole `aorta probe` invocation and continues with 1+2+4 (no per-trial spam, no failure). Detector IDs: `tier3:amdgpu_reset`, `tier3:sdma_timeout`, `tier3:vm_l2_fault`, `tier3:xgmi_link_error`, `tier3:pcie_aer_fatal`, `tier3:vram_growth`, `tier3:thermal_throttle`. | 5 | `tests/probe/classifier/test_tier3.py` with a fake `dmesg` script on `PATH` (via `monkeypatch.setenv("PATH", ...)`) and a missing-`dmesg` test asserting the single warning and continued operation. |
| 2.4 | Tier 4 built-in library fires for: Python traceback (`^Traceback \(most recent call last\):` followed by an exception line), HIP error (`hipError_*`), CUDA error (`cudaError_*`), ROCm error (`Error code: \d+`), NCCL/RCCL error (`NCCL error|RCCL ERROR`), collective timeout (`Watchdog caught collective operation timeout`), NaN signature (`loss(?: is)? NaN|loss=nan`). Detector IDs prefixed `tier4:`. | 5 | `tests/probe/classifier/test_tier4.py` — one parameterised test per pattern, with a fixture file under `tests/probe/fixtures/tier4_logs/<id>.log`. |
| 2.5 | `aorta probe list-patterns` exits 0 and prints every Tier 4 detector ID + sample regex. `--version` prints `aorta probe pattern library v1 (aorta 0.2.0)` exactly. | 3 | `tests/probe/test_list_patterns.py`. |
| 2.6 | `custom_patterns[*].match.regex` is compile-validated at recipe load. Invalid regex → `RecipeSchemaError` with the pattern `id` and the `re.error` message. | 4 | `tests/probe/test_recipe.py::test_invalid_custom_regex_rejected`. |
| 2.7 | `custom_patterns[*].match.condition` is sandbox-validated at recipe load (§2.E). Whitelist violations → `RecipeSchemaError` listing the rejected AST node type and source line. | 5 | `tests/probe/test_sandbox.py::test_hostile_inputs_rejected` parameterised over the corpus in §2.E.4. |
| 2.8 | Verdict precedence is exactly: (a) any Tier 1–4 detector fires OR any `custom_patterns[*]` with `on_match: fail` fires → trial fails, `result.json::verdict == "fail"`; (b) `result.json::failure_detectors_fired` lists ALL that fired in encounter order (not just the first); (c) if any `custom_patterns[*]` has `required_for_pass: true` and none of them fired → `verdict == "fail"` and detector ID `meta:missing_pass_signal` appears in `failure_detectors_fired`; (d) otherwise → `verdict == "pass"`. `on_match: warn` populates a separate `warn_detectors_fired: list[str]` and does NOT change `verdict`; `on_match: info` only populates `capture`. | 5 | `tests/probe/classifier/test_verdict_precedence.py` — table-driven parameterised test covering each branch and the `required_for_pass` case. |
| 2.9 | `result.json` shape (post-Phase-2): `{"verdict": "pass"\|"fail", "exit_code": int, "walltime_sec": float, "peak_vram_mib": int\|null, "argv": list[str], "cell_name": str, "trial_index": int, "failure_detectors_fired": list[str], "warn_detectors_fired": list[str], "capture": dict[str, str\|float\|int], "tier_durations_ms": dict[str, float]}`. | 4 | `tests/probe/test_subprocess_workload.py::test_result_json_phase_2_shape` with a JSON schema (jsonschema dep optional — hand-rolled key/type assertion is acceptable). |
| 2.10 | `matrix.md` gains `Top failure` and `Top warn` columns immediately after `Failures`. Each shows the detector ID with the highest fire count across the cell's trials. Built-in (`tier4:*`) and custom IDs are listed as peers (no namespace prefix discrimination beyond their own ID prefix). Hidden when no cell has a non-empty `failure_detectors_fired` / `warn_detectors_fired`. | 3 | `tests/triage/test_output_layout.py::test_matrix_md_top_failure_warn_columns`. |
| 2.11 | When `dmesg` is unavailable, the `tier3 disabled` log line appears at most once per `aorta probe` invocation (not per cell, not per trial). | 2 | `tests/probe/classifier/test_tier3.py::test_dmesg_missing_log_once`. |
| 2.12 | The sandbox evaluator runs the `condition` only when the regex matches, and only with the variables documented in §2.E.1. The expression is `compile(..., mode="eval")` then `eval(code, restricted_globals, restricted_locals)`. Whitelist enforcement is at parse time (AST walk), not at eval time, so a hostile input cannot reach eval. | 5 | `tests/probe/test_sandbox.py::test_no_eval_reach_for_rejected_input`. |
| 2.13 | All Tier 2–4 detectors are unit-testable without a real GPU or real `dmesg`. Test fixtures use shim scripts (`tests/probe/fixtures/bin/dmesg`, `tests/probe/fixtures/bin/amd-smi`) prepended to `PATH` via `monkeypatch.setenv`. | 3 | Reviewer checklist item: every Phase 2 test runs under `pytest -m "not gpu and not rocm"` (existing markers in `pyproject.toml`). |
| 2.14 | Phase 1 contract (resume, dry-run, shared-engine) is unchanged. Re-running an already-completed cell still skips it; `--dry-run` still prints cells without executing. | 2 | Re-run `tests/probe/test_resume.py`, `test_dry_run.py`, `test_shared_engine.py` — must all pass with no source change. |
|   | **Total weight** | **56** |  |

## 2.C File-by-File Deliverables

### New files

| Path | Purpose |
|---|---|
| `src/aorta/probe/classifier/__init__.py` | Re-export `classify_trial(...)`. |
| `src/aorta/probe/classifier/tier1_process.py` | Exit/signal/timeout/coredump. |
| `src/aorta/probe/classifier/tier2_hang.py` | Two-of-three hang predicate with grace window. |
| `src/aorta/probe/classifier/tier3_kernel.py` | dmesg + amd-smi scanning. |
| `src/aorta/probe/classifier/tier4_patterns.py` | Built-in pattern library + `BUILTIN_PATTERN_VERSION`. |
| `src/aorta/probe/classifier/tier5_custom.py` | User custom_patterns runner. |
| `src/aorta/probe/classifier/verdict.py` | Precedence resolver (§2.B row 2.8). |
| `src/aorta/probe/sandbox.py` | AST-whitelist `condition` evaluator. |
| `src/aorta/cli/probe.py` (modified) | Add `list-patterns` subcommand to the existing `aorta probe` group. |
| `tests/probe/classifier/__init__.py` | Package marker. |
| `tests/probe/classifier/test_tier1.py` | Per-detector tests. |
| `tests/probe/classifier/test_tier2.py` | Hang predicate + grace. |
| `tests/probe/classifier/test_tier3.py` | dmesg/amd-smi shim tests, missing-binary path. |
| `tests/probe/classifier/test_tier4.py` | Pattern fixture tests. |
| `tests/probe/classifier/test_verdict_precedence.py` | Table-driven precedence cases. |
| `tests/probe/test_sandbox.py` | Sandbox happy-path + hostile-input fixture corpus (§2.E.4). |
| `tests/probe/test_list_patterns.py` | `list-patterns` subcommand. |
| `tests/probe/fixtures/tier4_logs/*.log` | One per built-in pattern. |
| `tests/probe/fixtures/bin/{dmesg,amd-smi}` | Test shims (executable bash scripts emitting fixture content). |
| `tests/probe/fixtures/custom_patterns_hostile.yaml` | A recipe with one `condition` per hostile-input class for the sandbox test. |

### Modified files

| Path | Change |
|---|---|
| `src/aorta/workloads/_subprocess.py` | `run()` invokes `classify_trial` post-exit and populates the Phase-2 `result.json` shape. |
| `src/aorta/triage/output.py` | `write_matrix_md` adds `Top failure` / `Top warn` columns gated on any cell having data. |
| `src/aorta/triage/matrix.py` | `CellStats` gains `top_failure_detector_id: str \| None` and `top_warn_detector_id: str \| None`; aggregation reads `result.json` for these. |
| `src/aorta/probe/recipe_builder.py` | Hand `custom_patterns` block through to the validator + the `ProbeExtras` attached to `Recipe`. |
| `src/aorta/triage/recipe.py` | Accept `custom_patterns`, `condition`, `hang_grace_period_at_start`, `hang_window_sec` keys when `mode == "probe"`. |
| `docs/probe-188/usage.md` | Section on the 5-tier classifier, the `condition` sandbox whitelist, and `list-patterns`. |
| `docs/probe-188/classifier.md` | New: detector ID reference. |

## 2.D Documentation Deliverables

- `docs/probe-188/classifier.md` — full detector ID reference.
- `docs/probe-188/sandbox.md` — `condition` whitelist + worked examples + the rejected-AST-types corpus.
- `docs/probe-188/usage.md` updated with `list-patterns` and the `custom_patterns` example.
- `aorta probe list-patterns --help` Click string.

## 2.E `condition` Sandbox Specification (P0 security gate)

### 2.E.1 — Variables in scope (read-only)

| Name | Type | Source |
|---|---|---|
| `capture` | `dict[str, str]` | Named capture groups from the matched regex. Indexed only with `capture['name']`. |
| `exit_code` | `int` | From Tier 1. |
| `walltime_sec` | `float` | From Tier 1. |
| `peak_vram_mib` | `int \| None` | From `amd-smi` post-exit (None if unavailable). Condition expressions that reference it when None MUST not blow up — `peak_vram_mib` is bound to `0` when actual is None to keep arithmetic total. |

### 2.E.2 — Callable whitelist

`float`, `int`, `len`, `math.isnan`, `math.isinf`. **Nothing else.** `math` is the only allowed module, and only its two named functions. No `getattr`, no `hasattr`, no `type`, no `isinstance`.

### 2.E.3 — Implementation (recommended)

```python
import ast

_ALLOWED_NODES = {
    ast.Expression, ast.BoolOp, ast.BinOp, ast.UnaryOp, ast.Compare,
    ast.Call, ast.Subscript, ast.Name, ast.Constant, ast.IfExp,
    ast.And, ast.Or, ast.Not, ast.Eq, ast.NotEq, ast.Lt, ast.LtE,
    ast.Gt, ast.GtE, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod,
    ast.FloorDiv, ast.Pow, ast.USub, ast.UAdd, ast.Load,
}
_ALLOWED_NAMES = {"capture", "exit_code", "walltime_sec", "peak_vram_mib", "math"}
_ALLOWED_CALLS = {"float", "int", "len", "math.isnan", "math.isinf"}

def validate_and_compile(expr: str) -> CodeType:
    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if type(node) not in _ALLOWED_NODES and not isinstance(node, ast.Attribute):
            raise SandboxError(f"forbidden AST node: {type(node).__name__}")
        if isinstance(node, ast.Attribute):
            # Only math.isnan / math.isinf attribute access is allowed.
            if not (isinstance(node.value, ast.Name) and node.value.id == "math"
                    and node.attr in {"isnan", "isinf"}):
                raise SandboxError(f"forbidden attribute access: {ast.unparse(node)}")
        if isinstance(node, ast.Name) and node.id not in _ALLOWED_NAMES:
            raise SandboxError(f"forbidden name: {node.id}")
        if isinstance(node, ast.Subscript):
            # capture[...] only; reject e.g. foo[0].
            if not (isinstance(node.value, ast.Name) and node.value.id == "capture"):
                raise SandboxError("subscript only allowed on capture[...]")
        if isinstance(node, ast.Call):
            name = ast.unparse(node.func)
            if name not in _ALLOWED_CALLS:
                raise SandboxError(f"forbidden call: {name}")
    return compile(tree, "<condition>", "eval")
```

Eval is `eval(code, {"__builtins__": {}, "math": math}, {"capture": ..., "exit_code": ..., "walltime_sec": ..., "peak_vram_mib": ...})`. Empty `__builtins__` neutralises `__import__`.

**Alternative considered:** `RestrictedPython`. Rejected — heavyweight dep, large attack surface, more permissive than the issue's whitelist by default. The hand-rolled walker is ~80 lines and entirely auditable.

### 2.E.4 — Hostile-input corpus (test fixture)

The Phase 2 PR ships `tests/probe/fixtures/conditions/hostile.txt`, one expression per line, all of which MUST be rejected at recipe load:

```
__import__('os').system('rm -rf /')
(0).__class__.__bases__[0].__subclasses__()
().__class__.__mro__[-1]
open('/etc/passwd').read()
exec("import os; os.system('id')")
eval("1+1")
lambda x: x
[x for x in range(10)]
{k: v for k, v in capture.items()}
capture.update({'pwn': '1'})
type(capture)
getattr(capture, 'pop')('eval_loss')
capture['x'].__class__
math.__loader__.load_module('os')
2 ** 1000000000  # length-restricted: reject expressions > 256 chars
```

Each line is a parameterised test case in `tests/probe/test_sandbox.py::test_hostile_inputs_rejected`. The expression length cap (256 chars, after stripping) is rubric-required and prevents resource exhaustion at `ast.parse` time.

### 2.E.5 — Regex DoS hardening

The Phase-1 compile-validation (#1.B 1.6) and Phase-2 regex validation (#2.B 2.6) must run `re.compile(pattern)` once at load. Phase 2 must additionally **cap each per-trial regex run** with `re.search(..., string[:MAX_LOG_BYTES])` where `MAX_LOG_BYTES = 10 * 1024 * 1024` (10 MiB) — stdout/stderr larger than this are scanned in 10MiB windows. This is the only mitigation for catastrophic backtracking that doesn't require swapping the regex engine; document it in `docs/probe-188/sandbox.md`.

## 2.F CI / Lint / Build Gates

Same as Phase 1 plus:

- `pytest tests/probe/classifier/ tests/probe/test_sandbox.py tests/probe/test_list_patterns.py` — must all pass.
- `mypy src/aorta/probe/classifier src/aorta/probe/sandbox.py`.
- No new third-party dependencies (Tier 3 `dmesg`/`amd-smi` are detect-and-skip per open question #3 recommendation).

## 2.G Definition of Done (Phase 2)

- [ ] All 14 functional requirements in §2.B pass.
- [ ] Sandbox hostile-input corpus is committed and all entries are rejected.
- [ ] Phase 1 regression gates pass unchanged.
- [ ] `aorta probe list-patterns` and `--version` work.
- [ ] PR title: `probe: built-in 5-tier classifier + sandboxed custom_patterns (#188 phase 2)`.
- [ ] PR description quotes the §2.E sandbox spec and gets a security-reviewer approval (Open Question #1 — must be resolved before merge of Phase 2, not just Phase 3).
- [ ] `docs/probe-188/classifier.md` and `docs/probe-188/sandbox.md` published.

## 2.H Do-Not-Do List (Phase 2)

- Do **not** ship `aorta bundle` or `redaction:` semantics.
- Do **not** make `dmesg` or `amd-smi` a hard dependency; both detect-and-skip.
- Do **not** swap the regex engine for `regex` or `re2` — keep `re` and use length-capped windows.
- Do **not** introduce a Python sandbox library (`RestrictedPython`, `pysandbox`, etc.). Hand-rolled AST walker only.
- Do **not** capture trial environment variables yet (Phase 3 redaction owns that to prevent secret leakage).
- Do **not** alter the `aorta triage run` matrix.md format for non-probe cells (the new columns are gated on probe-mode data presence).

---

# PHASE 3 — `aorta bundle` + Redaction + Handout Templates

**Goal.** Land redaction (P0 security gate), integrate the (separately-built) `aorta bundle` command, and ship generic recipe templates. **Gated on `aorta bundle` upstream PR landing.**

## 3.A Scope

**In:**
- New module `src/aorta/probe/redaction.py` — env-key glob scrubbing, path rewriting, IPv4/IPv6 rewriting; emits a per-file count manifest.
- Integration with `aorta bundle <output-dir>/<ticket>/`: bundle reads the recipe's `redaction:` block, applies it to every file in the trial tree before bundling, refuses without `--ticket`, supports `--review` interactive confirmation.
- `recipes/probe-template-torchrun.yaml`, `recipes/probe-template-buck2.yaml`, `recipes/probe-template-bash.yaml` — generic handout templates.
- `docs/probe-188/redaction.md`, `docs/probe-188/handout-templates.md`.

**Out:**
- Network upload (issue-out-of-scope).
- Auto-discovering the launch command (issue-out-of-scope).
- The implementation of `aorta bundle` itself — that's a separate ticket; this PR consumes its public API.

## 3.B Functional Requirements (Weighted)

| # | Requirement | Weight | Verification |
|---|---|---|---|
| 3.1 | `aorta bundle <output-dir>/<ticket>/` is called without `--ticket` (or `<ticket>` resolves to `_no_ticket_`) → exits non-zero with a `ClickException` pointing to the `--ticket` flag in `aorta probe`. | 4 | `tests/probe/test_bundle_integration.py::test_refuses_no_ticket`. |
| 3.2 | `aorta bundle --review <dir>` prints the manifest (cell list, per-file redaction counts, total size, redaction summary) and pauses for `[y/N]` confirmation; `n` aborts with exit 1, `y` proceeds. | 3 | `tests/probe/test_bundle_integration.py::test_review_pause_and_confirm` using `CliRunner(input="y\n")` / `input="n\n"`. |
| 3.3 | `scrub_env_keys: ["AWS_*", "*_TOKEN"]` removes matching keys from every captured-env block (per-trial `result.json::env`, host-env snapshot, per-environment snapshots) using `fnmatch.fnmatchcase`. Match is case-sensitive (env vars are case-sensitive on Linux). | 5 | `tests/probe/test_redaction.py::test_env_key_glob` with fixture env blocks. |
| 3.4 | `scrub_paths: true` rewrites every absolute filesystem path (regex `/(?:[A-Za-z0-9_.\-]+/)+[A-Za-z0-9_.\-]+`) found in `stdout.log`, `stderr.log`, and any string value in `result.json::capture` to `<PATH:N>` where `N` is the per-bundle deduplication index. A mapping from `<PATH:N>` → original path is NOT written (the whole point is to not leak the path). | 5 | `tests/probe/test_redaction.py::test_path_rewrite` with a fixture log containing 3 unique paths and 5 occurrences. |
| 3.5 | `scrub_ip_addresses: true` rewrites IPv4 (`\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b`) and IPv6 (RFC-5952-shape regex from `ipaddress`-grade validators) to `<IPV4:N>` / `<IPV6:N>`. | 4 | `tests/probe/test_redaction.py::test_ip_rewrite` with a fixture log mixing v4 and v6. |
| 3.6 | The bundle manifest (`bundle/manifest.json`) records, per scrubbed file: `{"path": str, "env_keys_removed": int, "paths_rewritten": int, "ips_rewritten": int, "bytes_in": int, "bytes_out": int}`. | 4 | `tests/probe/test_bundle_integration.py::test_manifest_records_per_file_counts`. |
| 3.7 | `recipes/probe-template-torchrun.yaml`, `recipes/probe-template-buck2.yaml`, `recipes/probe-template-bash.yaml` all pass `aorta probe --recipe <template> --dry-run -- echo hi` with exit 0. | 3 | `tests/probe/test_handout_templates.py` — parameterised over the three templates. |
| 3.8 | Phase 1 + Phase 2 contracts are unchanged. Specifically: `aorta probe` without a `redaction:` block is byte-equivalent to Phase 2 behaviour; `aorta bundle` is a separate command, not implicit in `aorta probe`. | 3 | All Phase 1/2 tests still pass. |
| 3.9 | Redaction is **applied to a copy**, not in-place. The bundle directory contains scrubbed copies; the original `<output-dir>/<ticket>/` is untouched. | 4 | `tests/probe/test_bundle_integration.py::test_originals_untouched`. |
| 3.10 | A hostile log designed to inflate path regex backtracking (e.g. 100kB of `/` characters) is processed in < 5 seconds for a 10MiB file (regex-DoS guard from §2.E.5 applies here too). | 3 | `tests/probe/test_redaction.py::test_redaction_dos_bound` with `pytest-timeout` marker. |
|   | **Total weight** | **38** |  |

## 3.C File-by-File Deliverables

### New files

| Path | Purpose |
|---|---|
| `src/aorta/probe/redaction.py` | All three scrubbers; per-file count return type. |
| `src/aorta/probe/bundle_hook.py` | Adapter between `aorta.probe.redaction` and `aorta.bundle` (the latter is a separate ticket). |
| `recipes/probe-template-torchrun.yaml` | torchrun-shaped probe recipe. |
| `recipes/probe-template-buck2.yaml` | `buck2 run`-shaped probe recipe. |
| `recipes/probe-template-bash.yaml` | Generic `bash launch.sh`-shaped probe recipe. |
| `tests/probe/test_redaction.py` | Three scrubbers + DoS bound. |
| `tests/probe/test_bundle_integration.py` | End-to-end bundle invocation with `--ticket`, `--review`, manifest check. |
| `tests/probe/test_handout_templates.py` | Each template `--dry-run`s. |
| `tests/probe/fixtures/redaction_input.log` | Fixture log with paths, IPs, env-var values. |
| `docs/probe-188/redaction.md` | Glob/path/IP rewrite semantics, security-reviewer sign-off note. |
| `docs/probe-188/handout-templates.md` | Per-template walkthrough. |

### Modified files

| Path | Change |
|---|---|
| `src/aorta/triage/recipe.py` | Accept `redaction:` block when `mode == "probe"`, with strict key validation (`scrub_env_keys: list[str]`, `scrub_paths: bool`, `scrub_ip_addresses: bool`). |
| `src/aorta/probe/recipe_builder.py` | Thread `redaction:` onto `ProbeExtras`. |
| `docs/probe-188/usage.md` | Add a "redaction + bundle" section. |
| `README.md` | Cross-link to `aorta bundle` once that PR has landed. |

## 3.D Documentation Deliverables

- `docs/probe-188/redaction.md` — exhaustive scrubber reference, security-review sign-off appended at the bottom.
- `docs/probe-188/handout-templates.md` — when to use each template.
- `recipes/README.md` — probe-template section.

## 3.E Definition of Done (Phase 3)

- [ ] All 10 functional requirements in §3.B pass.
- [ ] `aorta bundle` upstream PR has landed (Phase 3 cannot merge before).
- [ ] Security-reviewer approval recorded in the PR description (Open Question #1).
- [ ] Bundle manifest format documented in `docs/probe-188/redaction.md` and matches the implementation byte-for-byte.
- [ ] Phase 1 + 2 regression suite passes unchanged.
- [ ] PR title: `probe: bundle integration + redaction + handout templates (#188 phase 3)`.

## 3.F Do-Not-Do List (Phase 3)

- Do **not** reimplement `aorta bundle` in this PR. Consume the public API only.
- Do **not** make redaction the default; it activates only when `redaction:` appears in the recipe.
- Do **not** alter `aorta probe`'s exit semantics — `aorta bundle` is a separate user step.
- Do **not** ship network-upload functionality.
- Do **not** auto-detect "secret-looking" env vars; the glob list is recipe-authoritative.

---

# Cross-Phase Items

## X.1 Open Question Resolutions (with Phase Mapping)

| # | Open Question | Recommendation | Blocks |
|---|---|---|---|
| 1 | Security-review owner for redaction | **Block Phase 2 merge** on `oyazdanb` naming a security reviewer in the issue (the sandbox is in Phase 2; redaction is in Phase 3 but the sandbox is also a security artifact). Same reviewer ideally signs off on both. | Phase 2 + Phase 3 |
| 2 | `aorta bundle` status | Land Phase 1 and Phase 2 independently. Phase 3 cannot start without the bundle PR. Implementer files a follow-up issue `aorta bundle: scope + contract` at Phase 1 PR merge time so the team can prioritise. | Phase 3 only |
| 3 | `py-spy` dep model | **Vendor-detect-and-skip** (issue's own recommendation). `tier2_hang.py` calls `shutil.which("py-spy")` once per run; if absent, hang detection still fires but does not attach a stack dump. No `pyproject.toml` change. | Phase 2 |
| 4 | Recipe `schema_version` | **Keep `1`, differentiate by `mode:`** (issue's own recommendation). Implemented per F2 and FR 1.16. | Phase 1 |
| 5 | Tier 4 pattern library ownership | **Recommend** the AORTA team (specifically the issue's CODEOWNERS, currently `@mycpuorg`) owns version bumps; new patterns require a PR that bumps `BUILTIN_PATTERN_VERSION`, adds a fixture log under `tests/probe/fixtures/tier4_logs/`, and updates `docs/probe-188/classifier.md`. **Phase 2 PR codifies this in `docs/probe-188/classifier.md::ownership`.** | Phase 2 (process doc); does not block code merge |

## X.2 Risk Register

| # | Risk | Mitigation |
|---|---|---|
| R1 | **Engine drift** — Phase 2 needs new per-trial output shape; tempting to add a probe-only runner. | Phase 1 FR 1.5 (`tests/probe/test_shared_engine.py`) and a code-review checklist item explicitly forbidding any new file under `src/aorta/probe/` whose name contains `runner` or `dispatcher`. CI gate: `tests/probe/test_shared_engine.py` is in the required-pass set on every probe-touching PR. |
| R2 | **Regex DoS in `custom_patterns`** — user supplies `^(a+)+$` against a 100MB log. | Compile-validation at load (rejects unparseable regex); per-scan window cap (10MiB per `re.search`); per-trial wall-clock timeout from the recipe; documented in `docs/probe-188/sandbox.md`. |
| R3 | **dmesg / amd-smi permission denied** — Tier 3 needs root or `CAP_SYSLOG`; common in container deployments. | Detect-and-skip (Phase 2 FR 2.3 + 2.11); single warning per invocation; Tiers 1+2+4 continue. Documented in `docs/probe-188/classifier.md`. |
| R4 | **Sandbox bypass** — a clever `condition` reaches `__import__` or attribute walking. | AST-walk whitelist (§2.E.3) rejects at parse time; empty `__builtins__` at eval time as defence-in-depth; hostile-input corpus (§2.E.4) is the regression suite; security-reviewer sign-off required for Phase 2 merge (Open Question #1). |
| R5 | **Env-passthrough leaking secrets** — `--env-passthrough-mode file` writes `KEY=VALUE\n` lines to disk, including the user's `*_TOKEN` env vars. | Phase 1 documents the hazard in `docs/probe-188/usage.md`; the `<trial_dir>/probe.env` file is `chmod 0600` at write time (FR 1.10); Phase 3 redaction strips matching keys from the bundle. |

## X.3 Scoring

Per-phase pass thresholds, with explicit hard-gate vs soft-criterion separation.

### Phase 1 (total 77 points)

- **Hard gates (45 pts; ALL must pass)**: FR 1.1, 1.3, 1.4, 1.5, 1.6, 1.9, 1.10, 1.11, 1.12, 1.13. A miss on any of these is a blocking failure.
- **Quality criteria (32 pts)**: every other FR row.
- **Threshold to merge: ≥ 70 / 77 AND every hard gate passes AND every item in §1.G Definition of Done is checked.**
- **Below 70 but every hard gate passes:** PR can land with explicit follow-up issues filed for each missing FR; reviewer sign-off required.
- **Any hard-gate failure:** blocking; no waiver.

### Phase 2 (total 56 points)

- **Hard gates (28 pts; ALL must pass)**: FR 2.1, 2.7, 2.8, 2.9, 2.12. Plus the security-reviewer sign-off (Open Question #1) is a binary blocker.
- **Quality criteria (28 pts)**: every other FR row.
- **Threshold to merge: ≥ 50 / 56 AND every hard gate passes AND security sign-off recorded.**

### Phase 3 (total 38 points)

- **Hard gates (22 pts; ALL must pass)**: FR 3.1, 3.3, 3.4, 3.5, 3.6, 3.9. Plus `aorta bundle` upstream PR landed.
- **Quality criteria (16 pts)**: every other FR row.
- **Threshold to merge: ≥ 34 / 38 AND every hard gate passes AND `aorta bundle` is on `main`.**

### Failure-class definitions

- **Blocking failure (`fail:blocker`):** any hard gate, any sandbox/redaction security check, any documented security-reviewer requirement.
- **Material failure (`fail:material`):** any quality-criterion miss totalling > 10% of the phase's quality points.
- **Nit (`nit`):** lint/style/doc-wording miss with no functional effect; reviewer may waive in the PR thread.

## X.4 Code-Review Checklist (paste into PR description, per phase)

- [ ] No new file under `src/aorta/probe/` whose name contains `runner` or `dispatcher` (engine-reuse gate, all phases).
- [ ] No new `subprocess` import under `src/aorta/triage/` (preserves #151 grep-test).
- [ ] Every recipe-schema addition is reflected in `_VALID_TOP_LEVEL` / `_VALID_CELL_KEYS` AND has a unit test asserting unknown-key rejection.
- [ ] Every reserved `_aorta_*` key the dispatcher injects is documented in the `RunRequest` docstring.
- [ ] Every new third-party dependency is justified (Phase 2 + 3 expect NONE).
- [ ] Phase 2: hostile-input corpus is fully red on a deliberately-buggy sandbox (mutation test).
- [ ] Phase 3: bundle redaction is verified against a fixture log with secrets the test asserts are absent from the bundled output.

---

# Files Read to Anchor This Rubric

Every file path cited in the rubric was read or grep'd; the audit list:

| Path | Read for |
|---|---|
| `src/aorta/cli/__init__.py` | CLI entry-point registration pattern. |
| `src/aorta/cli/triage.py` | `aorta triage run` shape, exception-bridging convention, `ClickException` discipline. |
| `src/aorta/cli/run.py` | `aorta run` "thin-shell" handler contract (anchor for FR 1.15). |
| `src/aorta/triage/runner.py` | `run_recipe` signature, output-dir layout, per-cell exception handling, dry-run early-return, preflight validation, `_aorta_log_prefix` semantics. |
| `src/aorta/triage/recipe.py` | `_VALID_TOP_LEVEL`, `_RESERVED_WORKLOAD_CONFIG_PREFIX`, `_build_recipe`, `Recipe` dataclass, `_validate_no_mitigation_collisions`. |
| `src/aorta/triage/output.py` | `resolve_run_dir`, `safe_slug`, `NO_TICKET_SLUG`, `write_matrix_md` table-builder. |
| `src/aorta/workloads/__init__.py` | Workload package shape. |
| `src/aorta/workloads/_base.py` | `Workload` ABC, `WorkloadResult`, lifecycle, `launch_mode`. |
| `src/aorta/run/dispatcher.py` | `RunRequest` dataclass, `_aorta_*` reserved-prefix rejection (line 193), `_aorta_environment` / `_aorta_save_logs` / `_aorta_log_prefix` injection pattern (line 346, 441, 452), `save_logs` per-trial stdout/stderr capture. |
| `src/aorta/run/discovery.py` | Workload discovery via `aorta.workloads` entry-point group. |
| `src/aorta/run/results.py` | `TrialResult` shape (anchor for the probe-mode `result.json` divergence note). |
| `src/aorta/run/collectors.py` | `KNOWN_RECIPES` (collector recipe validation). |
| `src/aorta/registry/__init__.py` | Public registry API. |
| `src/aorta/registry/types.py` | `Mitigation`, `Environment` dataclasses. |
| `src/aorta/instrumentation/environment.py` (header) | A1 env-probe contract (fail-soft, schema 1.1+). |
| `tests/triage/test_runner_b1_api.py` (header) | Existing mock-`run_trials` pattern that probe-mode's shared-engine test mirrors. |
| `tests/run/test_cli_parsing.py` (header) | Thin-shell-handler test pattern (anchor for FR 1.15). |
| `tests/conftest.py` | Shared fixtures (none relevant to probe). |
| `recipes/example-fsdp-smoke.yaml` | Existing recipe shape (triage-mode) for back-compat assertion. |
| `pyproject.toml` | Project metadata, optional-deps, `[project.entry-points."aorta.workloads"]` (commented out), `[project.scripts] aorta=aorta.cli:main`, lint/format config. |
| `.pre-commit-config.yaml` | Active hooks (trailing-whitespace, end-of-file-fixer, check-yaml). |
| `BUCK` | Top-level `python_library` + `python_binary` definitions; `src/aorta/**/*.py` glob (confirms no per-file BUCK changes needed for new modules). |
| `CODEOWNERS` | `@mycpuorg` reviewer routing (only one owner; sandbox/redaction PRs need an additional explicit security reviewer per Open Question #1). |
| `docs/probe-188/` | Empty directory confirms no prior probe scaffolding. |
| `gh issue view 188 --repo ROCm/aorta` | Live issue text (matches the summary in the prompt; no drift). |
| `gh pr list --state all --search 'probe'` | No prior probe scaffolding PR. |
| `gh pr list --state all --search 'bundle'` | Confirms no `aorta bundle` PR — Phase 3 blocker is real (F5). |
| `git status` / `git log -1` | Branch `users/vivekag/aorta-probe-188` at `b79da71`. |
