# Agentic Testing with `aorta agent`

This guide explains how **agentic testing** works in AORTA: what the
`aorta agent` command does, whether it uses a real LLM, what happens under
the hood, and how to read the output.

For the high-level design rationale, see
[aorta-probe-agent.md](aorta-probe-agent.md). For probe-mode mechanics
(recipes, classifiers, artifacts), see
[probe-188/usage.md](../probe-188/usage.md).

---

## What is agentic testing here?

**Agentic testing** means a **closed loop** instead of a one-shot run:

1. Run your repro (opaque command) under probe.
2. Read structured results (verdict, detectors, capture fields).
3. **Decide what to try next** (label the failure, pick a mitigation).
4. Run again with an expanded mitigation axis.
5. Stop when the repro passes, the budget is exhausted, or there is nothing
   left to try.

Today, `aorta probe` does step 1–2 across a **fixed matrix** you write in
YAML. `aorta agent` automates steps 3–5 on top of the same engine.

The loop is **agentic** because it maintains state, makes sequential
decisions, and adapts the next experiment from prior results — even when no
external LLM is involved.

---

## Are we using an actual LLM?

**By default: no.**

| Setting | LLM used? | How decisions are made |
|---------|-----------|-------------------------|
| Default (`--llm-backend fake`) | **No** | Deterministic `FakeLLMProposer`: heuristics on detector IDs + round-robin through registered mitigations |
| `--llm-backend litellm` | **Yes** | LiteLLM calls your configured model; requires `pip install 'aorta[agent]'` and provider API keys |

The CLI default is **`fake`** so tests, CI, and local smoke runs work with
**zero API calls** and fully reproducible behavior.

### When an LLM *is* used (optional)

The LLM runs at **one stage only**: the **proposer step**, after probe has
already executed a cell and the **deterministic 5-tier classifier** has
written `result.json`.

```mermaid
sequenceDiagram
    participant CLI as aorta agent
    participant Loop as agent loop
    participant Probe as run_recipe / probe
    participant Classifier as 5-tier classifier
    participant Proposer as fake or LiteLLM

    CLI->>Loop: argv + ticket + policy
    Loop->>Probe: run none-none cell
    Probe->>Classifier: stdout/stderr/exit/hang signals
    Classifier->>Probe: verdict pass/fail + detectors
    Probe->>Loop: result.json
    alt non-baseline cell passed
        Loop->>CLI: outcome converged
    else still searching
        Loop->>Proposer: cell summaries + candidates + tried list
        Proposer->>Loop: category, hypothesis, next_mitigations, stop
        Loop->>Probe: grow mitigation_axis, run next cell
    end
```

**The LLM never:**

- Sets `pass` / `fail` (only the classifier does).
- Changes your repro command (argv stays fixed).
- Proposes raw shell or env outside the **mitigations registry**.

**The LLM only:**

- Labels the failure (`rccl_hang`, `illegal_mem`, …).
- Writes a short hypothesis string.
- Picks **registered mitigation names** from the candidate list.
- Says whether to stop searching.

Install optional LLM support:

```bash
pip install 'aorta[agent]'
export OPENAI_API_KEY=...   # or other provider LiteLLM supports
```

Then:

```bash
aorta agent --llm-backend litellm --llm-model gpt-4o-mini ...
```

---

## How is it agentic *without* an LLM?

The **`fake`** backend still implements a full agent loop:

| Agent property | Implementation |
|----------------|----------------|
| **Perception** | Reads `result.json` per cell: `verdict`, `failure_detectors_fired`, `capture` |
| **Memory** | `agent_log.jsonl` + on-disk probe cells; `wake()` resumes after crash |
| **Planning** | Infers category from detector IDs (e.g. `tier2:*` → `rccl_hang`); picks next untried mitigation from registry order |
| **Action** | Appends mitigation to axis, calls `run_recipe` with `flat_resume` |
| **Termination** | Stops on baseline pass, converged mitigation, budget, or exhausted candidates |
| **Guardrails** | `AgentPolicy`: max iterations, wall time, registry-only names, optional approval gate |

So “agentic” here means **autonomous search over a mitigation space**, not
“must call Claude/GPT.” The LLM is an **optional upgrade** for smarter
mitigation ordering and richer hypotheses — not a requirement.

---

## Quick start

```bash
# From repo root with src on PYTHONPATH, or after pip install -e .
PYTHONPATH=src aorta agent \
  --output ./agent_results \
  --ticket ROCM-EXAMPLE \
  -- \
  python3 my_repro.py --steps 100
```

Required: literal `--` before your command (same rule as `aorta probe`).

Useful flags:

| Flag | Purpose |
|------|---------|
| `--symptom "..."` | Hint for proposer (fake or LLM) |
| `--max-iterations N` | Cap mitigation proposals (default 8) |
| `--mitigation NAME` | Restrict search (repeatable) |
| `--mitigations-file sidecar.json` | Extra registered mitigations |
| `--llm-backend litellm` | Enable real LLM proposer |
| `--dry-run` | Plan cells without executing |
| `--bundle` | Run `aorta bundle` after loop (needs recipe redaction) |
| `-v` / `-vv` | Progress logging |

---

## Examples and expected output

### Example 1 — Healthy repro (baseline passes)

Command:

```bash
PYTHONPATH=src aorta agent \
  --output /tmp/agent_out \
  --ticket smoke-hello \
  -- \
  echo hello
```

**What happens under the hood:**

1. Agent builds a probe recipe with `mitigation_axis: [none]`.
2. `run_recipe` runs cell `none-none` → executes `echo hello`.
3. Classifier sees exit code 0 → `verdict: pass`.
4. Fake proposer sees baseline pass → `stop_reason: baseline_pass`.
5. Loop writes `agent_report.md` and stops (no mitigation search).

**Expected CLI output:**

```
Agent outcome: baseline_pass — Baseline passed — no mitigation search needed.
Wrote /tmp/agent_out/smoke-hello/agent_report.md
Baseline cell (none-none) passed. The repro succeeds without mitigations; no search was run.
```

**Key artifacts:**

```
/tmp/agent_out/smoke-hello/
  agent_log.jsonl
  agent_report.md
  none-none/trial_0/result.json   # verdict: pass
  matrix.json
  host_env.json
```

---

### Example 2 — Failing repro, fake proposer searches mitigations

Command:

```bash
PYTHONPATH=src aorta agent \
  --output /tmp/agent_out \
  --ticket smoke-fail \
  --max-iterations 3 \
  --mitigation none \
  --mitigation tf32_off \
  --mitigation xnack \
  -- \
  python3 -c 'import sys; sys.exit(1)'
```

**What happens under the hood:**

1. **Iteration 0:** Run `none-none` → exit 1 → `verdict: fail`,
   detectors include `tier1:exit_nonzero`.
2. **Proposer (fake):** category `launch_error` or `unknown`; proposes
   `tf32_off` (first untried candidate in allowlist order).
3. **Iteration 1:** Recipe axis `[none, tf32_off]`; `none-none` skipped
   (flat resume); run `tf32_off-none` → still fails.
4. **Proposer:** proposes `xnack`.
5. **Iteration 2:** Run `xnack-none` → if still fail and budget hit →
   `exhausted_candidates` or `policy_stop`.

If **`tf32_off-none` passes** (hypothetically):

```
Agent outcome: converged — Mitigation found — repro passes with a non-baseline cell.
Wrote /tmp/agent_out/smoke-fail/agent_report.md
Re-run the repro with mitigation `tf32_off` applied (see cell `tf32_off-none` probe.env or matrix).
```

**Sample `none-none/trial_0/result.json` (abbreviated):**

```json
{
  "verdict": "fail",
  "exit_code": 1,
  "cell_name": "none-none",
  "failure_detectors_fired": ["tier1:exit_nonzero"],
  "argv": ["python3", "-c", "import sys; sys.exit(1)"]
}
```

---

### Example 3 — Symptom hint + LLM backend

Command:

```bash
pip install 'aorta[agent]'
export OPENAI_API_KEY=sk-...

PYTHONPATH=src aorta agent \
  --output /tmp/agent_out \
  --ticket smoke-llm \
  --symptom "RCCL hang after checkpoint" \
  --llm-backend litellm \
  --llm-model gpt-4o-mini \
  --mitigation none \
  --mitigation nccl_launch_order_implicit \
  --mitigation tf32_off \
  -- \
  ./my_training_repro.sh
```

**What happens under the hood:**

1. Baseline cell runs and fails (typical for a real repro).
2. Classifier populates detectors (e.g. hang tier, stderr patterns).
3. **LiteLLM** receives JSON context: symptom, cell summaries, candidate
   mitigations, already-tried list.
4. Model returns structured JSON:
   `{category, hypothesis, next_mitigations[], confidence, stop}`.
5. `AgentPolicy.validate_step()` drops any name not in the registry.
6. Next cell runs with proposed mitigation env vars applied by probe.

The LLM may pick `nccl_launch_order_implicit` first because the symptom
mentions RCCL — unlike fake mode, which always takes the first untried name
in sorted allowlist order.

---

### Example 4 — Resume after interrupt

Re-run the **same** command with the same `--output` and `--ticket`:

```bash
PYTHONPATH=src aorta agent --output /tmp/agent_out --ticket smoke-fail -- ...
```

**Under the hood:**

- `wake()` reads `agent_log.jsonl` and existing cell directories.
- `run_recipe(..., resume_existing=True, layout="flat_resume")` skips a
  trial only when its own `trial_<n>/result.json` is complete (checked
  per trial via `aorta.probe.resume.is_trial_complete`), so a cell reruns
  the specific trials whose result is missing, incomplete, or corrupt —
  not just when `trial_0` is absent.
- Search continues from the last untried mitigation — no duplicate work.

---

## Outcome reference

| Outcome | Meaning | Typical next step |
|---------|---------|-------------------|
| `baseline_pass` | `none-none` passed | No mitigations needed |
| `converged` | Some `{mitigation}-none` passed | Ship that mitigation to customer / gate |
| `exhausted_candidates` | No mitigations left in allowlist/registry | Manual matrix or new sidecar mitigations |
| `agent_stop` | Proposer set `stop` (LLM or fake) | Read `agent_report.md` hypothesis |
| `approval_required` | Mitigation needs ack (`--require-approval`) | Operator approves, re-run |
| `walltime_exhausted` | `--max-walltime-sec` hit | Re-run same ticket to resume |
| `policy_stop` | e.g. `--max-iterations` hit | Increase budget or narrow allowlist |

---

## Under the hood — component map

```
aorta agent (CLI)
    └── run_agent_loop()          src/aorta/agent/loop.py
            ├── wake()            replay agent_log.jsonl + cell verdicts
            ├── build_probe_recipe_from_dict()
            ├── run_recipe()      same engine as aorta probe
            │       └── SubprocessWorkload + 5-tier classifier
            ├── _read_cell_summaries()  from trial_*/result.json
            ├── proposer.propose()       fake OR LiteLLM
            ├── AgentPolicy.validate_step()
            └── write_agent_report()
```

### Probe classifier (verdict source of truth)

Every trial’s `verdict` comes from `aorta.probe.classifier`, not from the
agent:

1. **Tier 1** — process exit code
2. **Tier 2** — hang monitor (stdout stall window)
3. **Tier 3** — kernel / GPU signals
4. **Tier 4** — built-in stderr regex catalogue
5. **Tier 5** — recipe `custom_patterns`

See [classifier.md](../probe-188/classifier.md).

### Mitigations registry

Proposed names must resolve via `aorta.registry.get_mitigation()`. Built-ins
include `none`, `tf32_off`, `xnack`, and many ROCm env-flag bundles in
`src/aorta/registry/mitigations.py`. Plugins register via the
`aorta.mitigations` entry-point group.

### State file (`agent_log.jsonl`)

Append-only JSON lines, e.g.:

```json
{"ts": "2026-06-04T12:00:00+00:00", "type": "session_start", "ticket": "smoke-hello", "llm_backend": "fake", ...}
{"ts": "...", "type": "llm_step", "category": "unknown", "hypothesis": "Baseline cell passed...", "stop": true, "stop_reason": "baseline_pass"}
{"ts": "...", "type": "search_stopped", "outcome": "baseline_pass", "stop_reason": "baseline_pass"}
```

### Report (`agent_report.md`)

One-page markdown: category, hypothesis, mitigation search table, evidence
chain (`capture` fields), recommended next action.

---

## Fake vs LiteLLM — decision guide

| Use **fake** (default) when… | Use **litellm** when… |
|------------------------------|------------------------|
| CI, unit tests, offline dev | You want symptom-aware mitigation ordering |
| Reproducible demo | Large registry — LLM can prioritize likely fixes |
| No API keys / air-gapped | Richer hypotheses in `agent_report.md` |

Both backends share the same loop, policy, probe engine, and artifact layout.

---

## Comparison: `aorta probe` vs `aorta agent`

| | `aorta probe` | `aorta agent` |
|---|---------------|---------------|
| Matrix | You write full YAML axes | Grows axis iteration by iteration |
| Who picks next mitigation | You | Proposer (fake or LLM) |
| Verdict | Classifier | Classifier (unchanged) |
| argv | Opaque, fixed | Opaque, fixed |
| Resume | Per ticket dir | Same + `agent_log.jsonl` |
| LLM | Never | Optional at propose step only |

For a known matrix (regression gate), use **`aorta probe`**. For exploratory
“find a mitigation that makes this pass,” use **`aorta agent`**.

---

## Troubleshooting

**Confusing `agent_stop` message** — upgrade to latest branch; baseline pass
should report `baseline_pass` with a clear success line.

**Search does nothing after first run** — ticket dir already has state; use a
fresh `--ticket` or inspect `agent_log.jsonl`.

**`ImportError: LiteLLM` / `does not provide the extra 'agent'`** — your venv
has an **old** `aorta` wheel (e.g. from PyPI) without the `[agent]` extra.
`PYTHONPATH=src` loads new agent code, but `litellm` was never installed.
From the **aorta repo root** on branch `feature/aorta-probe-agent`:

```bash
pip install -e '.[agent]'
# or, minimal fix:
pip install litellm
```

Then retry `--llm-backend litellm`.

**All mitigations fail** — expected for hard repros; outcome
`exhausted_candidates`; inspect `failure_detectors_fired` in
`agent_report.md` and consider manual probe matrix or new sidecar mitigations.

---

## Related docs

- [aorta-probe-agent.md](aorta-probe-agent.md) — design deck + build phases
- [probe-188/usage.md](../probe-188/usage.md) — probe recipes and artifacts
- [probe-188/classifier.md](../probe-188/classifier.md) — detector IDs
- [probe-188/bundle.md](../probe-188/bundle.md) — packaging for handoff
