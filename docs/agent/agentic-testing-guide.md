# Agentic Testing with `aorta agent mitigate`

This guide explains how **agentic testing** works in AORTA: what the
`aorta agent mitigate` command does, whether it uses a real LLM, what happens under
the hood, and how to read the output.

For the high-level design rationale, see
[aorta-probe-agent.md](aorta-probe-agent.md). For probe-mode mechanics
(recipes, classifiers, artifacts), see
[probe/usage.md](../probe/usage.md).

> **Command name.** `aorta agent` is now a namespace: `aorta agent <name>`
> dispatches to one of the agents registered under the `aorta.agents`
> entry-point group, and the closed-loop mitigation search described here is
> `aorta agent mitigate`. The old bare form, `aorta agent -- <command>`, still
> works for one release; it prints a deprecation notice on **stderr** (so
> stdout stays parseable) and runs the same command. See
> [`src/aorta/registry/README.md`](../../src/aorta/registry/README.md) for how
> to register another agent.

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
YAML. `aorta agent mitigate` automates steps 3–5 on top of the same engine.

The loop is **agentic** because it maintains state, makes sequential
decisions, and adapts the next experiment from prior results — even when no
external LLM is involved.

---

## Are we using an actual LLM?

**By default: no.**

There are five values for `--llm-backend`, and only three of them call a
remote model:

| `--llm-backend` | Remote call? | Needs | How decisions are made |
|-----------------|--------------|-------|-------------------------|
| `fake` *(default)* | **No** | nothing — base install | Deterministic `FakeLLMProposer`: heuristics on detector IDs + round-robin through registered mitigations |
| `local` | **No** | `amd-aorta[local-classifier]` + staged weights | `LocalClassifierProposer` runs a local calibrated encoder. Three typed questions, one forward pass, no tokens generated |
| `litellm` | **Yes** | `amd-aorta[chat-cli]`, or `amd-aorta[agent]` alone | Shared chat provider layer, falling back to a direct LiteLLM call when the chat extra is absent |
| `openai` | **Yes** | `amd-aorta[chat-cli]` | Shared chat provider layer |
| `vllm` | **Yes** | `amd-aorta[chat-cli]` | Shared chat provider layer, against a model you host |

The CLI default is **`fake`** so tests, CI, and local smoke runs work with
**zero API calls** and fully reproducible behavior. `fake` is also the only
backend that needs no extra at all: it imports nothing and reaches nothing,
which is what makes the test suite and `--dry-run` hermetic.

`litellm`, `openai` and `vllm` read the **shared chat provider layer** — the
same `~/.config/aorta/chat.toml` or `AORTA_CHAT_*` settings `aorta chat` uses,
so an endpoint, gateway header or auth scheme is configured once and both front
doors read it. `litellm` is the exception that also still works on an
`[agent]`-only install, because that combination shipped before `aorta.chat`
existed and must keep working.

> **Adding a backend?** `AGENT_LLM_BACKENDS` in `src/aorta/agent/llm.py` is the
> source of truth. The `click.Choice` list in `src/aorta/cli/agent_mitigate.py`
> is deliberately hard-coded rather than imported from it, because that
> decorator runs at import time and `aorta --help` must not pay for that
> module's imports — so the two can drift silently, and
> `tests/agent/test_llm_providers.py` asserts they have not. That test is the
> only thing standing between adding a backend to `make_proposer` and it being
> unreachable from the command line, since Click rejects an unlisted value
> before `make_proposer` is ever called.

### When a model *is* used (optional)

Whichever backend you pick, the model runs at **one stage only**: the
**proposer step**, after probe has already executed a cell and the
**deterministic 5-tier classifier** has written `result.json`.

```mermaid
sequenceDiagram
    participant CLI as aorta agent mitigate
    participant Loop as agent loop
    participant Probe as run_recipe / probe
    participant Classifier as 5-tier classifier
    participant Proposer as fake / local / LLM backend

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

**The proposer never:**

- Sets `pass` / `fail` (only the classifier does).
- Changes your repro command (argv stays fixed).
- Proposes raw shell or env outside the **mitigations registry**.

**The proposer only:**

- Labels the failure (`rccl_hang`, `illegal_mem`, …).
- Writes a short hypothesis string.
- Picks **registered mitigation names** from the candidate list.
- Says whether to stop searching.

Those boundaries hold for every backend. `AgentPolicy.validate_step()` re-checks
every proposed name against the registry regardless of which proposer produced
it — three implementations share one guard, so it has to.

Install the backend you want:

```bash
# Remote model through the shared chat provider settings (openai / vllm / litellm)
pip install 'amd-aorta[chat-cli]'

# litellm also still works on the older, chat-free extra
pip install 'amd-aorta[agent]'
export OPENAI_API_KEY=...   # or another provider LiteLLM supports

# Local calibrated encoder, no network at propose time
pip install 'amd-aorta[local-classifier]'
```

Then:

```bash
aorta agent mitigate --llm-backend litellm --llm-model gpt-4o-mini ...
aorta agent mitigate --llm-backend vllm ...
aorta agent mitigate --llm-backend local ...
```

For `openai` and `vllm`, `--llm-model` is optional: the chat profile already
names a model. For `local` it names a **checkpoint**, not a model on an
endpoint — see [Example 3b](#example-3b--local-encoder-backend-local).

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
“must call Claude/GPT.” A model is an **optional upgrade** for smarter
mitigation ordering and richer hypotheses — not a requirement.

`local` sits between the two. It is a learned model, so it orders mitigations by
something better than registry order, but it never generates text and never
leaves the machine, so it keeps the offline property `fake` has. What it cannot
do is write a hypothesis: `LocalClassifierProposer` templates that string the same way
`FakeLLMProposer` does, because a model that emits no tokens has nothing to say.
If the prose in `agent_report.md` is what you are after, you want one of the
three LLM backends.

---

## Quick start

```bash
# From repo root with src on PYTHONPATH, or after pip install -e .
PYTHONPATH=src aorta agent mitigate \
  --output ./agent_results \
  --ticket ROCM-EXAMPLE \
  -- \
  python3 my_repro.py --steps 100
```

Required: literal `--` before your command (same rule as `aorta probe`).

Useful flags:

| Flag | Purpose |
|------|---------|
| `--symptom "..."` | Hint for the proposer (any backend) |
| `--max-iterations N` | Cap mitigation proposals (default 8) |
| `--mitigation NAME` | Restrict search (repeatable) |
| `--mitigations-file sidecar.json` | Extra registered mitigations |
| `--llm-backend NAME` | Proposer backend: `fake` (default), `local`, `litellm`, `openai`, `vllm` |
| `--llm-model NAME` | Model for the selected backend; a **checkpoint name or local fine-tune directory** for `local`. Defaults to whatever the chat profile configures (`gpt-4o-mini` on the standalone `litellm` path, `laya-typed-decisions` for `local`) |
| `--dry-run` | Plan cells without executing |
| `--bundle` | Run `aorta bundle` after loop (needs recipe redaction) |
| `-v` / `-vv` | Progress logging |

---

## Examples and expected output

### Example 1 — Healthy repro (baseline passes)

Command:

```bash
PYTHONPATH=src aorta agent mitigate \
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
PYTHONPATH=src aorta agent mitigate \
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

### Example 3 — Symptom hint + remote LLM backend

Command:

```bash
pip install 'amd-aorta[agent]'
export OPENAI_API_KEY=sk-...

PYTHONPATH=src aorta agent mitigate \
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

Swap `--llm-backend litellm --llm-model gpt-4o-mini` for `--llm-backend openai`
or `--llm-backend vllm` to route the same call through the chat profile's
configured provider instead.

---

### Example 3b — Local encoder backend (`local`)

Same loop, same artifacts, no network call at the propose step:

```bash
pip install 'amd-aorta[local-classifier]'

PYTHONPATH=src aorta agent mitigate \
  --output /tmp/agent_out \
  --ticket smoke-local \
  --symptom "RCCL hang after checkpoint" \
  --llm-backend local \
  --mitigation none \
  --mitigation nccl_launch_order_implicit \
  --mitigation tf32_off \
  -- \
  ./my_training_repro.sh
```

**What happens under the hood:**

1. Baseline cell runs and fails; the classifier populates detectors, exactly as
   in Example 3.
2. `LocalClassifierProposer` builds **three typed questions over one state**: a choice over
   the mitigations that are actually left, a choice over `PROBE_CATEGORIES`, and
   a yes/no on whether to stop searching.
3. All three are answered in **one forward pass** — the encoder batches
   questions over a shared state — and each answer carries a probability rather
   than a number a prompt asked a model to invent.
4. `AgentPolicy.validate_step()` re-checks the proposed name, as it does for
   every backend.

Two properties worth knowing before you read the output:

- **It cannot name a mitigation that was not offered.** The candidate list *is*
  the answer space, so the filtering the LLM backends do after the fact is
  structural here.
- **The stop threshold is a policy choice, not a measurement.**
  `DEFAULT_CLASSIFIER_STOP_THRESHOLD` sits above the 0.5 midpoint because the two
  mistakes cost differently: a false stop ends an investigation and reports a
  category nobody went on to test, while a false continue costs one more probe
  cell, and `AgentPolicy` already bounds how many of those there can be. Nothing
  on this path applies the per-question-type temperature fit that would make the
  probabilities calibrated, so treat the threshold as an error preference made in
  the absence of a fit — not as a figure derived from one. See
  [`docs/local-classifier-packaging.md`](../local-classifier-packaging.md) on why a probability is a
  function of the checkpoint *and* the fit applied to it.

`--llm-model` selects the checkpoint; it defaults to the fine-tuned
`laya-typed-decisions` rather than the base checkpoint, whose published numbers
sit below a majority-class baseline on typed decisions. (The encoder behind
this backend is currently the Laya model, which is why the checkpoint names
and the error messages below mention it.)

---

### Example 4 — Resume after interrupt

Re-run the **same** command with the same `--output` and `--ticket`:

```bash
PYTHONPATH=src aorta agent mitigate --output /tmp/agent_out --ticket smoke-fail -- ...
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
| `agent_stop` | Proposer set `stop` (any backend) | Read `agent_report.md` hypothesis |
| `approval_required` | Mitigation needs ack (`--require-approval`) | Operator approves, re-run |
| `walltime_exhausted` | `--max-walltime-sec` hit | Re-run same ticket to resume |
| `policy_stop` | e.g. `--max-iterations` hit | Increase budget or narrow allowlist |

---

## Under the hood — component map

```
aorta agent mitigate (CLI)
    └── run_agent_loop()          src/aorta/agent/loop.py
            ├── wake()            replay agent_log.jsonl + cell verdicts
            ├── build_probe_recipe_from_dict()
            ├── run_recipe()      same engine as aorta probe
            │       └── SubprocessWorkload + 5-tier classifier
            ├── _read_cell_summaries()  from trial_*/result.json
            ├── proposer.propose()       fake | local | litellm/openai/vllm
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

See [classifier.md](../probe/classifier.md).

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

## Which backend — decision guide

| Pick… | When |
|-------|------|
| **`fake`** *(default)* | CI, unit tests, offline dev, reproducible demos. No extra, no keys, no weights, no network. The only backend a base install can run |
| **`local`** | You want symptom-aware ordering without a network call or an API key: an air-gapped node, a customer site, or a loop you do not want metered. Needs the `[local-classifier]` extra and staged weights |
| **`litellm`** | You already configure a provider through LiteLLM, or you are on an `[agent]`-only install where the chat extra is absent |
| **`openai`** | The chat profile already points at OpenAI and you want one place to configure it |
| **`vllm`** | You host the model yourself and want the prose without the third party |

All five share the same loop, policy, probe engine, and artifact layout, and all
five are subject to the same registry check on whatever they propose.

Two axes decide it in practice. **Does a remote call cost you anything** — money,
an egress rule, or a customer's data leaving their machine — rules out the three
LLM backends. **Do you need prose in `agent_report.md`** rules out `fake` and
`local`, both of which template the hypothesis rather than writing one.

---

## Comparison: `aorta probe` vs `aorta agent mitigate`

| | `aorta probe` | `aorta agent mitigate` |
|---|---------------|---------------|
| Matrix | You write full YAML axes | Grows axis iteration by iteration |
| Who picks next mitigation | You | Proposer (`fake`, `local`, or an LLM backend) |
| Verdict | Classifier | Classifier (unchanged) |
| argv | Opaque, fixed | Opaque, fixed |
| Resume | Per ticket dir | Same + `agent_log.jsonl` |
| LLM | Never | Optional at propose step only |

For a known matrix (regression gate), use **`aorta probe`**. For exploratory
“find a mitigation that makes this pass,” use **`aorta agent mitigate`**.

---

## Troubleshooting

**Confusing `agent_stop` message** — upgrade to latest branch; baseline pass
should report `baseline_pass` with a clear success line.

**Search does nothing after first run** — ticket dir already has state; use a
fresh `--ticket` or inspect `agent_log.jsonl`.

**`ImportError: LiteLLM` / `does not provide the extra 'agent'`** — your venv
has an **old** `aorta` wheel (e.g. from PyPI) without the `[agent]` extra.
`PYTHONPATH=src` loads new agent code, but `litellm` was never installed. From
the **aorta repo root**:

```bash
pip install -e '.[agent]'
# or, minimal fix:
pip install litellm
```

Then retry `--llm-backend litellm`.

**`--llm-backend=openai is configured through the shared chat provider layer`** —
`openai` and `vllm` read the chat profile, so they need `amd-aorta[chat-cli]`.
Install it, then configure the endpoint once in `~/.config/aorta/chat.toml` or
`AORTA_CHAT_*`; `aorta chat doctor` will tell you whether that configuration
resolves. `litellm` does not hit this, because it falls back to a direct call
when the chat extra is absent.

**`ClassifierUnavailableError: Laya is required for a real typed-decision predictor`** —
`--llm-backend=local` needs `pip install 'amd-aorta[local-classifier]'`. It is a separate
extra because it resolves torch, which nothing reachable from `[chat-cli]` is
allowed to do.

**`ClassifierUnavailableError: could not load the Laya checkpoint …`** — the extra is
installed but the weights are not. They download on first use, so this is the
ordinary no-egress failure as often as it is a bad checkpoint name. Stage the
checkpoint from a machine with egress, or pass `--llm-model` pointing at a local
fine-tune directory. Note that a `local` run loads weights on the **first
proposer call**, not at startup, so this surfaces after the baseline cell has
already run rather than immediately.

**A `local` run stops earlier or later than you expected** — the stop threshold
is a policy choice about which error to prefer, not a calibrated cutoff; see
[Example 3b](#example-3b--local-encoder-backend-local). Raise `--max-iterations`
if you want the search to keep going regardless.

**All mitigations fail** — expected for hard repros; outcome
`exhausted_candidates`; inspect `failure_detectors_fired` in
`agent_report.md` and consider manual probe matrix or new sidecar mitigations.

---

## Related docs

- [aorta-probe-agent.md](aorta-probe-agent.md) — design deck + build phases
- [probe/usage.md](../probe/usage.md) — probe recipes and artifacts
- [probe/classifier.md](../probe/classifier.md) — detector IDs
- [probe/bundle.md](../probe/bundle.md) — packaging for handoff
