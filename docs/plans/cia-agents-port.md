---
title: "Cluster Intelligence Agents — port into ROCm/aorta"
branch: feat/aorta-agent
base_commit: ca0e61f
base_pr: "#422 (feat/aorta-chat — aorta_llm merged as src/aorta/chat/)"
source: AMD-ROCm-Internal/aorta_llm @ aorta_agent
planner_run_date: 2026-09-01
---

# Cluster Intelligence Agents — port into `ROCm/aorta`

> Bring Launch, Watch and Autopsy into the repo beside `aorta chat`, so a pasted
> kernel or training script is diagnosed by the tool most likely to find the
> cause, with the agents called directly rather than spawned.

Target behaviour, end to end:

1. The user pastes a kernel, some assembly, or a training script.
2. An LLM reads the prompt, extracts the code, and ranks the diagnostic tools by
   what evidence each returns.
3. The chosen tool runs *that* code through the AORTA CLI on a GPU node.
4. Watch polls the log the run produces and alerts when confident.
5. Autopsy classifies the bundle and cites the evidence.
6. While this happens the UI shows each step; the answer is three parts — the
   bug, how it was found and with which tool, and the fix.

---

## 0. Findings that may revise the request

Read before implementing. Each is a gap between the request and this codebase.

### F1 — `aorta.agent` already owns an autopsy taxonomy

`src/aorta/agent/llm.py:20` defines `AUTOPSY_CATEGORIES` with eight members.
The CIA taxonomy (`schemas/taxonomy.v0.1.json` in the source repo) has eleven:
the same eight, plus `gpu_race`, `numeric_silent` and `tooling_gap`.

The two loops are **not** duplicates and should both survive:

| | `aorta.agent` | CIA Autopsy |
| --- | --- | --- |
| input | symptom + probe cell summaries + registered mitigations | bundle of adapter artifacts |
| output | category + `next_mitigations` + `stop` | category + cited rationale + `next_probe` |
| purpose | search for a mitigation that fixes it | explain what caused it |

**Resolution.** Extend `AUTOPSY_CATEGORIES` with the three new members and have
CIA's Autopsy import it. Do not port `taxonomy.v0.1.json`; a second category
list in the same package is the thing to avoid. The probe agent gains three
categories it will simply never propose, which is harmless.

### F2 — DSPy is a second LLM configuration surface

`src/aorta/agent/llm.py:356` (`ChatProviderProposer`) records a locked decision:
the shared chat provider layer is where LLM configuration lives, read from
`~/.config/aorta/chat.toml` or `AORTA_CHAT_*`, and both `aorta chat` and
`aorta agent` use it. Watch and Autopsy currently reach an LLM through DSPy with
their own `LITELLM_API_BASE` / `LITELLM_API_KEY` / `LITELLM_MODEL`.

That surface has already failed once in the source repo: when the agents moved
from subprocesses to in-process calls, nothing set `LITELLM_API_BASE`, both fell
back to `localhost:4000`, and Autopsy — which swallows a router failure and
classifies from adapters alone — kept returning plausible verdicts with its LLM
reasoning silently switched off.

**Resolution, deliberate.** Port DSPy as-is for this PR and converge afterwards.
Rewriting both agents onto the provider layer is a larger change than the port
itself, and doing it blind would land two unverified things at once.

**Debt this creates, to be tracked, not forgotten:**

- a second place to configure a model, contradicting the locked decision;
- DSPy and litellm as dependencies of the new extra;
- the failure above can recur, because it is silent by construction.

Mitigation until convergence: Autopsy must **log** when its router is
unavailable rather than only falling through to the adapters, so the degraded
mode is visible. That is a small change and belongs in this PR.

### F3 — The repo is scrubbed; the source is not

`docs/` and `src/` contain no internal hostnames or paths (commits `029c749`,
`10491d8`). The CIA source hardcodes throughout:

- absolute paths under one user's home for the jobs root, the sanitizer build
  and the proxy's env file;
- a specific login/compute node naming scheme;
- proxy discovery by Slurm job name;
- a partition name that has since changed and now rejects the account.

**Resolution.** Everything above becomes settings on the existing pydantic
`Settings` (`src/aorta/chat/config.py:107`), with neutral defaults, reached
through `AORTA_CHAT_*` or the TOML profile like every other knob. Nothing lands
with a site-specific default.

### F4 — Launch assumes Slurm

`submit_sbatch` is the only launcher. A public repo will see users with no
scheduler at all.

**Resolution for this PR.** Keep Slurm as the only backend but put it behind one
seam (`launch(command, ...) -> JobHandle`) so a local backend is a new
implementation rather than a refactor. Do not build the local backend yet —
one implementation is not enough to generalise from.

### F5 — Chat is optional; the agents should not be

Core `aorta` has two dependencies. `chat-cli` adds eleven and `chat-ui` adds
Chainlit. Launch, Watch and Autopsy are useful with no chatbot present — from
the CLI, from CI, from a script.

**Resolution.** The agents live at `src/aorta/cia/` and depend on nothing from
`chat`. `chat` depends on *them*. Progress reaches the UI through a callback the
caller supplies, so the agents stay headless and the tokens still appear.

### F6 — `aorta.chat` must contain no Click

`src/aorta/cli/chat.py` is the only Click module for chat, enforced by
`tests/cli/test_chat_boundaries.py`, and chat imports lazily because the CLI
imports command modules eagerly while chat is an optional extra.

**Resolution.** New CLI surface goes in `src/aorta/cli/`, imports stay inside
callbacks, and the boundary test is extended to cover `aorta.cia`.

---

## 1. What lands where

| source (`aorta_llm`) | destination | note |
| --- | --- | --- |
| `deploy/` | `src/aorta/cia/launch/` | Launch; sbatch behind one seam (F4) |
| `watchdog/` | `src/aorta/cia/watch/` | Watch; poll loop, log finder, bundle writer |
| `autopsy/` | `src/aorta/cia/autopsy/` | Autopsy; adapters, router, orchestrator |
| `llm/config.py` | `src/aorta/cia/llm.py` | DSPy config, kept under protest (F2) |
| `schemas/taxonomy*.json` | — | dropped; use `AUTOPSY_CATEGORIES` (F1) |
| `src/cia.py` | `src/aorta/cia/triage.py` | the Launch→Watch→Autopsy driver |
| `src/tools/capabilities.py` | `src/aorta/chat/tools/capabilities.py` | tool catalogue the selector ranks |
| `src/tools/cluster.py` | `src/aorta/chat/tools/cluster.py` | the tools themselves |
| `src/tools/{asm,kernel}_harness.py` | `src/aorta/chat/tools/harness/` | staging pasted source |
| `src/graph/nodes.py::selector_node` | `src/aorta/chat/graph/nodes.py` | already has a graph to extend |

Not ported: the canned demo tools (`run_nan_demo`, `run_waitcheck_demo`,
`run_gpu_triage`, `run_nan_fix_check`, `run_waitcheck_via_cia`). They run fixed
built-in workloads and ignore what the user pasted, which is the opposite of the
target behaviour. Measured in the source repo: given a pasted RMSNorm with a
NaN, the selector chose `run_nan_demo`, which then ran a *different*, hardcoded
RMSNorm. A canned tool is worse than an absent one, because the model picks it
and silently substitutes the demo.

Their replacement is one general tool, `triage_workload(source=... | command=...)`,
over the driver's existing arbitrary-command path.

## 2. Tool selection, without a routing table

The selector is one LLM call at temperature 0 over a catalogue of plain
descriptions — what each tool does and what evidence it returns — asking only
whether that evidence would answer this problem. No symptom keywords, no
category lookup.

Two properties to preserve in the port:

- **Descriptions carry no routing.** Across the seven diagnostic tools in the
  source repo the words *race*, *NaN*, *hazard* and *deadlock* appear once in
  total. `run_nan_demo` never says "NaN"; it says "until a computed value stops
  being finite". The model gets from a user's "loss goes to NaN" to that tool on
  meaning alone.
- **Selection is advisory.** On any failure the agent still sees every tool. The
  one thing enforced in code is structural: a tool that analyses pasted source
  is dropped when nothing was pasted.

## 3. Showing the work

The chat UI showed "Thinking…" and then an answer, discarding the selector's
reasoning, the plan, every tool call and the critic's objection. Two mechanisms,
both already proven in the source repo:

- **Per node.** `invoke_agent` takes an optional callback and streams the graph
  (`stream_mode=["updates", "values", "custom"]`) instead of awaiting it, so
  each node renders a step as it completes.
- **Per tool.** A node that runs for minutes is the one that must not be silent.
  Tools announce themselves through the custom stream *before* they block, so
  the step naming the tool appears immediately. Measured: the tool event fires
  at +39s where its node does not finish until +44s, and a cluster triage keeps
  that node busy for five minutes.

## 4. The answer

Three labelled parts, every time, whichever tool ran:

- **The bug** — what is wrong, in the user's own code.
- **How we found it** — which tool, and the evidence it returned: the signal,
  the file and line, the confidence.
- **The fix** — the change, quotable verbatim.

Confidence is reported, not hidden, and it tracks how hard the evidence is:
`0.95` when a sanitizer watched two waves collide or a debugger read a stopped
wave, `0.62` when only a log said so, `0.55` for a static hazard on a path that
may never execute.

## 5. Sequencing

Three PRs off `feat/aorta-agent`, each independently reviewable:

1. **The agents.** `src/aorta/cia/` with Launch, Watch, Autopsy; taxonomy
   unified per F1; paths and hosts made settings per F3; Slurm behind the seam
   per F4; the degraded-router log per F2. No chat changes. Provable with a
   command and no chatbot.
2. **The tools.** Catalogue, harnesses, the general `triage_workload`, and the
   selector node. Chat gains the ability to choose and run a tool on pasted
   code.
3. **The reporting.** Streamed steps, the tool-start events, and the three-part
   answer.

## 6. Verification

Per PR:

1. Registry/catalogue: every tool has a non-empty description; no tool declares
   a symptom keyword.
2. Structural gating: pasted-source tools are dropped when nothing is pasted,
   kept when something is.
3. Selector: four pasted-code prompts, including a bug never seen before, each
   selecting a tool that acts on the pasted code.
4. Boundary: `tests/cli/test_chat_boundaries.py` extended — no Click in
   `aorta.cia`, no `aorta.chat` import from `aorta.cia`.
5. Headless: Launch→Watch→Autopsy driven from a script with the chat extras
   uninstalled.
6. End to end, on hardware, for each of the three instruments: the sanitizer
   actually ran (state `ran`, not `not_checked`), Watch alerted on its own
   evidence, and Autopsy was triggered *by Watch* rather than by the fallback.

That last check is not pedantry. In the source repo all three demos returned a
correct-looking verdict while the sanitizer had not run, Watch had not alerted
and Autopsy had reached its answer through a fallback path — because every layer
degrades quietly. A verdict alone does not prove the pipeline ran.
