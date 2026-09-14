# `aorta chat`

An interactive assistant over the AORTA codebase and over the run artifacts on
this machine. It answers questions like "how does the probe engine choose which
mitigations to try?" by retrieving the relevant source, and "why did cell
`tf32_off-local` fail in yesterday's sweep?" by reading that run's
`matrix.json`.

It is **opt-in**. `pip install amd-aorta` still installs `pyyaml` and `click`
and nothing else; chat lives behind the `chat-cli` / `chat-ui` / `chat-all`
extras.

```bash
pip install 'amd-aorta[chat-cli]'
aorta chat config init --profile openai
aorta chat ask "What does the five-tier failure classifier check?"
aorta chat                       # interactive REPL
```

## `aorta chat` or `aorta agent`?

Both drive an LLM, and the line between them is whether you sit and watch.

| | `aorta chat` | `aorta agent <name>` |
| --- | --- | --- |
| Shape | Conversational, human-in-the-loop | Autonomous, runs to a verdict |
| Output | Prose in your terminal (or JSON) | Artifacts on disk |
| Example | "Which mitigations touch hipBLASLt?" | `aorta agent mitigate` — closed-loop mitigation search |

`aorta agent` is a namespace over the `aorta.agents` entry-point group; see
[the agentic testing guide](../agent/agentic-testing-guide.md).

## Guides

| Guide | Description |
| --- | --- |
| [Installation](installation.md) | The three extras, the Python range, and the sqlite requirement |
| [Configuration](configuration.md) | The profile file, precedence, secrets, and every setting |
| [Providers](providers.md) | Local vLLM, OpenAI-compatible, LiteLLM; gateway auth; what a question costs |
| [The RAG index](rag-index.md) | What is indexed, when to rebuild, and why a stale index is dangerous |
| [Redaction](redaction.md) | What leaves the machine, what is rewritten first, and what is **not** |
| [Adding a tool](extending.md) | Contributing an agent tool from your own package |

## What a turn looks like

The answer is delivered the same way in both front doors and has not changed:
one complete reply per question. In the browser a `Thinking...` placeholder is
posted and then replaced by the finished answer; nothing is streamed token by
token, and there is no partial text to read while the model is still deciding.

What is new is what happens *between* those two moments, and only in the
browser.

A turn that calls a diagnostic tool can run for several minutes while a cluster
job compiles, queues and runs. Previously nothing was shown for that whole time
— steps are reported when a graph node finishes, so the one node that takes real
time was the one that said nothing, and a browser with no traffic for five
minutes reports the backend as unreachable rather than busy. So the UI now shows
a step as each node completes, and a tool announces itself *before* it blocks
rather than after it returns.

| | Browser (`aorta chat ui`) | CLI (`aorta chat`, `aorta chat ask`) |
| --- | --- | --- |
| Progress during the turn | A step per node, and one when a tool starts | None |
| The answer | Replaces the placeholder, complete | Rendered when it is ready |
| Token-by-token streaming | No | No |

The tool announcement carries the tool's name and nothing else. It deliberately
does not carry the arguments: for `triage_kernel_source` those are the user's
entire pasted kernel, and the step renders the name.

Progress is best-effort. If the browser has gone — a closed tab during a
five-minute job — reporting fails, the failure is logged once, and the run
continues to completion rather than being abandoned along with the session.

Mechanically: `invoke_agent` awaits the graph when no progress callback is
passed, which is what the CLI does, and streams it when one is. The CLI path is
byte-for-byte the one that was there before.


## Commands

| Command | Description |
| --- | --- |
| `aorta chat` | Interactive REPL |
| `aorta chat ask "..."` | Answer once and exit; `--json` / `--plain` for piping |
| `aorta chat ui` | Chainlit web UI (needs `chat-ui`) |
| `aorta chat tools` | List the agent tools, built-in and plugin-contributed |
| `aorta chat index build\|fetch\|digest\|eval` | Manage the retrieval index; `fetch` takes the prebuilt one |
| `aorta chat doctor` | Check the extras, the backend, the model cache, and the index manifest |
| `aorta chat config init\|show\|validate` | Create and inspect the profile |

`aorta chat --help` is authoritative for flags.
