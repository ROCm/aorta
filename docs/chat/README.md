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

The placeholder now steps aside as soon as there is something better to show.
It is what you see until the first step appears — the router has to finish
before anything can be reported, and that is an LLM call — and it is removed at
that point rather than sitting above the steps until the answer lands.

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
| Progress during the turn | A step per node, and one held open while a tool runs | None |
| The answer | Replaces the placeholder, complete | Rendered when it is ready |
| Token-by-token streaming | No | No |

These are the steps, and the placeholder is gone by the time the first one
appears — until then it is all there is to show, because the router has to
finish before anything can be reported and that is an LLM call. A node that
records nothing renders no step and does not retire the placeholder, so the
screen is never empty while the first real node runs.

| Node | Shown as |
| --- | --- |
| `router` | Deciding whether this needs a job |
| `select` | Choosing a diagnostic tool |
| `plan` | Planning the steps |
| `act` | Running tools |
| `critic` | Checking the answer |

Nodes not listed here render nothing: plumbing like `retrieve` stays out of the
way rather than reporting itself as progress.

A tool announces itself twice: once before it blocks, and once when it returns.
The step opens on the first and is held until the second, so a job that takes
five minutes reads as running for five minutes. Closing it on the announcement
alone — which is what rendering a step per event gets you — stamped it finished
the instant it began, and the wait the announcement exists to explain was the
part the UI showed as over. The completion is emitted from a `finally`, because
a step retained until it arrives is only safe if it always arrives.

Each announcement carries the tool's name, an id tying the pair together, and on
the completion how long the call took. It deliberately does not carry the
arguments: for `triage_kernel_source` those are the user's entire pasted kernel,
and the step renders the name. The id is there because one turn can call the
same tool more than once, and closing by name alone would end the wrong step.

Progress is best-effort. If the browser has gone — a closed tab during a
five-minute job — reporting fails, the failure is logged once, and the run
continues to completion rather than being abandoned along with the session.

### What changed against the integration's contract

The `aorta_llm` architecture notes describe `invoke_agent` as the single point
both front doors converge on, awaiting the graph and returning one complete
answer. That is still what the CLI gets, byte for byte: with no progress
callback `invoke_agent` awaits `ainvoke` exactly as it did.

Passing a callback now streams the graph instead. This is a real change to that
contract and worth stating plainly, because the part of it people remember —
"not streamed token by token" — is *not* what changed:

| Stream mode | Carries | Rendered as |
| --- | --- | --- |
| `updates` | the delta from each node as it finishes | a step per node |
| `custom` | the tool announcements described above | a step held open for the tool |
| `values` | the accumulated state; the last one is the answer | the replacement message |

`messages` is the mode that would emit tokens, and it is deliberately not
requested. The answer is still assembled once, at the end, and still arrives as
one message replacing the placeholder. What the stream added is progress
*between* those two moments, not partial text within the answer.


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
