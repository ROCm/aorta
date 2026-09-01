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

## Commands

| Command | Description |
| --- | --- |
| `aorta chat` | Interactive REPL |
| `aorta chat ask "..."` | Answer once and exit; `--json` / `--plain` for piping |
| `aorta chat ui` | Chainlit web UI (needs `chat-ui`) |
| `aorta chat tools` | List the agent tools, built-in and plugin-contributed |
| `aorta chat config init\|show\|validate` | Create and inspect the profile |

`aorta chat --help` is authoritative for flags.
