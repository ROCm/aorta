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

To add Cluster Intelligence diagnostics, combine `cia` with whichever chat
interface you install:

```bash
pip install 'amd-aorta[chat-cli,cia]'  # terminal, Python 3.11–3.14
pip install 'amd-aorta[chat-ui,cia]'   # Chainlit, Python 3.11–3.13
pip install 'amd-aorta[chat-all,cia]'  # UI + LiteLLM, Python 3.11–3.13

# Editable checkout:
uv pip install -e ".[chat-cli,cia]"
```

`allow_cluster_jobs` remains `false`. The installed CIA layer initially adds
only `list_cluster_jobs` and `read_autopsy_report`; enabling the setting adds
the three submitting tools (`triage_kernel_source`, `triage_assembly_source`,
`triage_workload`). Run `aorta chat tools` to verify the effective list. If
`cia` is absent, chat remains usable and reports that those five tools were not
offered, with `pip install 'amd-aorta[cia]'` as the remedy. See
[installation](installation.md#add-cia-backed-cluster-diagnostics) for all
published/editable combinations and
[cluster configuration and security](configuration.md#the-cluster-diagnostic-tools)
before enabling submissions.

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
| [Installation](installation.md) | Chat/CIA extra combinations, Python ranges, and the sqlite requirement |
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
| `aorta chat index build\|fetch\|digest\|eval` | Manage the retrieval index; `fetch` takes the prebuilt one |
| `aorta chat doctor` | Check the extras, the backend, the tool protocol, the model cache, the embedding profile, and the index manifest |
| `aorta chat config init\|show\|validate` | Create and inspect the profile |

`aorta chat --help` is authoritative for flags.
