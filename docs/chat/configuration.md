# Configuring `aorta chat`

## Where settings come from

Four sources, highest priority first:

1. **Command-line flags** — `--llm-provider`, `--llm-model`, `--no-redact`.
2. **Environment** — every setting is `AORTA_CHAT_` plus the upper-cased name,
   so `chunk_size` is `AORTA_CHAT_CHUNK_SIZE`. The prefix is not decoration: a
   bare `CHUNK_SIZE` or `ALLOWED_COMMANDS` in a public tool that runs inside
   other people's job scripts is asking for a collision.
3. **The profile file** — `$XDG_CONFIG_HOME/aorta/chat.toml`, defaulting to
   `~/.config/aorta/chat.toml`.
4. **Built-in defaults.**

The environment outranking the file is deliberate: a one-off `export` or a CI
job always wins over whatever is on disk.

Unknown keys in the profile are ignored at load time rather than rejected, so a
file written by a newer AORTA does not stop an older one from starting.
`aorta chat config validate` is where they are reported.

## Creating a profile

```bash
aorta chat config init --profile openai
```

`--profile` picks a starting point and then the wizard prompts for the few
fields that profile needs:

| `--profile` | For |
| --- | --- |
| `openai` | OpenAI itself |
| `openai-compatible` | Any OpenAI-wire endpoint: OpenRouter, Groq, Together, Fireworks, a self-hosted gateway |
| `azure-apim` | An Azure API Management gateway, which wants the key in a named header |
| `anthropic` | Native Anthropic protocol through LiteLLM (needs `chat-all`) |
| `local-vllm` | A vLLM server you run yourself |

`--no-input` writes the template without prompting, for scripting. `--force`
overwrites an existing file.

```bash
aorta chat config show          # effective settings, credentials masked
aorta chat config show --json   # same, machine-readable
aorta chat config validate      # parses? no dead keys? not world-readable?
```

## Secrets

The profile holds your API key, and it is created mode `0600` — the mode is set
on the file descriptor before any bytes are written, so the key is never briefly
world-readable, and it is re-applied if a previous run or a careless editor left
the file at `0644`.

Putting a credential at rest in a predictable path inside a tool whose day job
is collecting diagnostic bundles obliges two guards, and both are in place:

- **`aorta chat config show` masks keys** to their length and last four
  characters. `--reveal` prints them in full. This exists because the likeliest
  leak is not an attacker; it is a customer pasting their own config into a
  support ticket.
- **`aorta bundle` refuses to package the profile.** The chat config path is
  excluded explicitly, not by a filename convention.

`aorta chat config validate` also fails a profile that holds a credential at a
permissive mode, which on a shared node is a real finding rather than a style
note.

If you would rather not store the key at all, leave it out of the file and
export `AORTA_CHAT_REMOTE_LLM_API_KEY` instead; the environment outranks the
file.

## Settings

Every name below is a TOML key in the profile, and `AORTA_CHAT_<NAME>` in the
environment.

### Selectors

| Setting | Default | Meaning |
| --- | --- | --- |
| `llm_provider` | `vllm` | `vllm` (local server) / `openai` (any OpenAI-wire endpoint) / `litellm` (native Anthropic, Gemini, Bedrock). An unknown value raises, listing the accepted names. |
| `embedding_provider` | `local` | `local` (a small model on CPU) / `remote` (an embeddings API). Independent of `llm_provider`. |
| `llm_tool_mode` | `text` | `text` parses `ACTION: tool(arg="v")` lines out of the reply; `native` uses the provider's function-calling API. Reasoning models need `native` — see [providers](providers.md#tool-calling-and-reasoning-models). |

### Local vLLM (`llm_provider = "vllm"`)

| Setting | Default |
| --- | --- |
| `vllm_base_url` | `http://localhost:8000/v1` |
| `vllm_model` | `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct` |
| `vllm_api_key` | `EMPTY` — vLLM ignores it, but the OpenAI client rejects an empty string |

### Remote chat (`llm_provider = "openai"` or `"litellm"`)

| Setting | Default | Meaning |
| --- | --- | --- |
| `remote_llm_model` | `gpt-4o-mini` | The model id as the provider names it. For `litellm`, LiteLLM's own id format. |
| `remote_llm_api_key` | *(empty)* | Required for `openai`; a missing value fails preflight rather than mid-query. Used by `litellm` too **when set** — only when it is empty does LiteLLM fall back to its own standard variables (`ANTHROPIC_API_KEY`, ...). |
| `remote_llm_base_url` | *(empty)* | Empty means the provider default. Set it for anything else. |
| `remote_llm_auth_header` | *(empty)* | Header name for a gateway that does not take a bearer token. Honoured by both `openai` and `litellm`. |
| `remote_llm_extra_headers` | *(empty)* | Extra headers a gateway wants, as `user=alice,x-tenant=acme` or a JSON object. Honoured by both `openai` and `litellm`. Values are masked by `config show` and count as a credential for the 0600 check, because a gateway key put here is as sensitive as `remote_llm_api_key`. |

### Call limits

| Setting | Default | Meaning |
| --- | --- | --- |
| `llm_max_tokens` | *(unset)* | Cap on generated tokens per call. Omit the key entirely rather than setting it empty. |
| `llm_timeout` | `120` | Seconds before one request is abandoned. |
| `llm_max_retries` | `2` | Transport-level retries per call. Multiplies spend on a flaky endpoint. |

### Embeddings

| Setting | Default | Meaning |
| --- | --- | --- |
| `embedding_model` | `BAAI/bge-small-en-v1.5` | Local model. |
| `model_cache_path` | `$XDG_CACHE_HOME/aorta/chat/models` | Where the local model's ONNX weights are cached. `HF_HOME` overrides it, which is what [air-gapped pre-seeding](rag-index.md#air-gapped-nodes) uses. Explicit rather than `fastembed`'s own `/tmp/fastembed_cache`, which a reboot wipes and other users on a shared node can write. |
| `remote_embedding_model` | `text-embedding-3-small` | Also decides the collection name, since dimensions differ per model. |
| `remote_embedding_api_key` | *(empty)* | Separate from the chat key, so the two can use different providers. |
| `remote_embedding_base_url` | *(empty)* | Empty means the provider default. |
| `remote_embedding_auth_header` / `remote_embedding_extra_headers` | *(empty)* | As on the chat side. Behind a gateway you normally set both or neither. |

### Corpus and index

| Setting | Default | Meaning |
| --- | --- | --- |
| `aorta_path` | the installed `aorta` package | The source tree retrieval and the file tools are scoped to. Only ever read. |
| `runs_path` | the working directory | Where your own sweep output directories live, for the run-artifact tools. |
| `index_path` | `$XDG_CACHE_HOME/aorta/chat/index.sqlite` | The vector index, one file. |
| `repo_map_path` | `$XDG_CACHE_HOME/aorta/chat/repo_map.md` | The generated function/class index. |
| `repo_map_prompt_max_chars` | `20000` | Cap on how much of the map is injected into the planner's prompt; `0` disables the cap. The `search_repo_map` tool still queries the whole file. |
| `chunk_size` / `chunk_overlap` | `512` / `50` | Indexer text splitter. Changing either invalidates the index. |

Nothing writable defaults inside `site-packages`. An installed wheel is
read-only on a shared node, and a tool that writes into its own install
directory cannot be pip-upgraded cleanly.

### Retrieval and the agent loop

| Setting | Default | Meaning |
| --- | --- | --- |
| `retriever_k` / `retriever_fetch_k` | `12` / `30` | Chunks returned, and candidates fetched to select from. |
| `search_tool_k` | `10` | Results from the `search_code` tool. |
| `max_act_rounds` / `max_act_rounds_search` | `5` / `8` | Tool-loop budget for ordinary and search-shaped questions. The single biggest lever on cost. |
| `max_retry_iterations` | `3` | Critic retry budget. `0` disables the retry loop. |

### Command execution and egress

| Setting | Default | Meaning |
| --- | --- | --- |
| `enable_shell_tool` | `false` | Register `run_terminal_command`. Off by default: it hands a model-authored string to a shell, so the agent is not given one unless you say so. While off, the tool is absent from the registry and from the prompts, not merely refused at call time. |
| `allowed_commands` | `python,pytest,make,pip,grep,wc,head,tail,cat,ls,find` | Allowlist for `run_terminal_command`, applied per pipeline stage. Command chaining and redirection (`;`, `&`, backticks, `$(...)`, `>`, `<`) are refused, since the allowlist checks executables. Accepts `a,b,c` or a JSON list. |
| `command_timeout` | `60` | Seconds before a `run_terminal_command` command is killed. |
| `redact` | `true` | Rewrite filesystem paths and IP addresses out of outbound LLM requests. Does not cover the remote-embedding path. Read [redaction](redaction.md) before turning this off — and read it anyway for what it does **not** cover. |

## Example profile

```toml
# ~/.config/aorta/chat.toml
llm_provider = "openai"
remote_llm_model = "gpt-4o-mini"
remote_llm_api_key = "sk-..."
llm_tool_mode = "native"

embedding_provider = "local"

aorta_path = "/home/me/src/aorta"
runs_path = "/home/me/sweeps"
max_act_rounds_search = 4
```
