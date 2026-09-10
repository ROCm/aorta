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
| `embedding_provider` | `local` | `local` (a small model on CPU) / `remote` (an embeddings API). Independent of `llm_provider`, and every `config init` profile writes `local` — including the remote-chat ones, because the published index is built with the local model and cannot be read by any other. `remote` is a manual choice with consequences: see [configuring a remote embedding provider](#configuring-a-remote-embedding-provider-by-hand). |
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
| `embedding_model` | `BAAI/bge-small-en-v1.5` | Local model. Cannot be blank: an empty or whitespace-only value selects no model, so it is refused when the settings load. Remove the setting to take the default. |
| `model_cache_path` | `$XDG_CACHE_HOME/aorta/chat/models` | Where the local model's ONNX weights are cached. `HF_HOME` overrides it, which is what [air-gapped pre-seeding](rag-index.md#air-gapped-nodes) uses. Explicit rather than `fastembed`'s own `/tmp/fastembed_cache`, which a reboot wipes and other users on a shared node can write. |

The five `remote_embedding_*` settings below are read by the embedding path only
when `embedding_provider = "remote"`, which no profile selects. Setting them
without also setting `embedding_provider` changes nothing about how the index is
built or queried — though `remote_embedding_api_key` and
`remote_embedding_extra_headers` count as credentials for `config validate`'s
mode check either way. Setting them together with the selector is the procedure
[below](#configuring-a-remote-embedding-provider-by-hand), and it obliges a
local index rebuild.

| Setting | Default | Meaning |
| --- | --- | --- |
| `remote_embedding_model` | `text-embedding-3-small` | Also decides the collection name, since dimensions differ per model. Cannot be blank, on the same rule as `embedding_model`. |
| `remote_embedding_api_key` | *(empty)* | Separate from the chat key, so the two can use different providers. Required: an empty value raises rather than falling back to the local model. |
| `remote_embedding_base_url` | *(empty)* | Empty means the provider default, which for an OpenAI-compatible client is `api.openai.com`. Set it for anything else — a gateway header with an empty base URL sends your corpus to OpenAI. |
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

## Configuring a remote embedding provider by hand

`embedding_provider = "remote"` is supported but selected by nothing: no
`config init` profile writes it, and there is no flag or prompt for it. It is a
deliberate, manual change, and this is the whole procedure.

### When this is the right choice

**One case, and it is narrow: a node that can reach an embeddings API, cannot
reach Hugging Face, and cannot have its model cache pre-seeded.** The local
embedder downloads ~65 MB of ONNX weights from Hugging Face on first use, so a
host firewalled off from it — but allowed out to a corporate gateway — has
nothing to embed with until those weights arrive by some other route.

Copying them in is that other route, and it is the better one: the
[pre-seeded cache](rag-index.md#air-gapped-nodes) keeps embeddings local, free
and unmetered on a host that cannot reach Hugging Face at all. Try it first.
Remote embeddings are what is left when it is impractical — no second machine on
the same AORTA version, no way to move 65 MB onto the node, or a policy against
carrying model weights — and that is the only reason the code path exists.

Everything else that looks like a reason is not one:

- **A remote chat model is not a reason.** The two selectors are independent.
  Remote generation with local embeddings is the normal shape and the cheaper
  one, because retrieval then costs nothing per query.
- **Quality is not a reason worth the trade here.** Retrieval over one source
  tree is not where a larger embedding model earns its price, and the published
  index is only usable with the local model.
- **An air-gapped node is not a reason** — it cannot reach the embeddings API
  either. Pre-seed the model cache instead; see
  [air-gapped nodes](rag-index.md#air-gapped-nodes).

### What it costs you before you start

Read these three before editing anything. Each of them is a consequence, not a
risk to be managed.

1. **The published index becomes unusable for as long as this setting is in
   place.** CI builds the index asset with default settings, so its vectors are
   the local model's. An index is only meaningful to the model that produced it,
   and the manifest check refuses a mismatch rather than answering from it. So
   `aorta chat index fetch` will not leave you with a usable index — it refuses
   the published asset rather than installing it — and an already-fetched index
   will refuse every query. You take over building the index yourself, on every
   AORTA upgrade, until you [go back](#going-back).
2. **Every query costs money, not just the build.** The one-off index build
   embeds the whole corpus; after that every question embeds the query text to
   retrieve source, and retrieval also searches the run-artifact collection on
   every question — so once you have run `index runs`, the floor is two
   embedding calls per question, not one. Each `search_code` or
   `search_run_artifacts` tool call adds another on top. The build
   is the large number — the public corpus is roughly 5 MB of text over ~360
   files, so on the order of one to two million tokens — but the per-query calls
   are the ones that never stop. Check your provider's own price list: at
   `text-embedding-3-small`'s published rate the build is cents, and at a large
   model's, or through a gateway that adds a markup, it is not.
3. **Redaction does not cover this path.** `redact = true` rewrites paths and IP
   addresses out of *LLM* requests. The embeddings request is a different
   request and is sent verbatim. That matters most for the run-artifact
   collection, which holds your own `matrix.json` and `env.json` and can carry
   customer hostnames, filesystem layouts and environment variables — all of
   which would be sent, unredacted, to the embeddings endpoint at
   `index runs` time and never rewritten. [redaction](redaction.md) has the
   full scope. If that corpus must not leave the machine, stop here.

### The procedure

**1. Set the selector and the five settings it turns on.** Six in total, of
which four are required and two apply only behind a gateway:

| Setting | | |
| --- | --- | --- |
| `embedding_provider` | required | The selector. Nothing below reaches an embeddings API without it. |
| `remote_embedding_model` | required | Has a default, but set it explicitly — it names the collection. |
| `remote_embedding_base_url` | required in practice | Only omit it if you mean OpenAI's own API. |
| `remote_embedding_api_key` | required | Empty raises at first use. |
| `remote_embedding_auth_header` | gateway only | Header name instead of a bearer token. For Azure API Management that is exactly `Ocp-Apim-Subscription-Key`. |
| `remote_embedding_extra_headers` | gateway only | Any other headers the gateway wants. |

**Coming from the `azure-apim` profile?** That profile pre-fills
`remote_llm_auth_header = "Ocp-Apim-Subscription-Key"` for the chat side, and it
used to pre-fill `remote_embedding_auth_header` with the same string. It no
longer does — embeddings are local in that profile now, so there was nothing for
it to configure — which makes this the one value the wizard used to hand you and
no longer does. It is the same string on both sides.

In `~/.config/aorta/chat.toml`:

```toml
# The selector. Without this nothing below reaches an embeddings API.
embedding_provider = "remote"

# The model. Also decides the collection name, because dimensions differ per
# model and two models' vectors cannot share a table.
remote_embedding_model = "text-embedding-3-small"

# The endpoint. Do not leave this empty unless you really mean OpenAI's own
# API: empty resolves to api.openai.com, so an empty base URL plus a gateway
# auth header sends your corpus to OpenAI with a header it does not read.
remote_embedding_base_url = "https://gateway.example.com/openai/v1"

# The key. Separate from remote_llm_api_key, so chat and embeddings can use
# different providers. Empty raises at first use rather than silently falling
# back to the local model.
remote_embedding_api_key = "..."

# The last two only behind a gateway that does not take a bearer token. Same
# names and same meaning as the remote_llm_* pair; omit both against a provider
# that takes an Authorization header, which is most of them.
remote_embedding_auth_header = "Ocp-Apim-Subscription-Key"
remote_embedding_extra_headers = { user = "alice", x-tenant = "acme" }
```

Or in the environment, which outranks the file:

```bash
export AORTA_CHAT_EMBEDDING_PROVIDER=remote
export AORTA_CHAT_REMOTE_EMBEDDING_MODEL=text-embedding-3-small
export AORTA_CHAT_REMOTE_EMBEDDING_BASE_URL=https://gateway.example.com/openai/v1
export AORTA_CHAT_REMOTE_EMBEDDING_API_KEY=...
export AORTA_CHAT_REMOTE_EMBEDDING_AUTH_HEADER=Ocp-Apim-Subscription-Key
export AORTA_CHAT_REMOTE_EMBEDDING_EXTRA_HEADERS=user=alice,x-tenant=acme
```

`remote_embedding_api_key` and the values in `remote_embedding_extra_headers`
are both treated as credentials: masked by `config show`, and enough on their
own to make `config validate` fail a profile that is not `0600`.

**2. Confirm what it resolved to, before spending anything.**

```bash
aorta chat config show
```

Check the endpoint is the one you meant. An empty or mistyped
`remote_embedding_base_url` is the failure worth catching here, because the
symptom is a `401` from a third party you did not choose rather than an error
about your configuration.

**3. Rebuild the index. This step is mandatory, not a refresh.**

```bash
aorta chat index build     # the source collection
aorta chat index runs      # the run-artifact collection, if you use it
```

`index runs` is the one to think twice about: it is the command that sends your
own `matrix.json` and `env.json` to the embeddings endpoint, and it re-sends
them every time you rebuild. Skipping it leaves the run-artifact tools without
an index they can read, which is a worse assistant but not an egress you did
not choose.

Not `index fetch` — it refuses the published asset rather than installing it,
correctly, because that asset is the local model's. `index build`
embeds everything under `aorta_path` through the provider you just configured,
and writes it to a collection named after that provider and model, so the local
collection already in the file is not overwritten.

Know what you are giving up in coverage as well as in cost: `aorta_path`
defaults to the installed `aorta` package, so a local build indexes the code
but not `docs/` or `README.md`, which the published asset does carry. Point
`aorta_path` at a source checkout if you want the prose too. See
[the RAG index](rag-index.md#managing-it).

**4. Verify.**

```bash
aorta chat doctor
```

The index check should pass, and the `embedding provider` line should name your
model and the endpoint you configured.

`provider default` where you expected an endpoint is **not** in itself a
failure: that is what the line prints whenever `remote_embedding_base_url` is
empty, which is the supported way to mean OpenAI's own API. It is a symptom only
if you meant to point at a gateway — in which case step 1 did not take effect,
and your corpus is going to OpenAI with a header it does not read.

A refusal here means the index and the configuration still disagree — usually
step 3 was skipped, or was run before step 1 took effect.

### Going back

Remove or unset the six settings and re-fetch:

```bash
aorta chat index fetch
```

The index this replaces is the one step 3 built locally, and that is the point:
going back means giving up the local build for the published asset. If the
command declines for that reason — a downloaded asset can always be downloaded
again, where a local build may not be reproducible — the refusal names the flag
that overrides it, and overriding is the right answer here.

That is the whole way back **only if `embedding_model` is still at its default**.
It is not one of the six, and it is the one other setting the published asset is
checked against: `manifest.validate` refuses on the model name, on the collection
name and on the embedding identity, and for the local provider all three are
derived from `embedding_model`. A hand-set value is therefore refused exactly as
a remote provider was. Clear it as well, or keep it and rebuild with
`aorta chat index build`.

`chunk_size` and `chunk_overlap` are a different case and do **not** block the
re-fetch. They are compared, but as warnings: the vectors are still the published
model's, and the chunker only ran at build time, so what a mismatch costs you is
that retrieved spans are not the size the prompt budget was tuned for. Restore
them if you changed them, but they will not stop `index fetch` and they never
made the published asset unusable.

The two providers use different collection names and coexist in the one
`.sqlite` file, so switching to remote never disturbed the local collection:
the remote `index build` staged a database that lacked it, and installing a
staged database carries over whatever collections the incoming one is missing.

Coming back is not symmetric, because the published asset *does* contain a local
collection. On the default `embedding_model` it has the same name as yours, so
`index fetch` installs the published one over it, and restores the manifest that
the remote `index build` had rewritten to name the remote model. That is what
you want when your local collection came from the published asset in the first
place. When you built it yourself over a wider corpus — an `aorta_path` pointing
at a source checkout, so `docs/` and `README.md` were indexed too — reach for
`aorta chat index build` instead: it restores the manifest just as well and
keeps your own corpus rather than replacing it with the package-only build.

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
