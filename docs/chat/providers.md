# Providers

Two independent selectors: which backend generates text (`llm_provider`), and
which model turns text into vectors (`embedding_provider`). Mixing them is
normal — remote generation with local embeddings is the cheap default, because
retrieval then costs nothing.

## Chat backends

| `llm_provider` | Speaks | Use for | Extra |
| --- | --- | --- | --- |
| `vllm` | OpenAI wire, local server | A model you serve yourself on an AMD GPU | `chat-cli` |
| `openai` | OpenAI wire | OpenAI, OpenRouter, Groq, Together, Fireworks, DeepSeek, Mistral, xAI, and any gateway speaking the same protocol | `chat-cli` |
| `litellm` | LiteLLM's routing | Anthropic, Gemini and Bedrock, whose wire protocols are not OpenAI-compatible; also Azure OpenAI Service | `chat-all` |

Adding a backend is a new module under `aorta/chat/inference/providers/`
exposing `name`, `get_chat_model()`, `preflight()` and `describe()`, plus one
entry in that package's factory. No node, graph or tool code changes.

### Local vLLM

```toml
llm_provider = "vllm"
vllm_base_url = "http://localhost:8000/v1"
vllm_model = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
```

Preflight polls the server's `/health` and warns rather than failing, so a
server that is still loading does not abort the session.

### An OpenAI-compatible endpoint

```toml
llm_provider = "openai"
remote_llm_model = "gpt-4o-mini"
remote_llm_api_key = "sk-..."
# Only for a non-OpenAI endpoint:
# remote_llm_base_url = "https://openrouter.ai/api/v1"
```

Preflight validates that a key is present without making a network call, so a
missing key fails at startup instead of mid-query.

### Azure OpenAI Service

Azure OpenAI is **not** OpenAI-wire-compatible: it rewrites the URL path to
`/openai/deployments/<deployment>` and requires an `api-version` query
parameter, neither of which the `openai` backend can express. Route it through
LiteLLM.

```toml
llm_provider = "litellm"
remote_llm_model = "azure/<your-deployment-name>"
```

The credentials do not go in `remote_llm_api_key`; LiteLLM reads its own
environment variables, and Azure needs all three:

```bash
export AZURE_API_KEY=...
export AZURE_API_BASE=https://<resource>.openai.azure.com
export AZURE_API_VERSION=2024-02-01
```

`AZURE_API_VERSION` must be in the environment — `ChatLiteLLM` exposes no
`api_version` field, so there is no setting for it. Note the model name is the
**deployment** name prefixed with `azure/`, not the underlying model name.

### Anthropic, Gemini, Bedrock

```toml
llm_provider = "litellm"
remote_llm_model = "claude-sonnet-4-5"
```

LiteLLM reads `ANTHROPIC_API_KEY`, `GEMINI_API_KEY` and friends itself; AORTA
never touches them, so `remote_llm_api_key` is ignored for this backend.

Current Claude Opus builds accept only `temperature=1` and LiteLLM raises rather
than negotiating. The graph asks for 0.0 and 0.1, so this backend enables
LiteLLM's `drop_params` — which means routing and criticism are less
deterministic on those models than on one that honours temperature.

## Gateway authentication

`remote_llm_api_key` on its own assumes the key travels as
`Authorization: Bearer <key>`, which is what public providers expect. Corporate
gateways often disagree. Five families, and only the first three are "put this
string somewhere static", which is why one header mechanism covers them:

| Family | How the key travels | Seen on | Supported by |
| --- | --- | --- | --- |
| Bearer token | `Authorization: Bearer <key>` | OpenAI, OpenRouter, Groq, Together, Fireworks, DeepSeek, Mistral, xAI, vLLM | `openai`, out of the box |
| Custom static header | One named header | Azure API Management (`Ocp-Apim-Subscription-Key`), Azure OpenAI (`api-key`), Anthropic direct (`x-api-key`), Gemini (`x-goog-api-key`) | `openai` + `remote_llm_auth_header` |
| Query parameter | `?key=<key>` | Google AI Studio, legacy endpoints | `openai`, by putting it in `remote_llm_base_url` |
| OAuth2 / token exchange | Short-lived bearer, refreshed | Entra ID for Azure OpenAI, Vertex AI service accounts, IBM watsonx | `litellm` |
| Request signing | Signature over the request | AWS Bedrock (SigV4) | `litellm` |

The last two need refresh or signing logic, which is deliberately not
reimplemented here — that is what the `litellm` backend is for.

### Worked example: an API Management gateway

```toml
llm_provider = "openai"
remote_llm_base_url = "https://gateway.example.com/openai"
remote_llm_model = "gpt-oss-20b"
remote_llm_api_key = "<your subscription key>"
remote_llm_auth_header = "Ocp-Apim-Subscription-Key"
remote_llm_extra_headers = { user = "your-username" }
llm_tool_mode = "native"
```

Two details that surprise people:

- The `user` header here is attribution and quota, not authentication. That is
  why extra headers are a separate setting from the auth one — a JSON blob
  holding a live credential is far easier to leak into a log line, a pasted
  config or a PR diff than a lone `remote_llm_api_key`. Keeping them apart is
  also what lets the startup line name the auth header while guaranteeing no
  value is printed.
- The secret still goes in `remote_llm_api_key`. When `remote_llm_auth_header`
  is set, the bearer slot receives the throwaway string `unused`, because the
  OpenAI client rejects an empty `api_key` even when the gateway ignores it.

Preflight reports the header name and never the key:

```
Using remote OpenAI-compatible -- gpt-oss-20b at https://gateway.example.com/openai (auth: Ocp-Apim-Subscription-Key header, plus user)
```

A gateway may also expose a *native* Anthropic route on a different path of the
same host, speaking Anthropic's protocol rather than OpenAI's — so finding one
tells you nothing about the other. `curl` the path with `/v1/messages`: if it
answers, use `llm_provider = "litellm"` with an `anthropic/`-prefixed model
name, not `openai`.

## Tool calling and reasoning models

The agent's act loop needs the model to call tools. Two protocols, chosen by
`llm_tool_mode`:

| | `text` (default) | `native` |
| --- | --- | --- |
| How | The model writes `ACTION: search_code(query="x")` and AORTA parses it | The provider's function-calling API returns structured `tool_calls` |
| Endpoint requirement | None | Must accept the `tools` parameter |
| Local vLLM | Works as shipped | Needs `--enable-auto-tool-choice` and a `--tool-call-parser` |
| Reasoning models | **Does not work** | Works |

A reasoning model puts its working in a separate channel and returns empty
`content` when it wants to act. It never writes the `ACTION:` line, so the parse
finds nothing, the loop re-prompts, and the query ends with no answer having
spent the whole retry budget. Measured against `gpt-oss-20b` on one search
query: 0 parseable actions in 8 rounds, 11 billed calls, no answer. The same
query in `native` mode drove 8 real tool calls and returned a complete answer.

The symptom is distinctive — `finish_reason` is `stop`, output tokens are
non-zero, and `content` is empty — and the logs say so:
`act_node ... produced no text despite N output tokens`.

Both protocols run the same tools, retrieval and critic, and both are guarded
the same way: an empty reply is never used as the answer, unproductive rounds
are capped at two, a repeated identical tool call is answered with "you already
asked that" rather than re-run, an unknown or protocol-mangled tool name returns
an error the model can read instead of aborting the request, and the final
synthesis call runs with no tools bound (offered tools, a model that has not
found what it wants keeps calling them and returns no prose).

## What a question costs

The agent is agentic, not a single completion, so one question fans out:

| Path | Calls |
| --- | --- |
| Question (route → retrieve → answer) | 2 |
| Action, first pass (route → plan → retrieve → act → critic) | 3 + up to `max_act_rounds`, plus one synthesis call if the loop is exhausted |
| Each critic rejection | Replays act + critic, up to `max_retry_iterations` times |

A search-shaped action query can therefore reach about 12 calls in one pass, and
several critic passes can push a single question past forty. Most action queries
land in the 4–6 range in practice, because the act loop stops as soon as the
model answers without a tool call and the critic usually accepts first time.
With `embedding_provider = "remote"`, each retrieval and each `search_code` call
adds one embedding call on top.

Against a metered endpoint that is real money, so the remote backends log the
per-query call count at INFO, visible without `--verbose`:

```
aorta.chat.inference.callcount INFO Remote LLM calls for this query: 7
```

It is a process-wide total read as a before/after delta, so concurrent UI
sessions inflate each other's numbers — a spend indicator, not an accounting
record. The local backend does not attach the counter.

Knobs that lower the bill, roughly in order of effect:

| Setting | Effect |
| --- | --- |
| `max_act_rounds_search` / `max_act_rounds` | Hard cap on the most expensive loop. Lowering the search budget to 3–4 is the single biggest saving. |
| `max_retry_iterations` | `0` removes the critic's multiplier on everything above. |
| `llm_max_tokens` | Caps output tokens per call. |
| `retriever_k` / `search_tool_k` | Fewer chunks means a smaller prompt, and prompt tokens dominate a long act loop. |
| `llm_max_retries` | Lower it on an unreliable endpoint, so failures do not silently triple. |
| `embedding_provider = "local"` | Keeps all retrieval free even when generation is remote. |
| `remote_llm_model` | A smaller model in the same family is usually the cheapest change of all. |

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `LLM backend unavailable: unknown LLM provider: 'gpt4'` | `llm_provider` must be `vllm`, `openai` or `litellm`. |
| `remote_llm_api_key is not set, and llm_provider=openai requires it` | Set the key, or export `AORTA_CHAT_REMOTE_LLM_API_KEY`. |
| `llm_provider=litellm needs both litellm and langchain-litellm` | `pip install 'amd-aorta[chat-all]'`. The lazy import cannot tell which of the two is missing, so it names both. |
| `401` / `403` from a gateway whose key works in `curl` | The gateway wants a named header, not a bearer token. Set `remote_llm_auth_header`. |
| `Access denied due to missing subscription key` | Azure API Management's wording for the same thing: `remote_llm_auth_header = "Ocp-Apim-Subscription-Key"`. |
| `Incorrect API key provided: unused` from `platform.openai.com` | `remote_llm_auth_header` is set but `remote_llm_base_url` is empty, so the request went to OpenAI. The preflight line says `at the provider default endpoint` when this is wrong. |
| `404` on an `*.openai.azure.com` endpoint | Azure OpenAI needs the `litellm` backend, not `openai`. |
| `missing_keys: ['AZURE_API_VERSION', ...]` | Export all three `AZURE_*` variables; there is no setting for `api_version`. |
| `I could not complete that request...` | The model returned empty content. On a reasoning model, set `llm_tool_mode = "native"`. |
| Many `Act round N: ... re-prompting` lines and no answer | Same cause. Set `llm_tool_mode = "native"`. |
| `Waiting for vLLM at ...` when you meant to go remote | `llm_provider` is still `vllm`. Check the backend line printed at startup. |
| The call-count line never appears | Expected on `llm_provider = "vllm"`; only the remote backends attach the counter. |
| `extra header 'user' is missing '='` | `remote_llm_extra_headers` takes `name=value` pairs or a JSON object. |
