# Providers

Two independent selectors: which backend generates text (`llm_provider`), and
which model turns text into vectors (`embedding_provider`). Mixing them is
normal — remote generation with local embeddings is the cheap default, because
retrieval then costs nothing.

Everything on this page is about `llm_provider`. `embedding_provider` is
`local` for every profile `aorta chat config init` writes, including the remote
ones, and the published index is only readable that way. Choosing a remote
chat model is not a reason to change it; the narrow case that is, and what it
costs, are in
[configuring a remote embedding provider by hand](configuration.md#configuring-a-remote-embedding-provider-by-hand).

## Chat backends

| `llm_provider` | Speaks | Use for | Extra |
| --- | --- | --- | --- |
| `vllm` | OpenAI wire, local server | A model you serve yourself on an AMD GPU | `chat-cli` |
| `openai` | OpenAI wire | OpenAI, OpenRouter, Groq, Together, Fireworks, DeepSeek, Mistral, xAI, and any gateway speaking the same protocol | `chat-cli` |
| `litellm` | LiteLLM's routing | Anthropic, Gemini and Bedrock, whose wire protocols are not OpenAI-compatible; also Azure OpenAI Service | `chat-all` |

`chat-all` is `chat-ui` plus LiteLLM and nothing else; see
[installation](installation.md#the-three-extras) for what each extra costs.

Adding a backend is a new module under `aorta/chat/inference/providers/`
exposing `name`, `get_chat_model()`, `preflight()`, `probe()`,
`unreachable_hint()` and `describe()`, plus one entry in that package's
factory. No node, graph or tool code changes.

`preflight()` and `probe()` are separate because they answer to different
callers. A session's preflight may be permissive -- the local vLLM backend waits
up to five minutes for a server that is still loading weights and then starts
anyway, rather than refuse a warm-up the user can see progressing. `probe()` is
the raising counterpart, on a short budget, and is what `aorta chat doctor`
calls: a diagnostic that inherits the permissive behaviour reports a healthy
backend when there is none. `unreachable_hint()` supplies the text printed
instead of a traceback when a query fails to connect.

Only `vllm` reaches the network to answer `probe()`. The two remote backends
validate configuration and stop there, because a reachability check against a
metered endpoint means billing the operator for running `doctor` -- so a remote
provider whose key is valid and whose endpoint is down still reports `ok`, and
the failure surfaces on the first query as `unreachable_hint()`.

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

With `remote_llm_api_key` empty, LiteLLM reads `ANTHROPIC_API_KEY`,
`GEMINI_API_KEY` and friends itself and AORTA does not touch them. Set
`remote_llm_api_key` and it is passed to LiteLLM explicitly instead — which is
what makes the gateway flow below work on this backend.

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

### Automatic escalation to `native`

**AORTA now acts on that signature rather than only logging it.** On seeing it,
the query is retried once through `native`; if that retry answers, the protocol
is kept for the rest of the process and one log line says so. The default stays
`text`, because flipping it globally would break a stock local vLLM (see the
endpoint row in the table above) — this is detection, not a new default.

Four things follow from that:

- **An explicit `llm_tool_mode` is never overridden**, whether it comes from
  the profile file or `AORTA_CHAT_LLM_TOOL_MODE`. If you set `text`
  deliberately, set it; the escalation only ever moves the built-in default.
- **The wasted spend is bounded; the retry itself is not capped to one call.**
  What is bounded is the cost of *failure*: the escalated retry tolerates a
  single empty reply rather than the usual two, so a model that is silent on
  `native` as well costs exactly one extra call. A retry that is *working* —
  calling tools and getting results — is not truncated; it runs the ordinary act
  loop under the ordinary round budget, because cutting a productive loop off at
  one round would spend the call and throw away the answer it was about to
  reach. So the worst case is the normal budget, not double it, and the protocol
  moves once per process rather than once per query.
- **The switch is thrown only once native has answered.** An endpoint that
  refuses the protocol must not be able to select it, which is what throwing the
  switch up front let it do: the refusal became the state for every later query.
  A failed retry therefore changes nothing except a counter, and the query falls
  back exactly as it would have if the escalation had never fired. Two such
  failures write native off for the rest of the process — two rather than one
  because a timeout and a permanent refusal arrive identically, and what tells
  them apart is whether it happens again. A retry that comes back *silent*
  counts the same as one that errored: "the request returned" is not "the
  protocol works", and a model that says nothing on either protocol must not be
  billed for a native round on every query from then on.
  "Answered" means the model returned prose **or** made at least one tool call.
  A retry that drove a real tool call and then hit a backend error has proved
  the protocol works, so it moves the protocol and does *not* spend one of the
  two failures — otherwise two transient 503s after working tool calls would
  strand the process on `text` for good. Only a failure with no tool call
  behind it counts, which is the shape a refused `tools` payload actually has.
- **The scope is the process, not the conversation.** Under `aorta chat` that is
  the same thing, but `aorta chat ui` serves many browser sessions from one
  server, and there the escalation is shared by all of them. That is deliberate:
  what was detected is a property of the *model* — it emits reasoning instead of
  an `ACTION:` line — so it is equally true for every session talking to that
  endpoint, and sharing it means only the first query in the process pays the
  wasted round. Nothing about a conversation is carried across; the state is a
  flag and a counter about protocol support. Sessions that dead-end at the same
  moment each get their own retry — the shared state records the outcome, it
  does not ration the attempt.

### Reading the protocol that is actually in force

Two places name it, and both come from this change. The `LLM backend:` line
logged at startup names the protocol alongside the provider. The escalation logs
a line of its own when it fires, and that one names the protocol itself rather
than pointing at the startup banner, so it stands on its own wherever it is
read — including in a server log where the startup line has scrolled away.

`aorta chat doctor` is the third place, and what it says there is owned by
[#463](https://github.com/ROCm/aorta/pull/463) rather than by this change: it
adds a hint to the tool-mode line naming the configured protocol and what it
costs. The two startup signals above are what this change contributes and they
do not depend on #463 having landed.

One gap in them: the startup line comes from the CLI entry points, so
`aorta chat` and `aorta chat ask` get it and `aorta chat ui` does not — its
Chainlit welcome banner names the provider but not the protocol, which is the
front door where the process-wide scope above matters most.
[#468](https://github.com/ROCm/aorta/issues/468) tracks putting it there. Until
it does, a UI operator reads the protocol from the escalation warning in the
server log, or from `aorta chat doctor`.

### Answering from retrieved context when the act loop gives up

When the act loop gives up, it makes one tool-free attempt to answer from the
context `retrieve` already gathered, and labels that answer as having used no
tools. **This is independent of the escalation to `native`** and worth stating
separately, because the two are often confused: it fires whenever the loop
abandons with no tool run, including under an explicitly configured
`llm_tool_mode = "text"`, where the escalation is refused outright and no native
retry is attempted at all. So an action-routed question to a reasoning model
does not come back empty-handed even when the protocol never moves.

It applies only when no tool ran at all: once one has — including one that
returned an error, and including one the escalated native retry made before the
backend fell over — "I could not use my tools" would be untrue, so that query
gets the plain give-up notice instead. A tool *outage* is therefore not covered
by it. What is covered is a model that can drive neither protocol, and an
endpoint that refuses the escalated one (a stock local vLLM without
`--enable-auto-tool-choice` and a matching `--tool-call-parser` does): that
refusal never moves the protocol — the switch is thrown only once native has
answered, so there is nothing to roll back — and it lands on this same fallback
rather than escaping as an error. It takes two such failures to write native
off for the rest of the process, so one transient timeout does not disable the
escalation; the warning names which attempt it was.

`aorta chat doctor` reports the resolved mode as its own check, and warns before
you spend a query on it when `text` is paired with a model whose name reads as a
reasoning one — a locally served one as much as a remote one, since the channel
is the model's rather than the endpoint's. A deployment can be served under any
name, so for a name the check does not recognise — or a provider with no model
name to read — it still names `native`, and what turning it on costs here, in
the hint under the line it prints. Under `native` the same line names what the
endpoint has to accept, because nothing in that report tests it: no probe sends
a request carrying `tools`. A local vLLM is asked for `/health`, which a server
that rejects `tools` answers normally, and the remote backends are not called
at all — their `probe()` is a configuration preflight, deliberately, so that a
diagnostic cannot bill you for a round trip. The startup line names the
resolved protocol alongside the provider.

The one path that still ends with no answer is a model that also returns empty
content on the tool-free route. That is rarer than it sounds: the reporter's
transcript shows the same question answered correctly through the `question`
route in 2 calls while the `action` route returned nothing in 4, because the
empty-content behaviour belongs to the tool protocols and not to the model.

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
adds one embedding call on top. That recurring bill is the second reason no
profile selects it; the first is that the published index is built with the
local model, so a remote embedder makes `index fetch` unusable.
[The procedure for choosing it](configuration.md#configuring-a-remote-embedding-provider-by-hand)
covers both.

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
| `embedding_provider = "local"` | Keeps all retrieval free even when generation is remote. Already the case unless you set it by hand. |
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
| `I wasn't able to answer that: something in my own configuration...` | The act loop produced no usable text. Which attempts it made first depends on `llm_tool_mode` and on whether a tool had already run, so read the warning logged beside this message — it names the step that gave up. `aorta chat doctor` covers the configuration faults that reach this message by other routes. |
| `this process will use native from here` | Not an error. The model returned no answer and no tool call under `text` — the line names the signature that was observed — and native answered it, so chat switched protocol for the rest of this process. Set `llm_tool_mode` yourself to pin it either way. |
| `The escalated native tool-calling request failed ... without making a tool call` | The retry was tried and the request did not come back, having called nothing. If it is a local vLLM, it needs both `--enable-auto-tool-choice` and a matching `--tool-call-parser` (see the endpoint row in the table above); otherwise the endpoint may simply have been unwell. The protocol does *not* move, and the line says which attempt it was — after the second, native is not tried again in this process. Set `llm_tool_mode = "text"` to skip the attempt entirely. |
| `The escalated native tool-calling request failed before it could call anything` | The same outcome, one step earlier: the backend could not even be built or the tool schemas could not be bound, so no request was made. Counts as an attempt in the same way. Read the exception named on the line — this is a backend or configuration fault, not a protocol one. |
| `... drove N tool call(s) and then failed` | Structured tool calling *works* on this endpoint and the backend fell over afterwards. So the protocol **does** move to `native`, and this deliberately does not count against the two-failure budget — otherwise two transient errors after working tool calls would strand the process on `text`. The answer is built from whatever those calls gathered. |
| `The escalated native tool-calling request returned no answer and no tool call either` | The retry reached the endpoint and the model was as silent on `native` as it was on `text`, so the protocol is not what it is failing on. `text` stays in force, and this counts as one of the two attempts above. Nothing here is a configuration fault; the model cannot drive either protocol for this query. |
| An answer prefixed `I could not use my tools for this question` | The act loop gave up and the answer came from retrieved context alone, so anything needing a live lookup is missing from it. Same underlying cause as the row above. |
| Many `Act round N: ... re-prompting` lines and no answer | Same cause. Set `llm_tool_mode = "native"`. |
| `Waiting for vLLM at ...` when you meant to go remote | `llm_provider` is still `vllm`. Check the backend line printed at startup. |
| The call-count line never appears | Expected on `llm_provider = "vllm"`; only the remote backends attach the counter. |
| `extra header #N is missing '='` | `remote_llm_extra_headers` takes `name=value` pairs or a JSON object. `#N` is the position in the comma-separated list, counted from 1 — the entry itself is not quoted back, because a header value may be a credential. |
