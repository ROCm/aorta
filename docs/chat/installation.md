# Installing `aorta chat`

Chat is an extra, not part of the base install. `pip install amd-aorta` remains
`pyyaml` plus `click` — about 9 MB and a few seconds on a customer node — and
chat is a couple of hundred megabytes on top, so it is opted into deliberately.

## The three extras

Sizes are measured `site-packages` on Python 3.12, base install included.

| Extra | Adds | Python | Size |
| --- | --- | --- | --- |
| `chat-cli` (alias: `chat`) | LangChain / LangGraph, the OpenAI client, sqlite-vec, fastembed, rich | 3.11+ | ~210 MB |
| `chat-ui` | `chat-cli` plus Chainlit, for `aorta chat ui` | 3.11–3.13 | ~305 MB |
| `chat-all` | `chat-ui` plus LiteLLM, for native Anthropic / Gemini / Bedrock | 3.11–3.13 | ~420 MB |

The embedding model's weights (~65 MB — `fastembed` serves this model from a
quantised ONNX re-host, not the 130 MB fp32 build) are fetched on first use and
cached outside the environment, so they are not in those numbers.

> **There is deliberately no torch tier, not even an opt-in one.** Embeddings
> run `BAAI/bge-small-en-v1.5` on `onnxruntime` via `fastembed`, so no extra can
> resolve a PyTorch build. This is a choice about what pip resolves rather than
> a performance preference: `sentence-transformers` hard-requires torch, and on
> a machine without torch already present pip resolves the default PyPI wheel,
> which is the **CUDA** build — roughly 2.7 GB of `nvidia-*` libraries for a
> 384-dimension text embedder, on an AMD node. The nightly and release index
> jobs fail if `torch`, any `nvidia-*` wheel, `chromadb` or
> `sentence-transformers` appears in the environment.

```bash
# Published package
pip install 'amd-aorta[chat-cli]'

# Editable source checkout
uv pip install -e ".[chat-cli]"
```

`pip install 'amd-aorta[chat]'` is an alias for `chat-cli`, because that is what
most people type first.

## Add CIA-backed cluster diagnostics

The Cluster Intelligence Agents are a separate `cia` extra. Install it beside
the chat interface you use; installing a chat extra alone deliberately leaves
all five cluster tools out.

```bash
# Published package: choose one chat interface.
pip install 'amd-aorta[chat-cli,cia]'
pip install 'amd-aorta[chat-ui,cia]'
pip install 'amd-aorta[chat-all,cia]'

# Editable source checkout: the same combinations.
uv pip install -e ".[chat-cli,cia]"
uv pip install -e ".[chat-ui,cia]"
uv pip install -e ".[chat-all,cia]"
```

The headless `cia` extra supports Python 3.10–3.14, but combining it with chat
keeps the chosen chat interface's range: `chat-cli` supports 3.11–3.14, while
`chat-ui` and `chat-all` support 3.11–3.13.

Installing `cia` does **not** grant permission to submit work. With the default
`allow_cluster_jobs = false`, these read-only, jobs-root-contained tools are
available:

- `list_cluster_jobs`
- `read_autopsy_report`

The three tools that write staged inputs and submit scheduler work remain
absent until an operator sets `allow_cluster_jobs = true`:

- `triage_kernel_source`
- `triage_assembly_source`
- `triage_workload`

Set it persistently in `~/.config/aorta/chat.toml`, or for one process:

```bash
export AORTA_CHAT_ALLOW_CLUSTER_JOBS=true
```

Before enabling it, configure the shared jobs path, GPU architecture, scheduler,
and sanitizer backend in
[the cluster diagnostic settings](configuration.md#the-cluster-diagnostic-tools),
and read [why submitting tools are a separate security capability](extending.md#the-exception-and-why-it-is-one)
plus the [redaction boundary](redaction.md).

Verify the effective registry rather than inferring it from what pip installed:

```bash
aorta chat tools
aorta chat tools --json
```

At the safe default, the first command includes `list_cluster_jobs` and
`read_autopsy_report`, but no `triage_*` tools. After enabling cluster jobs it
includes all five. Without `cia`, chat still starts and its ordinary code/run
tools still work; startup prints that the five cluster diagnostic tools were
not offered and names the remedy:
`pip install 'amd-aorta[cia]'`.

**No GPU.** Retrieval runs a small embedding model on CPU and generation happens
wherever your provider lives; no extra pulls a torch build at all, which matters
on a ROCm node for the reason in the callout above.

## Python range

The rest of AORTA supports 3.10 through 3.14. Chat is narrower at both ends, and
neither bound is arbitrary.

- **Floor: 3.11.** The profile file is read with the standard library's
  `tomllib`, which arrived in 3.11. On 3.10, `aorta chat` prints a one-line
  explanation and exits; every other `aorta` command is unaffected.
- **Ceiling for the UI: below 3.14.** Chainlit declares
  `Requires-Python: >=3.10,<3.14`, so `chat-ui` and `chat-all` cannot install it
  on 3.14. `chat-cli` itself is fine on 3.14.

Python packaging has no way to give an *extra* its own `requires-python`, so the
range is expressed as environment markers on each dependency. The consequence is
worth knowing, because it is quiet: on an out-of-range interpreter the extra
still installs **successfully** and simply contributes nothing. `aorta chat`
therefore re-checks the interpreter itself, and `aorta chat ui` on 3.14 tells
you Chainlit is the reason rather than suggesting you install an extra you
already have.

## sqlite

The vector index is a single sqlite file using the
[`sqlite-vec`](https://github.com/asg017/sqlite-vec) extension, so your Python's
sqlite3 must be **3.41 or newer** and must have been built with loadable
extension support. Current distributions are fine. Enterprise Linux is often
not: RHEL 9 and CentOS Stream 9 ship sqlite 3.34.1.

Fixing it needs no root, because the wheel carries its own sqlite:

```bash
pip install 'amd-aorta[chat-sqlite]'     # or: pip install pysqlite3-binary
```

Nothing else changes. Chat checks the version before it opens the index and
swaps in `pysqlite3` only when the built-in is too old, so a current distro
installs nothing and behaves identically. When the build is too old *and* the
wheel is absent, the error names the package to install rather than surfacing a
message from inside the extension.

## Verify

```bash
aorta chat --help          # the extra is installed
aorta chat config show     # the profile resolves
aorta chat tools           # the agent tools loaded
```

If `aorta chat` reports that it needs the `chat-cli` extra on a machine where
you just installed it, check the interpreter version against the table above —
that is what an extra resolving to nothing looks like from the outside.
