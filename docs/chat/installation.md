# Installing `aorta chat`

Chat is an extra, not part of the base install. `pip install amd-aorta` remains
`pyyaml` plus `click`, which is what makes it a few seconds on a customer node;
chat adds roughly 400 MB and ~120 distributions, so it is opted into
deliberately.

## The three extras

| Extra | Adds | Python | Rough size |
| --- | --- | --- | --- |
| `chat-cli` (alias: `chat`) | LangChain / LangGraph, the OpenAI client, sqlite-vec, rich | 3.11+ | ~400 MB |
| `chat-ui` | `chat-cli` plus Chainlit, for `aorta chat ui` | 3.11–3.13 | ~460 MB |
| `chat-all` | `chat-ui` plus LiteLLM, for native Anthropic / Gemini / Bedrock | 3.11–3.13 | ~560 MB |

```bash
# Published package
pip install 'amd-aorta[chat-cli]'

# Editable source checkout
uv pip install -e ".[chat-cli]"
```

`pip install 'amd-aorta[chat]'` is an alias for `chat-cli`, because that is what
most people type first.

**No PyTorch, no CUDA, no GPU.** Retrieval runs a small embedding model on CPU
and generation happens wherever your provider lives. Nothing in these extras
pulls a torch build, which matters on a ROCm node: a dependency that
hard-requires torch makes pip resolve the default PyPI wheel, and that is the
**CUDA** build.

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
