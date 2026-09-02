# What leaves the machine

When `llm_provider` is `openai` or `litellm`, every prompt AORTA builds is sent
to a third party. Those prompts are not just your question: they carry retrieved
chunks, tool output, and — because the assistant indexes your own sweep results
— the contents of your `matrix.json` and `env.json`.

Each of those is individually reasonable and together they are an exfiltration
path. So outbound text is rewritten by default, and chat says so the first time
it changes anything.

## What is redacted

**Filesystem paths and IP addresses. That is the whole list.**

| Rewritten | Not rewritten |
| --- | --- |
| Absolute filesystem paths | API keys, tokens, passwords |
| IPv4 addresses | Hostnames and cluster names |
| IPv6 addresses | Customer, project or ticket identifiers |
| | Usernames, e-mail addresses, employee IDs |
| | GPU serial numbers, node IDs, MAC addresses |
| | Source code and its comments |

The rewriters are the same ones `aorta bundle` uses to make a diagnostic bundle
shareable — one implementation, one place to keep correct — applied here to the
message list on its way out.

## Read that table again before you rely on it

The limitation is worth stating plainly, because a user who trusts an overstated
claim is worse off than one who knows the boundary:

- **Credentials are not scrubbed from message text.** Chat never *puts* a key in
  a message body — keys travel as request headers, which this never sees — but
  if a key reaches a prompt some other way (it is in a file the model read, or
  in the output of `run_terminal_command`, or you typed it), it is sent verbatim.
- **Customer and project identifiers are not scrubbed.** A customer name
  survives redaction only *incidentally*, when it happens to sit inside a path
  that gets rewritten. A ticket ID in `matrix.json`, a cluster name in a
  hostname field, or a codename in a comment goes out untouched.
- **Redaction is not a compliance control.** Treat it as a way to avoid
  broadcasting your directory layout and internal addressing, not as a
  guarantee about what a prompt contains.

If your threat model needs more than that, use `llm_provider = "vllm"` against a
model you host — **and** keep `embedding_provider = "local"`, which is the
default. A local LLM alone does not mean nothing leaves the machine: with
`embedding_provider = "remote"`, indexing sends the corpus text (including your
rendered run artifacts) to the embeddings API, and every retrieval sends the
query. Neither call goes through the chat-message redactor described here.
Together, those two settings are what make the claim hold.

## The notice

On the first outbound request of a session that actually changed something, chat
prints one line:

```
aorta chat: redacted 3 filesystem paths and 1 IPv4 address from the outbound request.
Disable with --no-redact, or 'redact = false' in ~/.config/aorta/chat.toml.
```

Three deliberate properties:

- **It names what was removed.** A gate that works silently trains people to
  distrust the tool when an answer looks wrong for an unrelated reason.
- **It names the way out**, because knowing something was removed and not how to
  stop it leaves you stuck.
- **It goes to stderr**, and to the real stderr rather than the one quiet mode
  redirects. `--json` and `--plain` stay clean on stdout, so a piped session
  still yields parseable output.

It appears once per session, and only when a redaction actually happened — a
session whose prompts contained no paths is not told about a rewrite that never
occurred.

### In the web UI

stderr is the *server's* console under `aorta chat ui`, which is not a place the
person typing can see, so the notice is also delivered into the browser session
that caused it — once, after the answer it applies to.

"Once per session" is therefore per **browser** session, not per server process.
The state lives on a `NoticeState` the Chainlit handler owns and binds around
each turn (`redaction.use_notice_state`), so one user's redaction cannot consume
another user's disclosure. The CLI front doors are one session per process and
use the process-wide state without binding anything.

The welcome message states the scope before you type, since the per-request
notice necessarily arrives after you have already sent something.

## Turning it off

```bash
aorta chat --no-redact ask "..."     # one invocation
```

```toml
# ~/.config/aorta/chat.toml
redact = false                        # permanently
```

The flag is tri-state on purpose: omitting it means "no opinion", so a profile
setting `redact = false` stays in force rather than being silently re-enabled by
the default.

Turning it off is a reasonable choice when the model is local, or when the paths
matter to the answer — "why did this fail" questions sometimes hinge on the
exact directory layout, and a rewritten path can make the model's reasoning
worse. It is a bad choice against a metered public endpoint on a customer node.
