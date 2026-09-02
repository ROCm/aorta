# Adding an agent tool

`aorta chat` ships nine tools — file listing and reading, semantic and regex
search, the repo-map lookup, and four readers over your own run artifacts. A
tenth, the sandboxed shell, is registered only when you set
[`enable_shell_tool`](configuration.md); until then it is in neither the
registry nor the prompts. Your package can add more through the
`aorta.chat_tools` entry-point group, with no edit to AORTA itself.

This is the same mechanism `aorta.workloads`, `aorta.mitigations` and
`aorta.agents` already use, so if you have registered a workload before, none of
this will be new.

## Write the tool

A tool is a LangChain `BaseTool`, which in practice means a function with the
`@tool` decorator. The **docstring is the specification the model reads**, so it
is not a comment — write it for the reader who has to decide whether to call it.

```python
# my_package/chat_tools.py
from langchain_core.tools import tool


@tool
def read_fabric_counters(node: str) -> str:
    """Read the Infinity Fabric performance counters for one node.

    Args:
        node: Hostname of the node to read, as it appears in `list_runs` output.

    Returns:
        One counter per line as `name=value`, or a line beginning with `Error:`
        if the node is unreachable.
    """
    ...
```

Two rules the built-in tools follow and yours should too:

- **Return a string, always.** Errors are returned as text (`"Error: ..."`), not
  raised. A raised exception aborts the graph run; a returned error gives the
  model a chance to correct itself, which is usually what you want.
- **Bound what you touch.** The built-in file tools resolve every path relative
  to the configured source root and refuse anything that escapes it. A tool with
  no such bound is a tool an LLM can point anywhere.

## Register it

```toml
# my_package/pyproject.toml
[project.entry-points."aorta.chat_tools"]
read_fabric_counters = "my_package.chat_tools:read_fabric_counters"
```

**The entry-point name must equal the tool's own name.** They are two strings in
two files and only the second one reaches the model — `bind_tools` sends
`tool.name`, so the provider echoes `tool.name` back, and a registry keyed on a
different entry-point name would never find it. A tool whose names disagree is
skipped at load with a message saying so, rather than being registered and
failing on every call.

## Check it

```bash
pip install -e .        # your package, into the same environment as aorta
aorta chat tools
```

```
grep_code  [aorta]
    Search for a regex pattern across files in the AORTA codebase.
list_files  [aorta]
    List files and directories under the given path inside the AORTA codebase.
...
read_fabric_counters  [my-package]
    Read the Infinity Fabric performance counters for one node.
...
```

`aorta chat tools --json` is the machine-readable form. If your tool is missing,
the reason was printed on stderr — see the failure table below.

## What happens when a plugin is wrong

The registry's failure semantics are deliberately asymmetric, and the asymmetry
is the design:

| Situation | Result |
| --- | --- |
| Entry point fails to import | Logged with the full traceback, **skipped** |
| Entry point resolves to something that is not a `BaseTool` | Logged, **skipped** |
| Entry-point name disagrees with the tool's own name | Logged, **skipped** |
| Two packages register the same tool name | **Raises** |
| A built-in is broken | **Raises** |

A broken third-party package is skipped because one bad install must not take
the whole assistant down. A **collision** raises, because silently shadowing
`run_terminal_command` with someone else's idea of it is not a thing to recover
from quietly — and because the alternative, a load order that quietly picks a
winner, is a bug nobody can reproduce. Rename one of the two.

A broken built-in raises because it is AORTA's own bug: skipping it would
silently drop a shipped tool.

## Where plugin tools are offered

Both tool-calling protocols see them, by different routes:

- **`llm_tool_mode = "native"`** sends every tool's schema through the
  provider's function-calling API, so a plugin tool is offered with no further
  work.
- **`llm_tool_mode = "text"`** has only the prompt, so plugin tools are appended
  to the tool list the model is shown, with their first docstring line and their
  source package. With no plugins installed the prompt is byte-identical to what
  it was before this extension point existed.

The registry is read once when the graph module is imported, so a plugin
installed while a REPL session is open is picked up on the next session.

## Should you use this?

Probably only if you are extending what the *assistant* can do. Two adjacent
seams are often the better fit:

- To add an autonomous, run-to-verdict workflow, register an
  [agent](../agent/agentic-testing-guide.md) under `aorta.agents` instead.
- To *consume* AORTA's results from your own program, you do not need a plugin
  at all — run the CLI and read `matrix.json` and `env.json` through
  `aorta.artifacts`, which needs no chat extra. Note that it is **internal for
  now**: it is importable from a base `pip install amd-aorta` and deliberately
  depends only on the standard library, but it is scoped to what AORTA's own
  tooling needs today and its names may change without a deprecation cycle.
  Pin your AORTA version if you build on it.
