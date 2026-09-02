"""Import-boundary discipline for ``aorta chat``.

Four invariants, all of which conventions alone would erode:

1. ``aorta.chat`` contains no Click. The Click layer is ``aorta.cli``, and
   nothing else.
2. Nothing in ``aorta.*`` imports ``aorta.chat`` except the two sanctioned
   importers below. This is what keeps a later extraction into its own
   distribution a directory move rather than an archaeology project.
3. Each sanctioned importer imports ``aorta.chat`` only inside function bodies.
4. ``import aorta.cli`` does not pull in langchain. Rules 1-3 are the mechanism;
   this is the property they exist to protect, so it is asserted directly.

**Why there are two sanctioned importers, not one.** ``aorta/cli/chat.py`` is
the original: the Click entry for the chat command. ``aorta/agent/llm.py`` was
added by Phase 5b, because locked Decision 7a makes the chat provider factory
*the* provider layer and ports the agent's proposer onto it -- so that
``vllm`` / ``openai`` / ``litellm`` are configured once for both front doors
instead of twice. That decision and this rule are in genuine tension, and the
resolution is recorded here rather than left as an undocumented erosion:

* The edge is **deferred**, so the cost is zero for anyone not using it, and
  ``import aorta.agent`` stays free of langchain. Asserted below.
* The edge is **one function**, ``ChatProviderProposer._chat_model``.
* Extraction is still a directory move, but it would have to take the provider
  layer with it or leave a shim -- ``aorta agent --llm-backend=openai`` depends
  on it. That is the price of 7a, and it is deliberate.

If a third importer ever wants to join, that is the signal that the provider
layer belongs in core rather than under ``aorta.chat``, and this list should be
replaced by that move instead of being extended again.

These live under ``tests/cli/`` rather than ``tests/chat/`` on purpose: they are
pure AST and stdlib, so they must run on a base install, where ``tests/chat/``
is skipped for want of the chat extra.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
_CHAT_PKG = _SRC / "aorta" / "chat"
_CIA_PKG = _SRC / "aorta" / "cia"
_CLI_CHAT = _SRC / "aorta" / "cli" / "chat.py"
_AGENT_LLM = _SRC / "aorta" / "agent" / "llm.py"

#: The only modules allowed to import ``aorta.chat``, and each one's reason.
#: See the module docstring before adding a third.
_SANCTIONED_IMPORTERS = {
    _CLI_CHAT: "the Click entry for `aorta chat`",
    _AGENT_LLM: "Decision 7a: the agent proposer on the shared provider layer",
}

#: Third-party packages ``aorta/cli/chat.py`` may import at module scope.
#: Everything else there must be stdlib.
_CLI_CHAT_ALLOWED_THIRD_PARTY = frozenset({"click"})


def _python_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _imported_names(node: ast.AST) -> list[str]:
    """Dotted module names imported by a single Import/ImportFrom node."""
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if isinstance(node, ast.ImportFrom):
        if node.level > 0 or not node.module:
            # Relative: internal by definition, and ``node.module`` carries no
            # package prefix, so recording it would mis-flag it.
            return []
        return [node.module]
    return []


def _module_scope_imports(tree: ast.Module) -> list[tuple[int, str]]:
    """Imports at module level only -- not the ones nested in functions.

    Walks ``tree.body`` and the bodies of module-level ``if`` / ``try``
    statements (where a conditional import legitimately lives), but never
    descends into a function or class.
    """
    found: list[tuple[int, str]] = []
    pending: list[ast.stmt] = list(tree.body)
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            found.extend((node.lineno, name) for name in _imported_names(node))
        elif isinstance(node, (ast.If, ast.Try)):
            pending.extend(node.body)
            pending.extend(node.orelse)
            pending.extend(getattr(node, "finalbody", []))
            for handler in getattr(node, "handlers", []):
                pending.extend(handler.body)
    return found


# ── Rule 1: no Click under aorta.chat ─────────────────────────────────────


def test_aorta_chat_contains_no_click():
    """``aorta.chat`` is the functional layer; Click lives in ``aorta.cli``.

    Catches both an ``import click`` and a ``@click.command`` applied to a
    function that happened to get a click import from elsewhere.
    """
    offenders: list[str] = []
    for path in _python_files(_CHAT_PKG):
        tree = _parse(path)
        rel = path.relative_to(_REPO_ROOT)
        for node in ast.walk(tree):
            for name in _imported_names(node):
                if name == "click" or name.startswith("click."):
                    offenders.append(f"{rel}:{node.lineno}: imports {name!r}")
            # A decorator spelled ``@click.command`` / ``@group.command``
            # cannot appear without an import, but a `from x import click`
            # alias could -- so check the decorator text too.
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                for decorator in node.decorator_list:
                    text = ast.unparse(decorator)
                    if text.split("(")[0].split(".")[0] == "click":
                        offenders.append(f"{rel}:{node.lineno}: @{text}")
    assert not offenders, "Click found under src/aorta/chat/:\n  " + "\n  ".join(offenders)


# ── Rule 2: only aorta.cli.chat may import aorta.chat ─────────────────────


def test_only_sanctioned_modules_import_aorta_chat():
    """Keeps the dependency arrow pointing one way: chat -> core, never back."""
    offenders: list[str] = []
    for path in _python_files(_SRC / "aorta"):
        if path in _SANCTIONED_IMPORTERS or _CHAT_PKG in path.parents or path.parent == _CHAT_PKG:
            continue
        rel = path.relative_to(_REPO_ROOT)
        for node in ast.walk(_parse(path)):
            for name in _imported_names(node):
                if name == "aorta.chat" or name.startswith("aorta.chat."):
                    offenders.append(f"{rel}:{node.lineno}: imports {name!r}")
    allowed = ", ".join(sorted(p.relative_to(_SRC).as_posix() for p in _SANCTIONED_IMPORTERS))
    assert not offenders, (
        "aorta.chat imported from an unsanctioned module:\n  "
        + "\n  ".join(offenders)
        + f"\n\naorta.chat may import aorta.*; the reverse may only go through {allowed}, "
        "so the subpackage stays extractable. Adding a third importer is the signal "
        "that the provider layer belongs in core -- see this module's docstring."
    )


@pytest.mark.parametrize(
    "path", sorted(_SANCTIONED_IMPORTERS, key=lambda p: p.as_posix()), ids=lambda p: p.name
)
def test_sanctioned_importers_defer_every_chat_import(path: Path):
    """The exception is deferred imports only.

    A module-scope ``aorta.chat`` import in either sanctioned module would put
    langchain on the critical path of every ``aorta`` invocation, which is the
    cost rule 2 exists to prevent -- the extractability argument is the second
    reason, not the only one.
    """
    module_scope = {name for _, name in _module_scope_imports(_parse(path))}
    leaked = {n for n in module_scope if n == "aorta.chat" or n.startswith("aorta.chat.")}
    assert not leaked, (
        f"{path.relative_to(_SRC)} imports {sorted(leaked)} at module scope; move it "
        "into the function that needs it so the chat extra stays optional."
    )


# ── Rule 3: aorta/cli/chat.py defers every chat import ────────────────────


def test_cli_chat_module_scope_is_click_and_stdlib():
    """The reason ``aorta --help`` does not import langchain.

    ``aorta/cli/__init__.py`` imports every command module eagerly, so anything
    at this module's scope is on the critical path of every single ``aorta``
    invocation -- including for users who never installed the chat extra.
    """
    stdlib = set(sys.stdlib_module_names)
    offenders: list[str] = []
    for lineno, name in _module_scope_imports(_parse(_CLI_CHAT)):
        top = name.split(".", 1)[0]
        if top in stdlib or top in _CLI_CHAT_ALLOWED_THIRD_PARTY:
            continue
        offenders.append(f"aorta/cli/chat.py:{lineno}: imports {name!r} at module scope")
    assert not offenders, "\n  ".join(["module scope must be click + stdlib:", *offenders])


def test_load_turns_a_missing_extra_into_an_install_hint(monkeypatch):
    """An absent external dependency must read as advice, not as a traceback."""
    import click

    from aorta.cli import chat as cli_chat

    def _missing_langchain(name: str):
        raise ModuleNotFoundError("No module named 'langchain'", name="langchain")

    monkeypatch.setattr(cli_chat.importlib, "import_module", _missing_langchain)
    with pytest.raises(click.ClickException) as exc:
        cli_chat._load("session")
    assert "amd-aorta[chat-cli]" in str(exc.value)
    assert "langchain" in str(exc.value)


def test_load_lets_a_broken_chat_submodule_surface():
    """The other half of the distinction, and the reason ``_load`` exists.

    A ``ModuleNotFoundError`` naming an ``aorta.chat`` module is a real bug -- a
    typo, or a file that never got packaged -- and advising the user to install
    something they already have would bury it. Same rule as ``cli/bench.py``.
    """
    from aorta.cli.chat import _load

    with pytest.raises(ModuleNotFoundError) as exc:
        _load("no_such_submodule")
    assert exc.value.name == "aorta.chat.no_such_submodule"


# ── Rule 4: the property all of the above protects ────────────────────────

_HEAVY_PREFIXES = (
    "langchain",
    "langchain_core",
    "langchain_community",
    "langchain_openai",
    "langchain_text_splitters",
    "langgraph",
    "openai",
    "pydantic",
    "pydantic_settings",
    # Deleted in Phase 4 (Decision 19a) and listed anyway: they are the CUDA-on
    # -AMD hazard, so a reappearance in aorta.cli's import graph is worth a
    # failing test even though nothing imports them today.
    "sentence_transformers",
    "torch",
    "chainlit",
    "chromadb",
    "sqlite_vec",
    "fastembed",
    "onnxruntime",
    "rich",
)


def test_importing_aorta_cli_does_not_pull_in_langchain():
    """Asserted in a subprocess: ``sys.modules`` in-process is already dirty.

    pytest has imported the world by the time this runs, so the only honest
    measurement is a fresh interpreter that imports nothing but ``aorta.cli``.
    """
    import subprocess

    probe = (
        "import sys, json, aorta.cli;"
        f"heavy={_HEAVY_PREFIXES!r};"
        "print(json.dumps(sorted(m for m in sys.modules "
        "if m.split('.')[0] in heavy)))"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
    )
    leaked = __import__("json").loads(out.stdout)
    assert leaked == [], (
        f"import aorta.cli pulled in {leaked}. Every chat import in "
        "aorta/cli/chat.py must live inside a command callback."
    )


def test_importing_the_agent_proposer_does_not_pull_in_langchain():
    """The other half of the Decision 7a exception being affordable.

    ``aorta.agent.llm`` is imported by every ``aorta agent mitigate`` run,
    including the default ``--llm-backend=fake``, which must stay fully offline.
    If its chat seam ever moved to module scope, the fake path would start
    paying for langchain -- and on a base install it would stop working.
    """
    import subprocess

    probe = (
        "import sys, json, aorta.agent.llm;"
        f"heavy={_HEAVY_PREFIXES!r};"
        "print(json.dumps(sorted(m for m in sys.modules "
        "if m.split('.')[0] in heavy)))"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    leaked = __import__("json").loads(out.stdout)
    assert leaked == [], (
        f"import aorta.agent.llm pulled in {leaked}. The chat provider seam must "
        "stay inside ChatProviderProposer._chat_model."
    )


def test_the_default_agent_backend_is_offline_and_needs_no_extra():
    """``fake`` is the default, and is what the test suite and --dry-run rely on."""
    from aorta.agent.llm import FakeLLMProposer, make_proposer

    assert isinstance(make_proposer("fake"), FakeLLMProposer)


def test_importing_aorta_cli_does_not_pull_in_asyncio():
    """A stdlib guard, because the rule's reason is cost, not third-partyness.

    ``asyncio`` costs ~33 ms to import -- about 12% of ``aorta --help`` -- and
    only ``aorta chat`` needs it. It passes the module-scope rule on a
    technicality, so it gets its own assertion.
    """
    import subprocess

    probe = "import sys, aorta.cli; print('asyncio' in sys.modules)"
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "False", (
        "import aorta.cli now pulls in asyncio; move the import into the "
        "command callback that awaits the agent."
    )


# ── The same two rules, for aorta.cia ─────────────────────────────────────

#: What only the chat extras provide. ``aorta.cia`` may pull its own extra --
#: dspy and what dspy pulls -- but reaching any of these would mean the agents
#: cannot be used without installing a chatbot, which is the property the
#: package was placed outside ``aorta.chat`` to keep.
_CHAT_ONLY_PREFIXES = (
    "langchain",
    "langchain_core",
    "langchain_community",
    "langchain_openai",
    "langchain_text_splitters",
    "langgraph",
    "chainlit",
    "chromadb",
    "sentence_transformers",
    "fastembed",
    "torch",
)


def test_aorta_cia_contains_no_click():
    """Same rule as ``aorta.chat``, same reason: Click lives in ``aorta.cli``.

    The agents arrived from a repository where each was its own console script,
    so the argparse entry points were dropped on the way in. This is what keeps
    them from growing back.
    """
    offenders: list[str] = []
    for path in _python_files(_CIA_PKG):
        rel = path.relative_to(_REPO_ROOT)
        for node in ast.walk(_parse(path)):
            for name in _imported_names(node):
                if name == "click" or name.startswith("click."):
                    offenders.append(f"{rel}:{node.lineno}: imports {name!r}")
    assert not offenders, "Click found under src/aorta/cia/:\n  " + "\n  ".join(offenders)


def test_importing_the_agents_does_not_require_the_chat_extras():
    """Launch, Watch and Autopsy are useful with no chatbot present.

    From a script, from CI, from ``aorta`` itself. Core has two dependencies;
    the chat extras add eleven plus Chainlit, and a cluster job submitter has no
    business requiring them. Measured in a fresh interpreter, because
    ``sys.modules`` is already dirty by the time pytest runs this.
    """
    import subprocess

    probe = (
        "import sys, json, aorta.cia, aorta.cia.launch, aorta.cia.autopsy.orchestrator;"
        f"chat_only={_CHAT_ONLY_PREFIXES!r};"
        "print(json.dumps(sorted(m for m in sys.modules "
        "if m.split('.')[0] in chat_only)))"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    leaked = __import__("json").loads(out.stdout)
    assert leaked == [], (
        f"importing the agents pulled in {leaked}, which only the chat extras "
        "provide. The agents must stay usable on a base install plus their own "
        "extra."
    )
