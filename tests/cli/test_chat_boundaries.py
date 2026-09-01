"""Import-boundary discipline for ``aorta chat``.

Four invariants, all of which conventions alone would erode:

1. ``aorta.chat`` contains no Click. The Click layer is ``aorta.cli``, and
   nothing else.
2. Nothing in ``aorta.*`` imports ``aorta.chat``, except ``aorta.cli.chat``,
   which is the single sanctioned entry. This is what keeps a later extraction
   into its own distribution a directory move rather than an archaeology
   project.
3. ``aorta.cli.chat`` imports ``aorta.chat`` only inside function bodies. Its
   module scope is click plus stdlib.
4. ``import aorta.cli`` does not pull in langchain. Rules 1-3 are the mechanism;
   this is the property they exist to protect, so it is asserted directly.

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
_CLI_CHAT = _SRC / "aorta" / "cli" / "chat.py"

#: The one module allowed to import ``aorta.chat``.
_SANCTIONED_IMPORTER = _CLI_CHAT

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


def test_only_cli_chat_imports_aorta_chat():
    """Keeps the dependency arrow pointing one way: chat -> core, never back."""
    offenders: list[str] = []
    for path in _python_files(_SRC / "aorta"):
        if path == _SANCTIONED_IMPORTER or _CHAT_PKG in path.parents or path.parent == _CHAT_PKG:
            continue
        rel = path.relative_to(_REPO_ROOT)
        for node in ast.walk(_parse(path)):
            for name in _imported_names(node):
                if name == "aorta.chat" or name.startswith("aorta.chat."):
                    offenders.append(f"{rel}:{node.lineno}: imports {name!r}")
    assert not offenders, (
        "aorta.chat imported from outside aorta/cli/chat.py:\n  "
        + "\n  ".join(offenders)
        + "\n\naorta.chat may import aorta.*; the reverse must go through "
        "aorta/cli/chat.py so the subpackage stays extractable."
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


def test_cli_chat_imports_aorta_chat_only_inside_functions():
    """Every ``aorta.chat`` import must sit in a callback, behind ``_load``."""
    module_scope = {name for _, name in _module_scope_imports(_parse(_CLI_CHAT))}
    leaked = {n for n in module_scope if n == "aorta.chat" or n.startswith("aorta.chat.")}
    assert not leaked, (
        f"aorta/cli/chat.py imports {sorted(leaked)} at module scope; move it "
        "into the command callback so the chat extra stays optional."
    )


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
    "langchain_huggingface",
    "langgraph",
    "openai",
    "pydantic",
    "pydantic_settings",
    "sentence_transformers",
    "torch",
    "chainlit",
    "chromadb",
    "sqlite_vec",
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
