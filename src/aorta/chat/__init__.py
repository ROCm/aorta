"""``aorta chat`` functionality: LangGraph agent, RAG, providers, tools.

Everything under this package is plain Python. There is deliberately no Click
here -- the whole Click surface for ``aorta chat`` lives in
:mod:`aorta.cli.chat`, which is the single sanctioned importer of this package
from within ``aorta.*``. ``tests/cli/test_chat_boundaries.py`` enforces both
halves of that rule against the AST.

Nothing heavy is imported at module scope. ``aorta.cli`` imports every command
module eagerly, so a langchain import here would land on the critical path of
``aorta --help`` for every user, including those who never installed the chat
extra. Import the submodule you need:

    from aorta.chat.session import invoke_agent
"""

from __future__ import annotations

#: Install hint shared by the CLI's ``_load`` helper and the extras-gated
#: providers, so a missing extra reads the same way wherever it surfaces.
INSTALL_HINT = "pip install 'amd-aorta[chat-cli]'"

__all__ = ["INSTALL_HINT"]
