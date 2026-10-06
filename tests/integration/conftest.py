"""The integration suite needs a running model and the chat stack to reach it.

These are marked ``integration`` and skip at run time when there is no endpoint,
but a skip at run time is too late for an import: the modules here import
``langchain_core`` and ``aorta.chat.graph`` at the top, and the CPU lane
installs neither before collecting the whole tree. Collection then fails for the
directory instead of skipping it.

Same guard tests/chat and tests/cia use, for the same reason.
"""

from __future__ import annotations

import importlib.util

CHAT_EXTRA_INSTALLED = importlib.util.find_spec("langchain_core") is not None

if not CHAT_EXTRA_INSTALLED:  # pragma: no cover - exercised on a base install
    collect_ignore_glob = ["test_*.py"]
