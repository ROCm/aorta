"""Shared fixtures for the ``aorta`` CLI test suite."""

from __future__ import annotations

from collections import namedtuple

import pytest

#: ``sys.version_info`` is not constructible, and a bare tuple would not carry
#: the ``.major`` / ``.minor`` that the chat refusal messages format.
_VersionInfo = namedtuple("version_info", "major minor micro releaselevel serial")


@pytest.fixture()
def pin_python(monkeypatch):
    """Pin the interpreter version that ``aorta/cli/chat.py`` sees.

    ``_require_python`` guards every chat entry point, so on 3.10 -- which the
    CPU matrix covers, and where the chat extra cannot be installed at all
    (Decision 13a) -- it is what raises, ahead of the import paths several of
    these tests are actually about. Pinning the version keeps each test
    measuring what it claims on every interpreter in the matrix rather than on
    four out of five, and lets the 3.10 refusal be asserted from all of them
    instead of only from the one leg that happens to run it.
    """

    def _pin(minor: int = 11) -> None:
        from aorta.cli import chat as cli_chat

        monkeypatch.setattr(cli_chat.sys, "version_info", _VersionInfo(3, minor, 0, "final", 0))

    return _pin
