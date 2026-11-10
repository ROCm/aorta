import os

import pytest

from aorlta.config import SITE_ENV


@pytest.fixture(autouse=True)
def _no_site_file(monkeypatch):
    """A site file named in the environment would change every config a test loads."""
    monkeypatch.delenv(SITE_ENV, raising=False)


@pytest.fixture(autouse=True)
def _hipcc_as_it_was():
    """Building the ASan checker's factory sets $HIPCC for aorta's canary when it is unset: no test leaves it set."""
    before = os.environ.get("HIPCC")
    yield
    if before is None:
        os.environ.pop("HIPCC", None)
    else:
        os.environ["HIPCC"] = before
