import pytest

from aorlta.config import SITE_ENV


@pytest.fixture(autouse=True)
def _no_site_file(monkeypatch):
    """A site file named in the environment would change every config a test loads."""
    monkeypatch.delenv(SITE_ENV, raising=False)
