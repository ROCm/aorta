import pytest

from aorlta.orchestrator.fakes import FakeRunSpec


@pytest.fixture
def make_spec(tmp_path):
    """A FakeRunSpec rooted in this test's directory; `name` keeps several runs of one test apart."""

    def make(name: str = "a", **kw) -> FakeRunSpec:
        return FakeRunSpec(root=str(tmp_path / name / "runs"), **kw)

    return make
