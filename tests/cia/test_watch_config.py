"""The shipped config describes what Watch does, and reaches an installed Watch.

Two ways a config file lies. It can carry keys nothing reads -- ``model``,
``poll_interval_min_sec`` and ``poll_interval_max_sec`` had no reader, under a
comment promising an adaptive interval "tuned to step cadence" that was never
implemented, so the file described behaviour a reader could not get and offered
settings that changed nothing.

And it can fail to ship. Watch loads it through ``Path(__file__).parent``, so
outside ``package-data`` it is absent from a wheel, ``_load_watch_config``
returns ``{}``, and the built-in defaults apply with no error: a different poll
interval and a shorter list of expectations than the file here specifies.
"""

from __future__ import annotations

import pathlib
import tomllib

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
CONFIG = ROOT / "src" / "aorta" / "cia" / "watch" / "watch_config.yaml"
SOURCES = " ".join(
    p.read_text(encoding="utf-8")
    for p in (ROOT / "src" / "aorta" / "cia").rglob("*.py")
)


@pytest.fixture(scope="module")
def config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


class TestEveryKeyHasAReader:
    """A setting nothing reads is a promise the file cannot keep."""

    def test_watch_keys(self, config):
        unread = [k for k in config["watch"] if f'"{k}"' not in SOURCES]
        assert not unread, f"nothing reads these: {unread}"

    def test_log_finder_keys(self, config):
        unread = [k for k in config["log_finder"] if f'"{k}"' not in SOURCES]
        assert not unread, f"nothing reads these: {unread}"

    @pytest.mark.parametrize(
        "gone", ["model", "poll_interval_min_sec", "poll_interval_max_sec"]
    )
    def test_the_keys_with_no_reader_are_gone(self, config, gone):
        assert gone not in config["watch"]

    def test_the_keys_that_do_work_remain(self, config):
        assert config["watch"]["confidence_threshold"] == 0.70
        assert config["watch"]["poll_interval_sec"] == 60
        assert len(config["watch"]["expectations"]) > 3


class TestItDoesNotDescribeWhatItDoesNotDo:
    def test_no_adaptive_polling_is_promised(self):
        text = CONFIG.read_text(encoding="utf-8").lower()
        for claim in ("self-tunes", "updated dynamically", "re-arms", "backs off when log"):
            assert claim not in text, f"still promises: {claim}"

    def test_the_interval_is_described_as_the_constant_it_is(self):
        assert "constant" in CONFIG.read_text(encoding="utf-8").lower()


class TestItReachesAnInstalledWatch:
    @pytest.fixture(scope="class")
    def package_data(self) -> list[str]:
        data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        return data["tool"]["setuptools"]["package-data"]["aorta"]

    def test_the_config_is_in_the_wheel(self, package_data):
        """Absent, the loader returns {} and the defaults apply silently."""
        assert "cia/watch/watch_config.yaml" in package_data

    def test_the_path_is_relative_to_the_package(self, package_data):
        entry = next(p for p in package_data if p.endswith("watch_config.yaml"))
        assert (ROOT / "src" / "aorta" / entry).is_file(), entry

    def test_the_loader_still_looks_beside_itself(self):
        """Which is why it has to be packaged rather than found on the tree."""
        import inspect

        from aorta.cia.watch import poll

        assert "Path(__file__).parent" in inspect.getsource(poll._load_watch_config)


def test_the_defaults_in_code_match_the_file_where_both_exist(config):
    """A fallback that disagrees with the file is a third behaviour."""
    import inspect

    from aorta.cia.watch import poll

    source = inspect.getsource(poll.poll_jobs)
    assert f'"confidence_threshold", {config["watch"]["confidence_threshold"]}' in source
