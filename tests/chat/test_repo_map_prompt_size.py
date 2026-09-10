"""The cap on how much repo map reaches a prompt.

`plan_node` injects the map straight into a system message. AORTA's real map is
around 3 MB -- several hundred thousand tokens -- which overflows any context
window, and on a metered endpoint does so at a price. The cap makes that safe
only because `search_repo_map` still reads the whole file, so these tests pin
both halves of that bargain.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aorta.chat.config import settings
from aorta.chat.rag.repo_map import load_repo_map


@pytest.fixture()
def big_map(tmp_path: Path, monkeypatch) -> str:
    """A repo map far larger than any sensible prompt budget."""
    text = "".join(f"line {i} of the repository map\n" for i in range(4000))
    path = tmp_path / "repo_map.md"
    path.write_text(text, encoding="utf-8")
    monkeypatch.setattr(settings, "repo_map_path", str(path))
    monkeypatch.setattr(settings, "repo_map_prompt_max_chars", 1_000)
    return text


class TestTruncation:
    def test_a_large_map_is_capped(self, big_map: str):
        loaded = load_repo_map()
        assert len(loaded) < len(big_map)

    def test_the_beginning_is_preserved(self, big_map: str):
        """Truncation keeps the top of the tree, which is the useful part."""
        assert load_repo_map().startswith("line 0 of the repository map")

    def test_the_model_is_told_it_was_truncated_and_what_to_do(self, big_map: str):
        loaded = load_repo_map()
        assert "truncated" in loaded
        assert "search_repo_map" in loaded

    def test_the_notice_reports_both_sizes(self, big_map: str):
        loaded = load_repo_map()
        assert f"{len(big_map):,}" in loaded

    def test_an_explicit_limit_wins_over_the_setting(self, big_map: str):
        assert len(load_repo_map(max_chars=200)) < len(load_repo_map())

    def test_zero_means_the_whole_file(self, big_map: str):
        """What search_repo_map relies on; without it the cap loses information."""
        assert load_repo_map(max_chars=0) == big_map


class TestNoTruncationNeeded:
    def test_a_small_map_is_returned_verbatim(self, tmp_path: Path, monkeypatch):
        text = "├── src/\n└── README.md\n"
        path = tmp_path / "repo_map.md"
        path.write_text(text, encoding="utf-8")
        monkeypatch.setattr(settings, "repo_map_path", str(path))
        monkeypatch.setattr(settings, "repo_map_prompt_max_chars", 20_000)
        loaded = load_repo_map()
        assert loaded == text
        assert "truncated" not in loaded

    def test_a_missing_map_still_explains_itself(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(settings, "repo_map_path", str(tmp_path / "absent.md"))
        assert "not yet generated" in load_repo_map()


class TestSearchToolReadsTheFullMap:
    def test_search_repo_map_matches_beyond_the_prompt_cap(
        self, big_map: str
    ):
        """A term only present late in the file must still be findable."""
        from aorta.chat.tools.search import search_repo_map

        result = search_repo_map.invoke({"query": "line 3999"})
        assert "line 3999" in result


class TestDefault:
    def test_the_default_cap_is_promptable(self):
        """Roughly 5k tokens: room for the map without crowding out the query."""
        assert 0 < settings.repo_map_prompt_max_chars <= 50_000
