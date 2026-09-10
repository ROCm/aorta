"""The last gate before an index is published, which has to fail closed.

The script exists to prove every chunk in a publishable index came from a
git-tracked path, because the index embeds source text verbatim. A chunk whose
provenance it cannot read is therefore the one case most worth stopping on --
and skipping those let an index holding unverifiable text still print ``OK``,
including one where nothing at all was collected.
"""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import subprocess
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "verify_chat_index_sources.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("verify_chat_index_sources", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def verify():
    return _load_script()


def _index_with(tmp_path: Path, metadata_rows: list[str]) -> Path:
    """An index file holding one chunk table with the given metadata blobs."""
    index = tmp_path / "index.sqlite"
    conn = sqlite3.connect(index)
    conn.execute("CREATE TABLE chunks_aorta_local (metadata TEXT)")
    conn.executemany(
        "INSERT INTO chunks_aorta_local (metadata) VALUES (?)",
        [(row,) for row in metadata_rows],
    )
    conn.commit()
    conn.close()
    return index


class TestIndexSources:
    def test_well_formed_metadata_is_collected(self, verify, tmp_path: Path):
        index = _index_with(
            tmp_path,
            [json.dumps({"source": "src/aorta/cli/chat.py"}), json.dumps({"source": "README.md"})],
        )
        assert verify.index_sources(index) == {"src/aorta/cli/chat.py", "README.md"}

    def test_metadata_that_is_not_json_stops_the_run(self, verify, tmp_path: Path):
        index = _index_with(tmp_path, [json.dumps({"source": "README.md"}), "{not json"])
        with pytest.raises(SystemExit) as exc:
            verify.index_sources(index)
        assert "not" in str(exc.value)

    @pytest.mark.parametrize("row", [{}, {"source": ""}, {"source": None}, {"source": 3}])
    def test_a_chunk_with_no_usable_source_stops_the_run(self, verify, tmp_path: Path, row):
        index = _index_with(tmp_path, [json.dumps(row)])
        with pytest.raises(SystemExit) as exc:
            verify.index_sources(index)
        assert "provenance" in str(exc.value)


class TestMain:
    def test_an_index_yielding_no_sources_is_not_reported_as_publishable(
        self, verify, tmp_path: Path, capsys
    ):
        """"All zero paths are tracked" is not a pass this guard may print."""
        index = _index_with(tmp_path, [])
        repo = tmp_path / "repo"
        repo.mkdir()
        subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)

        assert verify.main([str(index), "--repo", str(repo)]) == 1
        assert "nothing" in capsys.readouterr().err
