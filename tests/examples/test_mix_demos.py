"""Mixing demonstrations for a continual cycle: whole episodes, one system prompt, reproducible."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples" / "rl"))

import mix_demos  # noqa: E402


def write_demos(path: Path, source: str, episodes: int, steps: int, sha: str = "a" * 64) -> Path:
    rows = [{"scenario_id": f"s{e}", "source": source, "episode": 0, "step": k + 1, "prompt": "p",
             "completion": f"{source}-{e}-{k}"} for e in range(episodes) for k in range(steps)]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    path.with_suffix(".stats.json").write_text(json.dumps({"system_sha256": sha}))
    return path


def rows(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_whole_episodes_are_sampled_reproducibly(tmp_path):
    new = write_demos(tmp_path / "new.jsonl", "column:new", 5, 3)
    old = write_demos(tmp_path / "old.jsonl", "column:old", 40, 4)
    outs = []
    for i in range(2):
        out = tmp_path / f"mix{i}.jsonl"
        assert mix_demos.main(["--out", str(out), str(new), f"{old}:7"]) == 0
        outs.append(rows(out))
    assert outs[0] == outs[1]
    mixed = outs[0]
    by_episode = {}
    for r in mixed:
        by_episode.setdefault((r["source"], r["scenario_id"]), []).append(r["step"])
    assert sum(1 for k in by_episode if k[0] == "column:new") == 5
    assert sum(1 for k in by_episode if k[0] == "column:old") == 7
    assert all(steps == [1, 2, 3] for k, steps in by_episode.items() if k[0] == "column:new")
    assert all(steps == [1, 2, 3, 4] for k, steps in by_episode.items() if k[0] == "column:old")
    stats = json.loads((tmp_path / "mix0.stats.json").read_text())
    assert stats["system_sha256"] == "a" * 64 and stats["episodes"] == 12 and stats["samples"] == 5 * 3 + 7 * 4
    other = tmp_path / "mix-seed.jsonl"
    assert mix_demos.main(["--out", str(other), "--seed", "1", str(new), f"{old}:7"]) == 0
    assert rows(other) != mixed


def test_inputs_from_two_system_prompts_are_refused(tmp_path, capsys):
    a = write_demos(tmp_path / "a.jsonl", "column:a", 3, 2, sha="a" * 64)
    b = write_demos(tmp_path / "b.jsonl", "column:b", 3, 2, sha="b" * 64)
    assert mix_demos.main(["--out", str(tmp_path / "m.jsonl"), str(a), str(b)]) == 3
    assert "2 system prompt(s)" in capsys.readouterr().err
    assert not (tmp_path / "m.jsonl").exists()


def test_a_file_without_stats_or_an_episode_twice_is_refused(tmp_path):
    a = write_demos(tmp_path / "a.jsonl", "column:a", 3, 2)
    bare = tmp_path / "bare.jsonl"
    bare.write_text(a.read_text())
    assert mix_demos.main(["--out", str(tmp_path / "m.jsonl"), str(bare)]) == 3
    assert mix_demos.main(["--out", str(tmp_path / "m.jsonl"), str(a), str(a)]) == 3
    assert mix_demos.main(["--out", str(tmp_path / "m.jsonl"), f"{a}:-1"]) == 3


def test_an_episode_is_named_by_its_iteration_too(tmp_path):
    a = write_demos(tmp_path / "a.jsonl", "wire:/run/wire.jsonl", 1, 2)
    rows_a = rows(a)
    both = [dict(r, iteration=i) for i in (1, 2) for r in rows_a]
    a.write_text("".join(json.dumps(r) + "\n" for r in both))
    assert mix_demos.main(["--out", str(tmp_path / "m.jsonl"), str(a)]) == 0
    assert json.loads((tmp_path / "m.stats.json").read_text())["episodes"] == 2
