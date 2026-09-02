"""Bundle fixtures for the agent tests.

Autopsy reads a directory, not arguments, so almost every test here needs one.
These build the smallest bundle that ``run_autopsy`` will accept, and let a
test add exactly the evidence it is about.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_MANIFEST = """\
schema_version: '0.1'
job_id: {job_id}
failure_at: '2026-01-01T00:00:00Z'
nodes:
- hostname: ''
  rank: 0
paths:
{paths}
metadata:
  scheduler: slurm
  launcher: sbatch
"""


@pytest.fixture
def make_bundle(tmp_path: Path):
    """Build a bundle directory. ``artifacts`` maps a manifest key to content.

    Content is written under the path the manifest advertises, so an adapter
    finds it the same way it would in a real run. A key with no content is a
    path the manifest promises and the bundle does not have -- which is a case
    worth testing, not an accident.
    """

    def _build(job_id: str = "test-job", **artifacts: str | dict | None) -> Path:
        root = tmp_path / job_id
        root.mkdir(parents=True, exist_ok=True)

        relative = {
            "stderr": "logs/watch.stderr.log",
            "sanitizer_report": "aorta/sanitizer_report.json",
            "aorta_matrix": "aorta/matrix.json",
            "rocgdb_session": "rocgdb/session.log",
        }
        declared = {k: v for k, v in relative.items() if k in artifacts}
        paths = "\n".join(f"  {k}: {v}" for k, v in declared.items()) or "  {}"
        (root / "manifest.yaml").write_text(
            _MANIFEST.format(job_id=job_id, paths=paths), encoding="utf-8"
        )

        for key, content in artifacts.items():
            if content is None:
                continue
            target = root / relative[key]
            target.parent.mkdir(parents=True, exist_ok=True)
            text = json.dumps(content) if isinstance(content, (dict, list)) else content
            target.write_text(text, encoding="utf-8")
        return root

    return _build


@pytest.fixture
def bundle(make_bundle) -> Path:
    """A bundle with a manifest and nothing else."""
    return make_bundle()


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Repository root, for the tests that read shipped files as text."""
    return Path(__file__).resolve().parents[2]
