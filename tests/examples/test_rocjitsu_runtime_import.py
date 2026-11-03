"""The rocjitsu inference runtime is installed, while example imports remain compatible."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_runtime_imports_with_only_src_on_pythonpath(tmp_path: Path) -> None:
    imported = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from aorta.rocjitsu import Client, ArchiveRunConfig, attach_archive_run; "
                "import aorta.rocjitsu.environment"
            ),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join((str(ROOT / "src"), os.environ.get("PYTHONPATH", ""))),
        },
    )

    assert imported.returncode == 0, imported.stderr


def test_legacy_environment_name_aliases_the_package_module(monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(ROOT / "examples" / "rl"))
    sys.modules.pop("rocjitsu_env", None)

    import rocjitsu_env

    from aorta.rocjitsu import environment

    assert rocjitsu_env is environment


def test_runtime_redaction_covers_mapping_keys_and_endpoint_text() -> None:
    from aorta.rocjitsu import runtime

    secret = "SECRET-KEY"
    assert runtime._redact({secret: secret}, secret) == {
        "<redacted>": "<redacted>"
    }
    endpoint = "http://user:SECRET-KEY@example.test/v1?token=SECRET-KEY#SECRET-KEY"
    redacted = runtime._redact(runtime._redacted_endpoint(endpoint), secret)
    assert secret not in redacted
    assert "user" not in redacted
