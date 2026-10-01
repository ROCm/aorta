"""The two-command Ruby cluster quickstart must remain safe and executable."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = (
    ROOT / "scripts" / "chat" / "start_qwen_vllm.sh",
    ROOT / "scripts" / "chat" / "start_aorta_chat.sh",
)


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda path: path.name)
def test_launcher_is_executable_and_valid_bash(script: Path) -> None:
    assert os.access(script, os.X_OK)

    checked = subprocess.run(
        ["bash", "-n", str(script)],
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert checked.returncode == 0, checked.stderr


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda path: path.name)
def test_launcher_help_needs_no_cluster_access(script: Path) -> None:
    shown = subprocess.run(
        [str(script), "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert shown.returncode == 0, shown.stderr
    assert "Usage:" in shown.stdout


def test_model_launcher_never_uses_the_occupied_dashboard_port() -> None:
    source = SCRIPTS[0].read_text(encoding="utf-8")

    assert "PORT_START=\"${AORTA_QWEN_PORT_START:-8001}\"" in source
    assert "((PORT_START >= 8001))" in source
    assert "localhost:8000" in source  # only the explicit refusal guard/help
    assert "*\"localhost:8000\"*" in source
    assert "--enable-auto-tool-choice" in source
    assert "--tool-call-parser qwen3_xml" in source
    assert '"aorta_tool_protocol_probe"' in source


def test_ui_requires_the_published_endpoint_and_cleans_up() -> None:
    source = SCRIPTS[1].read_text(encoding="utf-8")

    assert '[[ -s "$ENDPOINT_FILE" ]]' in source
    assert "*\"localhost:8000\"*" in source
    assert "ExitOnForwardFailure=yes" in source
    assert "trap cleanup EXIT" in source
    assert "trap cleanup_remote_worker EXIT HUP INT TERM" in source
    assert 'MODE="stop"' in source
    assert "aorta-chat-stop.requested" in source
    assert 'AORTA_CHAT_UI_STARTUP_TIMEOUT:-180' in source
    assert "export AORTA_CHAT_LLM_TOOL_MODE=native" in source
