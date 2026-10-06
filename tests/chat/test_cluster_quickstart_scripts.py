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
QUICKSTART = ROOT / "docs" / "chat" / "qwen-vllm-cluster-quickstart.md"


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
    assert "((PORT_END <= 65535))" in source
    assert "localhost:8000" in source  # only the explicit refusal guard/help
    assert "*\"localhost:8000\"*" in source
    assert "--enable-auto-tool-choice" in source
    assert "--tool-call-parser qwen3_xml" in source
    assert '"aorta_tool_protocol_probe"' in source
    assert "PROGRESS_INTERVAL=30" in source
    assert "--verbose" in source
    assert "Still waiting for Qwen" in source
    assert "Still waiting for Slurm job" in source
    assert "Qwen is ready after %s" in source
    assert "Qwen vLLM is stopped after %s" in source
    assert 'PENDING_FILE="${RUNTIME_DIR}/qwen38-pending-job"' in source
    assert 'LOCK_DIR="${RUNTIME_DIR}/qwen38-launch.lock"' in source
    assert 'while ! mkdir "$LOCK_DIR"' in source
    assert "acquire_launch_lock" in source
    assert "release_launch_lock" in source
    assert 'job_status "$active_job" || true' not in source
    assert source.count('--max-time "$request_timeout"') == 3
    assert 'message.get("reasoning_content")' in source


def test_model_launcher_rejects_ports_above_the_tcp_limit_before_submission() -> None:
    started = subprocess.run(
        [str(SCRIPTS[0]), "--port-start", "70000", "--port-end", "70000"],
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert started.returncode != 0
    assert "--port-end must be at most 65535" in started.stderr


def test_reused_endpoint_probes_the_model_recorded_by_its_worker() -> None:
    source = SCRIPTS[0].read_text(encoding="utf-8")
    adoption = 'MODEL="$recorded_model"'
    first_probe = 'AORTA_EXPECTED_MODEL="$MODEL"'

    assert 'recorded_model="${AORTA_CHAT_VLLM_MODEL:-}"' in source
    assert source.index(adoption) < source.index(first_probe)


def test_manual_cleanup_preserves_the_endpoint_when_slurm_is_unreachable() -> None:
    source = QUICKSTART.read_text(encoding="utf-8")
    cleanup = source[
        source.index("## 3. Submit the model server") :
        source.index("Submit through the login node")
    ]

    assert 'if ! status="$(' in cleanup
    assert "could not query Slurm job" in cleanup
    assert "preserving ${ENDPOINT_FILE}" in cleanup
    assert cleanup.index("exit 1") < cleanup.index('rm -f "${ENDPOINT_FILE}"')


def test_manual_shutdown_confirms_an_empty_queue_before_reporting_success() -> None:
    source = QUICKSTART.read_text(encoding="utf-8")
    shutdown = source[source.index("## 7. Stop everything") :]

    assert "while true; do" in shutdown
    assert 'if ! JOB_STATUS="$(' in shutdown
    assert "shutdown is unconfirmed" in shutdown
    assert '[ -n "${JOB_STATUS}" ] || break' in shutdown
    assert shutdown.index("shutdown is unconfirmed") < shutdown.index(
        'echo "vLLM job ${QWEN_VLLM_JOB_ID} has stopped."'
    )


def test_quickstart_documents_the_native_function_call_gate() -> None:
    source = QUICKSTART.read_text(encoding="utf-8")
    probe = source[
        source.index("Reproduce the launcher's required native function-call check") :
        source.index("## 5. Start the Chat UI")
    ]

    for required in (
        '"tools": [{',
        '"parameters": {',
        '"required": ["source"]',
        '"tool_choice": {"type": "function"',
        'name = "aorta_tool_protocol_probe"',
        'message.get("tool_calls")',
        'arguments.get("source") != "READY"',
    ):
        assert required in probe


def test_ui_requires_the_published_endpoint_and_cleans_up() -> None:
    source = SCRIPTS[1].read_text(encoding="utf-8")

    assert '[[ -s "$ENDPOINT_FILE" ]]' in source
    assert "*\"localhost:8000\"*" in source
    assert "ExitOnForwardFailure=yes" in source
    assert "trap cleanup EXIT" in source
    assert "trap cleanup_remote_worker EXIT HUP INT TERM" in source
    assert 'MODE="stop"' in source
    assert 'aorta-chat-ui-${UI_PORT}.pid' in source
    assert 'aorta-chat-stop-${UI_PORT}.requested' in source
    assert 'UI_LOCK_DIR="${RUNTIME_DIR}/aorta-chat-ui-${UI_PORT}.lock"' in source
    assert "acquire_ui_lock" in source
    assert "release_ui_lock" in source
    assert "Starting AORTA Chat on port" in source
    assert "--port ${UI_PORT}" in source
    assert 'AORTA_CHAT_UI_STARTUP_TIMEOUT:-360' in source
    assert "export AORTA_CHAT_LLM_PROVIDER=vllm" in source
    assert "export AORTA_CHAT_LLM_TOOL_MODE=native" in source


def test_ui_cleanup_waits_for_the_entire_process_group() -> None:
    source = SCRIPTS[1].read_text(encoding="utf-8")

    assert source.count('/bin/kill -0 -- "-$1"') >= 2
    assert 'group_alive "$pid"' in source
    assert 'process_group_alive "$ui_pid"' in source
    assert 'kill -0 "$pid"' not in source
    assert 'kill -0 "$ui_pid"' not in source
    assert "did not stop; preserving" in source


def test_slurm_query_failure_preserves_the_endpoint_record(tmp_path: Path) -> None:
    """A login-host outage is not evidence that the recorded job died."""
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    endpoint = runtime / "qwen38-endpoint.env"
    endpoint.write_text(
        "\n".join(
            (
                "export QWEN_VLLM_JOB_ID=12345",
                "export AORTA_CHAT_VLLM_BASE_URL=http://compute:8001/v1",
                "export AORTA_CHAT_VLLM_MODEL=Qwen/Qwen3.8-27B",
                "",
            )
        ),
        encoding="utf-8",
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_ssh = fake_bin / "ssh"
    fake_ssh.write_text("#!/bin/sh\nexit 42\n", encoding="utf-8")
    fake_ssh.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "AORTA_CHAT_RUNTIME_DIR": str(runtime),
        "AORTA_QWEN_HF_CACHE": str(tmp_path / "hf-cache"),
    }

    started = subprocess.run(
        [str(SCRIPTS[0])],
        capture_output=True,
        text=True,
        timeout=10,
        env=env,
    )

    assert started.returncode != 0
    assert "could not query Slurm job 12345" in started.stderr
    assert endpoint.is_file(), "transient SSH failure discarded the live-job record"


def test_a_purged_job_is_a_successful_empty_queue_result(tmp_path: Path) -> None:
    """A completed job is absent from the user's queue, not a query outage."""
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    endpoint = runtime / "qwen38-endpoint.env"
    endpoint.write_text(
        "\n".join(
            (
                "export QWEN_VLLM_JOB_ID=12345",
                "export AORTA_CHAT_VLLM_BASE_URL=http://compute:8001/v1",
                "export AORTA_CHAT_VLLM_MODEL=Qwen/Qwen3.8-27B",
                "",
            )
        ),
        encoding="utf-8",
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_ssh = fake_bin / "ssh"
    fake_ssh.write_text(
        """#!/bin/sh
case "$*" in
  *"squeue -h -j"*)
    echo "Invalid job id specified" >&2
    exit 1
    ;;
  *"squeue -h -u"*)
    exit 0
    ;;
  *)
    echo "unexpected ssh command: $*" >&2
    exit 99
    ;;
esac
""",
        encoding="utf-8",
    )
    fake_ssh.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "AORTA_CHAT_RUNTIME_DIR": str(runtime),
        "AORTA_QWEN_HF_CACHE": str(tmp_path / "hf-cache"),
    }

    stopped = subprocess.run(
        [str(SCRIPTS[0]), "--stop"],
        capture_output=True,
        text=True,
        timeout=10,
        env=env,
    )

    assert stopped.returncode == 0, stopped.stderr
    assert "Qwen vLLM is stopped" in stopped.stdout
    assert not endpoint.exists()
