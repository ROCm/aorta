"""The two-command Ruby cluster quickstart must remain safe and executable."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = (
    ROOT / "scripts" / "chat" / "start_qwen_vllm.sh",
    ROOT / "scripts" / "chat" / "start_aorta_chat.sh",
)
QUICKSTART = ROOT / "docs" / "chat" / "qwen-vllm-cluster-quickstart.md"


def _spawn_orphaned_ui_group(repo: Path, *, owned: bool) -> int:
    if owned:
        argv0 = (
            f"{repo}/.venv-ui/bin/python -m chainlit run "
            f"{repo}/src/aorta/chat/ui/app.py -h --host 127.0.0.1 --port 8080"
        )
    else:
        argv0 = "unrelated-user-service"
    code = (
        "import os\n"
        "print(os.getpgrp(), flush=True)\n"
        "if os.fork() == 0:\n"
        f"    os.execl('/bin/sleep', {argv0!r}, '30')\n"
        "os._exit(0)\n"
    )
    leader = subprocess.Popen(
        ["setsid", sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        text=True,
    )
    assert leader.stdout is not None
    pgid = int(leader.stdout.readline())
    leader.wait(timeout=5)
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        try:
            os.killpg(pgid, 0)
            return pgid
        except ProcessLookupError:
            time.sleep(0.02)
    raise AssertionError("orphaned process group never became visible")


def _fake_ui_commands(tmp_path: Path) -> Path:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "ssh").write_text(
        """#!/bin/bash
last=""
for arg in "$@"; do
  last="$arg"
done
case " $* " in
  *" ss -H -ltn "*) exit 0 ;;
  *" -tt "*) exit 0 ;;
esac
exec bash -c "$last"
""",
        encoding="utf-8",
    )
    for name in ("ssh", "ss", "curl"):
        fake = fake_bin / name
        if not fake.exists():
            fake.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        fake.chmod(0o755)
    return fake_bin


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


@pytest.mark.parametrize(
    ("script", "login_variable"),
    (
        (SCRIPTS[0], "AORTA_QWEN_LOGIN_HOST"),
        (SCRIPTS[1], "AORTA_CHAT_LOGIN_HOST"),
    ),
    ids=lambda value: value.name if isinstance(value, Path) else value,
)
def test_launcher_requires_site_specific_connection_settings(
    script: Path, login_variable: str
) -> None:
    env = dict(os.environ)
    env.pop(login_variable, None)
    env.pop("AORTA_CHAT_RUNTIME_DIR", None)

    started = subprocess.run(
        [str(script)],
        capture_output=True,
        text=True,
        timeout=10,
        env=env,
    )

    assert started.returncode != 0
    assert "--login-host is required" in started.stderr


def test_launchers_ship_no_private_host_or_directory_defaults() -> None:
    model_source = SCRIPTS[0].read_text(encoding="utf-8")
    ui_source = SCRIPTS[1].read_text(encoding="utf-8")

    assert 'LOGIN_HOST="${AORTA_QWEN_LOGIN_HOST:-}"' in model_source
    assert 'LOGIN_HOST="${AORTA_CHAT_LOGIN_HOST:-}"' in ui_source
    assert 'RUNTIME_DIR="${AORTA_CHAT_RUNTIME_DIR:-}"' in model_source
    assert 'RUNTIME_DIR="${AORTA_CHAT_RUNTIME_DIR:-}"' in ui_source
    assert 'HF_CACHE="${AORTA_QWEN_HF_CACHE:-}"' in model_source
    assert 'HF_CACHE="${RUNTIME_DIR}/hf-cache"' in model_source
    for source in (model_source, ui_source):
        assert "ruby-slurmlogin" not in source
        assert "/apps/avsharma" not in source


def test_ruby_runbook_supplies_the_required_site_configuration() -> None:
    source = QUICKSTART.read_text(encoding="utf-8")
    quickstart = source[
        source.index("## Two-command quickstart") : source.index("## 1. Prepare the AORTA checkout")
    ]

    assert "ruby-slurmlogin01.rckg.g03.cpe.ice.amd.com" in quickstart
    assert "/apps/avsharma/aorta-chat-runtime" in quickstart
    assert quickstart.count("--login-host") == 4
    assert quickstart.count("--runtime-dir") == 4
    assert "git switch cia_models" not in source


@pytest.mark.skipif(os.name != "posix", reason="requires bash launcher execution")
def test_model_worker_uses_safe_ports_and_qwen_parsers(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    docker_log = tmp_path / "docker.log"
    docker = fake_bin / "docker"
    docker.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$*" >>"$DOCKER_LOG"\n',
        encoding="utf-8",
    )
    docker.chmod(0o755)

    worker = subprocess.run(
        [
            str(SCRIPTS[0]),
            "--worker",
            "--login-host",
            "login.example.invalid",
            "--runtime-dir",
            str(runtime),
            "--hf-cache",
            str(tmp_path / "hf-cache"),
            "--port-start",
            "18001",
            "--port-end",
            "18010",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        env={
            **os.environ,
            "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
            "DOCKER_LOG": str(docker_log),
            "SLURM_JOB_ID": "12345",
        },
    )

    assert worker.returncode == 0, worker.stderr
    commands = docker_log.read_text(encoding="utf-8")
    assert "vllm serve Qwen/Qwen3.8-27B" in commands
    assert "--enable-auto-tool-choice" in commands
    assert "--tool-call-parser qwen3_xml" in commands
    assert "--reasoning-parser qwen3" in commands
    assert "--default-chat-template-kwargs" not in commands
    assert "--port 8000" not in commands


def test_model_launcher_rejects_ports_above_the_tcp_limit_before_submission() -> None:
    started = subprocess.run(
        [str(SCRIPTS[0]), "--port-start", "70000", "--port-end", "70000"],
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert started.returncode != 0
    assert "--port-end must be at most 65535" in started.stderr


@pytest.mark.skipif(os.name != "posix", reason="requires bash launcher execution")
def test_submitted_job_is_cancelled_until_pending_state_is_durable(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    scancel_log = tmp_path / "scancel.log"
    (fake_bin / "ssh").write_text(
        """#!/bin/bash
last=""
for arg in "$@"; do
  last="$arg"
done
exec bash -c "$last"
""",
        encoding="utf-8",
    )
    (fake_bin / "sbatch").write_text("#!/bin/sh\nprintf '12345\\n'\n", encoding="utf-8")
    (fake_bin / "mv").write_text("#!/bin/sh\nexit 73\n", encoding="utf-8")
    (fake_bin / "scancel").write_text(
        '#!/bin/sh\nprintf "%s\\n" "$1" >>"$SCANCEL_LOG"\n',
        encoding="utf-8",
    )
    for fake in fake_bin.iterdir():
        fake.chmod(0o755)
    started = subprocess.run(
        [str(SCRIPTS[0])],
        capture_output=True,
        text=True,
        timeout=10,
        env={
            **os.environ,
            "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
            "SCANCEL_LOG": str(scancel_log),
            "AORTA_QWEN_LOGIN_HOST": "login.example.invalid",
            "AORTA_CHAT_RUNTIME_DIR": str(runtime),
            "AORTA_QWEN_HF_CACHE": str(tmp_path / "hf-cache"),
        },
    )

    assert started.returncode != 0
    assert scancel_log.read_text(encoding="utf-8").splitlines() == ["12345"]
    assert not (runtime / "qwen38-pending-job").exists()


def test_manual_cleanup_preserves_the_endpoint_when_slurm_is_unreachable() -> None:
    source = QUICKSTART.read_text(encoding="utf-8")
    cleanup = source[
        source.index("## 3. Submit the model server") : source.index(
            "Submit through the login node"
        )
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
        source.index("Reproduce `aorta chat doctor`'s native function-call check") : source.index(
            "## 5. Start the Chat UI"
        )
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


def test_quickstart_keeps_thinking_as_a_per_request_choice() -> None:
    source = QUICKSTART.read_text(encoding="utf-8")

    assert "--reasoning-parser qwen3" in source
    assert "--default-chat-template-kwargs" not in source
    assert '"chat_template_kwargs": {"enable_thinking": false}' in source


def test_ui_uses_the_published_endpoint_and_doctor_gate() -> None:
    source = SCRIPTS[1].read_text(encoding="utf-8")

    assert '[[ -s "$ENDPOINT_FILE" ]]' in source
    assert '*"localhost:8000"*' in source
    assert "export AORTA_CHAT_LLM_PROVIDER=vllm" in source
    assert "export AORTA_CHAT_LLM_TOOL_MODE=native" in source
    assert '"${REPO_ROOT}/.venv-ui/bin/aorta" chat doctor' in source
    assert source.index('"${REPO_ROOT}/.venv-ui/bin/aorta" chat doctor') < source.index(
        'setsid "${REPO_ROOT}/.venv-ui/bin/aorta" chat ui'
    )
    assert source.count("validate_cia_model") >= 3
    assert "CIA requires the endpoint to serve Qwen/Qwen3.8-27B" in source


def test_cia_ui_refuses_an_endpoint_serving_a_different_model(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "qwen38-endpoint.env").write_text(
        "\n".join(
            (
                "export QWEN_VLLM_JOB_ID=12345",
                "export AORTA_CHAT_VLLM_BASE_URL=http://compute:8001/v1",
                "export AORTA_CHAT_VLLM_MODEL=Other/Model",
                "",
            )
        ),
        encoding="utf-8",
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    for name in ("ssh", "ss", "curl"):
        fake = fake_bin / name
        fake.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        fake.chmod(0o755)

    started = subprocess.run(
        [str(SCRIPTS[1])],
        capture_output=True,
        text=True,
        timeout=10,
        env={
            **os.environ,
            "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
            "AORTA_CHAT_LOGIN_HOST": "login.example.invalid",
            "AORTA_CHAT_RUNTIME_DIR": str(runtime),
        },
    )

    assert started.returncode != 0
    assert (
        "CIA requires the endpoint to serve Qwen/Qwen3.8-27B; " "got Other/Model (use --no-cia)"
    ) in started.stderr


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX process groups")
def test_stop_cleans_chainlit_children_after_the_leader_exits(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()
    pgid = _spawn_orphaned_ui_group(repo, owned=True)
    pid_file = runtime / "aorta-chat-ui-8080.pid"
    pid_file.write_text(f"{pgid}\n", encoding="utf-8")
    fake_bin = _fake_ui_commands(tmp_path)
    try:
        stopped = subprocess.run(
            [str(SCRIPTS[1]), "--stop", "--repo", str(repo)],
            capture_output=True,
            text=True,
            timeout=10,
            env={
                **os.environ,
                "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
                "AORTA_CHAT_LOGIN_HOST": "login.example.invalid",
                "AORTA_CHAT_RUNTIME_DIR": str(runtime),
            },
        )

        assert stopped.returncode == 0, stopped.stderr
        assert not pid_file.exists()
        with pytest.raises(ProcessLookupError):
            os.killpg(pgid, 0)
    finally:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX process groups")
def test_cleanup_reports_a_refused_remote_ui_stop(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "qwen38-endpoint.env").write_text(
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
    repo = tmp_path / "repo"
    aorta = repo / ".venv-ui" / "bin" / "aorta"
    aorta.parent.mkdir(parents=True)
    aorta.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    aorta.chmod(0o755)
    pgid = _spawn_orphaned_ui_group(repo, owned=False)
    pid_file = runtime / "aorta-chat-ui-8080.pid"
    pid_file.write_text(f"{pgid}\n", encoding="utf-8")
    fake_bin = _fake_ui_commands(tmp_path)
    try:
        started = subprocess.run(
            [str(SCRIPTS[1]), "--repo", str(repo), "--no-cia"],
            capture_output=True,
            text=True,
            timeout=10,
            env={
                **os.environ,
                "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
                "AORTA_CHAT_LOGIN_HOST": "login.example.invalid",
                "AORTA_CHAT_RUNTIME_DIR": str(runtime),
            },
        )

        assert started.returncode != 0
        assert "remote AORTA Chat UI is still running" in started.stderr
        assert pid_file.is_file()
    finally:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def test_ui_stop_ignores_an_unrecorded_matching_tunnel(tmp_path: Path) -> None:
    """The command shape is not ownership; only the PID file grants it."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    for name in ("ssh", "ss"):
        fake = fake_bin / name
        fake.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        fake.chmod(0o755)

    manual = subprocess.Popen(
        [
            "bash",
            "-c",
            (
                'exec -a "ssh -N -L 8080:127.0.0.1:8080 '
                'ruby-slurmlogin01.rckg.g03.cpe.ice.amd.com" sleep 30'
            ),
        ]
    )
    try:
        time.sleep(0.1)
        stopped = subprocess.run(
            [str(SCRIPTS[1]), "--stop"],
            capture_output=True,
            text=True,
            timeout=10,
            env={
                **os.environ,
                "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
                "AORTA_CHAT_LOGIN_HOST": "login.example.invalid",
                "AORTA_CHAT_RUNTIME_DIR": str(tmp_path / "runtime"),
            },
        )

        assert stopped.returncode == 0, stopped.stderr
        assert manual.poll() is None, "an unrecorded user tunnel was killed"
    finally:
        manual.terminate()
        manual.wait(timeout=5)


def test_ui_stop_cancels_a_startup_before_pid_publication(tmp_path: Path) -> None:
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
    repo = tmp_path / "repo"
    aorta = repo / ".venv-ui" / "bin" / "aorta"
    aorta.parent.mkdir(parents=True)
    aorta.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    aorta.chmod(0o755)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "ssh").write_text(
        """#!/bin/sh
case " $* " in
  *" -tt "*) sleep 30 ;;
  *) exit 0 ;;
esac
""",
        encoding="utf-8",
    )
    for name in ("ssh", "ss", "curl"):
        fake = fake_bin / name
        if not fake.exists():
            fake.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        fake.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "AORTA_CHAT_LOGIN_HOST": "login.example.invalid",
        "AORTA_CHAT_RUNTIME_DIR": str(runtime),
    }
    command = [
        str(SCRIPTS[1]),
        "--repo",
        str(repo),
        "--startup-timeout",
        "10",
        "--no-cia",
    ]

    starting = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        lock = runtime / "aorta-chat-ui-8080.lock"
        deadline = time.monotonic() + 3
        while not lock.is_dir() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert lock.is_dir(), "startup never acquired its lock"

        stopped = subprocess.run(
            [str(SCRIPTS[1]), "--stop", "--repo", str(repo)],
            capture_output=True,
            text=True,
            timeout=10,
            env=env,
        )
        _stdout, startup_stderr = starting.communicate(timeout=5)

        assert stopped.returncode == 0, stopped.stderr
        assert starting.returncode != 0
        assert "cancelled by --stop" in startup_stderr
        assert not (runtime / "aorta-chat-stop-8080.requested").exists()
    finally:
        if starting.poll() is None:
            starting.terminate()
            starting.wait(timeout=5)


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
        "AORTA_QWEN_LOGIN_HOST": "login.example.invalid",
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
        "AORTA_QWEN_LOGIN_HOST": "login.example.invalid",
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
