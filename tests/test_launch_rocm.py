"""Tests for scripts/launch_rocm.sh argument forwarding."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "launch_rocm.sh"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="bash is required")


@pytest.fixture
def stub_bin(tmp_path):
    """PATH dir with a `python` shim and a `torchrun` that records its argv."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "python").symlink_to(sys.executable)
    torchrun = bin_dir / "torchrun"
    torchrun.write_text('#!/usr/bin/env bash\nprintf \'%s\\n\' "$@" > "$TORCHRUN_ARGS_FILE"\n')
    torchrun.chmod(0o755)
    return bin_dir


def run_launcher(stub_bin, tmp_path, *args):
    args_file = tmp_path / "torchrun_args.txt"
    env = {
        **os.environ,
        "PATH": f"{stub_bin}{os.pathsep}{os.environ['PATH']}",
        "TORCHRUN_ARGS_FILE": str(args_file),
        "NPROC": "1",
        "HIP_VISIBLE_DEVICES": "0",
    }
    subprocess.run(
        ["bash", str(SCRIPT), *args],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return args_file.read_text().splitlines()


def test_extra_args_are_forwarded_to_train_py(stub_bin, tmp_path):
    argv = run_launcher(
        stub_bin,
        tmp_path,
        "my.yaml",
        "--override",
        "training.max_steps=5",
        "--override",
        "profiling.tensorboard=false",
    )
    assert argv[argv.index("--config") + 1] == "my.yaml"
    assert argv[-4:] == [
        "--override",
        "training.max_steps=5",
        "--override",
        "profiling.tensorboard=false",
    ]


def test_extra_args_with_spaces_stay_single_tokens(stub_bin, tmp_path):
    argv = run_launcher(stub_bin, tmp_path, "my.yaml", "--override", "a=b c")
    assert argv[-2:] == ["--override", "a=b c"]


def test_config_only_is_unchanged(stub_bin, tmp_path):
    argv = run_launcher(stub_bin, tmp_path, "my.yaml")
    assert argv[-3:] == ["--config", "my.yaml", "--enable-rocm-metrics"]


def test_no_args_uses_default_config(stub_bin, tmp_path):
    argv = run_launcher(stub_bin, tmp_path)
    assert argv[argv.index("--config") + 1] == str(REPO_ROOT / "config" / "default.yaml")
    assert argv[-1] == "--enable-rocm-metrics"
