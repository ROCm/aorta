"""Tests for the ``layer_numerics`` collector submodule (no GPU / no torch).

Covers the platform-side contract only: the collector is registered as a
known recipe, the script ships and is discoverable, and ``build_env``
produces the expected ``NANLOG_*`` bundle. The logger's runtime hook
behavior lives in the verbatim upstream script and is exercised on a GPU
host, not here.
"""

from __future__ import annotations

from pathlib import Path

from aorta.instrumentation.layer_numerics import (
    OUTPUT_SUBDIR,
    SCRIPT_PATH,
    build_env,
)
from aorta.run.collectors import KNOWN_RECIPES


def test_registered_as_known_recipe() -> None:
    assert "layer_numerics" in KNOWN_RECIPES


def test_script_path_exists_and_is_the_logger() -> None:
    assert SCRIPT_PATH.is_file()
    assert SCRIPT_PATH.name == "instrument_nan_logger.py"
    # Sanity: it's the real logger, not an empty placeholder.
    text = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "NANLOG_DIR" in text
    assert 'if __name__ == "__main__"' in text  # standalone entry preserved


def test_build_env_points_nanlog_dir_into_results_tree() -> None:
    env = build_env(Path("/tmp/run/cell0"))
    assert env["NANLOG_DIR"] == str(Path("/tmp/run/cell0") / OUTPUT_SUBDIR)


def test_build_env_accepts_str_results_dir() -> None:
    env = build_env("relative/results")
    assert env["NANLOG_DIR"] == str(Path("relative/results") / OUTPUT_SUBDIR)


def test_build_env_defaults_present() -> None:
    env = build_env(Path("/tmp/x"))
    assert env["NANLOG_PRE_CONTEXT"] == "10"
    assert env["NANLOG_SAMPLE_EVERY"] == "50"
    assert env["NANLOG_CHANNELS"] == "act,input,igrad,weight,bias,wgrad,bgrad"
    # Only NANLOG_* keys are emitted (never leaks unrelated environment).
    assert all(k.startswith("NANLOG_") for k in env)


def test_build_env_overrides_win() -> None:
    env = build_env(
        Path("/tmp/x"),
        overrides={"NANLOG_WATCH_NAMES": "encoder.blocks", "NANLOG_SAMPLE_EVERY": "1"},
    )
    assert env["NANLOG_WATCH_NAMES"] == "encoder.blocks"
    assert env["NANLOG_SAMPLE_EVERY"] == "1"  # override beats the default


def test_build_env_override_can_redirect_output() -> None:
    env = build_env(Path("/tmp/x"), overrides={"NANLOG_DIR": "/custom/out"})
    assert env["NANLOG_DIR"] == "/custom/out"


def test_build_env_does_not_mutate_defaults() -> None:
    first = build_env(Path("/a"))
    build_env(Path("/b"), overrides={"NANLOG_SAMPLE_EVERY": "999"})
    # A later call with overrides must not bleed into an earlier bundle
    # (guards against a shared-dict aliasing bug).
    assert first["NANLOG_SAMPLE_EVERY"] == "50"
    assert build_env(Path("/a"))["NANLOG_DIR"] == str(Path("/a") / OUTPUT_SUBDIR)
