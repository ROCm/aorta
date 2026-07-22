"""Tests for the shared ``docker_env_flags`` helper (env-forwarding only)."""

import os

import pytest

from aorta.run import docker_env_flags


def test_empty_mapping_yields_no_flags():
    assert docker_env_flags({}) == []


def test_single_pair():
    assert docker_env_flags({"FOO": "bar"}) == ["-e", "FOO=bar"]


def test_output_is_deterministic_sorted_by_key():
    # Insertion order deliberately not sorted; output must be key-sorted so the
    # same mapping always renders byte-identical argv.
    env = {"ZED": "3", "ALPHA": "1", "MIKE": "2"}
    assert docker_env_flags(env) == [
        "-e",
        "ALPHA=1",
        "-e",
        "MIKE=2",
        "-e",
        "ZED=3",
    ]
    # Same content, different insertion order -> identical output.
    assert docker_env_flags({"MIKE": "2", "ZED": "3", "ALPHA": "1"}) == docker_env_flags(env)


def test_value_with_equals_and_spaces_preserved():
    # Only the FIRST '=' delimits key/value for docker; the value may itself
    # contain '=' and spaces. The helper emits a single KEY=VALUE token.
    flags = docker_env_flags({"CONN": "a=b c=d"})
    assert flags == ["-e", "CONN=a=b c=d"]


def test_never_reads_os_environ(monkeypatch):
    # A var present in the ambient environment but NOT in the explicit mapping
    # must never appear in the output -- the helper cannot leak host env.
    monkeypatch.setenv("AORTA_TEST_LEAK_CANARY", "leaked")
    assert "AORTA_TEST_LEAK_CANARY" in os.environ
    flags = docker_env_flags({"SAFE": "1"})
    assert flags == ["-e", "SAFE=1"]
    assert not any("LEAK_CANARY" in tok for tok in flags)


def test_non_mapping_raises_type_error():
    with pytest.raises(TypeError, match="dict"):
        docker_env_flags(["FOO=bar"])  # type: ignore[arg-type]


def test_non_string_value_raises_without_echoing_value():
    with pytest.raises(TypeError) as exc:
        docker_env_flags({"TOKEN": 12345})  # type: ignore[dict-item]
    # The (possibly secret) value must not appear in the error text.
    assert "12345" not in str(exc.value)
    assert "TOKEN" in str(exc.value)


def test_non_string_key_is_rejected_before_sorting():
    with pytest.raises(TypeError, match="keys must be str"):
        docker_env_flags({"GOOD": "1", 2: "bad"})  # type: ignore[dict-item]


def test_invalid_key_name_raises_value_error():
    with pytest.raises(ValueError, match="POSIX"):
        docker_env_flags({"1BAD": "x"})
    with pytest.raises(ValueError, match="POSIX"):
        docker_env_flags({"has space": "x"})


def test_nul_value_rejected_without_echoing_value():
    # A NUL byte cannot survive an execve'd ``docker run -e`` argument; reject
    # it for parity with the dispatcher's os.environ validation. The value must
    # not be echoed in the error (it may be a secret).
    with pytest.raises(ValueError, match="NUL") as exc:
        docker_env_flags({"TOKEN": "abc\x00def"})
    assert "abc" not in str(exc.value) and "def" not in str(exc.value)
    assert "TOKEN" in str(exc.value)
