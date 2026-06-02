"""Unit tests for ``aorta.probe.redaction`` (issue #188 Phase 3)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from aorta.bundle.errors import RedactionError
from aorta.probe.redaction import (
    RedactingRedactor,
    RedactionCfg,
    _line_windows,
    scrub_env_keys,
    scrub_text,
)
from aorta.probe.sandbox import MAX_LOG_BYTES

FIXTURES = Path(__file__).parent / "fixtures"


def test_env_key_glob():
    env = {
        "AWS_ACCESS_KEY_ID": "AKIA",
        "AWS_SECRET_ACCESS_KEY": "secret",
        "PATH": "/usr/bin",
        "SAFE": "ok",
    }
    scrubbed, removed = scrub_env_keys(env, ("AWS_*",))
    assert removed == 2
    assert scrubbed == {"PATH": "/usr/bin", "SAFE": "ok"}


def test_env_key_glob_case_sensitive():
    env = {"aws_secret_key": "lower", "AWS_SECRET_KEY": "upper"}
    scrubbed, removed = scrub_env_keys(env, ("AWS_*",))
    assert removed == 1
    assert "aws_secret_key" in scrubbed
    assert "AWS_SECRET_KEY" not in scrubbed


def test_path_rewrite():
    text = (
        "a /home/user/a/data/file.txt\n"
        "b /home/user/a/data/file.txt\n"
        "c /home/user/b/other.log\n"
        "d /opt/third/path\n"
        "e /opt/third/path\n"
    )
    out, paths, v4, v6 = scrub_text(text, scrub_paths=True, scrub_ip_addresses=False)
    assert paths == 5
    assert "<PATH:0>" in out
    assert "<PATH:1>" in out
    assert "<PATH:2>" in out
    assert "/home/user" not in out
    assert v4 == 0 and v6 == 0


def test_path_rewrite_no_reverse_mapping_persisted(tmp_path: Path):
    cfg = RedactionCfg(scrub_paths=True)
    redactor = RedactingRedactor(cfg)
    src = tmp_path / "stdout.log"
    dst = tmp_path / "out" / "stdout.log"
    src.write_text("loaded /secret/path/file\n", encoding="utf-8")
    counts = redactor.scrub_file(src, dst)
    assert counts.paths_rewritten == 1
    bundled = dst.read_text(encoding="utf-8")
    assert "/secret/path/file" not in bundled
    assert "<PATH:0>" in bundled
    assert "mapping" not in bundled.lower()


def test_ip_rewrite():
    text = "host 192.168.0.1 and 2001:db8::1 here"
    out, paths, v4, v6 = scrub_text(
        text,
        scrub_paths=False,
        scrub_ip_addresses=True,
    )
    assert paths == 0
    assert v4 == 1
    assert v6 == 1
    assert "<IPV4:0>" in out
    assert "<IPV6:0>" in out
    assert "192.168.0.1" not in out


@pytest.mark.parametrize(
    "snippet",
    [
        "loopback ::1 ok",
        "fe80 link fe80::1 ok",
        "bracket [::1] ok",
        "url http://[::1]:8080/path",
    ],
)
def test_ipv6_compressed_and_bracketed_forms(snippet: str):
    out, _, v4, v6 = scrub_text(
        snippet,
        scrub_paths=False,
        scrub_ip_addresses=True,
    )
    assert v4 == 0
    assert v6 >= 1
    assert "::1" not in out
    assert "[::1]" not in out


def test_redactor_kind_string():
    assert RedactingRedactor(RedactionCfg()).kind == "probe.v1"


def test_redaction_dos_bound():
    """10 MiB slash run completes within 5 seconds (regex DoS guard)."""
    blob = "/" * MAX_LOG_BYTES
    t0 = time.perf_counter()
    scrub_text(blob, scrub_paths=True, scrub_ip_addresses=False)
    assert time.perf_counter() - t0 < 5.0


def test_redacting_redactor_scrubs_result_json_env(tmp_path: Path):
    cfg = RedactionCfg(
        scrub_env_keys=("AWS_*", "HOME", "USER"),
        scrub_paths=True,
    )
    redactor = RedactingRedactor(cfg)
    src = tmp_path / "result.json"
    dst = tmp_path / "out" / "result.json"
    src.write_text(
        json.dumps(
            {
                "verdict": "pass",
                "env": {"AWS_TOKEN": "x", "HIP_VISIBLE_DEVICES": "0"},
                "argv": ["/home/user/train.py"],
            }
        ),
        encoding="utf-8",
    )
    counts = redactor.scrub_file(src, dst)
    assert counts.env_keys_removed == 1
    doc = json.loads(dst.read_text(encoding="utf-8"))
    assert "AWS_TOKEN" not in doc["env"]
    assert "<PATH:" in doc["argv"][0]


def test_result_json_env_values_scrubbed(tmp_path: Path):
    """Retained env values are path/IP-scrubbed, not just key-filtered.

    Removing matching keys left a kept key's value (e.g. a *_PATH var)
    leaking an absolute path/IP into the bundle even with scrub_paths on
    (Copilot review). result.json env now matches host_env.json.
    """
    cfg = RedactionCfg(
        scrub_env_keys=("AWS_*",),
        scrub_paths=True,
        scrub_ip_addresses=True,
    )
    redactor = RedactingRedactor(cfg)
    src = tmp_path / "result.json"
    dst = tmp_path / "out" / "result.json"
    src.write_text(
        json.dumps(
            {
                "verdict": "pass",
                "env": {
                    "AWS_TOKEN": "drop-me",
                    "LD_LIBRARY_PATH": "/home/customer/lib",
                    "MASTER_ADDR": "192.168.1.42",
                },
            }
        ),
        encoding="utf-8",
    )
    counts = redactor.scrub_file(src, dst)
    doc = json.loads(dst.read_text(encoding="utf-8"))
    assert "AWS_TOKEN" not in doc["env"]
    assert counts.env_keys_removed == 1
    assert "/home/customer/lib" not in doc["env"]["LD_LIBRARY_PATH"]
    assert "192.168.1.42" not in doc["env"]["MASTER_ADDR"]
    assert counts.paths_rewritten >= 1
    assert counts.ips_rewritten >= 1


def test_fixture_log_scrubs_paths_and_ips(tmp_path: Path):
    raw = (FIXTURES / "redaction_input.txt").read_text(encoding="utf-8")
    cfg = RedactionCfg(
        scrub_paths=True,
        scrub_ip_addresses=True,
    )
    redactor = RedactingRedactor(cfg)
    src = tmp_path / "stdout.log"
    dst = tmp_path / "out" / "stdout.log"
    src.write_text(raw, encoding="utf-8")
    counts = redactor.scrub_file(src, dst)
    out = dst.read_text(encoding="utf-8")
    assert counts.paths_rewritten >= 3
    assert counts.ips_rewritten >= 2
    assert "/home/customer" not in out
    assert "192.168.1.42" not in out


def test_invalid_utf8_log_still_scrubbed(tmp_path: Path):
    """A stray non-UTF-8 byte must not disable scrubbing (oyazdanb review).

    stdout.log / stderr.log are raw subprocess bytes by design; the
    redactor used to fail open (byte-copy the file) on the first
    UnicodeDecodeError, silently leaking every path/IP after it. It now
    decodes with errors="replace" and keeps scrubbing.
    """
    cfg = RedactionCfg(scrub_paths=True, scrub_ip_addresses=True)
    redactor = RedactingRedactor(cfg)
    src = tmp_path / "stdout.log"
    dst = tmp_path / "out" / "stdout.log"
    src.write_bytes(b"bad\xff  /home/user/secret 192.168.1.1\n")
    counts = redactor.scrub_file(src, dst)
    out = dst.read_text(encoding="utf-8")
    assert counts.paths_rewritten >= 1
    assert counts.ips_rewritten >= 1
    assert "/home/user/secret" not in out
    assert "192.168.1.1" not in out


def test_line_windows_reconstructs_input():
    text = "alpha\nbeta\r\ngamma\rdelta\nno-trailing-newline"
    assert "".join(_line_windows(text)) == text


def test_scrub_text_spans_window_seam(monkeypatch):
    """A path/IP must not be missed when it lands on a window boundary.

    The old fixed-slice windowing cut tokens in half at ``i*MAX_LOG_BYTES``
    so neither regex pass matched them. Line-aware windows never split a
    line, so an IP sitting just past the byte budget is still scrubbed.
    """
    monkeypatch.setattr("aorta.probe.redaction.MAX_LOG_BYTES", 20)
    text = "x" * 15 + "\n" + "192.168.1.1 /home/user/secret\n"
    out, paths, v4, v6 = scrub_text(text, scrub_paths=True, scrub_ip_addresses=True)
    assert v4 == 1
    assert paths == 1
    assert "192.168.1.1" not in out
    assert "/home/user/secret" not in out


def test_corrupt_result_json_fails_closed(tmp_path: Path):
    """Corrupt result.json fails closed with a typed BundleError (Issue E).

    A raw JSONDecodeError would escape staging as a traceback and is not
    an OSError, so the writer's OSError wrap would miss it -- and partial
    handling risks an unredacted bundle. RedactionError is a BundleError,
    so the CLI fails closed (no bundle written).
    """
    cfg = RedactionCfg(scrub_paths=True)
    redactor = RedactingRedactor(cfg)
    src = tmp_path / "result.json"
    dst = tmp_path / "out" / "result.json"
    src.write_text("{ this is not valid json", encoding="utf-8")
    with pytest.raises(RedactionError) as excinfo:
        redactor.scrub_file(src, dst)
    assert excinfo.value.path == src
    assert not dst.exists()


def test_corrupt_host_env_json_fails_closed(tmp_path: Path):
    cfg = RedactionCfg(scrub_env_keys=("AWS_*",))
    redactor = RedactingRedactor(cfg)
    src = tmp_path / "host_env.json"
    dst = tmp_path / "out" / "host_env.json"
    src.write_text("not json at all", encoding="utf-8")
    with pytest.raises(RedactionError):
        redactor.scrub_file(src, dst)


def test_probe_env_scrubs_env_keys(tmp_path: Path):
    cfg = RedactionCfg(scrub_env_keys=("AWS_*", "HOME", "USER"))
    redactor = RedactingRedactor(cfg)
    src = tmp_path / "probe.env"
    dst = tmp_path / "out" / "probe.env"
    src.write_text(
        "AWS_SECRET_ACCESS_KEY=supersecret\nHOME=/home/user\nSAFE=1\n",
        encoding="utf-8",
    )
    counts = redactor.scrub_file(src, dst)
    out = dst.read_text(encoding="utf-8")
    assert counts.env_keys_removed == 2
    assert "supersecret" not in out
    assert "SAFE=1" in out
