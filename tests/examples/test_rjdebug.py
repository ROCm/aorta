"""rjdebug against a scripted OpenAI-compatible server, on a toy corpus-v2 archive.

No model and no GPU: a standard-library HTTP server answers every chat request
from a script, so what is checked is rjdebug's half of the contract -- the
request it sends, how it reads the reply, that the environment scores the
episode -- and that a column it writes replays to the same episodes.
"""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples" / "rl"))
sys.path.insert(0, str(ROOT / "tests" / "examples"))

import rjdebug  # noqa: E402
import rocjitsu_env as rj  # noqa: E402
from test_rocjitsu_env_v2 import _toy_corpus  # noqa: E402

TOOL = '{"tool": "waitcheck"}'
ANSWER = '{"answer": {"class": "waitcnt", "kernel": "k000", "offset": "0x0"}}'


class Scripted:
    """An endpoint that answers chat requests from a script and keeps every request body."""

    def __init__(self, replies: list[str], props: dict | None = None, status: int = 200) -> None:
        self.replies, self.props, self.status, self.bodies = list(replies), props, status, []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, code, doc):
                data = json.dumps(doc).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/props" and outer.props is not None:
                    self._send(200, outer.props)
                else:
                    self._send(404, {"error": "not found"})

            def do_POST(self):
                outer.bodies.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                if outer.status != 200:
                    self._send(outer.status, {"error": "context length exceeded"})
                    return
                text = outer.replies.pop(0) if outer.replies else ANSWER
                self._send(200, {"choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                                 "usage": {"prompt_tokens": 3, "completion_tokens": 2}})

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def __enter__(self) -> Scripted:
        return self

    def __exit__(self, *exc) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def archive(tmp_path, monkeypatch):
    monkeypatch.setenv(rj.PROMPT_ENV, "procedure")
    monkeypatch.setenv(rj.REWARD_ENV, "v2")
    return str(_toy_corpus(tmp_path / "corpus"))


def test_an_episode_is_played_through_the_endpoint_and_scored_by_the_environment(archive):
    scenarios = rjdebug.load_scenarios(archive, "test", ["ran@toy"], "v2")
    with Scripted([TOOL, ANSWER]) as server:
        client = rjdebug.Client(server.url, "m", timeout=10)
        system, [result] = rjdebug.run(scenarios, client, episodes=1, concurrency=1, progress=False)
    assert client.flavor == "generic"
    assert result.error is None and result.episode.outcome == "exact"
    assert result.episode.score().total == pytest.approx(0.5 + 1.0 + 4.0 - 0.05)
    first, second = server.bodies
    assert [m["role"] for m in first["messages"]] == ["system", "user"]
    assert first["messages"][0]["content"] == system == rj.system_for(scenarios)
    assert {k: first[k] for k in rjdebug.EVAL_SAMPLING} == rjdebug.EVAL_SAMPLING
    assert first["chat_template_kwargs"] == {"enable_thinking": False}
    assert [first["seed"], second["seed"]] == [rjdebug.episode_seed(rjdebug.EVAL_SEED, "ran@toy", 0, t) for t in (1, 2)]


def test_llama_server_is_detected_and_its_prefill_echo_is_not_read_as_the_reply(archive):
    scenarios = rjdebug.load_scenarios(archive, "test", ["ran@toy"], "v2")
    props = {"default_generation_settings": {"n_ctx": 4096}, "chat_template": "t", "total_slots": 1}
    with Scripted([rjdebug.THINK_OFF + ANSWER], props=props) as server:
        client = rjdebug.Client(server.url, "m", timeout=10)
        _, [result] = rjdebug.run(scenarios, client, episodes=1, concurrency=1, progress=False)
    assert client.flavor == "llamacpp"
    assert server.bodies[0]["samplers"] == rjdebug.LLAMACPP_SAMPLERS and server.bodies[0]["min_p"] == 0.0
    assert result.episode.outcome == "exact" and result.steps[0]["prefill_echo"] is True


def test_a_client_error_is_reported_once_and_never_scored_as_the_models_reply(archive):
    scenarios = rjdebug.load_scenarios(archive, "test", ["ran@toy"], "v2")
    with Scripted([], status=400) as server:
        client = rjdebug.Client(server.url, "m", timeout=10)
        _, [result] = rjdebug.run(scenarios, client, episodes=1, concurrency=1, progress=False)
    assert len(server.bodies) == 1, "a 4xx is not retried"
    assert "HTTP 400" in result.error and result.steps == []


def test_a_column_it_writes_replays_to_the_same_episodes(archive, tmp_path):
    with Scripted([TOOL, ANSWER, ANSWER, TOOL, TOOL, ANSWER]) as server:
        played = tmp_path / "played"
        assert rjdebug.main(["--endpoint", server.url, "--model", "m", "--archive", archive, "--split", "test",
                             "--episodes", "2", "--concurrency", "1", "--out", str(played), "--quiet"]) == 0
    replayed = tmp_path / "replayed"
    assert rjdebug.main(["--replay", str(played / "column.json"), "--archive", archive,
                         "--out", str(replayed), "--quiet"]) == 0
    check = json.loads((replayed / "summary.json").read_text())["replay_check"]
    assert check["episodes"] == 6 and check["identical"] == 6 and not check["mismatches"]
