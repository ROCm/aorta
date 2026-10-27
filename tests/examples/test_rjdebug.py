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

    def __init__(self, replies: list[str], props: dict | None = None, status: int = 200,
                 truncate: bool = False, statuses: list[int] | None = None, docs: list[dict] | None = None) -> None:
        self.replies, self.props, self.status, self.bodies = list(replies), props, status, []
        self.truncate, self.statuses, self.docs = truncate, list(statuses or []), list(docs or [])
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, code, doc):
                data = json.dumps(doc).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data) + (100 if outer.truncate else 0)))
                self.end_headers()
                self.wfile.write(data)
                if outer.truncate:  # the connection closes short of Content-Length
                    self.close_connection = True

            def do_GET(self):
                if self.path == "/props" and outer.props is not None:
                    self._send(200, outer.props)
                else:
                    self._send(404, {"error": "not found"})

            def do_POST(self):
                outer.bodies.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                status = outer.statuses.pop(0) if outer.statuses else outer.status
                if status != 200:
                    self._send(status, {"error": "context length exceeded"})
                    return
                if outer.docs:
                    self._send(200, outer.docs.pop(0))
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


def _play(server, archive, out, *flags):
    return rjdebug.main(["--endpoint", server.url, "--model", "m", "--archive", archive, "--split", "test",
                         "--scenario", "ran@toy", "--episodes", "1", "--concurrency", "1", "--out", str(out),
                         "--quiet", *flags])


def test_the_api_key_never_reaches_the_column(archive, tmp_path):
    assert rjdebug.redacted_argv(["--api-key", "S1", "--ap=S2", "--archive", "a"]) == [
        "--api-key", "<redacted>", "--ap=<redacted>", "--archive", "a"]
    with Scripted([ANSWER]) as server:
        assert _play(server, archive, tmp_path / "o", "--api-key", "SECRET-ONE", "--api=SECRET-TWO") == 0
    assert "SECRET" not in (tmp_path / "o" / "column.json").read_text()


def test_no_extra_params_sends_only_openai_fields_even_to_llama_server(archive):
    scenarios = rjdebug.load_scenarios(archive, "test", ["ran@toy"], "v2")
    props = {"default_generation_settings": {"n_ctx": 4096}, "chat_template": "t", "total_slots": 1}
    with Scripted([ANSWER], props=props) as server:
        client = rjdebug.Client(server.url, "m", timeout=10, extra_params=False)
        rjdebug.run(scenarios, client, episodes=1, concurrency=1, progress=False)
    assert client.flavor == "llamacpp"
    assert set(server.bodies[0]) <= {"model", "max_tokens", "temperature", "top_p", "seed", "stream", "n", "messages"}


def test_a_column_records_top_k_only_when_the_requests_carry_it(archive, tmp_path):
    with Scripted([ANSWER, ANSWER]) as server:
        assert _play(server, archive, tmp_path / "a") == 0
        assert _play(server, archive, tmp_path / "b", "--no-extra-params") == 0
    assert json.loads((tmp_path / "a" / "column.json").read_text())["config"]["top_k"] == 20
    assert json.loads((tmp_path / "b" / "column.json").read_text())["config"]["top_k"] is None


def test_a_budget_the_system_prompt_does_not_state_is_refused(archive, tmp_path):
    with Scripted([TOOL, TOOL]) as server:
        assert _play(server, archive, tmp_path / "short", "--max-turns", "2") == 2
        assert server.bodies == []
        assert _play(server, archive, tmp_path / "played", "--max-turns", str(rj.MAX_TURNS)) == 0
    path = tmp_path / "played" / "column.json"
    column = json.loads(path.read_text())
    assert column["config"]["max_episode_steps"] == rj.MAX_TURNS
    column["config"]["max_episode_steps"] = 2
    path.write_text(json.dumps(column))
    assert rjdebug.main(["--replay", str(path), "--archive", archive, "--out", str(tmp_path / "r"), "--quiet"]) == 2


@pytest.mark.parametrize("flags", [["--scenario", "ran@toy"], ["--limit", "1"]])
def test_a_replay_of_some_scenarios_is_held_only_to_those(archive, tmp_path, flags):
    with Scripted([TOOL, ANSWER, ANSWER, TOOL, TOOL, ANSWER]) as server:
        played = tmp_path / "played"
        assert rjdebug.main(["--endpoint", server.url, "--model", "m", "--archive", archive, "--split", "test",
                             "--episodes", "2", "--concurrency", "1", "--out", str(played), "--quiet"]) == 0
    assert len(json.loads((played / "column.json").read_text())["config"]["scenarios"]) > 1
    assert rjdebug.main(["--replay", str(played / "column.json"), "--archive", archive, "--out",
                         str(tmp_path / "r"), "--quiet", *flags]) == 0
    check = json.loads((tmp_path / "r" / "summary.json").read_text())["replay_check"]
    assert check["episodes"] == 2 and check["identical"] == 2 and not check["mismatches"]


def _llamacpp(template):
    return {"default_generation_settings": {"n_ctx": 4096}, "chat_template": template, "total_slots": 1}


@pytest.mark.parametrize("change", ["template", "extra_params"])
def test_columns_drawn_under_different_protocols_do_not_pair(archive, tmp_path, change):
    import eval_rocjitsu

    with Scripted([ANSWER, ANSWER], props=_llamacpp("t")) as server:
        assert _play(server, archive, tmp_path / "a") == 0
        if change == "template":
            server.props = _llamacpp("u")
        assert _play(server, archive, tmp_path / "b", *(["--no-extra-params"] if change == "extra_params" else [])) == 0
    with Scripted([ANSWER], props=_llamacpp("t")) as server:
        assert _play(server, archive, tmp_path / "c") == 0
    a, b, c = (json.loads((tmp_path / d / "column.json").read_text()) for d in "abc")
    assert a["config"]["generation"]["flavor"] == "llamacpp"
    eval_rocjitsu.compare(a, c)
    with pytest.raises(SystemExit, match="generation"):
        eval_rocjitsu.compare(a, b)
    assert rjdebug.main(["--replay", str(tmp_path / "a" / "column.json"), "--archive", archive,
                         "--out", str(tmp_path / "ra"), "--quiet"]) == 0
    replayed = json.loads((tmp_path / "ra" / "column.json").read_text())
    assert replayed["config"]["generation"] == a["config"]["generation"]


@pytest.mark.parametrize("status", [200, 400, 503])
def test_a_body_cut_short_is_an_episode_error_not_a_crash(archive, status):
    scenarios = rjdebug.load_scenarios(archive, "test", ["ran@toy"], "v2")
    with Scripted([ANSWER], status=status, truncate=True) as server:
        client = rjdebug.Client(server.url, "m", timeout=10, retries=0)
        _, [result] = rjdebug.run(scenarios, client, episodes=1, concurrency=1, progress=False)
    assert result.error is not None and result.steps == []


def test_a_run_with_a_failed_request_writes_an_incomplete_column_that_does_not_pair(archive, tmp_path):
    import eval_rocjitsu

    with Scripted([], status=400) as server:
        assert _play(server, archive, tmp_path / "o") == 1
    column = json.loads((tmp_path / "o" / "column.json").read_text())
    assert column["incomplete"] is True and column["errors"] == 1 and "metrics" not in column
    assert json.loads((tmp_path / "o" / "summary.json").read_text())["incomplete"] is True
    with pytest.raises(SystemExit, match="incomplete"):
        eval_rocjitsu.compare(column, column)


def test_the_default_seed_is_the_evaluations(archive):
    import eval_rocjitsu

    args = eval_rocjitsu.build_parser().parse_args(["--corpus-root", "c", "--out", "o"])
    assert rjdebug.EVAL_SEED == args.seed


def test_a_replay_scores_under_the_reward_the_column_was_played_under(archive, tmp_path):
    with Scripted([TOOL, ANSWER]) as server:
        assert _play(server, archive, tmp_path / "played", "--reward", "v1") == 0
    assert rjdebug.main(["--replay", str(tmp_path / "played" / "column.json"), "--archive", archive,
                         "--out", str(tmp_path / "replayed"), "--quiet"]) == 0
    check = json.loads((tmp_path / "replayed" / "summary.json").read_text())["replay_check"]
    assert check["episodes"] == 1 and check["identical"] == 1


def test_a_replay_refuses_a_column_played_under_another_system_prompt(archive, tmp_path):
    with Scripted([ANSWER]) as server:
        assert _play(server, archive, tmp_path / "played") == 0
    path = tmp_path / "played" / "column.json"
    column = json.loads(path.read_text())
    column["config"]["system_sha256"] = "0" * 64
    path.write_text(json.dumps(column))
    assert rjdebug.main(["--replay", str(path), "--archive", archive, "--out", str(tmp_path / "r"), "--quiet"]) == 2


def test_an_explicit_llamacpp_flavor_still_records_the_chat_template(archive, tmp_path):
    import eval_rocjitsu

    with Scripted([ANSWER, ANSWER], props=_llamacpp("t")) as server:
        assert _play(server, archive, tmp_path / "a", "--flavor", "llamacpp") == 0
        server.props = _llamacpp("u")
        assert _play(server, archive, tmp_path / "b", "--flavor", "llamacpp") == 0
    a, b = (json.loads((tmp_path / d / "column.json").read_text()) for d in "ab")
    with pytest.raises(SystemExit, match="generation"):
        eval_rocjitsu.compare(a, b)


def test_an_explicit_flavor_whose_server_does_not_answer_is_refused():
    with Scripted([]) as server:
        with pytest.raises(SystemExit, match="props"):
            rjdebug.Client(server.url, "m", flavor="llamacpp", timeout=10)
        with pytest.raises(SystemExit, match="api/version"):
            rjdebug.Client(server.url, "m", flavor="ollama", timeout=10)


def test_a_rate_limit_is_retried_and_other_client_errors_are_not(archive, monkeypatch):
    monkeypatch.setattr(rjdebug.time, "sleep", lambda s: None)
    scenarios = rjdebug.load_scenarios(archive, "test", ["ran@toy"], "v2")
    with Scripted([ANSWER], statuses=[429, 429]) as server:
        client = rjdebug.Client(server.url, "m", timeout=10)
        _, [result] = rjdebug.run(scenarios, client, episodes=1, concurrency=1, progress=False)
    assert len(server.bodies) == 3 and result.error is None and result.episode.outcome == "exact"


@pytest.mark.parametrize("change", ["missing", "other"])
def test_a_replay_refuses_a_column_played_against_another_archive(archive, tmp_path, change):
    with Scripted([ANSWER]) as server:
        assert _play(server, archive, tmp_path / "played") == 0
    path = tmp_path / "played" / "column.json"
    column = json.loads(path.read_text())
    if change == "missing":
        del column["config"]["corpus"]
    else:
        column["config"]["corpus"]["ran@toy"] = "0" * 64
    path.write_text(json.dumps(column))
    assert rjdebug.main(["--replay", str(path), "--archive", archive, "--out", str(tmp_path / "r"), "--quiet"]) == 2
    assert not (tmp_path / "r").exists()


@pytest.mark.parametrize("doc", [
    {"choices": [None]},
    {"choices": ["x"]},
    {"choices": [{"message": "x"}]},
    {"choices": [{"message": {"content": ["x"]}}]},
    {"choices": [{"message": {"content": "x", "reasoning_content": {"a": 1}}}]},
    {"choices": [{}]},
    {"choices": [{"message": None}]},
    {"choices": [{"message": {"content": None}}]},
    {"choices": [{"message": {"content": "x"}}], "usage": ["unexpected"]},
    {"choices": [{"message": {"content": "x"}}], "timings": ["unexpected"]},
    {"choices": [{"message": {"content": "x"}}], "usage": {"prompt_tokens": "3"}},
    {"choices": [{"message": {"content": "x"}}], "usage": {"completion_tokens": True}},
], ids=["null-choice", "string-choice", "string-message", "list-content", "dict-reasoning", "empty-choice",
        "null-message", "null-content", "list-usage", "list-timings", "string-token-count", "bool-token-count"])
def test_a_malformed_reply_is_an_episode_error_not_a_crash(archive, doc):
    scenarios = rjdebug.load_scenarios(archive, "test", ["ran@toy"], "v2")
    with Scripted([], docs=[doc]) as server:
        client = rjdebug.Client(server.url, "m", timeout=10, retries=0)
        _, [result] = rjdebug.run(scenarios, client, episodes=1, concurrency=1, progress=False)
    assert result.error is not None and result.steps == []


def test_an_explicitly_empty_reply_is_still_the_models_reply(archive):
    scenarios = rjdebug.load_scenarios(archive, "test", ["ran@toy"], "v2")
    with Scripted([""]) as server:
        client = rjdebug.Client(server.url, "m", timeout=10, retries=0)
        _, [result] = rjdebug.run(scenarios, client, episodes=1, concurrency=1, progress=False)
    assert result.error is None and result.steps


def test_a_replay_fails_when_the_column_lost_an_episodes_summary(archive, tmp_path):
    with Scripted([ANSWER, ANSWER]) as server:
        assert _play(server, archive, tmp_path / "played", "--episodes", "2") == 0
    path = tmp_path / "played" / "column.json"
    column = json.loads(path.read_text())
    [group] = column["groups"]
    assert len(group["episodes"]) == 2
    group["episodes"].pop()
    path.write_text(json.dumps(column))
    assert rjdebug.main(["--replay", str(path), "--archive", archive, "--out", str(tmp_path / "r"), "--quiet"]) != 0
    check = json.loads((tmp_path / "r" / "summary.json").read_text())["replay_check"]
    assert check["identical"] == 1 and check["mismatches"] == ["ran@toy#1"]
