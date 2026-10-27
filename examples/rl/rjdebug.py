#!/usr/bin/env python3
"""rjdebug: play rocjitsu GPU-debugging episodes against any OpenAI-compatible endpoint.

    rjdebug.py --endpoint http://127.0.0.1:8080/v1 --model rjx \\
        --archive <corpus>/corpus-v2/archive --split test_reencode \\
        [--scenario SID ...] [--episodes 2] [--transcript] [--concurrency 8] [--out DIR]

The policy is whatever model the endpoint serves (llama.cpp ``llama-server``,
vLLM, TokenSpeed/SGLang, Ollama, LM Studio). Observations and scoring are the
training environment's own (``examples/rl/rocjitsu_env.py`` of this checkout, or
of the one ``--env-code`` names), so a score here means what it means in training.

What is reproduced exactly, and how it is checked
=================================================
* **The prompt.** Each turn is the two-message chat the trainer and the
  evaluation harness built (``train_grpo_step.chat_prompt``): the system prompt
  of the corpus (``rocjitsu_env.system_for``; profile ``procedure``, reward
  ``v2``) and ONE user message that re-renders the whole episode so far
  (``Episode.observe``). The Qwen3 template is applied with
  ``enable_thinking=False``. In ``--mode chat`` the server applies the template
  (``chat_template_kwargs``); in ``--mode completions`` this file renders it
  (:func:`render_prompt`) and sends raw text. Both were checked against
  ``transformers.apply_chat_template`` token for token.
* **Sampling.** T 0.7, top-p 0.95, top-k 20, at most 192 new tokens, one
  request per reply, seeded per (seed, scenario, episode, turn) with the
  harness's ``eval_episodes.episode_seed``. For llama.cpp the sampler chain is
  pinned to HF's order (temperature before top-p) and its non-HF samplers
  (min-p 0.05 by default, repetition penalties, DRY, XTC) are switched off.
  A seed reproduces a server's own draw; it cannot reproduce HF's torch RNG, so
  a server column pairs with an HF column by scenario, not by episode.
* **The reply.** What the model generated, as HF decodes it: ``--mode
  completions`` returns exactly that; in ``--mode chat`` llama-server (with
  ``reasoning_format: none``) prefixes the content with the template's empty
  think block, which belongs to the prompt, so one leading ``<think>\\n\\n
  </think>\\n\\n`` is removed (counted as ``prefill_echo_stripped``).
* **Parsing and scoring.** The reply text goes to ``Episode.act`` unchanged.
  ``--replay COLUMN.json`` feeds a harness column's recorded replies through
  the same loop instead of an endpoint and requires every episode to come out
  with the recorded outcome and score.

Output (``--out``): ``column.json`` in the harness's column format (``groups``,
``wire``, ``config``, ``metrics``; two runs on one server pair with ``eval_rocjitsu.py --compare``;
a run with a failed request is marked ``incomplete`` and has no ``metrics``),
``summary.json`` / ``summary.txt``, and with ``--transcript`` one readable
transcript per episode under ``transcripts/``.

Tool outputs are replayed from the archive: nothing here executes a kernel.
Standard library only (plus the environment's own imports: PyYAML).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import http.client
import json
import os
import random
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: The checkout this file is in; ``--env-code`` or ``$RJDEBUG_ENV_CODE`` names another one.
DEFAULT_ENV_CODE = str(Path(__file__).resolve().parents[2])
ENV_CODE_VAR = "RJDEBUG_ENV_CODE"
#: sha256 of the system prompt the trained model was evaluated under.
TRAINED_SYSTEM_SHA256 = {("procedure", "v2"): "a0a084cbaef3bdff39fef234e6967f68269958c4a2e6c252cc680decb52d7aec"}
#: The evaluation protocol's sampling (eval_rocjitsu defaults, run with --top-k 20 --max-new-tokens 192).
EVAL_SAMPLING = {"temperature": 0.7, "top_p": 0.95, "top_k": 20, "max_tokens": 192}
EVAL_SEED = 20261005
#: What Qwen3's template appends after ``<|im_start|>assistant\n`` when ``enable_thinking`` is false.
THINK_OFF = "<think>\n\n</think>\n\n"
#: llama.cpp's sampler chain in HF ``generate`` order: top-k and temperature commute, top-p sees scaled logits.
LLAMACPP_SAMPLERS = ["top_k", "temperature", "top_p"]
LLAMACPP_NEUTRAL = {"min_p": 0.0, "typical_p": 1.0, "repeat_penalty": 1.0, "presence_penalty": 0.0,
                    "frequency_penalty": 0.0, "dry_multiplier": 0.0, "xtc_probability": 0.0, "top_n_sigma": -1.0}

_rj: Any = None


def load_env(code: str | os.PathLike[str] | None = None) -> Any:
    """Import ``rocjitsu_env`` (and the harness helpers) from a code snapshot; idempotent."""
    global _rj
    if _rj is not None:
        if code is not None and Path(_rj.__file__).resolve().parent != (Path(code) / "examples" / "rl").resolve():
            raise SystemExit(f"rocjitsu_env is already loaded from {_rj.__file__}; "
                             f"one process cannot switch to --env-code {code}")
        return _rj
    root = Path(code or os.environ.get(ENV_CODE_VAR) or DEFAULT_ENV_CODE)
    rl = root / "examples" / "rl"
    if not (rl / "rocjitsu_env.py").is_file():
        raise SystemExit(f"no rocjitsu_env.py under {rl}; pass --env-code or set {ENV_CODE_VAR}")
    for p in (str(root / "src"), str(rl)):
        while p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)
    import rocjitsu_env  # noqa: PLC0415

    if Path(rocjitsu_env.__file__).resolve().parent != rl.resolve():
        raise SystemExit(f"rocjitsu_env was imported from {rocjitsu_env.__file__}, not from {rl}")
    _rj = rocjitsu_env
    return _rj


def render_prompt(system: str, user: str) -> str:
    """Qwen3's chat template for ``[system, user]``, ``add_generation_prompt=True, enable_thinking=False``."""
    return (f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n{THINK_OFF}")


def messages(system: str, user: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def episode_seed(base: int, scenario_id: str, episode: int, step: int) -> int:
    """``eval_episodes.episode_seed``: the harness's per-reply seed (imported and checked in :func:`main`)."""
    import zlib  # noqa: PLC0415

    return (base + zlib.crc32(f"{scenario_id}/{episode}/{step}".encode())) % (2**31 - 1)


# ---------------------------------------------------------------------------
# the endpoint
# ---------------------------------------------------------------------------


class EndpointError(RuntimeError):
    """The server failed to produce a reply (transport or HTTP error); never scored as the model's reply."""


@dataclass
class Reply:
    text: str
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_s: float = 0.0
    timings: dict[str, Any] | None = None
    reasoning_split: bool = False
    prefill_echo: bool = False


class _RefuseRedirects(urllib.request.HTTPRedirectHandler):
    """Raise on a redirect instead of following it: a followed redirect would carry the API key to its target."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001, ARG002
        raise urllib.error.HTTPError(req.full_url, code, f"redirect to {newurl} refused", headers, fp)


_NO_REDIRECTS = urllib.request.build_opener(_RefuseRedirects)


@dataclass
class Client:
    """One OpenAI-compatible endpoint. ``endpoint`` is the API base (``.../v1``; added if missing)."""

    endpoint: str
    model: str
    mode: str = "chat"  # "chat" -> /chat/completions; "completions" -> /completions with a client-rendered prompt
    flavor: str = "auto"  # "auto" | "llamacpp" | "ollama" | "generic"
    api_key: str | None = None
    timeout: float = 900.0
    retries: int = 3
    sampling: dict[str, Any] = field(default_factory=lambda: dict(EVAL_SAMPLING))
    extra_params: bool = True  # send top_k / min_p / chat_template_kwargs (not in the OpenAI spec)
    server: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        base = self.endpoint.rstrip("/")
        self.endpoint = base if base.endswith("/v1") else base + "/v1"
        self.root = self.endpoint[: -len("/v1")]
        if self.flavor == "auto":
            self.flavor = ("llamacpp" if self._llamacpp_props() else
                           "ollama" if self._ollama_version() else "generic")
        elif self.flavor == "llamacpp" and not self._llamacpp_props():
            # The column records the chat template's hash from /props; without it two templates would pair.
            raise SystemExit(f"--flavor llamacpp: {self.root}/props did not answer, so the chat template is unknown")
        elif self.flavor == "ollama" and not self._ollama_version():
            raise SystemExit(f"--flavor ollama: {self.root}/api/version did not answer, so the server is unknown")
        if self.flavor == "ollama" and self.mode == "completions":
            raise SystemExit("Ollama's /v1/completions applies the model template to the prompt; use --mode chat")

    # -- plumbing -------------------------------------------------------------

    def _request(self, url: str, body: dict[str, Any] | None = None, timeout: float | None = None) -> Any:
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(url, data=data, method="GET" if body is None else "POST")
        req.add_header("Content-Type", "application/json")
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")
        with _NO_REDIRECTS.open(req, timeout=timeout or self.timeout) as resp:
            return json.loads(resp.read().decode())

    def _llamacpp_props(self) -> bool:
        try:
            props = self._request(self.root + "/props", timeout=10)
        except (urllib.error.URLError, http.client.HTTPException, OSError, ValueError):
            return False
        if not isinstance(props, dict) or "default_generation_settings" not in props:
            return False
        self.server = {
            "flavor": "llamacpp",
            "model_path": props.get("model_path"),
            "build_info": props.get("build_info"),
            "total_slots": props.get("total_slots"),
            "n_ctx": (props.get("default_generation_settings") or {}).get("n_ctx"),
            "chat_template_sha256": hashlib.sha256((props.get("chat_template") or "").encode()).hexdigest(),
        }
        return True

    def _ollama_version(self) -> bool:
        try:
            doc = self._request(self.root + "/api/version", timeout=10)
        except (urllib.error.URLError, http.client.HTTPException, OSError, ValueError):
            return False
        if not isinstance(doc, dict) or "version" not in doc:
            return False
        self.server = {"flavor": "ollama", "version": doc["version"]}
        return True

    def models(self) -> list[str]:
        try:
            doc = self._request(self.endpoint + "/models", timeout=30)
        except (urllib.error.URLError, http.client.HTTPException, OSError, ValueError) as exc:
            raise EndpointError(f"GET {self.endpoint}/models failed: {exc}") from exc
        return [m.get("id") for m in doc.get("data", [])]

    def body(self, system: str, user: str, seed: int) -> tuple[str, dict[str, Any]]:
        """The request a reply is drawn with: ``(url, json body)``."""
        s = self.sampling
        body: dict[str, Any] = {"model": self.model, "max_tokens": s["max_tokens"], "temperature": s["temperature"],
                                "top_p": s["top_p"], "seed": seed, "stream": False, "n": 1}
        if self.extra_params:
            body["top_k"] = s["top_k"]
            body["min_p"] = 0.0
        if self.flavor == "llamacpp" and self.extra_params:
            body.update(LLAMACPP_NEUTRAL)
            body["samplers"] = list(LLAMACPP_SAMPLERS)
            body["cache_prompt"] = True
        if self.mode == "completions":
            body["prompt"] = render_prompt(system, user)
            return self.endpoint + "/completions", body
        body["messages"] = messages(system, user)
        if self.extra_params:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        if self.flavor == "llamacpp" and self.extra_params:
            body["reasoning_format"] = "none"
        if self.flavor == "ollama":
            # Ollama renders the GGUF's chat template and ignores chat_template_kwargs; "none" is its thinking-off
            # switch on the OpenAI endpoint. top_k is not an OpenAI field there: it comes from the Modelfile.
            body["reasoning_effort"] = "none"
        return self.endpoint + "/chat/completions", body

    def complete(self, system: str, user: str, seed: int) -> Reply:
        url, body = self.body(system, user, seed)
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            t0 = time.time()
            try:
                doc = self._request(url, body)
            except urllib.error.HTTPError as exc:
                try:
                    detail = exc.read().decode(errors="replace")[:500]
                except (http.client.HTTPException, OSError) as body_exc:
                    detail = f"(error body unreadable: {body_exc!r})"
                last = EndpointError(f"HTTP {exc.code} {exc.reason} from {url}: {detail}")
                if exc.code < 500 and exc.code != 429:  # a refused redirect or non-429 4xx will not improve
                    raise last from exc
            except (urllib.error.URLError, http.client.HTTPException, OSError, ValueError) as exc:
                # http.client.IncompleteRead (a body cut short) is an HTTPException, not an OSError.
                last = EndpointError(f"{url}: {exc}")
            else:
                return self._reply(doc, time.time() - t0)
            time.sleep(min(30.0, 2.0 ** attempt))
        raise last or EndpointError(url)

    def _reply(self, doc: dict[str, Any], latency: float) -> Reply:
        try:
            choice = doc["choices"][0]
        except (KeyError, IndexError, TypeError) as exc:
            raise EndpointError(f"no choices in the response: {str(doc)[:300]}") from exc
        msg = choice.get("message") if isinstance(choice, dict) else None
        if not isinstance(choice, dict) or (self.mode != "completions" and not isinstance(msg, (dict, type(None)))):
            raise EndpointError(f"malformed choice in the response: {str(doc)[:300]}")
        fields = [choice.get("text")] if self.mode == "completions" else \
            [(msg or {}).get("content"), (msg or {}).get("reasoning_content")]
        if any(f is not None and not isinstance(f, str) for f in fields):
            raise EndpointError(f"reply text is not a string: {str(doc)[:300]}")
        if all(f is None for f in fields):  # an empty string is a reply; no text at all is not
            raise EndpointError(f"no reply text in the response: {str(doc)[:300]}")
        usage, timings = doc.get("usage"), doc.get("timings")
        if not isinstance(usage, (dict, type(None))) or not isinstance(timings, (dict, type(None))) or any(
                not isinstance(n, (int, type(None))) or isinstance(n, bool)
                for n in ((usage or {}).get("prompt_tokens"), (usage or {}).get("completion_tokens"))):
            raise EndpointError(f"malformed usage or timings in the response: {str(doc)[:300]}")
        split = echo = False
        if self.mode == "completions":
            text = choice.get("text")
        else:
            msg = choice.get("message") or {}
            text = msg.get("content")
            reasoning = msg.get("reasoning_content")
            if reasoning:  # the server split a thinking block out; put it back where the model wrote it
                text = f"<think>\n{reasoning}\n</think>\n\n{text or ''}"
                split = True
            elif text and text.startswith(THINK_OFF):
                # llama-server (reasoning_format none) returns the template's empty think block, which is part of
                # the prompt, at the head of the content; the reply the model generated starts after it.
                text = text[len(THINK_OFF):]
                echo = True
        if text is None:
            text = ""
        usage = usage or {}
        return Reply(text=text, finish_reason=choice.get("finish_reason"), prompt_tokens=usage.get("prompt_tokens"),
                     completion_tokens=usage.get("completion_tokens"), latency_s=round(latency, 3),
                     timings=timings, reasoning_split=split, prefill_echo=echo)


def enforces_top_k(client: Any) -> bool:
    """Whether the requests carry ``top_k``: not with ``--no-extra-params``, and not to Ollama, which takes it
    from its Modelfile. A replay keeps the value its column recorded."""
    if isinstance(client, ReplayClient):
        return True
    return client.extra_params and client.flavor != "ollama"


def generation_record(client: Any) -> dict[str, Any]:
    """How the replies were drawn, beyond the sampling fields: ``config.generation``, which ``compare`` checks.
    A replay keeps the record of the column it replays."""
    if isinstance(client, ReplayClient):
        return client.generation
    return {"client": "rjdebug", "mode": client.mode, "flavor": client.flavor, "extra_params": client.extra_params,
            "chat_template_sha256": client.server.get("chat_template_sha256"),
            "server_version": client.server.get("version")}


def redacted_argv(argv: list[str]) -> list[str]:
    """``argv`` with the value of ``--api-key``, or of any abbreviation argparse accepts for it, replaced."""
    out, hide = [], False
    for arg in argv:
        if hide:
            out.append("<redacted>")
            hide = False
            continue
        name, eq, _ = arg.partition("=")
        if len(name) >= 4 and "--api-key".startswith(name):
            out.append(f"{name}=<redacted>" if eq else name)
            hide = not eq
            continue
        out.append(arg)
    return out


class ReplayClient:
    """A harness column's recorded replies, served in place of an endpoint (``--replay``)."""

    mode = "replay"
    flavor = "replay"
    endpoint = "replay"
    server: dict[str, Any] = {}

    def __init__(self, column: dict[str, Any]) -> None:
        self.column = column
        self.model = column["config"]["source"]
        self.generation = column["config"].get("generation")
        self.raw = {(w["scenario_id"], w["episode"], w["step"]): w["raw"] for w in column["wire"]}
        self.sampling = {k: column["config"].get(k2) for k, k2 in
                         (("temperature", "temperature"), ("top_p", "top_p"), ("top_k", "top_k"),
                          ("max_tokens", "max_new_tokens"))}

    def reply_for(self, sid: str, index: int, step: int) -> Reply:
        try:
            return Reply(text=self.raw[(sid, index, step)])
        except KeyError:
            raise EndpointError(f"the column has no recorded reply for {sid}#{index} step {step}") from None


# ---------------------------------------------------------------------------
# episodes
# ---------------------------------------------------------------------------


@dataclass
class EpisodeResult:
    scenario_id: str
    index: int
    episode: Any  # rocjitsu_env.Episode
    steps: list[dict[str, Any]]
    error: str | None = None
    seconds: float = 0.0


def play_episode(scenario: Any, index: int, client: Client | ReplayClient, system: str, *, seed: int = EVAL_SEED,
                 max_turns: int | None = None) -> EpisodeResult:
    """One episode: observe, ask the endpoint for one reply, act, until the episode ends."""
    rj = load_env()
    ep = rj.Episode(scenario, index=index, budget=max_turns or rj.MAX_TURNS)
    steps: list[dict[str, Any]] = []
    t0 = time.time()
    sid = scenario.scenario_id
    turn = 0
    while True:
        user = ep.observe()
        if user is None:
            break
        turn += 1
        s = episode_seed(seed, sid, index, turn)
        try:
            reply = (client.reply_for(sid, index, turn) if isinstance(client, ReplayClient)
                     else client.complete(system, user, s))
        except EndpointError as exc:
            return EpisodeResult(sid, index, ep, steps, error=str(exc), seconds=time.time() - t0)
        ep.act(reply.text)
        rec = ep.steps[-1]
        steps.append({"step": turn, "seed": s, "prompt": user, "raw": reply.text, "kind": rec.action.kind,
                      "tool": rec.action.tool, "kernel": rec.action.kernel, "error": rec.action.error,
                      "finish_reason": reply.finish_reason, "prompt_tokens": reply.prompt_tokens,
                      "completion_tokens": reply.completion_tokens, "latency_s": reply.latency_s,
                      "reasoning_split": reply.reasoning_split, "prefill_echo": reply.prefill_echo,
                      "timings": {k: reply.timings.get(k) for k in ("prompt_n", "prompt_ms", "predicted_n",
                                                                     "predicted_ms", "cache_n")
                                  if k in reply.timings} if reply.timings else None})
    return EpisodeResult(sid, index, ep, steps, seconds=time.time() - t0)


def load_scenarios(archive: str, split: str, only: list[str], reward: str, limit: int = 0) -> list[Any]:
    """The split's scenarios in the harness's order (``eval_rocjitsu.pick_scenarios``: faulty by kernel/op/id, then clean)."""
    rj = load_env()
    import eval_rocjitsu  # noqa: PLC0415

    scenarios = eval_rocjitsu.pick_scenarios(rj.load_corpus(archive, split=split, only=only, reward=reward), 0, -1)
    return scenarios[:limit] if limit else scenarios


def run(scenarios: list[Any], client: Client | ReplayClient, *, episodes: int = 2, seed: int = EVAL_SEED,
        concurrency: int = 4, max_turns: int | None = None, progress: bool = True) -> tuple[str, list[EpisodeResult]]:
    """Play ``episodes`` episodes of every scenario; returns ``(system prompt, results in scenario/episode order)``."""
    rj = load_env()
    system = rj.system_for(scenarios)
    jobs = [(s, i) for s in scenarios for i in range(episodes)]
    results: dict[tuple[str, int], EpisodeResult] = {}
    lock = threading.Lock()
    t0 = time.time()

    def one(job: tuple[Any, int]) -> EpisodeResult:
        res = play_episode(job[0], job[1], client, system, seed=seed, max_turns=max_turns)
        with lock:
            results[(res.scenario_id, res.index)] = res
            if progress:
                done = len(results)
                tag = "ERROR " + (res.error or "")[:120] if res.error else (
                    f"{res.episode.outcome:<13} {res.episode.score().total:+6.2f} {res.episode.turns_used} turn(s)")
                print(f"[{done:>4}/{len(jobs)} {time.time() - t0:6.0f}s] {res.scenario_id[:44]:<44} "
                      f"#{res.index} {tag}", flush=True)
        return res

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        list(pool.map(one, jobs))
    return system, [results[(s.scenario_id, i)] for s, i in jobs]


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------


def bootstrap_mean(values: list[float], draws: int = 4000, seed: int = 0) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(draws))
    return {"n": len(values), "mean": round(statistics.mean(values), 4),
            "ci95": [round(means[int(0.025 * draws)], 4), round(means[int(0.975 * draws) - 1], 4)]}


def column(scenarios: list[Any], results: list[EpisodeResult], client: Client | ReplayClient, system: str, *,
           split: str, episodes: int, seed: int, max_turns: int, argv: list[str]) -> dict[str, Any]:
    """The harness's column format (``eval_rocjitsu.run_column``), from the played episodes."""
    rj = load_env()
    import eval_rocjitsu  # noqa: PLC0415

    by_sid: dict[str, list[EpisodeResult]] = {}
    for r in results:
        if r.error is None:
            by_sid.setdefault(r.scenario_id, []).append(r)
    groups, wire = [], []
    for s in scenarios:
        played = by_sid.get(s.scenario_id, [])
        if not played:
            continue
        eps = [r.episode for r in played]
        rewards = [e.score().total for e in eps]
        mean = statistics.mean(rewards)
        sd = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
        depth = max(e.turns_used for e in eps)
        groups.append(rj.group_stats(s, eps, rewards, [0.0] * len(eps), mean, sd, depth))
        for r in played:
            ep = r.episode
            for st in r.steps:
                wire.append({"scenario_id": s.scenario_id, "episode": r.index, "step": st["step"], "raw": st["raw"],
                             "kind": st["kind"], "tool": st["tool"], "kernel": st["kernel"], "error": st["error"],
                             "reward": ep.score().total, "advantage": 0.0, "terminal": ep.outcome,
                             "solved": ep.solved, "episode_steps": ep.turns_used, "tool_cost": round(ep.tool_cost(), 4),
                             "seed": st["seed"], "finish_reason": st["finish_reason"],
                             "prompt_tokens": st["prompt_tokens"], "completion_tokens": st["completion_tokens"],
                             "latency_s": st["latency_s"]})
    config = {
        "source": f"{client.endpoint}#{client.model}", "seed": seed, "split": split, "episodes": episodes,
        "temperature": client.sampling["temperature"], "top_p": client.sampling["top_p"],
        "top_k": client.sampling["top_k"] if enforces_top_k(client) else None,
        "max_new_tokens": client.sampling["max_tokens"], "generation": generation_record(client),
        "max_episode_steps": max_turns, "param_dtype": None, "seeding": "per-episode-step (server-side seed)",
        "scenarios": [s.scenario_id for s in scenarios], "corpus": {s.scenario_id: s.digest for s in scenarios},
        "blas_backend": None, "system_sha256": hashlib.sha256(system.encode()).hexdigest(),
        "prompt_profile": rj.resolve_prompt(), "reward": scenarios[0].reward if scenarios else None,
        "runtime": {"python": sys.version.split()[0], "client": "rjdebug.py"},
        "client": {"mode": client.mode, "flavor": client.flavor, "server": client.server,
                   "argv": redacted_argv(argv)},
    }
    doc: dict[str, Any] = {"config": config, "groups": groups, "wire": wire}
    errors = sum(1 for r in results if r.error is not None)
    if errors:
        # Failed episodes are not in the groups, so metrics over the rest would not describe config.scenarios.
        doc.update(incomplete=True, errors=errors)
    else:
        doc["metrics"] = eval_rocjitsu.summarize(eval_rocjitsu.episode_rows({"groups": groups}))
    return doc


def summary(col: dict[str, Any], results: list[EpisodeResult], wall: float) -> dict[str, Any]:
    """The run's numbers; for an incomplete column they are over the episodes that finished."""
    rj = load_env()
    import eval_rocjitsu  # noqa: PLC0415

    rows = [e for g in col["groups"] for e in g["episodes"]]
    metrics = col["metrics"] if "metrics" in col else eval_rocjitsu.summarize(rows)
    clean = [r for r in rows if rj.is_clean_id(r["scenario_id"])]
    faulty = [r for r in rows if not rj.is_clean_id(r["scenario_id"])]
    per_scn: dict[str, list[float]] = {}
    for r in rows:
        per_scn.setdefault(r["scenario_id"], []).append(r["total"])
    outcomes = Counter(r["outcome"] for r in rows)
    steps = [st for r in results if r.error is None for st in r.steps]
    ptok = [st["prompt_tokens"] for st in steps if st["prompt_tokens"] is not None]
    ctok = [st["completion_tokens"] for st in steps if st["completion_tokens"] is not None]
    out = {
        "episodes": len(rows),
        "scenarios": len(per_scn),
        "errors": sum(1 for r in results if r.error is not None),
        "error_examples": [f"{r.scenario_id}#{r.index}: {r.error}" for r in results if r.error][:5],
        "mean_score": bootstrap_mean([statistics.mean(v) for v in per_scn.values()]),
        "mean_score_faulty": bootstrap_mean([statistics.mean(v) for k, v in per_scn.items() if not rj.is_clean_id(k)]),
        "mean_score_clean": bootstrap_mean([statistics.mean(v) for k, v in per_scn.items() if rj.is_clean_id(k)]),
        "solved_rate": metrics["all"]["solved_rate"] if metrics["all"] else None,
        "exact_rate_faulty": round(sum(r["outcome"] == "exact" for r in faulty) / len(faulty), 4) if faulty else None,
        "near_rate_faulty": round(sum(r["outcome"] == "near" for r in faulty) / len(faulty), 4) if faulty else None,
        "missed_bug_rate_faulty": round(outcomes["missed_bug"] / len(faulty), 4) if faulty else None,
        "kernel_wrong_rate_faulty": round(outcomes["kernel_wrong"] / len(faulty), 4) if faulty else None,
        "false_alarms": outcomes["false_alarm"],
        "false_alarm_rate_clean": round(outcomes["false_alarm"] / len(clean), 4) if clean else None,
        "no_answer_rate": round(outcomes["no_answer"] / len(rows), 4) if rows else None,
        "malformed_replies": sum(1 for st in steps if st["kind"] == "malformed"),
        "replies": len(steps),
        "truncated_replies": sum(1 for st in steps if st["finish_reason"] == "length"),
        "reasoning_split_replies": sum(1 for st in steps if st.get("reasoning_split")),
        "prefill_echo_stripped": sum(1 for st in steps if st.get("prefill_echo")),
        "mean_turns": metrics["all"]["mean_turns"] if metrics["all"] else None,
        "mean_tool_cost": metrics["all"]["mean_tool_cost"] if metrics["all"] else None,
        "outcomes": dict(outcomes),
        "tokens": {"prompt_total": sum(ptok), "completion_total": sum(ctok),
                   "prompt_max": max(ptok) if ptok else None,
                   "prompt_mean": round(statistics.mean(ptok), 1) if ptok else None,
                   "completion_mean": round(statistics.mean(ctok), 1) if ctok else None},
        "wall_seconds": round(wall, 1),
        "completion_tokens_per_second": round(sum(ctok) / wall, 1) if wall and ctok else None,
        "incomplete": bool(col.get("incomplete")),
        "metrics": metrics,
    }
    return out


def format_summary(s: dict[str, Any], col: dict[str, Any]) -> str:
    def ci(d: dict[str, Any]) -> str:
        return f"{d['mean']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]" if d.get("n") else "-"

    cfg = col["config"]
    lines = [
        f"rjdebug  {cfg['source']}  split {cfg['split']}  {s['scenarios']} scenario(s) x {cfg['episodes']} episode(s)"
        f"  seed {cfg['seed']}  mode {cfg['client']['mode']}/{cfg['client']['flavor']}",
        f"  mean score (by scenario, 95% CI)   {ci(s['mean_score'])}",
        f"    faulty scenarios                 {ci(s['mean_score_faulty'])}",
        f"    clean controls                   {ci(s['mean_score_clean'])}",
        f"  solved rate                        {s['solved_rate']}",
        f"  exact location (faulty episodes)   {s['exact_rate_faulty']}",
        f"  near location (faulty episodes)    {s['near_rate_faulty']}",
        f"  missed bug (faulty episodes)       {s['missed_bug_rate_faulty']}",
        f"  false alarms (clean episodes)      {s['false_alarms']} ({s['false_alarm_rate_clean']})",
        f"  no answer                          {s['no_answer_rate']}",
        f"  malformed replies                  {s['malformed_replies']} of {s['replies']}"
        f" (truncated at max_tokens: {s['truncated_replies']})",
        f"  mean turns / tool cost             {s['mean_turns']} / {s['mean_tool_cost']}",
        f"  outcomes                           {json.dumps(s['outcomes'])}",
        f"  tokens: prompt {s['tokens']['prompt_total']} (max {s['tokens']['prompt_max']}),"
        f" completion {s['tokens']['completion_total']}; wall {s['wall_seconds']} s;"
        f" {s['completion_tokens_per_second']} completion tok/s",
    ]
    if s["errors"]:
        lines.append(f"  ENDPOINT ERRORS: {s['errors']} episode(s) not scored, e.g. {s['error_examples'][:2]}")
    return "\n".join(lines)


def write_transcripts(root: Path, results: list[EpisodeResult]) -> None:
    rj = load_env()
    for r in results:
        ep = r.episode
        s = ep.scenario
        d = root / r.scenario_id.replace("/", "_")
        d.mkdir(parents=True, exist_ok=True)
        truth = ("clean control (no bug)" if s.clean else
                 f"{s.family} in {s.kernel} at {', '.join(rj.hexoff(o) for o in s.offsets)} (operator {s.op})")
        out = [f"# {r.scenario_id} episode {r.index}", "", f"ground truth: {truth}", ""]
        for st in r.steps:
            out += [f"## turn {st['step']} (seed {st['seed']})", "", "### prompt (user message)", "", "```",
                    st["prompt"], "```", "", "### reply", "", "```", st["raw"], "```", "",
                    f"parsed: {st['kind']}" + (f" {st['tool']} {st['kernel']}".rstrip() if st["kind"] == "tool" else "")
                    + (f" -- {st['error']}" if st["error"] else ""), ""]
        if r.error:
            out += [f"**endpoint error, episode not scored:** {r.error}", ""]
        else:
            out += ["## result", "", f"outcome {ep.outcome}, score {ep.score().total:+.2f}, answer {json.dumps(ep.answer)}",
                    "", "events: " + ", ".join(f"{e.name} {e.points:+.2f}" for e in ep.events), ""]
        (d / f"ep{r.index:03d}.md").write_text("\n".join(out), encoding="utf-8")
        with open(d / f"ep{r.index:03d}.jsonl", "w", encoding="utf-8") as fh:
            for st in r.steps:
                fh.write(json.dumps(st) + "\n")
            fh.write(json.dumps({"summary": ep.summary(), "events": [e.as_dict() for e in ep.events],
                                 "error": r.error}) + "\n")


def check_replay(col: dict[str, Any], recorded: dict[str, Any], selected: set[str]) -> dict[str, Any]:
    """Hold every replayed episode to the column it was replayed from, over the scenarios ``selected``."""
    want = {(e["scenario_id"], e["index"]): e for g in recorded["groups"] for e in g["episodes"]
            if e["scenario_id"] in selected}
    got = {(e["scenario_id"], e["index"]): e for g in col["groups"] for e in g["episodes"]}
    keys = ("outcome", "steps", "solved", "tool_cost", "answer", "sequence", "total")
    same = [k for k in want if k in got and all(got[k][f] == want[k][f] for f in keys)]
    bad = sorted(want.keys() - set(same)) + sorted(got.keys() - want.keys())
    return {"episodes": len(want), "replayed": len(got), "identical": len(same),
            "mismatches": [f"{k[0]}#{k[1]}" for k in bad[:10]]}


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--endpoint", help="OpenAI-compatible API base, e.g. http://127.0.0.1:8080/v1")
    p.add_argument("--model", default="rjx", help="model name the server expects (llama-server ignores it)")
    p.add_argument("--archive", required=True, help="corpus archive, e.g. .../corpus-v2/archive")
    p.add_argument("--split", default="test", help="split of the archive (corpus-v2: val, test, test_reencode, ...)")
    p.add_argument("--scenario", action="append", default=[], metavar="SID", help="only this scenario (repeatable)")
    p.add_argument("--limit", type=int, default=0, help="only the first N scenarios of the split, in harness order")
    p.add_argument("--episodes", type=int, default=2, help="episodes per scenario (evaluation protocol: 2)")
    p.add_argument("--concurrency", type=int, default=4, help="episodes in flight (match the server's slots)")
    p.add_argument("--transcript", action="store_true", help="write one readable transcript per episode")
    p.add_argument("--out", type=Path, default=None, help="output directory (default rjdebug-<split>-<time>)")
    p.add_argument("--seed", type=int, default=EVAL_SEED, help="base seed; reply seeds as eval_episodes.episode_seed")
    p.add_argument("--mode", choices=("chat", "completions"), default="chat",
                   help="chat: the server applies the chat template; completions: this tool renders it")
    p.add_argument("--flavor", choices=("auto", "llamacpp", "ollama", "generic"), default="auto",
                   help="llamacpp pins the sampler chain to HF's order (auto-detected from GET /props); "
                        "ollama turns thinking off with reasoning_effort none (auto-detected from GET /api/version)")
    p.add_argument("--no-extra-params", action="store_true",
                   help="send only OpenAI-spec fields (drops top_k, min_p, chat_template_kwargs)")
    p.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    p.add_argument("--timeout", type=float, default=900.0, help="seconds per request")
    p.add_argument("--temperature", type=float, default=EVAL_SAMPLING["temperature"])
    p.add_argument("--top-p", type=float, default=EVAL_SAMPLING["top_p"])
    p.add_argument("--top-k", type=int, default=EVAL_SAMPLING["top_k"])
    p.add_argument("--max-tokens", type=int, default=EVAL_SAMPLING["max_tokens"])
    p.add_argument("--max-turns", type=int, default=None,
                   help="turn budget; only the environment's 8, which the system prompt states")
    p.add_argument("--prompt", default="procedure", help="prompt profile the model was trained under: procedure")
    p.add_argument("--reward", default="v2", help="reward version the model was trained under: v2")
    p.add_argument("--env-code", default=None, help=f"code snapshot with examples/rl/rocjitsu_env.py "
                                                     f"(default ${ENV_CODE_VAR} or {DEFAULT_ENV_CODE})")
    p.add_argument("--replay", type=Path, default=None, metavar="COLUMN_JSON",
                   help="offline: replay a harness column's recorded replies and check every episode reproduces")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    args = build_parser().parse_args(argv)
    if args.episodes < 1:
        print("--episodes must be >= 1", file=sys.stderr)
        return 2
    recorded = json.loads(args.replay.read_text()) if args.replay else None
    if recorded is not None:
        # Score the replay under the prompt and reward the column was played under.
        args.prompt = recorded["config"].get("prompt_profile") or args.prompt
        args.reward = recorded["config"].get("reward") or args.reward
    os.environ["AORTA_RJ_PROMPT"] = args.prompt
    os.environ["AORTA_RJ_REWARD"] = args.reward
    rj = load_env(args.env_code)
    try:
        from eval_episodes import episode_seed as harness_seed  # noqa: PLC0415
    except ImportError:
        harness_seed = None
    if harness_seed is not None and harness_seed(EVAL_SEED, "x/y", 1, 2) != episode_seed(EVAL_SEED, "x/y", 1, 2):
        raise SystemExit("episode_seed here differs from eval_episodes.episode_seed")
    if recorded is not None:
        client: Client | ReplayClient = ReplayClient(recorded)
        args.split = recorded["config"]["split"]
        args.episodes = recorded["config"]["episodes"]
        args.seed = recorded["config"]["seed"]
        args.max_turns = args.max_turns or recorded["config"].get("max_episode_steps")
        only = list(args.scenario) or list(recorded["config"]["scenarios"])
    else:
        if not args.endpoint:
            print("--endpoint is required (or --replay)", file=sys.stderr)
            return 2
        client = Client(args.endpoint, args.model, mode=args.mode, flavor=args.flavor, api_key=args.api_key,
                        timeout=args.timeout, extra_params=not args.no_extra_params,
                        sampling={"temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k,
                                  "max_tokens": args.max_tokens})
        only = list(args.scenario)
    if args.max_turns not in (None, rj.MAX_TURNS):
        # The system prompt states the environment's budget; the training and HF evaluator refuse others too.
        print(f"a turn budget of {args.max_turns} is not supported: the system prompt states {rj.MAX_TURNS}",
              file=sys.stderr)
        return 2
    scenarios = load_scenarios(args.archive, args.split, only, args.reward, args.limit)
    if not scenarios:
        print("no scenarios selected", file=sys.stderr)
        return 2
    if recorded is not None:
        digests = recorded["config"].get("corpus")
        for s in scenarios:
            got = digests.get(s.scenario_id) if isinstance(digests, dict) else None
            if got != s.digest:
                # A changed tool observation can keep the outcome and score, so the episodes alone would not show it.
                print(f"{args.replay} was played against another archive of {s.scenario_id} (recorded digest "
                      f"{str(got)[:12]}, archive {s.digest[:12]})", file=sys.stderr)
                return 2
    system = rj.system_for(scenarios)
    sha = hashlib.sha256(system.encode()).hexdigest()
    if recorded is not None and recorded["config"].get("system_sha256") not in (None, sha):
        print(f"{args.replay} was played under system prompt {recorded['config']['system_sha256'][:12]}; this run "
              f"renders {sha[:12]} (pass the --prompt and --reward it was played under)", file=sys.stderr)
        return 2
    trained = TRAINED_SYSTEM_SHA256.get((args.prompt, args.reward))
    if not args.quiet:
        print(f"[rjdebug] {len(scenarios)} scenario(s) x {args.episodes} episode(s); system prompt sha256 {sha[:12]}"
              f" ({'the trained one' if sha == trained else 'NOT the prompt the model was trained under'})",
              flush=True)
        if not isinstance(client, ReplayClient):
            print(f"[rjdebug] endpoint {client.endpoint} mode {client.mode} flavor {client.flavor} "
                  f"{json.dumps(client.server) if client.server else ''}", flush=True)
    out = args.out or Path(f"rjdebug-{args.split}-{time.strftime('%Y%m%d-%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)
    max_turns = args.max_turns or rj.MAX_TURNS
    t0 = time.time()
    system, results = run(scenarios, client, episodes=args.episodes, seed=args.seed, concurrency=args.concurrency,
                          max_turns=max_turns, progress=not args.quiet)
    wall = time.time() - t0
    col = column(scenarios, results, client, system, split=args.split, episodes=args.episodes, seed=args.seed,
                 max_turns=max_turns, argv=argv)
    summ = summary(col, results, wall)
    if recorded is not None:
        summ["replay_check"] = check_replay(col, recorded, {s.scenario_id for s in scenarios})
    (out / "column.json").write_text(json.dumps(col, indent=1))
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    text = format_summary(summ, col)
    if recorded is not None:
        rc = summ["replay_check"]
        text += (f"\n  replay check: {rc['identical']} of {rc['episodes']} episode(s) identical to the recorded column"
                 + (f"; MISMATCHES {rc['mismatches']}" if rc["mismatches"] else ""))
    (out / "summary.txt").write_text(text + "\n")
    if args.transcript:
        write_transcripts(out / "transcripts", results)
    print(text, flush=True)
    print(f"[rjdebug] wrote {out}/column.json, summary.json, summary.txt"
          + (", transcripts/" if args.transcript else ""), flush=True)
    if summ["errors"]:
        return 1
    if recorded is not None and summ["replay_check"]["mismatches"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
