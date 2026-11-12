"""Compile-repair chains, and the rollout of one iteration (design Section 6.1).

Each chain moves through the state machine on its own thread; there is no barrier between turns:

    SUBMIT(t) -response-> VALIDATE -bad ids/logprobs-> END(logprob_error)
                             |
                          PARSE (rollout.parse_adapter) -no change-> END(no_change)
                             | rejected: -no repair turn for it, or t == max_repair_turns-> END(no_code_block)
                             |           else FEEDBACK (apply) ----------------------+
                             | edits                                                 |
                          COMPILE (CPU pool) -ok-> END(compiled)                     |
                             | fail                                                  |
                             +- t == max_repair_turns -> END(turns_exhausted)        |
                          FEEDBACK (compile) <---------------------------------------+
                             -> LEAK CHECK -fail-> END(quarantined)
                             | pass
                             +- len(ids_{t+1}) + max_new > max_model_len -> END(context_limit)
                          SUBMIT(t+1)

The parse and feedback components are chosen by name: `standin.parse` (the last ```cpp block) and `standin`
(the stand-in feedback). A component's source must be the run's. A parse adapter may also end a chain on a reply
that leaves the kernel unchanged, without a compile (`no_change`), and a feedback template may give a rejected
answer a repair turn of its own (`build_apply`); the stand-ins do neither, so a stand-in reply without a code block
ends the chain `no_code_block`.

A repair turn's ids are built as `rollout.repair_turn_ids` says: `append` (Appendix C's token append) or
`rerender` (the whole conversation rendered again through the chat template, the reply's text as an assistant
turn).

A request that fails (the engine died, the request timed out, or the engine answered with an error
or an unknown finish reason) ends the chain `engine_error`. So does reaching the iteration's generation
deadline: a chain past it starts no further turn (a request already running is not aborted; it ends
at the request timeout), and its id is in the result's `outstanding`.

At chain end the final answer is handed over at once (`on_candidate`), and the whole outcome too
(`on_chain_end`, so the orchestrator can release groups during generation), except for `engine_error` and
`quarantined` chains, which are unscored. `logprob_error` chains are handed over for scoring (logs)
but give no TrainingSample. A quarantined chain keeps only hashes: its turns are recorded without
their token ids.

Every request is one chain's one turn (never `n > 1`), with the turn's own seed from
`ids.IterationSeeds`, recorded in the turn.
"""

from __future__ import annotations

import statistics
import threading
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from ..config import RunConfig, component_source
from ..ids import IterationSeeds, chain_id, sha256_ids, sha256_text
from ..schemas import Candidate, ChainRecord, CompileResult, PromptRecord, TrainingSample, TurnRecord
from ..spans import SpanRecorder
from .compile_check import Compiler, ParseOutcome, parse_code
from .engine_pool import EnginePool, RolloutStop, logprobs_aligned, lpt_assign
from .feedback import Feedback, build_feedback
from .templates import (
    IM_END_ID, STOP_TOKEN_IDS, TemplateError, Tokenizer, next_turn_ids, rerendered_turn_ids, turn0_ids,
)

SCORED_ENDS = frozenset({"compiled", "no_change", "no_code_block", "turns_exhausted", "context_limit", "logprob_error"})
TRAINED_ENDS = frozenset({"compiled", "no_change", "no_code_block", "turns_exhausted", "context_limit"})
IDENTICAL_GROUPS_STOP = 0.05  # above this share of groups with all K answers identical, stop (sampler suspected)
REPAIR_TURN_IDS = ("append", "rerender")


class LeakCheck(Protocol):
    name: str

    def check(self, feedback: str, prompt: PromptRecord) -> bool:  # True: the feedback may be shown
        ...


class ParseAdapter(Protocol):
    name: str
    reply_stops: tuple[int, ...]  # the stop tokens dropped from the end of a reply before it is read

    def check_prompts(self, prompts: Sequence[PromptRecord], thinking: str, max_repair_turns: int) -> None:
        ...  # raises ValueError

    def check(self, text: str, thinking: str, prompt: PromptRecord) -> ParseOutcome:
        ...


class FeedbackTemplate(Protocol):
    name: str

    def build(self, errors: Sequence[dict], max_chars: int) -> Feedback:  # the compile turn
        ...

    # optional: build_apply(message, reason) -> Feedback, the turn for a rejected answer


class StandinParse:
    """`standin.parse`: the last ```cpp block of the reply (compile_check.parse_code). A reply without one ends
    the chain (no repair turn)."""

    name = "standin.parse"
    reply_stops = (IM_END_ID,)

    def check_prompts(self, prompts: Sequence[PromptRecord], thinking: str, max_repair_turns: int = 0) -> None:
        return None

    def check(self, text: str, thinking: str, prompt: PromptRecord) -> ParseOutcome:
        code = parse_code(text, thinking)
        return ParseOutcome("rejected") if code is None else ParseOutcome("edits", code=code)

    def parse(self, text: str, thinking: str, prompt: PromptRecord) -> str | None:
        return parse_code(text, thinking)


class StandinFeedback:
    """`standin`: the stand-in feedback (feedback.build_feedback)."""

    name = "standin"

    def build(self, errors: Sequence[dict], max_chars: int) -> Feedback:
        return build_feedback(errors, max_chars)


PARSE_ADAPTERS = ("standin.parse",)
FEEDBACK_TEMPLATES = ("standin",)


def _check_built(kind: str, name: str, built: Sequence[str]) -> None:
    if name not in built:
        raise RolloutStop(f"{kind} {name!r} is not available; built: {', '.join(built)}")


def _parse_adapters() -> tuple[str, ...]:
    from .. import plugins

    return PARSE_ADAPTERS + tuple(sorted(plugins.PARSE_ADAPTERS))


def _feedback_templates() -> tuple[str, ...]:
    from .. import plugins

    return FEEDBACK_TEMPLATES + tuple(sorted(plugins.FEEDBACK_TEMPLATES))


def _named(kind: str, obj: Any, name: str) -> Any:
    if getattr(obj, "name", None) != name:
        raise RolloutStop(f"the {kind} registered as {name!r} is named {getattr(obj, 'name', None)!r}")
    return obj


def check_adapter_names(parse_adapter: str, feedback_template: str) -> None:
    """Refuse a configured parse or feedback component that is not built or registered by a plugin, rather than
    silently use the stand-in in its place."""
    _check_built("parse adapter", parse_adapter, _parse_adapters())
    _check_built("feedback template", feedback_template, _feedback_templates())


def make_parse_adapter(name: str) -> ParseAdapter:
    """The configured parse: the stand-in one, or one a plugin registered."""
    from .. import plugins

    _check_built("parse adapter", name, _parse_adapters())
    if name in plugins.PARSE_ADAPTERS:
        return _named("parse adapter", plugins.PARSE_ADAPTERS[name](), name)
    return StandinParse()


def make_feedback_template(name: str) -> FeedbackTemplate:
    """The configured feedback: the stand-in one, or one a plugin registered."""
    from .. import plugins

    _check_built("feedback template", name, _feedback_templates())
    if name in plugins.FEEDBACK_TEMPLATES:
        return _named("feedback template", plugins.FEEDBACK_TEMPLATES[name](), name)
    return StandinFeedback()


def make_leak_check(name: str) -> LeakCheck | None:
    """The configured leak check. "none" is no check (stand-in prompts only); no other leak check is built."""
    if name == "none":
        return None
    raise RolloutStop(f"no leak check named {name!r}")


@dataclass(frozen=True)
class RolloutSettings:
    run_id: str
    run_seed: int
    source: str
    check_profile: str
    K: int
    max_repair_turns: int
    thinking: str
    temperature: float
    top_p: float
    top_k: int
    max_new_tokens: int
    max_feedback_chars: int
    max_model_len: int
    P: int | None = None  # if set, an iteration must have exactly P prompts
    parse_adapter: str = "standin.parse"
    feedback_template: str = "standin"
    repair_turn_ids: str = "append"  # REPAIR_TURN_IDS

    @classmethod
    def from_config(cls, cfg: RunConfig, max_model_len: int) -> RolloutSettings:
        r = cfg.rollout
        check_adapter_names(r.parse_adapter, r.feedback_template)
        return cls(
            run_id=cfg.run.run_id, run_seed=cfg.orchestrator.run_seed, source=cfg.run.source,
            check_profile=cfg.reward.profile, K=r.K, max_repair_turns=r.max_repair_turns, thinking=r.thinking,
            temperature=r.temperature, top_p=r.top_p, top_k=r.top_k, max_new_tokens=r.max_new_tokens,
            max_feedback_chars=r.max_feedback_chars, max_model_len=max_model_len, P=r.P,
            parse_adapter=r.parse_adapter, feedback_template=r.feedback_template, repair_turn_ids=r.repair_turn_ids,
        )

    def sampling_params(self, seed: int) -> dict:
        """Section 6.1, Sampling: one request per chain per turn, with its own seed."""
        return {"temperature": self.temperature, "top_p": self.top_p, "top_k": self.top_k,
                "max_new_tokens": self.max_new_tokens, "seed": seed, "stop_token_ids": list(STOP_TOKEN_IDS)}


@dataclass
class ChainOutcome:
    record: ChainRecord
    candidate: Candidate | None
    sample: TrainingSample | None
    group: int  # index of the prompt in the iteration
    cut_by_deadline: bool = False  # ended engine_error because the generation deadline had passed


@dataclass
class RolloutResult:
    iteration: int
    policy_version: str
    chains: list[ChainRecord]
    candidates: list[Candidate]
    samples: list[TrainingSample]
    metrics: dict
    stop_reasons: list[str] = field(default_factory=list)
    deadline_hit: bool = False
    outstanding: list[str] = field(default_factory=list)  # chain ids the generation deadline cut


def validate_response(res: dict, max_new: int) -> str | None:
    """None if the reply is usable; else the end reason it gives the chain."""
    fr = res.get("finish_reason")
    if not isinstance(fr, dict) or fr.get("type") not in ("stop", "length"):
        return "engine_error"
    ids = res.get("output_ids")
    if not isinstance(ids, list) or not ids or len(ids) > max_new:
        return "logprob_error"
    if fr["type"] == "stop" and isinstance(fr.get("matched"), int) and ids[-1] != fr["matched"]:
        return "logprob_error"
    if not logprobs_aligned(res):
        return "logprob_error"
    return None


def response_text(tokenizer: Tokenizer, response_ids: Sequence[int], stops: Sequence[int] = (IM_END_ID,)) -> str:
    """The reply as text, special tokens kept, without one final stop token (one of `stops`)."""
    ids = list(response_ids)
    if ids and ids[-1] in stops:
        ids = ids[:-1]
    return tokenizer.decode(ids)


def _pct(xs: Sequence[float], q: float) -> float | None:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * (len(xs) - 1)))] if xs else None


class RolloutManager:
    """C1: produce K chains per prompt from one policy version, streaming final answers."""

    def __init__(
        self,
        settings: RolloutSettings,
        pool: EnginePool,
        tokenizer: Tokenizer,
        compiler: Compiler,
        *,
        leak_check: LeakCheck | None = None,
        spans: SpanRecorder | None = None,
        compile_workers: int = 96,
        mean_response_tokens: float | None = None,
        parse_adapter: ParseAdapter | None = None,
        feedback_template: FeedbackTemplate | None = None,
    ):
        self.s = settings
        self.pool = pool
        self.tok = tokenizer
        self.compiler = compiler
        self.leak_check = leak_check
        self.parse_adapter = parse_adapter if parse_adapter is not None else make_parse_adapter(settings.parse_adapter)
        self.feedback_template = (feedback_template if feedback_template is not None
                                  else make_feedback_template(settings.feedback_template))
        for kind, got, want in (("parse adapter", self.parse_adapter.name, settings.parse_adapter),
                                ("feedback template", self.feedback_template.name, settings.feedback_template)):
            if got != want:
                raise RolloutStop(f"the {kind} given is {got!r}, the settings name {want!r}")
        if settings.repair_turn_ids not in REPAIR_TURN_IDS:
            raise RolloutStop(f"repair_turn_ids must be one of {REPAIR_TURN_IDS}, got {settings.repair_turn_ids!r}")
        self.reply_stops = tuple(self.parse_adapter.reply_stops)
        self.spans = spans if spans is not None else SpanRecorder()
        self.compile_workers = compile_workers
        self.mean_response_tokens = (
            float(mean_response_tokens) if mean_response_tokens is not None else settings.max_new_tokens / 2
        )

    # --- checks before anything is submitted ---------------------------------------------------------

    def _check_prompts(self, prompts: Sequence[PromptRecord]) -> list[list[int]]:
        ids_seen = [p.prompt_id for p in prompts]
        if not prompts:
            raise RolloutStop("an iteration needs at least one prompt")
        if len(set(ids_seen)) != len(ids_seen):
            raise RolloutStop("a prompt id repeats within the iteration")
        if self.s.P is not None and len(prompts) != self.s.P:
            raise RolloutStop(f"the iteration has {len(prompts)} prompts, rollout.P is {self.s.P}")
        for p in prompts:
            if p.source != self.s.source:
                raise RolloutStop(f"prompt {p.prompt_id!r} has source {p.source!r} in a {self.s.source} run")
        for kind, name in (("parse adapter", self.parse_adapter.name), ("feedback template", self.feedback_template.name)):
            if component_source(name) != self.s.source:
                raise RolloutStop(f"{kind} {name!r} is a {component_source(name)} component, in a {self.s.source} run")
        try:
            self.parse_adapter.check_prompts(prompts, self.s.thinking, self.s.max_repair_turns)
        except ValueError as exc:
            raise RolloutStop(f"{self.parse_adapter.name}: {exc}") from None
        if self.s.repair_turn_ids == "rerender" and self.s.max_repair_turns:
            for p in prompts:
                if p.messages is None:
                    raise RolloutStop(f"prompt {p.prompt_id!r} has no messages, and rebuilt repair turns re-render them")
        out = []
        for p in prompts:
            try:
                ids = turn0_ids(p, self.tok, self.s.thinking)
            except TemplateError as exc:
                raise RolloutStop(str(exc)) from None
            if len(ids) + self.s.max_new_tokens > self.s.max_model_len:
                raise RolloutStop(f"prompt {p.prompt_id!r}: {len(ids)} turn-0 ids leave no room for "
                                  f"{self.s.max_new_tokens} new tokens in max_model_len {self.s.max_model_len}")
            out.append(ids)
        return out

    # --- one chain -------------------------------------------------------------------------------

    def _run_chain(self, *, prompt: PromptRecord, group: int, ids0: list[int], k: int, engine: int,
                   seeds: IterationSeeds, iteration: int, policy_version: str,
                   compile_pool: ThreadPoolExecutor, deadline: float | None = None) -> ChainOutcome:
        s = self.s
        cid = chain_id(s.run_id, iteration, prompt.prompt_id, k)
        turns: list[TurnRecord] = []
        ids = ids0
        messages = list(prompt.messages or ())  # the conversation so far, for rebuilt repair turns
        inputs: list[list[int]] = []
        end: str | None = None
        cut = False
        for t in range(s.max_repair_turns + 1):
            if deadline is not None and self.spans.now() >= deadline:
                end, cut = "engine_error", True  # the generation deadline: no further turn starts
                break
            turn_seed = seeds.seed_for(prompt.prompt_id, k, t)
            sp = s.sampling_params(turn_seed)
            span = self.spans.start("gen.turn0" if t == 0 else "gen.repair", iteration, chain_id=cid,
                                    engine_id=engine, turn=t, input_len=len(ids))
            out = self.pool.generate(engine, ids, sp)
            res = out.result
            if not out.ok:
                span.end(error=str(res.get("kind")))
                end = "engine_error"
                break
            problem = validate_response(res, s.max_new_tokens)
            if problem == "engine_error":
                span.end(error="finish_reason")
                end = "engine_error"
                break
            cached = res.get("cached_tokens")
            if not (isinstance(cached, int) and not isinstance(cached, bool) and cached >= 0):
                cached = -1  # not reported: counted in the metrics, never a cache hit
            response_ids = [int(x) for x in res.get("output_ids") or []]
            rec = TurnRecord(
                turn=t, seed=turn_seed, input_ids_sha256=sha256_ids(ids), input_len=len(ids),
                response_ids=response_ids,
                response_logprobs=[float(v) for v in res["lp_vals"]] if problem is None else [],
                finish_reason=dict(res.get("finish_reason") or {}), cached_tokens=cached,
                ideal_cached_tokens=out.ideal_cached_tokens, t_submit=out.t_submit, t_done=out.t_done,
                compile=None, feedback_sha256=None, leak_check="not_run",
            )
            turns.append(rec)
            inputs.append(ids)
            span.end(output_len=len(response_ids), cached_tokens=cached, ideal_cached_tokens=out.ideal_cached_tokens,
                     finish=str(rec.finish_reason.get("type")))
            if problem is not None:
                end = problem
                break
            reply = response_text(self.tok, response_ids, self.reply_stops)
            parsed = self.parse_adapter.check(reply, s.thinking, prompt)
            if parsed.kind == "no_change":
                end = "no_change"  # nothing to compile: the gates decide
                break
            if parsed.kind == "rejected":
                build_apply = getattr(self.feedback_template, "build_apply", None)
                if parsed.rejection is None or build_apply is None or t == s.max_repair_turns:
                    end = "no_code_block"
                    break
                feedback = build_apply(*parsed.rejection).text
            else:
                with self.spans.span("compile", iteration, chain_id=cid, turn=t) as cspan:
                    result: CompileResult = compile_pool.submit(self.compiler.check, parsed.code,
                                                                f"c{group}-{k}-t{t}").result()
                    cspan.set(ok=result.ok, rc=result.rc, errors=len(result.errors))
                rec.compile = result
                if result.ok:
                    end = "compiled"
                    break
                if t == s.max_repair_turns:
                    end = "turns_exhausted"
                    break
                feedback = self.feedback_template.build(result.errors, s.max_feedback_chars).text
            rec.feedback_sha256 = sha256_text(feedback)
            if self.leak_check is not None:
                try:
                    passed = self.leak_check.check(feedback, prompt) is True
                except Exception:  # noqa: BLE001 - a failing check never lets feedback through
                    passed = False
                rec.leak_check = "pass" if passed else "fail"
                if not passed:
                    end = "quarantined"
                    break
            try:
                if s.repair_turn_ids == "rerender":
                    messages, ids_next = rerendered_turn_ids(self.tok, messages, reply, feedback, s.thinking)
                else:
                    ids_next = next_turn_ids(ids, response_ids, feedback, self.tok, s.thinking)
            except TemplateError:
                end = "quarantined"
                break
            if len(ids_next) + s.max_new_tokens > s.max_model_len:
                end = "context_limit"
                break
            ids = ids_next
        assert end is not None
        if end == "quarantined":
            turns = [replace(tr, response_ids=[], response_logprobs=[]) for tr in turns]
        record = ChainRecord(chain_id=cid, prompt_id=prompt.prompt_id, k=k, engine_id=engine,
                             policy_version=policy_version, turns=turns, final_turn=len(turns) - 1, end_reason=end)
        candidate = sample = None
        if end in SCORED_ENDS and turns:
            final = turns[-1]
            candidate = Candidate(
                candidate_id=cid, run_id=s.run_id, iteration=iteration, prompt_id=prompt.prompt_id, k=k,
                policy_version=policy_version,
                response_text=response_text(self.tok, final.response_ids, self.reply_stops),
                response_tokens=len(final.response_ids), max_new_tokens=s.max_new_tokens, task=prompt.task,
                check_profile=s.check_profile, rollout_compile=final.compile,
            )
            if end in TRAINED_ENDS:
                sample = TrainingSample(chain_id=cid, iteration=iteration, prompt_id=prompt.prompt_id,
                                        prompt_ids=list(inputs[-1]), response_ids=list(final.response_ids),
                                        response_logprobs=list(final.response_logprobs))
        return ChainOutcome(record, candidate, sample, group, cut_by_deadline=cut)

    # --- one iteration -------------------------------------------------------------------------------

    def run(
        self,
        prompts: Sequence[PromptRecord],
        iteration: int,
        policy_version: str,
        on_candidate: Callable[[Candidate], Any] | None = None,
        on_chain_end: Callable[[ChainOutcome], Any] | None = None,
        deadline: float | None = None,
    ) -> RolloutResult:
        """`deadline` is on this manager's span clock (`spans.now()`); None is no deadline."""
        s = self.s
        if s.temperature <= 0:
            raise RolloutStop("rollout temperature must be > 0")
        ids0 = self._check_prompts(prompts)
        if not all(self.pool.available):
            raise RolloutStop(f"engines not in service at iteration start: {self.pool.available}")
        seeds = IterationSeeds(s.run_seed, iteration, [p.prompt_id for p in prompts], s.K, s.max_repair_turns + 1)
        estimates = [len(ids) + s.K * self.mean_response_tokens for ids in ids0]
        assignment = lpt_assign(estimates, len(self.pool))
        self.pool.begin_iteration(iteration)
        callback_errors: list[str] = []
        lock = threading.Lock()
        outcomes: list[ChainOutcome] = []
        t_start = self.spans.now()

        def chain_task(g: int, k: int) -> None:
            outcome = self._run_chain(prompt=prompts[g], group=g, ids0=ids0[g], k=k, engine=assignment[g],
                                      seeds=seeds, iteration=iteration, policy_version=policy_version,
                                      compile_pool=compile_pool, deadline=deadline)
            for fn, what in ((on_candidate, outcome.candidate), (on_chain_end, outcome)):
                if fn is None or what is None:
                    continue
                try:
                    fn(what)
                except Exception as exc:  # noqa: BLE001 - reported as a stop reason, never dropped
                    with lock:
                        callback_errors.append(f"{outcome.record.chain_id}: {type(exc).__name__}: {exc}"[:300])
            with lock:
                outcomes.append(outcome)

        n_chains = len(prompts) * s.K
        with ThreadPoolExecutor(max_workers=self.compile_workers) as compile_pool:
            with ThreadPoolExecutor(max_workers=n_chains) as chains_pool:
                futures = [chains_pool.submit(chain_task, g, k) for g in range(len(prompts)) for k in range(s.K)]
                for f in futures:
                    f.result()
        t_end = self.spans.now()
        outcomes.sort(key=lambda o: (o.group, o.record.k))
        chains = [o.record for o in outcomes]
        metrics = self._metrics(iteration, prompts, outcomes, assignment, estimates, seeds, t_start, t_end)
        stop: list[str] = []
        if metrics["groups_all_identical_frac"] > IDENTICAL_GROUPS_STOP:
            stop.append(f"{metrics['groups_all_identical']} of {metrics['groups_compared']} groups have all "
                        f"{s.K} answers identical at T {s.temperature} (sampler suspected)")
        if self.pool.stop_reason:
            stop.append(self.pool.stop_reason)
        stop.extend(f"candidate hand-over failed: {e}" for e in callback_errors)
        turn0 = [o.record.turns[0] for o in outcomes if o.record.turns]
        if turn0:
            self.mean_response_tokens = statistics.fmean(len(tr.response_ids) for tr in turn0) or self.mean_response_tokens
        cut = [o.record.chain_id for o in outcomes if o.cut_by_deadline]
        metrics["cut_by_deadline"] = len(cut)
        return RolloutResult(
            iteration=iteration, policy_version=policy_version, chains=chains,
            candidates=[o.candidate for o in outcomes if o.candidate is not None],
            samples=[o.sample for o in outcomes if o.sample is not None], metrics=metrics, stop_reasons=stop,
            deadline_hit=bool(cut), outstanding=cut,
        )

    # --- metrics -----------------------------------------------------------------------------------

    def _metrics(self, iteration, prompts, outcomes, assignment, estimates, seeds, t_start, t_end) -> dict:
        s = self.s
        by_group: dict[int, list[ChainOutcome]] = {}
        for o in outcomes:
            by_group.setdefault(o.group, []).append(o)
        distinct, identical, compared = [], 0, 0
        for g in range(len(prompts)):
            answers = [tuple(o.record.turns[0].response_ids) for o in by_group.get(g, []) if o.record.turns]
            d = len(set(answers))
            distinct.append(d)
            if len(answers) >= 2:
                compared += 1
                identical += d == 1
        turns = [tr for o in outcomes for tr in o.record.turns]
        repair = [tr for tr in turns if tr.turn > 0]
        repair_ok = [tr for tr in repair if tr.cached_tokens >= tr.ideal_cached_tokens - 1]
        per_turn: dict[str, dict] = {}
        for t in range(s.max_repair_turns + 1):
            rows = [tr for tr in turns if tr.turn == t]
            lat = [tr.t_done - tr.t_submit for tr in rows]
            toks = [len(tr.response_ids) for tr in rows]
            per_turn[str(t)] = {
                "requests": len(rows),
                "latency_s_p50_p90_max": [_pct(lat, 0.5), _pct(lat, 0.9), max(lat) if lat else None],
                "completion_tokens": sum(toks),
                "completion_tokens_p50_max": [_pct(toks, 0.5), max(toks) if toks else None],
                "length_capped": sum(1 for tr in rows if tr.finish_reason.get("type") == "length"),
                "input_len_p50_max": [_pct([tr.input_len for tr in rows], 0.5), max((tr.input_len for tr in rows), default=None)],
                "cached_tokens": sum(max(tr.cached_tokens, 0) for tr in rows),
                "ideal_cached_tokens": sum(tr.ideal_cached_tokens for tr in rows),
                "compiled_ok": sum(1 for tr in rows if tr.compile is not None and tr.compile.ok),
            }
        wall = max(t_end - t_start, 1e-9)
        engines: dict[str, dict] = {}
        for e in range(len(self.pool)):
            rows = [tr for o in outcomes if o.record.engine_id == e for tr in o.record.turns]
            done = [tr.t_done for tr in rows]
            toks = sum(len(tr.response_ids) for tr in rows)
            engines[str(e)] = {
                "groups": sum(1 for a in assignment if a == e),
                "estimated_tokens": sum(est for est, a in zip(estimates, assignment) if a == e),
                "requests": len(rows), "completion_tokens": toks,
                "finish_s": (max(done) - t_start) if done else None,
                "tokens_per_s": toks / ((max(done) - t_start) or 1e-9) if done else None,
            }
        finish = [v["finish_s"] for v in engines.values() if v["finish_s"] is not None]
        end_reasons = Counter(o.record.end_reason for o in outcomes)
        n_turns = len(turns)
        aligned = sum(1 for tr in turns if tr.response_logprobs and len(tr.response_logprobs) == len(tr.response_ids))
        return {
            "prompts": len(prompts), "K": s.K, "chains": len(outcomes),
            "end_reasons": dict(sorted(end_reasons.items())),
            "turns": n_turns, "requests": n_turns + end_reasons.get("engine_error", 0),
            "distinct_turn0_per_group": distinct,
            "distinct_turn0_mean": statistics.fmean(distinct) if distinct else 0.0,
            "groups_compared": compared, "groups_all_identical": identical,
            "groups_all_identical_frac": identical / compared if compared else 0.0,
            "logprobs_aligned_turns": aligned, "logprobs_aligned_frac": aligned / n_turns if n_turns else 0.0,
            "repair_turns": len(repair), "repair_cache_ok": len(repair_ok),
            "repair_cache_ok_frac": len(repair_ok) / len(repair) if repair else None,
            "repair_cache_miss_rate": 1 - len(repair_ok) / len(repair) if repair else None,
            "cached_tokens_missing": sum(1 for tr in turns if tr.cached_tokens < 0),
            "per_turn": per_turn,
            "generation_wall_s": wall,
            "completion_tokens": sum(len(tr.response_ids) for tr in turns),
            "tokens_per_s": sum(len(tr.response_ids) for tr in turns) / wall,
            "engines": engines,
            "engine_finish_spread_s": (max(finish) - min(finish)) if finish else None,
            "salted_seeds": [{"prompt_id": e.prompt_id, "k": e.k, "turn": e.turn, "salt": e.salt} for e in seeds.salted()],
            "assignment": {p.prompt_id: a for p, a in zip(prompts, assignment)},
            "candidates": sum(1 for o in outcomes if o.candidate is not None),
            "samples": sum(1 for o in outcomes if o.sample is not None),
            "mean_response_tokens_used": self.mean_response_tokens,
            "deaths": [dict(d) for d in self.pool.deaths if d["iteration"] == iteration],
        }


def chain_requests_expected(chains: Sequence[ChainRecord]) -> int:
    """One request per chain per turn, plus one for a chain that ended on a failed request."""
    return sum(len(c.turns) + (1 if c.end_reason == "engine_error" else 0) for c in chains)
