"""C8.T3: a scan of every log, summary and TensorBoard event finds no substring of any prompt, feedback or source text
(planted canary strings must not appear), except a component's error message, which a stand-in run keeps and any other
source keeps only as a hash (Section 6.10 item 3).

Canaries are planted where the text really flows: in each prompt (its messages and so its token ids), in the code of
every reply the fake engine writes (so in every candidate's source), in every feedback text, and in the message of an
exception a component raises at iteration 3, which stops the run. The scan reads every file under the run directory
except the sealed evidence store, and everything logged.
"""

import logging
from pathlib import Path

from aorlta.episodes.writer import complete_iterations, read_episodes
from aorlta.orchestrator.fakes import FakeRunSpec, build_fake_run
from aorlta.orchestrator.ports import describe_error
from aorlta.rollout.fake_engine import BYTE_TOKEN_BASE

CANARIES = {"prompt": "CANARY-PROMPT-7f3a19", "source": "CANARY-SOURCE-9b1c44", "feedback": "CANARY-FEEDBACK-2d4e07"}


def scan(root: Path, extra_text: str = "") -> list[tuple[str, str]]:
    hits = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and "evidence" not in p.relative_to(root).parts:
            data = p.read_bytes()
            hits += [(str(p.relative_to(root)), name) for name, c in CANARIES.items() if c.encode() in data]
    hits += [("<log>", name) for name, c in CANARIES.items() if c in extra_text]
    return hits


def _run(tmp_path, caplog):
    spec = FakeRunSpec(root=str(tmp_path / "runs"), iterations=3, eval_every=1, leak_check=True,
                       canary=CANARIES["prompt"], min_scored_fraction=0.5,
                       rollout={"canary_source": CANARIES["source"], "canary_feedback": CANARIES["feedback"],
                                "raise_at_iteration": 3})
    r = build_fake_run(spec)
    with caplog.at_level(logging.DEBUG):
        res = r.orchestrator.run()
    return r, res


def _decoded(ids):
    return bytes(t - BYTE_TOKEN_BASE for t in ids if BYTE_TOKEN_BASE <= t < BYTE_TOKEN_BASE + 256).decode("utf-8", "replace")


def test_c8_t3_no_prompt_feedback_or_source_text_reaches_any_record_or_log(tmp_path, caplog, capsys):
    r, res = _run(tmp_path, caplog)
    assert res["status"] == "stopped" and res["stop"]["reason"] == "rollout_failed"
    assert CANARIES["prompt"] in res["stop"]["detail"]["error"]  # a stand-in run keeps the component's message
    # the canaries were really in the data the run handled
    assert all(CANARIES["prompt"] in (p.messages or [{}])[0].get("content", "") for p in r.pool.records())
    assert any(CANARIES["source"] in st["cand"].response_text for st in r.reward.cands.values())
    eps = [e for n in complete_iterations(r.cfg.run_dir) for e in read_episodes(r.cfg.run_dir, n)]
    assert any(CANARIES["source"] in _decoded(t["response_ids"]) for e in eps for t in e["turns"])  # as token ids
    assert any(t["feedback_sha256"] for e in eps for t in e["turns"])  # feedback was built, and kept as a hash
    # ... and none of it as text, anywhere but that message (the scan finds it, so the scan works)
    out = capsys.readouterr()
    hits = scan(r.cfg.run_dir, caplog.text + out.out + out.err)
    assert hits == [("run_result.json", "prompt")], hits
    assert {p.name for p in r.cfg.run_dir.iterdir()} >= {"iterations", "tb", "ckpt", "eval", "run_result.json"}


def test_c8_t3_for_any_other_source_a_components_message_is_kept_only_as_a_hash():
    exc = RuntimeError(f"failed while handling {CANARIES['prompt']}")
    kept, hashed = describe_error(exc, "standin"), describe_error(exc, "other")
    assert CANARIES["prompt"] in kept
    assert CANARIES["prompt"] not in hashed and hashed.startswith("RuntimeError (message sha256 ")
