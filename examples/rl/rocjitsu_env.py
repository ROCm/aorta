#!/usr/bin/env python3
"""The rocjitsu GPU-debugging environment as multi-turn GRPO episodes.

What an episode is
==================
A workload of GPU kernels, at most one of which carries one synchronization
fault injected into its machine code (or none: a clean control). The policy
calls tools, each with a fixed cost, and then answers ``{class, kernel,
offset}``. Ground truth is exact because the fault was injected at a known
instruction. Tool outputs are served from an archive recorded once per
scenario when the corpus is built, so an episode runs
offline in milliseconds and the GPU is paid once per scenario, not per step.

The interface is the one ``train_grpo_step.py`` drives for ``episode_env``:
:class:`Episode` with :meth:`Episode.observe` / :meth:`Episode.act`, and
:func:`rollout_scenario` with the same signature and the same flattening into
:class:`episode_env.Sample` -- one advantage per episode, repeated on every
step, normalised within the group of episodes on one scenario.

Design decisions (each the recommended option; reasons kept beside them)
========================================================================
1. **The points table is the prototype's, verbatim** (``POINTS``), fixed before
   any policy was scored. A test pins it value for value.
2. **The turn budget is the prototype's 8**, and every reply -- tool call,
   answer or malformed -- spends one turn. Running out is ``no_answer``.
3. **One user message per turn carrying the whole history**, not a chat
   transcript. The trainer's unit of loss is ``(user prompt, completion)``
   under one system prompt, and a rendered state keeps that unit unchanged.
4. **Parsing reuses the shipped ``aorta.agent.llm._answer_text``** (drop a
   terminated reasoning block, unwrap a code fence) and then takes the first
   JSON object in what remains. The prototype's base-model numbers were taken
   with that leniency; whether the strict path would have parsed is recorded
   per step (``strict_json``) so format drift is visible rather than hidden.
5. **An answer that fails validation is a malformed action**, not an answer:
   a class outside ``barrier|waitcnt|none`` or a non-object ``answer``. It
   costs ``malformed`` and the episode continues, as for any invalid action.
   An answer with a missing or unparseable offset is still an answer; it can
   earn kernel and class but no location (the prototype's rule).
6. **Offsets are shown in one notation, hex, everywhere**, and an answer may
   give hex text or a decimal integer. The prototype mixed hex listings with
   decimal race reports, which made unit conversion part of the task.
7. **No answer leaks into the prompt**: no scenario id, no mutant path, no
   operator name, no unmodified copy of the faulty kernel. ``disasm`` shows the
   faulty listing; tool stdout/stderr (which name the mutant file) is never
   rendered. A test asserts all of it on every scenario of the archive.
8. **Bounded prompts**: ``disasm`` returns at most :data:`DISASM_MAX_LINES`
   instructions per call and the history is elided oldest-first past
   :data:`MAX_HISTORY_CHARS`, so the loss never sees an unbounded sequence.

Known limit, stated because it does not go away: every scenario shares one
workload, so a held-out kernel's *unmodified* code is visible in training
episodes whose fault is elsewhere. :func:`rollout_scenario` counts ``disasm``
calls per kernel so that exposure is measured, not assumed.

Corpus v2 (``corpus-v2/archive``, detected by its ``scenarios.jsonl``)
======================================================================
Real PyTorch workloads, each with its own kernel set (``Scenario.kernels``); a
kernel is named by its corpus id (``k017``) or its ELF symbol. Tools are the
reference environment's (``rjenv2.py``, kept with the corpus build, not in this
repository) and return the
objects it returns, value for value -- a replay test holds the two to identical
observations and scores: ``list_kernels``, ``run_native`` (one archived run per
call; ``NOT_RUN``, at no cost, for a scenario recorded without native execution,
``native_source`` "not_run"), ``run_sim`` with ``seed`` 0-3, ``waitcheck`` with
an optional kernel, ``disasm`` (the mutated listing for the faulty kernel) and
``consan`` (not archived: an error object, free). The answer class ``memorder``
is added. The
policy sees those objects rendered as text with every offset in hex
(decision 6); ``disasm`` renders at most :data:`DISASM_MAX_LINES` lines.
Differences from ``rjenv2.py``, each deliberate and each the v1 decision named:
an invalid class or a non-object answer is malformed (5); hex text is accepted
for answer offsets and ``disasm`` ranges, and a garbage range is ignored (6).
The system prompt is the same for every scenario (the trainer has one), so the
per-scenario facts -- the workload's description and kernel count -- open each
user message, as ``rjenv2.Episode.observe`` states them.

Rewards: ``v1`` (:data:`POINTS`, the prototype's) and ``v2`` (:data:`POINTS_V2`,
pre-registered in ``docs/rl-rocjitsu-reward-v2.md``), chosen per corpus load (``reward=``,
else ``$AORTA_RJ_REWARD``, else ``v1``) so every earlier result reproduces. A v1-layout
corpus takes ``v1`` only, the reward its system prompt states.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sys
import zlib
from collections import Counter
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from aorta.agent.llm import _answer_text

sys.path.insert(0, str(Path(__file__).resolve().parent))

from episode_env import Sample  # noqa: E402
from event_reward import Event, EventScore  # noqa: E402

__all__ = [
    "CLASSES",
    "CLASSES_V2",
    "Corpus",
    "CorpusV2",
    "Episode",
    "MAX_TURNS",
    "POINTS",
    "POINTS_V2",
    "REWARD_VERSIONS",
    "SYSTEM",
    "Scenario",
    "TurnBudget",
    "corpus_layout",
    "kernel_paid",
    "load_corpus",
    "parse_action",
    "parse_action_v2",
    "reprice",
    "reward_spec",
    "reward_spec_sha256",
    "rollout_scenario",
]

#: Where the corpus lives, if ``--corpus-root`` is not given.
CORPUS_ROOT_ENV = "AORTA_RJ_CORPUS_ROOT"

#: The prototype's pre-registered points (its ``rjenv.py``), verbatim.
POINTS: dict[str, float] = {
    # terminal, decided by the answer and the injected ground truth
    "kernel_correct": 2.0,
    "class_correct": 1.0,
    "location_exact": 4.0,
    "location_near": 2.0,
    "kernel_wrong": -1.0,
    "clean_correct": 7.0,
    "false_alarm": -2.0,
    "missed_bug": -1.0,
    "no_answer": -1.0,
    "malformed": -1.0,
    # per-call tool prices (GPU time priced above CPU time)
    "cost_run_native": -0.5,
    "cost_run_sim_kernel": -0.1,
    "cost_run_sim_all": -0.6,
    "cost_waitcheck": -0.05,
    "cost_consan": -1.0,
    "cost_disasm": -0.02,
}
#: The best score an episode can earn: kernel + class + exact, or a clean "none".
MAX_SCORE = POINTS["kernel_correct"] + POINTS["class_correct"] + POINTS["location_exact"]
MAX_TURNS = 8
#: "near" is within this many instructions of an accepted offset.
NEAR_INSTRUCTIONS = 2
CLASSES = ("barrier", "waitcnt", "none")
TOOLS = ("run_native", "run_sim", "waitcheck", "consan", "disasm")
DISASM_MAX_LINES = 128
MAX_HISTORY_CHARS = 16000
ELIDED = "(output elided to keep the prompt short; call the tool again to see it)"

#: Corpus v2 (``rjenv2.py``): one more class and tool, schedule seeds, and its tool prices --
#: v1's for the tools both layouts have, ``list_kernels``, a ``run_sim all`` priced per kernel
#: of the workload, and ``consan`` free because it is not archived (no event is recorded).
CLASSES_V2 = ("barrier", "waitcnt", "memorder", "none")
TOOLS_V2 = ("list_kernels", "run_native", "run_sim", "waitcheck", "disasm", "consan")
SEEDS_V2 = (0, 1, 2, 3)
COSTS_V2: dict[str, float] = {
    "cost_list_kernels": -0.05,
    "cost_run_native": -0.5,
    "cost_run_sim_kernel": -0.1,
    "cost_run_sim_all_base": -0.1,
    "cost_run_sim_all_per_kernel": -0.05,
    "cost_waitcheck": -0.05,
    "cost_disasm": -0.02,
    "cost_consan": 0.0,
}

#: The reward versions. ``v1`` is :data:`POINTS` as it always was; ``v2`` is
#: pre-registered in ``docs/rl-rocjitsu-reward-v2.md`` before any val/test score under it was
#: seen. v2 changes exactly one thing: the kernel is worth +0.5 instead of +2,
#: and is paid only together with the right class or an exact/near offset, so
#: naming the right kernel and nothing else earns 0. Every other value -- the
#: offset credit, the clean-control credit, every penalty and tool price -- is v1's.
REWARD_ENV = "AORTA_RJ_REWARD"
REWARD_VERSIONS = ("v1", "v2")
POINTS_V2: dict[str, float] = {**POINTS, "kernel_correct": 0.5}
KERNEL_PAID_ONLY_WITH: dict[str, tuple[str, ...]] = {
    "v1": (),
    "v2": ("class_correct", "location_exact", "location_near"),
}
TERMINAL_EVENTS = ("kernel_correct", "class_correct", "location_exact", "location_near", "kernel_wrong",
                   "clean_correct", "false_alarm", "missed_bug", "no_answer", "malformed")


def resolve_reward(reward: str | None = None) -> str:
    """``reward``, else ``$AORTA_RJ_REWARD``, else ``v1``; anything else is refused."""
    value = reward or os.environ.get(REWARD_ENV) or "v1"
    if value not in REWARD_VERSIONS:
        raise SystemExit(f"reward must be one of {list(REWARD_VERSIONS)}, got {value!r}")
    return value


def check_layout_reward(layout: str, reward: str) -> None:
    """Refuse a reward the corpus layout's system prompt cannot state.

    The v1-layout prompt prints v1's points and no kernel rule, so a v1 corpus
    scored under v2 would be trained and evaluated against a reward other than
    the one shown to the policy.
    """
    if layout == "v1" and reward != "v1":
        raise SystemExit(f"reward {reward!r} is not supported on a v1-layout corpus: its system prompt states "
                         f"reward v1 only (pass reward='v1', or unset ${REWARD_ENV})")


def reward_points(reward: str) -> dict[str, float]:
    return POINTS_V2 if resolve_reward(reward) == "v2" else POINTS


def reward_spec(reward: str) -> dict[str, Any]:
    """Everything a reward version decides, as one canonical object (hashed in docs/rl-rocjitsu-reward-v2.md).

    Tool prices belong to the corpus layout, not to the reward version: both
    versions charge the same price for the same call.
    """
    version = resolve_reward(reward)
    pts = reward_points(version)
    return {
        "version": version,
        "terminal": {k: pts[k] for k in TERMINAL_EVENTS},
        "kernel_paid_only_with": list(KERNEL_PAID_ONLY_WITH[version]),
        "near_instructions": NEAR_INSTRUCTIONS,
        "tool_costs": {"v1_layout": {k: v for k, v in POINTS.items() if k.startswith("cost_")},
                       "v2_layout": dict(COSTS_V2)},
    }


def reward_spec_sha256(reward: str) -> str:
    text = json.dumps(reward_spec(reward), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def kernel_paid(reward: str, names: Collection[str]) -> bool:
    """Whether a right kernel is paid, given the answer's other terminal events."""
    need = KERNEL_PAID_ONLY_WITH[resolve_reward(reward)]
    return not need or any(n in names for n in need)


def reprice(events: Sequence[tuple[str, float]], reward: str) -> list[tuple[str, float]]:
    """``(name, points)`` events re-priced under ``reward``: terminal events take
    its points (the kernel 0 when :func:`kernel_paid` says so), tool costs are kept."""
    pts = reward_points(reward)
    names = {name for name, _ in events}
    out = []
    for name, value in events:
        if name == "kernel_correct" and not kernel_paid(reward, names):
            out.append((name, 0.0))
        elif name in TERMINAL_EVENTS:
            out.append((name, pts[name]))
        else:
            out.append((name, value))
    return out


def op_class(op: str) -> str:
    """The answer class an injection operator belongs to."""
    if op == "none":
        return "none"
    return "barrier" if op.startswith("barrier") else "waitcnt"


def hexoff(value: int | None) -> str:
    return "?" if value is None else f"0x{int(value):04x}"


# ---------------------------------------------------------------------------
# the archive
# ---------------------------------------------------------------------------


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_disasm(text: str) -> dict[str, list[tuple[int, str]]]:
    """``llvm-objdump`` text -> ``{kernel: [(kernel-relative offset, insn)]}``."""
    out: dict[str, list[tuple[int, str]]] = {}
    cur: str | None = None
    base = 0
    for line in text.splitlines():
        head = re.match(r"^([0-9a-f]+) <([^>]+)>:", line)
        if head:
            cur, base = head.group(2), int(head.group(1), 16)
            out[cur] = []
            continue
        body = re.search(r"//\s*([0-9A-Fa-f]+):\s*([0-9A-Fa-f]{8})", line)
        if cur and body:
            out[cur].append((int(body.group(1), 16) - base, line.split("//")[0].strip()))
    return out


class Corpus:
    """Archived tool outputs: per-mutant records plus per-kernel baselines.

    The layout is the prototype campaign's: ``sites.json`` (its ``meta`` holds
    kernel starts and instruction offsets), ``results.jsonl`` (one record per
    mutant, plus one ``BASELINE:<kernel>`` record per kernel), the original
    object's disassembly, and ``analysis.json`` (each mutant's class).
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        files = corpus_files(self.root)
        self.meta = json.loads(files["sites"].read_text())["meta"]
        self.base: dict[str, dict[str, Any]] = {}
        self.mut: dict[str, dict[str, Any]] = {}
        with open(files["results"], encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec["site_id"].startswith("BASELINE:"):
                    self.base[rec["kernel"]] = rec
                else:
                    self.mut[rec["site_id"]] = rec
        self.disasm = parse_disasm(files["disasm"].read_text())
        self.rows = {r["site_id"]: r for r in json.loads(files["analysis"].read_text())["rows"]}
        self.kernels = list(self.meta.get("kernel_order") or self._kernel_order())
        missing = [k for k in self.kernels if k not in self.base]
        if missing:
            raise ValueError(f"{self.root}: no BASELINE record for kernel(s) {missing}")
        self.digest = hashlib.sha256(
            "".join(_file_digest(files[k]) for k in ("sites", "results", "disasm")).encode()
        ).hexdigest()

    def _kernel_order(self) -> list[str]:
        starts = self.meta["kernel_starts"]
        return sorted(starts, key=lambda k: starts[k])


def corpus_files(root: Path) -> dict[str, Path]:
    """The four archive files, refusing a corpus that lacks any of them."""
    want = {
        "sites": ("sites.json",),
        "results": ("results.jsonl", "results_clean.jsonl"),
        "disasm": ("suite.dis", "disasm.txt"),
        "analysis": ("analysis.json",),
    }
    found: dict[str, Path] = {}
    for key, names in want.items():
        hit = next((root / n for n in names if (root / n).is_file()), None)
        if hit is None:
            raise SystemExit(f"corpus {root}: missing {key} file (one of {list(names)})")
        found[key] = hit
    return found


@dataclass(frozen=True)
class Scenario:
    """One episode's ground truth. Shared by every episode on it."""

    scenario_id: str
    kernel: str  # the faulty kernel; "" on a clean control
    op: str  # the injection operator; "none" on a clean control
    offsets: tuple[int, ...]  # accepted exact offsets
    insn_offsets: tuple[int, ...]  # the faulty kernel's instruction offsets, for "near"
    record: dict[str, Any] | None = field(default=None, compare=False, repr=False)
    corpus: Any = field(default=None, compare=False, repr=False)  # Corpus | CorpusV2
    digest: str = ""
    answer_class: str = ""  # v2: the record's class; v1: derived from the operator
    kernels: tuple[str, ...] = ()  # v2: the workload's kernels; v1: the corpus's
    workload: str = ""
    layout: str = "v1"
    reward: str = "v1"

    @property
    def clean(self) -> bool:
        return self.op == "none"

    @property
    def family(self) -> str:
        return self.answer_class or op_class(self.op)


def scenario_from_corpus(corpus: Corpus, scenario_id: str, reward: str = "v1") -> Scenario:
    check_layout_reward("v1", reward)
    digest = hashlib.sha256(f"{corpus.digest}/{scenario_id}".encode()).hexdigest()
    if scenario_id.startswith("CLEAN:"):
        return Scenario(scenario_id, "", "none", (), (), None, corpus, digest, reward=reward)
    rec = corpus.mut[scenario_id]
    offsets = [rec["kernel_offset"]]
    if rec.get("moved_to") is not None:
        offsets.append(rec["moved_to"])
    return Scenario(
        scenario_id=scenario_id,
        kernel=rec["kernel"],
        op=rec["op"],
        offsets=tuple(int(o) for o in offsets),
        insn_offsets=tuple(int(o) for o in corpus.meta["kernel_insn_offsets"][rec["kernel"]]),
        record=rec,
        corpus=corpus,
        digest=digest,
        reward=reward,
    )


# ---------------------------------------------------------------------------
# corpus v2: scenarios.jsonl + kernels.json + workloads.json + listings/
# ---------------------------------------------------------------------------

V2_FILES = ("scenarios.jsonl", "kernels.json", "workloads.json")
#: v2's splits, disjoint except ``test_reencode`` (a subset of ``test``).
V2_SPLITS = ("train", "val", "test", "test_seen_kernel", "test_reencode")
V2_PARTITION = ("train", "val", "test", "test_seen_kernel")


#: Clean-control ids: ``CLEAN:<split>:<n>`` in v1, ``CLEAN@<workload>#<n>`` in v2.
CLEAN_PREFIXES = ("CLEAN:", "CLEAN@")


def is_clean_id(scenario_id: str) -> bool:
    return scenario_id.startswith(CLEAN_PREFIXES)


def corpus_layout(root: str | Path) -> str:
    """``v2`` when the directory holds a ``scenarios.jsonl``, else ``v1``."""
    return "v2" if (Path(root) / "scenarios.jsonl").is_file() else "v1"


class CorpusV2:
    """The corpus-v2 archive, read as ``rjenv2.Archive`` reads it."""

    layout = "v2"

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        missing = [n for n in (*V2_FILES, "listings") if not (self.root / n).exists()]
        if missing:
            raise SystemExit(f"corpus {self.root}: v2 layout is missing {missing}")
        self.kernel_info: dict[str, dict[str, Any]] = json.loads((self.root / "kernels.json").read_text())
        self.workloads: dict[str, dict[str, Any]] = json.loads((self.root / "workloads.json").read_text())
        self.sym2kid = {k["symbol"]: kid for kid, k in self.kernel_info.items()}
        self.recs: dict[str, dict[str, Any]] = {}
        with open(self.root / "scenarios.jsonl", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rec = json.loads(line)
                    self.recs[rec["sid"]] = rec
        # A kernel with no instructions has an empty ``[]`` file; an absent file is a broken archive,
        # which would otherwise load and serve ``disasm`` and near-location scoring nothing.
        reachable = {k for w in self.workloads.values() for k in w.get("kernels", [])}
        reachable |= {r["kid"] for r in self.recs.values() if r.get("kid")}
        absent = [k for k in sorted(reachable) if not self._listing_path(k).is_file()]
        if absent:
            raise SystemExit(f"corpus {self.root}: {len(absent)} kernel(s) have no listing file: "
                             + ", ".join(f"{k} ({self._listing_path(k)})" for k in absent[:10]))
        self._listings: dict[str, list[tuple[int, str]]] = {}
        parts = [_file_digest(self.root / n) for n in V2_FILES]
        parts += [_file_digest(p) for p in sorted((self.root / "listings").glob("*.json"))]
        self.digest = hashlib.sha256("".join(parts).encode()).hexdigest()

    def _listing_path(self, kid: str) -> Path:
        return self.root / "listings" / f"{kid}.json"

    def listing(self, kid: str) -> list[tuple[int, str]]:
        """The ORIGINAL listing of ``kid``; the faulty kernel is served mutated (:func:`mutate_listing_v2`)."""
        if kid not in self._listings:
            path = self._listing_path(kid)
            if not path.is_file():
                raise SystemExit(f"corpus {self.root}: kernel {kid} has no listing file ({path})")
            self._listings[kid] = [tuple(x) for x in json.loads(path.read_text())]
        return self._listings[kid]

    def split_ids(self, name: str) -> list[str]:
        """A split's ids: ``split.json`` (prepared corpus), else ``splits/<name>.txt``, else the records' field."""
        spec_path = self.root / "split.json"
        if spec_path.is_file():
            spec = json.loads(spec_path.read_text())
            if name not in spec.get("splits", {}):
                raise SystemExit(f"corpus {self.root}: split.json has no split {name!r} "
                                 f"(has {sorted(spec.get('splits', {}))})")
            ids = list(spec["splits"][name])
        elif (self.root / "splits" / f"{name}.txt").is_file():
            ids = [x.strip() for x in (self.root / "splits" / f"{name}.txt").read_text().splitlines() if x.strip()]
        elif name in V2_SPLITS:
            ids = [s for s, r in self.recs.items() if r.get("split") == name]
        else:
            raise SystemExit(f"corpus {self.root}: no split {name!r}; v2 splits are {list(V2_SPLITS)}")
        unknown = [s for s in ids if s not in self.recs]
        if unknown:
            raise SystemExit(f"split {name!r} names {len(unknown)} id(s) with no record: {unknown[:5]}")
        return ids


def mutate_listing_v2(listing: list[tuple[int, str]], rec: dict[str, Any]) -> list[tuple[int, str]]:
    """``rjenv2.mutate_listing``: the scenario's byte edit applied to the disassembly."""
    m = rec.get("mutation")
    if not m:
        return listing
    lo = m["range_off"]
    hi = lo + len(m["old"]) // 2
    win = [tuple(x) for x in m["window"]]
    return [x for x in listing if x[0] < lo] + win + [x for x in listing if x[0] >= hi]


def scenario_from_v2(corpus: CorpusV2, scenario_id: str, reward: str = "v1") -> Scenario:
    rec = corpus.recs[scenario_id]
    kid = rec.get("kid") or ""
    return Scenario(
        scenario_id=scenario_id,
        kernel=kid,
        op=rec.get("op", "none"),
        offsets=tuple(int(o) for o in rec.get("offsets", [])),
        insn_offsets=tuple(int(o) for o, _ in corpus.listing(kid)) if kid else (),
        record=rec,
        corpus=corpus,
        digest=hashlib.sha256(f"{corpus.digest}/{scenario_id}".encode()).hexdigest(),
        answer_class=rec.get("cls", "none"),
        kernels=tuple(corpus.workloads[rec["workload"]]["kernels"]),
        workload=rec["workload"],
        layout="v2",
        reward=reward,
    )


def load_corpus_v2(base: Path, only: Sequence[str], split: str | None, reward: str) -> list[Scenario]:
    corpus = CorpusV2(base)
    if split is None:
        ids = [s for name in V2_PARTITION for s in corpus.split_ids(name)]
    else:
        ids = corpus.split_ids(split)
    unknown = [s for s in only if s not in ids]
    if unknown:
        raise SystemExit(f"scenario id(s) {unknown[:5]} are not in split {split!r}")
    return [scenario_from_v2(corpus, s, reward) for s in ids if not only or s in only]


def resolve_corpus_root(given: str | Path | None) -> Path:
    import os

    value = given if given else os.environ.get(CORPUS_ROOT_ENV)
    if not value:
        raise SystemExit(f"no corpus: pass --corpus-root or set {CORPUS_ROOT_ENV}")
    root = Path(value)
    if not root.is_dir():
        raise SystemExit(f"corpus root {root} is not a directory")
    return root


def read_split(root: Path) -> dict[str, Any]:
    path = root / "split.json"
    if not path.is_file():
        raise SystemExit(f"corpus {root}: no split.json (prepare_rocjitsu_corpus.py writes it)")
    return json.loads(path.read_text())


def load_corpus(
    root: str | Path | None = None,
    *,
    only: Sequence[str] = (),
    split: str | None = "train",
    reward: str | None = None,
) -> list[Scenario]:
    """The scenarios of one split, optionally narrowed to ``only``.

    An id in ``only`` that is not in the split is an error, not a silent drop.
    ``split=None`` takes every id the split file names, train then test (v2:
    train, val, test, test_seen_kernel). The layout is detected
    (:func:`corpus_layout`); ``reward`` is resolved by :func:`resolve_reward`
    and carried by every scenario, so episodes and the system prompt agree.
    """
    base = resolve_corpus_root(root)
    reward = resolve_reward(reward)
    if corpus_layout(base) == "v2":
        return load_corpus_v2(base, only, split, reward)
    check_layout_reward("v1", reward)
    spec = read_split(base)
    if split is None:
        ids = list(spec["train"]) + list(spec["test"])
    else:
        if split not in ("train", "test"):
            raise SystemExit(f"--split must be train or test, got {split!r}")
        ids = list(spec[split])
    unknown = [s for s in only if s not in ids]
    if unknown:
        raise SystemExit(f"scenario id(s) {unknown[:5]} are not in split {split!r}")
    corpus = Corpus(base)
    wanted = [s for s in ids if not only or s in only]
    absent = [s for s in wanted if not s.startswith("CLEAN:") and s not in corpus.mut]
    if absent:
        raise SystemExit(f"split names {len(absent)} id(s) with no archived record: {absent[:5]}")
    return [scenario_from_corpus(corpus, s, reward) for s in wanted]


def describe(scenario: Scenario) -> str:
    """One line for the training log -- never shown to the policy."""
    return f"{scenario.scenario_id:<30} op={scenario.op:<15} kernel={scenario.kernel or '-'}"


def with_reward(scenarios: Sequence[Scenario], reward: str) -> list[Scenario]:
    """The same scenarios scored under another reward version."""
    reward = resolve_reward(reward)
    for layout in {s.layout for s in scenarios}:
        check_layout_reward(layout, reward)
    return [replace(s, reward=reward) for s in scenarios]


# ---------------------------------------------------------------------------
# tools over the archive (structured results; the prototype's semantics)
# ---------------------------------------------------------------------------


def _kernel_result(run: dict[str, Any], kernel: str) -> dict[str, Any] | None:
    for x in run.get("kernels") or []:
        if x["kernel"] == kernel:
            return x
    return None


def mutated_listing(listing: list[tuple[int, str]], rec: dict[str, Any]) -> list[tuple[int, str]]:
    """The faulty kernel's listing, rebuilt from the original by the injected edit."""
    out = list(listing)
    index = {off: i for i, (off, _) in enumerate(out)}
    op, off = rec["op"], rec["kernel_offset"]
    i = index.get(off)
    if i is None:
        return out
    if op.endswith("_drop"):
        out[i] = (off, "s_nop 0")
    elif op.startswith("waitcnt_weaken"):
        m = re.match(r"(vmcnt|lgkmcnt)\((\d+)\)->\1\((\d+)\)", rec.get("note", ""))
        if m:
            out[i] = (off, out[i][1].replace(f"{m.group(1)}({m.group(2)})", f"{m.group(1)}({m.group(3)})"))
    elif op in ("barrier_late", "waitcnt_late") and i + 1 < len(out):
        a, b = out[i], out[i + 1]
        out[i], out[i + 1] = (a[0], b[1]), (rec["moved_to"], a[1])
    elif op == "barrier_early" and i > 0:
        a, b = out[i - 1], out[i]
        out[i - 1], out[i] = (a[0], b[1]), (b[0], a[1])
    return out


class Tools:
    """The five tools, answered from the archive for one scenario."""

    def __init__(self, scenario: Scenario, rng: random.Random) -> None:
        if scenario.corpus is None:
            raise ValueError(f"{scenario.scenario_id}: scenario has no corpus")
        self.s, self.a, self.rng = scenario, scenario.corpus, rng

    def _rec_for(self, kernel: str) -> dict[str, Any]:
        if self.s.record is not None and kernel == self.s.kernel:
            return self.s.record
        return self.a.base[kernel]

    def run_native(self, kernel: str) -> list[dict[str, Any]]:
        out = []
        for k in self.a.kernels if kernel == "all" else [kernel]:
            run = self.rng.choice(self._rec_for(k)["native"])
            kr = _kernel_result(run, k)
            gold = _kernel_result(self.a.base[k]["native"][0], k)
            if kr is None:
                out.append({"kernel": k, "status": "CRASH", "rc": run.get("rc")})
            else:
                out.append({"kernel": k, "status": kr["status"], "bad": kr.get("bad", 0),
                            "output_matches_golden": gold is not None and kr["digest"] == gold["digest"]})
        return out

    def run_sim(self, kernel: str) -> list[dict[str, Any]]:
        out = []
        for k in self.a.kernels if kernel == "all" else [kernel]:
            sim = self._rec_for(k)["sim"][0]
            kr = _kernel_result(sim, k)
            races = [
                {"type": r["type"], "access": r["access"], "wave": r["wave"], "lane": r["lane"],
                 "instructions": [{"offset": m.get("kernel_offset"), "insn": m["insn"], "wave": m["wave"]}
                                  for m in r["marks"]]}
                for r in sim.get("races") or []
            ]
            grouped: dict[tuple[Any, ...], int] = {}
            for m in sim.get("memwait") or []:
                key = (m.get("kernel_offset"), m.get("producer_offset"), m["reg"], m["counter"], m["threshold"])
                grouped[key] = grouped.get(key, 0) + 1
            memwait = [{"consumer_offset": a, "producer_offset": b, "reg": c, "counter": d,
                        "required_threshold": e, "count": n} for (a, b, c, d, e), n in grouped.items()]
            out.append({"kernel": k, "status": kr["status"] if kr else "CRASH", "races": races[:8],
                        "n_races": len(races), "memory_wait": memwait[:8], "n_memory_wait": len(memwait)})
        return out

    def waitcheck(self) -> list[dict[str, Any]]:
        rec = self.s.record if self.s.record is not None else self.a.base[self.a.kernels[0]]
        tv = self.a.meta["text_vaddr"]
        starts = self.a.meta["kernel_starts"]
        diags = []
        for d in rec["waitcheck"].get("diagnostics") or []:
            v = tv + d["text_offset"]
            k = max((kk for kk in starts if starts[kk] <= v), key=lambda kk: starts[kk])
            ks = starts[k] - tv
            diags.append({
                "kernel": k, "offset": d["text_offset"] - ks,
                "what": f"{d['what']} {d['wait']} before {d['kind']} of {d['reg']}",
                "producer_offset": d.get("producer_text_offset", 0) - ks, "producer": d.get("producer_insn"),
                "consumer_offset": d.get("consumer_text_offset", 0) - ks, "consumer": d.get("consumer_insn"),
            })
        return diags

    def consan(self, kernel: str) -> dict[str, Any]:
        c = self._rec_for(kernel).get("consan") or {}
        ks = self.a.meta["kernel_starts"][kernel] - self.a.meta["text_vaddr"]
        conflicts = []
        for x in c.get("conflicts") or []:
            try:
                conflicts.append({"first_offset": int(x["first_instruction"], 16) - ks,
                                  "second_offset": int(x["second_instruction"], 16) - ks,
                                  "workgroup": x.get("workgroup")})
            except (KeyError, TypeError, ValueError):
                pass
        return {"kernel": kernel, "conflicts": conflicts[:8], "n_conflicts": len(conflicts),
                "verdict": (c.get("verdicts") or [""])[-1][:160]}

    def listing(self, kernel: str) -> list[tuple[int, str]]:
        listing = self.a.disasm.get(kernel, [])
        if self.s.record is not None and kernel == self.s.kernel:
            listing = mutated_listing(listing, self.s.record)
        return listing

    def disasm(self, kernel: str, start: int | None, end: int | None) -> dict[str, Any]:
        lo = start if start is not None else 0
        hi = end if end is not None else 1 << 30
        lines = [(o, t) for o, t in self.listing(kernel) if lo <= o <= hi]
        return {"kernel": kernel, "lines": lines[:DISASM_MAX_LINES],
                "omitted": max(0, len(lines) - DISASM_MAX_LINES)}


class ToolsV2:
    """``rjenv2.Episode``'s tools, ported line for line: the same objects for the same calls.

    One ``random.Random`` per episode draws the native run, once per
    ``run_native`` call (``all`` included), exactly as ``rjenv2`` draws it.
    """

    def __init__(self, scenario: Scenario, rng: random.Random) -> None:
        if scenario.corpus is None:
            raise ValueError(f"{scenario.scenario_id}: scenario has no corpus")
        self.s, self.a, self.rng = scenario, scenario.corpus, rng
        self.workload = self.a.workloads[scenario.workload]

    def resolve(self, kernel: Any) -> str | None:
        """A kernel id or ELF symbol of this workload -> its id; anything else -> None."""
        try:
            if kernel in self.s.kernels:
                return kernel
            kid = self.a.sym2kid.get(kernel)
        except TypeError:
            return None
        return kid if kid in self.s.kernels else None

    def _short(self, kid: str) -> str:
        d = self.a.kernel_info[kid].get("demangled", kid)
        return d if len(d) <= 100 else d[:97] + "..."

    def _native_rep(self) -> dict[str, Any] | None:
        reps = [n for n in self.s.record["native"] if n.get("ops") is not None]
        return self.rng.choice(reps) if reps else None

    def list_kernels(self) -> list[dict[str, Any]]:
        return [dict(kernel=k["kernel"], name=self._short(k["kernel"]), dispatches=k["dispatches"], ops=k["ops"][:4])
                for k in self.workload["trace"]]

    def native_not_run(self) -> bool:
        """Recorded without native execution: there is no run to serve, and serving none costs nothing."""
        return self.s.record.get("native_source") == "not_run"

    def native(self, kernel: str) -> list[dict[str, Any]]:
        kids = list(self.s.kernels) if kernel == "all" else [kernel]
        if self.native_not_run():
            return [dict(kernel=kid, status="NOT_RUN", note="no native run exists for this scenario") for kid in kids]
        rep = self._native_rep()
        out = []
        for kid in kids:
            if rep is None or rep.get("rc") not in (0, None):
                out.append(dict(kernel=kid, status="CRASH", rc=(rep or {}).get("rc")))
                continue
            ops = [o for o in self.workload["ops"] if kid in o["kernels"]]
            if not ops:
                out.append(dict(kernel=kid, status="NO_CHECK", note="launched outside any checked op"))
                continue
            bad = [rep["ops"].get(str(o["i"])) for o in ops]
            fails = [b for b in bad if b and b[0] is False]
            out.append(dict(kernel=kid, status="FAIL" if fails else "PASS", bad=sum((b[2] or 0) for b in fails),
                            output_matches_golden=not fails, checked_ops=len(ops)))
        return out

    def sim(self, kernel: str, seed: int = 0) -> list[dict[str, Any]]:
        kids = list(self.s.kernels) if kernel == "all" else [kernel]
        rec, ops_of = self.s.record, self.workload["ops"]
        out = []
        if seed == 0:
            s = rec["sim"]
            for kid in kids:
                st = [s["ops"].get(str(o["i"])) for o in ops_of if kid in o["kernels"]]
                races = [dict(type=r["type"], access=r["access"], wave=r["wave"], lane=r["lane"],
                              instructions=r["instructions"]) for r in s["races"] if r["kernel"] == kid]
                mw = [{kk: v for kk, v in m.items() if kk != "kernel"} for m in s["memwait"] if m["kernel"] == kid]
                out.append(dict(kernel=kid, status="FAIL" if any(x and x[0] is False for x in st) else "PASS",
                                races=races[:8], n_races=len(races), memory_wait=mw[:8]))
            return out
        q = rec.get("simq", {}).get(str(seed))
        for kid in kids:
            st = [q["ops"].get(str(o["i"])) for o in ops_of if kid in o["kernels"]] if q else []
            out.append(dict(kernel=kid, seed=seed, status="FAIL" if any(x and x[0] is False for x in st) else "PASS",
                            note="schedule-seed run: output check only"))
        return out

    def waitcheck(self, kernel: str | None = None) -> list[dict[str, Any]]:
        diags = self.s.record.get("waitcheck_view") or []
        if kernel and kernel != "all":
            diags = [d for d in diags if d["kernel"] == kernel]
        return diags[:32]

    def listing(self, kernel: str) -> list[tuple[int, str]]:
        lst = self.a.listing(kernel)
        if self.s.kernel and kernel == self.s.kernel:
            lst = mutate_listing_v2(lst, self.s.record)
        return lst

    def disasm(self, kernel: str, start: int | None = None, end: int | None = None) -> str:
        lo = int(start) if start is not None else -1
        hi = int(end) if end is not None else 1 << 30
        return "\n".join(f"{off:#06x}: {txt}" for off, txt in self.listing(kernel) if lo <= off <= hi)

    def consan(self, kernel: str) -> dict[str, Any]:
        return dict(kernel=kernel, error="consan is not archived in corpus v2")


# ---------------------------------------------------------------------------
# rendering: what the policy is shown
# ---------------------------------------------------------------------------


def render_result(tool: str, kernel: str, result: Any) -> str:
    if tool == "waitcheck":
        if not result:
            return "waitcheck: no diagnostics in any kernel."
        lines = [f"waitcheck: {len(result)} diagnostic(s)."]
        for d in result[:12]:
            lines.append(f"  {d['kernel']} {hexoff(d['offset'])}: {d['what']}; producer "
                         f"{hexoff(d['producer_offset'])} {d['producer']} -> consumer "
                         f"{hexoff(d['consumer_offset'])} {d['consumer']}")
        if len(result) > 12:
            lines.append(f"  ... {len(result) - 12} more not shown")
        return "\n".join(lines)
    if tool == "run_sim":
        lines = [f"run_sim {kernel} (simulator, race detector, memory-wait diagnostics):"]
        quiet = [r["kernel"] for r in result if not r["n_races"] and not r["n_memory_wait"]]
        for r in result:
            if r["kernel"] in quiet and kernel == "all":
                continue
            lines.append(f"  {r['kernel']}: simulated output {r['status']}, {r['n_races']} race "
                         f"report(s), {r['n_memory_wait']} memory-wait warning(s)")
            for race in r["races"]:
                marks = " <-> ".join(f"{hexoff(m['offset'])} {m['insn']} [wave {m['wave']}]"
                                     for m in race["instructions"])
                lines.append(f"    race {race['type']} {race['access']} wave {race['wave']} "
                             f"lane {race['lane']}: {marks}")
            for m in r["memory_wait"]:
                lines.append(f"    memory-wait: {hexoff(m['consumer_offset'])} reads {m['reg']} before "
                             f"{m['counter']}<={m['required_threshold']}; producer "
                             f"{hexoff(m['producer_offset'])} (x{m['count']})")
            if r["n_races"] > len(r["races"]) or r["n_memory_wait"] > len(r["memory_wait"]):
                lines.append("    (further reports not shown)")
        # Known limitation, kept as in render_result_v2: a quiet FAIL/CRASH kernel reads "no findings".
        if quiet and kernel == "all":
            lines.append(f"  no findings, simulated output PASS: {', '.join(quiet)}"
                         if all(r["status"] == "PASS" for r in result if r["kernel"] in quiet)
                         else f"  no findings: {', '.join(quiet)}")
        return "\n".join(lines)
    if tool == "run_native":
        parts = []
        for r in result:
            if r["status"] == "CRASH":
                parts.append(f"  {r['kernel']}: CRASH (exit code {r.get('rc')})")
                continue
            text = f"  {r['kernel']}: {r['status']}"
            if r.get("bad"):
                text += f" ({r['bad']} wrong output elements)"
            if not r.get("output_matches_golden", True):
                text += ", output differs from the known-good run"
            parts.append(text)
        return f"run_native {kernel} (real GPU, output checked against a CPU reference):\n" + "\n".join(parts)
    if tool == "consan":
        lines = [f"consan {kernel}: {result['n_conflicts']} conflict(s)."]
        for c in result["conflicts"]:
            lines.append(f"  {hexoff(c['first_offset'])} <-> {hexoff(c['second_offset'])} "
                         f"(workgroup {c['workgroup']})")
        if result["verdict"]:
            lines.append(f"  verdict: {result['verdict']}")
        return "\n".join(lines)
    if tool == "disasm":
        lines = [f"disasm {kernel}:"] + [f"  {hexoff(o)}: {t}" for o, t in result["lines"]]
        if not result["lines"]:
            lines.append("  (no instructions in that range)")
        if result["omitted"]:
            lines.append(f"  ... {result['omitted']} more instruction(s) in that range not shown; "
                         f"request start/end to see them")
        return "\n".join(lines)
    raise ValueError(f"unknown tool {tool!r}")


def render_result_v2(tool: str, kernel: str, result: Any, seed: int = 0) -> str:
    """A v2 tool's object as text: every field ``rjenv2`` returns, offsets in hex."""
    if tool == "list_kernels":
        lines = [f"list_kernels: {len(result)} kernel(s), in dispatch order:"]
        for k in result:
            ops = ", ".join(k["ops"]) if k["ops"] else "-"
            lines.append(f"  {k['kernel']}: {k['dispatches']} dispatch(es); ops {ops}; {k['name']}")
        return "\n".join(lines)
    if tool == "waitcheck":
        scope = f" {kernel}" if kernel else ""
        if not result:
            return f"waitcheck{scope}: no diagnostics."
        lines = [f"waitcheck{scope}: {len(result)} diagnostic(s)."]
        for d in result:
            lines.append(f"  {d['kernel']} {hexoff(d['offset'])}: {d['what']}; producer "
                         f"{hexoff(d['producer_offset'])} {d['producer']} -> consumer "
                         f"{hexoff(d['consumer_offset'])} {d['consumer']}")
        return "\n".join(lines)
    if tool == "run_native":
        if result and all(r["status"] == "NOT_RUN" for r in result):
            return (f"run_native {kernel}: NOT_RUN -- this scenario has no real-GPU run, so there is no output "
                    f"check to report (no cost):\n" + "\n".join(f"  {r['kernel']}: NOT_RUN" for r in result))
        parts = []
        for r in result:
            st = r["status"]
            if st == "CRASH":
                parts.append(f"  {r['kernel']}: CRASH (exit code {r.get('rc')})")
            elif st == "NO_CHECK":
                parts.append(f"  {r['kernel']}: NO_CHECK ({r.get('note')})")
            else:
                text = f"  {r['kernel']}: {st}"
                if r.get("bad"):
                    text += f" ({r['bad']} wrong output elements)"
                if not r.get("output_matches_golden", True):
                    text += ", output differs from the CPU reference"
                parts.append(text + f"; {r.get('checked_ops')} checked op(s)")
        return (f"run_native {kernel} (one real GPU run; the ops each kernel ran checked against a CPU "
                f"float64 reference):\n" + "\n".join(parts))
    if tool == "run_sim" and seed:
        lines = [f"run_sim {kernel} seed {seed} (simulator under another workgroup interleaving; output "
                 f"check only):"]
        lines += [f"  {r['kernel']}: simulated output {r['status']}" for r in result]
        return "\n".join(lines)
    if tool == "run_sim":
        lines = [f"run_sim {kernel} (simulator, race detector, memory-wait diagnostics):"]
        quiet = [r["kernel"] for r in result if not r["n_races"] and not r["memory_wait"]]
        for r in result:
            if r["kernel"] in quiet and kernel == "all":
                continue
            lines.append(f"  {r['kernel']}: simulated output {r['status']}, {r['n_races']} race "
                         f"report(s), {len(r['memory_wait'])} memory-wait warning(s) shown")
            for race in r["races"]:
                marks = " <-> ".join(f"{hexoff(m.get('offset'))} {m.get('insn')} [wave {m.get('wave')}]"
                                     for m in race["instructions"])
                lines.append(f"    race {race['type']} {race['access']} wave {race['wave']} "
                             f"lane {race['lane']}: {marks}")
            for m in r["memory_wait"]:
                lines.append(f"    memory-wait: {hexoff(m.get('consumer_offset'))} reads {m.get('reg')} before "
                             f"{m.get('counter')}<={m.get('required_threshold')}; producer "
                             f"{hexoff(m.get('producer_offset'))} (x{m.get('count')})")
            if r["n_races"] > len(r["races"]):
                lines.append(f"    ({r['n_races'] - len(r['races'])} further race report(s) not shown)")
            if len(r["memory_wait"]) >= 8:
                lines.append("    (at most 8 memory-wait warnings are shown)")
        # Known limitation, kept: the released model was trained on this text (a quiet FAIL reads "no findings").
        if quiet and kernel == "all":
            lines.append(f"  no findings, simulated output PASS: {', '.join(quiet)}"
                         if all(r["status"] == "PASS" for r in result if r["kernel"] in quiet)
                         else f"  no findings: {', '.join(quiet)}")
        return "\n".join(lines)
    if tool == "consan":
        return f"consan {kernel}: {result['error']} (no cost)."
    if tool == "disasm":
        rows = result.splitlines() if result else []
        lines = [f"disasm {kernel}:"] + [f"  {row}" for row in rows[:DISASM_MAX_LINES]]
        if not rows:
            lines.append("  (no instructions in that range)")
        if len(rows) > DISASM_MAX_LINES:
            lines.append(f"  ... {len(rows) - DISASM_MAX_LINES} more instruction(s) in that range not shown; "
                         f"request start/end to see them")
        return "\n".join(lines)
    raise ValueError(f"unknown tool {tool!r}")


# ---------------------------------------------------------------------------
# parsing and validating a reply
# ---------------------------------------------------------------------------


def _first_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text, match.start())
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def parse_offset(value: Any) -> int | None:
    """An answer offset: a JSON integer, or hex (``"0x1c"``) / decimal text."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if re.fullmatch(r"0x[0-9a-f]+", text):
            return int(text, 16)
        if re.fullmatch(r"\d+", text):
            return int(text, 10)
    return None


@dataclass(frozen=True)
class Action:
    """A validated reply: a tool call, an answer, or why it was neither."""

    kind: str  # "tool" | "answer" | "malformed"
    tool: str = ""
    kernel: str = ""
    start: int | None = None
    end: int | None = None
    answer: dict[str, Any] | None = None
    error: str = ""
    parsed_json: bool = False  # a JSON object was found in the reply
    strict_json: bool = False  # the shipped strict parse would have accepted it
    seed: int = 0  # v2 run_sim schedule seed


def parse_action(raw: str, kernels: Sequence[str]) -> Action:
    """Turn one reply into an :class:`Action`. Never raises."""
    if not isinstance(raw, str) or not raw.strip():
        return Action("malformed", error="empty reply")
    try:
        text = _answer_text(raw)
    except ValueError as exc:
        return Action("malformed", error=str(exc))
    strict = False
    try:
        obj = json.loads(text)
        strict = isinstance(obj, dict)
    except json.JSONDecodeError:
        obj = None
    if not isinstance(obj, dict):
        obj = _first_object(text)
    if obj is None:
        return Action("malformed", error="no JSON object in the reply")
    common = {"parsed_json": True, "strict_json": strict}
    if "answer" in obj:
        ans = obj["answer"]
        if not isinstance(ans, dict):
            return Action("malformed", error="'answer' must be an object", **common)
        cls = ans.get("class")
        cls = cls.strip().lower() if isinstance(cls, str) else cls
        if cls not in CLASSES:
            return Action("malformed", error=f"answer class must be one of {list(CLASSES)}", **common)
        kernel = ans.get("kernel")
        return Action("answer", answer={"class": cls, "kernel": kernel if isinstance(kernel, str) else None,
                                        "offset": parse_offset(ans.get("offset"))}, **common)
    tool = obj.get("tool")
    if tool not in TOOLS:
        return Action("malformed", error=f"unknown tool {tool!r}; tools are {list(TOOLS)}", **common)
    if tool == "waitcheck":
        return Action("tool", tool=tool, **common)
    kernel = obj.get("kernel", "all")
    if tool in ("run_native", "run_sim"):
        if kernel != "all" and kernel not in kernels:
            return Action("malformed", error=f"unknown kernel {kernel!r}", **common)
        return Action("tool", tool=tool, kernel=kernel, **common)
    if kernel not in kernels:
        return Action("malformed", error=f"{tool} needs one kernel name, got {kernel!r}", **common)
    start = parse_offset(obj.get("start")) if tool == "disasm" else None
    end = parse_offset(obj.get("end")) if tool == "disasm" else None
    return Action("tool", tool=tool, kernel=kernel, start=start, end=end, **common)


def tool_cost_event(action: Action) -> str:
    if action.tool == "run_sim":
        return "cost_run_sim_all" if action.kernel == "all" else "cost_run_sim_kernel"
    return f"cost_{action.tool}"


def _reply_object(raw: Any) -> tuple[dict[str, Any] | None, bool, str]:
    """:func:`parse_action`'s extraction: ``(object, strict, error)``."""
    if not isinstance(raw, str) or not raw.strip():
        return None, False, "empty reply"
    try:
        text = _answer_text(raw)
    except ValueError as exc:
        return None, False, str(exc)
    strict = False
    try:
        obj = json.loads(text)
        strict = isinstance(obj, dict)
    except json.JSONDecodeError:
        obj = None
    if not isinstance(obj, dict):
        obj = _first_object(text)
    if obj is None:
        return None, False, "no JSON object in the reply"
    return obj, strict, ""


def parse_seed(value: Any) -> int | None:
    """A ``run_sim`` seed: absent or null is 0, else a whole number in :data:`SEEDS_V2`; anything else is None.

    Unlike ``rjenv2``'s ``int(seed or 0)``, a bool, text, a fraction or a non-finite number is refused, not
    truncated or raised on (``int(1e309)`` is an ``OverflowError``). Membership is tested before any
    conversion, so nothing outside :data:`SEEDS_V2` is ever converted; a bool is refused explicitly
    because ``True == 1``."""
    if value is None:
        return 0
    if isinstance(value, bool):
        return None
    return int(value) if value in SEEDS_V2 else None


def parse_action_v2(raw: str, resolve: Callable[[Any], str | None]) -> Action:
    """A v2 reply -> :class:`Action`, with ``rjenv2.Episode.act``'s kernel and seed rules.

    ``resolve`` maps a kernel id or ELF symbol of the workload to its id. As in
    ``rjenv2``: ``run_native``/``run_sim``/``disasm``/``consan`` need a kernel
    that resolves (``all`` for the first two and ``consan``), ``waitcheck``
    reads every kernel unless its ``kernel`` resolves, and a ``run_sim`` seed
    outside 0-3 is invalid. Never raises.
    """
    obj, strict, error = _reply_object(raw)
    if obj is None:
        return Action("malformed", error=error)
    common = {"parsed_json": True, "strict_json": strict}
    if "answer" in obj:
        ans = obj["answer"]
        if not isinstance(ans, dict):
            return Action("malformed", error="'answer' must be an object", **common)
        cls = ans.get("class")
        cls = cls.strip().lower() if isinstance(cls, str) else cls
        if cls not in CLASSES_V2:
            return Action("malformed", error=f"answer class must be one of {list(CLASSES_V2)}", **common)
        kernel = ans.get("kernel")
        return Action("answer", answer={"class": cls, "kernel": kernel if isinstance(kernel, str) else None,
                                        "offset": parse_offset(ans.get("offset"))}, **common)
    tool = obj.get("tool")
    if not isinstance(tool, str) or tool not in TOOLS_V2:
        return Action("malformed", error=f"unknown tool {tool!r}; tools are {list(TOOLS_V2[:-1])}", **common)
    if tool == "list_kernels":
        return Action("tool", tool=tool, **common)
    kernel = obj.get("kernel", "all")
    kid = "all" if kernel == "all" else resolve(kernel)
    if tool == "waitcheck":
        return Action("tool", tool=tool, kernel="" if kid in (None, "all") else kid, **common)
    if not kid:
        return Action("malformed", error=f"unknown kernel {kernel!r}; list_kernels names this workload's kernels",
                      **common)
    if tool == "run_sim":
        seed = parse_seed(obj.get("seed"))
        if seed is None:
            return Action("malformed", error=f"seed must be one of {list(SEEDS_V2)}", **common)
        return Action("tool", tool=tool, kernel=kid, seed=seed, **common)
    if tool in ("run_native", "consan"):
        return Action("tool", tool=tool, kernel=kid, **common)
    if kid == "all":
        return Action("malformed", error="disasm needs one kernel", **common)
    return Action("tool", tool=tool, kernel=kid, start=parse_offset(obj.get("start")),
                  end=parse_offset(obj.get("end")), **common)


# ---------------------------------------------------------------------------
# the system prompt
# ---------------------------------------------------------------------------


def system_prompt(kernels: Sequence[str], max_turns: int = MAX_TURNS) -> str:
    p = POINTS
    return (
        "You are an expert AMD GPU (gfx950, CDNA4, wave64) debugging agent. A test workload runs "
        f"{len(kernels)} HIP kernels: {', '.join(kernels)}. At most ONE kernel has ONE synchronization "
        "bug in its machine code: an s_barrier that is missing or misplaced (waves of a workgroup race "
        "on LDS), or an s_waitcnt that is missing, too weak or misplaced (an instruction uses a "
        "register or LDS value before the memory operation producing it has completed). Some "
        "workloads have no bug. Find the faulty kernel and the byte offset of the faulty "
        "synchronization instruction, or answer that there is no bug.\n"
        "Each turn, reply with exactly ONE JSON object and nothing else: a tool call or your answer.\n"
        "Tools (the cost is subtracted from your score):\n"
        f'  {{"tool": "waitcheck"}}  static wait-hazard analysis of every kernel (cost {-p["cost_waitcheck"]:g})\n'
        f'  {{"tool": "run_sim", "kernel": "<kernel>"}}  CPU simulator with a race detector and '
        f'memory-wait diagnostics (cost {-p["cost_run_sim_kernel"]:g}; "kernel": "all" runs every '
        f'kernel, cost {-p["cost_run_sim_all"]:g})\n'
        f'  {{"tool": "run_native", "kernel": "<kernel>|all"}}  real GPU run, output checked against '
        f'a CPU reference (cost {-p["cost_run_native"]:g})\n'
        f'  {{"tool": "consan", "kernel": "<kernel>"}}  ConSan LDS race sanitizer on the GPU '
        f'(cost {-p["cost_consan"]:g})\n'
        f'  {{"tool": "disasm", "kernel": "<kernel>", "start": "0x0000", "end": "0x0100"}}  instruction '
        f'listing (cost {-p["cost_disasm"]:g}; start/end optional; at most {DISASM_MAX_LINES} '
        "instructions per call)\n"
        "Final answer, which ends the episode:\n"
        '  {"answer": {"class": "barrier" or "waitcnt", "kernel": "<kernel>", "offset": "0x00bc"}}\n'
        '  {"answer": {"class": "none"}} if there is no bug\n'
        "Offsets are byte offsets from the start of the kernel, in hex as the tools print them. For a "
        "misplaced instruction, either where it is or where it belongs counts as exact.\n"
        f"Scoring: +{p['kernel_correct']:g} correct kernel, +{p['class_correct']:g} correct class, "
        f"+{p['location_exact']:g} exact offset (+{p['location_near']:g} if within "
        f"{NEAR_INSTRUCTIONS} instructions); +{p['clean_correct']:g} for \"none\" when there is no bug; "
        f"{p['false_alarm']:g} for claiming a bug when there is none; {p['missed_bug']:g} for \"none\" "
        f"when there is a bug; {p['kernel_wrong']:g} for the wrong kernel; {p['malformed']:g} for an "
        f"invalid reply; {p['no_answer']:g} if you have not answered after {max_turns} turns. Every "
        "reply uses one turn."
    )


#: The suite's kernels, in the order its workload runs them. ``SYSTEM`` is the
#: prompt for this kernel set; a corpus with other kernels gets its own via
#: :func:`system_prompt`, and the trainer reads it off the corpus.
SUITE_KERNELS = ("k_reduce", "k_transpose", "k_matmul", "k_scan", "k_hist", "k_softmax",
                 "k_bitonic", "k_stencil", "k_wavesum")
SYSTEM = system_prompt(SUITE_KERNELS)


#: The prompt profiles: ``plain`` is the v2 prompt as pre-registered; ``procedure``
#: appends :data:`PROCEDURE_V2`, the scripted recipe's strategy in words. Chosen
#: by argument, else ``$AORTA_RJ_PROMPT``, else ``plain``.
PROMPT_ENV = "AORTA_RJ_PROMPT"
PROMPT_PROFILES = ("plain", "procedure", "procedure_quirks")
PROCEDURE_V2 = (
    "\nA procedure that works on most workloads:\n"
    "1. Run waitcheck with no kernel; it checks every kernel at once. A diagnostic names an instruction that reads a "
    "value (the consumer) before the memory operation producing it (the producer) is guaranteed to have completed.\n"
    "2. If waitcheck reports nothing, run the simulator on all kernels (run_sim, kernel \"all\", seed 0). A race "
    "report names instructions of two waves that touch the same LDS location; a memory-wait warning names a consumer "
    "that reads a register before its producer completed.\n"
    "3. Read the code around the instructions the evidence names with disasm, giving a start and end around them.\n"
    "4. The faulty instruction is the synchronization instruction that should separate producer and consumer: an "
    "s_waitcnt that waits for too little (class waitcnt), or an s_barrier missing from between two waves' LDS "
    "accesses (class barrier), or one that has moved so it no longer separates them. Answer its offset.\n"
    "5. If neither waitcheck nor the simulator finds anything, answer none."
)


#: Appended after :data:`PROCEDURE_V2` by the ``procedure_quirks`` profile: a quirk of the
#: tools that a model cannot learn from its own rollouts, because without it no rollout
#: ever answers correctly where it applies (every clean control whose only waitcheck
#: diagnostics are of this kind is a false alarm for every model measured). On the train
#: split 41 of the 45 clean controls with any diagnostic carry only this kind, and it sits
#: at the injected offset 2 times against 181 for every other kind.
QUIRKS_V2 = (
    "\nA known quirk of waitcheck: a diagnostic \"missing s_waitcnt lgkmcnt(0) before def of sN\" whose producer and "
    "consumer are both s_load instructions (a later scalar load overwrites a register an earlier one is still "
    "loading) is present in unmodified compiler output and is not the bug. Never answer it. If it is waitcheck's "
    "only kind of diagnostic, treat waitcheck as having reported nothing and continue with step 2."
)


def resolve_prompt(profile: str | None = None) -> str:
    """``profile``, else ``$AORTA_RJ_PROMPT``, else ``plain``; anything else is refused."""
    value = profile or os.environ.get(PROMPT_ENV) or "plain"
    if value not in PROMPT_PROFILES:
        raise SystemExit(f"prompt profile must be one of {list(PROMPT_PROFILES)}, got {value!r}")
    return value


def system_prompt_v2(reward: str = "v1", max_turns: int = MAX_TURNS, profile: str | None = None) -> str:
    """The corpus-v2 system prompt: tools at ``rjenv2``'s prices, the scoring of ``reward``.

    The kernels belong to each scenario's workload, so they are not listed here:
    ``list_kernels`` names them, and every user turn states the workload.
    """
    chosen = resolve_prompt(profile)
    procedure = PROCEDURE_V2 if chosen in ("procedure", "procedure_quirks") else ""
    quirks = QUIRKS_V2 if chosen == "procedure_quirks" else ""
    return _system_prompt_v2_plain(reward, max_turns) + procedure + quirks


def _system_prompt_v2_plain(reward: str, max_turns: int) -> str:
    p, c = reward_points(reward), COSTS_V2
    kernel_rule = ("" if not KERNEL_PAID_ONLY_WITH[resolve_reward(reward)]
                   else " (the kernel is paid only together with the correct class or a correct offset)")
    return (
        "You are an expert AMD GPU (gfx950, CDNA4, wave64) debugging agent. A real GPU workload (PyTorch, "
        "hipBLASLt, rocBLAS, Composable Kernel or rocPRIM kernels) may contain ONE synchronization bug in one "
        "kernel's machine code: an s_barrier that is missing or misplaced, an s_waitcnt that is missing, too weak "
        "or misplaced, an LDS access moved across a barrier, or a global store moved past a release "
        "(cross-workgroup ordering). Some workloads have no bug. Find the faulty kernel and the byte offset of "
        "the faulty instruction, or answer that there is no bug.\n"
        "Each turn, reply with exactly ONE JSON object and nothing else: a tool call or your answer. A kernel is "
        "named by its id (as list_kernels prints it) or its ELF symbol.\n"
        "Tools (the cost is subtracted from your score):\n"
        f'  {{"tool": "list_kernels"}}  the workload\'s kernels, dispatch counts and ops (cost {-c["cost_list_kernels"]:g})\n'
        f'  {{"tool": "waitcheck", "kernel": "<kernel>"}}  static wait-hazard analysis; omit "kernel" for every '
        f'kernel (cost {-c["cost_waitcheck"]:g})\n'
        f'  {{"tool": "run_sim", "kernel": "<kernel>|all", "seed": 0}}  CPU simulator; seed 0 runs the race '
        f'detector and memory-wait diagnostics, seeds 1-3 other workgroup interleavings with an output check '
        f'only (cost {-c["cost_run_sim_kernel"]:g} per kernel; "all" costs {-c["cost_run_sim_all_base"]:g} + '
        f'{-c["cost_run_sim_all_per_kernel"]:g} per kernel)\n'
        f'  {{"tool": "run_native", "kernel": "<kernel>|all"}}  one real GPU run, each kernel\'s ops checked '
        f'against a CPU reference (cost {-c["cost_run_native"]:g})\n'
        f'  {{"tool": "disasm", "kernel": "<kernel>", "start": "0x0000", "end": "0x0100"}}  instruction listing '
        f'(cost {-c["cost_disasm"]:g}; start/end optional; at most {DISASM_MAX_LINES} instructions per call)\n'
        "Final answer, which ends the episode:\n"
        '  {"answer": {"class": "barrier" or "waitcnt" or "memorder", "kernel": "<kernel>", "offset": "0x00bc"}}\n'
        '  {"answer": {"class": "none"}} if there is no bug\n'
        "Offsets are byte offsets from the start of the kernel, in hex as the tools print them. For a misplaced "
        "instruction, either where it is or where it belongs counts as exact.\n"
        f"Scoring: +{p['kernel_correct']:g} correct kernel{kernel_rule}, +{p['class_correct']:g} correct class, "
        f"+{p['location_exact']:g} exact offset (+{p['location_near']:g} if within {NEAR_INSTRUCTIONS} "
        f"instructions); +{p['clean_correct']:g} for \"none\" when there is no bug; {p['false_alarm']:g} for "
        f"claiming a bug when there is none; {p['missed_bug']:g} for \"none\" when there is a bug; "
        f"{p['kernel_wrong']:g} for the wrong kernel; {p['malformed']:g} for an invalid reply; "
        f"{p['no_answer']:g} if you have not answered after {max_turns} turns. Every reply uses one turn."
    )


def system_for(scenarios: Sequence[Scenario]) -> str:
    """The system prompt for a corpus: one kernel set (v1) or one reward (v2), or a refusal."""
    layouts = {s.layout for s in scenarios}
    if layouts == {"v2"}:
        rewards = {s.reward for s in scenarios}
        if len(rewards) != 1:
            raise ValueError(f"scenarios carry {len(rewards)} reward versions; one system prompt cannot serve them")
        return system_prompt_v2(next(iter(rewards)))
    if len(layouts) != 1:
        raise ValueError("scenarios mix corpus layouts; one system prompt cannot serve them")
    kernel_sets = {tuple(s.corpus.kernels) for s in scenarios if s.corpus is not None}
    if len(kernel_sets) != 1:
        raise ValueError(f"scenarios span {len(kernel_sets)} kernel sets; one system prompt cannot serve them")
    return system_prompt(next(iter(kernel_sets)))


# ---------------------------------------------------------------------------
# the episode
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TurnBudget:
    """The only policy this environment has: how many replies an episode gets."""

    max_iterations: int = MAX_TURNS


def make_policy(max_episode_steps: int) -> TurnBudget:
    if max_episode_steps < 1:
        raise ValueError("an episode needs at least one turn")
    return TurnBudget(max_iterations=max_episode_steps)


@dataclass
class StepRecord:
    n: int
    prompt: str
    raw: str
    action: Action
    cost: float = 0.0
    header: str = ""
    body: str = ""


def episode_seed(scenario_id: str, index: int) -> int:
    """The native-run sampling stream for one episode: stable across processes."""
    return zlib.crc32(f"{scenario_id}/{index}".encode())


class Episode:
    """One debugging episode, advanced one reply at a time."""

    def __init__(self, scenario: Scenario, index: int = 0, budget: int = MAX_TURNS,
                 seed: int | None = None) -> None:
        if budget < 1:
            raise ValueError("an episode needs at least one turn")
        self.scenario, self.index, self.budget = scenario, index, budget
        self.rng = random.Random(episode_seed(scenario.scenario_id, index) if seed is None else seed)
        self.v2 = scenario.layout == "v2"
        self.tools: Any = ToolsV2(scenario, self.rng) if self.v2 else Tools(scenario, self.rng)
        self.kernels = list(scenario.kernels) if self.v2 else list(scenario.corpus.kernels)
        self.reward = resolve_reward(scenario.reward)
        self.points = reward_points(self.reward)
        self.steps: list[StepRecord] = []
        self.events: list[Event] = []
        self.done = False
        self.answer: dict[str, Any] | None = None
        self.outcome = "in_progress"
        self.last_result: Any = None
        self.pending_prompt = ""

    # -- what the policy sees ------------------------------------------------

    @property
    def turns_used(self) -> int:
        return len(self.steps)

    def tool_cost(self) -> float:
        return -sum(e.points for e in self.events if e.name.startswith("cost_"))

    def _history(self) -> list[str]:
        blocks = [[s.header, s.body] for s in self.steps]
        total = sum(len(h) + len(b) + 2 for h, b in blocks)
        # Oldest first, listings before everything else: a listing is the
        # largest result and the cheapest to fetch again.
        # A header is the call's JSON ('{"tool": "disasm", ...}'), so the listing test never matches;
        # kept, as the models were trained on this history (README, "Known limitations").
        order = sorted(range(len(blocks)), key=lambda i: (not self.steps[i].header.startswith("disasm"), i))
        for i in order:
            if total <= MAX_HISTORY_CHARS:
                break
            if blocks[i][1] and blocks[i][1] != ELIDED:
                total -= len(blocks[i][1]) - len(ELIDED)
                blocks[i][1] = ELIDED
        return [f"[turn {s.n}] {h}\n{b}" for s, (h, b) in zip(self.steps, blocks)]

    def observe(self) -> str | None:
        if self.done:
            return None
        turn = self.turns_used + 1
        left = self.budget - turn
        head = [f"Turn {turn} of {self.budget}."]
        if self.v2:
            workload = self.scenario.corpus.workloads[self.scenario.workload]
            head.insert(0, f"Workload: {workload['description']}; {len(self.kernels)} distinct kernels.")
        history = self._history()
        if history:
            head.append(f"Tool cost so far: {self.tool_cost():.2f}. Previous turns:")
            head.extend(history)
        else:
            head.append("No tools called yet.")
        if left == 0:
            head.append("This is your LAST turn: answer now, or the episode ends with no answer.")
        else:
            head.append(f"After this turn you have {left} turn(s) left.")
        head.append("Reply with ONE JSON object: a tool call or your answer.")
        prompt = "\n".join(head)
        self.pending_prompt = prompt
        return prompt

    # -- what the policy does ------------------------------------------------

    def _event(self, name: str, why: str, step: int | None = None, points: float | None = None) -> None:
        value = self.points[name] if points is None else points
        self.events.append(Event(name, value, why, step if step is not None else self.turns_used))

    def act(self, raw: str) -> None:
        if self.done:
            return
        action = parse_action_v2(raw, self.tools.resolve) if self.v2 else parse_action(raw, self.kernels)
        record = StepRecord(n=self.turns_used + 1, prompt=self.pending_prompt, raw=raw, action=action)
        self.steps.append(record)
        if action.kind == "malformed":
            self._event("malformed", action.error, record.n)
            snippet = " ".join(str(raw).split())[:160]
            record.header = "invalid reply"
            record.body = f"({action.error}; cost {-POINTS['malformed']:g}) your reply was: {snippet}"
        elif action.kind == "answer":
            self._score_answer(action.answer or {}, record.n)
            return
        else:
            self._run_tool(action, record)
        if self.turns_used >= self.budget and not self.done:
            self._event("no_answer", f"{self.budget} turns used without an answer", record.n)
            self.done = True
            self.outcome = "no_answer"

    def _run_tool_v2(self, action: Action, record: StepRecord) -> None:
        """``rjenv2.Episode.act``'s dispatch and prices; ``consan`` records no event."""
        tool, kernel, t = action.tool, action.kernel, self.tools
        call: dict[str, Any] = {"tool": tool, **({"kernel": kernel} if kernel else {})}
        if tool == "list_kernels":
            result, name, cost = t.list_kernels(), "cost_list_kernels", COSTS_V2["cost_list_kernels"]
        elif tool == "run_native":
            result = t.native(kernel)
            name, cost = (None, 0.0) if t.native_not_run() else ("cost_run_native", COSTS_V2["cost_run_native"])
        elif tool == "run_sim":
            result = t.sim(kernel, action.seed)
            if kernel == "all":
                name = "cost_run_sim_all"
                cost = round(COSTS_V2["cost_run_sim_all_base"]
                             + COSTS_V2["cost_run_sim_all_per_kernel"] * len(self.kernels), 4)
            else:
                name, cost = "cost_run_sim_kernel", COSTS_V2["cost_run_sim_kernel"]
            if action.seed:
                call["seed"] = action.seed
        elif tool == "waitcheck":
            result, name, cost = t.waitcheck(kernel or None), "cost_waitcheck", COSTS_V2["cost_waitcheck"]
        elif tool == "consan":
            result, name, cost = t.consan(kernel), None, 0.0
        else:
            result, name, cost = t.disasm(kernel, action.start, action.end), "cost_disasm", COSTS_V2["cost_disasm"]
            if action.start is not None:
                call["start"] = hexoff(action.start)
            if action.end is not None:
                call["end"] = hexoff(action.end)
        if name is not None:
            self._event(name, f"{tool} {kernel}".strip(), record.n, points=cost)
        record.cost = -cost
        record.header = json.dumps(call)
        record.body = render_result_v2(tool, kernel, result, action.seed)
        self.last_result = result

    def _run_tool(self, action: Action, record: StepRecord) -> None:
        if self.v2:
            return self._run_tool_v2(action, record)
        tool, kernel = action.tool, action.kernel
        if tool == "waitcheck":
            result = self.tools.waitcheck()
            call = {"tool": "waitcheck"}
        elif tool == "run_native":
            result = self.tools.run_native(kernel)
            call = {"tool": tool, "kernel": kernel}
        elif tool == "run_sim":
            result = self.tools.run_sim(kernel)
            call = {"tool": tool, "kernel": kernel}
        elif tool == "consan":
            result = self.tools.consan(kernel)
            call = {"tool": tool, "kernel": kernel}
        else:
            result = self.tools.disasm(kernel, action.start, action.end)
            call = {"tool": tool, "kernel": kernel}
            if action.start is not None:
                call["start"] = hexoff(action.start)
            if action.end is not None:
                call["end"] = hexoff(action.end)
        name = tool_cost_event(action)
        self._event(name, f"{tool} {kernel}".strip(), record.n)
        record.cost = -POINTS[name]
        record.header = json.dumps(call)
        record.body = render_result(tool, kernel or "all", result)
        self.last_result = result

    def _score_answer(self, ans: dict[str, Any], step: int) -> None:
        self._judge_answer(ans, step)
        names = {e.name for e in self.events}
        if "kernel_correct" in names and not kernel_paid(self.reward, names):
            self.events = [Event(e.name, 0.0, f"{e.why}; {self.reward} pays it only with the class or a location",
                                 e.step) if e.name == "kernel_correct" else e for e in self.events]

    def _judge_answer(self, ans: dict[str, Any], step: int) -> None:
        self.done = True
        self.answer = ans
        s = self.scenario
        cls = ans.get("class")
        if s.clean:
            if cls == "none":
                self._event("clean_correct", "answered none on a clean control", step)
                self.outcome = "clean_correct"
            else:
                self._event("false_alarm", f"claimed a {cls} bug on a clean control", step)
                self.outcome = "false_alarm"
            return
        if cls == "none":
            self._event("missed_bug", f"answered none; the bug is in {s.kernel}", step)
            self.outcome = "missed_bug"
            return
        named = self.tools.resolve(ans.get("kernel")) if self.v2 else ans.get("kernel")
        if named != s.kernel:
            self._event("kernel_wrong", f"answered {ans.get('kernel')!r}", step)
            self.outcome = "kernel_wrong"
            return
        self._event("kernel_correct", f"kernel {s.kernel}", step)
        if cls == s.family:
            self._event("class_correct", f"class {cls}", step)
        off = ans.get("offset")
        self.outcome = "kernel_only"
        if off is None:
            return
        if off in s.offsets:
            self._event("location_exact", f"offset {hexoff(off)}", step)
            self.outcome = "exact"
        elif self._near(off):
            self._event("location_near", f"offset {hexoff(off)} within {NEAR_INSTRUCTIONS} instructions", step)
            self.outcome = "near"

    def _near(self, off: int) -> bool:
        insns = self.scenario.insn_offsets
        if not insns:
            return False
        # Only an instruction of the kernel, as served (mutated) or as built, can be near; the snap
        # below would otherwise credit an offset past either end of the kernel or inside an instruction.
        if off not in insns and off not in {o for o, _ in self.tools.listing(self.scenario.kernel)}:
            return False
        index = {o: i for i, o in enumerate(insns)}
        nearest = min(range(len(insns)), key=lambda j: abs(insns[j] - off))
        # Distances are counted as built, so an accepted offset that only the served listing has is skipped;
        # kept, as the reference environment and every recorded evaluation score it so (README, "Known limitations").
        return any(abs(nearest - index[t]) <= NEAR_INSTRUCTIONS for t in self.scenario.offsets if t in index)

    # -- the result ------------------------------------------------------------

    @property
    def solved(self) -> bool:
        """Full credit: right kernel, class and exact offset, or "none" on a clean control."""
        names = {e.name for e in self.events}
        if self.scenario.clean:
            return "clean_correct" in names
        return {"kernel_correct", "class_correct", "location_exact"} <= names

    def score(self) -> EventScore:
        return EventScore(events=list(self.events), terminal=self.outcome, cells=0)

    def tool_sequence(self) -> str:
        """The episode's actions as one string, for entrenchment counts."""
        parts = []
        for s in self.steps:
            a = s.action
            if a.kind == "tool":
                parts.append(a.tool + ("" if a.tool == "waitcheck" or not a.kernel
                                       else (":all" if a.kernel == "all" else ":k")))
            else:
                parts.append(a.kind)
        return ">".join(parts)

    def summary(self) -> dict[str, Any]:
        return {"scenario_id": self.scenario.scenario_id, "index": self.index, "outcome": self.outcome,
                "steps": self.turns_used, "solved": self.solved, "tool_cost": round(self.tool_cost(), 4),
                "answer": self.answer, "sequence": self.tool_sequence(),
                "total": round(self.score().total, 4)}


# ---------------------------------------------------------------------------
# lockstep rollout, in episode_env's shape
# ---------------------------------------------------------------------------


def pending(episodes: Sequence[Episode]) -> list[tuple[Episode, str]]:
    out = []
    for episode in episodes:
        prompt = episode.observe()
        if prompt is not None:
            out.append((episode, prompt))
    return out


def _write_episode_log(root: Path, episode: Episode) -> None:
    path = root / episode.scenario.scenario_id.replace("/", "_") / f"ep{episode.index:03d}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for s in episode.steps:
            handle.write(json.dumps({"step": s.n, "prompt": s.prompt, "raw": s.raw,
                                     "kind": s.action.kind, "error": s.action.error}) + "\n")
        handle.write(json.dumps({"summary": episode.summary(),
                                 "events": [e.as_dict() for e in episode.events]}) + "\n")


def group_stats(scenario: Scenario, episodes: Sequence[Episode], rewards: Sequence[float],
                adv: Sequence[float], mean: float, sd: float, depth: int) -> dict[str, Any]:
    n = max(len(episodes), 1)
    events: Counter[str] = Counter(e.name for ep in episodes for e in ep.events)
    tools_by_kernel: Counter[str] = Counter(
        f"{s.action.tool}:{s.action.kernel or '-'}" for ep in episodes for s in ep.steps
        if s.action.kind == "tool")
    steps_total = sum(ep.turns_used for ep in episodes)
    step_one = [ep.steps[0].raw for ep in episodes if ep.steps]
    exact = sum(ep.outcome == "exact" for ep in episodes)
    return {
        "scenario_id": scenario.scenario_id,
        "family": scenario.family,
        "op": scenario.op,
        "kernel": scenario.kernel,
        "n": len(episodes),
        "reward_mean": round(mean, 4),
        "reward_sd": round(sd, 4),
        "reward_spread": round(max(rewards) - min(rewards), 4) if rewards else 0.0,
        "advantage_spread": round(max(adv) - min(adv), 4) if adv else 0.0,
        "distinct_completions": len(set(step_one)),
        "steps_total": steps_total,
        "parsed_json": sum(s.action.parsed_json for ep in episodes for s in ep.steps),
        "strict_json": sum(s.action.strict_json for ep in episodes for s in ep.steps),
        "rewards": [round(v, 4) for v in rewards],
        "max_depth": depth,
        "mean_steps": round(steps_total / n, 3),
        "terminals": dict(Counter(ep.outcome for ep in episodes)),
        "events": dict(events),
        # `converged_rate` is the name the trainer prints; here it is the solved rate.
        "converged_rate": round(sum(ep.solved for ep in episodes) / n, 4),
        "solved_rate": round(sum(ep.solved for ep in episodes) / n, 4),
        "exact_rate": round(exact / n, 4),
        "answered_rate": round(sum(ep.answer is not None for ep in episodes) / n, 4),
        "mean_tool_cost": round(sum(ep.tool_cost() for ep in episodes) / n, 4),
        "sequences": dict(Counter(ep.tool_sequence() for ep in episodes)),
        "tool_calls": dict(tools_by_kernel),
        "solved": [int(ep.solved) for ep in episodes],
        "episodes": [ep.summary() for ep in episodes],
    }


def sequence_stats(sequences: Counter[str] | dict[str, int]) -> dict[str, Any]:
    """How concentrated the episodes' action sequences are: the entrenchment check.

    ``top_share`` is the most common sequence's share of episodes and
    ``entropy_norm`` the Shannon entropy over sequences divided by its maximum
    for that many episodes (``log n``), so 0.0 is every episode identical and
    1.0 is every episode different.
    """
    import math

    counts = Counter(sequences)
    n = sum(counts.values())
    if not n:
        return {"episodes": 0, "distinct": 0, "top": None, "top_share": None, "entropy_norm": None}
    top, top_n = counts.most_common(1)[0]
    entropy = max(0.0, -sum((c / n) * math.log(c / n) for c in counts.values()))
    return {"episodes": n, "distinct": len(counts), "top": top, "top_share": round(top_n / n, 4),
            "entropy_norm": round(entropy / math.log(n), 4) if n > 1 else 0.0}


def iteration_metrics(groups: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """One iteration's environment metrics, from its groups (printed and logged)."""
    n = sum(g["n"] for g in groups) or 1
    spreads = [g["advantage_spread"] for g in groups]
    events: Counter[str] = Counter()
    sequences: Counter[str] = Counter()
    first_tools: Counter[str] = Counter()
    for g in groups:
        events.update(g["events"])
        sequences.update(g["sequences"])
        for seq, c in g["sequences"].items():
            first_tools[seq.split(">")[0]] += c
    stats = sequence_stats(sequences)
    return {
        "episodes": n,
        "solved_rate": round(sum(g["solved_rate"] * g["n"] for g in groups) / n, 4),
        "exact_rate": round(sum(g["exact_rate"] * g["n"] for g in groups) / n, 4),
        "answered_rate": round(sum(g["answered_rate"] * g["n"] for g in groups) / n, 4),
        "mean_tool_cost": round(sum(g["mean_tool_cost"] * g["n"] for g in groups) / n, 4),
        "mean_steps": round(sum(g["mean_steps"] * g["n"] for g in groups) / n, 3),
        "malformed": events.get("malformed", 0),
        "no_answer": events.get("no_answer", 0),
        "groups_flat": sum(s <= 1e-6 for s in spreads),
        "advantage_spread_min": round(min(spreads), 4) if spreads else None,
        "advantage_spread_mean": round(sum(spreads) / len(spreads), 4) if spreads else None,
        "sequence_distinct": stats["distinct"],
        "sequence_top_share": stats["top_share"],
        "sequence_entropy_norm": stats["entropy_norm"],
        "first_tool": dict(first_tools.most_common(4)),
    }


def rollout_scenario(
    scenario: Scenario,
    count: int,
    policy: Any,
    generate: Callable[..., list[str]],
    *,
    advantage_fn: Callable[[list[float]], tuple[list[float], float, float]],
    log_root: Path | None = None,
    log_first: int = 0,
    keyed: bool = False,
) -> tuple[dict[str, Any], list[Sample], list[dict[str, Any]]]:
    """Advance ``count`` episodes on one scenario in lockstep, score, flatten.

    ``episode_env.rollout_scenario``'s contract: every live episode is advanced
    by one batched ``generate`` call per step, a generator returning the wrong
    number of replies is an error, and with ``keyed=True`` the generator gets
    ``(episode index, step)`` per prompt so it can seed each reply on its own.
    The advantage is the episode's, repeated on each of its steps.
    """
    budget = getattr(policy, "max_iterations", MAX_TURNS)
    episodes = [Episode(scenario, index=i, budget=budget) for i in range(count)]
    depth = 0
    while True:
        live = pending(episodes)
        if not live:
            break
        depth += 1
        prompts = [prompt for _, prompt in live]
        if keyed:
            texts = generate(prompts, [(episode.index, depth) for episode, _ in live])
        else:
            texts = generate(prompts)
        if len(texts) != len(live):
            raise ValueError(f"generate returned {len(texts)} completion(s) for {len(live)} prompt(s)")
        for (episode, _), text in zip(live, texts, strict=True):
            episode.act(text)

    rewards = [episode.score().total for episode in episodes]
    adv, mean, sd = advantage_fn(rewards)
    samples: list[Sample] = []
    wire: list[dict[str, Any]] = []
    for episode, a, reward in zip(episodes, adv, rewards, strict=True):
        for record in episode.steps:
            samples.append(Sample(scenario_id=scenario.scenario_id, prompt=record.prompt,
                                  completion=record.raw, advantage=a, episode=episode.index,
                                  step=record.n))
            wire.append({"scenario_id": scenario.scenario_id, "episode": episode.index, "step": record.n,
                         "raw": record.raw, "kind": record.action.kind, "tool": record.action.tool,
                         "kernel": record.action.kernel, "error": record.action.error,
                         "reward": reward, "advantage": a, "terminal": episode.outcome,
                         "solved": episode.solved, "episode_steps": episode.turns_used,
                         "tool_cost": round(episode.tool_cost(), 4)})
        if log_root is not None and episode.index < log_first:
            _write_episode_log(log_root, episode)
    return group_stats(scenario, episodes, rewards, adv, mean, sd, depth), samples, wire
