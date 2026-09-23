"""The exact questions the chat graph asks Laya, and the thresholds it reads.

**One definition, because the alternative has already happened once.** Track B
found that the corpus builder under ``src/aorta/laya/corpus/`` had independently
written its own phrasing of the questions the inference path asks. Fine-tuning
on one wording and asking another does not fail loudly -- it answers slightly
worse, for a reason nobody would go looking for. So every question string the
router and the selector ask lives here, and anything that builds a chat corpus
imports these rather than re-typing them.

There is no chat corpus builder yet, and when one is written it will not be able
to import this module: ``tests/cli/test_chat_boundaries.py`` allows
``aorta.chat`` to import ``aorta.*`` and forbids the reverse, so a builder under
``aorta/laya/corpus/`` cannot reach in here. That is a real problem for exactly
the defect above, and the honest options are to move these constants into
``aorta/laya/`` once a builder needs them, or to make the corpus package a
sanctioned importer. It is recorded here rather than solved because both options
edit a package this change does not own.
``tests/chat/test_laya_questions.py`` guards the interim with a test that fails
if the text of either question appears in more than one file in the tree.

**Every question here is a noul, and that is deliberate.** The plan's reason is
the ``head_max_len`` budget split: every option of one question shares one token
budget, so a wide option list leaves few tokens per label. That reason holds on
every install.

The second reason is calibration, and it holds only on some, which is worth
being exact about. Laya 0.3.5 clamps fitted temperatures into [0.5, 5.0] because
the shipped ``choice:11+`` bucket is 0.1006 -- a temperature below 1 sharpens
logits rather than softening them, so that bucket multiplies them roughly
tenfold and publishes a 0.24 top probability as 0.99. A single ``choice`` over
the tool registry lands in that bucket on a full install and not on a bare one:
measured against this tree, ``[chat-cli]`` alone registers nine described tools,
and adding ``[cia]`` with ``allow_cluster_jobs`` brings five more, at which point
a choice would be ``choice:14``. Entry-point plugins only add to that. So the
wide choice is safe today for a subset of users and unsafe for the rest, which
is the worst shape a defect can have.

N independent nouls are ``noul:2`` on every install, whatever is registered.
The cost is nothing, because M questions about one state is one forward pass.
``tests/chat/test_laya_questions.py`` asserts both halves, including the
counterfactual, so neither claim can quietly stop being true.
"""

from __future__ import annotations

from aorta.laya.predictor import Noul

# ── router ─────────────────────────────────────────────────────────────────

#: The router's whole decision, as one yes/no.
#:
#: Phrased so that *true* means the action branch, matching ``ROUTER_PROMPT``'s
#: own two categories rather than inverting them. The glosses carry the two
#: clauses of that prompt that are not restatements of the categories: the
#: "find, search, list, enumerate" rule, and the pasted-source-plus-symptom rule
#: that exists because answering such a turn means compiling the code and
#: running it under a sanitizer, which no amount of retrieved context
#: substitutes for.
#:
#: What is deliberately *not* carried over is the prompt's closing "If in doubt,
#: classify as action". That sentence is a calibration instruction written in
#: English, and it is the thing a threshold replaces: see
#: :data:`DEFAULT_ROUTER_THRESHOLD`.
ROUTER_QUESTION = Noul(
    question=(
        "Does answering this message require using tools -- searching the codebase, "
        "reading files, running a command, or diagnosing a workload on the cluster -- "
        "rather than answering it from retrieved context alone?"
    ),
    when_true=(
        "it asks to find, search, list or enumerate things, asks how to run "
        "something, or carries pasted source, assembly, a log or a stack trace "
        "together with a symptom such as wrong results, a crash or a hang"
    ),
    when_false=(
        "it is a specific question about the code or the system that retrieved "
        "context can answer on its own"
    ),
)

#: p(action) at or above which the router takes the action branch.
#:
#: Below the 0.5 midpoint on purpose, and this is where ``ROUTER_PROMPT``'s
#: closing "If in doubt, classify as action" ends up. The two mistakes do not
#: cost the same: a question sent down the action branch costs a plan call and a
#: tool round and still answers, while an action sent down the question branch
#: reaches a node with no tool access at all and dead-ends the turn -- the same
#: asymmetry ``_ROUTER_EMPTY_ROUTE`` and ``_ROUTER_UNPARSED_ROUTE`` were split
#: apart to handle on the LLM path.
#:
#: **This number is not a calibration figure and must not be reported as one.**
#: The Phase 1 measurement that fits one temperature per (question type, option
#: count) has not been run for either chat decision, and no chat corpus exists.
#: It is a policy choice about which error to prefer, made in the absence of a
#: fit rather than derived from one, and it is the number to re-derive first when
#: there is one -- see ``docs/laya-packaging.md`` on a probability being a
#: function of the checkpoint *and* the fit applied to it.
DEFAULT_ROUTER_THRESHOLD = 0.4


# ── selector ───────────────────────────────────────────────────────────────

#: One question per registered tool. ``{name}`` and ``{description}`` are filled
#: from the live registry, the same source ``catalogue()`` renders for the LLM
#: prompt, so a tool contributed through the ``aorta.chat_tools`` entry point is
#: rankable here too without being named anywhere.
#:
#: The wording is ``_SELECTOR_PROMPT``'s own criterion -- "Judge only on this:
#: would the evidence that tool returns actually answer this problem?" -- rather
#: than a fresh phrasing of it, so that the two paths are asking the same thing
#: and a comparison between them is a comparison of models rather than of
#: prompts.
SELECTOR_QUESTION_TEMPLATE = (
    "Would the evidence returned by the {name} tool actually answer this problem? "
    "The tool: {description}"
)

SELECTOR_WHEN_TRUE = "the evidence this tool returns would identify the cause"
SELECTOR_WHEN_FALSE = (
    "this tool returns evidence about something else, so running it costs a "
    "cluster job and points the engineer down the wrong path"
)

#: p(yes) a tool needs before it is recommended at all.
#:
#: It exists to keep a property the prompt states in words and a pure ranking
#: would lose: "If nothing fits, return fewer tools, or none." Ranking alone
#: always produces the top three, so a question no tool suits would still come
#: back with three recommendations, and the act node would be steered by a list
#: whose best entry the model thought was wrong.
#:
#: **Not a calibration figure either**, for the same reason as
#: :data:`DEFAULT_ROUTER_THRESHOLD`. 0.5 is the neutral reading of a noul and
#: nothing has been fitted; unlike the router's, its asymmetry genuinely is
#: neutral, because the shortlist is advisory in both directions -- a tool left
#: off is still bound and still callable.
DEFAULT_SELECTOR_THRESHOLD = 0.5


def tool_question(name: str, description: str) -> Noul:
    """The noul asked about one tool. One definition, per this module's docstring."""
    return Noul(
        question=SELECTOR_QUESTION_TEMPLATE.format(name=name, description=description),
        when_true=SELECTOR_WHEN_TRUE,
        when_false=SELECTOR_WHEN_FALSE,
    )


__all__ = [
    "DEFAULT_ROUTER_THRESHOLD",
    "DEFAULT_SELECTOR_THRESHOLD",
    "ROUTER_QUESTION",
    "SELECTOR_QUESTION_TEMPLATE",
    "SELECTOR_WHEN_FALSE",
    "SELECTOR_WHEN_TRUE",
    "tool_question",
]
