"""The egress redaction gate (Decision 16).

Three properties matter here and none of them is about the scrubbers, which
belong to ``aorta.probe.redaction`` and are tested with the bundle:

1. It is **on by default**, and off only when asked.
2. It rewrites the copy that leaves the machine and *not* the conversation
   state, so the user still sees their own paths echoed back.
3. The notice fires **once**, on stderr, and names both what went and how to
   stop it. A gate nobody is told about is a gate that gets blamed for a wrong
   answer.
"""

from __future__ import annotations

import ast
import io
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from aorta.chat import redaction
from aorta.chat.config import configure, reset_settings

CUSTOMER_TEXT = (
    "the run under /home/cust7/models/llama-70b failed on host 10.42.7.9 "
    "and the replica at [2001:db8::1] hung"
)


@pytest.fixture(autouse=True)
def _clean_session():
    """Every test starts with the notice unshown and the default settings."""
    redaction.reset_session_notice()
    reset_settings()
    yield
    redaction.reset_session_notice()
    reset_settings()


class TestDefaults:
    def test_redaction_is_on_without_configuration(self):
        """The default has to be the safe one; nobody opts in to not leaking."""
        text, summary = redaction.redact_text(CUSTOMER_TEXT)
        assert "/home/cust7" not in text
        assert "10.42.7.9" not in text
        assert summary.total == 3

    def test_no_redact_passes_text_through_byte_for_byte(self):
        configure(redact=False)
        text, summary = redaction.redact_text(CUSTOMER_TEXT)
        assert text == CUSTOMER_TEXT
        assert not summary

    def test_placeholders_replace_each_kind(self):
        text, _ = redaction.redact_text(CUSTOMER_TEXT)
        assert "<PATH:0>" in text
        assert "<IPV4:0>" in text
        assert "<IPV6:0>" in text


class TestMessageRewriting:
    def test_conversation_state_is_not_mutated(self):
        """The user's own history must keep their paths; only the wire copy loses them."""
        original = HumanMessage(content=CUSTOMER_TEXT)
        messages = [original]
        out, _ = redaction.redact_messages(messages)
        assert original.content == CUSTOMER_TEXT
        assert out[0] is not original
        assert "/home/cust7" not in out[0].content

    def test_unchanged_messages_keep_their_identity(self):
        """A no-op scrub must not clone.

        LangChain pairs a ToolMessage with the AIMessage that requested it, so
        needlessly rebuilding untouched messages risks that bookkeeping for no
        gain.
        """
        clean = SystemMessage(content="You are an assistant.")
        out, summary = redaction.redact_messages([clean])
        assert out[0] is clean
        assert not summary

    def test_non_string_content_is_passed_through_untouched(self):
        """Multimodal / tool-call blocks are not modelled, so they are not rewritten.

        Coercing one to text would corrupt the request rather than protect
        anything.
        """
        structured = AIMessage(content=[{"type": "text", "text": "/home/cust7/x"}])
        out, summary = redaction.redact_messages([structured])
        assert out[0] is structured
        assert not summary

    def test_counts_accumulate_across_the_whole_message_list(self):
        messages = [
            HumanMessage(content="/home/a/b failed"),
            ToolMessage(content="also /home/c/d and 10.0.0.1", tool_call_id="1"),
        ]
        _, summary = redaction.redact_messages(messages)
        assert summary.paths == 2
        assert summary.ipv4 == 1

    def test_no_redact_returns_the_same_list_object(self):
        configure(redact=False)
        messages = [HumanMessage(content=CUSTOMER_TEXT)]
        out, _ = redaction.redact_messages(messages)
        assert out is messages


class TestNoticeIsSessionLocal:
    """The notice state must not be shared by every session in the process.

    Under ``aorta chat ui`` one process serves many browser sessions. With a
    module-level flag, the first session to trigger a redaction consumed the
    disclosure for everybody, and every later user was silently redacted -- the
    one thing Decision 16 says must not happen. The CLI is one session per
    process, so it keeps using the process-wide state and binds nothing.
    """

    def test_two_bound_sessions_are_each_told(self):
        _, summary = redaction.redact_text(CUSTOMER_TEXT)
        first, second = redaction.NoticeState(), redaction.NoticeState()
        one, two = io.StringIO(), io.StringIO()
        with redaction.use_notice_state(first):
            assert redaction.emit_notice_once(summary, stream=one) is True
        with redaction.use_notice_state(second):
            assert redaction.emit_notice_once(summary, stream=two) is True
        assert one.getvalue().count("aorta chat: redacted") == 1
        assert two.getvalue().count("aorta chat: redacted") == 1

    async def test_overlapping_sessions_are_each_told(self):
        """Two sessions live at the same time, which is how Chainlit runs them.

        The sequential case above already rejects a bare module-level flag. What
        it cannot see is a binding that is itself process-wide: a module-level
        "current session" pointer, saved and restored around each block, answers
        every sequential arrangement identically to a context variable and hands
        one session the other's state the moment the two overlap. So each
        session is held inside its own binding until the other has emitted *and*
        looked -- the second rendezvous is what makes the lookup happen while a
        second binding is live, which is the only time the two differ.

        Both rendezvous are released in ``finally`` so a failing assertion fails
        this test rather than parking its peer for the rest of the run, and both
        waits are bounded so a failure *before* the counter -- which never
        reaches that ``finally`` -- fails it too instead of hanging the job.
        """
        import asyncio

        _, summary = redaction.redact_text(CUSTOMER_TEXT)
        both_emitted = asyncio.Event()
        both_looked = asyncio.Event()
        emitted = 0
        looked = 0
        timeout = 5.0

        async def meet(event: asyncio.Event, name: str) -> None:
            try:
                await asyncio.wait_for(event.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                raise AssertionError(
                    f"peer session never reached {name} within {timeout}s"
                ) from None

        async def session(stream: io.StringIO) -> None:
            nonlocal emitted, looked
            with redaction.use_notice_state(redaction.NoticeState()) as state:
                try:
                    assert redaction.emit_notice_once(summary, stream=stream) is True
                finally:
                    emitted += 1
                    if emitted == 2:
                        both_emitted.set()
                await meet(both_emitted, "both_emitted")
                try:
                    # This session's own state, and its own undrained notice,
                    # while the other session's binding is still in force.
                    assert redaction.current_notice_state() is state
                    assert redaction.take_pending_notice(state) is not None
                finally:
                    looked += 1
                    if looked == 2:
                        both_looked.set()
                await meet(both_looked, "both_looked")

        one, two = io.StringIO(), io.StringIO()
        await asyncio.gather(session(one), session(two))

        assert one.getvalue().count("aorta chat: redacted") == 1
        assert two.getvalue().count("aorta chat: redacted") == 1
        assert redaction.current_notice_state().emitted is False

    def test_one_session_is_still_told_only_once(self):
        _, summary = redaction.redact_text(CUSTOMER_TEXT)
        state = redaction.NoticeState()
        stream = io.StringIO()
        for _ in range(3):
            with redaction.use_notice_state(state):
                redaction.emit_notice_once(summary, stream=stream)
        assert stream.getvalue().count("aorta chat: redacted") == 1

    def test_a_bound_session_does_not_consume_the_process_notice(self):
        """A UI session must leave the CLI fallback state untouched."""
        _, summary = redaction.redact_text(CUSTOMER_TEXT)
        with redaction.use_notice_state(redaction.NoticeState()):
            redaction.emit_notice_once(summary, stream=io.StringIO())
        assert redaction.current_notice_state().emitted is False

    def test_the_binding_is_undone_on_the_way_out(self):
        state = redaction.NoticeState()
        with redaction.use_notice_state(state):
            assert redaction.current_notice_state() is state
        assert redaction.current_notice_state() is not state

    def test_state_survives_the_tasks_one_turn_spans(self):
        """The reason this is a mutable object and not a ``ContextVar[bool]``.

        A task copies the context at creation and its writes do not propagate
        back, so a bare flag flipped inside a graph node would be forgotten by
        the next turn and the notice would fire every time. Shared state does
        not have that problem.
        """
        import asyncio

        _, summary = redaction.redact_text(CUSTOMER_TEXT)
        state = redaction.NoticeState()
        stream = io.StringIO()

        async def turn() -> None:
            # A nested task is what LangGraph runs nodes in.
            await asyncio.create_task(
                asyncio.to_thread(redaction.emit_notice_once, summary, stream)
            )

        async def session() -> None:
            with redaction.use_notice_state(state):
                await turn()
                await turn()

        asyncio.run(session())
        assert stream.getvalue().count("aorta chat: redacted") == 1


class TestNoticeDelivery:
    """A front door that is not a terminal has to be able to render it itself."""

    def test_the_line_is_parked_for_the_caller_to_show(self):
        _, summary = redaction.redact_text(CUSTOMER_TEXT)
        state = redaction.NoticeState()
        with redaction.use_notice_state(state):
            redaction.emit_notice_once(summary, stream=io.StringIO())
        assert state.pending is not None
        assert "aorta chat: redacted" in state.pending

    def test_draining_it_yields_it_exactly_once(self):
        _, summary = redaction.redact_text(CUSTOMER_TEXT)
        state = redaction.NoticeState()
        with redaction.use_notice_state(state):
            redaction.emit_notice_once(summary, stream=io.StringIO())
        assert redaction.take_pending_notice(state) is not None
        assert redaction.take_pending_notice(state) is None

    def test_nothing_is_parked_when_nothing_was_redacted(self):
        state = redaction.NoticeState()
        _, summary = redaction.redact_text("what does the router node do?")
        with redaction.use_notice_state(state):
            redaction.emit_notice_once(summary, stream=io.StringIO())
        assert redaction.take_pending_notice(state) is None


class TestNotice:
    def test_notice_names_what_went_and_how_to_disable_it(self):
        _, summary = redaction.redact_text(CUSTOMER_TEXT)
        line = redaction.notice_line(summary)
        assert "filesystem path" in line
        assert "IPv4" in line
        assert "--no-redact" in line
        assert "redact = false" in line

    def test_notice_fires_once_per_session(self):
        stream = io.StringIO()
        _, summary = redaction.redact_text(CUSTOMER_TEXT)
        assert redaction.emit_notice_once(summary, stream=stream) is True
        assert redaction.emit_notice_once(summary, stream=stream) is False
        assert stream.getvalue().count("aorta chat: redacted") == 1

    def test_nothing_is_announced_when_nothing_was_redacted(self):
        """A session of path-free prompts must not be told about a redaction."""
        stream = io.StringIO()
        _, summary = redaction.redact_text("what does the router node do?")
        assert redaction.emit_notice_once(summary, stream=stream) is False
        assert stream.getvalue() == ""

    def test_redact_for_send_writes_the_notice_to_stderr(self, capfd):
        """stderr, so --json and --plain stay machine-parseable on stdout.

        ``capfd`` rather than ``capsys``: the notice deliberately targets
        ``sys.__stderr__``, because quiet mode -- the default -- repoints
        ``sys.stderr`` at ``os.devnull``. Only fd-level capture sees it, which
        is also the proof that it survives that repointing.
        """
        redaction.redact_for_send([HumanMessage(content=CUSTOMER_TEXT)])
        captured = capfd.readouterr()
        assert captured.out == ""
        assert "aorta chat: redacted" in captured.err

    def test_the_notice_survives_quiet_mode_repointing_stderr(self, capfd, monkeypatch):
        """The regression this guards is Decision 16's whole point.

        ``aorta chat`` without ``-v`` sends stderr to /dev/null before the first
        query. A notice written to ``sys.stderr`` would be silently discarded
        exactly when the user most needs it.
        """
        import os

        with open(os.devnull, "w") as devnull:
            monkeypatch.setattr(sys_module(), "stderr", devnull)
            redaction.redact_for_send([HumanMessage(content=CUSTOMER_TEXT)])
        assert "aorta chat: redacted" in capfd.readouterr().err


class TestSummaryWording:
    @pytest.mark.parametrize(
        ("summary", "expected"),
        [
            (redaction.RedactionSummary(), "nothing"),
            (redaction.RedactionSummary(paths=1), "1 filesystem path"),
            (redaction.RedactionSummary(paths=2), "2 filesystem paths"),
            (redaction.RedactionSummary(ipv4=1), "1 IPv4 address"),
            (redaction.RedactionSummary(ipv4=3), "3 IPv4 addresses"),
            (
                redaction.RedactionSummary(paths=1, ipv4=1),
                "1 filesystem path and 1 IPv4 address",
            ),
            (
                redaction.RedactionSummary(paths=1, ipv4=1, ipv6=2),
                "1 filesystem path, 1 IPv4 address and 2 IPv6 addresses",
            ),
        ],
    )
    def test_describe_reads_as_prose(self, summary, expected):
        assert summary.describe() == expected


#: Receivers in ``graph/nodes.py`` that are not chat models, and are therefore
#: outside what ``_send`` guards. ``_send`` is the *chat-model* gate, and that
#: is the whole of what this exemption means.
#:
#: It emphatically does not mean nothing leaves the machine. A tool runs local
#: code, but a retriever only queries the local sqlite index under the default
#: ``embedding_provider = "local"``; set it to ``"remote"`` and
#: ``retriever.ainvoke(last_human)`` sends the raw user query to the embeddings
#: API, unredacted -- ``docs/chat/redaction.md`` documents that path and says
#: plainly that it does not go through the chat-message redactor. So the reason
#: these two are exempt is that they are not the thing ``_send`` gates, not
#: that they are egress-free.
#:
#: Both are awaited rather than called, so neither blocks the event loop out
#: from under a concurrent Chainlit session (issue #444). Anything not named
#: here still has to go through ``_send``.
_NON_MODEL_AINVOKE_RECEIVERS = frozenset({"retriever", "tool_fn"})

#: Every way a LangChain model can be driven asynchronously, all of which are
#: egress and none of which may skip ``_send``. Only ``ainvoke`` is used today,
#: so the extra names guard the future rather than the present -- but a guard
#: that only knows the verb currently in use stops being a guard on the first
#: day someone reaches for a different one, and streaming is the obvious
#: candidate.
_MODEL_INVOCATION_VERBS = frozenset(
    {
        "ainvoke",
        "abatch",
        "abatch_as_completed",
        "agenerate",
        "agenerate_prompt",
        "astream",
        "astream_events",
        "astream_log",
        "atransform",
    }
)

#: Async public methods on ``BaseChatModel`` that do *not* drive a request, so
#: their absence from :data:`_MODEL_INVOCATION_VERBS` is correct rather than a
#: gap. ``as_tool``/``assign`` build a runnable, ``asdict`` serialises. They are
#: listed because the completeness test below works by subtraction: naming what
#: is deliberately not egress is what lets it treat everything else as egress.
_NON_INVOKING_ASYNC_ATTRS = frozenset({"as_tool", "asdict", "assign"})

#: How each AST node that can bind a name exposes the expression it binds
#: *from*. The guard below walks bindings rather than assignments because the
#: shapes are not interchangeable: ``retriever: BaseChatModel = _get_llm()`` is
#: an ``ast.AnnAssign``, and a check that only visited ``ast.Assign`` waved it
#: straight through while the allowlist still exempted ``retriever.ainvoke(...)``
#: from the chokepoint test above.
_BINDING_SOURCE_ATTR = {
    ast.Assign: "value",
    ast.AnnAssign: "value",
    ast.AugAssign: "value",
    ast.NamedExpr: "value",
    ast.For: "iter",
    ast.AsyncFor: "iter",
    ast.comprehension: "iter",
    ast.withitem: "context_expr",
}


def _binding_source(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> str | None:
    """The source text the binding at *node* takes its value from.

    ``""`` when the form binds no value at all (a bare annotation), and ``None``
    when it is a shape :data:`_BINDING_SOURCE_ATTR` cannot read -- a function
    parameter, say, whose value comes from a caller this file cannot see. The
    caller fails on ``None`` rather than passing, because a guard that goes
    quiet on an unfamiliar shape is a guard that can be walked around by
    reaching for one.
    """
    current = node
    while True:
        parent = parents.get(current)
        if parent is None:
            return None
        attr = _BINDING_SOURCE_ATTR.get(type(parent))
        if attr is not None:
            value = getattr(parent, attr, None)
            return "" if value is None else ast.unparse(value)
        if isinstance(
            parent,
            (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            return None
        current = parent


#: ``(node type, field)`` pairs that carry an identifier which *provably cannot*
#: be a binding, and are therefore the only string fields the scan below skips.
#:
#: This list is the inverse of the one that used to be here, and the inversion
#: is the point. Enumerating the shapes that *do* bind sprang a leak three times
#: running -- ``AnnAssign``, then ``import x as y``, then ``match`` captures and
#: ``except ... as name`` -- because that design's default for a shape it had
#: not been taught was *allow*, and the next way past a guard is by definition
#: one its author had not thought of. So the default is now the other way: any
#: string field holding an allowlisted receiver name counts as a binding this
#: guard cannot read, and a new binding syntax fails it instead of slipping
#: through. ``def``/``async def``/``class`` names, which the enumerated version
#: missed, need no entry to be caught -- that is what changed.
#:
#: Each exemption is a *read* of a name owned elsewhere, not a binding of one
#: here: ``obj.retriever`` reads an attribute off an object, a string literal is
#: data, ``f(retriever=x)`` names a parameter in the callee's signature, and
#: ``from retriever import x`` names a module rather than binding ``retriever``.
#: Adding to this list means asserting a field cannot bind, which is a much
#: harder thing to be wrong about by omission.
_NON_BINDING_IDENTIFIER_FIELDS = {
    (ast.Attribute, "attr"),
    (ast.Constant, "value"),
    (ast.keyword, "arg"),
    (ast.ImportFrom, "module"),
}


def _module_bound_names(source: str) -> set[str]:
    """Every name *source* binds, in any scope, according to CPython itself.

    The completeness oracle for the scan below, and the second half of failing
    closed. ``symtable`` is the compiler's own binding analysis, so it is closed
    by construction: it already knows every binding form the language has, and
    it will know the next one without this file being edited. That is exactly
    the property an enumerated table could not have.

    It is used only to answer "is this name bound at all?", because that is the
    question it answers without any scope-naming guesswork -- comprehensions are
    inlined on 3.12 and named on older versions, lambdas and classes have their
    own conventions, and a guard that mismatched those would cry wolf on
    ordinary code. Cross-checked against the AST scan, which is what supplies
    line numbers and the bound expressions that can clear a binding.
    """
    # Imported here rather than at module scope: this file's import block is
    # shared with another PR in the same series, and a local import keeps the
    # two sets of additions textually apart.
    import symtable

    bound: set[str] = set()

    def visit(table) -> None:
        for symbol in table.get_symbols():
            if symbol.is_assigned() or symbol.is_imported() or symbol.is_parameter():
                bound.add(symbol.get_name())
        for child in table.get_children():
            visit(child)

    visit(symtable.symtable(source, "<guard>", "exec"))
    return bound


def _alias_bound_name(node: ast.alias) -> str:
    """The name an ``import`` statement actually binds.

    ``import a.b as c`` binds ``c``; ``import a.b`` binds ``a``, not ``a.b``,
    even though ``alias.name`` holds the whole dotted path. Reading the field
    literally meant ``import retriever.client`` recorded ``retriever.client``,
    which matches no allowlisted receiver -- so the import went unnoticed while
    ``retriever.ainvoke(...)`` stayed exempt from the chokepoint test.
    """
    return node.asname or node.name.split(".", 1)[0]

#: The only bindings ``graph/nodes.py`` may give the allowlisted receiver names,
#: as ``ast.unparse`` renders them. Pinning the expressions rather than only
#: rejecting ``_get_llm`` is what closes the escapes the walk cannot see on its
#: own -- an alias (``llm = _get_llm()`` then ``retriever = llm``) reads as
#: innocent one step at a time, and a two-line detour is not a technique anyone
#: has to be clever to find. Editing this map is meant to be the deliberate act.
_PERMITTED_RECEIVER_BINDINGS = {
    "retriever": {"get_retriever()"},
    "tool_fn": {"TOOL_REGISTRY.get(name)"},
}


def _receiver_bindings(source: str) -> list[tuple[str, str | None, int]]:
    """Every place *source* binds an allowlisted receiver name, and what from.

    Yields ``(name, bound_expression_source, lineno)``, with ``None`` for a form
    :func:`_binding_source` cannot read.

    Two rules, and no list of binding shapes to keep up to date:

    * An ``ast.Name`` in ``Store``/``Del`` context is a binding, and
      :func:`_binding_source` tries to read what it is bound to. This is the
      only case that can be *cleared*.
    * Any other node carrying a string field whose value is an allowlisted
      receiver name is a binding this guard cannot read, unless the field is in
      :data:`_NON_BINDING_IDENTIFIER_FIELDS`. That covers ``def``, ``async
      def``, ``class``, ``import``, parameters, ``match`` captures, ``except
      ... as`` and anything the language grows later, without naming any of
      them.

    It does *not* resolve values through intermediate variables -- that is what
    :data:`_PERMITTED_RECEIVER_BINDINGS` is for.
    """
    tree = ast.parse(source)
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    bindings = []

    def record(name: str, assigned: str | None, node: ast.AST) -> None:
        if name in _NON_MODEL_AINVOKE_RECEIVERS:
            bindings.append((name, assigned, getattr(node, "lineno", 0)))

    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, (ast.Store, ast.Del)):
                record(node.id, _binding_source(node, parents), node)
            continue
        if isinstance(node, ast.alias):
            # The one node whose field semantics are conditional rather than
            # positional: which of `name`/`asname` binds depends on the other.
            # Handled here so the generic rule below cannot report `retriever`
            # for `import retriever.client as rc`, which binds only `rc`.
            record(_alias_bound_name(node), None, node)
            continue
        for field in node._fields:
            value = getattr(node, field, None)
            for item in value if isinstance(value, list) else [value]:
                if not isinstance(item, str) or not item.isidentifier():
                    continue
                if (type(node), field) in _NON_BINDING_IDENTIFIER_FIELDS:
                    continue
                # Nothing on the node bears an expression this could be cleared
                # against: a pattern capture takes its value from a subject
                # several nodes away, a handler takes whichever exception was
                # raised, a parameter takes whatever a caller passed, and a
                # `def` takes whatever its decorators return.
                record(item, None, node)
    return bindings


def _unaccounted_bound_names(source: str) -> list[str]:
    """Allowlisted receivers CPython says *source* binds and the scan missed.

    The cross-check that makes the scan's completeness testable instead of
    assumed. If ``symtable`` reports a binding of an allowlisted name and the
    AST walk found none, the walk has a blind spot -- and the honest thing for a
    guard to do about a blind spot is fail, since the whole point of the
    allowlist is a claim that these names never hold a chat model.
    """
    found = {name for name, _assigned, _lineno in _receiver_bindings(source)}
    return sorted(
        (_module_bound_names(source) & set(_NON_MODEL_AINVOKE_RECEIVERS)) - found
    )


#: The function that hands out a chat model. Smuggling one past ``_send`` means
#: getting its return value onto an allowlisted receiver name, so this is the
#: root the scan below chases.
_MODEL_PRODUCER = "_get_llm"


def _model_producer_aliases(source: str) -> frozenset[str]:
    """Names in *source* that are ``_get_llm`` under another name.

    A guard that matched the callee spelling at the binding site read
    ``retriever = _get_llm()`` correctly and ``get_retriever = _get_llm`` /
    ``retriever = get_retriever()`` not at all -- the model still lands on an
    allowlisted receiver, one line later, with the gate none the wiser.

    So the root is chased to a fixpoint: any name bound to a *bare* reference
    to a known producer (``Name`` or ``Attribute``, not a call) is itself a
    producer, and the pass repeats until nothing new appears. Two hops and
    ``nodes._get_llm`` are covered by construction rather than by being listed.

    This is deliberately not general dataflow. A model smuggled through a
    container, a class attribute or a function's return value would not be
    found, and cannot be by a scan of this kind. What makes that acceptable is
    that this is the *second* line of defence: the runtime chokepoint test in
    :class:`TestGraphChokepoint` asserts every model call actually observed
    goes through ``_send``, and this scan exists to stop a *future* edit
    quietly re-earning an allowlist exemption it no longer deserves. The
    honest claim is "harder to do by accident", not "impossible".
    """
    aliases = {_MODEL_PRODUCER}
    tree = ast.parse(source)
    while True:
        grown = False
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
                value = node.value
                if value is None or not isinstance(value, (ast.Name, ast.Attribute)):
                    # A *call* produces a model instance, not another alias for
                    # the producer, so it is the binding scan's business.
                    continue
                if ast.unparse(value).split(".")[-1] not in aliases:
                    continue
                targets = (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
                for target in targets:
                    for sub in ast.walk(target):
                        if isinstance(sub, ast.Name) and sub.id not in aliases:
                            aliases.add(sub.id)
                            grown = True
        if not grown:
            return frozenset(aliases)


def _calls_a_model_producer(assigned: str, aliases: frozenset[str]) -> bool:
    """Whether the bound expression *assigned* reaches a model producer.

    Substring rather than a parse of the callee, and over-reporting on purpose:
    a tuple unpack hides which element lands on which name, and for a guard the
    safe direction is to complain about a binding it cannot fully attribute.
    """
    return any(alias in assigned for alias in aliases)


def _smuggled_model_bindings(source: str) -> list[str]:
    """Complaints about allowlisted receiver names in *source* bound to a model.

    Reports a binding it cannot read as well as one it can see is wrong, so the
    two failures the guard has to survive -- a new smuggling route and a new
    syntax for an old one -- both come out as a message rather than a pass.
    """
    complaints = []
    aliases = _model_producer_aliases(source)
    for name, assigned, lineno in _receiver_bindings(source):
        if assigned is None:
            complaints.append(
                f"line {lineno}: '{name}' is allowlisted out of the _send gate "
                "and is bound by a form this guard cannot read, so it cannot "
                "show that no chat model reaches it"
            )
        elif _calls_a_model_producer(assigned, aliases):
            complaints.append(
                f"line {lineno}: '{name}' is allowlisted out of the _send gate "
                f"but is assigned a chat model: {assigned}"
            )
    for name in _unaccounted_bound_names(source):
        complaints.append(
            f"'{name}' is allowlisted out of the _send gate and is bound "
            "somewhere this guard's AST scan does not see, per symtable, so it "
            "cannot show that no chat model reaches it"
        )
    return complaints


class TestGraphChokepoint:
    async def test_every_node_send_goes_through_the_gate(self):
        """``_send`` is the single seam, so a node added later cannot bypass it."""
        from unittest.mock import AsyncMock, MagicMock

        from aorta.chat.graph import nodes

        llm = MagicMock()
        llm.ainvoke = AsyncMock(return_value=AIMessage(content="ok"))
        await nodes._send(llm, [HumanMessage(content=CUSTOMER_TEXT)])

        sent = llm.ainvoke.await_args.args[0]
        assert "/home/cust7" not in sent[0].content

    def test_no_node_calls_a_model_directly(self):
        """The gate is only worth anything if it is the only way out.

        Asserted against the source rather than by mocking, because the failure
        being guarded against is a *new* call site, which no existing test would
        exercise.

        Every async invocation verb, not just ``ainvoke``: nothing in
        ``src/aorta/chat/`` streams today, so a bare ``ainvoke`` check passes,
        but streaming is the natural next thing a UI wants and ``plan_node`` and
        ``answer_node`` already build their model without ``streaming=False``.
        An ``astream`` added later would bypass ``_send`` *and* the guard whose
        job is to notice that, which is the one combination this test cannot
        afford.
        """
        source = Path(nodes_path()).read_text(encoding="utf-8")
        offenders = []
        for node in ast.walk(ast.parse(source)):
            if (
                not isinstance(node, ast.Attribute)
                or node.attr not in _MODEL_INVOCATION_VERBS
            ):
                continue
            receiver = node.value
            if (
                isinstance(receiver, ast.Name)
                and receiver.id in _NON_MODEL_AINVOKE_RECEIVERS
            ):
                continue
            # `_send` itself is the sanctioned caller.
            offenders.append((node.lineno, node.attr))
        # One permitted occurrence: the call inside `_send`.
        assert len(offenders) == 1, (
            f"graph/nodes.py invokes a model at {offenders}; every outbound "
            "call must go through _send so the redaction gate applies."
        )

    def test_the_allowlist_cannot_be_used_to_smuggle_a_model_out(self):
        """The narrowing above is only safe while those names hold no model.

        The gate is about messages leaving for a provider, so awaiting a
        retriever or a tool is not egress -- both stay on this machine, and both
        are awaited to keep the event loop free (issue #444). But an allowlist
        by receiver name is only as good as what those names are bound to, so
        this pins that neither is ever assigned from ``_get_llm``.
        """
        source = Path(nodes_path()).read_text(encoding="utf-8")
        assert _smuggled_model_bindings(source) == []

    def test_the_allowlisted_names_are_bound_only_to_what_they_claim(self):
        """The stronger half: pin the bindings, not just reject ``_get_llm``.

        The check above answers "is a model assigned here?", and a two-line
        detour answers no to it -- ``llm = _get_llm()`` then ``retriever = llm``
        mentions ``_get_llm`` on neither line that matters. An allowlist of
        *bindings* has no such gap, and it also makes the check above
        non-vacuous: a rename that left ``_NON_MODEL_AINVOKE_RECEIVERS`` behind
        would otherwise keep reporting green against nothing.
        """
        source = Path(nodes_path()).read_text(encoding="utf-8")
        found: dict[str, set[str]] = {}
        for name, assigned, _lineno in _receiver_bindings(source):
            found.setdefault(name, set()).add(assigned)
        assert found == {
            name: set(expected)
            for name, expected in _PERMITTED_RECEIVER_BINDINGS.items()
        }, (
            "graph/nodes.py binds a receiver that is allowlisted out of the "
            "_send gate somewhere new. Confirm it still holds no chat model, "
            "then update _PERMITTED_RECEIVER_BINDINGS deliberately."
        )

    def test_an_alias_of_a_model_is_caught_by_the_pinned_bindings(self):
        """The escape the ``_get_llm`` walk cannot see, named as its own case."""
        aliased = "llm = _get_llm()\nretriever = llm\n"
        assert _smuggled_model_bindings(aliased) == []
        found = {name for name, _assigned, _lineno in _receiver_bindings(aliased)}
        assert found == {"retriever"}
        assert ("retriever", "llm", 2) in _receiver_bindings(aliased)

    @pytest.mark.parametrize(
        "binding",
        [
            "retriever = _get_llm()",
            # The one the previous version of this guard let through: an
            # annotation makes it an `ast.AnnAssign`, which `ast.Assign` alone
            # never visited, while the allowlist above still exempted
            # `retriever.ainvoke(...)` from the chokepoint test.
            "retriever: BaseChatModel = _get_llm()",
            "retriever, other = _get_llm(), 1",
            "if (tool_fn := _get_llm()):\n    pass",
            "with _get_llm() as retriever:\n    pass",
            "for tool_fn in [_get_llm()]:\n    pass",
            "candidates = [tool_fn for tool_fn in [_get_llm()]]",
        ],
    )
    def test_every_binding_form_is_seen_by_the_guard(self, binding):
        """One syntax for "bind this name" is not the same as all of them."""
        assert _smuggled_model_bindings(binding), binding

    @pytest.mark.parametrize(
        "binding",
        [
            # Matching the callee's spelling at the binding site read the direct
            # form and none of these: the producer arrives under a different
            # name, and the model lands on the allowlisted receiver one line
            # later with the gate none the wiser.
            'get_retriever = _get_llm\nretriever = get_retriever()',
            "f = nodes._get_llm\nretriever = f()",
            "a = _get_llm\nb = a\nretriever = b()",
            # Order must not matter: the closure is computed over the whole
            # module before any binding is judged.
            "retriever = g()\ng = _get_llm",
            "h: Any = _get_llm\nretriever = h()",
            "(w := _get_llm)\nretriever = w()",
        ],
    )
    def test_renaming_the_producer_does_not_hide_it(self, binding):
        """An alias of ``_get_llm`` is ``_get_llm`` for the guard's purposes."""
        assert _smuggled_model_bindings(binding), binding

    @pytest.mark.parametrize(
        "binding",
        [
            # The cost of the alias closure is false positives, so these pin
            # the other direction: a real producer that was never aliased to
            # `_get_llm` must not start being reported.
            "retriever = get_retriever()",
            "tool_fn = TOOL_REGISTRY.get(name)",
            "from x import get_retriever\nretriever = get_retriever()",
            "def get_retriever():\n    pass\nretriever = get_retriever()",
        ],
    )
    def test_an_unaliased_producer_is_not_reported(self, binding):
        """A guard that cried wolf on ordinary code would be edited around."""
        assert not _smuggled_model_bindings(binding), binding

    @pytest.mark.parametrize(
        "binding",
        [
            "async def f(retriever):\n    pass\n",
            "from aorta.chat.graph.nodes import _get_llm as tool_fn",
            # Every one of these binds through a *string* field on the node
            # rather than an `ast.Name(Store)`, so the Store walk saw nothing at
            # all and the allowlist went on exempting the receiver from the
            # chokepoint test -- a model reaching `.ainvoke` with `_send`
            # skipped entirely.
            "match _get_llm():\n    case retriever:\n        pass\n",
            "match x:\n    case object() as retriever:\n        pass\n",
            "match _get_llm():\n    case [*tool_fn]:\n        pass\n",
            "match _get_llm():\n    case {**retriever}:\n        pass\n",
            "try:\n    pass\nexcept Exception as retriever:\n    pass\n",
            # `def`, `async def` and `class` bind through `name` too, and the
            # enumerated version of this guard did not list them -- the fourth
            # leak, and the one that prompted the redesign. A decorator can
            # return anything, so a decorated `def retriever` is a real route:
            # `@something` over `def retriever(): ...` leaves `retriever`
            # holding whatever the decorator returned, which may be a model.
            # None of these needed a new entry to be caught.
            "def retriever():\n    return _get_llm()\n",
            "async def retriever():\n    return _get_llm()\n",
            "class tool_fn:\n    pass\n",
            "@wraps\ndef retriever():\n    pass\n",
            # A dotted import binds only its first component, but `alias.name`
            # holds the whole path -- so this recorded `retriever.client`,
            # matched no allowlisted receiver, and went unreported while
            # `retriever.ainvoke(...)` stayed exempt from the chokepoint test.
            "import retriever.client\n",
            "import tool_fn.a.b\n",
        ],
    )
    def test_a_binding_the_guard_cannot_read_is_reported_not_ignored(self, binding):
        """None of these has a value this file can inspect, so none is cleared.

        Reported rather than skipped: staying silent on an unfamiliar shape is
        how an allowlist stops being a guard, since the next smuggling route is
        by definition one this code has not seen.
        """
        complaints = _smuggled_model_bindings(binding)
        assert len(complaints) == 1
        assert "cannot read" in complaints[0]

    @pytest.mark.parametrize(
        "source",
        [
            "match x:\n    case _:\n        pass\n",
            "match x:\n    case {'a': 1}:\n        pass\n",
            "try:\n    pass\nexcept Exception:\n    pass\n",
            # The cost of the inverted default is that ordinary *reads* of these
            # identifiers must not be mistaken for bindings, so each of the
            # exemptions in `_NON_BINDING_IDENTIFIER_FIELDS` gets a case. A
            # guard that cried wolf on `obj.retriever` would be edited out of
            # the way, which is the failure mode this half exists to prevent.
            "obj.retriever\n",
            "x = 'retriever'\n",
            "f(retriever=1)\n",
            "from retriever import thing\n",
            "print(tool_fn)\n",
            # Binds `rc`, not `retriever`: `asname` replaces the dotted path
            # rather than adding to it, so reading both fields would report a
            # name this statement never introduces.
            "import retriever.client as rc\n",
        ],
    )
    def test_a_form_that_binds_no_name_is_not_reported(self, source):
        """The other half of failing closed: it must not fail *noisily*.

        ``case _:``, a mapping pattern with no ``**rest`` and a bare ``except:``
        all carry the string field the check above reads, set to ``None``. A
        guard that reported the *node* rather than the name it binds would
        complain about all three, and one that cries wolf on ordinary syntax
        gets edited out of the way.
        """
        assert _smuggled_model_bindings(source) == []

    def test_symtable_catches_a_binding_the_ast_scan_cannot_see(self, monkeypatch):
        """The scan's completeness is cross-checked, not asserted.

        This is the half that makes a fifth patch unnecessary rather than
        overdue. The scan above reports unknown *shapes*, but it is still code
        in this file reading nodes it was written against; ``symtable`` is
        CPython's own binding analysis, so it already knows every binding form
        the language has and will know the next one without this file changing.

        Driven by blinding the AST scan completely, because a blind spot that
        can be constructed in Python today would be a bug to fix rather than a
        case to pin -- the assertion worth making is that the oracle does not
        depend on the scan being right.
        """
        import sys

        # Patched through this module's own object rather than by dotted path:
        # the importable name for a test module depends on pytest's import mode
        # and rootdir, and a target that resolves here but not in CI would make
        # this pass for the wrong reason. No `raising=False` -- the attribute
        # must exist, and a rename should fail this test rather than skip it.
        monkeypatch.setattr(
            sys.modules[__name__], "_receiver_bindings", lambda _source: []
        )
        complaints = _smuggled_model_bindings("retriever = _get_llm()")
        assert len(complaints) == 1
        assert "symtable" in complaints[0]
        assert "retriever" in complaints[0]

    def test_the_oracle_stays_quiet_when_the_scan_accounts_for_everything(self):
        """The other direction: agreement between the two must not complain.

        ``graph/nodes.py`` binds both allowlisted names, so this would fire on
        the real file if the two disagreed about ordinary code -- which is what
        makes the assertion in
        ``test_the_allowlist_cannot_be_used_to_smuggle_a_model_out`` non-vacuous
        rather than green because nothing is being compared.
        """
        source = Path(nodes_path()).read_text(encoding="utf-8")
        assert _unaccounted_bound_names(source) == []
        assert _module_bound_names(source) >= set(_NON_MODEL_AINVOKE_RECEIVERS)

    def test_the_guarded_verbs_cover_every_async_api_the_model_exposes(self):
        """``BaseChatModel`` is the oracle, so a langchain upgrade cannot open a gap.

        An enumerated verb list is only as good as the day it was written: this
        one was missing ``agenerate_prompt``, so a direct
        ``llm.agenerate_prompt(...)`` would have sent prompts outside ``_send``
        with the chokepoint test still green. Rather than add that one name,
        derive the expectation from the class itself and subtract the async
        attributes that provably do not drive a request
        (:data:`_NON_INVOKING_ASYNC_ATTRS`).

        So the failure mode is inverted. A future langchain that adds an async
        verb fails here -- loudly, naming the verb -- instead of silently
        widening the hole the guard exists to close.
        """
        from langchain_core.language_models.chat_models import BaseChatModel

        exposed = {
            name
            for name in dir(BaseChatModel)
            if name.startswith("a")
            and not name.startswith("_")
            and callable(getattr(BaseChatModel, name, None))
        }
        unguarded = exposed - _NON_INVOKING_ASYNC_ATTRS - _MODEL_INVOCATION_VERBS
        assert not unguarded, (
            f"BaseChatModel exposes async verb(s) the redaction chokepoint does "
            f"not guard: {sorted(unguarded)}. Add each to "
            f"_MODEL_INVOCATION_VERBS, or to _NON_INVOKING_ASYNC_ATTRS if it "
            f"cannot drive a request."
        )
        # And the exclusion list must not rot into a way to hide a real verb:
        # every name in it has to still exist on the class.
        assert _NON_INVOKING_ASYNC_ATTRS <= exposed


def nodes_path() -> str:
    from aorta.chat.graph import nodes

    return nodes.__file__


def sys_module():
    import sys

    return sys
