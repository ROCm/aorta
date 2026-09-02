"""``aorta chat`` -- interactive assistant over the AORTA codebase.

The only Click code for chat. Everything it drives lives in :mod:`aorta.chat`,
which contains no Click at all; ``tests/cli/test_chat_boundaries.py`` enforces
both directions against the AST.

**Module scope is click plus stdlib, deliberately.** ``aorta/cli/__init__.py``
imports every command module eagerly, so a single ``from aorta.chat...`` at
module scope here would put langchain, langgraph and openai on the critical path
of ``aorta --help`` -- for every user, including the ones who never installed
the chat extra. Every chat import therefore happens inside a command callback,
behind :func:`_load`, which is the same discipline (and the same
external-versus-internal ``ModuleNotFoundError`` distinction) as
``aorta/cli/bench.py``.

Ported from the ``src/cli.py`` of a standalone tool in an internal AMD
repository, which was argparse; the REPL, the three output modes and the
quiet-mode logging setup are that module's.
"""

from __future__ import annotations

import atexit
import contextlib
import importlib
import json
import logging
import os
import sys
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import click

logger = logging.getLogger(__name__)

#: Decision 13a. Enforced here as well as by the extras' environment markers,
#: because an extra whose every dependency is marked out of range installs
#: *successfully* and does nothing -- so without this the user's next signal
#: would be an ImportError for langchain right after pip said "Successfully
#: installed".
_MIN_PYTHON = (3, 11)

#: Chainlit's own ceiling, so ``aorta chat ui`` can tell a py3.14 user why the
#: chat-ui extra installed cleanly and gave them nothing.
_MAX_PYTHON_UI = (3, 14)

_INSTALL_HINT = (
    "'aorta chat' requires the chat-cli extra.\n"
    "Install it with:  pip install 'amd-aorta[chat-cli]'"
)

#: Hard-coded rather than read from ``aorta.chat.inference.providers.factory``:
#: decorators run at import time, so enumerating the factory here would import
#: langchain on every ``aorta --help``. ``tests/chat/test_cli_choices.py``
#: fails if this list and the factory's registry drift apart.
_LLM_PROVIDERS = ("litellm", "openai", "vllm")

#: Likewise hard-coded against ``aorta.chat.config.PROFILE_TEMPLATES``.
_CONFIG_PROFILES = ("anthropic", "azure-apim", "local-vllm", "openai", "openai-compatible")


def _require_python() -> None:
    """Refuse politely on an interpreter chat does not support."""
    if sys.version_info < _MIN_PYTHON:
        want = ".".join(str(part) for part in _MIN_PYTHON)
        have = f"{sys.version_info.major}.{sys.version_info.minor}"
        raise click.ClickException(
            f"aorta chat needs Python {want} or newer; this interpreter is {have}. "
            "The rest of aorta supports 3.10, so this affects 'aorta chat' only."
        )


def _load(name: str) -> Any:
    """Import ``aorta.chat.<name>``, turning a missing extra into a clear hint.

    Only a missing *external* dependency is swallowed. A ``ModuleNotFoundError``
    naming an ``aorta.chat`` module is a real bug -- a typo, a file that never
    got packaged -- and must surface as a traceback rather than as advice to
    install something the user already has. Same rule as ``cli/bench.py``.
    """
    _require_python()
    try:
        return importlib.import_module(f"aorta.chat.{name}")
    except ModuleNotFoundError as exc:
        missing = exc.name or ""
        if missing == "aorta.chat" or missing.startswith("aorta.chat."):
            raise
        raise click.ClickException(f"{_INSTALL_HINT}\n(missing: {missing})") from exc


# ── output renderers ──────────────────────────────────────────────────────


def _render_rich(reply: str) -> None:
    from rich.console import Console
    from rich.markdown import Markdown

    console = Console()
    console.print()
    console.print(Markdown(reply))
    console.print()


def _render_plain(reply: str) -> None:
    sys.stdout.write(reply)
    sys.stdout.write("\n")
    sys.stdout.flush()


def _render_json(query: str, reply: str, result: dict) -> None:
    obj = {"query": query, "response": reply, "iteration": result.get("iteration", 0)}
    sys.stdout.write(json.dumps(obj, ensure_ascii=False))
    sys.stdout.write("\n")
    sys.stdout.flush()


def _output_mode(as_json: bool, plain: bool) -> str:
    if as_json:
        return "json"
    if plain:
        return "plain"
    return "rich"


# ── logging / noise suppression ───────────────────────────────────────────


def _quiet_mode() -> None:
    """Suppress noisy third-party output that bypasses the logging system.

    Silences LangChain's DeprecationWarnings, tqdm progress bars, and
    fastembed's per-batch loguru chatter.
    """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    os.environ.setdefault("TQDM_DISABLE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    for noisy in (
        "httpx",
        "httpcore",
        "fastembed",
        "aorta.chat.rag",
        "openai",
        "langchain",
        "langgraph",
        "markdown_it",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # Keep backend readiness, remote call counts, and agent routing/act
    # messages visible.
    for useful in ("aorta.chat.session", "aorta.chat.inference", "aorta.chat.graph.nodes"):
        logging.getLogger(useful).setLevel(logging.INFO)


_stderr_suppressed = False


@contextlib.contextmanager
def _suppress_stderr_noise() -> Iterator[None]:
    """Redirect stderr to /dev/null for the first invocation only.

    Catches model-loading banners that go straight to stderr with no logging
    hook to silence them -- historically sentence-transformers' LOAD REPORT
    table, now onnxruntime's provider warnings. After the first call the model
    is loaded, so later calls are clean.
    """
    global _stderr_suppressed
    if _stderr_suppressed:
        yield
        return
    _stderr_suppressed = True
    real_stderr = sys.stderr
    devnull = open(os.devnull, "w")  # noqa: SIM115 - closed in the finally below
    sys.stderr = devnull
    try:
        yield
    finally:
        # Restored here rather than at process exit. In a REPL the first query
        # is one of many, and leaving stderr on /dev/null for the rest of the
        # session discarded every later warning, backend diagnostic and
        # traceback -- including the INFO loggers `_quiet_mode` deliberately
        # keeps.
        sys.stderr = real_stderr
        devnull.close()


# ── REPL ──────────────────────────────────────────────────────────────────

#: Enough to scroll back through a working session, small enough to stay fast.
HISTORY_LENGTH = 1000

#: The REPL prompt. It has to be handed to ``input()`` rather than printed
#: beforehand: readline works out where the line starts from the prompt it was
#: given, so a prompt it never saw gets treated as part of the editable line and
#: is erased the moment Up redraws it.
PROMPT = "aorta> "

#: ``\001``/``\002`` are readline's RL_PROMPT_START_IGNORE and
#: RL_PROMPT_END_IGNORE. They bracket bytes that occupy no columns, so readline
#: measures this prompt as the 7 visible characters rather than the 18 it
#: contains. Without them the colour escapes are counted and the cursor lands in
#: the wrong column, which corrupts the line on every redraw.
PROMPT_COLOUR = f"\001\033[1;32m\002{PROMPT}\001\033[0m\002"


def _history_path() -> Path:
    """Where the REPL remembers past queries between sessions.

    A dotfile directly in ``$HOME`` in the original standalone tool; now under
    the XDG cache with the rest of chat's regenerable state.
    """
    from aorta._user_paths import chat_cache_dir

    return chat_cache_dir() / "repl_history"


def _repl_prompt(output_mode: str, readline_active: bool) -> str:
    """Pick a prompt whose escapes suit the reader that will consume them."""
    if output_mode != "rich" or not sys.stdout.isatty():
        # Piped or machine-readable output: colour would be noise, and in JSON
        # mode the prompt is already unwanted on stdout.
        return PROMPT
    if readline_active:
        return PROMPT_COLOUR
    # No readline, so nothing measures the prompt and the ignore markers would
    # be printed literally.
    return f"\033[1;32m{PROMPT}\033[0m"


def _save_history(readline_module: Any, path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        readline_module.write_history_file(path)
    except OSError as exc:  # read-only or full home directory
        logger.debug("Could not write REPL history to %s: %s", path, exc)


def _enable_line_editing(history_file: Path) -> bool:
    """Give the REPL arrow-key history, line editing and Ctrl+R search.

    Importing ``readline`` *is* the mechanism: with it, the built-in ``input()``
    gains the full editing keymap, and without it an arrow key arrives as a raw
    escape sequence such as ``^[[A``.

    Returns whether readline is available, which decides how the prompt has to
    be written -- see :func:`_repl_prompt`.

    Everything here degrades quietly. History is a convenience, and no failure
    to read or write it is worth interrupting a session over.
    """
    try:
        import readline
    except ImportError:  # not available on Windows without a third-party wheel
        logger.debug("readline unavailable; REPL history and editing disabled.")
        return False

    try:
        readline.read_history_file(history_file)
    except FileNotFoundError:
        pass  # first run
    except OSError as exc:
        logger.debug("Could not read REPL history from %s: %s", history_file, exc)

    readline.set_history_length(HISTORY_LENGTH)
    atexit.register(_save_history, readline, history_file)
    return True


# ── async core ────────────────────────────────────────────────────────────


async def _ask_once(
    invoke_agent: Any,
    query: str,
    history: list,
    output_mode: str,
    quiet: bool,
) -> tuple[list, bool]:
    """Invoke the agent for one query and render it; also report success.

    The caller needs the second element: a one-shot ``aorta chat ask`` that
    exits 0 after failing to answer reports success to whatever is consuming
    the JSON or plain stream.
    """
    suppress = _suppress_stderr_noise() if quiet else contextlib.nullcontext()
    try:
        with suppress:
            reply, history, result = await invoke_agent(query, history)
    except Exception:
        logger.exception("Agent graph error")
        msg = "An error occurred while processing your request."
        if output_mode == "json":
            _render_json(query, msg, {})
        else:
            _render_plain(msg)
        return history, False

    if output_mode == "json":
        _render_json(query, reply, result)
    elif output_mode == "plain":
        _render_plain(reply)
    else:
        _render_rich(reply)
    return history, True


async def _interactive_loop(invoke_agent: Any, output_mode: str, quiet: bool) -> None:
    """Run a multi-turn REPL session."""
    banner = "AORTA Codebase Assistant  (type exit, quit, or /q to leave)"
    if output_mode == "rich":
        from rich.console import Console

        console = Console()
        console.print()
        console.print(f"[bold]{banner}[/bold]\n")
    else:
        # The banner and prompt go to stderr outside rich mode so a piped
        # --json run yields a clean JSONL stream on stdout.
        sys.stderr.write(f"{banner}\n\n")
        sys.stderr.flush()

    prompt = _repl_prompt(output_mode, _enable_line_editing(_history_path()))
    # Where the prompt is written decides whether stdout stays machine-readable.
    # ``input(prompt)`` puts it on stdout, so a --json REPL interleaved
    # "aorta> " with the JSON objects and emitted something that is not JSONL --
    # the very thing the banner above goes to stderr to avoid. On a terminal it
    # still has to go through input(), which is what lets readline measure the
    # line start and stop Up from erasing the prompt; piped, there is no redraw
    # to protect.
    prompt_via_input = sys.stdout.isatty()
    history: list = []

    while True:
        try:
            if prompt_via_input:
                query = input(prompt)
            else:
                sys.stderr.write(prompt)
                sys.stderr.flush()
                query = input()
        except (EOFError, KeyboardInterrupt):
            break

        stripped = query.strip().lower()
        if stripped in ("exit", "quit", "/q"):
            break
        if not stripped:
            continue

        history, _ = await _ask_once(invoke_agent, query, history, output_mode, quiet)


async def _run(query: str | None, output_mode: str, quiet: bool, no_wait: bool) -> bool:
    """Preflight the backend, then either answer once or start the REPL."""
    factory = _load("inference.providers.factory")
    session = _load("session")
    try:
        backend = factory.get_backend()
        if not no_wait:
            await backend.preflight()
    except (ImportError, ValueError) as exc:
        raise click.ClickException(f"LLM backend unavailable: {exc}") from exc
    logger.info("LLM backend: %s", backend.describe())

    if query is None:
        await _interactive_loop(session.invoke_agent, output_mode, quiet)
        # A REPL's exit status describes the session, not any one answer: the
        # user has already seen each failure and chosen to keep going.
        return True
    _, ok = await _ask_once(session.invoke_agent, query, [], output_mode, quiet)
    return ok


def _setup_logging(verbose: bool) -> bool:
    """Configure logging and return whether to run in quiet mode."""
    if not verbose:
        _quiet_mode()
        return True
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    # LiteLLM's debug records include the outbound request headers, which carry
    # the API key when a gateway needs it in a custom header. Verbose
    # diagnostics must not mean a key in the scrollback.
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    return False


def _common_options(command: Any) -> Any:
    """Options shared by ``aorta chat`` and ``aorta chat ask``.

    Both surfaces need them: the bare group runs the REPL, and ``ask`` must
    accept ``--json`` after the subcommand name rather than before it.
    """
    for decorator in reversed(
        (
            click.option(
                "--json",
                "as_json",
                is_flag=True,
                help="Emit each response as a JSON object (JSONL in the REPL).",
            ),
            click.option(
                "--plain",
                is_flag=True,
                help="Plain text with no formatting (pipe-friendly).",
            ),
            click.option(
                "--llm-provider",
                type=click.Choice(_LLM_PROVIDERS),
                default=None,
                help="Override the configured LLM provider for this run.",
            ),
            click.option(
                "--llm-model",
                default=None,
                help="Override the model name of the selected provider.",
            ),
            click.option(
                "--no-wait",
                is_flag=True,
                help="Skip the LLM backend preflight check.",
            ),
            click.option(
                "--no-redact",
                is_flag=True,
                help=(
                    "Send filesystem paths and IP addresses to the LLM unredacted "
                    "(default: redact, and say so on stderr)."
                ),
            ),
            click.option("-v", "--verbose", is_flag=True, help="Debug-level logging."),
        )
    ):
        command = decorator(command)
    return command


def _dispatch(
    query: str | None,
    as_json: bool,
    plain: bool,
    llm_provider: str | None,
    llm_model: str | None,
    no_wait: bool,
    no_redact: bool,
    verbose: bool,
) -> None:
    """Shared body of the bare group and ``ask``."""
    if as_json and plain:
        raise click.UsageError("--json and --plain are mutually exclusive.")
    # asyncio is stdlib, so it does not break the module-scope rule -- but it
    # costs ~33 ms to import, which is 12% of `aorta --help` for a command
    # almost nobody is running. The rule's reason applies even where its letter
    # does not.
    import asyncio

    config = _load("config")
    # One call, because `configure` rebuilds the settings from its arguments
    # alone -- a second call would drop the first one's overrides. redact stays
    # None unless the flag was given, so a profile's `redact = false` survives.
    config.apply_cli_overrides(
        provider=llm_provider,
        model=llm_model,
        redact=False if no_redact else None,
    )
    quiet = _setup_logging(verbose)
    if not asyncio.run(_run(query, _output_mode(as_json, plain), quiet, no_wait)):
        # The answer was never produced, so the advertised JSON and plain piping
        # modes must not report success to whatever is reading them. The message
        # is already rendered on the requested stream; this only sets the status.
        raise SystemExit(1)


# ── Click surface ─────────────────────────────────────────────────────────


@click.group(name="chat", invoke_without_command=True)
@_common_options
@click.pass_context
def chat(
    ctx: click.Context,
    as_json: bool,
    plain: bool,
    llm_provider: str | None,
    llm_model: str | None,
    no_wait: bool,
    no_redact: bool,
    verbose: bool,
) -> None:
    """Ask questions about the AORTA codebase (interactive REPL).

    Bare 'aorta chat' starts the REPL; 'aorta chat ask "..."' answers once and
    exits. Requires the chat-cli extra:

      pip install 'amd-aorta[chat-cli]'
    """
    if ctx.invoked_subcommand is not None:
        return
    _dispatch(None, as_json, plain, llm_provider, llm_model, no_wait, no_redact, verbose)


@chat.command(name="ask")
@click.argument("query")
@_common_options
def ask(
    query: str,
    as_json: bool,
    plain: bool,
    llm_provider: str | None,
    llm_model: str | None,
    no_wait: bool,
    no_redact: bool,
    verbose: bool,
) -> None:
    """Answer QUERY once and exit."""
    _dispatch(query, as_json, plain, llm_provider, llm_model, no_wait, no_redact, verbose)


@chat.command(name="ui")
@click.option("--host", default="127.0.0.1", show_default=True, help="Bind address.")
@click.option("--port", default=8000, show_default=True, type=int, help="Bind port.")
def ui(host: str, port: int) -> None:
    """Serve the Chainlit web UI (needs the chat-ui extra)."""
    _require_python()
    import importlib.util
    import subprocess

    if importlib.util.find_spec("chainlit") is None:
        if sys.version_info >= _MAX_PYTHON_UI:
            # Saying "install the chat-ui extra" here would send the user round
            # a loop: pip installs it successfully and Chainlit's marker
            # excludes it, so nothing changes. Name the real cause instead.
            want = ".".join(str(part) for part in _MAX_PYTHON_UI)
            have = f"{sys.version_info.major}.{sys.version_info.minor}"
            raise click.ClickException(
                f"Chainlit does not support Python {have} (it declares "
                f"< {want}), so the chat-ui extra installs nothing on this "
                "interpreter. Use a 3.11-3.13 environment for the web UI; "
                "'aorta chat' and 'aorta chat ask' work here."
            )
        raise click.ClickException(
            "'aorta chat ui' requires the chat-ui extra.\n"
            "Install it with:  pip install 'amd-aorta[chat-ui]'"
        )
    # find_spec rather than import: locating the app must not import chainlit
    # twice, and the module is handed to `chainlit run` as a path anyway.
    spec = importlib.util.find_spec("aorta.chat.ui.app")
    if spec is None or spec.origin is None:
        raise click.ClickException("could not locate aorta.chat.ui.app on disk")
    raise SystemExit(
        subprocess.call(
            [
                sys.executable,
                "-m",
                "chainlit",
                "run",
                spec.origin,
                "-h",
                "--host",
                host,
                "--port",
                str(port),
            ]
        )
    )


@chat.command(name="tools")
@click.option("--json", "as_json", is_flag=True, help="Emit the registry as JSON.")
def tools(as_json: bool) -> None:
    """List the agent tools, built-in and plugin-contributed.

    A package that registers an `aorta.chat_tools` entry point but does not
    appear here was skipped at load, and said why on stderr.
    """
    plugins = _load("plugins")
    registry = plugins.load_chat_tools()
    if as_json:
        click.echo(
            json.dumps(
                {
                    name: {
                        "source_package": entry.source_package,
                        "description": entry.tool.description,
                    }
                    for name, entry in sorted(registry.items())
                },
                indent=2,
            )
        )
        return
    for name, entry in sorted(registry.items()):
        summary = (entry.tool.description or "").strip().splitlines()
        click.echo(f"{name}  [{entry.source_package}]")
        if summary:
            click.echo(f"    {summary[0]}")


@chat.group(name="config")
def config_group() -> None:
    """Create and inspect the chat profile (~/.config/aorta/chat.toml)."""


@config_group.command(name="init")
@click.option(
    "--profile",
    type=click.Choice(_CONFIG_PROFILES),
    required=True,
    help="Starting point for the generated profile.",
)
@click.option("--force", is_flag=True, help="Overwrite an existing profile.")
@click.option(
    "--no-input",
    is_flag=True,
    help="Write the profile template without prompting (for scripting).",
)
def config_init(profile: str, force: bool, no_input: bool) -> None:
    """Write a chat profile for the chosen --profile.

    The file is created mode 0600 because it may hold an API key. 'aorta
    bundle' refuses to package it, and 'aorta chat config show' masks the key
    unless --reveal is passed.
    """
    config = _load("config")
    from aorta._user_paths import chat_config_path

    target = chat_config_path()
    if target.exists() and not force:
        raise click.ClickException(f"{target} already exists; pass --force to overwrite.")

    values = dict(config.PROFILE_TEMPLATES[profile])
    if not no_input:
        for field in config.PROFILE_PROMPTS[profile]:
            secret = field in config.SECRET_FIELDS
            answer = click.prompt(
                field.replace("_", " "),
                default=values.get(field, ""),
                hide_input=secret,
                show_default=not secret,
            )
            if answer:
                values[field] = answer
    written = config.write_profile(values)
    click.echo(f"Wrote {written} (mode {config.PROFILE_FILE_MODE:04o})")
    click.echo("Review it with: aorta chat config show")


@config_group.command(name="show")
@click.option(
    "--reveal",
    is_flag=True,
    help="Print API keys in full instead of masking them.",
)
@click.option("--json", "as_json", is_flag=True, help="Emit the settings as JSON.")
def config_show(reveal: bool, as_json: bool) -> None:
    """Print the effective configuration, with credentials masked."""
    config = _load("config")
    from aorta._user_paths import chat_config_path

    values = config.effective_settings(reveal=reveal)
    if as_json:
        click.echo(json.dumps(values, indent=2, sort_keys=True, default=str))
        return
    path = chat_config_path()
    click.echo(f"profile: {path}{'' if path.exists() else ' (not present)'}")
    click.echo("")
    for key in sorted(values):
        click.echo(f"  {key} = {values[key]!r}")
    if not reveal:
        click.echo("")
        click.echo("API keys are masked. Pass --reveal to print them.")


@config_group.command(name="validate")
def config_validate() -> None:
    """Check the profile parses, has no dead keys, and is not world-readable."""
    config = _load("config")
    problems = config.validate_profile()
    if not problems:
        click.echo("Profile OK.")
        return
    for problem in problems:
        click.echo(f"error: {problem}", err=True)
    raise click.exceptions.Exit(1)


# ── index and doctor ──────────────────────────────────────────────────────


def _index_logging(verbose: bool) -> None:
    """Show build/fetch progress, which is minutes long and otherwise silent."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
    )
    for noisy in ("httpx", "httpcore", "fastembed", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _guard(action: Any) -> Any:
    """Run ``action``, turning chat's known failures into a clean CLI error.

    The messages these carry are the deliverable -- an index-mismatch refusal
    and a missing-model failure are written to be read by the person whose
    command just stopped -- so they are surfaced verbatim rather than wrapped in
    a traceback.
    """
    corpus = _load("rag.corpus")
    manifest = _load("rag.manifest")
    ops = _load("rag.index_ops")
    embeddings = _load("rag.embeddings.fastembed_bge")
    known = (
        corpus.PublicTreeError,
        embeddings.ModelUnavailableError,
        manifest.IndexMismatchError,
        manifest.ManifestError,
        ops.IndexFetchError,
        FileNotFoundError,
        # The embedding and LLM provider factories report bad configuration as
        # ValueError, message-first; a traceback would bury it.
        ValueError,
    )
    try:
        return action()
    except known as exc:
        raise click.ClickException(str(exc)) from exc


def _resolve_corpus(path: str | None, public_only: bool) -> Any:
    """Build the corpus spec for ``index build`` / ``index digest``."""
    corpus = _load("rag.corpus")
    config = _load("config")
    if public_only:
        # The CI path. Verifies the checkout is ROCm/aorta and restricts the
        # walk to git-tracked files, because the index embeds source verbatim.
        return corpus.published_corpus(Path(path or ".").resolve())
    return corpus.local_corpus(path or config.settings.aorta_path)


@chat.group(name="index")
def index_group() -> None:
    """Build, download or inspect the retrieval index."""


@index_group.command(name="build")
@click.option("--path", default=None, help="Corpus root. Defaults to the installed aorta package.")
@click.option("--output", default=None, help="Where to write the index. Defaults to the cache.")
@click.option(
    "--public-only",
    is_flag=True,
    help="Index only git-tracked files of a ROCm/aorta checkout (what CI publishes).",
)
@click.option("--json", "as_json", is_flag=True, help="Emit the build result as JSON.")
@click.option("-v", "--verbose", is_flag=True, help="Debug-level logging.")
def index_build(
    path: str | None, output: str | None, public_only: bool, as_json: bool, verbose: bool
) -> None:
    """Build the index from source on this machine.

    The air-gapped and developer path. Needs the embedding weights, which are
    downloaded once (~65 MB) unless the cache is pre-seeded -- run 'aorta chat
    doctor' first if this node has no egress.
    """
    _index_logging(verbose)
    ops = _load("rag.index_ops")
    result = _guard(lambda: ops.build_index(_resolve_corpus(path, public_only), index_path=output))

    if as_json:
        click.echo(
            json.dumps(
                {
                    "index": str(result.index_path),
                    "files": result.file_count,
                    "chunks": result.chunk_count,
                    "size_bytes": result.size_bytes,
                    "seconds": round(result.seconds, 1),
                    "corpus_digest": result.manifest.corpus_digest,
                    "collection": result.manifest.collection,
                    "dimensions": result.manifest.dimensions,
                },
                indent=2,
            )
        )
        return
    manifest = result.manifest
    click.echo(f"Built {result.index_path}")
    click.echo(f"  corpus      {result.corpus}")
    click.echo(f"  files       {result.file_count}")
    click.echo(f"  chunks      {result.chunk_count}")
    click.echo(f"  size        {result.size_bytes / (1024 * 1024):.1f} MB")
    click.echo(f"  time        {result.seconds:.1f}s")
    click.echo(f"  model       {manifest.embedding_model} @ {manifest.dimensions}d")
    click.echo(f"  collection  {manifest.collection}")
    click.echo(f"  digest      {manifest.corpus_digest}")


@index_group.command(name="fetch")
@click.option(
    "--version",
    default=None,
    help="Release version or tag to fetch. Overrides version matching.",
)
@click.option(
    "--from",
    "from_path",
    default=None,
    help="Side-load a staged index file instead of downloading (air-gapped path).",
)
@click.option("--output", default=None, help="Where to install it. Defaults to the cache.")
@click.option("--json", "as_json", is_flag=True, help="Emit the result as JSON.")
@click.option("-v", "--verbose", is_flag=True, help="Debug-level logging.")
def index_fetch(
    version: str | None,
    from_path: str | None,
    output: str | None,
    as_json: bool,
    verbose: bool,
) -> None:
    """Download the prebuilt index matching this aorta version.

    An exact release version takes that release's asset; a development version
    takes the rolling asset built from main and reports how far off it is.
    Pass --version to override, or --from to side-load a staged file.
    """
    if version and from_path:
        raise click.UsageError("--version and --from are mutually exclusive.")
    _index_logging(verbose)
    ops = _load("rag.index_ops")

    if from_path:
        result = _guard(lambda: ops.side_load(from_path, index_path=output))
    else:
        result = _guard(lambda: ops.fetch_index(version=version, index_path=output))

    if as_json:
        click.echo(
            json.dumps(
                {
                    "index": str(result.index_path),
                    "source": result.source,
                    "warnings": result.warnings,
                    "manifest": result.manifest.describe(),
                },
                indent=2,
            )
        )
        return
    click.echo(f"Installed {result.index_path}")
    click.echo(f"  source    {result.source}")
    click.echo(f"  built as  {result.manifest.describe()}")
    for warning in result.warnings:
        click.echo(f"warning: {warning}", err=True)


@index_group.command(name="digest")
@click.option("--path", default=None, help="Corpus root. Defaults to the installed aorta package.")
@click.option(
    "--public-only", is_flag=True, help="Restrict to git-tracked files of the public tree."
)
def index_digest(path: str | None, public_only: bool) -> None:
    """Print the corpus digest without building anything.

    CI compares this against the published manifest's digest and skips the
    rebuild when they match, so an unchanged corpus does not re-upload tens of
    megabytes every night.
    """
    ops = _load("rag.index_ops")
    digest, files = _guard(lambda: ops.compute_digest(_resolve_corpus(path, public_only)))
    click.echo(json.dumps({"corpus_digest": digest, "files": files}))


@index_group.command(name="eval")
@click.option("--questions", default=None, help="JSON question set. Defaults to the shipped one.")
@click.option("--k", default=None, type=int, help="Retrieval depth. Defaults to the configured k.")
@click.option(
    "--json", "as_json", is_flag=True, help="Emit scores and per-question detail as JSON."
)
def index_eval(questions: str | None, k: int | None, as_json: bool) -> None:
    """Score retrieval over a question set (Decision 19b).

    Run it before and after changing the embedding model, so the next model
    argument is settled with numbers from this corpus rather than leaderboards.
    """
    _index_logging(False)
    evaluation = _load("rag.eval")
    config = _load("config")
    depth = k or config.settings.retriever_k
    result = _guard(lambda: evaluation.evaluate(evaluation.load_questions(questions), k=depth))

    if as_json:
        click.echo(json.dumps(result.to_dict(), indent=2))
        return
    click.echo(f"model: {result.embedding_model}")
    click.echo(result.summary())
    for item in result.results:
        position = f"@{item.rank}" if item.hit else "MISS"
        click.echo(f"  {position:<6} {item.question.question}")
        if not item.hit:
            click.echo(f"         expected one of: {', '.join(item.question.expected)}")


#: Marker widths are fixed so the statuses line up into a scannable column.
_STATUS_MARK = {"ok": "[ ok ]", "warn": "[warn]", "fail": "[FAIL]", "skip": "[ -- ]"}


@chat.command(name="doctor")
@click.option(
    "--no-backend",
    is_flag=True,
    help="Skip the LLM backend preflight (which needs the network).",
)
@click.option("--json", "as_json", is_flag=True, help="Emit the report as JSON.")
def doctor(no_backend: bool, as_json: bool) -> None:
    """Report extras, backend reachability, index freshness and model cache.

    Exits non-zero if anything is broken, so it works as a CI or setup gate.
    """
    _require_python()
    module = _load("doctor")
    report = module.run_checks(backend=not no_backend)

    if as_json:
        click.echo(
            json.dumps(
                {
                    "ok": not report.failed,
                    "checks": [
                        {
                            "name": check.name,
                            "status": check.status,
                            "detail": check.detail,
                            "hint": check.hint,
                            "procedure": check.procedure,
                        }
                        for check in report.checks
                    ],
                },
                indent=2,
            )
        )
    else:
        for check in report.checks:
            mark = _STATUS_MARK.get(check.status, "[ ?? ]")
            click.echo(f"{mark} {check.name:<22} {check.detail}")
            for line in check.hint.splitlines():
                click.echo(f"       {line}")
            if check.procedure:
                click.echo("")
                for line in check.procedure.splitlines():
                    click.echo(f"  {line}")
                click.echo("")
    if report.failed:
        raise click.exceptions.Exit(1)


__all__ = ["chat"]
