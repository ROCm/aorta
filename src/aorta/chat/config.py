"""Configuration for ``aorta chat``.

Replaces ``aorta_llm``'s ``config/settings.py``. Three things changed in the
move, and all three are load-bearing:

1. **Defaults are re-anchored on the user, not on the package.** The original
   anchored the ``.env`` file, the AORTA path, the vector store and the repo map
   on ``Path(__file__).parent.parent`` -- which inside an installed wheel is
   ``site-packages/aorta/``. Every writable default now resolves under
   ``$XDG_CACHE_HOME``, and the profile file under ``$XDG_CONFIG_HOME``, via
   :mod:`aorta._user_paths`.
2. **Settings are built on first use, not at import.** The original ran
   ``settings = Settings()`` at module scope, so a malformed environment raised
   during *import* -- which, once the chat command is registered on the shared
   ``aorta`` CLI, would break ``aorta --help`` for everyone. :data:`settings` is
   now a proxy that builds the real object on first attribute access.
3. **Every environment variable is namespaced ``AORTA_CHAT_``.** Bare names like
   ``CHUNK_SIZE`` and ``ALLOWED_COMMANDS`` are far too collision-prone for a
   public package that runs inside other people's job scripts.

Resolution order, highest first: constructor arguments (which is how the CLI
applies its flags) > ``AORTA_CHAT_*`` environment > the TOML profile file >
built-in defaults.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Annotated, Any

import tomllib
from pydantic import Field, field_validator
from pydantic_settings import (
    BaseSettings,
    NoDecode,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from aorta._user_paths import chat_cache_dir, chat_config_path

logger = logging.getLogger(__name__)

#: Prefix on every environment variable this class reads.
ENV_PREFIX = "AORTA_CHAT_"

#: The installed ``aorta`` package directory. Used as the default RAG corpus:
#: it is real code the user demonstrably has, and it is only ever read.
_AORTA_PACKAGE_ROOT = Path(__file__).resolve().parents[1]


class ConfigFileError(RuntimeError):
    """The profile file exists but could not be read or parsed."""

    def __init__(self, path: Path, cause: Exception) -> None:
        super().__init__(
            f"could not read the chat profile at {path}: {cause}\n"
            "Fix the file, or move it aside and re-run 'aorta chat config init'."
        )
        self.path = path
        self.cause = cause


def read_profile(path: Path | None = None) -> dict[str, Any]:
    """Return the raw TOML mapping from the profile file, or ``{}`` if absent.

    Shared by the settings source and by ``aorta chat config show|validate``,
    which need the file's own contents rather than the merged result.
    """
    path = path or chat_config_path()
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except FileNotFoundError:
        return {}
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigFileError(path, exc) from exc


class _TomlProfileSource(PydanticBaseSettingsSource):
    """Reads ``$XDG_CONFIG_HOME/aorta/chat.toml`` with stdlib ``tomllib``.

    Ranked below the environment so a CI job or a one-off shell export always
    wins over the file, and above the built-in defaults so the file is what
    ``aorta chat config init`` writes to make a setting stick.

    Unknown keys are dropped rather than rejected: the file is hand-edited, and
    a profile written by a newer aorta must not stop an older one from starting.
    ``aorta chat config validate`` is where unknown keys get reported.
    """

    def __call__(self) -> dict[str, Any]:
        raw = read_profile()
        known = set(self.settings_cls.model_fields)
        return {key: value for key, value in raw.items() if key in known}

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
        # Required by the ABC but unused: __call__ returns the whole mapping in
        # one read, which is cheaper than one file open per field.
        raise NotImplementedError


class Settings(BaseSettings):
    """Every knob ``aorta chat`` reads. Construct via :func:`get_settings`."""

    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        extra="ignore",
    )

    # --- LLM provider selector ---
    # "vllm" uses the LOCAL block below; "openai" and "litellm" use the
    # REMOTE block. The two blocks are kept disjoint so either flow can be
    # removed without disturbing the other.
    llm_provider: str = "vllm"

    # --- LOCAL: vLLM inference ---
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_model: str = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
    vllm_api_key: str = "EMPTY"

    # --- REMOTE: API-key LLM provider ---
    # remote_llm_base_url is optional: empty means the provider default
    # (api.openai.com for "openai", LiteLLM routing for "litellm").
    # "litellm" reads its own keys from the standard LiteLLM environment
    # variables (ANTHROPIC_API_KEY, GEMINI_API_KEY, ...), not from here.
    remote_llm_model: str = "gpt-4o-mini"
    remote_llm_api_key: str = ""
    remote_llm_base_url: str = ""
    # Gateways that want the key in a custom header rather than as a bearer
    # token: set the header name here (Ocp-Apim-Subscription-Key for Azure API
    # Management, api-key for Azure OpenAI, x-api-key for Anthropic's own API).
    # remote_llm_extra_headers carries non-secret extras a gateway needs for
    # attribution or quota, e.g. {"user": "alice"}.
    remote_llm_auth_header: str = ""
    remote_llm_extra_headers: Annotated[dict[str, str], NoDecode] = {}

    # --- Tool-calling protocol ---
    # "text"   act_node asks for `ACTION: tool(arg="v")` lines and parses them.
    #          Works with any chat model, including a vLLM server started
    #          without --enable-auto-tool-choice.
    # "native" OpenAI function calling. Required by reasoning models such as
    #          gpt-oss, which put text in a reasoning channel and return empty
    #          content rather than emitting the ACTION: syntax -- every query
    #          then exhausts the retry budget and answers nothing.
    # Default "text" so an existing local deployment is unaffected.
    llm_tool_mode: str = "text"

    # --- LLM call limits (both providers) ---
    llm_max_tokens: int | None = None
    llm_timeout: float = 120.0
    llm_max_retries: int = 2

    # --- AORTA codebase ---
    # The RAG corpus. Defaults to the installed aorta package, which is read
    # only -- never written -- so pointing at site-packages is safe here in a
    # way that the index and repo-map paths below are not.
    aorta_path: str = str(_AORTA_PACKAGE_ROOT)

    # --- Run artifacts ---
    # Where this user's own aorta run directories live, for the run-artifact
    # tools and the second RAG collection (Decision 11b). Empty means the
    # working directory, because `aorta sweep run --output` is normally a
    # relative path and the operator runs the assistant from the same place.
    runs_path: str = ""

    # --- Vector store ---
    # A single sqlite-vec database file (not a Chroma directory): one file to
    # checksum, side-load, or delete. A default_factory rather than a literal so
    # $XDG_CACHE_HOME is read when the object is built, not when this module is
    # first imported -- otherwise a test (or a job script) that sets it would be
    # silently ignored.
    index_path: str = Field(default_factory=lambda: str(chat_cache_dir() / "index.sqlite"))

    # --- Egress redaction (Decision 16) ---
    # On by default. The run-artifact collection puts the user's own env.json
    # and matrix.json into retrieval, so a retrieved chunk can carry customer
    # paths and addresses towards a third-party API. Opt out per invocation
    # with `aorta chat --no-redact`, or permanently with `redact = false`.
    redact: bool = True

    # --- Embedding provider selector ---
    # "local" uses the LOCAL block below; "remote" uses the REMOTE block.
    embedding_provider: str = "local"

    # --- LOCAL: embeddings ---
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # --- REMOTE: embeddings ---
    # Vector dimensions differ from the local model, so the two providers
    # cannot share a collection.
    remote_embedding_model: str = "text-embedding-3-small"
    remote_embedding_api_key: str = ""
    remote_embedding_base_url: str = ""
    # Same custom-header support as the chat side; a customer behind a gateway
    # needs it for both or neither.
    remote_embedding_auth_header: str = ""
    remote_embedding_extra_headers: Annotated[dict[str, str], NoDecode] = {}

    # --- Indexer ---
    chunk_size: int = 512
    chunk_overlap: int = 50

    # --- Sandbox ---
    # NoDecode turns off pydantic-settings' JSON decoding for this field so the
    # comma-separated form loads; the validator below accepts both that and a
    # JSON list.
    allowed_commands: Annotated[list[str], NoDecode] = [
        "python",
        "pytest",
        "make",
        "pip",
        "grep",
        "wc",
        "head",
        "tail",
        "cat",
        "ls",
        "find",
    ]
    command_timeout: int = 60

    # --- Retrieval ---
    retriever_k: int = 12
    retriever_fetch_k: int = 30
    search_tool_k: int = 10

    # --- Agent ---
    max_retry_iterations: int = 3
    max_act_rounds: int = 5
    max_act_rounds_search: int = 8

    # --- Repo map ---
    repo_map_path: str = Field(default_factory=lambda: str(chat_cache_dir() / "repo_map.md"))
    # Cap on what plan_node injects into its system message. The full map for a
    # real codebase is megabytes, which overflows any context window -- and on a
    # metered endpoint, does so expensively. The search_repo_map tool still
    # queries the whole file, so nothing becomes unreachable. 0 disables the cap.
    repo_map_prompt_max_chars: int = 20_000

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Constructor args > ``AORTA_CHAT_*`` env > profile file > defaults.

        ``dotenv_settings`` and ``file_secret_settings`` are dropped. The
        original read a ``.env`` beside the package, which in a wheel means
        ``site-packages`` -- so it never resolved for an installed user anyway,
        and keeping it would give two files that configure the same tool.
        """
        return (init_settings, env_settings, _TomlProfileSource(settings_cls))

    @field_validator("allowed_commands", mode="before")
    @classmethod
    def _parse_allowed_commands(cls, value: Any) -> Any:
        """Accept ``a,b,c`` as well as ``["a", "b", "c"]`` from the environment.

        The comma-separated form is what anyone writing an env var by hand
        expects. JSON is still honoured so values written against
        pydantic-settings' default decoding keep loading unchanged.
        """
        if not isinstance(value, str):
            return value
        text = value.strip()
        if text.startswith("["):
            try:
                return json.loads(text)
            except ValueError as exc:
                raise ValueError(
                    f"{ENV_PREFIX}ALLOWED_COMMANDS looks like JSON but is not valid "
                    'JSON. Use either python,pytest,ls or ["python", "pytest"].'
                ) from exc
        return [item.strip() for item in text.split(",") if item.strip()]

    @field_validator(
        "remote_llm_extra_headers",
        "remote_embedding_extra_headers",
        mode="before",
    )
    @classmethod
    def _parse_extra_headers(cls, value: Any) -> Any:
        """Accept ``k=v,k2=v2`` as well as a JSON object.

        The comma form is what an env var written by hand looks like; JSON is
        the escape hatch for a value that itself contains a comma. Splitting on
        the first ``=`` only, so a base64-ish value keeps its padding.
        """
        if not isinstance(value, str):
            return value
        text = value.strip()
        if not text:
            return {}
        if text.startswith("{"):
            try:
                return json.loads(text)
            except ValueError as exc:
                raise ValueError(
                    "extra headers look like JSON but are not valid JSON. Use "
                    'either user=alice,x-tenant=amd or {"user": "alice"}.'
                ) from exc
        headers: dict[str, str] = {}
        for pair in text.split(","):
            if not pair.strip():
                continue
            if "=" not in pair:
                raise ValueError(
                    f"extra header {pair.strip()!r} is missing '='. Use "
                    'either user=alice,x-tenant=amd or {"user": "alice"}.'
                )
            name, _, header_value = pair.partition("=")
            headers[name.strip()] = header_value.strip()
        return headers

    @property
    def aorta_root(self) -> Path:
        return Path(self.aorta_path).resolve()

    @property
    def index_file(self) -> Path:
        return Path(self.index_path).expanduser()

    @property
    def runs_root(self) -> Path:
        """Sandbox root for the run-artifact tools and the run collection.

        Resolved on every access rather than cached, because the default is the
        working directory and a REPL session can outlive a ``cd``.
        """
        if not self.runs_path.strip():
            return Path.cwd().resolve()
        return Path(self.runs_path).expanduser().resolve()


#: Field names whose value is a credential. ``aorta chat config show`` masks
#: these unless ``--reveal`` is passed (the second half of Decision 9b, the
#: first being the ``aorta bundle`` exclusion in :mod:`aorta.bundle.writer`).
SECRET_FIELDS = frozenset(
    {
        "remote_llm_api_key",
        "remote_embedding_api_key",
        "vllm_api_key",
    }
)

_cached: Settings | None = None


def get_settings() -> Settings:
    """Return the process-wide :class:`Settings`, building it on first call."""
    global _cached
    if _cached is None:
        _cached = Settings()
    return _cached


def reset_settings() -> None:
    """Drop the cached settings so the next access re-reads env and file.

    Needed by tests, and by ``aorta chat config init``, which writes the profile
    file in a process that may already have read it.
    """
    global _cached
    _cached = None


def configure(**overrides: Any) -> Settings:
    """Rebuild the process-wide settings with *overrides* at top precedence.

    Constructor arguments are the highest-ranked source (see
    :meth:`Settings.settings_customise_sources`), which is how ``aorta chat``
    makes its flags outrank both the environment and the profile file. ``None``
    values are dropped so an unset flag means "no opinion" rather than "empty".
    """
    global _cached
    _cached = Settings(**{key: value for key, value in overrides.items() if value is not None})
    return _cached


def apply_cli_overrides(
    provider: str | None = None,
    model: str | None = None,
    redact: bool | None = None,
) -> Settings:
    """Apply the chat CLI's setting flags and return the new settings.

    Every CLI override is composed here rather than through separate
    :func:`configure` calls, because ``configure`` rebuilds the settings from
    the named arguments alone -- so a second call would discard the first
    call's overrides rather than adding to them.

    ``--llm-model`` names a model without saying which provider's field it
    belongs to, and the answer depends on the *resolved* provider -- which may
    itself have come from the profile file. So the provider is settled first and
    the model is folded in on a second pass, rather than both being guessed up
    front.

    ``redact`` is tri-state: ``None`` means the flag was absent, which must
    leave a profile's ``redact = false`` in force rather than re-asserting the
    default.
    """
    overrides: dict[str, Any] = {}
    if provider:
        overrides["llm_provider"] = provider
    if redact is not None:
        overrides["redact"] = redact
    resolved = configure(**overrides)
    if model:
        field = "vllm_model" if resolved.llm_provider == "vllm" else "remote_llm_model"
        resolved = configure(**overrides, **{field: model})
    return resolved


class _LazySettings:
    """Module-level ``settings`` that materialises on first attribute access.

    Call sites keep the ``settings.chunk_size`` shape they had in ``aorta_llm``,
    so the move did not touch ~15 modules, but importing any of them no longer
    validates the environment. Attribute *writes* forward to the real object as
    well, which is what lets a test ``monkeypatch.setattr(settings, ...)``.
    """

    __slots__ = ()

    def __getattr__(self, name: str) -> Any:
        return getattr(get_settings(), name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(get_settings(), name, value)

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"<lazy {'unloaded' if _cached is None else repr(_cached)}>"


#: The object every ``aorta.chat`` module imports.
settings = _LazySettings()


# ── profile generation (`aorta chat config init`) ──────────────────────────

#: Starting points for ``aorta chat config init``, keyed by ``--profile``.
#: ``aorta/cli/chat.py`` hard-codes the same names in its ``click.Choice`` --
#: decorators run at import time, so it cannot read this dict -- and
#: ``tests/chat/test_config_wizard.py`` fails if the two drift apart.
PROFILE_TEMPLATES: dict[str, dict[str, Any]] = {
    "openai": {
        "llm_provider": "openai",
        "remote_llm_model": "gpt-4o-mini",
        "embedding_provider": "remote",
        "remote_embedding_model": "text-embedding-3-small",
    },
    "openai-compatible": {
        "llm_provider": "openai",
        "remote_llm_model": "",
        "remote_llm_base_url": "",
        "embedding_provider": "remote",
    },
    "azure-apim": {
        "llm_provider": "openai",
        "remote_llm_model": "",
        "remote_llm_base_url": "",
        # Azure API Management reads the key from a named header rather than
        # from Authorization; without this the gateway answers 401 to a request
        # that looks correct.
        "remote_llm_auth_header": "Ocp-Apim-Subscription-Key",
        "embedding_provider": "remote",
        "remote_embedding_auth_header": "Ocp-Apim-Subscription-Key",
    },
    "anthropic": {
        # Native Anthropic wire protocol, which the openai backend cannot
        # speak; litellm needs the chat-all extra.
        "llm_provider": "litellm",
        "remote_llm_model": "claude-sonnet-4-5",
        "embedding_provider": "remote",
    },
    "local-vllm": {
        "llm_provider": "vllm",
        "vllm_base_url": "http://localhost:8000/v1",
        "vllm_model": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
        "embedding_provider": "local",
    },
}

#: Fields ``aorta chat config init`` asks about, per profile, in order. The
#: wizard prompts for exactly these; anything else is taken from the template.
PROFILE_PROMPTS: dict[str, tuple[str, ...]] = {
    "openai": ("remote_llm_model", "remote_llm_api_key"),
    "openai-compatible": ("remote_llm_base_url", "remote_llm_model", "remote_llm_api_key"),
    "azure-apim": ("remote_llm_base_url", "remote_llm_model", "remote_llm_api_key"),
    "anthropic": ("remote_llm_model", "remote_llm_api_key"),
    "local-vllm": ("vllm_base_url", "vllm_model"),
}

#: Mode the profile file is created with. Decision 9b keeps the API key at rest
#: in a predictable path, so owner-only is the minimum bar -- and the two guards
#: it obliges are the masked ``config show`` here and the ``aorta bundle``
#: exclusion in :mod:`aorta.bundle.writer`.
PROFILE_FILE_MODE = 0o600


def _toml_scalar(value: Any) -> str:
    """Render one TOML value.

    A deliberately small writer rather than a new dependency: the profile only
    ever holds the scalar, list-of-string and string-table field types declared
    on :class:`Settings`, and the base install is pyyaml + click.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_scalar(item) for item in value) + "]"
    if isinstance(value, dict):
        inner = ", ".join(f"{key} = {_toml_scalar(val)}" for key, val in value.items())
        return "{" + inner + "}"
    return json.dumps(str(value))  # TOML basic strings are JSON-escaped strings


def render_profile(values: dict[str, Any]) -> str:
    """Serialise *values* as the body of ``chat.toml``."""
    lines = [
        "# aorta chat profile. Written by 'aorta chat config init'.",
        "# Environment variables (AORTA_CHAT_*) override anything set here.",
        "",
    ]
    for key in sorted(values):
        lines.append(f"{key} = {_toml_scalar(values[key])}")
    return "\n".join(lines) + "\n"


def write_profile(values: dict[str, Any], path: Path | None = None) -> Path:
    """Write the profile file at :data:`PROFILE_FILE_MODE` and return its path.

    The mode is set on the file descriptor *before* any bytes are written, so a
    key is never briefly world-readable. An existing file is truncated but its
    mode is re-applied, because a previous run (or a careless editor) may have
    left it at 0644.
    """
    path = path or chat_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    body = render_profile(values)
    # os.open carrying the mode, rather than write_text followed by chmod: the
    # latter leaves a window in which the key is on disk under the process
    # umask. The explicit chmod after covers the pre-existing-file case, where
    # O_CREAT's mode argument is ignored.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, PROFILE_FILE_MODE)
    try:
        handle = open(fd, "w", encoding="utf-8")
    except BaseException:
        os.close(fd)
        raise
    with handle:  # takes ownership of fd
        handle.write(body)
    os.chmod(path, PROFILE_FILE_MODE)
    reset_settings()
    return path


def mask(value: str) -> str:
    """Render a credential as its length and last four characters.

    Enough to tell two keys apart in a support conversation, not enough to use.
    """
    if not value:
        return ""
    if len(value) <= 8:
        return "*" * len(value)
    return f"{'*' * (len(value) - 4)}{value[-4:]}"


def effective_settings(reveal: bool = False) -> dict[str, Any]:
    """The merged configuration as ``aorta chat config show`` prints it.

    Credentials are masked unless *reveal*, which is the first of the two
    guards Decision 9b obliges: the likeliest leak is a customer pasting their
    own config into a support ticket.
    """
    values = get_settings().model_dump()
    if not reveal:
        for name in SECRET_FIELDS:
            if name in values:
                values[name] = mask(str(values[name]))
    return values


def validate_profile(path: Path | None = None) -> list[str]:
    """Return the problems with the on-disk profile; empty means healthy.

    Reports unreadable/malformed files, keys that no longer exist, values that
    fail validation, and a profile holding a credential at a permissive mode --
    the last being a real finding on a shared node, not a style note.
    """
    path = path or chat_config_path()
    problems: list[str] = []
    if not path.exists():
        return [f"no profile at {path}; run 'aorta chat config init'"]

    try:
        raw = read_profile(path)
    except ConfigFileError as exc:
        return [str(exc)]

    unknown = sorted(set(raw) - set(Settings.model_fields))
    if unknown:
        problems.append(f"{path}: unknown key(s) {', '.join(unknown)} (ignored at load time)")

    try:
        Settings()
    except Exception as exc:  # pydantic ValidationError, or a field validator
        problems.append(f"{path}: {exc}")

    if any(raw.get(name) for name in SECRET_FIELDS):
        mode = path.stat().st_mode & 0o777
        if mode != PROFILE_FILE_MODE:
            problems.append(
                f"{path}: holds a credential but its mode is {mode:04o}, not "
                f"{PROFILE_FILE_MODE:04o}. Fix with: chmod 600 {path}"
            )
    return problems


__all__ = [
    "ENV_PREFIX",
    "PROFILE_FILE_MODE",
    "PROFILE_PROMPTS",
    "PROFILE_TEMPLATES",
    "SECRET_FIELDS",
    "ConfigFileError",
    "Settings",
    "apply_cli_overrides",
    "configure",
    "effective_settings",
    "get_settings",
    "mask",
    "read_profile",
    "render_profile",
    "reset_settings",
    "settings",
    "validate_profile",
    "write_profile",
]
