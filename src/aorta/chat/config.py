"""Centralized configuration loaded from environment variables / .env file."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
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
    #          without --enable-auto-tool-choice, which is how scripts/
    #          start_vllm.sh and docker-compose.yml start it.
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
    aorta_path: str = str(_PROJECT_ROOT / "aorta")

    # --- Vector store ---
    chroma_path: str = str(_PROJECT_ROOT / "data" / "chroma")

    # --- Embedding provider selector ---
    # "local" uses the LOCAL block below; "remote" uses the REMOTE block.
    embedding_provider: str = "local"

    # --- LOCAL: embeddings ---
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # --- REMOTE: embeddings ---
    # Vector dimensions differ from the local model, so the two providers
    # cannot share a Chroma collection.
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
    # comma-separated form in .env.example loads; the validator below accepts
    # both that and a JSON list.
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
    repo_map_path: str = str(_PROJECT_ROOT / "data" / "repo_map.md")
    # Cap on what plan_node injects into its system message. The full map for a
    # real codebase is megabytes, which overflows any context window -- and on a
    # metered endpoint, does so expensively. The search_repo_map tool still
    # queries the whole file, so nothing becomes unreachable. 0 disables the cap.
    repo_map_prompt_max_chars: int = 20_000

    @field_validator("allowed_commands", mode="before")
    @classmethod
    def _parse_allowed_commands(cls, value: Any) -> Any:
        """Accept ``a,b,c`` as well as ``["a", "b", "c"]`` from the environment.

        ``.env.example`` ships the comma-separated form, which is what anyone
        writing an env file by hand expects. JSON is still honoured so existing
        ``.env`` files written against pydantic-settings' default decoding keep
        loading unchanged.
        """
        if not isinstance(value, str):
            return value
        text = value.strip()
        if text.startswith("["):
            try:
                return json.loads(text)
            except ValueError as exc:
                raise ValueError(
                    "ALLOWED_COMMANDS looks like JSON but is not valid JSON. "
                    "Use either python,pytest,ls or [\"python\", \"pytest\"]."
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

        The comma form is what an env file written by hand looks like; JSON is
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


settings = Settings()
