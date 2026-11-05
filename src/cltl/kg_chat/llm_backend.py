"""
llm_backend.py
================

Which LLM every chat-completion call goes to: OpenAI, or a local server with an OpenAI-compatible
API, such as llama.cpp's

    llama-server -hf Qwen/Qwen2.5-7B-Instruct-GGUF:Q4_K_M --port 8080 -c 16384

whose API is at http://localhost:8080/v1. Both are reached through the openai SDK; only the base
URL, the API key and the model name differ. Call configure() once, before the first conversation
(cltl.diabetes.container does this from [cltl.diabetes] llm_backend etc.); without it, OpenAI is
used, as before.

The local server differs in two ways the callers take care of:
- No tool calling unless llama-server runs with --jinja, so the event extractor asks for its
  structured output as a JSON schema instead (see structured_output()).
- A small context window (-c), so the extractor keeps only the last turns of its history.
"""

import json
import os
import urllib.request
from typing import Optional

from openai import OpenAI

OPENAI = "openai"
LOCAL = "local"
BACKENDS = (OPENAI, LOCAL)

DEFAULT_LOCAL_BASE_URL = "http://localhost:8080/v1"
# llama-server serves the one model it was started with and ignores the name in the request.
DEFAULT_LOCAL_MODEL = "local"
# How many earlier user turns the event extractor keeps in its history with a local server.
DEFAULT_LOCAL_HISTORY_TURNS = 4

_backend = OPENAI
_base_url: Optional[str] = None
_model: Optional[str] = None
_timeout: Optional[float] = None
_tool_calls = False


def configure(backend: str = OPENAI, base_url: Optional[str] = None, model: Optional[str] = None,
              timeout: Optional[float] = None, tool_calls: bool = False) -> None:
    """`backend` is "openai" or "local"; `base_url` defaults to DEFAULT_LOCAL_BASE_URL for a local
    server (and to OpenAI's own for "openai"); `model` to the caller's default; `timeout`
    (seconds per request) to the caller's default. `tool_calls` (local only): the server does
    tool calling (llama-server started with --jinja), so the event extractor uses a forced tool
    call, as with OpenAI, instead of a JSON schema."""
    global _backend, _base_url, _model, _timeout, _tool_calls
    backend = (backend or OPENAI).strip().lower()
    if backend not in BACKENDS:
        raise ValueError(f"Unknown LLM backend {backend!r}, expected one of {BACKENDS}")
    _backend = backend
    _base_url = base_url or (DEFAULT_LOCAL_BASE_URL if backend == LOCAL else None)
    _model = model or None
    _timeout = timeout
    _tool_calls = bool(tool_calls)


def backend() -> str:
    return _backend


def is_local() -> bool:
    return _backend == LOCAL


def base_url() -> Optional[str]:
    return _base_url


def model(default: str) -> str:
    """The configured model, else `default` for OpenAI or DEFAULT_LOCAL_MODEL for a local server."""
    return _model or (DEFAULT_LOCAL_MODEL if is_local() else default)


def timeout(default: float) -> float:
    return _timeout or default


def api_key() -> str:
    """OPENAI_API_KEY; a local server needs none, but the SDK wants a non-empty one."""
    key = os.environ.get("OPENAI_API_KEY")
    if is_local():
        return key or "no-key"
    if not key:
        raise SystemExit("OPENAI_API_KEY environment variable not set.")
    return key


def client(default_timeout: float) -> OpenAI:
    """A new client for the configured backend. max_retries=0: a timeout surfaces at once, see
    chat_sessions._call_openai()."""
    return OpenAI(api_key=api_key(), base_url=base_url(), timeout=timeout(default_timeout),
                  max_retries=0)


def extraction_tool_calls() -> bool:
    """Whether the event extractor asks for a tool call (OpenAI, or llama-server with --jinja)
    rather than for JSON following its schema."""
    return not is_local() or _tool_calls


def decision_sampling() -> dict:
    """Extra request parameters for a call that DECIDES something -- event extraction, and the
    yes/no or JSON classifiers -- as opposed to one that writes a reply. llama-server samples at
    a temperature of ~0.8 by default, which made such a classifier flip now and then (YES on "At
    lunch, 1pm" in 2 of 20 tries); 0 gives the same answer for the same input. Nothing extra for
    OpenAI, whose reasoning models reject a temperature."""
    return {"temperature": 0} if is_local() else {}


def settings() -> tuple:
    """Everything a client depends on, so a cached client can tell it is out of date."""
    return _backend, _base_url, _timeout


def describe() -> str:
    """E.g. "local at http://localhost:8080/v1 (serving qwen3-30b...)" or "openai, model gpt-5.1"."""
    if not is_local():
        return f"openai, model {model('gpt-5.1')}"
    try:
        return f"local at {_base_url} (serving {served_model()})"
    except Exception as e:
        return f"local at {_base_url} (NOT REACHABLE: {e})"


def served_model(request_timeout: float = 5.0) -> str:
    """The model the local server serves, from its /models; raises when it is not reachable,
    e.g. while llama-server is still downloading or loading the model."""
    with urllib.request.urlopen(f"{_base_url.rstrip('/')}/models", timeout=request_timeout) as response:
        models = json.load(response).get("data", [])
    return ", ".join(m.get("id", "?") for m in models) or "no model"


def structured_output(name: str, schema: dict) -> dict:
    """The response_format for output that must follow `schema`, for a backend without tool
    calling: llama-server turns it into a grammar the model cannot step outside of."""
    return {"type": "json_schema", "json_schema": {"name": name, "schema": schema, "strict": True}}
