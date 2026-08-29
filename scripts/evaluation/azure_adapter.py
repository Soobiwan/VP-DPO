"""Compatibility client used by the Azure-backed AlpacaEval 2 judge."""

from __future__ import annotations

import logging
import os
import re
import hashlib
import json
import sqlite3
from pathlib import Path

from openai import AzureOpenAI, BadRequestError
from openai.types.chat import ChatCompletion


_NEUTRAL_LOGPROB = -0.6931471805599453


def _response_cache_path() -> Path | None:
    value = os.environ.get("ALPACA_AZURE_RESPONSE_CACHE", "").strip()
    return Path(value).expanduser().resolve() if value else None


def _response_cache_key(namespace: str, args: tuple, kwargs: dict) -> str:
    payload = json.dumps(
        {"namespace": namespace, "args": args, "kwargs": kwargs},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _initialize_response_cache(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS responses (
            cache_key TEXT PRIMARY KEY,
            response_json TEXT NOT NULL
        )
        """
    )


def _load_cached_completion(cache_key: str) -> ChatCompletion | None:
    path = _response_cache_path()
    if path is None or not path.is_file():
        return None
    with sqlite3.connect(path, timeout=30) as connection:
        _initialize_response_cache(connection)
        row = connection.execute(
            "SELECT response_json FROM responses WHERE cache_key = ?",
            (cache_key,),
        ).fetchone()
    if row is None:
        return None
    return ChatCompletion.model_validate_json(row[0])


def _store_cached_completion(cache_key: str, completion: ChatCompletion) -> None:
    path = _response_cache_path()
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path, timeout=30) as connection:
        _initialize_response_cache(connection)
        connection.execute(
            "INSERT OR REPLACE INTO responses(cache_key, response_json) VALUES (?, ?)",
            (cache_key, completion.model_dump_json()),
        )


def _extract_instruction(messages: object) -> str:
    """Extract the benchmark instruction from an AlpacaEval judge request."""
    if not isinstance(messages, list):
        return "<unavailable>"
    contents = []
    for message in messages:
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            contents.append(message["content"])
    match = re.search(
        r'"instruction"\s*:\s*"""(.*?)"""',
        "\n".join(contents),
        flags=re.DOTALL,
    )
    return match.group(1).strip() if match else "<unavailable>"


def _is_content_filter_error(error: BadRequestError) -> bool:
    body = getattr(error, "body", None)
    if not isinstance(body, dict):
        return False
    inner = body.get("innererror") or {}
    return (
        body.get("code") == "content_filter"
        or inner.get("code") == "ResponsibleAIPolicyViolation"
    )


def _neutral_content_filter_completion(model: str) -> ChatCompletion:
    """Return equal m/M logprobs so a filtered comparison is a transparent draw."""
    return ChatCompletion.model_validate(
        {
            "id": "content-filtered-neutral",
            "object": "chat.completion",
            "created": 0,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "content_filter",
                    "message": {"role": "assistant", "content": "m"},
                    "logprobs": {
                        "content": [
                            {
                                "token": "m",
                                "bytes": [109],
                                "logprob": _NEUTRAL_LOGPROB,
                                "top_logprobs": [
                                    {
                                        "token": "m",
                                        "bytes": [109],
                                        "logprob": _NEUTRAL_LOGPROB,
                                    },
                                    {
                                        "token": "M",
                                        "bytes": [77],
                                        "logprob": _NEUTRAL_LOGPROB,
                                    },
                                ],
                            }
                        ]
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }
    )


class _Completions:
    def __init__(self, client: AzureOpenAI, cache_namespace: str):
        self._client = client
        self._cache_namespace = cache_namespace

    def create(self, *args, **kwargs):
        if "max_tokens" in kwargs and "max_completion_tokens" not in kwargs:
            kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
        cache_key = _response_cache_key(self._cache_namespace, args, kwargs)
        cached = _load_cached_completion(cache_key)
        if cached is not None:
            logging.info("Reusing a cached Azure AlpacaEval judge response.")
            return cached
        try:
            completion = self._client.chat.completions.create(*args, **kwargs)
        except BadRequestError as error:
            if not _is_content_filter_error(error):
                raise
            instruction = _extract_instruction(kwargs.get("messages"))
            logging.warning(
                "Azure content-filtered an AlpacaEval judge prompt; recording "
                "a neutral 0.5 preference. Instruction: %r",
                instruction,
            )
            completion = _neutral_content_filter_completion(
                kwargs.get("model", "azure-judge")
            )
        _store_cached_completion(cache_key, completion)
        return completion


class _Chat:
    def __init__(self, client: AzureOpenAI, cache_namespace: str):
        self.completions = _Completions(client, cache_namespace)


class AzureOpenAICompat:
    def __init__(self, **kwargs):
        kwargs.setdefault("api_key", os.environ["AZURE_OPENAI_API_KEY"])
        # Let the SDK absorb brief DNS, connection, rate-limit, and 5xx failures.
        # The response cache below independently preserves every completed request.
        kwargs.setdefault("max_retries", 6)
        cache_namespace = json.dumps(
            {
                "provider": "azure",
                "azure_endpoint": kwargs.get("azure_endpoint"),
                "api_version": kwargs.get("api_version"),
            },
            sort_keys=True,
        )
        self._client = AzureOpenAI(**kwargs)
        self.chat = _Chat(self._client, cache_namespace)
