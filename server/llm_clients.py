"""
JUROR · server-side LLM clients.

Same four providers as the in-browser fallback (Anthropic, OpenAI, Google,
DeepSeek) but here the key sits in environment variables — the browser never
sees it.

For each provider we expose ``call(provider, system, user, max_tokens) -> (text, tokens)``.

Environment variables read at startup:

    ANTHROPIC_API_KEY   |  ANTHROPIC_MODEL    (default claude-opus-4-6)
    OPENAI_API_KEY      |  OPENAI_MODEL       (default gpt-5.4)
    GOOGLE_API_KEY      |  GOOGLE_MODEL       (default gemini-3.1-pro)
    DEEPSEEK_API_KEY    |  DEEPSEEK_MODEL     (default deepseek-chat)

If a provider key is missing, the client raises a clean RuntimeError so the
caller can surface the right error to the UI.
"""

from __future__ import annotations

import os
import logging
from typing import Optional, Tuple

import httpx

log = logging.getLogger("juror.llm")

REQUEST_TIMEOUT = 60.0


DEFAULTS = {
    "anthropic": ("ANTHROPIC_API_KEY", "ANTHROPIC_MODEL", "claude-opus-4-6"),
    "openai":    ("OPENAI_API_KEY",    "OPENAI_MODEL",    "gpt-5.4"),
    "google":    ("GOOGLE_API_KEY",    "GOOGLE_MODEL",    "gemini-3.1-pro"),
    "deepseek":  ("DEEPSEEK_API_KEY",  "DEEPSEEK_MODEL",  "deepseek-chat"),
}


def _resolve(provider: str) -> Tuple[str, str]:
    if provider not in DEFAULTS:
        raise ValueError(f"unknown provider: {provider}")
    key_env, model_env, default_model = DEFAULTS[provider]
    key   = os.environ.get(key_env, "").strip()
    model = os.environ.get(model_env, default_model).strip() or default_model
    if not key:
        raise RuntimeError(
            f"Provider '{provider}' has no key set. "
            f"Set the {key_env} environment variable on the server."
        )
    return key, model


async def call_anthropic(system: str, user: str, max_tokens: int) -> Tuple[str, int]:
    key, model = _resolve("anthropic")
    body = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    headers = {
        "x-api-key": key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as cli:
        r = await cli.post("https://api.anthropic.com/v1/messages",
                           headers=headers, json=body)
    r.raise_for_status()
    j = r.json()
    text = j["content"][0]["text"] if j.get("content") else ""
    usage = j.get("usage", {})
    tokens = int(usage.get("input_tokens", 0)) + int(usage.get("output_tokens", 0))
    return text, tokens


async def call_openai(system: str, user: str, max_tokens: int) -> Tuple[str, int]:
    key, model = _resolve("openai")
    body = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    }
    headers = {"Authorization": f"Bearer {key}", "content-type": "application/json"}
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as cli:
        r = await cli.post("https://api.openai.com/v1/chat/completions",
                           headers=headers, json=body)
    r.raise_for_status()
    j = r.json()
    text = j["choices"][0]["message"]["content"] or ""
    tokens = int(j.get("usage", {}).get("total_tokens", 0))
    return text, tokens


async def call_google(system: str, user: str, max_tokens: int) -> Tuple[str, int]:
    key, model = _resolve("google")
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/{model}"
           f":generateContent?key={key}")
    body = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents":          [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig":  {"temperature": 0.0, "maxOutputTokens": max_tokens},
    }
    headers = {"content-type": "application/json"}
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as cli:
        r = await cli.post(url, headers=headers, json=body)
    r.raise_for_status()
    j = r.json()
    text = ""
    try:
        text = j["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        pass
    tokens = int(j.get("usageMetadata", {}).get("totalTokenCount", 0))
    return text, tokens


async def call_deepseek(system: str, user: str, max_tokens: int) -> Tuple[str, int]:
    key, model = _resolve("deepseek")
    body = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    }
    headers = {"Authorization": f"Bearer {key}", "content-type": "application/json"}
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as cli:
        r = await cli.post("https://api.deepseek.com/v1/chat/completions",
                           headers=headers, json=body)
    r.raise_for_status()
    j = r.json()
    text = j["choices"][0]["message"]["content"] or ""
    tokens = int(j.get("usage", {}).get("total_tokens", 0))
    return text, tokens


PROVIDERS = {
    "anthropic": call_anthropic,
    "openai":    call_openai,
    "google":    call_google,
    "deepseek":  call_deepseek,
}


def available_providers() -> dict:
    """Return which providers currently have a key configured."""
    out = {}
    for name, (key_env, _, _) in DEFAULTS.items():
        out[name] = bool(os.environ.get(key_env))
    return out


async def call(provider: str, system: str, user: str,
               max_tokens: int = 220) -> Tuple[str, int]:
    """Dispatch to the right provider client."""
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider: {provider}")
    return await PROVIDERS[provider](system, user, max_tokens)
