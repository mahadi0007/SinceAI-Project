"""Shared LLM access with a provider-aware model fallback chain.

Chain entries in config copilot.models are "<provider>:<model>":
  featherless:Qwen/Qwen3-VL-30B-A3B-Instruct   OpenAI-compatible API (open models, no daily quota)
  gemini:gemini-3.5-flash                      Google Gemini (free tier ~20 requests/day/model)
A plain name without a prefix means gemini (backwards compatible).

Callers pass Gemini-style contents (a string, or a list of strings and google.genai types.Part
images); they are converted per provider. generate() tries each entry in order and moves on for
quota (429), overload (503), gated (403) or retired (404) models. Featherless 429s ("too many
concurrent requests") are retried with a short wait before moving on. Returns (text, "provider:model").
The last fallback ("saved answers") lives in the callers: diagnosis cache and label cache.
"""
from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from dotenv import load_dotenv

from pipeline.common import ROOT

Cfg = dict[str, Any]
SKIP_CODES = ("429", "503", "404", "403", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "NOT_FOUND", "gated")
_dead: dict[str, float] = {}          # entry -> time it failed with quota/retired/gated (skip for a while)


class ProviderError(RuntimeError):
    """An HTTP/API error with the status code in the message (so SKIP_CODES can classify it)."""


def _env(name: str) -> str:
    load_dotenv(ROOT / ".env")
    key = os.getenv(name, "")
    if not key:
        raise ProviderError(f"{name} not set in .env")
    return key


def split_entry(entry: str) -> tuple[str, str]:
    if ":" in entry and entry.split(":", 1)[0] in ("featherless", "gemini"):
        p, m = entry.split(":", 1)
        return p, m
    return "gemini", entry


# ---------- providers ----------

def _gemini(model: str, contents: Any, cc: Cfg, json_mode: bool, temperature: float) -> str:
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=_env(cc["api_key_env"]))
    kw: dict[str, Any] = {"temperature": temperature}
    if json_mode:
        kw["response_mime_type"] = "application/json"
    r = client.models.generate_content(model=model, contents=contents, config=types.GenerateContentConfig(**kw))
    return r.text or ""


def to_openai_content(contents: Any) -> list[dict]:
    """Gemini-style contents -> OpenAI chat 'content' parts (text + base64 image_url)."""
    items = contents if isinstance(contents, list) else [contents]
    parts: list[dict] = []
    for it in items:
        if isinstance(it, str):
            parts.append({"type": "text", "text": it})
            continue
        blob = getattr(it, "inline_data", None)
        if blob is not None and blob.data:
            b64 = base64.b64encode(blob.data).decode()
            parts.append({"type": "image_url", "image_url": {"url": f"data:{blob.mime_type};base64,{b64}"}})
        elif getattr(it, "text", None):
            parts.append({"type": "text", "text": it.text})
    return parts


def _featherless(model: str, contents: Any, cc: Cfg, json_mode: bool, temperature: float) -> str:
    fc = cc["featherless"]
    body = {"model": model, "temperature": temperature, "max_tokens": fc["max_tokens"],
            "messages": [{"role": "user", "content": to_openai_content(contents)}]}
    req = urllib.request.Request(fc["base_url"].rstrip("/") + "/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {_env(fc['api_key_env'])}",
                                          "Content-Type": "application/json"})
    for attempt in range(fc["retries_429"] + 1):
        try:
            with urllib.request.urlopen(req, timeout=fc["timeout_s"]) as r:
                out = json.loads(r.read())
            return out["choices"][0]["message"]["content"] or ""
        except urllib.error.HTTPError as e:
            msg = e.read()[:300].decode(errors="replace")
            if e.code == 429 and attempt < fc["retries_429"]:      # concurrency limit: wait, retry
                time.sleep(fc["backoff_s"] * (attempt + 1))
                continue
            raise ProviderError(f"{e.code} {msg}") from e
        except (urllib.error.URLError, TimeoutError) as e:
            raise ProviderError(f"503 network/timeout: {e}") from e
    raise ProviderError("429 retries exhausted")


PROVIDERS = {"gemini": _gemini, "featherless": _featherless}


# ---------- chain ----------

def generate(contents: Any, cc: Cfg, json_mode: bool = True, temperature: float | None = None) -> tuple[str, str]:
    temp = cc["temperature"] if temperature is None else temperature
    errors = []
    for entry in cc["models"]:
        if time.time() - _dead.get(entry, 0) < cc["dead_model_skip_s"]:
            continue
        provider, model = split_entry(entry)
        try:
            text = PROVIDERS[provider](model, contents, cc, json_mode, temp)
            if text.strip():
                return text, f"{provider}:{model}"
            errors.append(f"{entry}: empty response")
        except Exception as e:  # noqa: BLE001 - classify and fall through to next model
            s = str(e)
            errors.append(f"{entry}: {s[:100]}")
            if "not set in .env" in s or any(c in s for c in SKIP_CODES):
                if "503" not in s and "UNAVAILABLE" not in s:
                    _dead[entry] = time.time()       # quota / retired / gated / no key: skip for a while
                continue
            raise
    raise RuntimeError("all models failed: " + " | ".join(errors))
