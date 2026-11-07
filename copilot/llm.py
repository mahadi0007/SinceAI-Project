"""Shared Gemini access with a model fallback chain.

Free-tier keys have small per-model daily quotas (e.g. 20 requests/day) and models get
overloaded (503) or retired (404). generate() tries each model in config copilot.models
in order and moves on for those errors. Returns (text, model_used).
"""
from __future__ import annotations

import os
import time
from typing import Any

from dotenv import load_dotenv

from pipeline.common import ROOT

Cfg = dict[str, Any]
SKIP_CODES = ("429", "503", "404", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "NOT_FOUND")
_dead: dict[str, float] = {}          # model -> time it failed with quota/retired (skip for a while)


def _client(cc: Cfg) -> Any:
    from google import genai
    load_dotenv(ROOT / ".env")
    key = os.getenv(cc["api_key_env"], "")
    if not key:
        raise RuntimeError(f"{cc['api_key_env']} not set in .env")
    return genai.Client(api_key=key)


def generate(contents: Any, cc: Cfg, json_mode: bool = True, temperature: float | None = None) -> tuple[str, str]:
    from google.genai import types
    client = _client(cc)
    errors = []
    for model in cc["models"]:
        if time.time() - _dead.get(model, 0) < cc["dead_model_skip_s"]:
            continue
        kw: dict[str, Any] = {"temperature": cc["temperature"] if temperature is None else temperature}
        if json_mode:
            kw["response_mime_type"] = "application/json"
        try:
            r = client.models.generate_content(model=model, contents=contents,
                                               config=types.GenerateContentConfig(**kw))
            if r.text:
                return r.text, model
            errors.append(f"{model}: empty response")
        except Exception as e:  # noqa: BLE001 - classify and fall through to next model
            s = str(e)
            errors.append(f"{model}: {s[:80]}")
            if any(c in s for c in SKIP_CODES):
                if "503" not in s and "UNAVAILABLE" not in s:
                    _dead[model] = time.time()       # quota / retired: skip it for a while
                continue
            raise
    raise RuntimeError("all models failed: " + " | ".join(errors))
