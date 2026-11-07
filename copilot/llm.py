"""Provider-neutral LLM access (text + images) with a hard paid-request budget and a usage ledger.

Provider: Featherless (OpenAI-compatible, https://api.featherless.ai/v1). Key from .env
(FEATHERLESS_API_KEY by default; never printed). One model, NO automatic retry and NO model
cascade: every request that leaves the machine is a paid attempt and is counted BEFORE it is sent.

Budget: copilot.budget.max_paid_requests caps the total number of paid requests ever made from
this repo (ledger file, survives restarts). Raise it deliberately in config.yaml to spend more.
Offline: env VALMET_OFFLINE=1 (or copilot.offline: true) makes every call raise LLMUnavailable,
so callers fall back to caches/templates.

generate(prompt, cc, images=[png bytes], json_mode=..., purpose=...) -> (text, model)
"""
from __future__ import annotations

import base64
import json
import os
import re
import threading
import time
from typing import Any

from dotenv import load_dotenv

from pipeline.common import ROOT

Cfg = dict[str, Any]
_lock = threading.Lock()


class LLMUnavailable(RuntimeError):
    """Offline mode, missing key or exhausted budget: caller must use its offline path."""


class BudgetExceeded(LLMUnavailable):
    pass


def ledger_path(cc: Cfg):
    return ROOT / cc.get("budget", {}).get("ledger", "llm_ledger.jsonl")


def ledger(cc: Cfg) -> list[dict]:
    p = ledger_path(cc)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def usage_summary(cc: Cfg) -> dict:
    rows = ledger(cc)
    paid = [r for r in rows if r.get("sent")]
    return {"paid_requests": len(paid),
            "max_paid_requests": cc.get("budget", {}).get("max_paid_requests", 0),
            "prompt_tokens": sum(r.get("prompt_tokens") or 0 for r in paid),
            "completion_tokens": sum(r.get("completion_tokens") or 0 for r in paid),
            "est_cost_usd": round(sum(r.get("est_cost_usd") or 0 for r in paid), 5),
            "by_purpose": {p: sum(1 for r in paid if r.get("purpose") == p)
                           for p in sorted({r.get("purpose") for r in paid})}}


def _append(cc: Cfg, rec: dict) -> None:
    with open(ledger_path(cc), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def _reserve(cc: Cfg, purpose: str, model: str) -> int:
    """Count the request before sending it; refuse when the budget is spent."""
    with _lock:
        n = sum(1 for r in ledger(cc) if r.get("sent"))
        cap = int(cc.get("budget", {}).get("max_paid_requests", 0))
        if n >= cap:
            raise BudgetExceeded(f"paid-request budget spent ({n}/{cap}); raise copilot.budget.max_paid_requests")
        _append(cc, {"n": n + 1, "sent": True, "purpose": purpose, "model": model,
                     "at": time.strftime("%Y-%m-%d %H:%M:%S"), "status": "pending"})
        return n + 1


def _finish(cc: Cfg, n: int, **kw: Any) -> None:
    with _lock:
        rows = ledger(cc)
        for r in rows:
            if r.get("n") == n and r.get("status") == "pending":
                r.update(kw)
        ledger_path(cc).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def offline(cc: Cfg) -> bool:
    return os.environ.get("VALMET_OFFLINE") == "1" or bool(cc.get("offline"))


def _client(cc: Cfg) -> Any:
    from openai import OpenAI
    load_dotenv(ROOT / ".env")
    key = os.getenv(cc["api_key_env"], "")
    if not key:
        raise LLMUnavailable(f"{cc['api_key_env']} not set in .env")
    # max_retries=0: the SDK must never re-send a paid request on its own
    return OpenAI(base_url=cc["base_url"], api_key=key, max_retries=0, timeout=cc.get("timeout_s", 180))


def image_part(png: bytes) -> dict:
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}}


THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def generate(prompt: str | list[Any], cc: Cfg, json_mode: bool = True, temperature: float | None = None,
             images: list[bytes] | None = None, purpose: str = "other", max_tokens: int | None = None,
             system: str | None = None) -> tuple[str, str]:
    """One paid request (or LLMUnavailable). `prompt` may be a str or a list of str/png-bytes parts
    (interleaved text and images, in order)."""
    if offline(cc):
        raise LLMUnavailable("offline mode (VALMET_OFFLINE=1 / copilot.offline)")
    model = cc["model"]
    parts: list[dict] = []
    for p in (prompt if isinstance(prompt, list) else [prompt]):
        parts.append(image_part(p) if isinstance(p, (bytes, bytearray)) else {"type": "text", "text": str(p)})
    for im in images or []:
        parts.append(image_part(im))
    msgs: list[dict] = ([{"role": "system", "content": system}] if system else []) + \
        [{"role": "user", "content": parts}]
    kw: dict[str, Any] = {"model": model, "messages": msgs,
                          "temperature": cc["temperature"] if temperature is None else temperature,
                          "max_tokens": max_tokens or cc.get("max_tokens", 2000)}
    if cc.get("disable_thinking", True):
        kw["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    client = _client(cc)
    n = _reserve(cc, purpose, model)
    t0 = time.time()
    try:
        r = client.chat.completions.create(**kw)
    except Exception as e:  # noqa: BLE001 - recorded, never retried
        _finish(cc, n, status="error", error=str(e)[:300], latency_s=round(time.time() - t0, 1))
        raise
    u = getattr(r, "usage", None)
    pt, ct = (getattr(u, "prompt_tokens", None), getattr(u, "completion_tokens", None)) if u else (None, None)
    price = cc.get("pricing_usd_per_mtok", {})
    cost = ((pt or 0) * price.get("input", 0) + (ct or 0) * price.get("output", 0)) / 1e6
    text = (r.choices[0].message.content or "") if r.choices else ""
    text = THINK_RE.sub("", text).strip()
    _finish(cc, n, status="ok" if text else "empty", prompt_tokens=pt, completion_tokens=ct,
            est_cost_usd=round(cost, 6), latency_s=round(time.time() - t0, 1),
            finish_reason=r.choices[0].finish_reason if r.choices else None, n_images=sum(
                1 for p in parts if p["type"] == "image_url"))
    if not text:
        raise RuntimeError(f"{model}: empty response")
    return text, model
