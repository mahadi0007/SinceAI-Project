"""Free diagnosis for training and pre-inspection: the cached LLM answer for the same evidence, else the
template. Never calls the LLM and writes no output files (unlike copilot.agent.diagnose)."""
from __future__ import annotations

import json
from typing import Any

from copilot.agent import cache_path, describe_ids, template_diagnosis, validate_citations

Cfg = dict[str, Any]


def offline_diagnosis(pack: dict, cfg: Cfg) -> dict:
    cp = cache_path(cfg, pack["asset"]["tag"], None, pack)
    if cp.exists():
        c = json.loads(cp.read_text(encoding="utf-8"))
        d, source, kind = c["raw"], f"{c['model']} (cached answer from {c['at']}, same evidence)", "cached_llm"
    else:
        d, source, kind = template_diagnosis(pack), "template (rules over the Excel evidence)", "template"
    d = validate_citations(json.loads(json.dumps(d)), pack)
    d["meta"] = {"source": source, "kind": kind, "latency_s": 0.0, "question": None, "scene": pack["scene"],
                 "status": pack["current_state"], "sources": describe_ids(pack, d["citation_check"]["cited_ids"])}
    return d
