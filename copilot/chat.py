"""Viewer chat: quick commands (free, instant) first, then Claude Code headless with our MCP server.

Option C (quick): fixed phrases are handled locally without any LLM, e.g.
  "take me to M-006", "show P-005", "start tour", "tour past CV-003 and M-006",
  "which assets need attention", "status of P-002", "help".
Option A (Claude): anything else runs `claude -p` (the user's own Claude Code login, e.g. Pro/Max
subscription) with ONLY our MCP server (.mcp.json, --strict-mcp-config) and only its tools allowed.
Its viewer tools (show/walk/tour) reach the running viewer through out/viewer_cmd.jsonl.
"""
from __future__ import annotations

import glob
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable

from pipeline.common import ROOT

Cfg = dict[str, Any]
TAG_RE = re.compile(r"\b(OBJ-\d{4}|ASSET-\d{3}|[A-Z]{1,3}-\d{3})\b", re.I)
HELP = ("**Quick commands** (instant, free): `take me to M-006` · `show P-005` · `start tour` · "
        "`tour past CV-003 and M-006` · `which assets need attention` · `status of P-002`.  \n"
        "Anything else is answered by **Claude** with the copilot tools (~10-30 s).")


# ---------- option C: quick commands ----------

def tags_in(text: str) -> list[str]:
    return [t.upper() for t in TAG_RE.findall(text)]


WALK_RE = re.compile(r"\b(?:take me|go|walk|navigate|bring me)\b.*?\bto\b\s+(?:the\s+|a\s+|an\s+)?(?P<x>[a-z/ -]+?)\s*[?.!]*$")
SHOW_RE = re.compile(r"\b(?:show(?: me)?|where is|find|locate|highlight)\b\s+(?:the\s+|a\s+|an\s+)?(?P<x>[a-z/ -]+?)\s*[?.!]*$")


def label_target(text: str, labels: list[str] | None) -> tuple[str, str] | None:
    """('walk'|'show', phrase) when the request names a known object label, e.g. 'take me to the tank'."""
    if not labels:
        return None
    t = text.strip().lower()

    def words(s: str) -> set[str]:
        return {w.rstrip("s") for w in re.split(r"[\s/]+", s.lower()) if w}

    for cmd, rx in (("walk", WALK_RE), ("show", SHOW_RE)):
        m = rx.search(t)
        if not m:
            continue
        phrase = m.group("x").strip()
        ask = words(phrase) - {"nearest", "closest"}
        # only when every word belongs to one label ("tank", "tank/vessel", "electrical cabinet");
        # qualifiers like "worst pump" or "overheating motor" need reasoning -> Claude
        if ask and any(ask <= words(lb) for lb in labels):
            return cmd, phrase
    return None


def parse_quick(text: str, labels: list[str] | None = None) -> dict | None:
    """Recognise a fixed command; None means 'ask Claude'. `labels` = known vision labels in the scene."""
    t = text.strip().lower()
    tags = tags_in(text)
    if not tags:
        lt = label_target(text, labels)
        if lt:
            return {"cmd": lt[0], "target": lt[1]}
    if t in ("help", "?", "commands"):
        return {"cmd": "help"}
    if re.search(r"\b(tour|inspection round)\b", t) and re.search(r"\b(start|run|begin|do|tour past|tour via)\b", t):
        return {"cmd": "tour", "stops": [x for x in tags if not x.startswith("OBJ")]}
    if len(tags) == 1 and re.search(r"\b(take me|go|walk|navigate|bring me)\b", t):
        return {"cmd": "walk", "target": tags[0]}
    if len(tags) == 1 and re.search(r"\b(show|where is|find|locate|highlight)\b", t):
        return {"cmd": "show", "target": tags[0]}
    if len(tags) == 1 and re.search(r"\b(status|health|condition|state)\b", t) and len(t.split()) <= 6:
        return {"cmd": "status", "target": tags[0]}
    if not tags and re.search(r"\b(which|list|what)\b", t) and re.search(r"\b(attention|faulty|urgent|degraded|problem)", t) \
            and len(t.split()) <= 8:
        return {"cmd": "attention"}
    return None


def run_quick(q: dict, act: Callable[[str, str], str]) -> str:
    """Execute a quick command. `act(cmd, target)` performs viewer actions and returns a message."""
    from copilot import tools as T
    if q["cmd"] == "help":
        return HELP
    if q["cmd"] in ("walk", "show"):
        return act(q["cmd"], q["target"])
    if q["cmd"] == "tour":
        return act("tour", ",".join(q["stops"]))
    if q["cmd"] == "status":
        a = T.get_asset(q["target"])
        if not a["ok"]:
            return f"⚠️ {a['message']}"
        if not a["asset_record"]:
            return f"**{a['object_id']}** ({a['label']}): no asset record (not in the Excel pack)."
        return (f"**{a['tag']}** ({a['asset_class']}): {a['condition_state']}, health **{a['health_score']}**, "
                f"{a['issue_code']}, urgency {a['urgency']}.  \nNext check: {a['recommended_next_check']} "
                f"[{a['knowledge_doc_id']}]")
    if q["cmd"] == "attention":
        rows = []
        for cond in ("attention_required", "degraded"):
            for a in T.search_assets(condition=cond)["assets"]:
                rows.append(f"- **{a['tag']}** {a['asset_class']}: {cond}, health {a['health_score']}, {a['issue_code']}")
        return "**Needs attention** (most urgent first):\n" + "\n".join(rows) if rows else "Nothing flagged."
    return HELP


# ---------- option A: Claude Code headless + our MCP server ----------

def find_claude(cc: Cfg) -> str | None:
    """Configured path, `claude` on PATH, or the binary bundled with the Claude Code editor extension."""
    p = cc.get("claude_path")
    if p and p != "auto" and Path(p).exists():
        return p
    on_path = shutil.which("claude")
    if on_path:
        return on_path
    home = Path.home()
    pats = [str(home / d / "extensions" / "anthropic.claude-code-*" / "resources" / "native-binary" / "claude.exe")
            for d in (".antigravity-ide", ".vscode", ".vscode-insiders", ".cursor", ".windsurf")]
    hits = sorted((h for pat in pats for h in glob.glob(pat)), key=os.path.getmtime)
    return hits[-1] if hits else None


SETUP_HELP = ("**To enable Claude answers on this computer:** install Claude Code (https://claude.com/claude-code, "
              "or the Claude Code extension in VS Code / Antigravity), then log in once: open a terminal, run "
              "`claude`, type `/login` and sign in with a Claude Pro/Max account. Restart the viewer afterwards. "
              "If Claude Code is installed somewhere unusual, set `chat.claude_path` in config.yaml.")


def claude_status(cc: Cfg) -> dict:
    """Free check (no AI request): is Claude Code installed and logged in, and how (subscription vs API key)?"""
    exe = find_claude(cc)
    if not exe:
        return {"ready": False, "installed": False,
                "message": "⚠️ **Claude Code is not installed** on this computer: only quick commands work "
                           "(type `help`).  \n" + SETUP_HELP}
    try:
        p = subprocess.run([exe, "auth", "status"], capture_output=True, text=True, encoding="utf-8", timeout=30)
        st = json.loads(p.stdout)
    except (subprocess.SubprocessError, json.JSONDecodeError, OSError) as e:
        return {"ready": True, "installed": True, "path": exe,
                "message": f"🤖 Claude Code found (login status unknown: {type(e).__name__})."}
    if not st.get("loggedIn"):
        return {"ready": False, "installed": True, "path": exe,
                "message": "⚠️ **Claude Code is installed but not logged in**: only quick commands work "
                           "(type `help`).  \n" + SETUP_HELP}
    method = st.get("authMethod", "?")
    billed = method not in ("claude.ai",)
    note = ("⚠️ logged in with an **API key**: each Claude question is billed per use."
            if billed else "logged in with a Claude subscription (no extra cost, plan usage limits apply).")
    return {"ready": True, "installed": True, "path": exe, "auth_method": method,
            "message": f"🤖 **Claude ready**: {note}"}


def ask_claude(question: str, cc: Cfg) -> str:
    st = cc.get("_status") or claude_status(cc)
    if not st["ready"]:
        return st["message"]
    exe = st["path"]
    cmd = [exe, "-p", question, "--mcp-config", str(ROOT / ".mcp.json"), "--strict-mcp-config",
           "--allowedTools", "mcp__valmet-copilot", "--output-format", "json",
           "--append-system-prompt", cc["system_prompt"]]
    if cc.get("model"):
        cmd += ["--model", cc["model"]]
    try:
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                           timeout=cc["timeout_s"])
    except subprocess.TimeoutExpired:
        return f"⚠️ Claude did not answer within {cc['timeout_s']} s. Try a shorter question or a quick command."
    try:
        d = json.loads(p.stdout)
    except json.JSONDecodeError:
        return f"⚠️ Claude Code failed: {(p.stderr or p.stdout)[:300]}"
    if d.get("is_error"):
        res = str(d.get("result"))
        if "login" in res.lower() or "logged in" in res.lower() or "api key" in res.lower():
            return "⚠️ **Claude Code is not logged in** on this computer: only quick commands work.  \n" + SETUP_HELP
        return f"⚠️ Claude Code error: {res[:300]}"
    return f"{d.get('result', '').strip()}\n\n_Claude via MCP · {d.get('num_turns')} turns_"


def answer(question: str, cc: Cfg, act: Callable[[str, str], str]) -> tuple[str, str]:
    """Returns (answer markdown, route) where route is 'quick' or 'claude'."""
    from copilot.tools import scene
    labels = sorted({n["label"] for n in scene()["objects"] if n["kind"] == "object"})
    q = parse_quick(question, labels) if cc.get("quick_commands", True) else None
    if q is not None:
        return run_quick(q, act), "quick"
    return ask_claude(question, cc), "claude"
