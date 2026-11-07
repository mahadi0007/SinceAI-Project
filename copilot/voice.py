"""Spoken copilot briefing: ElevenLabs text-to-speech over the validated diagnosis.

Grounding: the script is built only from fields of diagnose()'s validated output
(condition_summary + next_safe_checks), never rephrased by another model. Evidence IDs are not spoken.
Audio is cached in out/voice_cache/ by a hash of voice + model + text, so a repeat costs no API call.
Any failure (no key, network, API error) returns None and the viewer stays text-only.

The voice does not care which LLM wrote the diagnosis (Featherless, Gemini, template): any dict with
condition_summary + next_safe_checks[{text}] works. To use it from other code:
    from copilot.voice import briefing, speak
    text, mp3 = briefing(diagnosis_dict, "P-002", cfg)   # script + mp3 bytes (or None)
    path = speak("Any sentence.")                        # mp3 file path (or None)

Run:
    python -m copilot.voice --dummy --play     # sample P-002 diagnosis: no LLM, no Excel, no pipeline
    python -m copilot.voice --text "Hello" --play
    python -m copilot.voice P-002              # real diagnosis (out/diagnosis_P-002.json, else diagnose())
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from pipeline.common import ROOT, load_config, out_path

Cfg = dict[str, Any]
API = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}?output_format={fmt}"
LOTO_RE = re.compile(r"\b(LOTO|lockout|lock-out|permit)", re.I)


CODE_RE = re.compile(r"\b[A-Za-z]+(?:_[A-Za-z]+)+\b")      # attention_required, CAVITATION_RISK


def _say(text: str) -> str:
    """Speakable wording only (same content): codes with underscores become words, first letter capital."""
    t = CODE_RE.sub(lambda m: m.group(0).replace("_", " ").lower(), text.strip())
    return t[:1].upper() + t[1:]


def speech_text(d: dict, tag: str, vc: Cfg) -> str:
    """'Arrived at P-002. <summary> First checks: 1. … 2. … <safety line>'."""
    raw = [c.get("text", "") if isinstance(c, dict) else str(c) for c in d.get("next_safe_checks") or []]
    checks = [_say(t).rstrip(".") for t in raw if t.strip() and not LOTO_RE.search(t)]
    checks = checks[:vc["max_checks"]]
    summary = _say(d.get("condition_summary") or "")
    if summary and summary[-1] not in ".!?":
        summary += "."
    parts = [f"Arrived at {tag}.", summary]
    if checks:
        parts.append("First checks: " + " ".join(f"{i + 1}. {c}." for i, c in enumerate(checks)))
    parts.append(vc["safety_line"])
    return " ".join(p for p in parts if p)


def _cache_file(cfg: Cfg, text: str) -> Path:
    vc = cfg["viewer"]["voice"]
    h = hashlib.sha1(f"{vc['voice_id']}|{vc['model_id']}|{vc['output_format']}|{text}".encode()).hexdigest()[:12]
    d = out_path(cfg, "voice_cache")
    d.mkdir(exist_ok=True)
    return d / f"{h}.mp3"


def synthesize(text: str, cfg: Cfg) -> bytes | None:
    """MP3 bytes for text (cache first), or None if voice is disabled/unavailable."""
    vc = cfg["viewer"]["voice"]
    if not vc.get("enabled", True):
        return None
    cf = _cache_file(cfg, text)
    if cf.exists():
        return cf.read_bytes()
    load_dotenv(ROOT / ".env")
    key = os.getenv(vc["api_key_env"], "")
    if not key or key == "your-key-here":
        print(f"[voice] {vc['api_key_env']} not set in .env; briefing stays text-only")
        return None
    req = urllib.request.Request(
        API.format(voice_id=vc["voice_id"], fmt=vc["output_format"]),
        data=json.dumps({"text": text, "model_id": vc["model_id"]}).encode(),
        headers={"xi-api-key": key, "Content-Type": "application/json", "Accept": "audio/mpeg"},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=vc["timeout_s"]) as r:
            audio = r.read()
    except urllib.error.HTTPError as e:
        print(f"[voice] ElevenLabs HTTP {e.code}: {e.read()[:200]!r}")
        return None
    except Exception as e:  # noqa: BLE001 - demo must not break
        print(f"[voice] ElevenLabs failed: {type(e).__name__}: {str(e)[:120]}")
        return None
    cf.write_bytes(audio)
    return audio


def briefing(d: dict, tag: str, cfg: Cfg) -> tuple[str, bytes | None]:
    """Spoken script + mp3 bytes for a diagnosis dict from any copilot model."""
    text = speech_text(d, tag, cfg["viewer"]["voice"])
    return text, synthesize(text, cfg)


def speak(text: str, cfg: Cfg | None = None) -> Path | None:
    """Simplest hook: any text -> cached mp3 file path (None if voice unavailable)."""
    cfg = cfg or load_config()
    return _cache_file(cfg, text) if synthesize(text, cfg) else None


# Dummy diagnosis in the exact shape copilot.agent.diagnose() returns (P-002 story from the Excel pack),
# for testing the voice without an LLM, the Excel pack or the pipeline outputs.
SAMPLE_DIAGNOSIS: dict = {
    "asset": "P-002 (ASSET-014, PROCESS_PUMP)",
    "condition_summary": "P-002 shows a cavitation risk: vibration has nearly tripled in three days while "
                         "discharge pressure and flow are falling.",
    "observations": [
        {"text": "Vibration rose from 1.9 to 5.2 mm/s; healthy pump P-003 averages 2.1 mm/s.",
         "evidence_ids": ["TEL-P-002-vibration_mm_s"]},
        {"text": "Pressure fell from 4.28 to 3.75 bar and flow from 71 to 63 percent.",
         "evidence_ids": ["TEL-P-002-pressure_bar"]}],
    "hypotheses": [{"text": "Insufficient suction head causing cavitation.", "confidence": 0.6,
                    "evidence_ids": ["CASE-0007"]}],
    "confidence": 0.62,
    "next_safe_checks": [
        {"text": "Read the suction pressure gauge and compare it with the design minimum.",
         "evidence_ids": ["DOC-PUMP-CAV"]},
        {"text": "Listen at the pump casing for a gravel-like crackling noise.", "evidence_ids": []},
        {"text": "Check that the suction strainer is not clogged.", "evidence_ids": ["WO-0112"]},
        {"text": "Follow LOTO and permits before any physical work.", "evidence_ids": ["DOC-FIELD-SAFETY"]}],
    "escalation": "Escalate to the reliability engineer if suction pressure is below the design minimum.",
}


def _play(path: Path) -> None:
    player = "afplay" if sys.platform == "darwin" else "ffplay"
    args = [player, str(path)] if player == "afplay" else [player, "-nodisp", "-autoexit", str(path)]
    try:
        subprocess.run(args, check=False)
    except FileNotFoundError:
        print(f"[voice] no '{player}' found; open the mp3 yourself")


def main() -> None:
    ap = argparse.ArgumentParser(description="ElevenLabs spoken briefing for the copilot diagnosis.")
    ap.add_argument("tag", nargs="?", help="asset tag (default: mapping.target_tag)")
    ap.add_argument("--dummy", action="store_true", help="use SAMPLE_DIAGNOSIS (no LLM / Excel / pipeline)")
    ap.add_argument("--text", help="speak this text instead of a diagnosis")
    ap.add_argument("--play", action="store_true", help="play the mp3 after generating it")
    a = ap.parse_args()
    cfg = load_config()
    tag = a.tag or cfg["mapping"]["target_tag"]
    if a.text:
        text = a.text
    else:
        if a.dummy:
            d = SAMPLE_DIAGNOSIS
        elif (f := out_path(cfg, f"diagnosis_{tag}.json")).exists():
            d = json.loads(f.read_text(encoding="utf-8"))
        else:
            from copilot.agent import diagnose
            d = diagnose(tag, None, cfg)
        text = speech_text(d, tag, cfg["viewer"]["voice"])
    print(text)
    path = speak(text, cfg)
    print(f"[voice] {path}" if path else "[voice] no audio")
    if path and a.play:
        _play(path)


if __name__ == "__main__":
    main()
