"""Pre-inspection brief: offline, all sections in order, only known evidence IDs, no ledger entry."""
import pytest

from copilot.brief import build_brief, write_brief
from pipeline.common import load_config, out_path, repo_path

CFG = load_config()
SECTIONS = ["Header", "Location and route", "Along the way", "Diagnosis", "What to verify on site", "Safety",
            "Bring / prepare", "Draft work order", "Evidence appendix"]
need = pytest.mark.skipif(not (repo_path(CFG["paths"]["xlsx"]).exists()
                               and all(out_path(CFG, f).exists() for f in ("scene_graph.json", "path.json", "navgrid.npz"))),
                          reason="Excel pack or out/ navigation artifacts missing")


@need
def test_brief_sections_and_citations(monkeypatch):
    monkeypatch.setenv("VALMET_OFFLINE", "1")
    b = build_brief("P-002", CFG)
    assert [name for name, _ in b["sections"]] == SECTIONS
    assert b["cited_ids"] and set(b["cited_ids"]) <= set(b["allowed_ids"])
    assert "template" in b["diagnosis_source"] or "cached" in b["diagnosis_source"]


@need
def test_write_brief_offline_no_ledger(monkeypatch):
    monkeypatch.setenv("VALMET_OFFLINE", "1")
    ledger = repo_path(CFG["copilot"]["budget"]["ledger"])
    before = ledger.read_text(encoding="utf-8") if ledger.exists() else None
    r = write_brief("P-002", CFG)
    after = ledger.read_text(encoding="utf-8") if ledger.exists() else None
    assert before == after
    md = open(r["md"], encoding="utf-8").read()
    html = open(r["html"], encoding="utf-8").read()
    assert [f"## {i}. {s}" in md for i, s in enumerate(SECTIONS, 1)] == [True] * len(SECTIONS)
    assert "LOTO" in md and "not AI output" in md and "DRAFT" in md
    assert "data:image/png;base64," in html and "<script" not in html
    assert open(r["png"], "rb").read(4) == b"\x89PNG"
