# Valmet Field Copilot: Since AI Hackathon 2026

Semantic 3D scene understanding and an AI field-service copilot for the Valmet challenge.
The pipeline turns the `Equipment Room.ply` Gaussian splat into framed, labeled assets,
maps the Excel builder pack onto them, walks a service worker along a collision-free path to
the faulty pump **P-002 (CAVITATION_RISK)**, and gives a grounded AI diagnosis on arrival.

## Pipeline

Each stage reads its inputs from `out/` and writes its outputs there, so stages can be rerun on their own.
All tunable values are in `config.yaml`.

| Stage | Module | Output |
|---|---|---|
| 1. Segmentation | `pipeline/segment.py`: voxel + outlier removal, normal-checked floor/ceiling/wall removal, DBSCAN, direction-class split of pipe networks, collinear pipe merge | `assets.json` (936 objects with 3D boxes) |
| 2. Labeling | `pipeline/label.py`: renders each object (front, side, context) and labels it zero-shot with Gemini vision from a fixed vocabulary, falling back to geometry rules | `assets_labeled.json`, `crops/`, `label_contact_sheet.png` |
| 3. Navigation | `pipeline/navgrid.py`: x–z occupancy grid, obstacles inflated by avatar radius, A*, line-of-sight smoothing, clearance check | `path.json`, `navgrid.png` |
| 4. Mapping | `pipeline/mapping.py`: Excel assets → objects (label match, geometry rule, seeded fallback) with explicit confidence | `scene_graph.json`, `scene_anchors.csv` |
| 5. Copilot | `copilot/evidence.py` builds the evidence pack; `copilot/agent.py` runs Gemini under the DOC-RAG-GUIDE + DOC-FIELD-SAFETY rules, then checks every cited ID | `diagnosis_<tag>.json` |
| 6. Viewer | `viewer/app.py`: Viser scene with splats, frames, walking worker, proximity panel, click-to-inspect, copilot panel; on arrival the worker faces the target and `copilot/voice.py` reads the validated diagnosis aloud (ElevenLabs) | http://localhost:8080 |

## Setup

```bash
python -m venv .venv            # Python 3.12 tested
.venv/Scripts/pip install -r requirements.txt     # Windows (use .venv/bin/pip on Linux/macOS)
cp .env.example .env            # put your FEATHERLESS_API_KEY (and optional ELEVENLABS_API_KEY) in .env
```

Put the challenge files in `data/` (not in git):
`data/Equipment Room.ply` (full, uncorrupted scan) and `data/builder_pack.xlsx`.
The older `boiler_room.ply` is the same room but only the back strip (2/3 of its rows are corrupted);
it still works by switching `paths.ply` and the `scene`/`known_planes` values in `config.yaml`.

## Run

```bash
VALMET_OUT=out_baseline python -m pipeline.segment     # baseline (keeps fine fragments + source rows)
python -m pipeline.repair geo                          # geometry-only pipe continuity -> out_geo/
python -m pipeline.propose RV-01 all                   # per review region: render, Qwen proposal (paid), SAM 2.1, visible support
python -m pipeline.repair context                      # geometry + context repairs -> out/ (repairs.json, reversible)
python -m pipeline.evaluate out_baseline out_geo out   # reviewed references in review/regions.yaml
python -m pipeline.compare                             # before/after images -> out_review/compare/
python -m pipeline.label        # --rules-only: no requests (paid budget: copilot.budget)
python -m pipeline.navgrid
python -m pipeline.mapping
python -m copilot.agent P-002   # optional: diagnosis from the command line
python -m copilot.voice --dummy --play   # test the ElevenLabs voice with a sample diagnosis (no LLM/data)
python -m copilot.voice P-002   # optional: build/cache the spoken briefing (out/voice_cache/)
VALMET_OUT=out_geo python -m pipeline.export_ifc     # IFC/BIM model with scan meshes + deviation -> scene.ifc
VALMET_OUT=out_geo python -m viewer.removable   # objects that can be moved/removed/duplicated (viewer.removable.include)
python -m viewer.app            # open http://localhost:8080 and press Play
```

## Notes

- `pipeline/load_splat.py` detects and drops corrupted rows (needed for `boiler_room.ply`; `Equipment Room.ply` is clean).
- The up axis is −Y (Equipment Room: floor y = 1.174, ceiling y = −0.863, walls at x = −4.67 / 3.73, z = −3.76 / 3.14).
  Native coordinates are kept, and the walking plane is x–z.
- Navigation uses the observed floor (`navgrid.floor_fill: observed`) and starts at the door in the back wall.
- Current results on Equipment Room: 936 objects, 935 named by Gemini vision, a 3.2-unit collision-free walk to P-002.
- Gemini models are tried in the order of `copilot.models`. On quota, overload or retired-model errors the next model is used. If all fail, the copilot shows the last good answer (marked as cached) or a template built from the same evidence.

## Boundary repair (Nov 2026)

LLM provider is Featherless (`Qwen/Qwen3.5-397B-A17B`, OpenAI-compatible, `copilot/llm.py`) with a hard paid-request cap and
ledger (`llm_ledger.jsonl`), no automatic retries; `VALMET_OFFLINE=1` uses caches/templates only. Measured results,
method and limitations: `review/RESULTS.md`. Tests: `python -m pytest -q tests`.
The navigation target is a scene coordinate (`navgrid.target.point`), so OBJ id shifts between segmentation versions are safe.

## Guided boundary correction (manual, no paid calls)

`python -m viewer.app` also starts the correction page (default http://localhost:8091, next free port if taken;
the "Boundary repair" panel links to it). Standalone: `python -m api.annotate` (own Viser preview). Run one at a time:
both share `review/corrections/`.

Flow: pick region/view -> New component (class optional, default `unknown`) -> Draw box / Include / Exclude ->
Preview mask (SAM 2.1 on the clean EWA render, cached embedding) -> Preview in 3D (contribution-weighted visible
support, `pipeline.propose.view_support` + `pipeline.repair.claim_support`) -> open a suggested second view, adjust ->
Accept (single-view needs the explicit "provisional" tick) -> Apply -> Undo if needed.

- Corrections and revision history: `review/corrections/` (prompts in original pixels, camera/render/image/buffer
  hashes, masks, selection). `out/` and the PLY are never modified.
- Apply replays all active accepted corrections over `out/` into `out_manual/` (`labels_before_manual`,
  `manual_owner`, per-object membership, `manual_manifest.json`). Stale corrections block Apply. Downstream
  artifacts of `out_manual/` are listed as stale until regenerated (label -> navgrid -> mapping -> tour).
- Disputed voxels (inside the mask in one view, visibly outside in another) are excluded and reported; unseen voxels
  are never claimed except with a whole object that passes support + coverage + spatial checks (reported as inferred).

## MCP server (ask the copilot from Claude Code / Claude Desktop)

`mcp_server.py` exposes the field-service data and actions as MCP tools (thin wrapper over `copilot/tools.py`):
`search_assets`, `get_asset`, `telemetry_trend`, `recent_events`, `maintenance_history`, `nearby`, `route_to`,
`plan_tour`, `draft_work_order`, `diagnose`, `budget_status`, and viewer control `show_in_viewer`,
`walk_in_viewer`, `tour_in_viewer` (the running viewer polls `out/viewer_cmd.jsonl`). Prompts: `shift_handover`, `investigate`.

- **Claude Code:** `.mcp.json` registers the server as `valmet-copilot`; open Claude Code in this folder and approve it.
  On Linux/macOS set `VALMET_PYTHON=.venv/bin/python` first (defaults to the Windows venv path).
- **Claude Desktop:** add to `claude_desktop_config.json`:
  `"valmet-copilot": {"command": "<repo>/.venv/Scripts/python.exe", "args": ["<repo>/mcp_server.py"]}`
- Example questions: *"Which pumps need attention?"*, *"Why is P-002 flagged? Cite evidence."*, *"Take me to M-006 in the viewer."*,
  *"Plan a tour past CV-003 and M-006."*
- Safety and cost: data tools are read-only, work orders are drafts, no physical control; only
  `diagnose(allow_paid=true)` can spend a request from the team's paid budget.

## Viewer chat ("Ask the MCP")

The viewer panel "Ask the MCP" has a chat box. It works on every computer, with or without Claude:

| Question type | Who answers | Needs |
|---|---|---|
| Quick commands: `take me to M-006`, `show P-005`, `start tour`, `tour past CV-003 and M-006`, `which assets need attention`, `status of P-002`, `help` | the viewer itself, instantly | nothing (free) |
| Anything else, e.g. *"take me to the motor that is overheating and tell me why"* | Claude Code in the background, using only our MCP tools (it can also move the worker / camera) | Claude Code installed and logged in |

The chat panel shows the Claude status when the viewer starts:
- **🤖 Claude ready (subscription)**: free-form questions work; they use your own Claude Pro/Max plan (no extra cost).
- **⚠️ API key**: works, but each Claude question is billed per use.
- **⚠️ Not installed / not logged in**: only quick commands work. To enable: install Claude Code
  (https://claude.com/claude-code or the VS Code / Antigravity extension), run `claude` in a terminal, type `/login`,
  then restart the viewer. If it is installed in an unusual place, set `chat.claude_path` in `config.yaml`.

Claude never spends the team's paid Featherless budget unless you explicitly ask for a fresh paid diagnosis.
