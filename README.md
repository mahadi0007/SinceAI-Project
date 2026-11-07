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
