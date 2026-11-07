# Valmet Field Copilot: Since AI Hackathon 2026

Semantic 3D scene understanding and an AI field-service copilot for the Valmet challenge.
The pipeline turns the `Industrial Boiler Room.ply` Gaussian splat into framed, labeled assets,
maps the Excel builder pack onto them, walks a service worker along a collision-free path to
the faulty pump **P-002 (CAVITATION_RISK)**, and gives a grounded AI diagnosis on arrival.

## Pipeline

Each stage reads its inputs from `out/` and writes its outputs there, so stages can be rerun on their own.
All tunable values are in `config.yaml`.

| Stage | Module | Output |
|---|---|---|
| 1. Segmentation | `pipeline/segment.py`: voxel + outlier removal, normal-checked floor/ceiling/wall removal, DBSCAN, direction-class split of pipe networks, collinear pipe merge | `assets.json` (286 objects with 3D boxes) |
| 2. Labeling | `pipeline/label.py`: renders each object (front, side, context) and labels it zero-shot with Gemini vision from a fixed vocabulary, falling back to geometry rules | `assets_labeled.json`, `crops/`, `label_contact_sheet.png` |
| 3. Navigation | `pipeline/navgrid.py`: x–z occupancy grid, obstacles inflated by avatar radius, A*, line-of-sight smoothing, clearance check | `path.json`, `navgrid.png` |
| 4. Mapping | `pipeline/mapping.py`: Excel assets → objects (label match, geometry rule, seeded fallback) with explicit confidence | `scene_graph.json`, `scene_anchors.csv` |
| 5. Copilot | `copilot/evidence.py` builds the evidence pack; `copilot/agent.py` runs Gemini under the DOC-RAG-GUIDE + DOC-FIELD-SAFETY rules, then checks every cited ID | `diagnosis_<tag>.json` |
| 6. Viewer | `viewer/app.py`: Viser scene with splats, frames, walking worker, proximity panel, click-to-inspect, copilot panel | http://localhost:8080 |

## Setup

```bash
python -m venv .venv            # Python 3.12 tested
.venv/Scripts/pip install -r requirements.txt     # Windows (use .venv/bin/pip on Linux/macOS)
cp .env.example .env            # put your GEMINI_API_KEY in .env
```

Put the challenge files in `data/` (not in git):
`data/boiler_room.ply` and `data/builder_pack.xlsx`.

## Run

```bash
python -m pipeline.segment
python -m pipeline.label        # --rules-only to skip Gemini
python -m pipeline.navgrid
python -m pipeline.mapping
python -m copilot.agent P-002   # optional: diagnosis from the command line
python -m viewer.app            # open http://localhost:8080 and press Play
```

## Notes

- The `.ply` file is corrupted after row ~502k. `pipeline/load_splat.py` reads only the valid part.
- The up axis is −Y (floor y ≈ 1.075). Native coordinates are kept, and the walking plane is x–z.
- The scan has no floor for x > 0.5, so `navgrid.floor_fill: footprint` assumes the floor continues under the scanned area.
- Gemini models are tried in the order of `copilot.models`. On quota, overload or retired-model errors the next model is used. If all fail, the copilot shows the last good answer (marked as cached) or a template built from the same evidence.
