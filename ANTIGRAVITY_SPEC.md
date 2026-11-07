# Valmet × Since AI 2026 — Build spec for Antigravity

Paste this file into Antigravity as the first prompt (Planning mode), or add it as a workspace rule.
Goal: a working end-to-end demo by **Sunday 09:00** (deadline 10:00).

## 1. What we are building

A Python pipeline + browser viewer that:
1. Loads `Industrial Boiler Room.ply` (3D Gaussian Splatting) and extracts as many objects as possible, each with a 3D bounding box and a label.
2. Maps the 30 assets from the Excel builder pack onto extracted objects (rule-based by type, random for the rest).
3. Computes a collision-free walking path on the floor to the faulty device and animates an avatar along it.
4. While walking, highlights assets within a radius and shows their Excel status.
5. At the target, an AI copilot produces a grounded diagnosis with cited evidence IDs, confidence and safe next steps.

Scoring: extraction density 30%, navigation + proximity 25%, AI grounding 20%, backend architecture 15%, demo 10%.
**Spend effort on 3D processing, not UI.**

## 2. Facts about the data (verified, do not re-discover)

### The .ply file
- Standard 3DGS layout, 59 float32 properties, header says 1,506,136 Gaussians.
- **Only the first ~502,478 rows are valid.** The rest is random bytes (NaN, ±3e38). Always load through `load_splat.py` (provided, tested). After opacity > 0.1 and floater filtering: ~284k Gaussians.
- **Up axis is −Y.** Floor plane y ≈ +1.075, ceiling y ≈ −0.889 (≈ 2 units floor-to-ceiling). Back wall plane z ≈ −3.445.
- Keep native coordinates. In Viser call `server.scene.set_up_direction("-y")`. The walking plane is **x–z**.
- Scene footprint after cleanup: x −5.2…4.3, z −4.2…−0.7 (a strip along one wall: pipes, valve manifold, a door, a stair/cage). Units look roughly metric (≈1 unit ≈ 1.1–1.2 m, from the door height) — make scale a config value.
- Quaternions are stored as rot_1, rot_2, rot_3, rot_0 (rot_0 = w). `load_splat.py` returns `quats_wxyz` and has `covariances()` for splat rendering.
- Baseline already tested: remove floor/ceiling/back-wall planes → voxel 0.02 → statistical outlier removal → DBSCAN(eps=0.045, min_points=12) gives **70 clusters with ≥60 points**. Problem: the pipe network on the right merges into one giant cluster → must be split (see 4.2).

### The Excel pack (16 sheets)
- `Asset_Master` (30 rows): 12 CONTROL_VALVE, 6 PROCESS_PUMP, 6 ELECTRIC_MOTOR, 3 PRESSURE_TRANSMITTER, 3 VIBRATION_SENSOR.
- `Scene_Anchors` (30): placeholder coordinates, `mapping_status = placeholder_until_valmet_scene_received`. **Our job is to replace these with extracted box centres/sizes and a real `mapping_confidence`.**
- `Current_State` (30): health_score, condition_state, issue_code, urgency, recommended_next_check, knowledge_doc_id.
- `Telemetry_Sample_3d` (4,320 rows, 30-min, 3 days): pressure, flow, valve position, vibration, bearing temp, motor current, signal quality…
- `Alarms_Events` (1,000), `Maintenance_History` (300), `Historical_Service_Cases` (50), `Knowledge_Docs` (8), `Data_Quality` (120), `Builder_Questions` (10), `API_Contract` (8 endpoints).
- `Knowledge_Docs` includes **DOC-RAG-GUIDE** (answer contract: name the asset, cite telemetry/event/doc IDs, state confidence, separate observation from hypothesis, propose the next safe verification step), **DOC-FIELD-SAFETY** (never replace LOTO/permits; recommend verification and escalation, no autonomous physical action) and **DOC-3D-MAPPING** (map objects to asset IDs with explicit confidence, never silently guess identity).
- Missing from the folder: `telemetry_full_14d_30min.csv` and the 38 markdown docs listed in `RAG_Corpus_Index`. Use the Excel sample and generate asset cards from Excel rows.

### Recommended faulty target: **P-002 (ASSET-014), PROCESS_PUMP, CAVITATION_RISK, health 45, urgency high**
Evidence over 3 days: vibration 1.9 → 5.2 mm/s, pressure 4.28 → 3.75 bar, flow 71 → 63 %. Healthy pump P-003 averages 2.1 mm/s. 29 CAVITATION_RISK events. Clear, citable story.
Other degraded/attention assets (use as "warning" along the path): P-005 FOULING (44), M-006 OVERHEATING (45), CV-003 POSITION_DEVIATION (52), VS-002 MOUNTING_ISSUE (58).

## 3. Repo layout

```
valmet_copilot/
  data/                      # NOT in git: ply, xlsx, videos
  config.yaml                # paths, scale, eps, radius, target tag, model name
  pipeline/
    load_splat.py            # provided
    segment.py               # planes + clustering + splitting + boxes  -> out/assets.json
    label.py                 # crop render + VLM/CLIP labels            -> out/assets_labeled.json
    navgrid.py               # occupancy grid + A* + smoothing           -> out/navgrid.npz, out/path.json
    mapping.py               # Excel -> assets, Scene_Anchors remap      -> out/scene_graph.json
  copilot/
    evidence.py              # build evidence pack for an asset (stats, events, WOs, cases, docs, DQ)
    agent.py                 # LLM call with DOC-RAG-GUIDE contract, JSON output
  api/main.py                # FastAPI implementing API_Contract routes
  viewer/app.py              # Viser simulation
  run_all.py                 # runs pipeline end to end
  README.md
```
Every stage writes JSON/NPZ to `out/` so stages can be developed in parallel and rerun independently.

## 4. Module specs

### 4.1 segment.py
1. `load_splat()` → points + colours.
2. Voxel downsample 0.02, statistical outlier removal (nb=20, std=2.0).
3. Remove structural planes: floor (|y−1.075|<0.05), ceiling (|y+0.889|<0.05), back wall (|z+3.445|<0.05); then RANSAC for any other large planes (> 5% of points). Keep walls/floor as their own assets ("wall", "floor") — they count.
4. DBSCAN (eps 0.045, min_points 12). Drop clusters < 60 points.
5. **Split oversized clusters** (bbox diagonal > 1.5 or > 5,000 points): recursive DBSCAN with smaller eps (0.03, 0.02) on xyz + Lab colour (colour weight ~0.3); also try separating thin vertical cylinders (pipes) via PCA linearity > 0.9 → label "pipe segment".
6. For each cluster: axis-aligned box in native coords (OBB optional), centroid, extent, point count, mean colour, PCA shape features (linearity, planarity, height above floor).
7. Output `assets.json`: `[{id, center, extent, n_points, color, features}]`. Print the count. **Target ≥ 100 objects.**

### 4.2 label.py
- For each asset, render a small orthographic image of its points (front view, −z direction) + a context image with the box drawn.
- Batch to a vision model (Gemini) with a fixed vocabulary: pump, motor, control valve, pressure transmitter/gauge, vibration sensor, pipe segment, tank/vessel, electrical cabinet, cable tray, ladder/stairs, cage/rack, door, sign, box/crate, trolley, wall, floor, other. Return `{label, confidence}` JSON.
- Fallback without API: rules from features (linear + tall → pipe segment, planar + large → wall/door, small + colourful → valve/gauge).
- Optional stretch (GPU, LUMI): render many views, SAM 2 masks, project to Gaussians, merge — only if the baseline is done.

### 4.3 navgrid.py
- Grid on x–z, cell 0.05.
- Walkable = cells containing floor-plane points (dilate 2 cells to fill holes), minus obstacles.
- Obstacles = points with height above floor between 0.10 and 1.80 units (y between 1.075−1.80 and 1.075−0.10), excluding floor/ceiling.
- Inflate obstacles by avatar radius 0.25.
- A* (8-connected) from a start cell (configurable, default: walkable cell at one end of the strip) to the walkable cell nearest the target asset's box face.
- Smooth with line-of-sight shortcutting (Bresenham on the inflated grid), then resample every 0.05.
- Validate: assert no path sample is in an obstacle cell; print min clearance. Save a PNG of the grid + path for the slides.

### 4.4 mapping.py
- Load all sheets with pandas.
- Map Excel classes to labels: PROCESS_PUMP→pump, ELECTRIC_MOTOR→motor, CONTROL_VALVE→control valve, PRESSURE_TRANSMITTER→gauge/transmitter, VIBRATION_SENSOR→small object attached to pump/motor. If too few matching labels, assign remaining Excel assets to the nearest plausible unmapped objects (random with fixed seed), `mapping_confidence` lowered accordingly (label match 0.8–0.9, random 0.3–0.5).
- Force the target (P-002) onto an object reachable by the path, near the far end of the strip (makes a longer, more interesting walk).
- All other extracted objects stay as scene objects with label + "no asset record" (DOC-3D-MAPPING: never silently guess).
- Write `scene_graph.json` and an updated `Scene_Anchors` CSV with real coordinates.

### 4.5 copilot
- `evidence.py`: for an asset, compute trend stats from telemetry (first-day vs last-day mean, slope, comparison to healthy same-class baseline), last N events with IDs, related work orders, matching `Historical_Service_Cases` by class + issue code, knowledge docs, and `Data_Quality` issues (lower confidence when present; Builder Q-06).
- `agent.py`: one LLM call, system prompt = DOC-RAG-GUIDE + DOC-FIELD-SAFETY text verbatim. Return JSON:
  `{asset, condition_summary, observations[{text, evidence_ids}], hypotheses[{text, confidence}], confidence, next_safe_checks[], escalation, work_order_draft}`.
- Validate that every cited ID exists in the evidence pack; drop or flag unknown IDs.
- Short one-line status per nearby asset while walking (no LLM needed: template from Current_State).

### 4.6 viewer/app.py (Viser)
- `server.scene.set_up_direction("-y")`.
- Render the splat with `server.scene.add_gaussian_splats(...)` using `covariances()`; fallback `add_point_cloud`.
- Draw all boxes faintly (colour by condition: healthy grey, watch amber, degraded orange, attention red, no-record white). Label text above each.
- Draw the path as a line; avatar = capsule/sphere mesh moving along samples at ~1 unit/s.
- Each frame: KD-tree query radius 1.5 → highlight those boxes (thicker/brighter) + list them in a GUI panel with tag, class, health, issue.
- GUI: Play/Pause/Reset, radius slider, "Ask copilot" text box, counter "N assets extracted · M mapped".
- On arrival: call copilot, show the JSON as formatted markdown in the panel; highlight cited evidence IDs.

### 4.7 api/main.py
FastAPI with the 8 routes from the `API_Contract` sheet (GET /api/assets, /api/assets/{id}, …/telemetry, …/events, …/documents, /api/scene/anchors, POST /api/copilot/query, POST /api/work-orders/draft). The viewer can import functions directly; the API shows production-readiness (15% architecture score).

## 5. Rules for the agent
- Python 3.11 venv. Libraries: numpy, pandas, openpyxl, open3d, scikit-learn, scipy, viser, fastapi, uvicorn, google-genai (or anthropic), pyyaml, matplotlib.
- Never commit the data files. Add `data/` and `out/` to .gitignore.
- After each module, run it and save a verification PNG in `out/` (cluster view, grid + path, etc.).
- Keep functions small and typed; config values in `config.yaml`, not hard-coded.
- API keys from `.env`.

## 6. Timeline
- Fri night: load + segment baseline + Excel loader + repo skeleton.
- Sat morning: cluster splitting + labels, navgrid + A*, evidence + agent v1.
- Sat afternoon: mapping + Viser integration + proximity; ask Valmet the open questions.
- Sat evening: push object count, polish path and copilot. Feature freeze ~00:00.
- Sun 06:00–09:00: demo video (≤5 min), 5–8 slides, README, technical note, submit.

## 7. Open questions for Valmet (Discord challenge channel)
1. The .ply has only ~1/3 valid rows (rest is random bytes). Is there a clean file? The walkthrough video shows pumps and tanks not in the valid part.
2. Scene scale/units?
3. Do walls, floor, pipe segments count as extracted assets?
4. Is the full 14-day telemetry CSV and the 38 markdown docs available?
