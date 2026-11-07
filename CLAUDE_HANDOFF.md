# SinceAI-Project — investigate boundaries, integrate Featherless, test one repair loop

Work in `C:\Users\Usuario\Projects\hackathon\SinceAI-Project` (mahadi0007/SinceAI-Project). This handoff supersedes the previous handoff. The user wants a reasoned detection strategy and incremental implementation, not another untested global grouping replacement. No fine-tuning.

## Facts established by review

- Reviewed local and remote revision: `5de414098cbc2f12deb900bdcec823afc9e0e9c0`. Recheck changes before editing; preserve local work.
- This project's `.venv\Scripts\python.exe` is Python 3.11.9. Runtime imports and pip check succeeded. OpenAI SDK and pytest were missing at review; install missing packages only here, through `.venv\Scripts\python.exe -m pip`. Never recreate a working environment or overwrite `.env`. Featherless token is already configured in `.env`; do not print it.
- Code currently uses Gemini despite the Featherless token. `copilot/llm.py` implements Gemini; `pipeline/label.py` and `copilot/agent.py` use it. `pipeline/navgrid.py` and `pipeline/mapping.py` hard-code comparisons to `gemini_vision` in several places.
- No `data/` or generated `out/` existed at review. Existing inputs: `C:\Users\Usuario\Projects\hackathon\sinceai2026\data\raw\Equipment Room.ply` and `C:\Users\Usuario\Projects\hackathon\sinceai2026\data\raw\equipment.xlsx`. Inspect and configure paths without modifying originals. README's 936 objects are historical results, not ground truth.
- Native coordinates: -Y up, x-z walking plane; approximate scale `scene.units_to_m = 1.15`. Do not import +Z/meters code from sinceai2026 without explicit conversion.

## Critical metadata finding — use it correctly

The workbook's `Scene_Anchors` has 30 rows, ALL with `mapping_status = placeholder_until_valmet_scene_received`, `synthetic_record = true`, and instructions to remap when the Valmet scene arrives. Bounding dimensions repeat by class. Asset_Master has 12 control valves, 6 process pumps, 6 motors, 3 pressure transmitters and 3 vibration sensors. These are synthetic inventory records, not proof of those exact counts or locations in this scan.

Use metadata to build the target ontology, prioritize useful equipment searches, preserve stable IDs after a verified mapping, and lower confidence for relevant Data_Quality issues. Do not force the scene to contain exactly 30 assets, fit boxes to placeholder dimensions, align synthetic anchors to the PLY, infer identity from an issue code, or interpret a connectivity protocol as physical piping topology. Telemetry cannot identify which visible shape is P-002.

Keep separate records for (a) visual component class, (b) physical instance boundary, (c) asset identity and mapping confidence, and (d) operating condition. A confident pump classification does not establish the pump's tag. User-confirmed real-scene anchors or readable real tags can establish identity; save their method and provenance. Current seeded/random/forced-demo assignments must remain explicitly synthetic/unverified and must not become detector training labels or validation truth.

## What is broken and the hypothesis to test

The current pipeline is geometry fragments -> boxes -> tight crops -> Gemini labels. The split stage divides large connected networks by local x/y/z/other classification and reclustering. Noise reassignment and later merges can also change boundaries. Naming the resulting fragments does not reconstruct an entire pump.

Test an additional contextual repair loop: retain fine geometric fragments; propose whole visible equipment from context; refine the proposed boundaries using prompted SAM; associate masks with visible 3D support; propose parent instances or splits; review the evidence. Geometry remains useful for infrastructure continuity and rejecting implausible joins. Do not globally increase DBSCAN epsilon, merge by box containment alone, or blindly merge pump and motor into one equipment ID. A pump and its motor can be separate components under an assembly.

An earlier separate experiment in the +Z sinceai2026 pipeline went from 1,056 input fragments to 938 objects under existing merges, versus 1,053 parents under a stricter experimental centerline/cylinder pass. Only three joins were accepted. This did NOT improve fragmentation. Do not copy that experiment as a finished solution or assume synthetic tests prove scan quality.

## First implementation milestone

### A. Integrate Featherless without changing detection behavior

Use the OpenAI-compatible API at `https://api.featherless.ai/v1`, loading `FEATHERLESS_API_KEY` from the local `.env`. The requested family is Qwen 3.5, not Qwen3-VL. A confirmed vision-capable Featherless model ID is `Qwen/Qwen3.5-397B-A17B`:
`https://featherless.ai/models/Qwen/Qwen3.5-397B-A17B`.
Check current availability/account access, actual price and image input support before the pilot. Do not invent a `Qwen3.5-VL` ID. A smaller model may be benchmarked if its vision endpoint is actually available.

Create a provider-neutral interface for text and images; adapt label and diagnosis callers. Preserve diagnosis grounding and validation; changing provider does not make citation-only checks sufficient. Replace Gemini-only source checks downstream with explicit supported vision-source handling and confidence gates. Test target selection and asset mapping with Featherless-labelled inputs.

Start with one worker and at most three paid inference requests in the initial pilot; bounded outputs and no automatic paid retry/model cascade. The user has 25 Featherless credits, not authorization to consume the whole balance immediately. Include automatic viewer/copilot calls in limits. Record usage and observed cost when available; request count alone does not guarantee cost. Validate JSON, current-batch IDs, allowed labels and finite [0,1] confidence. Hash provider/model/prompt and real image/support evidence into caches, preserve existing good outputs, and provide an offline path. No full-scene relabel before checking pilot results.

### B. Diagnose fragmentation on a fixed sample before choosing new thresholds

Run the existing pipeline to an isolated baseline output folder with input checksum, config and seed saved. Preserve source PLY row IDs through filtering, fine fragment labels before merging, and explicit label-to-asset ID lookup (labels currently index the full assets list INCLUDING structures).

Create an inspection view/contact sheet covering about 10 equipment regions: some pumps/motors, valves among pipes, and straight/parallel/bent pipe runs. Assign stable review region IDs. Save contextual views showing the current fragment IDs in distinct colors. Trace the worst examples through DBSCAN -> direction split -> reassignment -> merges, so we can distinguish over-splitting from already-mixed clusters. Use existing saved images if suitable; they do not substitute for calibrated visibility.

Manually specify desired component boundaries for this small review set (a click/box or several confirmed examples is acceptable). Define a component ontology: pump housing, motor, valve body plus its handwheel/actuator, pipe run, sensor, structure, unknown. Keep assembly membership distinct from asset identity. Record reviewed references rather than inventing pixel-accurate truth.

### C. Test contextual recognition plus masks on ONE region

Give Qwen 2–4 contextual views of one region, including full equipment and its surroundings, plus a companion fragment-ID overlay. Begin with visual evidence alone; compare metadata-assisted interpretation as a separate pass to detect suggestion bias. Metadata can narrow plausible classes but must not cause the model to invent expected equipment.

Ask for structured candidate equipment proposals: candidate ID, component class, per-view box or foreground point, visible fragment references if readable, completeness/occlusion notes, proposed fragment relationships, and uncertainty. Treat box/point coordinates and grouping as hypotheses, not automatic edits. Unknown must remain allowed. Avoid huge lists of every room fragment in one call.

Use candidate boxes/points to prompt SAM 2.1 Small/Tiny inference, adding negative points on nearby pipes/equipment when useful. Compare raw prompt boxes with the SAM masks. SAM does not automatically establish physical instances or identify the equipment. Verify local hardware and compatible packages in this project's .venv before installation; no training.

Before 2D-to-3D association, implement or verify calibrated cameras and matching depth/ID/visibility output. Associate only visible support. A projected Gaussian center inside a mask is not sufficient visibility evidence, and full ellipsoids can contribute even when centers fall outside. Document the approximation used. If renderer visibility is not yet available, stop the experiment at inspectable 2D masks and implement the visibility prerequisite before applying any 3D repair; do not silently project through walls.

For each visible fragment, measure mask support across views. Propose merging complementary support belonging to the same physical component; split a mixed fragment only using mask-supported point subsets. Adjacent motors/valves and occluded support need explicit uncertainty. No fixed universal vote threshold until reviewed examples establish one. Preserve original leaves, source indices, parent-child membership and a reversible edit log; refit boxes to the accepted support.

### D. Limit geometric grouping to evidence it can support

For infrastructure, evaluate direction, centerline offset, local normals/cylinder fit, thickness agreement, longitudinal gaps and whole-run fit. Account for short partial surfaces, nearby parallel pipes, elbows, tees, valves and flat strips. No transitive drift across a pipe network. A straight object may be a beam or cable tray rather than a pipe.

Do not remove all points near a structural plane; wall-mounted objects and floor-level bases matter. Keep physical geometry for navigation even when infrastructure boxes are hidden. A fragment/instance display toggle helps review but is not a detection-quality metric.

## Evaluate before scaling

Compare current baseline, geometry-only repair, and contextual Qwen + SAM repair on the SAME reviewed regions. Measure whether equipment is complete, distinct neighboring assets stay separate, fitted boxes contain relevant support without swallowing background, and points/identities are preserved. Report under-segmentation as well as fragmentation. Fewer boxes alone is not success. Show images and failure examples.

Add tests for adjacent parallel pipes, gaps, bends/tees, valve barriers, compact contained equipment, mixed-cluster splits, source membership, reprojection/occlusion, provider schema/cache/limits and downstream label-source handling. Preserve a baseline mode if candidate behavior is worse.

After a small sample shows an improvement, expand to the rest of the room. Regenerate mapping and paths against that segmentation version: sorted OBJ IDs can shift, so hard-coded OBJ-0269 must not silently point somewhere new. The eventual navigation API should accept a real-scene coordinate and find a reachable approach position with clearance; a collision obstacle remains an obstacle regardless of whether it has an equipment label.

## Deliverables for this milestone

1. Working Featherless provider integration and bounded pilot, with actual observations.
2. Metadata audit and explicit treatment of placeholder anchors.
3. Baseline fragment inspection of the fixed review regions.
4. One inspectable contextual-proposal -> SAM-mask -> visibility-supported repair example, or a clearly identified visibility prerequisite with evidence (no fabricated 3D repair).
5. Reviewed before/after results and the justified next step. Install packages only as they become needed in the local .venv; do not launch a full-scene experiment before the small test supports it.
