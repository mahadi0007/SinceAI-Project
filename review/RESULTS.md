# Boundary repair: measured results (2026-11-07)

Inputs (read-only): `../sinceai2026/data/raw/Equipment Room.ply` (sha256 in `out_baseline/segment_manifest.json`),
`../sinceai2026/data/raw/equipment.xlsx`. No fine-tuning. Reviewed references: `review/regions.yaml`
(approximate 3D boxes from visual review of calibrated renders; RC-03 revised once, documented in the file).

## Variants (same reviewed references, `python -m pipeline.evaluate out_baseline out_geo out`)

| variant | objects | mean IoU | mean completeness | mean leakage | fragmented refs | accidental merges |
|---|---|---|---|---|---|---|
| baseline (`out_baseline`) | 936 | 0.530 | 0.694 | 0.199 | 5 | 0 |
| geometry-only (`out_geo`) | 1007 | 0.532 | 0.694 | 0.196 | 5 | 0 |
| geometry + context (`out`) | 1011 | **0.596** | 0.672 | **0.050** | 5 | 0 |

Per reference (IoU baseline -> geometry+context):
- RC-03 handwheel valve inside mixed cluster OBJ-0009: 0.09 -> 0.40 (leakage 0.88 -> 0.00). Valve cut out of a 7,146-pt cluster.
- RC-05 valve (ring handwheel + body inside mixed OBJ-0020): 0.05 -> 0.38 (completeness 0.79 -> 0.97). Remaining leakage 0.43
  is mostly the reference box being too high/shallow (object spans y 0.41-0.86, box 0.33-0.68; see `out_review/closeups/rv02_valve_low.png`).
- RC-02 circulator pump B: 0.54 -> 0.52, 3 pieces -> 2. The head is now one clean object; the box (union of baseline 638+395)
  also contains a curved hose that the split correctly left separate, so the metric under-counts.
- RC-01 pump A, RC-06/07 parallel pipes, RC-09 vessel: unchanged. RC-08 vessel 0.75 -> 0.74 (38 voxels taken by an RV-09 "sensor").
- RC-04 U-bend: 4 -> 3 pieces; bends are deliberately not crossed by the geometry rules.

Complete-equipment repair example: RV-01 (`out_review/compare/RV-01_v*.png`) - handwheel valve separated from the wall
cluster; pump B head reassembled from two fragments; pump A kept whole.

## Geometry-only
56 pipe joins (baseline box rule: 101 joins + 26 containment merges). Joins need agreeing fitted axis, centreline offset
<= 0.9 x half-width + 0.015, thickness ratio <= 1.6, axial gap <= 0.15, no compact non-pipe piece in the gap, and the whole
joined run must still fit one line. Neutral on the references; removes box-containment swallowing. Its global effect on the
45 baseline joins it no longer makes is NOT reviewed.

## Contextual loop
Featherless `Qwen/Qwen3.5-397B-A17B` (image input confirmed; $0.55/$3.5 per M tokens) proposes whole components from 2
calibrated views + a fragment-ID overlay -> SAM 2.1 tiny (CPU) masks from expanded boxes -> per-voxel visible support from the
renderer's ID/depth buffers (no projection through occluders; depth gate for background seen through gaps) -> merge objects
with >= 60% visible-in-mask support, split mixed non-pipe objects using supported voxels only. Elongated pipe runs are never
merged or split by equipment; earlier candidates are never stolen; extent > 1.2 rejected (2 rejections). 22 repairs over 8 regions,
each logged in `out/repairs.json` (+ moved voxels in `repairs_voxels.npz`); revert one with `repair.context.disabled`.

Paid usage (`llm_ledger.jsonl`): 10 requests, 18.8k prompt + 8.6k completion tokens, ~$0.040 estimated. Pilot = 3
(RV-01 proposal, one 12-object label batch, RV-02 proposal), then 7 after the pilot (6 regions + metadata pass). No retries.

## Known failures / limitations
- Qwen classed both circulator pump heads as "sensor" (visual and metadata-assisted runs alike; no invented equipment from the
  metadata hint). Downstream this maps PT-00x tags onto pumps as `component_class_only`, identity unverified.
- RV-04 / RV-06 / RV-07..09 repairs are qualitative only (no references); RV-06 "pump + motor" and RV-04 "vessel" are plausible
  but unverified - inspect in the viewer and disable if wrong.
- Visibility uses opaque discs (opacity >= 0.2, 2-sigma geometric-mean footprint), not alpha-blended splatting.
- Thresholds are provisional, set from these few examples; references are approximate boxes.
- Asset identities remain synthetic (geometry_rule / forced_demo_target); workbook anchors are placeholders and unused as coordinates.
