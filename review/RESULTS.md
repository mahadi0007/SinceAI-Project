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

---

# v2 (2026-11-07): bug fixes + Gaussian rendering experiment

## Renderer
gsplat was not practical here: only a `py3-none-any` wheel that JIT-compiles CUDA kernels, no CUDA toolkit/nvcc,
CPU-only torch in `.venv`, and ~5 GB free disk (CUDA torch ~4.3 GB + toolkit ~3 GB). Documented alternative:
`pipeline/views.render_ewa` - EWA splatting on CPU (NumPy): covariance from quaternion + anisotropic scales,
perspective Jacobian, 0.3 px^2 low-pass, 3-sigma footprint, global per-pixel depth sort, front-to-back alpha
compositing, SH degree 0 colour. Visibility is NOT read from RGB/expected depth: every (pixel, Gaussian) weight
w = alpha*T is stored (w >= 0.004) and masks aggregate those weights. Median depth used for gating. Disc renderer
kept as baseline. Cost: ~15 s per 768x576 view (11 M contributions) vs 0.3 s; ~40 MB buffers per view.
Sheets: `out_review/render_compare/<region>_v*.png` (clean RGB | depth | visibility | fragment overlay).

Visual result: equipment becomes recognisable. It also showed v1 errors: RV-01's "red handwheel valve" is a red
circulator pump and the "pumps" are blue valve actuators (v1 reference classes were wrong); RV-06's disc-era
"pump/motor/vessel" proposals were an insulated pipe, a desk fan and a cardboard box.

## Bugs fixed
- SAM input: one immutable clean render per view; overlays drawn on a separate copy; image embedding computed
  once and reused for every prompt (`propose.run_sam`).
- Whole-group claims now need support >= 0.6 AND observed coverage >= 0.5 AND spatial consistency (>= 80% of the
  group within 0.05 of supported voxels); otherwise only the observed supported subset moves and unseen voxels stay
  put (counted as `uncertain_unseen`; no more filling).
- Artifacts bound by hashes (render_key = renderer params + cameras + PLY checksum + load filters; proposal key =
  model + prompt + exact image bytes, plus cams_key; sam_key; seg_rev). `check_fresh` rejects stale combinations
  (verified: re-masking made 4 older associations stale and repair rejected them).
- Competing claims collected first, resolved per voxel by support with a 0.2 margin; ties stay with the original
  object; >10% lost/ambiguous -> `provisional`. Same-class claims with >= 30% overlap are corroboration and are
  united. Result does not depend on file order (tested).

## Frozen references v2 (`review/regions.yaml`, hash in `review/refs_v2.sha256`)
Set on EWA renders via `pipeline/review_refs.py` (reviewer pixel box -> median-depth unprojection, extended along the
view direction). 10 tuning refs (RV-01/02/03) + 4 HOLD-OUT refs in RV-10, set before any proposal/repair ran there.
Evaluation refuses to run if the reference blocks change.

## Measured (frozen v2 references)
| variant | objects | IoU tuning (10) | IoU hold-out (4) | mean leakage | accidental merges |
|---|---|---|---|---|---|
| baseline | 936 | 0.391 | 0.252 | 0.362 | 2 |
| geometry-only | 1007 | 0.393 | 0.299 | 0.335 | 2 |
| v1 context (old rules, disc, pilot proposals) | 1011 | 0.515 | 0.299 | 0.226 | 2 |
| fixed rules, disc masks, pilot proposals | 1010 | 0.447 | 0.299 | 0.276 | 2 |
| fixed rules, EWA masks, pilot (disc-era) proposals | 1009 | 0.470 | 0.155 | 0.321 | 2 |
| fixed rules, disc masks, EWA proposals | 1017 | 0.451 | 0.305 | 0.198 | 1 |
| **fixed rules, EWA masks, EWA proposals (out/)** | 1011 | 0.434 | **0.520** | **0.152** | **0** |

Hold-out: dark circulator RH-01 0.03 -> 0.65, blue valve RH-02 0.13 -> 0.57, red circulator RH-04 0.06 -> 0.32;
RH-03 actuator 0.79 -> 0.54 (worse). The v1 rules score higher on the tuning refs partly because they claimed whole
groups from little evidence - the behaviour fixed here; they were never tested on the hold-out with new proposals.

## Failures / limits
- RC2-02 (red circulator, RV-01): 0.04 - Qwen's EWA box included the white pipe below the pump, the mask followed it.
- Qwen grounding on EWA: RV-01 actuator boxes placed above the actuators; actuators still called "sensor".
- EWA masks with disc-era proposals hurt the hold-out (0.155): coarse boxes from unrecognisable images land on other
  objects once the image is clear. Proposals must come from the renderer they are masked on.
- 5 regions without EWA proposals (RV-03/05/07/08/09) get no contextual repair in out/ (conservative).
- RV-04 repair (now the navigation target object, "pump") has no reference: unverified boundary, class and identity.
- Single-layer approximation: SH > 0 ignored; Gaussians below the load filter (opacity < 0.1) are not rendered.
- Paid usage now 15 requests, ~$0.059 total (5 new EWA proposals ~$0.019); ledger `llm_ledger.jsonl`.
