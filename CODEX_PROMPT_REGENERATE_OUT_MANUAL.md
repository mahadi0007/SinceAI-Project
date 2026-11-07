# Task: regenerate the stale downstream artifacts of segmentation version `out_manual/`

Repo: `C:\Users\Usuario\Projects\hackathon\SinceAI-Project` (Windows, PowerShell). Use ONLY `.venv\Scripts\python.exe`.
Work on the current working tree (uncommitted changes are intentional; do not revert, stash or reformat them). Do not commit.

## Context (verified facts, do not re-derive)

- `out/` = current base segmentation (1011 objects). It is READ-ONLY for this task.
- `out_manual/` = a separate segmentation version produced by `pipeline/correct.py:apply()` from `out/` plus 8 accepted
  manual boundary corrections (reviewer-drawn SAM masks + contribution-weighted visible support). It currently contains
  only `segment_points.npz`, `assets.json`, `segment_manifest.json`, `manual_manifest.json`. It has 1019 objects.
- `out_manual/manual_manifest.json` lists `downstream_stale`: `assets_labeled.json, scene_graph.json, scene_anchors.csv,
  path.json, navgrid.npz, navgrid.png, tour.json, tour.png, label_contact_sheet.png`. `pipeline/correct.py:downstream_stale(dir, name)`
  returns a reason while a listed file is missing or older than `manifest["applied_at"]`; `viewer/app.py:load_scene` refuses to start on it.
- All pipeline stages read/write through `pipeline.common.out_path(cfg, name)`, which honours env `VALMET_OUT`. Setting
  `VALMET_OUT=out_manual` redirects every stage to `out_manual/`.
- Object IDs: unchanged base objects keep their `OBJ-xxxx` id from `out/`. The 8 new manual objects have ids `MC-0004 ..
  MC-0012` (MC-0009 and MC-0013 do not exist as objects: rejected). 13 base objects lost voxels to them and were rebuilt:
  `OBJ-0012, OBJ-0016, OBJ-0024, OBJ-0045, OBJ-0048, OBJ-0108, OBJ-0152, OBJ-0262, OBJ-0327, OBJ-0420, OBJ-0697, OBJ-0776, OBJ-1016`.
  In `out_manual/assets.json` these 13 have NO `label` field, `label_stale: true`, and `label_before_manual` (their old label).
- `out_manual/assets.json` state of labels (objects only): 990 `geometry_rules`, 8 `featherless_vision` (unchanged objects;
  their records still carry `vlm_proposal`, which is where those labels came from), 8 `manual_review`, 13 with no label.
- The 8 manual objects carry `label` = reviewer `component_class` (`pump`, `motor` or `valve`), `label_source: "manual_review"`,
  and a `manual` block: `component_id, revision, boundary_status, evidence_level, component_class, assembly_id, asset
  {tag: null, status: "unassigned"}, membership [...]`. Two have `evidence_level: "views-disagree"` and
  `boundary_status: "accepted-provisional"` (MC-0010, MC-0011).
- There is NO `label_cache.json` in `out/`. The base labels came from geometry rules + `vlm_proposal`.

## Problem with simply re-running the stages

`python -m pipeline.label` rebuilds every label via `decide(a, g, lc)` in `pipeline/label.py` from (label cache | `vlm_proposal` |
geometry rules). It ignores `label_source: "manual_review"`, so the 8 manual classes would be overwritten by geometry rules
(`other` / `pipe fitting/flange`). Also `"valve"` is not in `cfg["label"]["vocab"]`, and `pipeline.common.is_vision_label` (used by
`mapping.py` label_match and `navgrid.py` target fallback) only accepts `label.vision_sources`, so manual classes would never count.
Running label WITHOUT `--rules-only` would also spend paid Featherless requests: forbidden for this task.

## Required code changes (minimal, tested)

1. `config.yaml`, under `label:` add:
   ```yaml
   manual_sources: [manual_review]       # reviewer-confirmed component CLASS (never asset identity)
   manual_class_vocab:                   # correction-tool class -> label vocab
     pump: pump
     motor: motor
     control valve: control valve
     sensor: pressure transmitter/gauge
     vessel: tank/vessel
     pipe run: pipe segment
   manual_label_overrides:               # per component id, reviewer decision (blue electric actuators = control valves)
     MC-0006: control valve
     MC-0007: control valve
     MC-0008: manual valve/handwheel
   ```
   Classes `unknown`/`other`, or a `valve` without an override, must NOT become a confident label: fall back to the
   normal decide() path (geometry rules) and keep the reviewer class in `label_manual_class`.

2. `pipeline/label.py`: in `run()`, before `a.update(decide(a, g, lc))`, if `a.get("label_source") in lc["manual_sources"]`
   and `a.get("manual")`: resolve `label = manual_label_overrides.get(a["id"]) or manual_class_vocab.get(a["manual"]["component_class"])`.
   If resolved and in `lc["vocab"]`, set `label`, `label_confidence: 1.0`, `label_source: "manual_review"`,
   `label_reason: f"reviewer boundary correction {cid} rev {rev} ({evidence_level})"` and skip decide(). Never write
   asset identity here. Objects with `manual` must also be excluded from the paid `todo` list (they need no VLM call).
   Keep `--rules-only` behaviour otherwise identical.

3. `pipeline/common.py:is_vision_label`: also accept sources in `cfg["label"].get("manual_sources", [])` (same label/conf
   checks). Do not rename the function; update its docstring. Downstream callers: `mapping.py:120`, `navgrid.py:281`.

4. `pipeline/mapping.py:scene_graph()` (around line 231) copies only selected object fields into scene-graph nodes. Add
   `"manual", "label_manual_class", "label_stale", "label_before_manual", "manual_changes"` to the tuple
   `("label_confidence", "label_source", ..., "repair", "assembly")` so provisional status and membership history reach the
   viewer. Do not change mapping rules, seeds or confidences.

5. Tests (`tests/test_pipeline.py` or a new `tests/test_manual_labels.py`):
   - manual pump -> label `pump`, conf 1.0, source `manual_review`; manual `valve` with override -> override; manual `valve`
     without override and manual `unknown` -> geometry-rule label with `label_manual_class` kept;
   - `is_vision_label` true for a `manual_review` pump with `labels=["pump"], min_conf=0.8`, false for `unknown`;
   - manual objects are never in the paid `todo` list.
   Run: `.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp <a temp dir>` (the default pytest temp dir is not
   writable in this sandbox). All existing tests (33) must still pass.

## Regeneration (PowerShell, in this exact order, all offline)

```powershell
$env:VALMET_OUT = "out_manual"
$env:VALMET_OFFLINE = "1"          # every LLM call raises LLMUnavailable -> no paid requests possible
.venv\Scripts\python.exe -m pipeline.label --rules-only     # -> assets_labeled.json, crops/, label_contact_sheet.png
.venv\Scripts\python.exe -m pipeline.navgrid                # -> navgrid.npz, navgrid.png, path.json
.venv\Scripts\python.exe -m pipeline.mapping                # -> scene_graph.json, scene_anchors.csv
.venv\Scripts\python.exe -m pipeline.tour                   # -> tour.json, tour.png
```

Expected: label prints `8/1019 from vision LLM` style counts (featherless_vision 8 + manual_review 8 counted if you extend that
print; report both separately). navgrid uses `navgrid.target.point` (config), so the target must resolve to the same object as
in `out/` (`OBJ-0086`, unchanged by the corrections); if it does not, STOP and report. Mapping uses `mapping.seed: 42`; its
random-fallback assignments may legitimately move because the object set changed: report them, do not tune anything.

## Hard constraints

- Do NOT modify or regenerate: `out/`, `out_baseline/`, `out_geo/`, `out_ctx_*`, `review/` (especially `review/corrections/`,
  the live correction store), the PLY / xlsx under `../sinceai2026/data/raw/`, `.env`, `llm_ledger.jsonl`.
- Do NOT run `pipeline.segment`, `pipeline.repair`, `pipeline.propose`, `pipeline.correct.apply`, or the correction tool's Apply
  (Apply deletes the downstream files you are creating, by design). A correction-tool server may be running on port 8094; leave it alone.
- No paid requests: keep `VALMET_OFFLINE=1` for every command, including any viewer start. Do not change `copilot.budget`.
- Do NOT copy `out/diagnosis_cache`, `out/voice_cache`, `out/diagnosis_*.json`, `out/evidence_*.json` or `out/crops` into
  `out_manual/`: they are bound to the old object set.
- Manual corrections establish component CLASS and boundary only. Never set `identity_verified`, never map a tag because of a
  manual class, never write `manual.asset.tag`. Assembly ids (`RV04-UNIT-R`, `RV04-UNIT-L`) are not asset identity.
- Keep provisional status visible: MC-0010 / MC-0011 must keep `manual.boundary_status = accepted-provisional` in every
  downstream record that copies object fields (scene_graph objects), not be upgraded.

## Validation (all must pass; paste the output)

Save this as `check_out_manual.py` in a temp folder OUTSIDE the repo, then run it from the repo root:
`$env:VALMET_OUT="out_manual"; $env:VALMET_OFFLINE="1"; $env:PYTHONPATH="."; .venv\Scripts\python.exe <temp>\check_out_manual.py`

```python
import json, numpy as np
from pathlib import Path
from pipeline.correct import downstream_stale, DOWNSTREAM
from pipeline.scene import seg_revision
d = Path("out_manual")
man = json.loads((d / "manual_manifest.json").read_text())
assert seg_revision(d) == man["seg_rev"], "out_manual segmentation changed: someone re-applied or edited it"
assert seg_revision("out") == man["base_seg_rev"], "out/ was modified"
for n in DOWNSTREAM:
    assert downstream_stale(d, n) is None, downstream_stale(d, n)
s = np.load(d / "segment_points.npz"); lab = json.loads((d / "assets_labeled.json").read_text())
assert [a["id"] for a in lab] == list(s["label_ids"]) and len(lab) == 1025       # 1019 objects + 6 structures
mc = {a["id"]: a for a in lab if a["id"].startswith("MC-")}
assert sorted(mc) == ["MC-0004","MC-0005","MC-0006","MC-0007","MC-0008","MC-0010","MC-0011","MC-0012"]
exp = {"MC-0004":"pump","MC-0005":"pump","MC-0006":"control valve","MC-0007":"control valve",
       "MC-0008":"manual valve/handwheel","MC-0010":"motor","MC-0011":"pump","MC-0012":"motor"}
assert all(mc[k]["label"] == v and mc[k]["label_source"] == "manual_review" for k, v in exp.items())
assert all(a.get("label") for a in lab if a["kind"] == "object")                    # no unlabelled object left
sg = json.loads((d / "scene_graph.json").read_text()); ids = {a["id"] for a in lab}
assert sg["counts"]["objects"] == 1019 and all(o["id"] in ids for o in sg["objects"])
path = json.loads((d / "path.json").read_text()); assert path["target_object"] == "OBJ-0086", path["target_object"]
assert sg["target"]["object_id"] == path["target_object"]
tour = json.loads((d / "tour.json").read_text())
assert all(st["object_id"] in ids for st in tour["stops"]) and tour["stops"][-1]["object_id"] == path["target_object"]
assert not any((o.get("asset") or {}).get("identity_verified") for o in sg["objects"] if o["id"].startswith("MC-"))
nodes = {o["id"]: o for o in sg["objects"]}
assert all(nodes[k]["manual"]["boundary_status"] == "accepted-provisional" for k in ("MC-0010", "MC-0011"))
print("OK")
```

Then start the viewer once on the new version to prove it loads, and stop it:
`$env:VALMET_OUT="out_manual"; $env:VALMET_OFFLINE="1"; .venv\Scripts\python.exe -m viewer.app` -> must print
`[viewer] http://localhost:<port>  (1019 objects, ...)` with no stale-artifact SystemExit. (Port may differ: it picks a free one.)

## Report back

1. Diff of code changes (config, label.py, common.py, tests) and the pytest summary line.
2. Label counts by `label_source` for `out_manual/assets_labeled.json`, and the labels of the 13 rebuilt base objects
   (old `label_before_manual` -> new label).
3. Mapping diff `out/scene_graph.json` vs `out_manual/scene_graph.json`: for every workbook tag, old object id -> new object id,
   method and confidence; flag every tag whose object changed and whether it now sits on an `MC-` object. Expected
   attention points: tags previously on OBJ-0262 (CV-003), OBJ-0152 (P-001), OBJ-0776 (PT-001), which lost voxels.
4. Tour stops old vs new.
5. Validation script output and the viewer start line. Anything that failed or was skipped, stated plainly.
