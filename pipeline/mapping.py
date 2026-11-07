"""Stage 4: Excel assets -> extracted objects -> out/scene_graph.json, out/scene_anchors.csv

Inputs: data/builder_pack.xlsx, out/assets_labeled.json (or out/assets.json), out/path.json.
Rules (DOC-3D-MAPPING: explicit confidence, never silently guess identity):
  1. target asset (P-002) -> the object navgrid walked to       (method forced_demo_target)
  2. VLM label matches the asset class                          (label_match, 0.8-0.9)
  3. geometric rule from segment hints/features                 (geometry_rule, 0.55-0.7)
  4. any remaining unmapped object, nearest to the path, seeded (random_fallback, 0.3-0.5)
All other objects stay scene objects with status "no asset record".

Identity: a class-level match (VLM says "pump") does NOT establish which tag (P-002) the object is.
Every mapping therefore carries identity_verified=False and identity_basis, unless a user-confirmed
real-scene anchor or a readable tag is supplied (mapping.verified_identities, with provenance).
Workbook Scene_Anchors are synthetic placeholders (mapping_status=placeholder_until_valmet_scene_received):
their coordinates/dimensions are never used as scene positions; they are kept as workbook_* columns.
"""
from __future__ import annotations

import json
from typing import Any, Callable

import numpy as np
import pandas as pd

from pipeline.common import is_vision_label, load_config, out_path, repo_path

Cfg = dict[str, Any]
Obj = dict[str, Any]


# ---------- inputs ----------

def load_excel(cfg: Cfg) -> dict[str, pd.DataFrame]:
    return pd.read_excel(repo_path(cfg["paths"]["xlsx"]), sheet_name=None)


def load_objects(cfg: Cfg) -> tuple[list[Obj], str]:
    for name in ("assets_labeled.json", "assets.json"):
        p = out_path(cfg, name)
        if p.exists():
            with open(p, encoding="utf-8") as f:
                return json.load(f), name
    raise FileNotFoundError("run pipeline.segment first")


def load_path(cfg: Cfg) -> dict:
    with open(out_path(cfg, "path.json"), encoding="utf-8") as f:
        return json.load(f)


# ---------- candidate rules ----------

def path_distance(obj: Obj, path_xz: np.ndarray) -> float:
    """Closest x-z distance between the path and the object's box (0 if the path touches it)."""
    lo = np.array([obj["bbox_min"][0], obj["bbox_min"][2]])
    hi = np.array([obj["bbox_max"][0], obj["bbox_max"][2]])
    gap = np.maximum(np.maximum(lo - path_xz, path_xz - hi), 0)
    return float(np.linalg.norm(gap, axis=1).min())


def is_equipment(o: Obj, r: Cfg) -> bool:
    f = o["features"]
    return (o["hint"] not in ("pipe segment",) and "overhead" not in o["tags"]
            and f["height_bottom"] <= r["max_height_bottom"] and o["n_points"] >= r["min_points"]
            and max(o["extent"]) <= r["max_extent"])


def is_valve(o: Obj, r: Cfg) -> bool:
    return o["hint"] == "fitting/valve" and f_diag(o) <= r["max_diag"]


def is_small(o: Obj, r: Cfg) -> bool:
    return o["hint"] != "pipe segment" and f_diag(o) <= r["max_diag"] and o["n_points"] >= r["min_points"]


def f_diag(o: Obj) -> float:
    return float(o["features"]["diag"])


RULES: dict[str, Callable[[Obj, Cfg], bool]] = {
    "equipment": is_equipment, "valve": is_valve, "small": is_small}


# ---------- assignment ----------

class Mapper:
    def __init__(self, objs: list[Obj], path_xz: np.ndarray, mc: Cfg, cfg: Cfg | None = None) -> None:
        self.cfg = cfg or {}
        self.objs = {o["id"]: o for o in objs if o["kind"] == "object"}
        self.dist = {k: path_distance(o, path_xz) for k, o in self.objs.items()}
        self.mc = mc
        self.rng = np.random.default_rng(mc["seed"])
        self.used: dict[str, str] = {}            # object id -> asset id
        self.result: dict[str, dict] = {}         # asset id -> mapping

    def _conf(self, key: str) -> float:
        lo, hi = self.mc["confidence"][key]
        return round(float(self.rng.uniform(lo, hi)), 2)

    def label_conf(self, obj: Obj) -> float:
        """Label-match confidence follows the VLM's own confidence, bounded by the tier range."""
        lo = self.mc["confidence"]["geometry_rule"][0]
        hi = self.mc["confidence"]["label_match"][1]
        return round(float(np.clip(obj.get("label_confidence", lo), lo, hi)), 2)

    def assign(self, asset_id: str, obj_id: str, method: str, conf: float, evidence: str,
               verified: dict | None = None) -> None:
        self.used[obj_id] = asset_id
        basis = {"verified_identity": "user_confirmed", "label_match": "component_class_only",
                 "label_match_navigation_target": "component_class_only",
                 "geometry_rule": "geometry_only_synthetic", "random_fallback": "synthetic_placeholder",
                 "forced_demo_target": "synthetic_demo"}[method]
        self.result[asset_id] = {"object_id": obj_id, "mapping_method": method,
                                 "mapping_confidence": conf, "mapping_evidence": evidence,
                                 "identity_verified": verified is not None, "identity_basis": basis,
                                 "synthetic_assignment": basis.endswith(("synthetic", "placeholder", "demo")),
                                 "identity_provenance": verified}

    def vision_match(self, o: Obj, labels: Any) -> bool:
        return is_vision_label(o, self.cfg, labels, self.mc.get("vision_min_conf", 0.0))

    def free(self, pred: Callable[[Obj], bool]) -> list[str]:
        return sorted((k for k, o in self.objs.items() if k not in self.used and pred(o)),
                      key=lambda k: self.dist[k])

    def pick(self, cands: list[str], near_path: bool) -> str:
        """Nearest to the path for priority assets; seeded random among the near pool otherwise."""
        if near_path:
            return cands[0]
        pool = cands[: max(1, int(len(cands) * self.mc["near_path_pool"]))]
        return pool[int(self.rng.integers(len(pool)))]

    def map_asset(self, row: pd.Series, anchor_near: str | None = None) -> None:
        cls, aid, tag = row["asset_class"], row["asset_id"], row["tag"]
        rule = self.mc["class_rules"][cls]
        near = tag in self.mc["near_path_tags"]
        labels = set(rule["labels"])
        cands = self.free(lambda o: self.vision_match(o, labels))
        if cands:
            k = self.pick(cands, near)
            self.assign(aid, k, "label_match", self.label_conf(self.objs[k]),
                        f"VLM label '{self.objs[k]['label']}' matches {cls}")
            return
        r = self.mc["geometry"][rule["geometry"]]
        cands = self.free(lambda o: RULES[rule["geometry"]](o, r))
        if anchor_near is not None and cands:      # sensors: closest to their host machine
            host = np.array(self.objs[anchor_near]["center"])
            cands.sort(key=lambda k: np.linalg.norm(np.array(self.objs[k]["center"]) - host))
            near = True
        if cands:
            k = self.pick(cands, near)
            self.assign(aid, k, "geometry_rule", self._conf("geometry_rule"),
                        f"rule '{rule['geometry']}' (hint={self.objs[k]['hint']}, "
                        f"h_bottom={self.objs[k]['features']['height_bottom']:.2f})")
            return
        cands = self.free(lambda o: "overhead" not in o["tags"])
        k = self.pick(cands, near)
        self.assign(aid, k, "random_fallback", self._conf("random_fallback"),
                    f"no {cls}-like object left; nearest-to-path unmapped object (seed {self.mc['seed']})")


def run_mapping(xl: dict[str, pd.DataFrame], objs: list[Obj], path: dict, mc: Cfg,
                cfg: Cfg | None = None) -> Mapper:
    path_xz = np.array(path["points"])[:, [0, 2]]
    m = Mapper(objs, path_xz, mc, cfg)
    am = xl["Asset_Master"]
    for v in mc.get("verified_identities") or []:       # user-confirmed real-scene identities first
        k = nearest_object(m, v["point"]) if v.get("point") is not None else v.get("object_id")
        if k and k in m.objs:
            aid = am[am["tag"] == v["tag"]]["asset_id"].iloc[0]
            m.assign(aid, k, "verified_identity", float(v.get("confidence", 0.95)),
                     f"user-confirmed: {v.get('method')}", verified={k2: v[k2] for k2 in v if k2 != "confidence"})
    target = am[am["tag"] == mc["target_tag"]].iloc[0]
    tobj = m.objs[path["target_object"]]
    if target["asset_id"] in m.result:
        pass
    elif m.vision_match(tobj, mc["class_rules"][target["asset_class"]]["labels"]):
        m.assign(target["asset_id"], tobj["id"], "label_match_navigation_target", m.label_conf(tobj),
                 f"navigation target; VLM label '{tobj['label']}' ({tobj['label_confidence']}) "
                 f"matches {target['asset_class']}")
    else:
        m.assign(target["asset_id"], tobj["id"], "forced_demo_target",
                 mc["confidence"]["forced_demo_target"],
                 "placed on the object reached by the navigation path; not identified from geometry")
    order = {c: i for i, c in enumerate(mc["class_order"])}
    rows = sorted((r for _, r in am.iterrows() if r["asset_id"] not in m.result),
                  key=lambda r: (r["tag"] not in mc["near_path_tags"], order[r["asset_class"]]))
    cls_of = dict(zip(am["asset_id"], am["asset_class"]))
    n_sensor = 0
    for r in rows:
        host = None
        if r["asset_class"] in mc["sensor_classes"]:   # attach sensors to pumps/motors in turn
            hosts = [v["object_id"] for a, v in m.result.items() if cls_of[a] in mc["sensor_hosts"]]
            host = hosts[n_sensor % len(hosts)] if hosts else None
            n_sensor += 1
        m.map_asset(r, host)
    return m


def nearest_object(m: Mapper, point: list[float]) -> str | None:
    p = np.array(point, float)

    def d(o: Obj) -> float:
        lo, hi = np.array(o["bbox_min"]), np.array(o["bbox_max"])
        return float(np.linalg.norm(np.maximum(np.maximum(lo - p, p - hi), 0)))
    best = min(m.objs.values(), key=d)
    return best["id"] if d(best) <= 0.15 else None


# ---------- outputs ----------

def asset_record(row: pd.Series, state: pd.Series) -> dict:
    keys_m = ["asset_id", "tag", "asset_class", "area_name", "criticality"]
    keys_s = ["health_score", "condition_state", "issue_code", "issue_summary", "urgency",
              "recommended_next_check", "knowledge_doc_id"]
    rec = {k: row[k] for k in keys_m}
    rec.update({k: state[k] for k in keys_s})
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in rec.items()}


def scene_graph(objs: list[Obj], m: Mapper, xl: dict[str, pd.DataFrame], path: dict,
                cfg: Cfg, source: str) -> dict:
    am = xl["Asset_Master"].set_index("asset_id")
    cs = xl["Current_State"].set_index("asset_id")
    by_obj = {v["object_id"]: (a, v) for a, v in m.result.items()}
    nodes = []
    for o in objs:
        node = {k: o[k] for k in ("id", "kind", "hint", "tags", "center", "extent",
                                  "bbox_min", "bbox_max", "n_points", "color")}
        node["label"] = o.get("label") or o["hint"] or ("structure" if o["kind"] == "structure" else "object")
        for k in ("label_confidence", "label_source", "label_model", "label_reason", "label_vlm", "features",
                  "fragments", "baseline_ids", "repair", "assembly", "manual", "label_manual_class",
                  "label_stale", "label_before_manual", "manual_changes"):
            if k in o:
                node[k] = o[k]
        node["obb"] = o.get("obb", {})
        if o["id"] in by_obj:
            aid, mp = by_obj[o["id"]]
            node["status"] = "mapped"
            node["asset"] = {**asset_record(am.loc[aid].rename(None).to_frame().T.assign(asset_id=aid).iloc[0],
                                            cs.loc[aid]), **mp}
        else:
            node["status"] = "no asset record"
            node["asset"] = None
        nodes.append(node)
    tgt = m.result[am[am["tag"] == cfg["mapping"]["target_tag"]].index[0]]
    return {"scene": {"up_direction": cfg["scene"]["up_direction"], "floor_y": cfg["scene"]["floor_y"],
                      "units_to_m": cfg["scene"]["units_to_m"], "object_source": source,
                      "segmentation": path.get("segmentation"),
                      "identity_note": "asset identities are synthetic/unverified unless identity_verified=true"},
            "target": {"tag": cfg["mapping"]["target_tag"], "object_id": tgt["object_id"]},
            "path": {"file": "path.json", "length": path["length"], "min_clearance": path["min_clearance"]},
            "counts": {"objects": sum(o["kind"] == "object" for o in objs),
                       "structures": sum(o["kind"] == "structure" for o in objs),
                       "mapped": len(m.result)},
            "objects": nodes}


def scene_anchors(xl: dict[str, pd.DataFrame], m: Mapper, objs: list[Obj], cfg: Cfg) -> pd.DataFrame:
    sa = xl["Scene_Anchors"].copy()
    by_id = {o["id"]: o for o in objs}
    s = cfg["scene"]["units_to_m"]
    keep = [c for c in sa.columns if c.endswith("_m") or c in ("mapping_status", "mapping_confidence",
                                                                "identity_methods", "remap_instruction")]
    for c in keep:
        sa[f"workbook_{c}"] = sa[c]                       # synthetic placeholder values, kept for audit only
    sa = sa.astype(object)
    for i, r in sa.iterrows():
        mp = m.result[r["asset_id"]]
        o = by_id[mp["object_id"]]
        for ax, k in enumerate("xyz"):
            sa.loc[i, f"{k}_local_m"] = round(o["center"][ax] * s, 3)
            sa.loc[i, f"bbox_{k}_m"] = round(o["extent"][ax] * s, 3)
            sa.loc[i, f"scene_{k}_native"] = o["center"][ax]
        sa.loc[i, "scene_path"] = f"/BoilerRoom/{o['id']}"
        sa.loc[i, "mapping_status"] = f"mapped_{mp['mapping_method']}"
        sa.loc[i, "identity_methods"] = mp["mapping_method"]
        sa.loc[i, "mapping_confidence"] = mp["mapping_confidence"]
        sa.loc[i, "remap_instruction"] = mp["mapping_evidence"]
        sa.loc[i, "object_id"] = o["id"]
        sa.loc[i, "identity_verified"] = mp["identity_verified"]
        sa.loc[i, "identity_basis"] = mp["identity_basis"]
    return sa


def print_summary(m: Mapper, xl: dict[str, pd.DataFrame], mc: Cfg) -> None:
    am = xl["Asset_Master"].set_index("asset_id")
    cs = xl["Current_State"].set_index("asset_id")
    methods: dict[str, int] = {}
    for v in m.result.values():
        methods[v["mapping_method"]] = methods.get(v["mapping_method"], 0) + 1
    print(f"[map] {len(m.result)}/{len(am)} assets mapped  methods {methods}")
    for aid, v in sorted(m.result.items(), key=lambda kv: m.dist[kv[1]["object_id"]]):
        tag = am.loc[aid, "tag"]
        flag = " <- TARGET" if tag == mc["target_tag"] else (" (watch)" if tag in mc["near_path_tags"] else "")
        print(f"  {tag:7s} {am.loc[aid, 'asset_class']:21s} -> {v['object_id']}  "
              f"conf {v['mapping_confidence']:.2f}  {v['mapping_method']:18s} {v['identity_basis']:24s} "
              f"path_dist {m.dist[v['object_id']]:.2f}  {cs.loc[aid, 'condition_state']}{flag}")


def run(cfg: Cfg) -> dict:
    mc = cfg["mapping"]
    xl = load_excel(cfg)
    objs, source = load_objects(cfg)
    path = load_path(cfg)
    m = run_mapping(xl, objs, path, mc, cfg)
    assert len(m.result) == len(xl["Asset_Master"]), "not all assets mapped"
    assert len(set(v["object_id"] for v in m.result.values())) == len(m.result), "object mapped twice"
    sg = scene_graph(objs, m, xl, path, cfg, source)
    with open(out_path(cfg, "scene_graph.json"), "w", encoding="utf-8") as f:
        json.dump(sg, f, indent=1, default=str)
    scene_anchors(xl, m, objs, cfg).to_csv(out_path(cfg, "scene_anchors.csv"), index=False)
    print_summary(m, xl, mc)
    print(f"[out] scene_graph.json ({sg['counts']}), scene_anchors.csv  objects from {source}")
    return sg


if __name__ == "__main__":
    run(load_config())
