"""Stage 7 (PoC): scene graph -> IFC4 BIM model -> out/scene.ifc

Inputs: out/scene_graph.json, out/segment_points.npz.
Pipe runs become fitted cylinders: PCA axis snapped to vertical / room axes, least-squares circle fit
across the axis (handles half-seen pipes), radius snapped to the nearest standard DN size, and ends
extended to meet a neighbouring pipe they nearly touch. Pumps, valves, motors and sensors become parametric
catalog shapes (pipeline/ifc_families.py) fitted to their upright box; fittings can become flange pairs snapped
onto the nearest pipe (ifc.fittings_as_flanges). Everything else, including noisy pipe fits, becomes a mesh of its own scan points,
so every scan object is in the model. Mapped Excel data goes into Pset_ValmetAsset.
Colours come from the scan: scan meshes get per-face colours (IfcIndexedColourMap), other elements one
surface style with their points' median colour.
Deviation: each scan point's distance to its element's surface -> out/deviation.npz; per-element mean / p90
in Pset_ScanExtraction.
Axes: scene (x, y, z) with up = -y  ->  IFC (x, z, -y) in metres, IFC up = +Z.
"""
from __future__ import annotations

import json
import os
from typing import Any

import ifcopenshell
import ifcopenshell.api
import ifcopenshell.geom
import ifcopenshell.guid
import ifcopenshell.util.element
import numpy as np
import open3d as o3d
from scipy import ndimage
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes

from pipeline.bim_provenance import backup, label_revision, scene_revision, write_bim_manifest
from pipeline.common import load_config, out_path
from pipeline.ifc_families import FAMILIES, flange

Cfg = dict[str, Any]

IFC_CLASS = {
    "PROCESS_PUMP": "IfcPump",
    "CONTROL_VALVE": "IfcValve",
    "ELECTRIC_MOTOR": "IfcElectricMotor",
    "PRESSURE_TRANSMITTER": "IfcSensor",
    "VIBRATION_SENSOR": "IfcSensor",
    "pipe segment": "IfcPipeSegment",
    "pipe fitting/flange": "IfcPipeFitting",
}
FLOOR_STANDING = {"IfcPump", "IfcElectricMotor"}
DN_OD_MM = [("DN15", 21.3), ("DN20", 26.9), ("DN25", 33.7), ("DN32", 42.4), ("DN40", 48.3), ("DN50", 60.3),
            ("DN65", 76.1), ("DN80", 88.9), ("DN100", 114.3), ("DN125", 139.7), ("DN150", 168.3),
            ("DN200", 219.1), ("DN250", 273.0), ("DN300", 323.9), ("DN350", 355.6), ("DN400", 406.4),
            ("DN500", 508.0)]
MAX_RADIUS_GAIN = 1.6
ASSET_FIELDS = ["asset_id", "tag", "asset_class", "area_name", "criticality", "health_score",
                "condition_state", "issue_code", "issue_summary", "urgency", "recommended_next_check",
                "mapping_method", "mapping_confidence"]


# scene axes -> IFC axes (a proper rotation, det +1): ifc = u2m * AXES @ scene
AXES = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])


def to_ifc(v: np.ndarray, u2m: float = 1.0) -> np.ndarray:
    """Scene coords (up = -y) -> IFC coords (up = +z), scaled to metres."""
    v = np.asarray(v, np.float64)
    return np.stack([v[..., 0], v[..., 2], -v[..., 1]], -1) * u2m


def from_ifc(v: np.ndarray, u2m: float = 1.0) -> np.ndarray:
    """IFC coords (up = +z, metres) -> scene coords (up = -y, scan units); inverse of to_ifc."""
    v = np.asarray(v, np.float64) / u2m
    return np.stack([v[..., 0], -v[..., 2], v[..., 1]], -1)


def rot_to_ifc(R: np.ndarray) -> np.ndarray:
    """Rotation matrix in scene axes -> the same rotation in IFC axes."""
    return AXES @ np.asarray(R, np.float64) @ AXES.T


def rot_from_ifc(R: np.ndarray) -> np.ndarray:
    return AXES.T @ np.asarray(R, np.float64) @ AXES


def snap_axis(v: np.ndarray, deg: float) -> np.ndarray:
    """Snap a direction to vertical, else to horizontal, then to the room's x / y axis, when within deg."""
    v = v / np.linalg.norm(v)
    cos, sin = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    if abs(v[2]) >= cos:
        return np.array([0, 0, 1.0])
    if abs(v[2]) > sin:
        return v
    h = np.array([v[0], v[1], 0.0]) / np.hypot(v[0], v[1])
    for a in (np.array([1.0, 0, 0]), np.array([0, 1.0, 0])):
        if abs(h @ a) >= cos:
            return a
    return h


def fit_circle(uv: np.ndarray) -> tuple[np.ndarray, float]:
    """Robust geometric circle fit (Kasa init); keeps the centroid/median fit if that is better."""
    def rms(c: np.ndarray, r: float) -> float:
        return float(np.sqrt(np.mean((np.linalg.norm(uv - c, axis=1) - r) ** 2)))

    best = (np.zeros(2), float(np.median(np.linalg.norm(uv, axis=1))))
    r_max = float(np.max(np.ptp(uv, axis=0)))
    A = np.c_[2 * uv, np.ones(len(uv))]
    a, b, k = np.linalg.lstsq(A, (uv ** 2).sum(1), rcond=None)[0]
    r0 = float(np.sqrt(max(k + a * a + b * b, 1e-6)))
    x0 = np.array([a, b, np.clip(r0, 0.005, r_max)]) if np.isfinite(r0) else np.r_[best[0], best[1]]
    if r_max > 0.005:
        res = least_squares(lambda p: np.linalg.norm(uv - p[:2], axis=1) - p[2], np.clip(x0, [-np.inf, -np.inf, 0.005],
                            [np.inf, np.inf, r_max]), bounds=([-np.inf, -np.inf, 0.005], [np.inf, np.inf, r_max]),
                            loss="soft_l1", f_scale=0.005)
        # a half-seen pipe's true radius is ~1.25x the centroid median; much larger = flat-arc blow-up
        if res.success and res.x[2] <= MAX_RADIUS_GAIN * best[1] and rms(res.x[:2], res.x[2]) < rms(*best):
            best = (res.x[:2], float(res.x[2]))
    return best


def snap_dn(r: float) -> tuple[str, float]:
    name, od = min(DN_OD_MM, key=lambda d: abs(np.log(d[1] / 2000 / r)))
    return name, od / 2000


def fit_cylinder(xyz: np.ndarray, snap_deg: float) -> dict[str, Any]:
    """PCA axis (snapped), circle fit across it, DN-snapped radius; RMS of |d - r| against the model."""
    c = xyz.mean(0)
    _, _, vt = np.linalg.svd(xyz - c, full_matrices=False)
    axis = snap_axis(vt[0], snap_deg)
    u = perpendicular(axis)
    v = np.cross(axis, u)
    rel = xyz - c
    t = rel @ axis
    uv = np.c_[rel @ u, rel @ v]
    cc, r_fit = fit_circle(uv)
    dn, r = snap_dn(r_fit)
    d = np.linalg.norm(uv - cc, axis=1)
    t0, t1 = np.percentile(t, [1, 99])
    centre = c + cc[0] * u + cc[1] * v
    return {"start": centre + t0 * axis, "axis": axis, "length": float(t1 - t0), "radius": r, "fit_radius": r_fit,
            "dn": dn, "rms": float(np.sqrt(np.mean((d - r) ** 2)))}


def join_pipes(pipes: list[dict[str, Any]], gap: float) -> int:
    """Extend pipe ends to the axis of a non-parallel pipe they nearly touch (elbows, tees). Returns joins."""
    S = np.array([p["start"] for p in pipes])
    D = np.array([p["axis"] for p in pipes])
    L = np.array([p["length"] for p in pipes])
    R = np.array([p["radius"] for p in pipes])
    moves: list[tuple[int, int, float]] = []
    for i in range(len(pipes)):
        for end in (0, 1):
            E = S[i] + end * L[i] * D[i]
            w = E - S
            tb = np.clip((w * D).sum(1), 0, L)
            dist = np.linalg.norm(w - tb[:, None] * D, axis=1) - R - R[i]
            dist[i] = np.inf
            dist[np.abs(D @ D[i]) > np.cos(np.radians(10))] = np.inf      # parallel: not a joint
            j = int(np.argmin(dist))
            if dist[j] > gap:
                continue
            w0, b = S[i] - S[j], D[i] @ D[j]
            d, e = D[i] @ w0, D[j] @ w0
            sa, sb = (b * e - d) / (1 - b * b), (e - b * d) / (1 - b * b)
            if np.linalg.norm(w0 + sa * D[i] - sb * D[j]) > R[i] + R[j] or not -gap <= sb <= L[j] + gap:
                continue                                                    # axes pass each other: no joint
            delta = sa - L[i] if end else -sa                               # > 0 extends
            if abs(delta) <= gap + R[j]:
                moves.append((i, end, float(delta)))
    for i, end, delta in moves:
        p = pipes[i]
        if end == 0:
            p["start"] = p["start"] - delta * p["axis"]
        p["length"] += delta
    for i, p in enumerate(pipes):                                           # trims must not collapse a pipe
        if p["length"] < 0.5 * L[i]:
            p["start"], p["length"] = S[i], float(L[i])
    return len(moves)


def upright_box(xyz: np.ndarray, snap_deg: float) -> dict[str, Any]:
    """Gravity-aligned box: yaw from 2D PCA (snapped to room axes), 1-99 percentile extents."""
    xy = xyz[:, :2] - xyz[:, :2].mean(0)
    _, _, vt = np.linalg.svd(xy, full_matrices=False) if len(xyz) >= 3 else (None, None, np.eye(2))
    x = snap_axis(np.array([vt[0][0], vt[0][1], 0.0]), snap_deg)
    z = np.array([0, 0, 1.0])
    y = np.cross(z, x)
    local = xyz @ np.stack([x, y, z], 1)
    lo, hi = np.percentile(local, [1, 99], axis=0)
    dims = np.maximum(hi - lo, 0.01)
    mid = (lo + hi) / 2
    origin = mid[0] * x + mid[1] * y + lo[2] * z            # bottom centre
    return {"origin": origin, "x": x, "z": z, "dims": dims}


def scan_mesh(xyz: np.ndarray, voxel: float, max_tris: int) -> tuple[np.ndarray, np.ndarray]:
    """Surface of the object's own points: voxel occupancy, closed and filled, smoothed, marching cubes,
    decimated. Falls back to the convex hull for tiny objects."""
    lo = xyz.min(0) - 3 * voxel
    idx = np.floor((xyz - lo) / voxel).astype(int)
    occ = np.zeros(idx.max(0) + 4, bool)
    occ[tuple(idx.T)] = True
    occ = ndimage.binary_fill_holes(ndimage.binary_closing(occ, iterations=1))
    field = ndimage.gaussian_filter(occ.astype(np.float32), 0.5)
    try:
        v, f, _, _ = marching_cubes(field, 0.5)
    except (ValueError, RuntimeError):
        v = f = np.zeros((0, 3))
    if len(f) == 0:
        hull = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz)).compute_convex_hull(joggle_inputs=True)[0]
        return np.asarray(hull.vertices), np.asarray(hull.triangles)
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector((v + 0.5) * voxel + lo),
                                  o3d.utility.Vector3iVector(f.astype(np.int32)))
    if len(f) > max_tris:
        m = m.simplify_quadric_decimation(max_tris)
    m.remove_degenerate_triangles()
    m.remove_unreferenced_vertices()
    return np.asarray(m.vertices), np.asarray(m.triangles)


def face_colours(verts: np.ndarray, tris: np.ndarray, xyz: np.ndarray, rgb: np.ndarray, k: int = 6) -> np.ndarray:
    """Mean colour of the k scan points nearest each face centre."""
    _, nn = cKDTree(xyz).query(verts[tris].mean(1), k=min(k, len(xyz)))
    return rgb[nn.reshape(len(tris), -1)].mean(1)


def perpendicular(z: np.ndarray) -> np.ndarray:
    a = np.array([1.0, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1.0, 0])
    x = np.cross(z, a)
    return x / np.linalg.norm(x)


class Writer:
    def __init__(self, name: str) -> None:
        f = self.f = ifcopenshell.api.run("project.create_file", version="IFC4")
        self.project = ifcopenshell.api.run("root.create_entity", f, ifc_class="IfcProject", name=name)
        metre = ifcopenshell.api.run("unit.add_si_unit", f, unit_type="LENGTHUNIT")
        ifcopenshell.api.run("unit.assign_unit", f, units=[metre])
        model = ifcopenshell.api.run("context.add_context", f, context_type="Model")
        self.body = ifcopenshell.api.run("context.add_context", f, context_type="Model",
                                         context_identifier="Body", target_view="MODEL_VIEW", parent=model)
        site = ifcopenshell.api.run("root.create_entity", f, ifc_class="IfcSite", name="Site")
        bldg = ifcopenshell.api.run("root.create_entity", f, ifc_class="IfcBuilding", name="Plant")
        self.storey = ifcopenshell.api.run("root.create_entity", f, ifc_class="IfcBuildingStorey",
                                           name="Equipment Room")
        ifcopenshell.api.run("aggregate.assign_object", f, relating_object=self.project, products=[site])
        ifcopenshell.api.run("aggregate.assign_object", f, relating_object=site, products=[bldg])
        ifcopenshell.api.run("aggregate.assign_object", f, relating_object=bldg, products=[self.storey])

    def _axis3d(self, origin: np.ndarray, z: np.ndarray, x: np.ndarray) -> Any:
        f = self.f
        return f.createIfcAxis2Placement3D(f.createIfcCartesianPoint([float(v) for v in origin]),
                                           f.createIfcDirection([float(v) for v in z]),
                                           f.createIfcDirection([float(v) for v in x]))

    def _extrusion(self, profile: Any, depth: float, base: np.ndarray | None = None,
                   z: np.ndarray | None = None) -> Any:
        f = self.f
        z = np.array([0, 0, 1.0]) if z is None else z
        return f.createIfcExtrudedAreaSolid(profile, self._axis3d(np.zeros(3) if base is None else base, z,
                                                                  perpendicular(z)),
                                            f.createIfcDirection([0.0, 0.0, 1.0]), float(depth))

    def element(self, ifc_class: str, name: str, origin: np.ndarray, z: np.ndarray, x: np.ndarray,
                solid: Any, psets: dict[str, dict[str, Any]], rgb: np.ndarray | None = None) -> Any:
        f = self.f
        el = ifcopenshell.api.run("root.create_entity", f, ifc_class=ifc_class, name=name)
        el.ObjectPlacement = f.createIfcLocalPlacement(None, self._axis3d(origin, z, x))
        items = solid if isinstance(solid, list) else [solid]
        kind = "Tessellation" if items[0].is_a("IfcTriangulatedFaceSet") else "SweptSolid"
        if rgb is not None:
            style = f.createIfcSurfaceStyle(name, "BOTH", [f.createIfcSurfaceStyleRendering(
                f.createIfcColourRgb(None, *[float(c) for c in rgb]), 0.0, None, None, None, None, None, None,
                "NOTDEFINED")])
            for it in items:
                f.createIfcStyledItem(it, [style], None)
        rep = f.createIfcShapeRepresentation(self.body, "Body", kind, items)
        el.Representation = f.createIfcProductDefinitionShape(None, None, [rep])
        ifcopenshell.api.run("spatial.assign_container", f, relating_structure=self.storey, products=[el])
        for pname, props in psets.items():
            ps = ifcopenshell.api.run("pset.add_pset", f, product=el, name=pname)
            ifcopenshell.api.run("pset.edit_pset", f, pset=ps,
                                 properties={k: v for k, v in props.items() if v is not None})
        return el

    def cylinder(self, r: float, length: float, base: np.ndarray | None = None,
                 axis: np.ndarray | None = None) -> Any:
        f = self.f
        prof = f.createIfcCircleProfileDef("AREA", None, f.createIfcAxis2Placement2D(
            f.createIfcCartesianPoint([0.0, 0.0])), float(max(r, 0.002)))
        return self._extrusion(prof, max(length, 0.002), base, axis)

    def box(self, dx: float, dy: float, dz: float, base: np.ndarray | None = None) -> Any:
        f = self.f
        prof = f.createIfcRectangleProfileDef("AREA", None, f.createIfcAxis2Placement2D(
            f.createIfcCartesianPoint([0.0, 0.0])), float(max(dx, 0.002)), float(max(dy, 0.002)))
        return self._extrusion(prof, max(dz, 0.002), base)

    def mesh(self, verts: np.ndarray, tris: np.ndarray, colours: np.ndarray | None = None) -> Any:
        f = self.f
        fs = f.createIfcTriangulatedFaceSet(f.createIfcCartesianPointList3D(verts.round(4).tolist()), None, None,
                                            (tris + 1).tolist(), None)
        if colours is not None:                                            # per face, palette of 5-bit colours
            q = np.clip(np.round(colours * 31), 0, 31).astype(int)
            pal, idx = np.unique(q, axis=0, return_inverse=True)
            f.createIfcIndexedColourMap(fs, 0.0, f.createIfcColourRgbList((pal / 31).tolist()),
                                        (idx.ravel() + 1).tolist())
        return fs

    def parts(self, parts: list[tuple]) -> list[Any]:
        """Family parts (see ifc_families) -> solids in the element's local frame."""
        return [self.cylinder(p[3], p[4], p[1], p[2]) if p[0] == "cyl" else self.box(*p[2], base=p[1])
                for p in parts]


def deviation(f: Any, elements: dict[int, Any], xyz: np.ndarray, lab: np.ndarray, u2m: float) -> np.ndarray:
    """Distance (m) from each scan point to its own element's tessellated surface; nan if not modelled.
    Writes mean / p90 per element into Pset_ScanExtraction."""
    s = ifcopenshell.geom.settings()
    s.set("use-world-coords", True)
    it = ifcopenshell.geom.iterator(s, f, os.cpu_count() or 1)
    it.initialize()
    shapes = {}
    while True:
        sh = it.get()
        shapes[sh.id] = sh.geometry
        if not it.next():
            break
    dev = np.full(len(xyz), np.nan)
    for i, el in elements.items():
        g = shapes.get(el.id())
        sel = np.where(lab == i)[0]
        if g is None or not len(sel):
            continue
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(np.array(g.verts, np.float32).reshape(-1, 3), np.array(g.faces, np.uint32).reshape(-1, 3))
        d = scene.compute_distance(o3d.core.Tensor(to_ifc(xyz[sel], u2m).astype(np.float32))).numpy()
        dev[sel] = d
        ps = ifcopenshell.util.element.get_psets(el)["Pset_ScanExtraction"]
        ifcopenshell.api.run("pset.edit_pset", f, pset=f.by_id(ps["id"]),
                             properties={"DeviationMeanMm": round(float(d.mean()) * 1000, 1),
                                         "DeviationP90Mm": round(float(np.percentile(d, 90)) * 1000, 1)})
    return dev


def export(cfg: Cfg) -> dict[str, Any]:
    u2m, ic = cfg["scene"]["units_to_m"], cfg["ifc"]
    with open(out_path(cfg, "scene_graph.json"), encoding="utf-8") as fh:
        sg = json.load(fh)
    seg = np.load(out_path(cfg, "segment_points.npz"))
    xyz, lab, rgb = seg["xyz"].astype(np.float64), seg["labels"], np.clip(seg["rgb"], 0, 1).astype(np.float64)
    order = np.argsort(lab, kind="stable")
    bounds = np.searchsorted(lab[order], np.arange(len(sg["objects"]) + 1))

    rev = scene_revision(cfg, lab)
    w = Writer("Valmet Equipment Room")
    prov = ifcopenshell.api.run("pset.add_pset", w.f, product=w.project, name="Pset_ScanProvenance")
    ifcopenshell.api.run("pset.edit_pset", w.f, pset=prov, properties={
        "SegmentationRevision": rev["segmentation"], "SourcePlySha256": rev["ply_sha256"],
        "ObjectCount": rev["n_objects"], "UnitsToMetres": float(u2m),
        "CoordinateMap": "IFC (x, y, z) = UnitsToMetres * scene (x, z, -y); scene up = -y"})
    stats: dict[str, Any] = {"pipes": 0, "meshes": 0, "families": 0, "flanges": 0, "noisy_pipes": 0,
                             "rms_mm": [], "classes": {}}
    pipes: list[dict[str, Any]] = []
    others: list[dict[str, Any]] = []
    for i, n in enumerate(sg["objects"]):
        if n["kind"] != "object":
            continue
        a = n["asset"] or {}
        ifc_class = IFC_CLASS.get(a.get("asset_class") or n["label"], "IfcBuildingElementProxy")
        name = f"{a['tag']} {n['id']}" if a else n["id"]
        psets = {"Pset_ScanExtraction": {"ObjectId": n["id"], "SegmentationRevision": rev["segmentation"],
                                         "Label": n["label"],
                                         "LabelConfidence": n["label_confidence"],
                                         "LabelSource": n["label_source"], "PointCount": n["n_points"]}}
        if a:
            psets["Pset_ValmetAsset"] = {k: a.get(k) for k in ASSET_FIELDS}
        pts = to_ifc(xyz[order[bounds[i]:bounds[i + 1]]], u2m)
        cols = rgb[order[bounds[i]:bounds[i + 1]]]
        med = np.median(cols, 0) if len(cols) else None

        fitted = False
        if n["label"] == "pipe segment" and n["features"]["linearity"] >= ic["min_pipe_linearity"] and len(pts) >= 10:
            c = fit_cylinder(pts, ic["snap_deg"])
            psets["Pset_ScanExtraction"].update(FitType="cylinder", FitRmsMm=round(c["rms"] * 1000, 1),
                                                NominalSize=c["dn"], RadiusMm=round(c["radius"] * 1000, 1),
                                                FitRadiusMm=round(c["fit_radius"] * 1000, 1))
            stats["rms_mm"].append(c["rms"] * 1000)
            if c["rms"] * 1000 <= ic["max_pipe_rms_mm"]:
                psets["Pset_ScanExtraction"]["ModelSource"] = "fitted cylinder"
                pipes.append({**c, "ifc_class": ifc_class, "name": name, "psets": psets, "obj": i, "rgb": med})
                fitted = True
            else:
                stats["noisy_pipes"] += 1                                   # kept: meshed below
        if not fitted:
            if ifc_class == "IfcPipeSegment" and n["features"]["linearity"] < ic["min_pipe_linearity"]:
                ifc_class = "IfcBuildingElementProxy"
            others.append({"ifc_class": ifc_class, "name": name, "psets": psets, "obj": i, "pts": pts,
                           "cols": cols, "rgb": med, "box": upright_box(pts, ic["snap_deg"])})

    joins = join_pipes(pipes, ic["join_gap_m"])
    elements: dict[int, Any] = {}                                           # scene object index -> element
    for p in pipes:
        elements[p["obj"]] = w.element(p["ifc_class"], p["name"], p["start"], p["axis"], perpendicular(p["axis"]),
                                       w.cylinder(p["radius"], p["length"]), p["psets"], p["rgb"])
        stats["pipes"] += 1
        stats["classes"][p["ifc_class"]] = stats["classes"].get(p["ifc_class"], 0) + 1
    PS = np.array([p["start"] for p in pipes])
    PD = np.array([p["axis"] for p in pipes])
    PL = np.array([p["length"] for p in pipes])

    floor_z = to_ifc(np.array([0, cfg["scene"]["floor_y"], 0]), u2m)[2]
    for o in others:
        b, cls, ps = o["box"], o["ifc_class"], o["psets"]["Pset_ScanExtraction"]
        L, W, H = b["dims"]
        gap = b["origin"][2] - floor_z
        if cls in FLOOR_STANDING and 0 < gap <= ic["floor_snap_m"]:      # stand it on the floor, keep its top
            b["origin"] = b["origin"] - gap * b["z"]
            H += gap
        x = b["x"] if L >= W else np.cross(b["z"], b["x"])
        L, W = max(L, W), min(L, W)
        if cls in FAMILIES:
            ps.update(FitType="box", ModelSource=f"catalog family ({FAMILIES[cls].__name__})")
            el = w.element(cls, o["name"], b["origin"], b["z"], x, w.parts(FAMILIES[cls](L, W, H)), o["psets"],
                           o["rgb"])
            stats["families"] += 1
        else:
            el = None
            if cls == "IfcPipeFitting" and ic["fittings_as_flanges"] and len(pipes):
                centre = b["origin"] + b["z"] * H / 2
                t = np.clip(((centre - PS) * PD).sum(1), 0, PL)
                dist = np.linalg.norm(centre - PS - t[:, None] * PD, axis=1)
                j = int(np.argmin(dist))
                if dist[j] <= ic["flange_snap_m"]:
                    p = pipes[j]
                    at = p["start"] + t[j] * p["axis"]
                    ps.update(FitType="flange", ModelSource=f"catalog family (flange) on {p['name']}")
                    el = w.element(cls, o["name"], at, p["axis"], perpendicular(p["axis"]),
                                   w.parts(flange(p["radius"])), o["psets"], o["rgb"])
                    stats["flanges"] += 1
            if el is None:
                v, t = scan_mesh(o["pts"], ic["mesh_voxel_m"], ic["mesh_max_tris"])
                ps.update(FitType="mesh", ModelSource="scan mesh")
                el = w.element(cls, o["name"], np.zeros(3), np.array([0, 0, 1.0]), np.array([1.0, 0, 0]),
                               w.mesh(v, t, face_colours(v, t, o["pts"], o["cols"])), o["psets"])
                stats["meshes"] += 1
        elements[o["obj"]] = el
        stats["classes"][cls] = stats["classes"].get(cls, 0) + 1

    dev = deviation(w.f, elements, xyz, lab, u2m)
    if label_revision(np.load(out_path(cfg, "segment_points.npz"))["labels"]) != rev["segmentation"]:
        raise RuntimeError("segment_points.npz changed during the export; nothing written")
    backup(cfg, ["scene.ifc", "deviation.npz", "bim_manifest.json"], "bim")
    np.savez_compressed(out_path(cfg, "deviation.npz"), dist_m=dev.astype(np.float32))
    ok = dev[np.isfinite(dev)] * 1000

    path = out_path(cfg, "scene.ifc")
    w.f.write(str(path))
    write_bim_manifest(cfg, rev, [sg["objects"][i]["id"] for i in sorted(elements)],
                       {k: v for k, v in stats.items() if k != "rms_mm"})
    rms = np.array(stats.pop("rms_mm"))
    print(f"[ifc] {len(elements)} elements: {stats['pipes']} pipes, {stats['families']} equipment families, "
          f"{stats['flanges']} flanges, {stats['meshes']} scan meshes ({stats['noisy_pipes']} noisy pipe fits meshed)"
          f" -> {path}")
    print(f"[dev] scan point -> model surface mm: median {np.median(ok):.0f}  p90 {np.percentile(ok, 90):.0f}"
          f"  ({len(ok)} points)")
    print(f"[ifc] classes {dict(sorted(stats['classes'].items(), key=lambda kv: -kv[1]))}")
    if len(rms):
        print(f"[fit] pipe RMS deviation mm: median {np.median(rms):.1f}  p90 {np.percentile(rms, 90):.1f}"
              f"  ({joins} pipe ends joined)")
    return stats


if __name__ == "__main__":
    export(load_config())
