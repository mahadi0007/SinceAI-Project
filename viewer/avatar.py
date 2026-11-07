"""Rigged worker avatar: a Mixamo character (converted to GLB) as Viser skinned meshes, with a walk
and an idle animation.

Input (not in git, see README "Worker avatar"): data/avatar/walking.glb (character + "Walking", In Place)
and optionally data/avatar/idle.glb (an idle animation for the same skeleton, without skin).
Viser's skinned meshes have one colour each (no textures), so every part of the character (body, vest,
helmet, ...) is its own skinned mesh coloured with the median of its texture.

How it is posed: viser bones are a flat list of rigid transforms in the mesh frame, rest pose =
inverse bind matrices. Both animations are sampled once at load time with forward kinematics over the
glTF node tree, so the viewer only sends precomputed bone poses. Finger bones are folded into the
hand and every part only gets the bones that move it, which keeps the per-frame traffic small.
No dependencies beyond numpy and Pillow; the GLB reader below is ~20 lines of struct/json.
"""
from __future__ import annotations

import io
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np

_CT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_NC = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
_FINGERS = ("Thumb", "Index", "Middle", "Ring", "Pinky")


# ---------- GLB reading ----------

def read_glb(path: Path) -> tuple[dict, bytes]:
    b = Path(path).read_bytes()
    if b[:4] != b"glTF":
        raise ValueError(f"{path} is not a binary glTF (.glb) file")
    n = struct.unpack_from("<I", b, 12)[0]
    js = json.loads(b[20:20 + n])
    off = 20 + n
    bn = struct.unpack_from("<I", b, off)[0] if off < len(b) else 0
    return js, b[off + 8: off + 8 + bn]


def accessor(js: dict, binc: bytes, i: int) -> np.ndarray:
    a = js["accessors"][i]
    bv = js["bufferViews"][a["bufferView"]]
    dt, nc = np.dtype(_CT[a["componentType"]]), _NC[a["type"]]
    start, stride = bv.get("byteOffset", 0) + a.get("byteOffset", 0), bv.get("byteStride", 0)
    if stride and stride != dt.itemsize * nc:
        raw = np.frombuffer(binc, np.uint8, count=stride * a["count"], offset=start).reshape(a["count"], stride)
        out = raw[:, :dt.itemsize * nc].copy().view(dt).reshape(a["count"], nc)
    else:
        out = np.frombuffer(binc, dt, count=a["count"] * nc, offset=start).reshape(a["count"], nc)
    if a.get("normalized") and dt.kind in "iu":
        out = out.astype(np.float32) / np.iinfo(dt).max
    return out


# ---------- math (quaternions are xyzw as in glTF unless named wxyz) ----------

def _quat_mat(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _mat_wxyz(m: np.ndarray) -> np.ndarray:
    """Rotation matrix (3x3, may carry tiny scale noise) -> unit quaternion wxyz."""
    u, _, vt = np.linalg.svd(m)
    r = u @ vt
    t = np.trace(r)
    if t > 0:
        s = 2 * np.sqrt(t + 1)
        q = [s / 4, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(r)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2 * np.sqrt(1 + r[i, i] - r[j, j] - r[k, k])
        q = [0.0] * 4
        q[0] = (r[k, j] - r[j, k]) / s
        q[1 + i] = s / 4
        q[1 + j] = (r[j, i] + r[i, j]) / s
        q[1 + k] = (r[k, i] + r[i, k]) / s
    q = np.array(q)
    return q / np.linalg.norm(q)


def _trs(t: np.ndarray, r: np.ndarray, s: np.ndarray) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = _quat_mat(r) * s
    m[:3, 3] = t
    return m


def _sample(times: np.ndarray, values: np.ndarray, t: float, rot: bool) -> np.ndarray:
    t = float(np.clip(t, times[0], times[-1]))
    i = int(np.clip(np.searchsorted(times, t, side="right") - 1, 0, len(times) - 2)) if len(times) > 1 else 0
    if len(times) == 1:
        return values[0]
    a = (t - times[i]) / max(times[i + 1] - times[i], 1e-9)
    v0, v1 = values[i], values[i + 1]
    if rot and np.dot(v0, v1) < 0:          # shortest path, then normalised lerp
        v1 = -v1
    v = (1 - a) * v0 + a * v1
    return v / np.linalg.norm(v) if rot else v


# ---------- rig ----------

class Rig:
    """Character parts (vertices, faces, colour, skin weights) + precomputed bone poses per animation."""

    def __init__(self, walk_glb: Path, idle_glb: Path | None, skip_parts: list[str], fps: float) -> None:
        js, binc = read_glb(walk_glb)
        self.nodes = js["nodes"]
        self.parent = {c: i for i, n in enumerate(self.nodes) for c in n.get("children", [])}
        self.by_name = {n.get("name", ""): i for i, n in enumerate(self.nodes)}
        self.rest = [(np.array(n.get("translation", [0, 0, 0]), float), np.array(n.get("rotation", [0, 0, 0, 1]), float),
                      np.array(n.get("scale", [1, 1, 1]), float)) for n in self.nodes]
        self.parts = self._load_parts(js, binc, skip_parts)
        self.anims: dict[str, dict[str, np.ndarray]] = {}
        walk = self._tracks(js, binc)
        self._bake("walk", walk, fps)
        if idle_glb is not None and Path(idle_glb).exists():
            ijs, ibin = read_glb(idle_glb)
            self._bake("idle", self._tracks(ijs, ibin), fps / 2)
        else:                                   # no idle file: hold the first walk frame
            self.anims["idle"] = {k: v[:1] for k, v in self.anims["walk"].items()}
        mins = np.min([p["verts"].min(0) for p in self.parts], 0)
        maxs = np.max([p["verts"].max(0) for p in self.parts], 0)
        self.height = float(maxs[1] - mins[1])  # metres, glTF +y up

    def _hand_of(self, j: int) -> int:
        """Finger bones -> the hand they belong to (fewer bones to animate)."""
        k = j
        while k in self.parent and any(f in self.nodes[k].get("name", "") for f in _FINGERS):
            k = self.parent[k]
        return k

    def _load_parts(self, js: dict, binc: bytes, skip: list[str]) -> list[dict]:
        images = self._images(js, binc)
        parts = []
        for n in self.nodes:
            if "mesh" not in n or "skin" not in n:
                continue
            mesh, skin = js["meshes"][n["mesh"]], js["skins"][n["skin"]]
            if mesh.get("name") in skip:
                continue
            ibm = accessor(js, binc, skin["inverseBindMatrices"]).reshape(-1, 4, 4).transpose(0, 2, 1)
            for p in mesh["primitives"]:
                at = p["attributes"]
                verts = accessor(js, binc, at["POSITION"]).astype(np.float32)
                faces = accessor(js, binc, p["indices"]).reshape(-1, 3).astype(np.uint32)
                jnt = accessor(js, binc, at["JOINTS_0"]).astype(int)
                wts = accessor(js, binc, at["WEIGHTS_0"]).astype(np.float32)
                node_of = np.array([self._hand_of(j) for j in skin["joints"]])     # skin joint -> kept node
                bone_nodes = sorted({int(node_of[jj]) for jj, ww in zip(jnt.ravel(), wts.ravel()) if ww > 1e-4})
                col = {b: c for c, b in enumerate(bone_nodes)}
                dense = np.zeros((len(verts), len(bone_nodes)), np.float32)
                for k in range(4):
                    np.add.at(dense, (np.arange(len(verts)), [col.get(int(node_of[j]), 0) for j in jnt[:, k]]), wts[:, k])
                dense /= np.maximum(dense.sum(1, keepdims=True), 1e-9)
                ibm_of = {int(j): ibm[i] for i, j in enumerate(skin["joints"])}
                rest = [np.linalg.inv(ibm_of[b]) for b in bone_nodes]
                parts.append({"name": mesh.get("name", f"part{len(parts)}"), "verts": verts, "faces": faces,
                              "weights": dense, "bones": bone_nodes, "color": self._color(js, p, at, binc, images),
                              "rest_wxyz": np.array([_mat_wxyz(m[:3, :3]) for m in rest]),
                              "rest_pos": np.array([m[:3, 3] for m in rest])})
        return parts

    @staticmethod
    def _images(js: dict, binc: bytes) -> list[Any]:
        from PIL import Image
        out = []
        for im in js.get("images", []):
            try:
                bv = js["bufferViews"][im["bufferView"]]
                o = bv.get("byteOffset", 0)
                out.append(Image.open(io.BytesIO(binc[o:o + bv["byteLength"]])).convert("RGBA"))
            except Exception:  # noqa: BLE001 - missing texture -> default colour
                out.append(None)
        return out

    @staticmethod
    def _color(js: dict, prim: dict, at: dict, binc: bytes, images: list[Any]) -> tuple[int, int, int]:
        """Median texture colour under this part's UVs (opaque texels only)."""
        try:
            mat = js["materials"][prim["material"]]["pbrMetallicRoughness"]
            img = images[js["textures"][mat["baseColorTexture"]["index"]]["source"]]
            uv = accessor(js, binc, at["TEXCOORD_0"]) % 1.0
            px = np.asarray(img)
            h, w = px.shape[:2]
            s = px[np.clip((uv[:, 1] * h).astype(int), 0, h - 1), np.clip((uv[:, 0] * w).astype(int), 0, w - 1)]
            s = s[s[:, 3] > 128] if (s[:, 3] > 128).any() else s
            return tuple(int(c) for c in np.median(s[:, :3], 0))
        except Exception:  # noqa: BLE001
            return (200, 200, 200)

    def _tracks(self, js: dict, binc: bytes) -> dict:
        """{node index in the walk file: {path: (times, values)}} matched by bone name."""
        out: dict[int, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
        a = js["animations"][0]
        for c in a["channels"]:
            name = js["nodes"][c["target"]["node"]].get("name", "")
            j = self.by_name.get(name) or self.by_name.get("mixamorig1:" + name.split(":")[-1])
            if j is None or c["target"]["path"] not in ("translation", "rotation"):
                continue
            s = a["samplers"][c["sampler"]]
            out.setdefault(j, {})[c["target"]["path"]] = (accessor(js, binc, s["input"]).ravel().astype(float),
                                                          accessor(js, binc, s["output"]).astype(float))
        return out

    def _bake(self, name: str, tracks: dict, fps: float) -> None:
        dur = max((tv[0][-1] for tr in tracks.values() for tv in tr.values()), default=0.0)
        n = max(1, int(round(dur * fps)))
        times = np.linspace(0, dur, n, endpoint=False)
        needed = sorted({b for p in self.parts for b in p["bones"]})
        world_frames = []
        for t in times:
            local = {}
            for j, (tr, r, s) in enumerate(self.rest):
                tk = tracks.get(j, {})
                tt = _sample(*tk["translation"], t, False) if "translation" in tk else tr
                rr = _sample(*tk["rotation"], t, True) if "rotation" in tk else r
                local[j] = _trs(tt, rr, s)
            cache: dict[int, np.ndarray] = {}

            def world(j: int) -> np.ndarray:
                if j not in cache:
                    cache[j] = world(self.parent[j]) @ local[j] if j in self.parent else local[j]
                return cache[j]
            world_frames.append({b: world(b) for b in needed})
        anim = {}
        for pi, p in enumerate(self.parts):
            anim[f"{pi}_wxyz"] = np.array([[_mat_wxyz(f[b][:3, :3]) for b in p["bones"]] for f in world_frames])
            anim[f"{pi}_pos"] = np.array([[f[b][:3, 3] for b in p["bones"]] for f in world_frames])
        anim["duration"] = np.array(dur)
        self.anims[name] = anim


class RiggedAvatar:
    """Skinned-mesh worker under `parent` (a viser frame whose local axes are forward +z, up = scene up).
    walk(distance) / idle(seconds) pose it; only changed frames are sent."""

    def __init__(self, server: Any, parent: str, rc: dict, units_to_m: float, up_sign: float) -> None:
        self.server = server
        self.rig = Rig(Path(rc["walk_glb"]), Path(rc["idle_glb"]) if rc.get("idle_glb") else None,
                       list(rc.get("skip_parts", [])), float(rc.get("fps", 30)))
        self.base_scale = float(rc["height_m"]) / self.rig.height / units_to_m   # metres in GLB -> scene units
        self.base_cycle = float(rc["cycle_m"]) / units_to_m                        # distance per walk cycle
        self.size = float(rc.get("size", 1.0))
        # glTF: +y up, +z forward. Scene (avatar frame): up = up_sign * y, forward +z.
        # 180 deg about z maps +y -> -y (scene up = -y) and keeps +z forward.
        wxyz = (0.0, 0.0, 0.0, 1.0) if up_sign < 0 else (1.0, 0.0, 0.0, 0.0)
        self.parent = parent
        server.scene.add_frame(f"{parent}/rig", show_axes=False, wxyz=wxyz)
        self.meshes: list[Any] = []
        self._build_meshes()

    @property
    def s(self) -> float:                       # GLB metres -> scene units, incl. the size multiplier
        return self.base_scale * self.size

    def _build_meshes(self) -> None:
        """(Re)send the parts with the size baked into vertices and bone positions: Viser's `scale` on a
        skinned mesh does not scale the bones, so the worker would render at its full GLB size."""
        for h in self.meshes:
            h.remove()
        self.meshes = []
        s = self.s
        for p in self.rig.parts:
            self.meshes.append(self.server.scene.add_mesh_skinned(
                f"{self.parent}/rig/{p['name']}", p["verts"] * s, p["faces"], bone_wxyzs=p["rest_wxyz"],
                bone_positions=p["rest_pos"] * s, skin_weights=p["weights"], color=p["color"],
                material="standard", flat_shading=False, side="double"))
        self._shown: tuple[str, int] | None = None
        self.frame("idle", 0)

    @property
    def height_units(self) -> float:
        return self.rig.height * self.s

    @property
    def cycle_units(self) -> float:            # a smaller worker takes shorter steps
        return self.base_cycle * self.size

    def set_size(self, size: float) -> None:
        """Visual size multiplier (the scene's metres-per-unit is an estimate, so this is tuned by eye)."""
        if abs(float(size) - self.size) < 1e-6:
            return
        self.size = float(size)
        self._build_meshes()

    def frame(self, anim: str, k: int) -> None:
        a = self.rig.anims[anim]
        n = len(a["0_wxyz"])
        k %= n
        if self._shown == (anim, k):
            return
        self._shown = (anim, k)
        with self.server.atomic():
            for i, h in enumerate(self.meshes):
                q, t = a[f"{i}_wxyz"][k], a[f"{i}_pos"][k] * self.s
                for b, bone in enumerate(h.bones):
                    bone.wxyz = q[b]
                    bone.position = t[b]

    def walk(self, distance: float) -> None:
        """Walk cycle phase from the distance walked, so steps match the speed (no foot sliding drift)."""
        n = len(self.rig.anims["walk"]["0_wxyz"])
        self.frame("walk", int((distance / self.cycle_units) % 1.0 * n))

    def idle(self, seconds: float) -> None:
        a = self.rig.anims["idle"]
        dur = float(a["duration"]) or 1.0
        self.frame("idle", int((seconds % dur) / dur * len(a["0_wxyz"])))
