"""Local boundary-correction page (FastAPI). Manual prompts only: no Qwen / paid requests.

Run embedded in the viewer (python -m viewer.app; the review-region panel links to it), or standalone:
    python -m api.annotate            -> http://localhost:8091  (+ its own Viser 3D preview on :8081)
Coordinates from the browser are ORIGINAL image pixels; the page converts from its displayed size.
"""
from __future__ import annotations

import base64
import io
import threading
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from pipeline import correct as cr
from pipeline.common import load_config

HERE = Path(__file__).resolve().parent
COLORS = {"selected": (40, 200, 70), "single_view": (255, 160, 0), "disputed": (230, 30, 30),
          "uncertain": (240, 220, 0), "inferred": (60, 140, 255)}


def overlay_png(mask: np.ndarray, color: tuple[int, int, int] = (0, 160, 255), alpha: int = 110) -> str:
    """Translucent RGBA mask + opaque outline; the clean image stays a separate layer in the page."""
    m = np.asarray(mask, bool)
    a = np.zeros(m.shape + (4,), np.uint8)
    a[m] = (*color, alpha)
    edge = m & ~(np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1))
    a[edge] = (*color, 255)
    b = io.BytesIO()
    Image.fromarray(a, "RGBA").save(b, "PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


class Service:
    """All page actions; framework-free so tests can drive it directly."""

    def __init__(self, cfg: dict, sink: Any = None, store: cr.Store | None = None, engine: cr.Engine | None = None):
        self.cfg = cfg
        self.cc = cr.ccfg(cfg)
        self.store = store or cr.Store(cfg)
        self.eng = engine or cr.Engine(cfg)
        self.sink = sink
        self.latest_seq: dict[tuple[str, str], int] = {}
        self.seq_lock = threading.Lock()
        self.selection: dict[str, dict] = {}          # last 3D preview per component (in memory)

    # --- read ---
    def views(self) -> list[dict]:
        return cr.list_views(self.cc["renderer"])

    def image_path(self, key: str) -> Path:
        from pipeline import propose as pp
        rid, k = cr.parse_key(key)
        return pp.PDIR / rid / self.cc["renderer"] / f"view{k}.png"

    def component(self, cid: str) -> dict:
        c = self.store.comp(cid)
        views = {}
        for key, e in c["draft"]["views"].items():
            views[key] = {k: e[k] for k in ("box", "pos", "neg", "score", "prompt_hash", "mask_px")}
            views[key]["overlay"] = overlay_png(cr.read_mask(self.store.root / e["mask_file"]))
            rid, k = cr.parse_key(key)
            views[key]["stale"] = cr.check_view_fresh(e, rid, k, self.cc["renderer"])
        return {"id": cid, "component_class": c["component_class"], "assembly_id": c["assembly_id"],
                "asset": c["asset"], "note": c["note"], "region": c["region"], "draft_views": views,
                "preview": c["draft"]["preview"], "status": self.store.status(cid),
                "revisions": [{k: r.get(k) for k in ("rev", "status", "t", "boundary_status", "undone_at")} |
                              {"evidence_level": (r.get("stats") or {}).get("evidence_level"),
                               "voxels": (r.get("stats") or {}).get("selected")} for r in c["revisions"]]}

    def components(self) -> dict:
        ap = self.store.data.get("applied")
        stale = None
        if ap:
            try:
                from pipeline.scene import seg_revision
                if seg_revision(cr.ROOT / self.cc["out_seg"]) != ap["seg_rev"]:
                    stale = "applied version on disk differs from the last apply"
            except FileNotFoundError:
                stale = "applied version missing on disk"
        pending = [cid for cid in self.store.data["components"]
                   if self.store.status(cid)["state"].startswith(("accepted, not applied", "applied state out"))]
        return {"components": [{"id": cid, "region": c["region"], "component_class": c["component_class"],
                                **self.store.status(cid)} for cid, c in self.store.data["components"].items()],
                "applied": ap, "applied_problem": stale, "pending_apply": pending,
                "base_seg": self.cc["base_seg"], "out_seg": self.cc["out_seg"], "classes": self.cc["classes"]}

    # --- write ---
    def create(self, region: str, component_class: str = "unknown", assembly_id: str | None = None,
               asset_tag: str | None = None, note: str = "") -> dict:
        if component_class not in self.cc["classes"]:
            raise ValueError(f"class must be one of {self.cc['classes']}")
        c = self.store.new(region, component_class, assembly_id, asset_tag, note)
        return self.component(c["id"])

    def update(self, cid: str, **kw: Any) -> dict:
        if kw.get("component_class") and kw["component_class"] not in self.cc["classes"]:
            raise ValueError(f"class must be one of {self.cc['classes']}")
        self.store.update_meta(cid, **kw)
        return self.component(cid)

    def mask(self, cid: str, key: str, box: Any, pos: Any, neg: Any, seq: int) -> dict:
        """SAM on the clean render. Responses of superseded requests (a newer seq for the same component/view
        arrived) are discarded and never saved."""
        self.store.comp(cid)
        with self.seq_lock:
            k = (cid, key)
            if seq < self.latest_seq.get(k, -1):
                return {"seq": seq, "stale": True}
            self.latest_seq[k] = seq
        hv = cr.view_hashes(*cr.parse_key(key), self.cc["renderer"])
        pr = cr.clean_prompt(box, pos, neg, hv["W"], hv["H"])
        m, score = self.eng.mask(key, pr)
        with self.seq_lock:
            if self.latest_seq.get((cid, key)) != seq:
                return {"seq": seq, "stale": True}
            ph = cr.sha_json({"view": key, **pr, "image": hv["image_sha"]})
            entry = {"view": key, **pr, "score": round(score, 3), "mask_px": int(m.sum()), "prompt_hash": ph,
                     "sam_model": self.cfg["propose"]["sam_model"], "orig_size": [hv["W"], hv["H"]],
                     **{f: hv[f] for f in ("render_key", "cams_key", "cam_sha", "camera", "image_sha", "buffers_sha")}}
            self.store.set_draft_view(cid, key, entry, m)
        self.selection.pop(cid, None)
        return {"seq": seq, "stale": False, "score": round(score, 3), "mask_px": int(m.sum()),
                "prompt_hash": ph, "overlay": overlay_png(m), "W": hv["W"], "H": hv["H"]}

    def drop_view(self, cid: str, key: str) -> dict:
        self.store.drop_draft_view(cid, key)
        self.selection.pop(cid, None)
        return self.component(cid)

    def _select(self, cid: str) -> dict:
        others = {o: self.store.selection(r)["selected"] for o, r in self.store.accepted() if o != cid}
        masks = self.store.draft_masks(cid)
        if not masks:
            raise ValueError("no mask yet: draw prompts and press 'Preview mask'")
        for key, e in self.store.comp(cid)["draft"]["views"].items():
            why = cr.check_view_fresh(e, *cr.parse_key(key), self.cc["renderer"])
            if why:
                raise ValueError(f"stale prompts ({why}); redraw them on the current render")
        return self.eng.support(masks, others)

    def preview3d(self, cid: str) -> dict:
        sel = self._select(cid)
        h = self.store.draft_hash(cid)
        with self.store.lock:
            c = self.store.comp(cid)
            c["draft"]["preview"] = {"hash": h, "base_seg_rev": self.eng.base_rev(), "stats": sel["stats"]}
            self.store.save()
        self.selection[cid] = sel
        self.push_preview(cid, sel)
        return {"preview_hash": h, "stats": sel["stats"], "warnings": sel["warnings"], "per_view": sel["per_view"],
                "objects": [{**m, "mode": "whole"} for m in sel["members"]] +
                           [{**s, "mode": "visible part"} for s in sel["splits"]],
                "skipped": sel["skipped"][:20],
                "next_views": self.eng.rank_views(sel["selected"], set(c["draft"]["views"]))[:6]}

    def guide(self, cid: str, key: str) -> dict:
        sel = self.selection.get(cid)
        if sel is None:
            r = self.store.active_rev(cid)
            if self.store.comp(cid)["draft"]["views"]:
                sel = self._select(cid)
                self.selection[cid] = sel
            elif r is not None:
                sel = {**self.store.selection(r)}
            else:
                return {"view": key, "selected": {"visible": [], "hidden": 0}, "disputed": {"visible": [], "hidden": 0}}
        return self.eng.guide(key, sel)

    def accept(self, cid: str, preview_hash: str, accept_provisional: bool = False) -> dict:
        rec = cr.accept(self.store, self.eng, cid, preview_hash, accept_provisional)
        self.selection.pop(cid, None)
        if self.sink:
            self.sink.clear_correction()             # the accepted frame replaces the preview layers
        self.push_accepted()
        return {"rev": rec["rev"], "boundary_status": rec["boundary_status"], "component": self.component(cid)}

    def reject(self, cid: str, reason: str = "") -> dict:
        rec = cr.reject(self.store, cid, reason)
        self.selection.pop(cid, None)
        if self.sink:
            self.sink.clear_correction()
        return {"rev": rec["rev"], "component": self.component(cid)}

    def undo(self, cid: str) -> dict:
        was_applied = (self.store.data.get("applied") or {}).get("revisions", {}).get(cid) is not None
        res = cr.undo(self.store, cid)
        res["reapplied"] = False
        if was_applied:                 # keep the materialised version equal to the accepted set
            res["manifest"] = cr.apply(self.store, self.cfg, self.eng)
            res["reapplied"] = True
        self.push_accepted()
        res["component"] = self.component(cid)
        return res

    def apply(self) -> dict:
        man = cr.apply(self.store, self.cfg, self.eng)
        self.push_accepted()
        return man

    # --- 3D ---
    def push_preview(self, cid: str, sel: dict) -> None:
        if not self.sink:
            return
        xyz = self.eng.scene().vox_xyz
        single = set(np.asarray(sel["single_view"]).tolist())
        selected = np.asarray(sel["selected"], np.int64)
        multi = selected[~np.isin(selected, list(single))] if single else selected
        layers = {"selected": xyz[multi], "single_view": xyz[np.asarray(sel["single_view"], np.int64)],
                  "disputed": xyz[np.asarray(sel["disputed"], np.int64)],
                  "uncertain": xyz[np.asarray(sel["uncertain"], np.int64)],
                  "inferred": xyz[np.asarray(sel["inferred"], np.int64)]}
        from pipeline.views import Camera
        key = next(iter(self.store.comp(cid)["draft"]["views"]), None)
        cam = self.eng.view(key)[0] if key else None
        self.sink.show_correction(cid, layers, COLORS, sel["stats"]["bbox"], cam)

    def push_accepted(self) -> None:
        if not self.sink:
            return
        items = []
        for cid, r in self.store.accepted():
            bb = r["stats"]["bbox"]
            if bb:
                items.append({"cid": cid, "bbox": bb, "cls": r["component"]["component_class"],
                              "status": self.store.status(cid)["state"], "evidence": r["stats"]["evidence_level"]})
        self.sink.show_accepted(items)


# ---------- FastAPI ----------

def create_app(svc: Service) -> Any:
    from fastapi import Body, FastAPI, HTTPException
    from fastapi.responses import FileResponse, HTMLResponse

    app = FastAPI(title="Boundary correction")

    def run(f: Any, *a: Any, **kw: Any) -> Any:
        try:
            return f(*a, **kw)
        except (ValueError, KeyError) as e:
            raise HTTPException(400, str(e).strip("'\""))

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (HERE / "annotate.html").read_text(encoding="utf-8")

    @app.get("/api/views")
    def views() -> list[dict]:
        return svc.views()

    @app.get("/api/image")
    def image(view: str) -> Any:
        p = svc.image_path(view)
        if not p.exists():
            raise HTTPException(404, "no such view")
        return FileResponse(p, media_type="image/png", headers={"Cache-Control": "no-store"})

    @app.get("/api/components")
    def components() -> dict:
        return svc.components()

    @app.get("/api/component")
    def component(cid: str) -> dict:
        return run(svc.component, cid)

    @app.post("/api/component")
    def create(b: dict = Body(...)) -> dict:
        return run(svc.create, b["region"], b.get("component_class") or "unknown", b.get("assembly_id"),
                   b.get("asset_tag"), b.get("note", ""))

    @app.post("/api/component/meta")
    def meta(b: dict = Body(...)) -> dict:
        return run(svc.update, b["cid"], **{k: b.get(k) for k in ("component_class", "assembly_id", "asset_tag", "note")})

    @app.post("/api/mask")
    def mask(b: dict = Body(...)) -> dict:
        return run(svc.mask, b["cid"], b["view"], b.get("box"), b.get("pos"), b.get("neg"), int(b.get("seq", 0)))

    @app.post("/api/drop_view")
    def drop_view(b: dict = Body(...)) -> dict:
        return run(svc.drop_view, b["cid"], b["view"])

    @app.post("/api/preview3d")
    def preview3d(b: dict = Body(...)) -> dict:
        return run(svc.preview3d, b["cid"])

    @app.get("/api/guide")
    def guide(cid: str, view: str) -> dict:
        return run(svc.guide, cid, view)

    @app.post("/api/accept")
    def accept(b: dict = Body(...)) -> dict:
        return run(svc.accept, b["cid"], b.get("preview_hash", ""), bool(b.get("accept_provisional")))

    @app.post("/api/reject")
    def reject(b: dict = Body(...)) -> dict:
        return run(svc.reject, b["cid"], b.get("reason", ""))

    @app.post("/api/undo")
    def undo(b: dict = Body(...)) -> dict:
        return run(svc.undo, b["cid"])

    @app.post("/api/apply")
    def apply() -> dict:
        return run(svc.apply)

    return app


def free_port(start: int, tries: int = 20) -> int:
    """First port >= start that can be bound on localhost (another tool may already hold the configured one)."""
    import socket
    for p in range(start, start + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as c:      # something already answering?
            c.settimeout(0.3)
            if c.connect_ex(("127.0.0.1", p)) == 0:
                continue
        ok = True
        for fam, host in ((socket.AF_INET, "0.0.0.0"), (socket.AF_INET6, "::")):
            try:
                with socket.socket(fam, socket.SOCK_STREAM) as s:
                    if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):   # Windows: fail if ANY socket holds the port
                        s.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
                    s.bind((host, p))
            except OSError:
                if fam == socket.AF_INET or socket.has_ipv6:
                    ok = False
        if ok:
            return p
    raise OSError(f"no free port in {start}-{start + tries - 1}")


def start_in_thread(cfg: dict, sink: Any = None) -> Service:
    import uvicorn
    svc = Service(cfg, sink)
    want = cr.ccfg(cfg)["port"]
    port = free_port(want)
    if port != want:
        print(f"[annotate] port {want} is in use; using {port}")
    svc.port = port
    server = uvicorn.Server(uvicorn.Config(create_app(svc), host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True, name="annotate").start()
    print(f"[annotate] boundary correction tool: http://localhost:{port}")
    return svc


class ViserSink:
    """Standalone 3D preview: base voxel cloud + correction layers in a small Viser scene."""

    def __init__(self, cfg: dict, eng: cr.Engine) -> None:
        import viser
        sc = eng.scene()
        rgb = np.load(sc.seg_dir / "segment_points.npz")["rgb"]
        self.port = free_port(cr.ccfg(cfg)["viser_port"])
        self.server = viser.ViserServer(port=self.port)
        self.server.scene.set_up_direction(cfg["scene"]["up_direction"])
        self.server.scene.add_point_cloud("/room", sc.vox_xyz.astype(np.float32),
                                          (np.clip(rgb, 0, 1) * 255).astype(np.uint8), point_size=0.008)
        self.h: list[Any] = []
        self.h_acc: list[Any] = []

    def clear_correction(self) -> None:
        for h in self.h:
            h.remove()
        self.h = []

    def show_correction(self, cid: str, layers: dict, colors: dict, bbox: Any, cam: Any) -> None:
        from viewer.app import box_edges
        self.clear_correction()
        sc = self.server.scene
        for name, P in layers.items():
            if len(P):
                self.h.append(sc.add_point_cloud(f"/correction/{name}", P.astype(np.float32), colors[name],
                                                 point_size=0.014))
        if bbox:
            self.h.append(sc.add_line_segments("/correction/box", box_edges(*bbox), (255, 255, 255), thickness=3.0,
                                               thickness_units="screen"))
        if cam is not None:
            for cl in self.server.get_clients().values():
                cl.camera.position = tuple(cam.C)
                cl.camera.look_at = tuple(cam.C + 1.5 * cam.R[2])

    def show_accepted(self, items: list[dict]) -> None:
        from viewer.app import box_edges
        for h in self.h_acc:
            h.remove()
        self.h_acc = []
        for it in items:
            col = (0, 220, 255) if it["status"] == "applied" else (255, 0, 200)
            self.h_acc.append(self.server.scene.add_line_segments(
                f"/accepted/{it['cid']}", box_edges(*it["bbox"]), col, thickness=2.5, thickness_units="screen"))
            self.h_acc.append(self.server.scene.add_label(
                f"/accepted/{it['cid']}_label", f"{it['cid']} {it['cls']} ({it['status']})",
                position=((it["bbox"][0][0] + it["bbox"][1][0]) / 2, it["bbox"][0][1] - 0.05,
                          (it["bbox"][0][2] + it["bbox"][1][2]) / 2), anchor="bottom-center"))


if __name__ == "__main__":
    import time
    cfg = load_config()
    svc = start_in_thread(cfg)
    svc.sink = ViserSink(cfg, svc.eng)
    svc.push_accepted()
    print(f"[annotate] 3D preview: http://localhost:{svc.sink.port}")
    while True:
        time.sleep(3600)
