"""Workbook metadata audit: what the builder pack can and cannot be used for.
Output: out_review/metadata_audit.json. Run: python -m pipeline.metadata_audit"""
from __future__ import annotations

import json

import pandas as pd

from pipeline.common import ROOT, load_config, repo_path


def run() -> dict:
    cfg = load_config()
    xl = pd.read_excel(repo_path(cfg["paths"]["xlsx"]), sheet_name=None)
    sa, am = xl["Scene_Anchors"], xl["Asset_Master"]
    dims = [c for c in sa.columns if c.startswith("bbox_")]
    rep = {
        "sheets": {k: list(v.shape) for k, v in xl.items()},
        "asset_classes": am["asset_class"].value_counts().to_dict(),
        "scene_anchors": {
            "rows": len(sa),
            "mapping_status": sa["mapping_status"].value_counts().to_dict() if "mapping_status" in sa else None,
            "synthetic_record": sa["synthetic_record"].value_counts().to_dict() if "synthetic_record" in sa else None,
            "distinct_bbox_dims_per_class": (sa.merge(am[["asset_id", "asset_class"]], on="asset_id")
                                             .groupby("asset_class")[dims].nunique().max(axis=1).to_dict()) if dims else None,
        },
        "policy": {
            "used_for": ["component vocabulary (pump, motor, control valve, transmitter, vibration sensor)",
                         "stable asset ids/tags once a verified real-scene identity exists",
                         "confidence penalties from Data_Quality issues (evidence pack)"],
            "not_used_for": ["scene coordinates (anchors are placeholders until the Valmet scene is received)",
                             "box sizes (dimensions repeat by class)",
                             "required object counts (12 valves / 6 pumps are synthetic inventory, not scene truth)",
                             "identity from issue codes or telemetry",
                             "physical pipe topology from connectivity/protocol fields"],
            "identity": "mapping.py marks every asset mapping identity_verified=false unless mapping.verified_identities "
                        "provides a user-confirmed real-scene anchor or readable tag with provenance",
        },
    }
    out = ROOT / "out_review" / "metadata_audit.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    print(json.dumps(rep["scene_anchors"], default=str), "\n", rep["asset_classes"])
    return rep


if __name__ == "__main__":
    run()
