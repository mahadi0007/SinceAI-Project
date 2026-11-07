"""MCP server: the Valmet field copilot's data and actions as tools for any MCP client
(Claude Code, Claude Desktop, ...). Thin wrapper around copilot/tools.py.

Run (stdio):  python mcp_server.py
Claude Code:  .mcp.json in the repo root registers it as "valmet-copilot".

Safety (DOC-FIELD-SAFETY): read-only data tools, DRAFT work orders only, no physical control.
Cost: only diagnose(allow_paid=true) can spend a paid LLM request from the team budget.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))      # run from anywhere

from mcp.server.mcpserver import MCPServer  # noqa: E402

from copilot import tools as T  # noqa: E402

mcp = MCPServer(
    "valmet-copilot",
    instructions=(
        "Field-service copilot for an industrial equipment room (Valmet challenge). Assets are identified by "
        "tag (e.g. P-002), asset id (ASSET-014) or 3D object id (OBJ-0269). Ground every statement in tool "
        "results and cite evidence IDs (TEL-*, EVT-*, WO-*, CASE-*, DOC-*, DQ-*). Never recommend autonomous "
        "physical action: recommend verification and escalation; site procedures, LOTO and permits apply. "
        "Use diagnose(allow_paid=false) unless the user explicitly asks for a fresh paid diagnosis. "
        "If the viewer is running, show_in_viewer / walk_in_viewer / tour_in_viewer move the camera or worker. "
        "Only ~30 objects have Excel records (search_assets); the scene has ~900 detected objects with vision "
        "labels (tanks, ladders, cabinets, trolleys, ...): use find_objects for those, then act on their "
        "object_id. walk_in_viewer / show_in_viewer also accept a label (e.g. 'tank') = nearest reachable one. "
        "move_in_viewer / turn_in_viewer / remove_in_viewer / restore_in_viewer edit the objects that cut out "
        "cleanly; key '' or 'it' = the object selected in the viewer."),
)


@mcp.tool()
def search_assets(condition: str | None = None, asset_class: str | None = None, issue: str | None = None,
                  text: str | None = None, limit: int = 30) -> dict[str, Any]:
    """Find equipment from the Excel pack that is mapped into the 3D scene, most urgent first.
    condition: attention_required | degraded | watch | healthy. asset_class: e.g. pump, motor, valve,
    transmitter, sensor. issue: e.g. CAVITATION, OVERHEATING. text: free words matched against tag/class/issue."""
    return T.search_assets(condition, asset_class, issue, text, limit)


@mcp.tool()
def find_objects(label: str | None = None, text: str | None = None, limit: int = 20) -> dict[str, Any]:
    """Find ANY of the ~900 detected 3D objects by vision label, including equipment WITHOUT an Excel
    record: e.g. tank/vessel, ladder/stairs, electrical cabinet, cable tray, trolley, box/crate, sign,
    manual valve, pump, motor. Use this when search_assets (Excel assets only) finds nothing.
    Returns object_ids usable with walk_in_viewer / show_in_viewer / route_to / nearby."""
    return T.find_objects(label, text, limit)


@mcp.tool()
def get_asset(key: str) -> dict[str, Any]:
    """Identity, condition (health, issue, urgency, next check), size and 3D mapping of one asset or object.
    key: tag (P-002), asset id (ASSET-014) or object id (OBJ-0269)."""
    return T.get_asset(key)


@mcp.tool()
def telemetry_trend(key: str, metric: str | None = None) -> dict[str, Any]:
    """3-day telemetry trend (first vs last 24 h, slope, healthy same-class peers) with evidence IDs.
    metric: optional filter, e.g. vibration, pressure, flow, bearing_temp, motor_current."""
    return T.telemetry_trend(key, metric)


@mcp.tool()
def recent_events(key: str, limit: int = 10) -> dict[str, Any]:
    """Alarm/event history of an asset: counts by event code and the most recent relevant events (EVT-ids)."""
    return T.recent_events(key, limit)


@mcp.tool()
def maintenance_history(key: str) -> dict[str, Any]:
    """Recent work orders (WO-ids), similar historical service cases (CASE-ids), data-quality issues
    (DQ-ids) and the knowledge documents for an asset."""
    return T.maintenance_history(key)


@mcp.tool()
def nearby(key: str, radius_m: float = 1.5) -> dict[str, Any]:
    """What is around an asset: mapped assets within radius_m metres (with status and distance) and
    counts of other scene objects by label."""
    return T.nearby(key, radius_m)


@mcp.tool()
def route_to(key: str) -> dict[str, Any]:
    """Collision-free walking route from the door to an asset/object: length and clearance in metres,
    or a clear message when no path exists."""
    return T.route_to(key)


@mcp.tool()
def plan_tour(stops: list[str] | None = None) -> dict[str, Any]:
    """Plan an inspection tour door -> stops -> faulty target P-002 (always ends there).
    stops: tags in visiting order; empty = automatic from Excel status (most urgent first)."""
    return T.plan_tour(stops)


@mcp.tool()
def draft_work_order(key: str) -> dict[str, Any]:
    """Draft (never submitted) work order from the asset's status and evidence. No LLM, no cost."""
    return T.draft_work_order(key)


@mcp.tool()
def diagnose(key: str, question: str | None = None, allow_paid: bool = False) -> dict[str, Any]:
    """Grounded diagnosis with validated evidence citations. allow_paid=false (default): cached answer for
    the same evidence or the rule template, free. allow_paid=true: may spend 1 request of the team's hard
    paid budget; only use when the user explicitly asks for a fresh AI diagnosis."""
    return T.diagnose(key, question, allow_paid)


@mcp.tool()
def budget_status() -> dict[str, Any]:
    """Paid LLM requests used so far versus the team's hard cap."""
    return T.budget_status()


@mcp.tool()
def show_in_viewer(key: str) -> dict[str, Any]:
    """In the running 3D viewer: select the asset/object and fly the camera to it."""
    return T.show_in_viewer(key)


@mcp.tool()
def walk_in_viewer(key: str) -> dict[str, Any]:
    """In the running 3D viewer: the worker walks from where it stands to the asset/object
    (collision-free; reports if there is no path or no asset record). key: tag, object id, or a
    vision label such as 'tank' (then the nearest reachable object with that label)."""
    return T.walk_in_viewer(key)


@mcp.tool()
def tour_in_viewer(stops: list[str] | None = None) -> dict[str, Any]:
    """In the running 3D viewer: start an inspection tour door -> stops -> faulty target P-002.
    stops: tags in order; empty = automatic from Excel status."""
    return T.tour_in_viewer(stops)


@mcp.tool()
def move_in_viewer(key: str = "", forward_m: float = 0.0, right_m: float = 0.0, up_m: float = 0.0) -> dict[str, Any]:
    """In the running 3D viewer: move an editable object by metres relative to the camera
    (forward = away from the viewer; negative values = back / left / down). key: object id (OBJ-0269),
    a copy (OBJ-0269_copy1), a vision label such as 'chair' (the movable one closest to the camera),
    or '' / 'it' for the object selected in the viewer."""
    return T.move_in_viewer(
        key=key,
        forward_m=forward_m,
        right_m=right_m,
        up_m=up_m,
    )


@mcp.tool()
def turn_in_viewer(key: str = "", degrees: float = 0.0) -> dict[str, Any]:
    """In the running 3D viewer: turn an editable object about the vertical axis
    (positive degrees = left / counter-clockwise seen from above, negative = right). key: as move_in_viewer."""
    return T.turn_in_viewer(
        key=key,
        degrees=degrees,
    )


@mcp.tool()
def remove_in_viewer(key: str = "") -> dict[str, Any]:
    """In the running 3D viewer: remove an editable object (an original can be restored, a copy is deleted).
    key: as move_in_viewer."""
    return T.remove_in_viewer(key=key)


@mcp.tool()
def restore_in_viewer(key: str = "") -> dict[str, Any]:
    """In the running 3D viewer: put an editable object back where it was in the scan and delete its copies.
    key: as move_in_viewer."""
    return T.restore_in_viewer(key=key)


@mcp.prompt()
def shift_handover() -> str:
    """Summarise the equipment that needs attention for a shift handover."""
    return ("Use search_assets for attention_required and degraded equipment, then telemetry_trend and "
            "recent_events for the top three. Write a short shift handover: per asset the status, the key "
            "evidence with IDs, and the next safe verification step. No physical action recommendations.")


@mcp.prompt()
def investigate(key: str) -> str:
    """Investigate one asset end to end."""
    return (f"Investigate {key}: get_asset, telemetry_trend, recent_events, maintenance_history and nearby. "
            f"Then diagnose(allow_paid=false). Separate observations (with evidence IDs) from hypotheses, state "
            f"confidence, and propose the next safe check. If the viewer runs, show_in_viewer({key!r}).")


if __name__ == "__main__":
    mcp.run("stdio")
