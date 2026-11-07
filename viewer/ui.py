"""Control-panel look: dark "control room" theme for the Viser GUI + small HTML components.

Viser has no custom widgets, so the design system is (1) one global stylesheet injected through an
HTML widget, which restyles Viser's own Mantine controls (tabs, buttons, inputs, folders, tables), and
(2) pure functions returning HTML strings for read-only pieces (header, status chips, asset hero,
tour stepper, audio card). Interactive controls stay real Viser controls, so callbacks are unchanged.
Selectors use stable hooks (data-testid, role, data-* attributes, mantine-* classes) where possible.
"""
from __future__ import annotations

import html
import re

ACCENT = (20, 184, 166)                     # teal; also passed to configure_theme(brand_color=...)

STATE = {                                    # condition_state -> (label, css var)
    "attention_required": ("Attention required", "--crit"),
    "degraded": ("Degraded", "--warn"),
    "watch": ("Watch", "--watch"),
    "healthy": ("Healthy", "--ok"),
}

# Tabler-style outline icons (24px grid, stroke 2), inlined so every HTML piece shares one icon system.
_ICON_PATHS = {
    "walk": '<circle cx="13" cy="4" r="1.6"/><path d="M7 21l3-4M16 21l-2-4-3-3 1-6M6 12l2-3 4-1 3 3 3 1"/>',
    "check": '<path d="M5 12l5 5L20 7"/>',
    "alert": '<path d="M12 9v4M12 17h.01"/><path d="M10.4 3.9L2.6 17.5A1.8 1.8 0 0 0 4.2 20h15.6a1.8 1.8 0 0 0 1.6-2.5L13.6 3.9a1.8 1.8 0 0 0-3.2 0z"/>',
    "info": '<circle cx="12" cy="12" r="9"/><path d="M12 8h.01M11 12h1v4h1"/>',
    "volume": '<path d="M15 8a5 5 0 0 1 0 8M17.7 5a9 9 0 0 1 0 14"/><path d="M6 15H4a1 1 0 0 1-1-1v-4a1 1 0 0 1 1-1h2l3.5-4.5A.8.8 0 0 1 11 5v14a.8.8 0 0 1-1.5.5L6 15"/>',
    "pin": '<circle cx="12" cy="11" r="3"/><path d="M17.7 16.7L13.4 21a2 2 0 0 1-2.8 0l-4.3-4.3a8 8 0 1 1 11.4 0z"/>',
    "click": '<path d="M3 12h3M12 3v3M7.8 7.8L5.6 5.6M16.2 7.8l2.2-2.2M7.8 16.2l-2.2 2.2M12 12l9 3-4 2-2 4-3-9"/>',
    "route": '<circle cx="6" cy="19" r="2"/><circle cx="18" cy="5" r="2"/><path d="M12 19h4.5a3.5 3.5 0 0 0 0-7h-8a3.5 3.5 0 0 1 0-7H12"/>',
    "door": '<path d="M14 12v.01M3 21h18M6 21V5a2 2 0 0 1 2-2h8a2 2 0 0 1 2 2v16"/>',
    "cube": '<path d="M12 3l8 4.5v9L12 21l-8-4.5v-9L12 3M12 12l8-4.5M12 12v9M12 12L4 7.5"/>',
}


def icon(name: str, size: int = 16) -> str:
    return (f'<svg class="fc-ic" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" '
            f'aria-hidden="true">{_ICON_PATHS[name]}</svg>')


def esc(s: object) -> str:
    return html.escape(str(s), quote=True)


def words(s: object) -> str:
    """'CAVITATION_RISK' -> 'Cavitation risk'."""
    t = str(s or "").replace("_", " ").strip()
    return t[:1].upper() + t[1:].lower() if t.isupper() or "_" in str(s or "") else t[:1].upper() + t[1:]


THEME_CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
:root {
  --bg: rgba(13, 17, 23, 0.9); --card: #161b22; --card-2: #1d242d; --card-3: #252d38;
  --line: rgba(255, 255, 255, 0.08); --line-2: rgba(255, 255, 255, 0.14);
  --text: #e6edf3; --muted: #9aa6b4; --faint: #6b7785;
  --accent: #14b8a6; --accent-hi: #2dd4bf; --accent-soft: rgba(20, 184, 166, 0.14); --accent-ink: #04201c;
  --crit: #fb7185; --warn: #fbbf24; --watch: #facc15; --ok: #4ade80; --none: #94a3b8;
  --r-card: 14px; --r-input: 10px; --r-pill: 999px;
  --shadow: 0 10px 30px rgba(0, 0, 0, 0.45), 0 2px 6px rgba(0, 0, 0, 0.3);
  --ease: cubic-bezier(0.16, 1, 0.3, 1);
  --mantine-radius-xs: 10px; --mantine-radius-sm: 12px;
}
/* ---------- panel shell ---------- */
[data-testid="floating-panel"] {
  background: var(--bg) !important; backdrop-filter: blur(16px) saturate(1.2);
  -webkit-backdrop-filter: blur(16px) saturate(1.2); border: 1px solid var(--line) !important;
  border-radius: 18px !important; box-shadow: var(--shadow) !important; overflow: hidden !important; }
[data-testid="floating-panel"], [data-testid="floating-panel"] * {
  font-family: 'IBM Plex Sans', system-ui, sans-serif; }
[data-testid="floating-panel"] { color: var(--text); font-size: 13px; }
[data-testid="floating-panel"] ::selection { background: rgba(20, 184, 166, 0.35); color: #fff; }
[data-testid="floating-panel"] .mantine-ScrollArea-viewport { scrollbar-width: thin; }
[data-testid="floating-panel"] .mantine-ScrollArea-thumb { background: var(--line-2) !important; }
[data-testid="floating-panel"] :focus-visible { outline: 2px solid var(--accent) !important; outline-offset: 2px; }
[data-testid="floating-panel"] input, [data-testid="floating-panel"] textarea { caret-color: var(--accent); }
/* ---------- tabs: segmented pill bar ---------- */
[role="tablist"] {
  background: var(--card) !important; background-image: none !important; border: 1px solid var(--line);
  border-radius: var(--r-pill); padding: 4px; margin: 2px 10px 12px; gap: 2px; flex-wrap: nowrap !important;
  font-size: 12.5px !important; }
[role="tab"] {
  flex: 1 1 auto; justify-content: center; height: 32px !important; padding: 0 10px !important;
  border-radius: var(--r-pill); box-shadow: none !important; color: var(--muted) !important;
  font-weight: 500 !important; transition: background-color 160ms var(--ease), color 160ms var(--ease); }
[role="tab"]:hover { color: var(--text) !important; background: rgba(255, 255, 255, 0.04) !important; }
[role="tab"][aria-selected="true"] {
  background: var(--accent-soft) !important; color: var(--accent-hi) !important; font-weight: 600 !important;
  box-shadow: inset 0 0 0 1px rgba(45, 212, 191, 0.35) !important; }
/* ---------- buttons: pills ---------- */
[data-testid="floating-panel"] .mantine-Button-root {
  --button-radius: var(--r-pill) !important; border-radius: var(--r-pill) !important; height: 36px !important;
  font-weight: 600; letter-spacing: 0.01em; font-size: 13px;
  transition: transform 140ms var(--ease), background-color 160ms var(--ease), box-shadow 160ms var(--ease); }
[data-testid="floating-panel"] .mantine-Button-root:active { transform: scale(0.98); }
[data-testid="floating-panel"] .mantine-Button-root[style*="--button-bg: var(--mantine-color-teal"],
[data-testid="floating-panel"] .mantine-Button-root:not([style*="--button-bg"]) {
  background: var(--accent) !important; color: var(--accent-ink) !important;
  box-shadow: 0 6px 16px rgba(20, 184, 166, 0.25); }
[data-testid="floating-panel"] .mantine-Button-root:not([style*="--button-bg"]):hover {
  background: var(--accent-hi) !important; }
[data-testid="floating-panel"] .mantine-Button-root[style*="--mantine-color-gray"] {
  background: var(--card-3) !important; color: var(--text) !important; box-shadow: inset 0 0 0 1px var(--line-2); }
[data-testid="floating-panel"] .mantine-Button-root[style*="--mantine-color-gray"]:hover {
  background: #2e3846 !important; }
[data-testid="floating-panel"] .mantine-Button-root[style*="--mantine-color-red"] {
  background: rgba(251, 113, 133, 0.14) !important; color: var(--crit) !important;
  box-shadow: inset 0 0 0 1px rgba(251, 113, 133, 0.35); }
[data-testid="floating-panel"] .mantine-Button-root[data-disabled], [data-testid="floating-panel"] .mantine-Button-root:disabled {
  opacity: 0.45; box-shadow: none; }
/* ---------- inputs ---------- */
[data-testid="floating-panel"] .mantine-Input-input {
  background: var(--card-2) !important; border: 1px solid var(--line) !important; color: var(--text) !important;
  border-radius: var(--r-input) !important; height: 34px !important; min-height: 34px !important;
  padding-inline: 10px !important; font-size: 13px !important; letter-spacing: 0 !important;
  transition: border-color 140ms var(--ease), box-shadow 140ms var(--ease); }
[data-testid="floating-panel"] .mantine-Input-input:hover { border-color: var(--line-2) !important; }
[data-testid="floating-panel"] .mantine-Input-input:focus {
  border-color: var(--accent) !important; box-shadow: 0 0 0 3px var(--accent-soft) !important; }
[data-testid="floating-panel"] .mantine-Input-input::placeholder { color: var(--faint); }
[data-testid="floating-panel"] .mantine-Text-root { color: var(--muted) !important; letter-spacing: 0 !important; }
.mantine-Combobox-dropdown, .mantine-Select-dropdown {
  background: #1a2029 !important; border: 1px solid var(--line-2) !important; border-radius: 12px !important;
  box-shadow: var(--shadow) !important; padding: 4px !important; }
.mantine-Combobox-option { border-radius: 8px !important; font-size: 13px !important; color: var(--text) !important; }
.mantine-Combobox-option[data-combobox-selected], .mantine-Combobox-option:hover {
  background: var(--accent-soft) !important; }
/* ---------- sliders ---------- */
[data-testid="floating-panel"] .mantine-Slider-track::before { background: var(--card-3) !important; }
[data-testid="floating-panel"] .mantine-Slider-bar { background: var(--accent) !important; }
[data-testid="floating-panel"] .mantine-Slider-thumb { border-color: var(--accent) !important; background: #fff !important; }
[data-testid="floating-panel"] .mantine-NumberInput-input { height: 26px !important; min-height: 26px !important; }
/* ---------- folders: cards ---------- */
[data-testid="floating-panel"] .mantine-Paper-root[data-with-border] {
  background: var(--card) !important; border: 1px solid var(--line) !important; border-radius: var(--r-card) !important;
  margin: 0 10px 10px !important; overflow: hidden; }
[data-testid="floating-panel"] .mantine-Paper-root[data-with-border] .mantine-Paper-root[data-with-border] {
  margin: 4px 8px 8px !important; background: var(--card-2) !important; }
[data-testid="floating-panel"] .mantine-Paper-root[data-with-border] > .mantine-Paper-root[style*="cursor: pointer"] {
  position: static !important; transform: none !important; left: auto !important;
  background: transparent !important; padding: 11px 14px 4px !important; font-weight: 600 !important; font-size: 13.5px;
  color: var(--text); display: flex; align-items: center; justify-content: space-between;
  transition: background-color 140ms var(--ease); }
[data-testid="floating-panel"] .mantine-Paper-root[data-with-border] > .mantine-Paper-root[style*="cursor: pointer"]:hover {
  background: rgba(255, 255, 255, 0.03) !important; }
[data-testid="floating-panel"] .mantine-Paper-root[data-with-border] > .mantine-Paper-root[style*="cursor: pointer"] svg {
  color: var(--faint); }
/* ---------- markdown ---------- */
[data-testid="floating-panel"] h1, [data-testid="floating-panel"] h2, [data-testid="floating-panel"] h3 {
  color: var(--text); font-weight: 650; letter-spacing: -0.01em; }
[data-testid="floating-panel"] h3 { font-size: 15px !important; margin: 14px 0 6px !important; }
[data-testid="floating-panel"] h4 {
  font-size: 11px !important; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted) !important;
  font-weight: 600 !important; margin: 16px 0 6px !important; padding-top: 12px; border-top: 1px solid var(--line); }
[data-testid="floating-panel"] p, [data-testid="floating-panel"] li { line-height: 1.5; }
[data-testid="floating-panel"] strong { color: var(--text); font-weight: 600; }
[data-testid="floating-panel"] em { color: var(--muted); }
[data-testid="floating-panel"] a { color: var(--accent-hi) !important; text-underline-offset: 3px; }
[data-testid="floating-panel"] code { background: var(--card-3); border-radius: 6px; padding: 1px 5px; font-size: 12px; }
[data-testid="floating-panel"] hr { border: 0; border-top: 1px solid var(--line); margin: 14px 0; }
[data-testid="floating-panel"] blockquote {
  margin: 6px 0; padding: 8px 12px; border: 0; border-radius: 12px; background: var(--accent-soft); color: var(--text); }
/* ---------- tables ---------- */
.mantine-Table-table { border-collapse: separate !important; border-spacing: 0; width: 100%; max-width: 100%;
  margin: 4px 0 8px; font-size: 12px; font-variant-numeric: tabular-nums; border: 1px solid var(--line-2);
  border-radius: 10px; overflow: hidden; }
.mantine-Table-table th, .mantine-Table-table td {
  border: 0 !important; border-bottom: 1px solid var(--line) !important; border-right: 1px solid var(--line) !important;
  padding: 5px 7px !important; vertical-align: top; overflow-wrap: break-word; word-break: normal; hyphens: auto;
  color: var(--text); }
.mantine-Table-table tr > :last-child { border-right: 0 !important; }
.mantine-Table-table tbody tr:last-child td { border-bottom: 0 !important; }
.mantine-Table-table:has(th:nth-child(5)) { font-size: 11.5px; }
.mantine-Table-table:has(th:nth-child(5)) th, .mantine-Table-table:has(th:nth-child(5)) td { padding: 4px 5px !important; }
.mantine-Table-table thead th { background: var(--card-3) !important; font-weight: 600; text-align: left !important;
  color: var(--muted) !important; font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em; }
.mantine-Table-table tbody td { background: var(--card-2); }
.mantine-Table-table tbody tr:nth-child(even) td { background: #1a2028; }
/* ---------- toggles (Viser checkboxes size xs -> switches) ---------- */
.mantine-Checkbox-root[data-size="xs"] .mantine-Checkbox-inner { width: 38px !important; height: 22px !important; }
.mantine-Checkbox-root[data-size="xs"] .mantine-Checkbox-input {
  appearance: none; width: 38px !important; height: 22px !important; border-radius: 11px !important;
  border: none !important; cursor: pointer; transition: background 180ms var(--ease);
  background: radial-gradient(circle at 11px 50%, #fff 8px, transparent 8.5px), var(--card-3) !important;
  box-shadow: inset 0 0 0 1px var(--line-2); }
.mantine-Checkbox-root[data-size="xs"] .mantine-Checkbox-input:checked {
  background: radial-gradient(circle at 27px 50%, #fff 8px, transparent 8.5px), var(--accent) !important;
  box-shadow: none; }
.mantine-Checkbox-root[data-size="xs"] .mantine-Checkbox-icon { display: none !important; }
/* ---------- progress ---------- */
[data-testid="floating-panel"] .mantine-Progress-root { background: var(--card-3) !important; border-radius: 99px;
  margin: 0 10px; }
[data-testid="floating-panel"] .mantine-Progress-section { background: var(--accent) !important; }
/* ---------- components (viewer/ui.py) ---------- */
.fc-ic { flex-shrink: 0; vertical-align: -3px; }
.fc-wrap { padding: 0 10px; }
.fc-head { display: flex; align-items: center; gap: 10px; padding: 4px 10px 12px; }
.fc-mark { width: 34px; height: 34px; border-radius: 10px; display: grid; place-items: center;
  background: var(--accent); color: var(--accent-ink); box-shadow: 0 6px 16px rgba(20, 184, 166, 0.3); }
.fc-title { font-size: 16px; font-weight: 700; letter-spacing: -0.01em; color: var(--text); line-height: 1.15; }
.fc-sub { font-size: 12px; color: var(--muted); margin-top: 1px; }
.fc-stats { display: flex; gap: 6px; padding: 0 10px 12px; flex-wrap: wrap; }
.fc-stat { flex: 1 1 0; min-width: 84px; background: var(--card); border: 1px solid var(--line); border-radius: 12px;
  padding: 8px 10px; }
.fc-stat b { display: block; font-size: 17px; font-weight: 700; color: var(--text); font-variant-numeric: tabular-nums; }
.fc-stat span { font-size: 11px; color: var(--muted); }
.fc-stat.crit b { color: var(--crit); }
.fc-lead { padding: 0 12px 10px; font-size: 13px; color: var(--muted); line-height: 1.45; }
.fc-lead strong { color: var(--text); }
.fc-pill { display: inline-flex; align-items: center; gap: 6px; height: 22px; padding: 0 9px; border-radius: 99px;
  font-size: 11.5px; font-weight: 600; white-space: nowrap; color: var(--c, var(--none));
  background: color-mix(in srgb, var(--c, var(--none)) 14%, transparent);
  box-shadow: inset 0 0 0 1px color-mix(in srgb, var(--c, var(--none)) 35%, transparent); }
.fc-pill i { width: 7px; height: 7px; border-radius: 50%; background: currentColor; }
.fc-chip { display: inline-flex; align-items: center; height: 22px; padding: 0 9px; border-radius: 99px;
  font-size: 11.5px; color: var(--muted); background: var(--card-3); }
.fc-status { display: flex; align-items: flex-start; gap: 10px; margin: 0 10px 10px; padding: 10px 12px;
  border-radius: 12px; background: var(--card); border: 1px solid var(--line); font-size: 13px; line-height: 1.4; }
.fc-status .fc-dot { width: 9px; height: 9px; border-radius: 50%; margin-top: 5px; flex-shrink: 0; background: var(--faint); }
.fc-status.walking { border-color: rgba(45, 212, 191, 0.35); background: linear-gradient(0deg, var(--accent-soft), var(--accent-soft)), var(--card); }
.fc-status.walking .fc-dot { background: var(--accent-hi); animation: fc-pulse 1.4s var(--ease) infinite; }
.fc-status.arrived .fc-dot { background: var(--ok); }
.fc-status.warn { border-color: rgba(251, 191, 36, 0.35); }
.fc-status.warn .fc-dot { background: var(--warn); }
.fc-status em { color: var(--muted); font-style: normal; }
@keyframes fc-pulse { 0% { box-shadow: 0 0 0 0 rgba(45, 212, 191, 0.55); } 100% { box-shadow: 0 0 0 9px rgba(45, 212, 191, 0); } }
.fc-opt { display: flex; gap: 10px; align-items: flex-start; padding: 2px 4px 8px; }
.fc-num { width: 24px; height: 24px; border-radius: 50%; display: grid; place-items: center; flex-shrink: 0;
  background: var(--accent-soft); color: var(--accent-hi); font-weight: 700; font-size: 12px;
  box-shadow: inset 0 0 0 1px rgba(45, 212, 191, 0.35); }
.fc-opt-t { font-size: 12.5px; color: var(--muted); line-height: 1.45; flex: 1; }
.fc-sel { display: flex; align-items: center; gap: 8px; margin: 0 4px 8px; padding: 8px 10px; border-radius: 10px;
  background: var(--card-2); border: 1px dashed var(--line-2); color: var(--muted); font-size: 12.5px; }
.fc-sel.on { border-style: solid; border-color: rgba(45, 212, 191, 0.35); color: var(--text); }
.fc-sel.on .fc-ic { color: var(--accent-hi); }
.fc-sel.fc-armed { background: var(--accent-soft); animation: fc-arm 1.6s var(--ease) infinite; }
@keyframes fc-arm { 50% { border-color: rgba(45, 212, 191, 0.85); } }
.fc-hero { margin: 0 10px 10px; padding: 14px; border-radius: var(--r-card); background: var(--card);
  border: 1px solid var(--line); box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04); }
.fc-hero-top { display: flex; justify-content: space-between; align-items: flex-start; gap: 8px; }
.fc-tag { font-size: 22px; font-weight: 700; letter-spacing: -0.02em; color: var(--text); line-height: 1.1; }
.fc-class { font-size: 12.5px; color: var(--muted); margin-top: 3px; }
.fc-meta { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 10px; }
.fc-health { margin-top: 12px; }
.fc-health-row { display: flex; justify-content: space-between; align-items: baseline; font-size: 11.5px; color: var(--muted); }
.fc-health-row b { font-size: 18px; color: var(--text); font-variant-numeric: tabular-nums; }
.fc-bar { height: 6px; border-radius: 99px; background: var(--card-3); overflow: hidden; margin-top: 5px; }
.fc-bar i { display: block; height: 100%; border-radius: 99px; background: var(--c, var(--accent));
  transition: width 400ms var(--ease); }
.fc-issue { margin-top: 12px; padding: 9px 11px; border-radius: 10px; font-size: 12.5px; line-height: 1.45;
  background: color-mix(in srgb, var(--c, var(--none)) 9%, var(--card-2)); color: var(--text); }
.fc-issue b { display: block; font-size: 12px; color: var(--c, var(--text)); margin-bottom: 2px; }
.fc-audio { margin: 0 10px 10px; padding: 10px 12px 8px; border-radius: var(--r-card);
  background: linear-gradient(135deg, rgba(20, 184, 166, 0.18), rgba(20, 184, 166, 0.05)), var(--card);
  border: 1px solid rgba(45, 212, 191, 0.3); }
.fc-audio-h { display: flex; align-items: center; gap: 8px; font-size: 12.5px; font-weight: 600; color: var(--accent-hi);
  margin-bottom: 6px; }
.fc-wave { display: inline-flex; gap: 2px; align-items: center; height: 12px; margin-left: auto; }
.fc-wave i { width: 2px; border-radius: 2px; background: var(--accent-hi); animation: fc-eq 1s ease-in-out infinite; }
.fc-wave i:nth-child(1) { height: 5px; } .fc-wave i:nth-child(2) { height: 11px; animation-delay: -0.3s; }
.fc-wave i:nth-child(3) { height: 7px; animation-delay: -0.6s; } .fc-wave i:nth-child(4) { height: 10px; animation-delay: -0.15s; }
@keyframes fc-eq { 50% { transform: scaleY(0.35); } }
.fc-audio audio { width: 100%; height: 34px; color-scheme: dark; border-radius: 99px; }
.fc-steps { display: flex; flex-wrap: wrap; align-items: center; gap: 4px; margin: 0 4px 10px; }
.fc-step { display: inline-flex; align-items: center; gap: 4px; height: 24px; padding: 0 8px; border-radius: 99px;
  font-size: 11.5px; font-weight: 600; color: var(--muted); box-shadow: inset 0 0 0 1px var(--line-2); }
.fc-step i { width: 6px; height: 6px; border-radius: 50%; background: var(--c, var(--none)); }
.fc-step.done { color: var(--faint); box-shadow: none; background: var(--card-2); }
.fc-step.now { color: var(--accent-ink); background: var(--accent); box-shadow: 0 4px 12px rgba(20, 184, 166, 0.3); }
.fc-step.now i { background: var(--accent-ink); }
.fc-sep { color: var(--faint); font-size: 11px; }
.fc-stop { margin: 0 4px 6px; font-size: 12.5px; color: var(--muted); line-height: 1.45; }
.fc-stop b { color: var(--text); }
/* hover info */
.vtip { position: relative; display: inline-flex; align-items: center; gap: 4px; cursor: help; font-size: 11.5px;
  color: var(--accent-hi); margin-top: 4px; }
.vtip .vtip-box { visibility: hidden; opacity: 0; transform: translateY(-3px);
  transition: opacity 160ms var(--ease), transform 160ms var(--ease);
  position: absolute; left: -6px; top: calc(100% + 8px); z-index: 1000; width: max-content; max-width: 280px;
  padding: 9px 12px; border-radius: 10px; line-height: 1.45; font-size: 12px; font-weight: 400;
  background: #0b0f14; color: var(--text); border: 1px solid rgba(45, 212, 191, 0.35);
  box-shadow: var(--shadow); pointer-events: none; }
.vtip:hover .vtip-box { visibility: visible; opacity: 1; transform: none; }
@media (prefers-reduced-motion: reduce) {
  [data-testid="floating-panel"] *, .fc-wave i, .fc-status .fc-dot { animation: none !important; transition: none !important; }
}
</style>"""


def _state(state: str | None) -> tuple[str, str]:
    return STATE.get(str(state or ""), (words(state) if state else "No record", "--none"))


def status_pill(state: str | None) -> str:
    label, var = _state(state)
    return f'<span class="fc-pill" style="--c: var({var})"><i></i>{esc(label)}</span>'


def health_bar(score: float | int | None, state: str | None) -> str:
    if score is None:
        return ""
    _, var = _state(state)
    pct = max(0.0, min(100.0, float(score)))
    return (f'<div class="fc-health"><div class="fc-health-row"><span>Health</span><span><b>{int(score)}</b> / 100'
            f'</span></div><div class="fc-bar" style="--c: var({var})"><i style="width:{pct:.0f}%"></i></div></div>')


def brand_header(n_assets: int, n_attention: int, n_objects: int) -> str:
    return (f'<div class="fc-head"><div class="fc-mark">{icon("route", 18)}</div><div>'
            f'<div class="fc-title">Field Copilot</div><div class="fc-sub">Equipment room · inspection assistant</div>'
            f'</div></div><div class="fc-stats">'
            f'<div class="fc-stat crit"><b>{n_attention}</b><span>need attention</span></div>'
            f'<div class="fc-stat"><b>{n_assets}</b><span>assets in Excel</span></div>'
            f'<div class="fc-stat"><b>{n_objects}</b><span>3D objects</span></div></div>')


def lead(text_html: str) -> str:
    return f'<div class="fc-lead">{text_html}</div>'


def tip(text: str, label: str = "What does this do?") -> str:
    return f'<div class="vtip">{icon("info", 13)}{esc(label)}<div class="vtip-box">{esc(text)}</div></div>'


def option_header(num: int, text: str, tip_text: str) -> str:
    return (f'<div class="fc-opt"><div class="fc-num">{num}</div><div class="fc-opt-t">{esc(text)}'
            f'<br>{tip(tip_text)}</div></div>')


def selected_chip(what_html: str | None) -> str:
    if not what_html:
        return f'<div class="fc-sel">{icon("click", 15)}Press the button, then click any object in the 3D scan.</div>'
    return f'<div class="fc-sel on">{icon("cube", 15)}<span>{what_html}</span></div>'


def pick_prompt() -> str:
    """Pick mode armed: the next click in the 3D scan is the destination."""
    return (f'<div class="fc-sel on fc-armed">{icon("click", 15)}<span><b>Click an object in the 3D scan</b>: the worker '
            f'walks to it. Press the button again to cancel.</span></div>')


_MD_BOLD = re.compile(r"\*\*(.+?)\*\*")
_MD_ITAL = re.compile(r"(?<![\w*])_(.+?)_(?![\w*])")
_LEADING = ("🚶", "✅", "⚠️", "🔊", "📋", "🤖")


def md_inline(text: str) -> str:
    """Tiny markdown subset (bold, italics, line breaks) to HTML, escaped."""
    s = esc(text)
    s = _MD_BOLD.sub(r"<b>\1</b>", s)
    s = _MD_ITAL.sub(r"<em>\1</em>", s)
    return s.replace("  \n", "<br>").replace("\n", "<br>")


def walk_status(text: str) -> str:
    """Live status card at the top of Walk from the existing status line ('🚶 Walking to …' etc.)."""
    if not text.strip():
        return ""
    kind = ("walking" if text.startswith("🚶") else "arrived" if text.startswith("✅")
            else "warn" if text.startswith("⚠️") else "")
    body = text
    for e in _LEADING:
        body = body.replace(e, "").strip() if body.startswith(e) else body
    return f'<div class="fc-status {kind}"><span class="fc-dot"></span><div>{md_inline(body.strip())}</div></div>'


def tour_stepper(stops: list[dict], current: int | None, done_upto: int) -> str:
    """Door › M-006 › CV-003 › … chips; done < done_upto, current = index being shown."""
    parts = [f'<span class="fc-step done">{icon("door", 12)}Door</span>']
    for i, st in enumerate(stops):
        _, var = _state(st.get("condition_state"))
        cls = "now" if i == current else "done" if i < done_upto else ""
        parts.append('<span class="fc-sep">›</span>'
                     f'<span class="fc-step {cls}" style="--c: var({var})"><i></i>{esc(st.get("tag", "?"))}</span>')
    return f'<div class="fc-steps">{"".join(parts)}</div>'


def stop_line(text: str) -> str:
    return f'<div class="fc-stop">{md_inline(text)}</div>' if text.strip() else ""


def asset_hero(a: dict, sources: list[str] | None = None) -> str:
    """Info tab's top card for an Excel asset (fields of the scene node's asset record); sources = Excel
    provenance chips (sheet · row)."""
    state = a.get("condition_state")
    _, var = _state(state)
    crit = str(a.get("criticality") or "").split(" - ")[0]
    meta = [status_pill(state)]
    if a.get("urgency"):
        meta.append(f'<span class="fc-chip">{esc(words(a["urgency"]))} urgency</span>')
    if crit:
        meta.append(f'<span class="fc-chip">Criticality {esc(crit)}</span>')
    issue = ""
    if a.get("issue_code") and state != "healthy":
        issue = (f'<div class="fc-issue" style="--c: var({var})"><b>{esc(words(a["issue_code"]))}</b>'
                 f'{esc(a.get("issue_summary", ""))}</div>')
    return (f'<div class="fc-hero"><div class="fc-hero-top"><div><div class="fc-tag">{esc(a.get("tag", "?"))}</div>'
            f'<div class="fc-class">{esc(words(a.get("asset_class", "")))}'
            f'{" · " + esc(a["area_name"]) if a.get("area_name") else ""}</div></div></div>'
            f'<div class="fc-meta">{"".join(meta)}</div>{health_bar(a.get("health_score"), state)}{issue}'
            f'{sources_line(sources or [])}</div>')


def object_hero(n: dict, u2m: float) -> str:
    """Info tab's top card for a scene object without an Excel record."""
    e = [x * u2m for x in n.get("extent", [0, 0, 0])]
    return (f'<div class="fc-hero"><div class="fc-tag">{esc(words(n.get("label", "object")))}</div>'
            f'<div class="fc-class">{esc(n["id"])} · {e[0]:.2f} × {e[1]:.2f} × {e[2]:.2f} m</div>'
            f'<div class="fc-meta">{status_pill(None)}<span class="fc-chip">Not in the Excel pack</span></div></div>')


def audio_card(b64: str, caption: str) -> str:
    return (f'<div class="fc-audio"><div class="fc-audio-h">{icon("volume", 15)}{esc(caption)}'
            f'<span class="fc-wave"><i></i><i></i><i></i><i></i></span></div>'
            f'<audio controls autoplay src="data:audio/mpeg;base64,{b64}"></audio></div>')


# ---------- simplified, script-ordered panel (Scene / Walk / Info / Edit / Ask) ----------

def brand_bar() -> str:
    return (f'<div class="fc-head"><div class="fc-mark">{icon("route", 18)}</div><div>'
            f'<div class="fc-title">Field Copilot</div><div class="fc-sub">Equipment room · inspection assistant</div>'
            f'</div></div>')


def scene_stats(objects: int, mapped: int, attention: int, scanning: bool = False) -> str:
    """Extraction counters (climb live while 'Replay extraction' runs)."""
    cls = " live" if scanning else ""
    return (f'<div class="fc-stats{cls}">'
            f'<div class="fc-stat"><b>{objects}</b><span>objects extracted</span></div>'
            f'<div class="fc-stat"><b>{mapped}</b><span>mapped to Excel</span></div>'
            f'<div class="fc-stat crit"><b>{attention}</b><span>need attention</span></div></div>')


def sources_line(chips: list[str]) -> str:
    """Excel provenance chips under the asset card: 'Asset_Master · row 15', ..."""
    return ('<div class="fc-src">' + "".join(f'<span class="fc-chip">{esc(c)}</span>' for c in chips)
            + "</div>") if chips else ""


EXTRA_CSS = """<style>
.fc-stats.live .fc-stat b { color: var(--accent-hi); }
.fc-src { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 10px; }
.fc-src .fc-chip { font-size: 11px; height: 20px; font-variant-numeric: tabular-nums; }
</style>"""
