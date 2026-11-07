# Valmet Field Copilot: Since AI Hackathon 2026

Turns a Gaussian-splat scan of a plant room into a 3D scene that knows its equipment. Objects are extracted
and labelled, the Valmet Excel pack is mapped onto them, and a worker walks a collision-free route to the faulty
pump **P-002**, where an AI copilot gives a cited, spoken diagnosis.

**Pipeline:** `segment → label → navgrid → mapping → tour → copilot → viewer`
(details per stage: [docs/DETAILS.md](docs/DETAILS.md))

## Setup and run: one command

```bash
git clone https://github.com/mahadi0007/SinceAI-Project.git && cd SinceAI-Project
./run_all.sh            # Windows: run_all.bat
```

`run_all` does everything and prints OK / FAIL per step:
1. Creates `.venv` with Python 3.12 (via `uv` if installed, else a `python3.12` on PATH) and installs
   `requirements.txt`. The first run takes a few minutes; later runs skip this.
2. Creates `.env` from `.env.example` if missing.
3. Checks that the challenge data is in place (see below).
4. Runs the pipeline (about 15 s, no API calls). It's skipped when the outputs already exist, so results you
   built are never overwritten.
5. Starts the viewer at http://localhost:8080.

Options: `--rebuild` (rerun the pipeline), `--vision` (label with the vision model, paid), `--bim` (also export the
IFC model), `--no-viewer` (stop after the pipeline).

**Challenge data (not in git).** Put the two files next to the repo, or change `paths` in `config.yaml`:

```
../sinceai2026/data/raw/Equipment Room.ply
../sinceai2026/data/raw/equipment.xlsx
```

**API keys in `.env` (all optional).** Without keys everything still runs, using rule-based labels and
evidence-only diagnoses.

| Key | Used for |
|---|---|
| `FEATHERLESS_API_KEY` | Vision labelling and live AI diagnoses (Qwen3.5). Paid requests are capped by `copilot.budget` in `config.yaml`. |
| `ELEVENLABS_API_KEY` | The spoken copilot briefings |

**macOS:** `run_all` fixes the known open3d/libusb load error itself. If it still fails, run `brew install libusb`.

## Manual steps (what run_all runs)

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt   # Windows: .venv\Scripts\pip
python -m pipeline.segment                 # splat -> 934 objects + floor/walls/ceiling
python -m pipeline.label --rules-only      # names from geometry rules (drop the flag to use the vision model)
python -m pipeline.navgrid                 # occupancy grid + route from the door to the target
python -m pipeline.mapping                 # Excel assets -> 3D objects, with confidence
python -m pipeline.tour                    # inspection tour past the assets that need attention
python -m viewer.app                       # open http://localhost:8080
```

Run the commands from the repo root with the venv's Python. Every stage writes to `out/` (set `VALMET_OUT` to
use another folder). All tuning values are in `config.yaml`.

## Using the viewer

| Tab | What you can do |
|---|---|
| **Scene** | Replay the extraction, colour boxes by Excel health, show the occupancy grid, switch Scan / BIM / Overlay |
| **Walk** | Walk to an equipment, pick any object in the scan to walk to, or play the inspection tour (POV switch) |
| **Info** | The clicked object and the selected equipment: Excel data with sheet/row sources, spoken summary, cited diagnosis |
| **Edit** | Move, remove or duplicate objects, then **Apply layout** to re-plan routes (needs `python -m viewer.removable`) |
| **Ask** | Chat: quick commands (`take me to M-006`, `start tour`) or questions answered by Claude through the MCP tools |

## Optional extras

| Feature | How |
|---|---|
| Diagnosis from the command line | `python -m copilot.agent P-002` (1 paid request when a Featherless key is set; cached after) |
| Test the voice | `python -m copilot.voice --dummy --play` |
| BIM model (IFC4) | `python -m pipeline.export_ifc`, then Display: BIM in the viewer |
| Rigged worker avatar | Mixamo download, see [docs/DETAILS.md](docs/DETAILS.md#worker-avatar-optional-mixamo); otherwise a simple figure is used |
| MCP server for Claude | `.mcp.json` registers `mcp_server.py`; on macOS/Linux set `VALMET_PYTHON=.venv/bin/python` |
| Boundary repair and manual correction | See [docs/DETAILS.md](docs/DETAILS.md) |

## Tests

```bash
python -m pytest -q tests
```

## Repo layout

| Folder | Contents |
|---|---|
| `run_all.py` (+ `.sh` / `.bat`) | One-command setup, pipeline and viewer |
| `pipeline/` | Extraction, labelling, navigation, mapping, tour, boundary repair, IFC export |
| `copilot/` | Evidence pack, LLM access with budget, diagnosis with citation check, voice, chat |
| `viewer/` | Viser viewer (`app.py`), UI theme (`ui.py`), avatar, editing |
| `api/`, `mcp_server.py` | Manual correction tool and the MCP server |
| `review/` | Reviewed reference regions and measured results (`review/RESULTS.md`) |
