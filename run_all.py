"""One command: set up the environment, build the scene, start the viewer.

    ./run_all.sh            (macOS / Linux)      run_all.bat            (Windows)
    python3 run_all.py      (any OS; standard library only, so it runs before anything is installed)

Steps (each prints OK / FAIL):
  1. Python environment: creates .venv with Python 3.12 (uv if available, else a python3.12 on PATH) and installs
     requirements.txt (again only when requirements.txt changes).
  2. .env: created from .env.example if missing (API keys are optional).
  3. Challenge data: checks paths.ply / paths.xlsx from config.yaml.
  4. Pipeline: segment -> label -> navgrid -> mapping -> tour. Skipped when the outputs already exist,
     so results you built (e.g. with boundary repair or paid vision labels) are never overwritten.
  5. Viewer: http://localhost:8080 (Ctrl+C to stop).

Options:
  --rebuild       run the pipeline even if outputs exist
  --vision        label objects with the vision model (paid Featherless requests, capped by copilot.budget)
  --bim           also export the IFC/BIM model (Display: BIM in the viewer)
  --no-viewer     stop after the pipeline (e.g. for CI)
Env VALMET_OUT=<folder> writes the pipeline outputs to another folder (default out/).
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
WIN = os.name == "nt"
PY = VENV / ("Scripts/python.exe" if WIN else "bin/python")
STAMP = VENV / ".requirements.sha1"
COLOR = sys.stdout.isatty() and not WIN or os.environ.get("WT_SESSION")
GREEN, RED, DIM, BOLD, END = (("\033[32m", "\033[31m", "\033[2m", "\033[1m", "\033[0m") if COLOR
                               else ("", "", "", "", ""))


def ok(msg: str) -> None:
    print(f"{GREEN}  OK  {END} {msg}", flush=True)


def fail(msg: str, hint: str = "") -> None:
    print(f"{RED}  FAIL{END} {msg}", flush=True)
    if hint:
        print(f"{DIM}        {hint}{END}", flush=True)
    sys.exit(1)


def step(title: str) -> None:
    print(f"\n{BOLD}{title}{END}", flush=True)


def run(cmd: list[str], env: dict | None = None, quiet: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=ROOT, env=env, text=True,
                          stdout=subprocess.PIPE if quiet else None, stderr=subprocess.STDOUT if quiet else None)


def last_lines(text: str | None, n: int = 12) -> str:
    return "\n".join((text or "").strip().splitlines()[-n:])


# ---------- 1. environment ----------

def make_venv() -> None:
    if PY.exists():
        return
    uv = shutil.which("uv")
    if uv:
        r = run([uv, "venv", "--python", "3.12", str(VENV)])
    else:
        py = next((p for p in ("python3.12", "python3.13", "python3.11") if shutil.which(p)), None)
        if py is None and WIN and shutil.which("py"):
            py = "py"
        if py is None:
            fail("No Python 3.11+ found to create .venv",
                 "Install Python 3.12 (python.org) or uv (https://docs.astral.sh/uv/), then run again.")
        r = run([py] + (["-3.12"] if py == "py" else []) + ["-m", "venv", str(VENV)])
    if r.returncode != 0 or not PY.exists():
        fail("Could not create .venv", last_lines(r.stdout))


def install_requirements() -> None:
    req = (ROOT / "requirements.txt").read_bytes()
    digest = hashlib.sha1(req).hexdigest()
    if STAMP.exists() and STAMP.read_text().strip() == digest:
        ok("packages up to date")
        return
    print(f"{DIM}        installing requirements.txt (first run takes a few minutes: torch, open3d, ...){END}",
          flush=True)
    uv = shutil.which("uv")
    if uv:
        r = run([uv, "pip", "install", "--python", str(PY), "-r", "requirements.txt"])
    else:
        run([str(PY), "-m", "ensurepip", "--upgrade"])
        r = run([str(PY), "-m", "pip", "install", "-r", "requirements.txt"])
    if r.returncode != 0:
        fail("pip install failed", last_lines(r.stdout))
    STAMP.write_text(digest)
    ok("packages installed")


def child_env() -> dict:
    """Environment for every step. macOS: open3d's wheel links Homebrew's libusb; if that is missing we point
    the loader at the pip package libusb-package (installed on demand by fix_open3d)."""
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    lib = next(iter(VENV.glob("lib/python3*/site-packages/libusb_package")), None)
    if sys.platform == "darwin" and lib is not None:
        env["DYLD_FALLBACK_LIBRARY_PATH"] = f"{lib}:{env.get('DYLD_FALLBACK_LIBRARY_PATH', '')}".rstrip(":")
    return env


def fix_open3d() -> dict:
    env = child_env()
    r = run([str(PY), "-c", "import open3d"], env=env)
    if r.returncode == 0:
        ok("open3d loads")
        return env
    if sys.platform == "darwin" and "libusb" in (r.stdout or ""):
        uv = shutil.which("uv")
        run([uv, "pip", "install", "--python", str(PY), "libusb-package"] if uv
            else [str(PY), "-m", "pip", "install", "libusb-package"])
        lib = next(iter(VENV.glob("lib/python3*/site-packages/libusb_package")), None)
        if lib is not None and not (lib / "libusb-1.0.0.dylib").exists() and (lib / "libusb-1.0.dylib").exists():
            (lib / "libusb-1.0.0.dylib").symlink_to("libusb-1.0.dylib")
        env = child_env()
        if run([str(PY), "-c", "import open3d"], env=env).returncode == 0:
            ok("open3d loads (libusb supplied by the libusb-package wheel)")
            return env
        fail("open3d cannot load libusb", "Run: brew install libusb   then run again.")
    fail("open3d does not import", last_lines(r.stdout))
    return env


# ---------- 2-3. config, keys, data ----------

def check_env_file() -> None:
    env = ROOT / ".env"
    if not env.exists():
        shutil.copy(ROOT / ".env.example", env)
        ok(".env created from .env.example (optional: add FEATHERLESS_API_KEY / ELEVENLABS_API_KEY)")
        return
    keys = [k for k in ("FEATHERLESS_API_KEY", "ELEVENLABS_API_KEY")
            if any(line.startswith(k + "=") and "your-key-here" not in line and line.strip() != k + "="
                   for line in env.read_text(encoding="utf-8").splitlines())]
    ok(".env present" + (f" ({', '.join(keys)} set)" if keys else " (no API keys: rule labels, evidence-only "
                                                                   "diagnoses, no voice)"))


def check_data(env: dict) -> str:
    code = ("from pipeline.common import load_config, repo_path, out_dir; c = load_config(); "
            "print(repo_path(c['paths']['ply'])); print(repo_path(c['paths']['xlsx'])); print(out_dir(c))")
    r = run([str(PY), "-c", code], env=env)
    if r.returncode != 0:
        fail("config.yaml could not be read", last_lines(r.stdout))
    ply, xlsx, out = r.stdout.strip().splitlines()[-3:]
    for p in (ply, xlsx):
        if not Path(p).exists():
            fail(f"missing {p}", "Put the challenge files there or change paths.ply / paths.xlsx in config.yaml.")
    ok(f"scan and Excel pack found ({Path(ply).name}, {Path(xlsx).name})")
    return out


# ---------- 4. pipeline ----------

def pipeline(env: dict, out: str, rebuild: bool, vision: bool, bim: bool) -> None:
    outputs = ("assets.json", "navgrid.npz", "path.json", "scene_graph.json", "tour.json")
    if not rebuild and all((Path(out) / f).exists() for f in outputs):
        ok(f"outputs already in {Path(out).name}/ (use --rebuild to regenerate)")
        stages = []
    else:
        stages = [("Extract objects", ["pipeline.segment"]),
                  ("Label objects", ["pipeline.label"] + ([] if vision else ["--rules-only"])),
                  ("Occupancy grid + route", ["pipeline.navgrid"]),
                  ("Map Excel assets", ["pipeline.mapping"]),
                  ("Inspection tour", ["pipeline.tour"])]
    if bim and (rebuild or not (Path(out) / "scene.ifc").exists()):
        stages.append(("BIM model (IFC)", ["pipeline.export_ifc"]))
    for title, args in stages:
        t0 = time.time()
        r = run([str(PY), "-m", *args], env=env)
        if r.returncode != 0:
            fail(f"{title}  (python -m {' '.join(args)})", last_lines(r.stdout))
        ok(f"{title}  {DIM}{time.time() - t0:.0f} s{END}")


# ---------- main ----------

def main() -> None:
    args = set(sys.argv[1:])
    unknown = args - {"--rebuild", "--vision", "--bim", "--no-viewer", "-h", "--help"}
    if unknown or args & {"-h", "--help"}:
        print(__doc__)
        sys.exit(1 if unknown else 0)
    print(f"{BOLD}Valmet Field Copilot · Team Overfit{END}")
    step("1/4 Environment")
    make_venv()
    ok(f".venv ({run([str(PY), '--version']).stdout.strip()})")
    install_requirements()
    env = fix_open3d()
    step("2/4 Configuration and data")
    check_env_file()
    out = check_data(env)
    step("3/4 Pipeline")
    pipeline(env, out, "--rebuild" in args, "--vision" in args, "--bim" in args)
    if "--no-viewer" in args:
        print(f"\n{GREEN}{BOLD}All steps passed.{END}")
        return
    step("4/4 Viewer")
    ok("starting: open http://localhost:8080 (Ctrl+C to stop; another port is used if 8080 is busy)")
    try:
        sys.exit(subprocess.call([str(PY), "-m", "viewer.app"], cwd=ROOT, env=env))
    except KeyboardInterrupt:
        print("\nViewer stopped.")


if __name__ == "__main__":
    main()
