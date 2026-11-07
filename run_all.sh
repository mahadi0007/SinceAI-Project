#!/usr/bin/env bash
# One command: environment -> data check -> pipeline -> viewer. Options: see run_all.py (--rebuild, --vision, --bim, --no-viewer)
cd "$(dirname "$0")"
if command -v python3 >/dev/null 2>&1; then exec python3 run_all.py "$@"; fi
if command -v uv >/dev/null 2>&1; then exec uv run --no-project --python 3.12 python run_all.py "$@"; fi
echo "Python 3 not found: install Python 3.12 (python.org) or uv (https://docs.astral.sh/uv/)." >&2; exit 1
