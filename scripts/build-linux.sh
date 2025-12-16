#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"

if [[ -z "${PYINSTALLER_BIN:-}" ]]; then
  VENV_PATHS=(
    "${VENV_PATH:-$PROJECT_ROOT/.venv-build}"
    "$PROJECT_ROOT/.venv"
  )
  for candidate in "${VENV_PATHS[@]}"; do
    if [[ -x "$candidate/bin/pyinstaller" ]]; then
      PYINSTALLER_BIN="$candidate/bin/pyinstaller"
      break
    fi
  done
fi

if [[ ! -x "$PYINSTALLER_BIN" ]]; then
  echo "PyInstaller not found at $PYINSTALLER_BIN" >&2
  echo "Create or activate a Linux virtual environment before running this script." >&2
  echo "Example:" >&2
  echo "  python3 -m venv .venv-build" >&2
  echo "  .venv-build/bin/pip install -r requirements.txt" >&2
  echo "  ./scripts/build-linux.sh" >&2
  exit 1
fi

pushd "$PROJECT_ROOT" >/dev/null
"$PYINSTALLER_BIN" --noconfirm SpaceParticles.spec
popd >/dev/null
