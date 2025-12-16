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
  echo "Activate your virtual environment or set PYINSTALLER_BIN before running this script." >&2
  echo "Example:" >&2
  echo "  python3 -m venv .venv-build" >&2
  echo "  .venv-build/bin/pip install -r requirements.txt" >&2
  echo "  ./scripts/build-mac.sh" >&2
  exit 1
fi

pushd "$PROJECT_ROOT" >/dev/null
"$PYINSTALLER_BIN" --noconfirm SpaceParticles.spec
popd >/dev/null

if [[ "$(uname -s)" == "Darwin" ]]; then
  APP_PATH="$PROJECT_ROOT/dist/SpaceParticles/SpaceParticles.app"
  if [[ -d "$APP_PATH" ]]; then
    if command -v codesign >/dev/null 2>&1; then
      export MACOS_ENTITLEMENTS_PATH="${MACOS_ENTITLEMENTS_PATH:-$PROJECT_ROOT/scripts/entitlements.plist}"
      "$PROJECT_ROOT/scripts/sign-mac.sh" "$APP_PATH"
      "$PROJECT_ROOT/scripts/notarize-mac.sh" "$APP_PATH"
      if command -v xattr >/dev/null 2>&1; then
        echo "Clearing extended attributes that would trigger quarantine prompts..."
        xattr -cr "$APP_PATH"
      fi
    else
      echo "codesign utility not found; skipping macOS bundle signing" >&2
    fi
  fi
fi
