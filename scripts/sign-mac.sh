#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"
APP_PATH_DEFAULT="$PROJECT_ROOT/dist/SpaceParticles/SpaceParticles.app"

APP_PATH="${1:-$APP_PATH_DEFAULT}"
IDENTITY="${CODESIGN_IDENTITY:-}"
ENTITLEMENTS="${MACOS_ENTITLEMENTS_PATH:-$PROJECT_ROOT/scripts/entitlements.plist}"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "macOS signing skipped: not running on Darwin." >&2
  exit 0
fi

if [[ -z "$IDENTITY" ]]; then
  echo "CODESIGN_IDENTITY is not set; skipping macOS signing." >&2
  exit 0
fi

if [[ ! -d "$APP_PATH" ]]; then
  echo "App bundle not found at $APP_PATH" >&2
  exit 1
fi

if [[ ! -f "$ENTITLEMENTS" ]]; then
  echo "Entitlements file not found at $ENTITLEMENTS" >&2
  exit 1
fi

echo "Signing $APP_PATH ..."
codesign \
  --force \
  --deep \
  --options runtime \
  --timestamp \
  --entitlements "$ENTITLEMENTS" \
  --sign "$IDENTITY" \
  "$APP_PATH"

echo "Verifying signature..."
codesign --verify --deep --strict --verbose=2 "$APP_PATH"
spctl --assess --verbose=2 "$APP_PATH"
