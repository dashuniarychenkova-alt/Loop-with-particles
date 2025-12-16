#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"
APP_PATH_DEFAULT="$PROJECT_ROOT/dist/SpaceParticles/SpaceParticles.app"
ZIP_PATH_DEFAULT="$PROJECT_ROOT/dist/SpaceParticles-notarize.zip"

APP_PATH="${1:-$APP_PATH_DEFAULT}"
ZIP_PATH="${2:-$ZIP_PATH_DEFAULT}"

APPLE_ID="${NOTARIZE_APPLE_ID:-}"
APPLE_PASSWORD="${NOTARIZE_APPLE_PASSWORD:-}"
APPLE_TEAM_ID="${NOTARIZE_TEAM_ID:-}"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "Notarization skipped: not running on macOS." >&2
  exit 0
fi

if [[ -z "$APPLE_ID" || -z "$APPLE_PASSWORD" || -z "$APPLE_TEAM_ID" ]]; then
  echo "Notarization skipped: NOTARIZE_APPLE_ID, NOTARIZE_APPLE_PASSWORD, or NOTARIZE_TEAM_ID missing." >&2
  exit 0
fi

if [[ ! -d "$APP_PATH" ]]; then
  echo "App bundle not found at $APP_PATH" >&2
  exit 1
fi

echo "Creating ZIP for notarization at $ZIP_PATH ..."
rm -f "$ZIP_PATH"
pushd "$(dirname "$APP_PATH")" >/dev/null
ditto -c -k --keepParent "$(basename "$APP_PATH")" "$ZIP_PATH"
popd >/dev/null

echo "Submitting for notarization..."
xcrun notarytool submit "$ZIP_PATH" \
  --apple-id "$APPLE_ID" \
  --password "$APPLE_PASSWORD" \
  --team-id "$APPLE_TEAM_ID" \
  --wait

echo "Stapling notarization ticket..."
xcrun stapler staple "$APP_PATH"
spctl --assess --verbose=2 "$APP_PATH"
