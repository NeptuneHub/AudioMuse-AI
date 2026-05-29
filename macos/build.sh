#!/usr/bin/env bash
set -euo pipefail

APP="dist/AudioMuse-AI.app"
ENTITLEMENTS="macos/entitlements.plist"

echo "==> Cleaning previous build"
rm -rf build dist

echo "==> Generating icons from screenshot/audiomuseai.png"
bash macos/make_icns.sh

echo "==> Running PyInstaller"
pyinstaller macos/AudioMuse-AI.spec --noconfirm

echo "==> Ad-hoc signing nested binaries"
find "$APP/Contents" \
  \( -name "*.dylib" -o -name "*.so" -o -name "redis-server" -o -name "postgres" \
     -o -name "initdb" -o -name "pg_ctl" -o -name "psql" -o -name "pg_isready" \) -print0 \
  | while IFS= read -r -d '' f; do
      codesign --force --timestamp=none --sign - --entitlements "$ENTITLEMENTS" "$f" 2>/dev/null || true
    done

echo "==> Ad-hoc signing the bundle"
codesign --force --deep --timestamp=none --sign - --entitlements "$ENTITLEMENTS" "$APP"

echo "==> Verifying signature (rejection by spctl is expected for an unsigned app)"
codesign --verify --verbose "$APP" || true
spctl -a -vv "$APP" || true

ARCH="$(uname -m)"
ZIP="dist/AudioMuse-AI-${ARCH}.zip"
echo "==> Packaging ${ZIP}"
ditto -c -k --keepParent "$APP" "$ZIP"

echo "==> Done: ${ZIP}"
echo "    End users must clear quarantine after download:"
echo "    xattr -dr com.apple.quarantine /Applications/AudioMuse-AI.app"
