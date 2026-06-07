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
ZIP="dist/AudioMuse-AI-${ARCH}-macos.zip"
echo "==> Packaging ${ZIP} (AudioMuse-AI.app + readme.md)"
# Stage the app + a plain-text install note, then archive both at the zip root.
# We archive with `ditto` (not `zip`): the bundle is >4 GB and needs ZIP64, which
# the legacy `zip` tool mishandles ("extra bytes" / corrupt archive). `cp -c`
# clones the app via APFS copy-on-write, so staging is instant and costs no extra
# disk while preserving the signature and the bundle's symlinks (ditto copy is the
# fallback on a non-APFS volume).
STAGE="dist/_pkg"
rm -rf "$STAGE"
mkdir -p "$STAGE"
cp -cR "$APP" "$STAGE/AudioMuse-AI.app" 2>/dev/null || ditto "$APP" "$STAGE/AudioMuse-AI.app"
cat > "$STAGE/readme.md" <<'EOF'
This AudioMuse-AI app is not signed to avoid Apple recurrent subscription cost. To have it working you need to:
- Move AudioMuse-AI.app in /Applications
- Open a terminal and run this command to authorize:
xattr -dr com.apple.quarantine /Applications/AudioMuse-AI.app

After this you can just open it like any other application.
EOF
rm -f "$ZIP"
ditto -c -k "$STAGE" "$ZIP"   # no --keepParent: app + readme land at the zip root
rm -rf "$STAGE"

echo "==> Done: ${ZIP} (expands to AudioMuse-AI.app + readme.md)"
echo "    End users must clear quarantine after download:"
echo "    xattr -dr com.apple.quarantine /Applications/AudioMuse-AI.app"
