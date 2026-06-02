#!/usr/bin/env bash
#
# Build the standalone Linux bundle with PyInstaller, then package it as a .deb
# and a .rpm with nfpm. Run from the repo root, inside the build venv:
#
#     source .venv-linux/bin/activate
#     PKG_VERSION=1.0.0 bash linux/build.sh
#
# Prerequisites (the CI workflow installs these):
#   * the Python deps from requirements/linux.txt (incl. pyinstaller, pgserver)
#   * the vendored redis-server + pg-contrib for this arch
#     (linux/vendor/...; built by linux/vendor/build-redis.sh and
#      linux/vendor/pg-contrib/build-pg-contrib.sh)
#   * nfpm on PATH (https://nfpm.goreleaser.com)
set -euo pipefail

PKG_VERSION="${PKG_VERSION:-0.0.0}"
# Strip a leading v from a git tag like v1.2.3 -> 1.2.3 (deb/rpm want a bare ver).
PKG_VERSION="${PKG_VERSION#v}"

UNAME_ARCH="$(uname -m)"   # x86_64 | aarch64
case "$UNAME_ARCH" in
  x86_64)  NFPM_ARCH="amd64" ;;
  aarch64) NFPM_ARCH="arm64" ;;
  *) echo "Unsupported arch: $UNAME_ARCH" >&2; exit 1 ;;
esac

echo "==> Cleaning previous build"
rm -rf build dist

echo "==> Verifying vendored native build inputs are present"
required=(
  "linux/vendor/redis/${UNAME_ARCH}/redis-server"
  "linux/vendor/pg-contrib/${UNAME_ARCH}/lib/unaccent.so"
  "linux/vendor/pg-contrib/${UNAME_ARCH}/lib/pg_trgm.so"
  "linux/vendor/pg-contrib/${UNAME_ARCH}/extension/unaccent.control"
  "linux/vendor/pg-contrib/${UNAME_ARCH}/extension/pg_trgm.control"
  "linux/vendor/pg-contrib/${UNAME_ARCH}/tsearch_data/unaccent.rules"
)
missing=0
for f in "${required[@]}"; do
  if [ ! -s "$f" ]; then echo "::error::Missing vendored file: $f"; missing=1; fi
done
[ "$missing" -eq 0 ] || { echo "Vendored inputs missing (see linux/vendor/*/README.md)." >&2; exit 1; }
chmod +x "linux/vendor/redis/${UNAME_ARCH}/redis-server"

echo "==> Running PyInstaller"
pyinstaller linux/AudioMuse-AI.spec --noconfirm

BUNDLE="dist/AudioMuse-AI"
[ -x "$BUNDLE/AudioMuse-AI" ] || { echo "::error::PyInstaller did not produce $BUNDLE/AudioMuse-AI"; exit 1; }

echo "==> Staging package payload"
STAGE="dist/_pkg"
rm -rf "$STAGE"
mkdir -p "$STAGE/opt" "$STAGE/usr/share/applications" \
         "$STAGE/usr/share/icons/hicolor/512x512/apps"
cp -a "$BUNDLE" "$STAGE/opt/AudioMuse-AI"
cp linux/packaging/AudioMuse-AI.desktop "$STAGE/usr/share/applications/AudioMuse-AI.desktop"
cp linux/packaging/AudioMuse-AI-stop.desktop "$STAGE/usr/share/applications/AudioMuse-AI-stop.desktop"
cp screenshot/audiomuseai.png "$STAGE/usr/share/icons/hicolor/512x512/apps/audiomuse-ai.png"

echo "==> Generating nfpm config"
mkdir -p dist
sed -e "s|@VERSION@|${PKG_VERSION}|g" \
    -e "s|@ARCH@|${NFPM_ARCH}|g" \
    -e "s|@STAGE@|${STAGE}|g" \
    linux/packaging/nfpm.yaml.in > dist/nfpm.yaml

echo "==> Building .deb and .rpm with nfpm (arch=${NFPM_ARCH})"
mkdir -p dist/pkg
nfpm package --config dist/nfpm.yaml --packager deb --target dist/pkg/
nfpm package --config dist/nfpm.yaml --packager rpm --target dist/pkg/

echo "==> Built packages:"
ls -lh dist/pkg/

# Normalize the output names so the workflow can find them deterministically.
DEB="$(ls dist/pkg/*.deb | head -n1)"
RPM="$(ls dist/pkg/*.rpm | head -n1)"
cp "$DEB" "dist/AudioMuse-AI-${UNAME_ARCH}.deb"
cp "$RPM" "dist/AudioMuse-AI-${UNAME_ARCH}.rpm"
echo "==> Done:"
echo "    dist/AudioMuse-AI-${UNAME_ARCH}.deb"
echo "    dist/AudioMuse-AI-${UNAME_ARCH}.rpm"
