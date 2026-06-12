#!/usr/bin/env bash
set -euo pipefail

SRC="screenshot/audiomuseai.png"
OUT_DIR="native-build/macos/assets"
WORK="$(mktemp -d)/AudioMuse-AI.iconset"
mkdir -p "$WORK" "$OUT_DIR"

for size in 16 32 128 256 512; do
  double=$((size * 2))
  sips -z "$size" "$size" "$SRC" --out "$WORK/icon_${size}x${size}.png" >/dev/null
  sips -z "$double" "$double" "$SRC" --out "$WORK/icon_${size}x${size}@2x.png" >/dev/null
done

iconutil -c icns "$WORK" -o "$OUT_DIR/AudioMuse-AI.icns"
sips -z 44 44 "$SRC" --out "$OUT_DIR/menubar-icon.png" >/dev/null

echo "Wrote $OUT_DIR/AudioMuse-AI.icns and $OUT_DIR/menubar-icon.png"
