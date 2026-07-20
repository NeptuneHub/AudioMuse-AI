# Vendored fpcalc (Chromaprint) - Linux

`x86_64/fpcalc` and `aarch64/fpcalc` are the prebuilt, statically linked Chromaprint
fpcalc binaries, committed to the repo so the standalone Linux build bundles the right
one automatically (no manual step). The launcher points the `FPCALC` env var at it at
runtime (native-build/linux/env.py, paths.py). In the Docker images fpcalc instead
comes from the `libchromaprint-tools` apt package.

- Version: Chromaprint 1.6.0 (fpcalc, statically linked - `file` reports "statically
  linked ... not a dynamic executable", so no ffmpeg/libav runtime deps are needed)
- Source: https://github.com/acoustid/chromaprint/releases/tag/v1.6.0
  (`chromaprint-fpcalc-1.6.0-linux-x86_64.tar.gz`, `...-linux-arm64.tar.gz`)
- License: LGPL-2.1-or-later; source available at the URL above.

To update: download the newer `linux-x86_64` / `linux-arm64` archives, extract
`fpcalc`, `chmod +x`, and replace `x86_64/fpcalc` / `aarch64/fpcalc`.
