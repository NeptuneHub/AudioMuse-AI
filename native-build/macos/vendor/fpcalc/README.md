# Vendored fpcalc (Chromaprint) - macOS

`arm64/fpcalc` and `x86_64/fpcalc` are the prebuilt Chromaprint fpcalc binaries,
committed to the repo so the standalone macOS build bundles the right one automatically
(no manual step). The launcher points the `FPCALC` env var at it at runtime
(native-build/macos/env.py, paths.py). In the Docker images fpcalc instead comes from
the `libchromaprint-tools` apt package.

- Version: Chromaprint 1.6.0 (fpcalc, ships its own ffmpeg decoder)
- Source: https://github.com/acoustid/chromaprint/releases/tag/v1.6.0
  (`chromaprint-fpcalc-1.6.0-macos-arm64.tar.gz`, `...-macos-x86_64.tar.gz`)
- License: LGPL-2.1-or-later; source available at the URL above.

To update: download the newer `macos-arm64` / `macos-x86_64` archives, extract
`fpcalc`, `chmod +x`, and replace `arm64/fpcalc` / `x86_64/fpcalc`.
