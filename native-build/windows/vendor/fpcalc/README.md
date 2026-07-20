# Vendored fpcalc (Chromaprint) - Windows

`amd64/fpcalc.exe` is the prebuilt, statically linked Chromaprint fpcalc binary,
committed to the repo so the standalone Windows build bundles it automatically (no
manual step). The launcher points the `FPCALC` env var at it at runtime
(native-build/windows/env.py, paths.py). In the Docker images fpcalc instead comes
from the `libchromaprint-tools` apt package.

- Version: Chromaprint 1.6.0 (fpcalc, statically linked with its own ffmpeg decoder)
- Source: https://github.com/acoustid/chromaprint/releases/tag/v1.6.0
  (`chromaprint-fpcalc-1.6.0-windows-x86_64.zip`)
- License: LGPL-2.1-or-later; source available at the URL above.

To update: download the newer `windows-x86_64` archive, extract `fpcalc.exe`, and
replace `amd64/fpcalc.exe`.
