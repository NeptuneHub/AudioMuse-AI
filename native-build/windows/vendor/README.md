# Windows vendored build inputs

Native binaries the Windows standalone build bundles but does not pull from PyPI.
`scripts/standalone/platforms/windows.py:prepare()` checks every file listed here
exists before PyInstaller runs and aborts the build with the missing path if not.
Layout is `vendor/<component>/<arch>/...` (arch is `amd64` or `arm64`).

## redis/<arch>/redis-server.exe
Embedded Redis used as the RQ broker. Built locally with `build-redis.bat`.

## pg-contrib/<arch>/
`unaccent` and `pg_trgm` contrib modules (`.dll` + `.control` + `.sql` + the
`unaccent.rules` dictionary), compiled against the PostgreSQL minor that the
pinned `pgserver` wheel ships (16.2 -- contrib is not ABI-stable across minors).
Regenerate with `pg-contrib/build-pg-contrib-cross.sh`.

## numkong/<arch>/libomp140.x86_64.dll (LLVM OpenMP runtime)
numkong's Windows wheel links the LLVM OpenMP runtime (`libomp140.x86_64.dll`)
but -- unlike its Linux and macOS wheels, which vendor `libgomp` / `libomp.dylib`
inside the wheel -- it ships only the `.pyd` and bundles nothing. Its own METADATA
even claims "There is no OpenMP dependency". Without this DLL `import numkong`
fails with "DLL load failed while importing _numkong", and the i8/f16 IVF cells
silently fall back to the slower NumPy distance path.

The spec bundles this next to the frozen `numkong` extension, and `prepare()`
also copies it next to the installed `_numkong*.pyd` in the build venv so the
extension imports during PyInstaller analysis and in local runs.

- Source: conda-forge `llvm-openmp` 22.1.8 (win-64), file `Library/bin/libomp.dll`,
  renamed to `libomp140.x86_64.dll` (same LLVM OpenMP runtime, ABI-compatible with
  the import name MSVC's `/openmp:llvm` emits).
- License: Apache-2.0 WITH LLVM-exception (see `amd64/LICENSE-libomp.txt`).
- arm64: not vendored yet. An arm64 build needs `arm64/libomp140.aarch64.dll`
  from the matching `llvm-openmp` win-arm64 package.
