import importlib.util
import shutil
from pathlib import Path

from ._pgserver import verify_pgserver_bundle

_OMP_DLL = {"amd64": "libomp140.x86_64.dll", "arm64": "libomp140.aarch64.dll"}


def prepare(ctx):
    arch = ctx.arch
    vendor = ctx.root / "native-build" / "windows" / "vendor"
    pg_contrib = vendor / "pg-contrib" / arch
    omp_dll = vendor / "numkong" / arch / _OMP_DLL[arch]
    required = [
        vendor / "redis" / arch / "redis-server.exe",
        pg_contrib / "lib" / "unaccent.dll",
        pg_contrib / "lib" / "pg_trgm.dll",
        pg_contrib / "extension" / "unaccent.control",
        pg_contrib / "extension" / "pg_trgm.control",
        pg_contrib / "tsearch_data" / "unaccent.rules",
        omp_dll,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        for m in missing:
            print(f"[ERROR] Missing vendored file: {m}")
        raise SystemExit("Vendored inputs missing (see native-build/windows/vendor/README.md).")

    _stage_numkong_openmp(omp_dll)


def _stage_numkong_openmp(omp_dll):
    # Drop numkong's unbundled libomp next to the installed extension.
    spec = importlib.util.find_spec("numkong")
    if not spec or not spec.origin:
        print("[WARN] numkong not installed in build venv; the i8 SIMD kernels will be absent.")
        return
    dest = Path(spec.origin).parent / omp_dll.name
    if not dest.exists():
        shutil.copy2(omp_dll, dest)


def package(ctx):
    if ctx.use_pgserver:
        verify_pgserver_bundle(ctx, strict=False)
    return [ctx.bundle_dir]
