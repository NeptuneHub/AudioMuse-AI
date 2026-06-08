"""Windows packaging: verify vendored inputs, smoke-test the embedded PG, zip.

Mirrors the old ``windows/build.bat`` (clean + PyInstaller live in ``build.py``).
The bundle is zipped with the stdlib ``zipfile`` (ZIP64 enabled, automatic for the
>4 GB tree) instead of PowerShell ``Compress-Archive`` so the orchestrator stays
pure Python.
"""

import zipfile

from ._pgserver import verify_pgserver_bundle


def prepare(ctx):
    arch = ctx.arch
    required = [
        ctx.root / "windows" / "vendor" / "redis" / arch / "redis-server.exe",
        ctx.root / "windows" / "vendor" / "pg-contrib" / arch / "lib" / "unaccent.dll",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        for m in missing:
            print(f"[ERROR] Missing vendored file: {m}")
        raise SystemExit("Vendored inputs missing (see windows/vendor/README.md).")


def _zip_dir(src_dir, out_path):
    base = src_dir.parent
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for path in sorted(src_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(base).as_posix())


def package(ctx):
    if ctx.use_pgserver:
        verify_pgserver_bundle(ctx)
    out = ctx.dist_dir / f"AudioMuse-AI-{ctx.arch}-windows.zip"
    if out.exists():
        out.unlink()
    print(f"==> Packaging {out.name}")
    _zip_dir(ctx.bundle_dir, out)
    print(f"==> ZIP: {out}")
    return [out]
