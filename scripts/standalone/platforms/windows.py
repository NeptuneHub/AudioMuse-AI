from ._pgserver import verify_pgserver_bundle


def prepare(ctx):
    arch = ctx.arch
    pg_contrib = ctx.root / "windows" / "vendor" / "pg-contrib" / arch
    required = [
        ctx.root / "windows" / "vendor" / "redis" / arch / "redis-server.exe",
        pg_contrib / "lib" / "unaccent.dll",
        pg_contrib / "lib" / "pg_trgm.dll",
        pg_contrib / "extension" / "unaccent.control",
        pg_contrib / "extension" / "pg_trgm.control",
        pg_contrib / "tsearch_data" / "unaccent.rules",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        for m in missing:
            print(f"[ERROR] Missing vendored file: {m}")
        raise SystemExit("Vendored inputs missing (see windows/vendor/README.md).")


def package(ctx):
    if ctx.use_pgserver:
        verify_pgserver_bundle(ctx, strict=False)
    return [ctx.bundle_dir]
