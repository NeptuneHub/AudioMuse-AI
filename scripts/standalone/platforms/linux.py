"""Linux packaging: verify per-arch inputs, fix the embedded PG, build .deb/.rpm.

Mirrors the old ``linux/build.sh`` (clean + PyInstaller live in ``build.py``).

Embedded PostgreSQL differs by arch: x86_64 grafts the vendored unaccent/pg_trgm
contrib into the pgserver wheel (and the shared :func:`verify_pgserver_bundle`
restores the wheel's loadable modules + smoke-tests initdb); aarch64 bundles a
whole from-source PostgreSQL tree under ``pgsql/`` (pgserver has no arm64 wheel),
which is added as PyInstaller *data* and therefore loses the executable bit, so
its ``pgsql/bin`` binaries get ``+x`` restored here (the server relocates via
rpath; ``embedded_pg`` also sets ``LD_LIBRARY_PATH`` defensively).

Only verifiable in CI (ubuntu x86_64 + ubuntu-arm aarch64).
"""

import shutil
import subprocess

from ._pgserver import verify_pgserver_bundle

_NFPM_ARCH = {"x86_64": "amd64", "aarch64": "arm64"}
_ICON_SIZES = (512, 256, 128, 64, 48, 32)


def _present(path):
    return path.exists() and path.stat().st_size > 0


def prepare(ctx):
    arch = ctx.arch
    vendor = ctx.root / "linux" / "vendor"
    if ctx.use_pgserver:
        required = [
            vendor / "redis" / arch / "redis-server",
            vendor / "pg-contrib" / arch / "lib" / "unaccent.so",
            vendor / "pg-contrib" / arch / "lib" / "pg_trgm.so",
            vendor / "pg-contrib" / arch / "extension" / "unaccent.control",
            vendor / "pg-contrib" / arch / "extension" / "pg_trgm.control",
            vendor / "pg-contrib" / arch / "tsearch_data" / "unaccent.rules",
        ]
    else:
        required = [
            vendor / "redis" / arch / "redis-server",
            vendor / "postgres" / arch / "bin" / "postgres",
            vendor / "postgres" / arch / "bin" / "initdb",
            vendor / "postgres" / arch / "bin" / "pg_ctl",
        ]
    missing = [str(p) for p in required if not _present(p)]
    if not ctx.use_pgserver:
        pgtree = vendor / "postgres" / arch
        for name in ("unaccent.so", "pg_trgm.so", "unaccent.control", "pg_trgm.control"):
            if not any(pgtree.rglob(name)):
                missing.append(f"contrib artifact in {pgtree}: {name}")
    if missing:
        for m in missing:
            print(f"::error::Missing vendored file: {m}")
        raise SystemExit("Vendored inputs missing (see linux/vendor/*/README.md).")
    (vendor / "redis" / arch / "redis-server").chmod(0o755)


def _restore_aarch64_exec_bits(ctx):
    pgbin = next(
        (p for p in ctx.bundle_dir.rglob("bin") if p.is_dir() and p.parent.name == "pgsql"),
        None,
    )
    if pgbin is None:
        raise SystemExit(
            "::error::Expected bundled PostgreSQL (pgsql/bin) in the bundle (aarch64 build)"
        )
    for f in pgbin.rglob("*"):
        if f.is_file():
            f.chmod(f.stat().st_mode | 0o111)
    print(f"==> Restored +x on bundled PostgreSQL binaries ({pgbin})")


def _stage(ctx):
    stage = ctx.dist_dir / "_pkg"
    shutil.rmtree(stage, ignore_errors=True)
    (stage / "opt").mkdir(parents=True)
    (stage / "usr" / "share" / "applications").mkdir(parents=True)
    (stage / "usr" / "lib" / "systemd" / "user").mkdir(parents=True)
    subprocess.run(["cp", "-a", str(ctx.bundle_dir), str(stage / "opt" / "AudioMuse-AI")], check=True)

    pkg = ctx.root / "linux" / "packaging"
    shutil.copy2(pkg / "AudioMuse-AI.desktop", stage / "usr" / "share" / "applications" / "AudioMuse-AI.desktop")
    shutil.copy2(pkg / "AudioMuse-AI-stop.desktop", stage / "usr" / "share" / "applications" / "AudioMuse-AI-stop.desktop")
    shutil.copy2(pkg / "audiomuse-ai.service", stage / "usr" / "lib" / "systemd" / "user" / "audiomuse-ai.service")

    for size in _ICON_SIZES:
        src = pkg / "icons" / f"audiomuse-ai_{size}.png"
        if not _present(src):
            raise SystemExit(f"::error::Missing square icon source: {src}")
        dst = stage / "usr" / "share" / "icons" / "hicolor" / f"{size}x{size}" / "apps"
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst / "audiomuse-ai.png")
    return stage


def package(ctx):
    if ctx.use_pgserver:
        verify_pgserver_bundle(ctx)
    else:
        _restore_aarch64_exec_bits(ctx)

    _stage(ctx)
    nfpm_arch = _NFPM_ARCH[ctx.arch]

    print("==> Generating nfpm config")
    template = (ctx.root / "linux" / "packaging" / "nfpm.yaml.in").read_text()
    content = (
        template.replace("@VERSION@", ctx.version)
        .replace("@ARCH@", nfpm_arch)
        .replace("@STAGE@", "dist/_pkg")
    )
    (ctx.dist_dir / "nfpm.yaml").write_text(content)

    print(f"==> Building .deb and .rpm with nfpm (arch={nfpm_arch})")
    pkg = ctx.dist_dir / "pkg"
    pkg.mkdir(parents=True, exist_ok=True)
    for packager in ("deb", "rpm"):
        subprocess.run(
            ["nfpm", "package", "--config", str(ctx.dist_dir / "nfpm.yaml"),
             "--packager", packager, "--target", str(pkg) + "/"],
            check=True,
            cwd=str(ctx.root),
        )

    deb = sorted(pkg.glob("*.deb"))[0]
    rpm = sorted(pkg.glob("*.rpm"))[0]
    out_deb = ctx.dist_dir / f"AudioMuse-AI-{ctx.arch}-linux.deb"
    out_rpm = ctx.dist_dir / f"AudioMuse-AI-{ctx.arch}-linux.rpm"
    shutil.copy2(deb, out_deb)
    shutil.copy2(rpm, out_rpm)
    print(f"==> Done:\n    {out_deb}\n    {out_rpm}")
    return [out_deb, out_rpm]
