# Standalone (native) build

One parameterized build for the macOS, Linux and Windows desktop bundles. This
replaces the three near-identical PyInstaller specs and the three platform build
scripts (`windows/build.bat`, `macos/build.sh`, `linux/build.sh`) that used to
drift out of sync.

## Layout

| File | Role |
|---|---|
| `../../AudioMuse-AI.spec` | The single shared PyInstaller spec (repo root, so `SPECPATH == repo root`). |
| `config.py` | `PLATFORMS` table — the one place every per-platform difference lives. Imported by both the spec and `build.py`. |
| `build.py` | Orchestrator: `python scripts/standalone/build.py --platform {macos,linux,windows}`. Owns the shared steps (clean, version, arch, PyInstaller); dispatches packaging. |
| `platforms/{windows,macos,linux}.py` | `prepare(ctx)` (pre-PyInstaller) and `package(ctx)` (post) — only the genuinely platform-specific parts. |
| `platforms/_pgserver.py` | Shared embedded-PostgreSQL bundle guard (Linux x86_64 + Windows). |
| `assemble_model.py` | Shared CI model download/trim/verify (replaces three copies). |

PyInstaller **cannot cross-compile**: each OS is built natively on its own runner
(the existing CI matrix). `build.py --platform` must match the host; it is also
exported as `AUDIOMUSE_BUILD_TARGET` so the spec selects the right `PLATFORMS`
entry. A bare `pyinstaller AudioMuse-AI.spec` falls back to the host OS.

Only the **Windows** bundle is buildable locally (`.venv-windows`); macOS and Linux
build only in CI.

## What stays platform-specific (intentionally not merged)

Runtime code (`*/launcher.py`, `supervisor.py`, `paths.py`, `env.py`,
`db_backend.py`, `embedded_pg.py`, `control_*.py`, `reverse_log.py`),
`macos/entitlements.plist`, `macos/make_icns.sh`, the `*/assets` icons,
`linux/packaging/*` (nfpm template, `.desktop`, `.service`, post-install scripts),
and the vendored-binary build scripts (`*/vendor/...`) are irreducibly
platform-specific and remain where they are.

## Load-bearing rationale (do not regress)

**`strip=False` in both `EXE` and `COLLECT`.** Most native deps are manylinux
wheels whose shared libraries were already rewritten by auditwheel/patchelf
(mangled, hashed sonames such as `libscipy_openblas-*.so` and pgserver's
`libpq-*.so`, plus injected RPATHs). Running GNU `strip` over a patchelf-modified
ELF breaks it — pgserver's `initdb`/`psql` SIGSEGV at load, or scipy's OpenBLAS
fails with "ELF load command address/offset not page-aligned" (crashing the
Flask/worker import of sklearn → scipy). On Windows, stripping `.pyd` files can
break them too. Stripping would save a few hundred MB, but a corrupted bundle
does not start at all, so correctness wins.

**The pgserver tree restore + initdb smoke test** (`platforms/_pgserver.py`,
applied on Linux x86_64 and Windows). The spec pulls the embedded PostgreSQL in
via `collect_data_files('pgserver')`, but that helper EXCLUDES shared libraries,
so every loadable module under `pginstall/lib/postgresql` is dropped: `plpgsql`,
`vector` (pgvector), `dict_snowball` (initdb's post-bootstrap text-search setup
loads it — initdb fails without it), `pgoutput`, the encoding converters, etc.
Without them initdb cannot create the cluster and supervisor startup fails. The
guard overlays the **complete, pristine** `pginstall` tree from the installed
wheel onto the bundle (re-adding the missing `.so`/`.dll` modules while leaving
the vendored `unaccent`/`pg_trgm` contrib the spec grafted in), then runs a real
`initdb` into a temp dir and fails the build loudly if it errors — rather than
shipping a package that cannot start. The smoke test is the real guarantee, so
the copy method itself is unimportant. On Windows this also de-risks the
tray-launch fix, whose first run calls `initdb` via `_preinit_scram`.

**The Windows tray-launch fix needs `pgserver.postgres_server` frozen.**
`windows/db_backend.py::_patch_pgserver_pg_ctl()` does a function-level
`import pgserver.postgres_server` to swap `pg_ctl` so it starts the cluster
detached from any console with a generous timeout. That submodule is listed as an
explicit hidden import for Windows in `config.py`; without it the patch raises
`ImportError` in the frozen app.

**macOS archives with `ditto -c -k`, not `zip`.** The app is >4 GB and needs
ZIP64, which the legacy `zip` tool mishandles ("extra bytes" / corrupt archive);
`ditto` also preserves the bundle's symlinks and the ad-hoc signature. Staging
clones the app with `cp -cR` (APFS copy-on-write) so it is instant and free while
preserving signature + symlinks, falling back to `ditto` on a non-APFS volume.
`shutil.copytree` is avoided (loses symlinks/signature, slow). Windows uses the
stdlib `zipfile` with ZIP64 enabled.

**aarch64 bundles a from-source PostgreSQL** under `pgsql/` (no arm64 pgserver
wheel). It is added as PyInstaller *data*, which does not preserve the executable
bit, so `platforms/linux.py` restores `+x` on `pgsql/bin/*` after PyInstaller (the
server relocates via rpath; `embedded_pg` also sets `LD_LIBRARY_PATH`).
