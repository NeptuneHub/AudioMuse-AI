# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the standalone Linux build.

Run from the repo root: ``pyinstaller linux/AudioMuse-AI.spec --noconfirm``.
Produces a one-dir bundle at ``dist/AudioMuse-AI/`` (the executable plus an
``_internal`` tree with Python, the libraries, the models and the embedded
PostgreSQL/Redis). ``linux/build.sh`` then turns that tree into a ``.deb`` and a
``.rpm``.

Builds for the architecture of the running Python (CI builds x86_64 and
aarch64). Embedded PostgreSQL is bundled per arch: the pgserver wheel on x86_64,
or a from-source PostgreSQL tree under ``pgsql/`` on aarch64 (pgserver has no
arm64 wheel) -- see ``USE_PGSERVER`` below.
"""

import glob
import os
import platform

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

arch = platform.machine()  # 'x86_64' or 'aarch64'
# x86_64 embeds PostgreSQL via the pgserver wheel; aarch64 has no pgserver wheel
# and instead bundles a from-source PostgreSQL tree (see
# linux/vendor/postgres/build-postgres.sh).
USE_PGSERVER = arch in ('x86_64', 'amd64')

# ``SPECPATH`` is the directory containing this spec (``<repo>/linux``); the repo
# root is its parent. Anchor every relative source path to the root so the build
# works regardless of CWD or PyInstaller version.
ROOT = os.path.dirname(SPECPATH)

datas = [
    (os.path.join(ROOT, 'templates'), 'templates'),
    (os.path.join(ROOT, 'static'), 'static'),
    (os.path.join(ROOT, 'model'), 'model'),
]
if USE_PGSERVER:
    datas += collect_data_files('pgserver')
else:
    # Bundle the entire from-source PostgreSQL install as opaque data under
    # ``pgsql/`` (preserving the bin/lib/share layout so the relocatable server
    # finds its support files). build.sh chmod +x's pgsql/bin/* afterwards
    # (datas don't carry the exec bit).
    datas += [(os.path.join(ROOT, 'linux/vendor/postgres', arch), 'pgsql')]
datas += collect_data_files('librosa')
datas += collect_data_files('resampy')
datas += collect_data_files('transformers', include_py_files=False)
datas += collect_data_files('flasgger')
datas += collect_data_files('wn')
datas += collect_data_files('langdetect')

binaries = [
    (os.path.join(ROOT, f'linux/vendor/redis/{arch}/redis-server'), '.'),
]
binaries += collect_dynamic_libs('av')
binaries += collect_dynamic_libs('voyager')
binaries += collect_dynamic_libs('psycopg2')

# pgserver (x86_64 only) bundles a minimal PostgreSQL (only plpgsql + pgvector);
# the schema needs the ``unaccent`` and ``pg_trgm`` contrib extensions, which it
# lacks. We vendor them (compiled against pgserver's own headers/ABI -- see
# linux/vendor/pg-contrib/README.md) and graft them into the bundled pgserver
# tree. (On aarch64 the from-source PostgreSQL already includes these contrib
# modules in its own tree, bundled wholesale as ``pgsql/`` above.)
if USE_PGSERVER:
    _pg_contrib = os.path.join(ROOT, 'linux/vendor/pg-contrib', arch)
    _pg_dst = 'pgserver/pginstall'
    for _f in glob.glob(os.path.join(_pg_contrib, 'extension', '*')):
        datas.append((_f, f'{_pg_dst}/share/postgresql/extension'))
    for _f in glob.glob(os.path.join(_pg_contrib, 'tsearch_data', '*')):
        datas.append((_f, f'{_pg_dst}/share/postgresql/tsearch_data'))
    for _f in glob.glob(os.path.join(_pg_contrib, 'lib', '*.so')):
        binaries.append((_f, f'{_pg_dst}/lib/postgresql'))

hiddenimports = [
    'app',
    'rq_worker',
    'rq_worker_high_priority',
    'rq_janitor',
    'restart_listener',
    'waitress',
    # Platform-agnostic helpers reused by linux.supervisor from the macos
    # package (no GUI deps). Listed explicitly so the frozen build includes
    # them without pulling in the rumps-based macos.launcher.
    'macos.control_ipc',
    'macos.reverse_log',
]
hiddenimports += collect_submodules('linux')
hiddenimports += collect_submodules('tasks')
hiddenimports += collect_submodules('lyrics')
hiddenimports += collect_submodules('sklearn')

# rumps/AppKit are macOS-only. On aarch64 also exclude pgserver: it isn't
# installed (no arm64 wheel) and the shared database.py only references it
# lazily inside functions the aarch64 build never calls.
excludes = ['rumps', 'AppKit', 'Foundation', 'objc']
if not USE_PGSERVER:
    excludes.append('pgserver')

a = Analysis(
    [os.path.join(ROOT, 'linux/launcher.py')],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[os.path.join(ROOT, 'macos/hooks')],  # hook-tasks.py is platform-agnostic
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(a.pure)

# strip=False: stripping is DISABLED on Linux because it corrupts the bundle.
# Most of our native deps are manylinux wheels whose shared libraries were
# already rewritten by auditwheel/patchelf (mangled, hashed sonames such as
# ``libscipy_openblas-b75cc656.so`` and pgserver's ``libpq-084d956f.so.5.16``,
# plus injected RPATHs). Running GNU ``strip`` over a patchelf-modified ELF
# breaks it -- the result either SIGSEGVs at load (pgserver's initdb/psql) or
# fails to load with "ELF load command address/offset not page-aligned" (scipy's
# OpenBLAS, which crashes the Flask/worker import of sklearn -> scipy). Stripping
# would save a few hundred MB, but a corrupted bundle does not start at all, so
# correctness wins. (The macOS spec also keeps stripping off.)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='AudioMuse-AI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name='AudioMuse-AI',
)
