# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the standalone Linux build.

Run from the repo root: ``pyinstaller linux/AudioMuse-AI.spec --noconfirm``.
Produces a one-dir bundle at ``dist/AudioMuse-AI/`` (the executable plus an
``_internal`` tree with Python, the libraries, the models and the embedded
PostgreSQL/Redis). ``linux/build.sh`` then turns that tree into a ``.deb`` and a
``.rpm``.

Builds for the architecture of the running Python (build once per arch:
x86_64 and aarch64), matching the macOS spec's per-arch approach -- onnxruntime /
PyAV / voyager wheels are arch-specific.
"""

import glob
import os
import platform

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

arch = platform.machine()  # 'x86_64' or 'aarch64'

# ``SPECPATH`` is the directory containing this spec (``<repo>/linux``); the repo
# root is its parent. Anchor every relative source path to the root so the build
# works regardless of CWD or PyInstaller version.
ROOT = os.path.dirname(SPECPATH)

datas = [
    (os.path.join(ROOT, 'templates'), 'templates'),
    (os.path.join(ROOT, 'static'), 'static'),
    (os.path.join(ROOT, 'model'), 'model'),
]
datas += collect_data_files('pgserver')
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

# pgserver bundles a minimal PostgreSQL 16.2 (only plpgsql + pgvector); the
# schema needs the ``unaccent`` and ``pg_trgm`` contrib extensions, which it
# lacks. We vendor them (compiled from PostgreSQL 16.2 source against pgserver's
# own headers/ABI -- see linux/vendor/pg-contrib/README.md) and graft them into
# the bundled pgserver tree. ``collect_data_files('pgserver')`` above lays down
# pginstall under ``pgserver/pginstall``; these land beside it.
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

a = Analysis(
    [os.path.join(ROOT, 'linux/launcher.py')],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[os.path.join(ROOT, 'macos/hooks')],  # hook-tasks.py is platform-agnostic
    hooksconfig={},
    runtime_hooks=[],
    excludes=['rumps', 'AppKit', 'Foundation', 'objc'],
    noarchive=False,
)

pyz = PYZ(a.pure)

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
