# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the standalone Windows build.

Run from the repo root: ``pyinstaller windows/AudioMuse-AI.spec --noconfirm``.
Produces a one-dir bundle at ``dist/AudioMuse-AI/`` (the executable plus an
``_internal`` tree with Python, the libraries, the models and the embedded
PostgreSQL/Redis). ``windows/build.bat`` then packages that tree into an MSI
installer using WiX.

Builds for the architecture of the running Python (CI builds on ``windows-latest``,
which is x86_64/amd64).
"""

import glob
import os
import platform

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

arch = platform.machine().lower()  # 'amd64' on Windows, normalize to what the build expects
# pgserver ships a Windows wheel; if not available, the fallback embedded_pg is used.
USE_PGSERVER = True  # will be checked at runtime by db_backend

# ``SPECPATH`` is the directory containing this spec (``<repo>/windows``); the repo
# root is its parent.
ROOT = os.path.dirname(SPECPATH)

datas = [
    (os.path.join(ROOT, 'templates'), 'templates'),
    (os.path.join(ROOT, 'static'), 'static'),
    (os.path.join(ROOT, 'model'), 'model'),
    # Root-level data file the app loads relative to config.py's __file__
    (os.path.join(ROOT, 'mood_centroids_real_080_clap.json'), '.'),
]
if USE_PGSERVER:
    try:
        datas += collect_data_files('pgserver')
    except Exception:
        USE_PGSERVER = False
if not USE_PGSERVER:
    # Bundle the entire PostgreSQL install as opaque data under ``pgsql/``.
    datas += [(os.path.join(ROOT, 'windows/vendor/postgres', arch), 'pgsql')]
datas += collect_data_files('librosa')
datas += collect_data_files('resampy')
datas += collect_data_files('transformers', include_py_files=False)
datas += collect_data_files('flasgger')
datas += collect_data_files('wn')
datas += collect_data_files('langdetect')

binaries = [
    (os.path.join(ROOT, f'windows/vendor/redis/{arch}/redis-server.exe'), '.'),
]
binaries += collect_dynamic_libs('av')
binaries += collect_dynamic_libs('voyager')
binaries += collect_dynamic_libs('psycopg2')

if USE_PGSERVER:
    _pg_contrib = os.path.join(ROOT, 'windows/vendor/pg-contrib', arch)
    _pg_dst = 'pgserver/pginstall'
    for _f in glob.glob(os.path.join(_pg_contrib, 'extension', '*')):
        datas.append((_f, f'{_pg_dst}/share/extension'))
    for _f in glob.glob(os.path.join(_pg_contrib, 'tsearch_data', '*')):
        datas.append((_f, f'{_pg_dst}/share/tsearch_data'))
    for _f in glob.glob(os.path.join(_pg_contrib, 'lib', '*.dll')):
        binaries.append((_f, f'{_pg_dst}/lib'))

hiddenimports = [
    'app',
    'rq_worker',
    'rq_worker_high_priority',
    'rq_janitor',
    'restart_listener',
    'waitress',
    'flasgger',
    # Platform-agnostic helpers reused by windows.supervisor from the macos
    # package (no GUI deps).
    'macos.reverse_log',
]
hiddenimports += collect_submodules('windows')
hiddenimports += collect_submodules('tasks')
hiddenimports += collect_submodules('lyrics')
hiddenimports += collect_submodules('sklearn')

# rumps/AppKit are macOS-only.
excludes = ['rumps', 'AppKit', 'Foundation', 'objc']
if not USE_PGSERVER:
    excludes.append('pgserver')

a = Analysis(
    [os.path.join(ROOT, 'windows/launcher.py')],
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

# strip=False matches Linux/macOS: stripping can corrupt manylinux/patchelf-modified
# binaries, and on Windows stripping .pyd files can break them.
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
    console=True,  # Windows: console app so users see output and can Ctrl+C
    icon=os.path.join(ROOT, 'windows/assets/AudioMuse-AI.ico'),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name='AudioMuse-AI',
)
