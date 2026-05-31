# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the standalone macOS app.

Run from the repo root: ``pyinstaller macos/AudioMuse-AI.spec --noconfirm``.
Builds for the architecture of the running Python (build once on Apple Silicon,
once on Intel -- universal2 is avoided because onnxruntime/PyAV/voyager wheels
are not reliably universal2).
"""

import glob
import os
import platform

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

arch = platform.machine()

# ``SPECPATH`` is the directory containing this spec (``<repo>/macos``); the repo
# root is its parent. Anchor every relative source path to the root so the build
# works regardless of CWD or PyInstaller version (PyInstaller 6.x resolves
# relative spec paths against SPECPATH, older versions against the CWD).
ROOT = os.path.dirname(SPECPATH)

datas = [
    (os.path.join(ROOT, 'templates'), 'templates'),
    (os.path.join(ROOT, 'static'), 'static'),
    (os.path.join(ROOT, 'model'), 'model'),
    (os.path.join(ROOT, 'macos/assets'), 'assets'),
]
datas += collect_data_files('pgserver')
datas += collect_data_files('librosa')
datas += collect_data_files('resampy')
datas += collect_data_files('transformers', include_py_files=False)
datas += collect_data_files('flasgger')
datas += collect_data_files('wn')
datas += collect_data_files('langdetect')

binaries = [
    (os.path.join(ROOT, f'macos/vendor/redis/{arch}/redis-server'), '.'),
]
binaries += collect_dynamic_libs('av')
binaries += collect_dynamic_libs('voyager')
binaries += collect_dynamic_libs('psycopg2')

# pgserver bundles a minimal PostgreSQL 16.2 (only plpgsql + pgvector); the
# schema needs the ``unaccent`` and ``pg_trgm`` contrib extensions, which it
# lacks. We vendor them (compiled from PostgreSQL 16.2 source against pgserver's
# own headers/ABI -- see macos/vendor/pg-contrib/README.md) and graft them into
# the bundled pgserver tree. ``collect_data_files('pgserver')`` above lays down
# pginstall under ``pgserver/pginstall``; these land beside it.
_pg_contrib = os.path.join(ROOT, 'macos/vendor/pg-contrib', arch)
_pg_dst = 'pgserver/pginstall'
for _f in glob.glob(os.path.join(_pg_contrib, 'extension', '*')):
    datas.append((_f, f'{_pg_dst}/share/postgresql/extension'))
for _f in glob.glob(os.path.join(_pg_contrib, 'tsearch_data', '*')):
    datas.append((_f, f'{_pg_dst}/share/postgresql/tsearch_data'))
for _f in glob.glob(os.path.join(_pg_contrib, 'lib', '*.dylib')):
    binaries.append((_f, f'{_pg_dst}/lib/postgresql'))

hiddenimports = [
    'app',
    'rq_worker',
    'rq_worker_high_priority',
    'rq_janitor',
    'restart_listener',
    'waitress',
    'rumps',
]
hiddenimports += collect_submodules('tasks')
hiddenimports += collect_submodules('lyrics')
hiddenimports += collect_submodules('macos')
hiddenimports += collect_submodules('sklearn')

a = Analysis(
    [os.path.join(ROOT, 'macos/launcher.py')],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[os.path.join(ROOT, 'macos/hooks')],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
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
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name='AudioMuse-AI',
)

app = BUNDLE(
    coll,
    name='AudioMuse-AI.app',
    icon=os.path.join(ROOT, 'macos/assets/AudioMuse-AI.icns'),
    bundle_identifier='ai.audiomuse.standalone',
    info_plist={
        'LSUIElement': True,
        'NSHighResolutionCapable': True,
        'CFBundleName': 'AudioMuse-AI',
        'CFBundleDisplayName': 'AudioMuse-AI',
        'CFBundleShortVersionString': '1.0.0',
    },
)
