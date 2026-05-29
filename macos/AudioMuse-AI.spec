# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the standalone macOS app.

Run from the repo root: ``pyinstaller macos/AudioMuse-AI.spec --noconfirm``.
Builds for the architecture of the running Python (build once on Apple Silicon,
once on Intel -- universal2 is avoided because onnxruntime/PyAV/voyager wheels
are not reliably universal2).
"""

import platform

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

arch = platform.machine()

datas = [
    ('templates', 'templates'),
    ('static', 'static'),
    ('model', 'model'),
    ('macos/assets', 'assets'),
]
datas += collect_data_files('pgserver')
datas += collect_data_files('librosa')
datas += collect_data_files('resampy')
datas += collect_data_files('transformers', include_py_files=False)
datas += collect_data_files('flasgger')
datas += collect_data_files('wn')
datas += collect_data_files('langdetect')

binaries = [
    (f'macos/vendor/redis/{arch}/redis-server', '.'),
]
binaries += collect_dynamic_libs('av')
binaries += collect_dynamic_libs('voyager')
binaries += collect_dynamic_libs('psycopg2')

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
    ['macos/launcher.py'],
    pathex=['.'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=['macos/hooks'],
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
    icon='macos/assets/AudioMuse-AI.icns',
    bundle_identifier='ai.audiomuse.standalone',
    info_plist={
        'LSUIElement': True,
        'NSHighResolutionCapable': True,
        'CFBundleName': 'AudioMuse-AI',
        'CFBundleDisplayName': 'AudioMuse-AI',
        'CFBundleShortVersionString': '1.0.0',
    },
)
