# -*- mode: python ; coding: utf-8 -*-
import glob
import importlib.util
import os
import platform

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

ROOT = SPECPATH

_cfg_path = os.path.join(ROOT, "scripts", "standalone", "config.py")
_cfg_spec = importlib.util.spec_from_file_location("_amai_build_config", _cfg_path)
_cfg = importlib.util.module_from_spec(_cfg_spec)
_cfg_spec.loader.exec_module(_cfg)

target = _cfg.resolve_target(os.environ.get("AUDIOMUSE_BUILD_TARGET"))
cfg = _cfg.PLATFORMS[target]
arch = _cfg.normalize_arch(platform.machine(), target)
USE_PGSERVER = _cfg.use_pgserver(cfg["use_pgserver"], arch)

_app_ver = _cfg.read_app_version(ROOT)
if cfg["bundle"]:
    cfg["bundle"]["info_plist"]["CFBundleShortVersionString"] = _app_ver or "0.0.0"

datas = [
    (os.path.join(ROOT, "templates"), "templates"),
    (os.path.join(ROOT, "static"), "static"),
    (os.path.join(ROOT, "model"), "model"),
    (os.path.join(ROOT, "mood_centroids_real_080_clap.json"), "."),
]
for _src, _dst in cfg["extra_datas"]:
    datas.append((os.path.join(ROOT, _src), _dst))

if USE_PGSERVER:
    try:
        datas += collect_data_files("pgserver")
    except Exception:
        USE_PGSERVER = False
if not USE_PGSERVER:
    datas += [(os.path.join(ROOT, cfg["vendor_dir"], "postgres", arch), "pgsql")]

for _pkg in ("librosa", "resampy", "flasgger", "wn", "langdetect"):
    datas += collect_data_files(_pkg)
datas += collect_data_files("transformers", include_py_files=False)

binaries = [
    (os.path.join(ROOT, cfg["vendor_dir"], "redis", arch, cfg["redis_bin"]), "."),
]
for _pkg in ("av", "voyager", "psycopg2"):
    binaries += collect_dynamic_libs(_pkg)

if USE_PGSERVER:
    _pg_contrib = os.path.join(ROOT, cfg["vendor_dir"], "pg-contrib", arch)
    _pg_dst = "pgserver/pginstall"
    for _f in glob.glob(os.path.join(_pg_contrib, "extension", "*")):
        datas.append((_f, f"{_pg_dst}/share/postgresql/extension"))
    for _f in glob.glob(os.path.join(_pg_contrib, "tsearch_data", "*")):
        datas.append((_f, f"{_pg_dst}/share/postgresql/tsearch_data"))
    for _f in glob.glob(os.path.join(_pg_contrib, "lib", cfg["pg_contrib_glob"])):
        binaries.append((_f, f"{_pg_dst}/lib/postgresql"))

hiddenimports = [
    "app",
    "rq_worker",
    "rq_worker_high_priority",
    "rq_janitor",
    "restart_listener",
    "waitress",
    "flasgger",
]
hiddenimports += cfg["extra_hiddenimports"]
for _mod in ("tasks", "lyrics", "sklearn", *cfg["collect_submodules"]):
    hiddenimports += collect_submodules(_mod)
hiddenimports = list(dict.fromkeys(hiddenimports))

excludes = list(cfg["excludes_base"])
if not USE_PGSERVER:
    excludes.append("pgserver")
    hiddenimports = [h for h in hiddenimports if not h.startswith("pgserver")]

a = Analysis(
    [os.path.join(ROOT, cfg["launcher"])],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[os.path.join(ROOT, "macos/hooks")],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AudioMuse-AI",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=cfg["console"],
    **({"icon": os.path.join(ROOT, cfg["exe_icon"])} if cfg["exe_icon"] else {}),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="AudioMuse-AI",
)

if cfg["bundle"]:
    _b = cfg["bundle"]
    app = BUNDLE(
        coll,
        name=_b["name"],
        icon=os.path.join(ROOT, _b["icon"]),
        bundle_identifier=_b["bundle_identifier"],
        info_plist=_b["info_plist"],
    )
