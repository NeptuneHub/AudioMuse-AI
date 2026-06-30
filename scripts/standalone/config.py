import ast
import os
import sys


def read_app_version(root):
    path = os.path.join(str(root), "config.py")
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=path)
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "APP_VERSION":
                    value = str(node.value.value)
                    return value[1:] if value.startswith("v") else value
    return "0.0.0"


PLATFORMS = {
    "windows": {
        "launcher": "native-build/windows/launcher.py",
        "vendor_dir": "native-build/windows/vendor",
        "redis_bin": "redis-server.exe",
        "pg_contrib_glob": "*.dll",
        "initdb_bin": "initdb.exe",
        "use_pgserver": "always",
        "console": True,
        "exe_icon": "native-build/windows/assets/AudioMuse-AI.ico",
        "extra_datas": [("native-build/windows/assets/AudioMuse-AI.ico", "assets")],
        "extra_hiddenimports": ["macos.reverse_log", "pgserver.postgres_server"],
        "collect_submodules": ["windows", "pystray", "PIL"],
        "excludes_base": ["rumps", "AppKit", "Foundation", "objc"],
        "bundle": None,
    },
    "macos": {
        "launcher": "native-build/macos/launcher.py",
        "vendor_dir": "native-build/macos/vendor",
        "redis_bin": "redis-server",
        "pg_contrib_glob": "*.dylib",
        "initdb_bin": "initdb",
        "use_pgserver": "always",
        "console": False,
        "exe_icon": None,
        "extra_datas": [("native-build/macos/assets", "assets")],
        "extra_hiddenimports": ["numeric_bootstrap", "rumps"],
        "collect_submodules": ["macos"],
        "excludes_base": [],
        "bundle": {
            "name": "AudioMuse-AI.app",
            "icon": "native-build/macos/assets/AudioMuse-AI.icns",
            "bundle_identifier": "ai.audiomuse.standalone",
            "info_plist": {
                "LSUIElement": True,
                "NSHighResolutionCapable": True,
                "CFBundleName": "AudioMuse-AI",
                "CFBundleDisplayName": "AudioMuse-AI",
            },
        },
    },
    "linux": {
        "launcher": "native-build/linux/launcher.py",
        "vendor_dir": "native-build/linux/vendor",
        "redis_bin": "redis-server",
        "pg_contrib_glob": "*.so",
        "initdb_bin": "initdb",
        "use_pgserver": "arch",
        "console": True,
        "exe_icon": None,
        "extra_datas": [],
        "extra_hiddenimports": ["macos.control_ipc", "macos.reverse_log"],
        "collect_submodules": ["linux"],
        "excludes_base": ["rumps", "AppKit", "Foundation", "objc"],
        "bundle": None,
    },
}

_SYS_PLATFORM_TO_TARGET = {
    "win32": "windows",
    "darwin": "macos",
    "linux": "linux",
}


def resolve_target(env_value):
    if env_value:
        target = env_value.strip().lower()
        if target in PLATFORMS:
            return target
        raise ValueError(
            f"Unknown AUDIOMUSE_BUILD_TARGET {env_value!r}; expected one of {sorted(PLATFORMS)}"
        )
    target = _SYS_PLATFORM_TO_TARGET.get(sys.platform)
    if target is None:
        raise ValueError(f"Unsupported host platform {sys.platform!r} for a native build")
    return target


def normalize_arch(machine, target):
    if target == "windows":
        arch = machine.lower()
        return "amd64" if arch == "x86_64" else arch
    return machine


def use_pgserver(policy, arch):
    if policy == "always":
        return True
    if policy == "arch":
        return arch in ("x86_64", "amd64")
    return False


_WINDOWS_OMP_DLLS = {"amd64": "libomp140.x86_64.dll", "arm64": "libomp140.aarch64.dll"}


def windows_omp_dll(arch):
    try:
        return _WINDOWS_OMP_DLLS[arch]
    except KeyError:
        raise ValueError(
            f"No vendored libomp for Windows arch {arch!r}; expected one of {sorted(_WINDOWS_OMP_DLLS)}"
        ) from None
