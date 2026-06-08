import ast
import os
import sys


def read_app_version(root):
    """Return APP_VERSION from the app's config.py (leading 'v' stripped).

    Parsed statically with ast -- config.py is never imported or executed, so
    there are no import side effects and no dependency on __file__ or the
    environment. build.py (deb/rpm version + banner) and the shared spec (macOS
    CFBundleShortVersionString) both call this, so every platform stamps the same
    manually-maintained version from config.py and no CI/tag value is used.
    """
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
        "launcher": "windows/launcher.py",
        "vendor_dir": "windows/vendor",
        "redis_bin": "redis-server.exe",
        "pg_contrib_glob": "*.dll",
        "initdb_bin": "initdb.exe",
        "use_pgserver": "always",
        "console": True,
        "exe_icon": "windows/assets/AudioMuse-AI.ico",
        "extra_datas": [("windows/assets/AudioMuse-AI.ico", "assets")],
        "extra_hiddenimports": ["macos.reverse_log", "pgserver.postgres_server"],
        "collect_submodules": ["windows", "pystray", "PIL"],
        "excludes_base": ["rumps", "AppKit", "Foundation", "objc"],
        "bundle": None,
    },
    "macos": {
        "launcher": "macos/launcher.py",
        "vendor_dir": "macos/vendor",
        "redis_bin": "redis-server",
        "pg_contrib_glob": "*.dylib",
        "initdb_bin": "initdb",
        "use_pgserver": "always",
        "console": False,
        "exe_icon": None,
        "extra_datas": [("macos/assets", "assets")],
        "extra_hiddenimports": ["numeric_bootstrap", "rumps"],
        "collect_submodules": ["macos"],
        "excludes_base": [],
        "bundle": {
            "name": "AudioMuse-AI.app",
            "icon": "macos/assets/AudioMuse-AI.icns",
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
        "launcher": "linux/launcher.py",
        "vendor_dir": "linux/vendor",
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
    """Return the build target, preferring ``AUDIOMUSE_BUILD_TARGET`` over the host.

    ``build.py`` exports ``AUDIOMUSE_BUILD_TARGET`` before invoking PyInstaller, so
    the spec reads it from here. A bare ``pyinstaller AudioMuse-AI.spec`` (no
    orchestrator) still works for a developer by falling back to the host's
    ``sys.platform`` -- PyInstaller cannot cross-compile, so the host OS is always
    the correct target in that case.
    """
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
    """Normalize ``platform.machine()`` to the vendor-dir arch name for ``target``.

    Windows reports ``AMD64``/``ARM64`` (and historically ``x86_64``); the vendored
    inputs live under ``amd64``/``arm64``, so lowercase and fold ``x86_64`` to
    ``amd64``. macOS (``arm64``) and Linux (``x86_64``/``aarch64``) already match
    their vendor dirs verbatim.
    """
    if target == "windows":
        arch = machine.lower()
        return "amd64" if arch == "x86_64" else arch
    return machine


def use_pgserver(policy, arch):
    """Resolve the embedded-PostgreSQL sourcing policy to a boolean.

    ``always`` -> the pgserver wheel (Windows/macOS, with a runtime fallback the
    spec applies on its own). ``arch`` -> the wheel only where it exists (Linux
    x86_64/amd64); aarch64 has no wheel and bundles a from-source tree instead.
    """
    if policy == "always":
        return True
    if policy == "arch":
        return arch in ("x86_64", "amd64")
    return False
