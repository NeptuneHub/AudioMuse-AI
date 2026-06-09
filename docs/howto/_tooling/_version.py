"""Resolve the application version from config.py without importing it.

config.py pulls in heavy runtime dependencies, so the version is read by
parsing the source with the ast module instead (same approach the standalone
build uses in scripts/standalone/config.py).
"""
import ast
import os
from pathlib import Path

# docs/howto/_tooling/_version.py -> repo root is three parents up.
REPO_ROOT = Path(__file__).resolve().parents[3]


def read_app_version(repo_root=REPO_ROOT):
    """Return APP_VERSION exactly as written in config.py, e.g. 'v2.1.4'."""
    cfg = os.path.join(str(repo_root), "config.py")
    with open(cfg, "r", encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "APP_VERSION":
                    if isinstance(node.value, ast.Constant):
                        return str(node.value.value)
    raise RuntimeError("APP_VERSION not found in config.py")


def folder_version(version):
    """Strip a leading 'v' so 'v2.1.4' -> '2.1.4' (used for the folder name)."""
    return version[1:] if version[:1] in ("v", "V") else version


def display_version(version):
    """Normalise to the 'vX.Y.Z' form shown in the document."""
    return version if version[:1] in ("v", "V") else "v" + version


def resolve(version=None, repo_root=REPO_ROOT):
    """Return (display, folder) for an explicit tag or the value in config.py."""
    raw = version or read_app_version(repo_root)
    return display_version(raw), folder_version(raw)


if __name__ == "__main__":
    disp, folder = resolve()
    print(disp, folder)
