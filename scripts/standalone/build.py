# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Top-level standalone-build orchestrator (PyInstaller packaging).

Main entry point (``build.py --platform {windows,macos,linux}``) that runs
PyInstaller against the selected platform launcher and drives the packaging
steps, delegating platform-specific staging and post-processing to the
matching module under ``platforms/``. Reads the app version from ``config.py``
and produces the frozen bundle for one target.

Main Features:
* Resolves the build context (target, arch, version, dist/bundle paths) and
  dispatches ``prepare``/packaging to the right ``platforms`` module.
* Runs PyInstaller and sanitizes the version string for package naming.
"""

import argparse
import os
import platform as _platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import config
from platforms import linux as _linux
from platforms import macos as _macos
from platforms import windows as _windows

ROOT = Path(__file__).resolve().parents[2]
DISPATCH = {"windows": _windows, "macos": _macos, "linux": _linux}


@dataclass
class Ctx:
    target: str
    arch: str
    version: str
    root: Path
    dist_dir: Path
    bundle_dir: Path
    app_path: Path
    use_pgserver: bool
    cfg: dict


def sanitize_version(raw):
    v = raw or "0.0.0"
    if v.startswith("v"):
        v = v[1:]
    v = re.sub(r"[^A-Za-z0-9.+~-]", "-", v)
    v = re.sub(r"-{2,}", "-", v).strip("-")
    return v or "0.0.0"


def _expected_output(ctx):
    if ctx.cfg["bundle"]:
        return ctx.app_path
    name = "AudioMuse-AI.exe" if ctx.target == "windows" else "AudioMuse-AI"
    return ctx.bundle_dir / name


def main():
    parser = argparse.ArgumentParser(description="Build the AudioMuse-AI standalone bundle.")
    parser.add_argument("--platform", required=True, choices=sorted(config.PLATFORMS))
    parser.add_argument("--arch", default=None)
    args = parser.parse_args()

    os.chdir(ROOT)
    target = args.platform
    cfg = config.PLATFORMS[target]
    arch = args.arch or config.normalize_arch(_platform.machine(), target)

    version = sanitize_version(config.read_app_version(ROOT))

    use_pgserver = config.use_pgserver(cfg["use_pgserver"], arch)

    dist_dir = ROOT / "dist"
    ctx = Ctx(
        target=target,
        arch=arch,
        version=version,
        root=ROOT,
        dist_dir=dist_dir,
        bundle_dir=dist_dir / "AudioMuse-AI",
        app_path=dist_dir / "AudioMuse-AI.app",
        use_pgserver=use_pgserver,
        cfg=cfg,
    )
    module = DISPATCH[target]

    print(f"==> Package version: {version}")
    print(f"==> Platform: {target}  Architecture: {arch}  pgserver: {use_pgserver}")

    print("==> Cleaning previous build")
    shutil.rmtree(ROOT / "build", ignore_errors=True)
    shutil.rmtree(dist_dir, ignore_errors=True)

    module.prepare(ctx)

    print("==> Running PyInstaller")
    env = {**os.environ, "AUDIOMUSE_BUILD_TARGET": target}
    subprocess.run(
        [sys.executable, "-m", "PyInstaller", "AudioMuse-AI.spec", "--noconfirm"],
        check=True,
        env=env,
    )

    out = _expected_output(ctx)
    if not out.exists():
        raise SystemExit(f"::error::PyInstaller did not produce {out}")

    artifacts = module.package(ctx)

    print("==> Done")
    for art in artifacts or []:
        print(f"    {art}")


if __name__ == "__main__":
    main()
