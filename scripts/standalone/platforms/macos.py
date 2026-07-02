# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""macOS packaging steps for the standalone build.

Platform module invoked by ``build.py`` to stage the macOS ``.app``: it
generates the ``.icns`` icon, ad-hoc code-signs the nested binaries and dylibs,
and assembles the distributable with the unsigned-app authorization README.
The Linux/Windows modules are the platform-specific siblings.

Main Features:
* Generates icons and ad-hoc signs bundled binaries (redis, postgres, dylibs).
* Ships a README instructing users to clear the quarantine xattr on the app.
"""

import os
import subprocess

_SIGN_SUFFIXES = (".dylib", ".so")
_SIGN_NAMES = {"redis-server", "postgres", "initdb", "pg_ctl", "psql", "pg_isready"}

_README = """This AudioMuse-AI app is not signed to avoid Apple recurrent subscription cost. To have it working you need to:
- Move AudioMuse-AI.app in /Applications
- Open a terminal and run this command to authorize:
xattr -dr com.apple.quarantine /Applications/AudioMuse-AI.app

After this you can just open it like any other application.
"""


def prepare(ctx):
    print("==> Generating icons from screenshot/audiomuseai.png")
    subprocess.run(["bash", "native-build/macos/make_icns.sh"], check=True, cwd=str(ctx.root))


def _retarget_shared_libomp(app):
    # sklearn and numkong both bundle libomp.dylib; PyInstaller rewrites every
    # extension's rpath to the Frameworks root and points the shared top-level
    # libomp.dylib symlink at whichever copy it collected first (sklearn's,
    # which is older). numkong's kernels need ___kmpc_dispatch_deinit, absent
    # from sklearn's copy, so `import numkong` fails and the i8 IVF path falls
    # back to NumPy. libomp only ever adds symbols, so the newer numkong copy
    # is safe for sklearn too - retarget the shared symlink at it.
    contents = os.path.join(str(app), "Contents")
    target = os.path.join("numkong", "__dot__dylibs", "libomp.dylib")
    if not os.path.exists(os.path.join(contents, "Frameworks", target)):
        print("[WARN] numkong libomp.dylib not found; leaving shared libomp symlink alone.")
        return
    for parent in ("Frameworks", "Resources"):
        link = os.path.join(contents, parent, "libomp.dylib")
        if os.path.islink(link):
            os.unlink(link)
            os.symlink(target, link)
            print(f"==> Retargeted {parent}/libomp.dylib -> {target}")


def _sign_nested(app, entitlements):
    print("==> Ad-hoc signing nested binaries")
    contents = os.path.join(str(app), "Contents")
    for root, _dirs, files in os.walk(contents):
        for name in files:
            if name.endswith(_SIGN_SUFFIXES) or name in _SIGN_NAMES:
                subprocess.run(
                    [
                        "codesign",
                        "--force",
                        "--timestamp=none",
                        "--sign",
                        "-",
                        "--entitlements",
                        entitlements,
                        os.path.join(root, name),
                    ],
                    stderr=subprocess.DEVNULL,
                )


def package(ctx):
    app = ctx.app_path
    entitlements = str(ctx.root / "native-build" / "macos" / "entitlements.plist")

    _retarget_shared_libomp(app)
    _sign_nested(app, entitlements)

    print("==> Ad-hoc signing the bundle")
    subprocess.run(
        [
            "codesign",
            "--force",
            "--deep",
            "--timestamp=none",
            "--sign",
            "-",
            "--entitlements",
            entitlements,
            str(app),
        ],
        check=True,
    )

    print("==> Verifying signature (rejection by spctl is expected for an unsigned app)")
    subprocess.run(["codesign", "--verify", "--verbose", str(app)])
    subprocess.run(["spctl", "-a", "-vv", str(app)])

    stage = ctx.dist_dir / "_pkg"
    subprocess.run(["rm", "-rf", str(stage)], check=True)
    stage.mkdir(parents=True)
    staged_app = stage / "AudioMuse-AI.app"
    if (
        subprocess.run(
            ["cp", "-cR", str(app), str(staged_app)], stderr=subprocess.DEVNULL
        ).returncode
        != 0
    ):
        subprocess.run(["ditto", str(app), str(staged_app)], check=True)
    (stage / "readme.md").write_text(_README)

    out = ctx.dist_dir / f"AudioMuse-AI-{ctx.arch}-macos.zip"
    if out.exists():
        out.unlink()
    subprocess.run(["ditto", "-c", "-k", str(stage), str(out)], check=True)
    subprocess.run(["rm", "-rf", str(stage)], check=True)
    print(f"==> Done: {out} (expands to AudioMuse-AI.app + readme.md)")
    return [out]
