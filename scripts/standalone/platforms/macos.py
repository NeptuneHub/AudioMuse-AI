"""macOS packaging: generate icons, ad-hoc codesign, stage + ditto into a zip.

Mirrors the old ``macos/build.sh`` (clean + PyInstaller live in ``build.py``).

Two non-obvious choices preserved from the original, do not regress:

* The bundle is archived with ``ditto -c -k``, NOT ``zip``/``zipfile``. The app is
  >4 GB and needs ZIP64, which the legacy ``zip`` tool mishandles ("extra bytes" /
  corrupt archive), and ditto also preserves the bundle's symlinks and the ad-hoc
  signature.
* Staging copies the app with ``cp -cR`` (APFS copy-on-write clone) so staging is
  instant and costs no extra disk while preserving the signature and symlinks,
  falling back to ``ditto`` on a non-APFS volume. ``shutil.copytree`` is avoided
  (loses symlinks/signature, slow on >4 GB).

Only verifiable in CI (macos-14, arm64) -- there is no local macOS build box.
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
    subprocess.run(["bash", "macos/make_icns.sh"], check=True, cwd=str(ctx.root))


def _sign_nested(app, entitlements):
    print("==> Ad-hoc signing nested binaries")
    contents = os.path.join(str(app), "Contents")
    for root, _dirs, files in os.walk(contents):
        for name in files:
            if name.endswith(_SIGN_SUFFIXES) or name in _SIGN_NAMES:
                subprocess.run(
                    ["codesign", "--force", "--timestamp=none", "--sign", "-",
                     "--entitlements", entitlements, os.path.join(root, name)],
                    stderr=subprocess.DEVNULL,
                )


def package(ctx):
    app = ctx.app_path
    entitlements = str(ctx.root / "macos" / "entitlements.plist")

    _sign_nested(app, entitlements)

    print("==> Ad-hoc signing the bundle")
    subprocess.run(
        ["codesign", "--force", "--deep", "--timestamp=none", "--sign", "-",
         "--entitlements", entitlements, str(app)],
        check=True,
    )

    print("==> Verifying signature (rejection by spctl is expected for an unsigned app)")
    subprocess.run(["codesign", "--verify", "--verbose", str(app)])
    subprocess.run(["spctl", "-a", "-vv", str(app)])

    out = ctx.dist_dir / f"AudioMuse-AI-{ctx.arch}-macos.zip"
    print(f"==> Packaging {out.name} (AudioMuse-AI.app + readme.md)")
    stage = ctx.dist_dir / "_pkg"
    subprocess.run(["rm", "-rf", str(stage)], check=True)
    stage.mkdir(parents=True)
    staged_app = stage / "AudioMuse-AI.app"
    if subprocess.run(["cp", "-cR", str(app), str(staged_app)], stderr=subprocess.DEVNULL).returncode != 0:
        subprocess.run(["ditto", str(app), str(staged_app)], check=True)
    (stage / "readme.md").write_text(_README)
    if out.exists():
        out.unlink()
    subprocess.run(["ditto", "-c", "-k", str(stage), str(out)], check=True)
    subprocess.run(["rm", "-rf", str(stage)], check=True)
    print(f"==> Done: {out} (expands to AudioMuse-AI.app + readme.md)")
    return [out]
