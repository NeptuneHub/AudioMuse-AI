# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Shared pgserver bundle staging and verification for standalone builds.

Helper used by the Linux/macOS/Windows packaging modules to restore the
pgserver ``pginstall`` tree into the frozen bundle (PyInstaller strips parts of
it) and smoke-test that the bundled embedded PostgreSQL is runnable before the
build is accepted.

Main Features:
* Merges the source ``pginstall`` and ``pgserver.libs`` trees into the bundle.
* ``verify_pgserver_bundle`` runs the bundled ``initdb`` to create a throwaway
  cluster (and checks ``initdb --version``); ``strict=False`` downgrades a
  failure to a warning (Windows best-effort).
"""

import os
import shutil
import subprocess
import tempfile

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _merge_tree(src, dst):
    for root, _dirs, files in os.walk(src):
        rel = os.path.relpath(root, src)
        target_root = dst if rel == "." else os.path.join(dst, rel)
        os.makedirs(target_root, exist_ok=True)
        for name in files:
            s = os.path.join(root, name)
            d = os.path.join(target_root, name)
            if os.path.islink(d) or os.path.exists(d):
                os.remove(d)
            shutil.copy2(s, d, follow_symlinks=False)


def _restore_and_smoke_test(ctx, pgserver):
    pg_pkg = os.path.dirname(os.path.abspath(pgserver.__file__))
    pg_site = os.path.dirname(pg_pkg)
    src_pginstall = os.path.join(pg_pkg, "pginstall")
    dst_pginstall = os.path.join(str(ctx.bundle_dir), "_internal", "pgserver", "pginstall")

    if not os.path.isdir(os.path.join(src_pginstall, "bin")) or not os.path.isdir(dst_pginstall):
        raise SystemExit(
            f"::error::Cannot locate pgserver pginstall to restore "
            f"(src={src_pginstall} dst={dst_pginstall})"
        )

    _merge_tree(src_pginstall, dst_pginstall)
    src_libs = os.path.join(pg_site, "pgserver.libs")
    dst_libs = os.path.join(str(ctx.bundle_dir), "_internal", "pgserver.libs")
    if os.path.isdir(src_libs) and os.path.isdir(dst_libs):
        _merge_tree(src_libs, dst_libs)
    print(f"==> Restored complete unstripped pgserver tree into {dst_pginstall}")

    initdb = os.path.join(dst_pginstall, "bin", ctx.cfg["initdb_bin"])
    win_flags = {"creationflags": _NO_WINDOW} if _NO_WINDOW else {}
    tmp = tempfile.mkdtemp()
    try:
        proc = subprocess.run(
            [
                initdb,
                "-D",
                os.path.join(tmp, "d"),
                "--auth=trust",
                "--encoding=utf8",
                "-U",
                "postgres",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            **win_flags,
        )
        if proc.returncode != 0:
            print("::error::Bundled initdb failed after restore:")
            print(proc.stdout)
            raise SystemExit(1)
        version = subprocess.run(
            [initdb, "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            **win_flags,
        ).stdout.strip()
        print(f"==> Verified bundled initdb creates a cluster ({version})")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def verify_pgserver_bundle(ctx, strict=True):
    try:
        import pgserver
    except Exception:
        print("==> pgserver not importable; skipping wheel restore (from-source tree assumed)")
        return

    try:
        _restore_and_smoke_test(ctx, pgserver)
    except (SystemExit, Exception) as exc:
        if strict:
            raise
        print(
            f"::warning::pgserver bundle check did not pass; continuing "
            f"(Windows best-effort, the bundle still runs initdb at first launch): {exc!r}"
        )
