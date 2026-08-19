# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The guard that decides whether an embedded PostgreSQL data directory is reusable.

Linux and Windows drive vendored PostgreSQL binaries the same way, so this
decision existed as two byte-identical copies plus a third partial one in the
Windows pgserver path. It is the one place that may erase user data, so a copy
that drifts is the copy that wipes a real cluster: the refusal below is the only
thing standing between a half-written data directory and a database somebody
still needs.

A directory counts as initialized once it carries our marker file, or once it
carries global/pg_control (an interrupted first run that did finish initdb, which
we then mark so the next boot answers from the marker alone).

Main Features:
* reset_data_dir clears only a directory that never became a cluster
* An existing cluster raises instead of being wiped, whatever asked for the reset
* The marker file is written best-effort: failing to mark never fails a boot
"""

import logging
import os
import shutil

logger = logging.getLogger("audiomuse.supervisor")

READY_MARKER = "audiomuse_initialized"


def has_cluster_data(data_dir):
    return os.path.exists(os.path.join(data_dir, "global", "pg_control"))


def initialized(data_dir):
    if os.path.exists(os.path.join(data_dir, READY_MARKER)):
        return True
    if not has_cluster_data(data_dir):
        return False
    try:
        with open(os.path.join(data_dir, READY_MARKER), "w", encoding="utf-8") as fh:
            fh.write("ok\n")
    except OSError:
        pass
    return True


def refuse_to_wipe_cluster(data_dir):
    if has_cluster_data(data_dir):
        raise RuntimeError(
            f"Refusing to wipe {data_dir}: it contains an existing PostgreSQL "
            "cluster (global/pg_control present). Back it up or remove it "
            "manually if you really want a fresh start."
        )


def reset_data_dir(data_dir):
    if not (os.path.isdir(data_dir) and os.listdir(data_dir)):
        return
    refuse_to_wipe_cluster(data_dir)
    logger.warning("Clearing incomplete PostgreSQL data dir %s before re-init", data_dir)
    for entry in os.listdir(data_dir):
        target = os.path.join(data_dir, entry)
        try:
            if os.path.isdir(target) and not os.path.islink(target):
                shutil.rmtree(target)
            else:
                os.unlink(target)
        except OSError:
            logger.exception("Could not remove %s", target)
