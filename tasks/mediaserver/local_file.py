# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Read a track from a mounted library instead of downloading it.

Every provider reports the file path it holds a track at, and an install whose
library is mounted into the container can read that file directly: no HTTP round
trip, no second copy of the bytes, no load on the media server. This module
turns a provider's reported path into something the analysis pipeline can open,
or returns None so the caller downloads as usual. It is provider-agnostic - all
six backends populate ``Path``/``FilePath``.

Two properties matter more than the speed:

* The pipeline DELETES whatever ``download_track`` returns (see the ``finally``
  in tasks/analysis/album.py). Handing back the library file would delete the
  user's music, so what is returned is always a symlink inside TEMP_DIR:
  removing it unlinks the link and never touches the target.
* The path is reported by the media server, which makes it untrusted input.
  Anything resolving outside the configured roots is refused, so a hostile or
  merely wrong path cannot turn analysis into an arbitrary-file reader.

Main Features:
* Rewrites the server's path onto the container's mount point via
  LOCAL_FILE_PATH_MAP, longest prefix first, tolerating file:// URLs and
  Windows separators.
* Refuses any path whose REAL location (symlinks resolved) falls outside
  LOCAL_FILE_ROOTS, and refuses everything when no root is configured.
* Reads only from roots this process CANNOT write to, so the library cannot be
  modified or deleted whatever the rest of the code does; a writable root is
  skipped unless LOCAL_FILE_REQUIRE_READONLY is turned off.
* Publishes the track as an atomically-replaced symlink in TEMP_DIR named for
  the track id, so concurrent workers cannot corrupt each other's link.
* Returns None on every failure so the caller falls back to downloading; local
  access is an optimisation and never a new way for analysis to fail.
"""

import logging
import os
import re
from urllib.parse import unquote, urlparse

import config

logger = logging.getLogger(__name__)

_warned = set()
_announced = False


def _warn_once(key, message, *args):
    if key in _warned:
        return
    _warned.add(key)
    logger.warning(message, *args)


def _enabled():
    return bool(config.LOCAL_FILE_ACCESS)


def _roots():
    raw = config.LOCAL_FILE_ROOTS or ''
    return [os.path.realpath(part.strip()) for part in raw.split(',') if part.strip()]


def _writable(path):
    """True when THIS process could write into ``path``.

    A read-only mount fails this even for root (access(2) reports EROFS), which
    is the property worth testing: not "is the mount flagged ro" but "can this
    process change anything here".
    """
    return os.access(path, os.W_OK)


def _usable_roots():
    """Roots this process may read from, dropping writable ones by default.

    Nothing here writes to the library, but a root the process CANNOT write to
    means no bug, dependency or later change can delete a track either. The
    check is skipped only when an operator turns LOCAL_FILE_REQUIRE_READONLY off.
    """
    roots = _roots()
    if not config.LOCAL_FILE_REQUIRE_READONLY:
        return roots

    usable = []
    for root in roots:
        if _writable(root):
            _warn_once(
                f'writable-root:{root}',
                'Local file access is SKIPPING %s because this process can write to '
                'it. Mount the library read-only (for example "%s:%s:ro") so the '
                'files cannot be modified or deleted, or set '
                'LOCAL_FILE_REQUIRE_READONLY=false to accept the risk. Tracks under '
                'this root are downloaded instead.',
                root, root, root,
            )
            continue
        usable.append(root)
    return usable


def _path_map():
    """``[(server_prefix, container_prefix)]``, longest server prefix first.

    Longest first so a specific mapping wins over a broader one covering the
    same tree.
    """
    raw = config.LOCAL_FILE_PATH_MAP or ''
    pairs = []
    for entry in raw.split(','):
        server_prefix, separator, container_prefix = entry.partition('=')
        if not separator:
            continue
        server_prefix = server_prefix.strip().rstrip('/\\')
        container_prefix = container_prefix.strip().rstrip('/')
        if server_prefix and container_prefix:
            pairs.append((server_prefix, container_prefix))
    pairs.sort(key=lambda pair: len(pair[0]), reverse=True)
    return pairs


def _from_file_url(text):
    if not text.lower().startswith('file://'):
        return text
    path = unquote(urlparse(text).path)
    # file:///C:/Music/x.mp3 parses to /C:/Music/x.mp3, which is not a path on
    # Windows; drop the leading slash a drive letter leaves behind.
    if re.match(r'^/[A-Za-z]:', path):
        path = path[1:]
    return path


def _mapped_path(raw_path):
    """The container-side path for a path the media server reported."""
    text = _from_file_url(str(raw_path or '').strip())
    if not text:
        return None
    for server_prefix, container_prefix in _path_map():
        if text.startswith(server_prefix):
            remainder = text[len(server_prefix):].replace('\\', '/')
            return container_prefix + remainder
    return text


def _within_roots(resolved, roots):
    for root in roots:
        try:
            if os.path.commonpath([root, resolved]) == root:
                return True
        except ValueError:
            # Different drives (Windows) or a mix of absolute and relative.
            continue
    return False


def _resolved_local_path(raw_path):
    mapped = _mapped_path(raw_path)
    if not mapped:
        return None

    roots = _usable_roots()
    if not roots:
        _warn_once(
            'no-roots',
            'LOCAL_FILE_ACCESS is enabled but no usable root is configured, so every '
            'path is refused and tracks are downloaded instead. Set LOCAL_FILE_ROOTS '
            'to the read-only mount the library lives on.',
        )
        return None

    # realpath first: the allowlist has to be checked against where the path
    # ACTUALLY lands, or a symlink inside the library would walk straight out of it.
    resolved = os.path.realpath(mapped)
    if not _within_roots(resolved, roots):
        _warn_once(
            'outside-roots',
            'Local file access refused: %s resolves to %s, outside LOCAL_FILE_ROOTS '
            '(%s). Tracks will be downloaded. This is logged once per worker.',
            mapped, resolved, ', '.join(roots),
        )
        return None

    if not os.path.isfile(resolved) or not os.access(resolved, os.R_OK):
        return None
    try:
        if os.path.getsize(resolved) <= 0:
            return None
    except OSError:
        return None
    return resolved


def _replace_link(target, link_path):
    """Point ``link_path`` at ``target``, replacing whatever is there atomically.

    A symlink is preferred because it crosses filesystems, which matters when
    TEMP_DIR is a tmpfs and the library is a mount. Windows refuses symlinks
    without a privilege the service rarely holds, so a hardlink is the fallback:
    equally safe here, since removing a hardlink only drops that name and the
    library keeps its own.

    Two workers can be handed the same track and would build the same link name;
    staging plus os.replace means neither ever sees a half-made link.
    """
    staging = f"{link_path}.{os.getpid()}.tmplink"
    if os.path.lexists(staging):
        os.remove(staging)
    try:
        os.symlink(target, staging)
    except (OSError, NotImplementedError):
        os.link(target, staging)
    os.replace(staging, link_path)


def link_local_copy(temp_dir, item):
    """A TEMP_DIR symlink to this track's file on disk, or None to download it."""
    if not _enabled():
        return None

    resolved = _resolved_local_path(item.get('Path') or item.get('FilePath'))
    if not resolved:
        return None

    track_id = item.get('Id') or item.get('id') or os.path.basename(resolved)
    extension = os.path.splitext(resolved)[1] or '.tmp'
    link_path = os.path.join(temp_dir, f"{track_id}{extension}")

    try:
        os.makedirs(temp_dir, exist_ok=True)
        _replace_link(resolved, link_path)
    except OSError as e:
        _warn_once(
            'link-failed',
            'Could not link %s into %s (%s); falling back to downloading. This is '
            'logged once per worker.',
            resolved, temp_dir, e,
        )
        return None

    global _announced
    if not _announced:
        _announced = True
        logger.info(
            'Local file access is serving tracks from disk (first hit: %s); no '
            'downloads are needed for files under LOCAL_FILE_ROOTS.',
            resolved,
        )
    return link_path
