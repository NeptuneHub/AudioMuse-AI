# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Shared, provider-agnostic helpers for the AudioMuse-AI media servers.

Used by every backend for cross-cutting concerns, keeping the per-provider
modules focused on their own API specifics.

Main Features:
* Detects rejected-credential errors (is_auth_error: HTTP 401/403 plus wording).
* Selects the best artist from provider metadata and normalizes path/format fields.
"""

import logging
import os
import re

logger = logging.getLogger(__name__)

_AUTH_STATUS_CODES = {401, 403}
_AUTH_TEXT_HINTS = (
    'unauthorized',
    'unauthorised',
    'forbidden',
    'wrong username',
    'wrong password',
    'invalid credentials',
    'invalid login',
    'authentication failed',
    'not authorized',
    'permission denied',
)


def is_auth_error(exc):
    seen = set()
    cur = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        status = getattr(getattr(cur, 'response', None), 'status_code', None)
        if status in _AUTH_STATUS_CODES:
            return True
        text = str(cur).lower()
        if any(hint in text for hint in _AUTH_TEXT_HINTS):
            return True
        cur = getattr(cur, '__cause__', None) or getattr(cur, '__context__', None)
    return False


def select_best_artist(item, title="Unknown"):
    if item.get('ArtistItems') and len(item['ArtistItems']) > 0:
        track_artist = item['ArtistItems'][0].get('Name', 'Unknown Artist')
        artist_id = item['ArtistItems'][0].get('Id')
    elif item.get('Artists') and len(item['Artists']) > 0:
        track_artist = item['Artists'][0]
        artist_id = None
    elif item.get('AlbumArtist'):
        track_artist = item['AlbumArtist']
        artist_id = None
    else:
        track_artist = 'Unknown Artist'
        artist_id = None

    return track_artist, artist_id


def detect_download_extension(item):
    file_extension = '.tmp'
    try:
        container = item.get('Container')
        if container and isinstance(container, str) and container.strip():
            safe_container = container.strip().replace('/', '').replace('\\', '')
            if safe_container:
                file_extension = f".{safe_container}"
                logger.debug(f"Using Container field for format: {file_extension}")
        elif item.get('Path'):
            file_extension = os.path.splitext(item['Path'])[1] or '.tmp'
    except Exception as e:
        logger.debug(f"Error getting format from Container/Path, using .tmp: {e}")
    return file_extension


def detect_path_format(tracks):
    def _is_absolute_path(path):
        if not path:
            return False
        path_str = str(path)
        lower = path_str.lower()
        return (
            path_str.startswith('/')
            or path_str.startswith('\\')
            or lower.startswith('file://')
            or re.match(r'^[A-Za-z]:[\\/]', path_str)
        )

    paths = []
    for track in tracks or []:
        if not isinstance(track, dict):
            continue
        path = track.get('path') or track.get('Path') or track.get('url') or track.get('Url')
        if path:
            paths.append(path)

    if not paths:
        return 'none'

    ratio = sum(1 for p in paths if _is_absolute_path(p)) / len(paths)
    if ratio >= 0.8:
        return 'absolute'
    if ratio <= 0.2:
        return 'relative'
    return 'mixed'
