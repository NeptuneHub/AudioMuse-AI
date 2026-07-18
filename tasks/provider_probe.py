# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Probe a media provider and normalise its track metadata.

Thin, provider-agnostic wrapper over the mediaserver clients, used by the
provider-migration flow and by the multi-server sweep to test connectivity,
enumerate libraries, and pull whole catalogues.

Main Features:
* Supports jellyfin, emby, navidrome, lyrion, and plex, rejecting any other
  provider type early.
* Normalises heterogeneous provider fields (Jellyfin/Emby PascalCase, Subsonic
  camelCase, and lower-case variants) into one flat track dict, coercing the
  year to an int; track lists are normalised in place so the raw provider list
  never coexists with a full normalised copy.
* The normalised dict carries exactly what the consumers read: the id, path and
  metadata the sweep matches on, plus the artist id and rating it aligns.
"""

from tasks import mediaserver


def _duration_seconds(item):
    seconds = item.get('DurationSeconds')
    if seconds is None:
        ticks = item.get('RunTimeTicks')
        if ticks is not None:
            try:
                seconds = float(ticks) / 10_000_000.0
            except (TypeError, ValueError):
                return None
        else:
            seconds = item.get('duration')
    try:
        seconds = float(seconds)
    except (TypeError, ValueError):
        return None
    return seconds if seconds > 0 else None


def _normalize_track(item):
    if item is None:
        return {
            'id': None,
            'path': None,
            'title': None,
            'artist': None,
            'artist_id': None,
            'album_artist': None,
            'album': None,
            'year': None,
            'rating': None,
            'duration': None,
        }

    def _try(*keys):
        for key in keys:
            value = item.get(key)
            if value is not None:
                return value
        return None

    year = _try('Year', 'year')
    if isinstance(year, str):
        try:
            year = int(year)
        except (TypeError, ValueError):
            year = None

    return {
        'id': _try('Id', 'id', 'track_id'),
        'path': _try('Path', 'path', 'url'),
        'title': _try('Name', 'name', 'title'),
        'artist': _try('AlbumArtist', 'artist', 'author'),
        'artist_id': _try('ArtistId', 'artistId', 'artist_id'),
        'album_artist': _try('OriginalAlbumArtist', 'albumArtist', 'AlbumArtist'),
        'album': _try('Album', 'album'),
        'year': year,
        'rating': _try('Rating', 'rating', 'userRating'),
        'duration': _duration_seconds(item),
    }


_SUPPORTED_PROVIDERS = {'jellyfin', 'emby', 'navidrome', 'lyrion', 'plex'}


def _normalize_provider_type(provider_type):
    t = (provider_type or '').lower()
    if t not in _SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Provider type '{provider_type}' not supported by migration probe. "
            f"Supported: {sorted(_SUPPORTED_PROVIDERS)}"
        )
    return t


def fetch_all_tracks(provider_type, creds, apply_filter=False):
    t = _normalize_provider_type(provider_type)
    items = mediaserver.get_all_songs(
        user_creds=creds, provider_type=t, apply_filter=apply_filter
    ) or []
    for i in range(len(items)):
        items[i] = _normalize_track(items[i])
    return items


def search_albums(provider_type, creds, query):
    t = _normalize_provider_type(provider_type)
    return mediaserver.search_albums(query, user_creds=creds, provider_type=t)


def get_album_tracks(provider_type, creds, album_id):
    t = _normalize_provider_type(provider_type)
    items = mediaserver.get_tracks_from_album(album_id, user_creds=creds, provider_type=t) or []
    for i in range(len(items)):
        items[i] = _normalize_track(items[i])
    return items


def test_connection(provider_type, creds):
    t = _normalize_provider_type(provider_type)
    return mediaserver.test_connection(user_creds=creds, provider_type=t)


def list_libraries(provider_type, creds):
    t = _normalize_provider_type(provider_type)
    return mediaserver.list_libraries(user_creds=creds, provider_type=t)
