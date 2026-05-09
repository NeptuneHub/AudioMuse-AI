# tasks/mediaserver_plex.py

import logging
import os
import threading

import requests

import config
from tasks.mediaserver_helper import detect_path_format

logger = logging.getLogger(__name__)

REQUESTS_TIMEOUT = 300
PLEX_PAGE_SIZE = 500
PLEX_PLAYLIST_BATCH_SIZE = 100

# Plex library item-type codes
_PLEX_TYPE_ARTIST = 8
_PLEX_TYPE_ALBUM = 9
_PLEX_TYPE_TRACK = 10
# Lyrics streams (subtitle-style) on a track's Media.Part.Stream array
_PLEX_LYRICS_STREAM_TYPE = 4

# ##############################################################################
# PLEX IMPLEMENTATION
# ##############################################################################

# Plex servers expose a stable machineIdentifier from `GET /` that's required
# in the playlist `uri=` parameter. Cache it per (base_url, token) so we don't
# refetch on every playlist mutation.
_machine_id_cache = {}
_machine_id_cache_lock = threading.Lock()


def _plex_base_url(user_creds=None):
    return (user_creds.get('url') if user_creds and user_creds.get('url') else config.PLEX_URL).rstrip('/')


def _plex_token(user_creds=None):
    if user_creds and user_creds.get('token'):
        return user_creds.get('token')
    return getattr(config, 'PLEX_TOKEN', None)


def _plex_headers(user_creds=None):
    """Return request headers for Plex JSON API. Token in header, not query string."""
    headers = {
        'Accept': 'application/json',
    }
    token = _plex_token(user_creds)
    if token:
        headers['X-Plex-Token'] = token
    return headers


def _plex_get_json(path, params=None, user_creds=None, timeout=REQUESTS_TIMEOUT):
    """GET a Plex endpoint and return the parsed `MediaContainer` body, or None on error."""
    url = f"{_plex_base_url(user_creds)}{path}"
    try:
        r = requests.get(url, headers=_plex_headers(user_creds), params=params, timeout=timeout)
        r.raise_for_status()
        return r.json().get('MediaContainer') or {}
    except Exception as e:
        logger.error(f"Plex GET {path} failed: {e}", exc_info=True)
        return None


def _machine_identifier(user_creds=None):
    """Return the Plex server's machineIdentifier, caching per (url, token)."""
    base = _plex_base_url(user_creds)
    token = _plex_token(user_creds) or ''
    cache_key = (base, token)
    with _machine_id_cache_lock:
        cached = _machine_id_cache.get(cache_key)
    if cached:
        return cached

    container = _plex_get_json('/', user_creds=user_creds)
    if not container:
        return None
    machine_id = container.get('machineIdentifier')
    if machine_id:
        with _machine_id_cache_lock:
            _machine_id_cache[cache_key] = machine_id
    return machine_id


def _music_section_ids(user_creds=None):
    """Return all music section ids for the configured Plex server (no filtering)."""
    container = _plex_get_json('/library/sections', user_creds=user_creds)
    if not container:
        return []
    sections = container.get('Directory') or []
    return [s.get('key') for s in sections if s.get('type') == 'artist' and s.get('key')]


def _get_target_section_ids():
    """Mirror of jellyfin's `_get_target_library_ids`.

    Returns:
      - None  -> no filtering, scan every music section
      - set() -> filtering active but nothing matched (caller should bail out empty)
      - set of ids -> the matching music section ids
    """
    library_names_str = getattr(config, 'MUSIC_LIBRARIES', '') or ''
    if not library_names_str.strip():
        return None

    target_names_lower = {name.strip().lower() for name in library_names_str.split(',') if name.strip()}

    container = _plex_get_json('/library/sections')
    if not container:
        return set()

    sections = container.get('Directory') or []
    music_sections = [s for s in sections if s.get('type') == 'artist']

    available = [s.get('title') for s in music_sections if s.get('title')]
    logger.info(f"Available Plex music libraries found: {available}")

    matched = set()
    for s in music_sections:
        name = (s.get('title') or '').lower()
        if name in target_names_lower and s.get('key'):
            matched.add(s.get('key'))

    if not matched:
        logger.warning(
            f"No matching Plex music libraries found for configured names: {sorted(target_names_lower)}. "
            "No albums will be analyzed."
        )
        return set()

    logger.info(f"Filtering analysis to {len(matched)} Plex section(s).")
    return matched


def _select_best_artist(item):
    """Plex track artist resolution.

    `originalTitle` holds the track artist on compilations; fall back to album artist
    (`grandparentTitle`) which is what most non-compilation tracks set.
    """
    original = (item.get('originalTitle') or '').strip()
    grandparent = (item.get('grandparentTitle') or '').strip()
    track_artist = original or grandparent or 'Unknown Artist'
    artist_id = item.get('grandparentRatingKey')
    return track_artist, artist_id


def _normalize_track(item):
    """Convert a Plex track Metadata dict into the AudioMuse track shape.

    The dispatcher and analysis code consume these dicts via keys like `Id`, `Name`,
    `AlbumArtist`, `Album`, `Year`, `Path`, `FilePath`, `Container`, `ArtistId`. We
    populate them from Plex equivalents:
      ratingKey      -> Id
      title          -> Name
      grandparentTitle / originalTitle -> AlbumArtist
      parentTitle    -> Album
      parentYear / year -> Year / ProductionYear
      Media[0].Part[0].file -> Path / FilePath
      Media[0].Part[0].container -> Container
    """
    artist_name, artist_id = _select_best_artist(item)
    media_list = item.get('Media') or []
    part = {}
    if media_list:
        parts = media_list[0].get('Part') or []
        if parts:
            part = parts[0] or {}

    file_path = part.get('file')
    container = part.get('container')

    user_rating = item.get('userRating')
    rating = None
    if user_rating is not None:
        try:
            # Plex userRating is 0..10 at half-star precision; AudioMuse expects 0..5.
            # Truncate toward zero to match LMS (`int(rating)/20`); avoid banker's
            # rounding so half-stars consistently round down.
            rating = int(float(user_rating) / 2)
        except (TypeError, ValueError):
            rating = None

    year = item.get('parentYear') or item.get('year')

    return {
        'Id': str(item.get('ratingKey')) if item.get('ratingKey') is not None else None,
        'Name': item.get('title'),
        'Album': item.get('parentTitle'),
        'AlbumArtist': artist_name,
        'OriginalAlbumArtist': item.get('grandparentTitle'),
        'ArtistId': str(artist_id) if artist_id is not None else None,
        'Year': year,
        'ProductionYear': year,
        'Path': file_path,
        'FilePath': file_path,
        'Container': container,
        'PartKey': part.get('key'),
        'Rating': rating,
        'PlayCount': item.get('viewCount'),
        'LastPlayedDate': item.get('lastViewedAt'),
    }


def _normalize_album(item):
    """Convert a Plex album Metadata dict into the AudioMuse album shape used by
    `get_recent_albums` / `search_albums`."""
    return {
        'Id': str(item.get('ratingKey')) if item.get('ratingKey') is not None else None,
        'Name': item.get('title'),
        'AlbumArtist': item.get('parentTitle'),
        'Year': item.get('year'),
        'ProductionYear': item.get('year'),
        'DateCreated': item.get('addedAt'),
    }


def _paginate(path, params, user_creds=None, page_size=PLEX_PAGE_SIZE):
    """Page through a Plex listing endpoint, returning all `Metadata` items.

    Plex caps responses (~50 by default) and exposes `X-Plex-Container-Start` /
    `X-Plex-Container-Size` for pagination. We request `page_size` per page and
    stop when the server returns fewer than `page_size` items.
    """
    out = []
    start = 0
    while True:
        page_params = dict(params or {})
        page_params['X-Plex-Container-Start'] = start
        page_params['X-Plex-Container-Size'] = page_size
        container = _plex_get_json(path, params=page_params, user_creds=user_creds)
        if container is None:
            break
        items = container.get('Metadata') or []
        if not items:
            break
        out.extend(items)
        if len(items) < page_size:
            break
        start += len(items)
    return out


# -----------------------------------------------------------------------------
# Public surface (mirrors mediaserver_jellyfin.py — see tasks/mediaserver.py
# for the exact dispatcher contract).
# -----------------------------------------------------------------------------


def list_libraries(user_creds=None):
    """List every music library on the Plex server (no filtering by MUSIC_LIBRARIES).

    Used by the setup wizard / migration assistant to render a checkbox list.
    """
    container = _plex_get_json('/library/sections', user_creds=user_creds)
    if not container:
        return []
    sections = container.get('Directory') or []
    return [
        {'id': str(s.get('key')), 'name': s.get('title')}
        for s in sections
        if s.get('type') == 'artist' and s.get('key') and s.get('title')
    ]


def test_connection(user_creds=None):
    """Probe Plex connectivity by listing a small page of tracks across music sections.

    Returns the canonical migration-probe shape: ok/error/sample_count/path_format/warnings.
    """
    try:
        section_ids = _music_section_ids(user_creds=user_creds)
        sample = []
        warnings = []
        if not section_ids:
            warnings.append('No Plex music libraries (type=artist) found on the server.')
        else:
            for section_id in section_ids:
                container = _plex_get_json(
                    f'/library/sections/{section_id}/all',
                    params={
                        'type': _PLEX_TYPE_TRACK,
                        'X-Plex-Container-Start': 0,
                        'X-Plex-Container-Size': 100,
                    },
                    user_creds=user_creds,
                )
                if container is None:
                    continue
                for item in container.get('Metadata') or []:
                    sample.append(_normalize_track(item))
                if len(sample) >= 100:
                    break

        path_format = detect_path_format(sample)
        return {
            'ok': True,
            'error': None,
            'sample_count': len(sample),
            'path_format': path_format,
            'warnings': warnings,
        }
    except Exception as e:
        logger.warning(f"Plex test_connection failed: {e}")
        return {'ok': False, 'error': str(e), 'sample_count': 0, 'path_format': 'none', 'warnings': []}


def get_recent_albums(limit):
    """Fetch recently-added Plex albums, optionally filtered by MUSIC_LIBRARIES.

    Mirrors jellyfin's three-case structure: filtering disabled / filtering
    active but no match / filtering active with matches.
    """
    target_section_ids = _get_target_section_ids()
    if isinstance(target_section_ids, set) and not target_section_ids:
        logger.warning(
            "Plex library filtering is active, but no matching libraries were found on the server. "
            "Returning no albums."
        )
        return []

    if target_section_ids is None:
        target_section_ids = _music_section_ids()
        logger.info("Scanning all Plex libraries for recent albums.")
    else:
        logger.info(f"Scanning {len(target_section_ids)} specific Plex libraries for recent albums.")

    fetch_all = (limit == 0)
    all_albums = []
    for section_id in target_section_ids:
        params = {
            'type': _PLEX_TYPE_ALBUM,
            'sort': 'addedAt:desc',
        }
        all_albums.extend(_paginate(f'/library/sections/{section_id}/all', params))

    if len(target_section_ids) > 1:
        all_albums.sort(key=lambda x: x.get('addedAt') or 0, reverse=True)

    normalized = [_normalize_album(a) for a in all_albums]
    if not fetch_all:
        return normalized[:limit]
    return normalized


def get_tracks_from_album(album_id, user_creds=None):
    """Fetch all tracks for a given Plex album ratingKey."""
    container = _plex_get_json(f'/library/metadata/{album_id}/children', user_creds=user_creds)
    if not container:
        return []
    items = container.get('Metadata') or []
    return [_normalize_track(item) for item in items]


def get_all_songs(user_creds=None):
    """Fetch every track from every Plex music library (or those matching MUSIC_LIBRARIES)."""
    target_section_ids = _get_target_section_ids()
    if isinstance(target_section_ids, set) and not target_section_ids:
        return []
    if target_section_ids is None:
        target_section_ids = _music_section_ids(user_creds=user_creds)

    out = []
    for section_id in target_section_ids:
        params = {'type': _PLEX_TYPE_TRACK}
        items = _paginate(f'/library/sections/{section_id}/all', params, user_creds=user_creds)
        for item in items:
            out.append(_normalize_track(item))
    return out


def search_albums(query, user_creds=None):
    """Search Plex albums by title across music libraries.

    Plex doesn't have a single global album-search endpoint that ignores
    section, so we fan out across music sections and merge.
    """
    target_section_ids = _music_section_ids(user_creds=user_creds)
    out = []
    for section_id in target_section_ids:
        container = _plex_get_json(
            f'/library/sections/{section_id}/search',
            params={'type': _PLEX_TYPE_ALBUM, 'title': query, 'limit': 10},
            user_creds=user_creds,
        )
        if not container:
            continue
        for item in container.get('Metadata') or []:
            out.append({
                'id': str(item.get('ratingKey')) if item.get('ratingKey') is not None else None,
                'name': item.get('title'),
                'artist': item.get('parentTitle'),
                'year': item.get('year'),
                'track_count': item.get('leafCount'),
            })
        if len(out) >= 10:
            break
    return out[:10]


def download_track(temp_dir, item):
    """Download a single Plex track by ratingKey to temp_dir.

    Plex serves the original file (untranscoded) at `{Part.key}?download=1`
    when the requesting account has Allow Downloads enabled.
    """
    try:
        track_id = item.get('Id') or item.get('ratingKey')
        part_key = item.get('PartKey')
        container = item.get('Container')

        if not part_key:
            metadata = _plex_get_json(f'/library/metadata/{track_id}')
            if not metadata:
                return None
            tracks = metadata.get('Metadata') or []
            if not tracks:
                return None
            normalized = _normalize_track(tracks[0])
            part_key = normalized.get('PartKey')
            container = container or normalized.get('Container')

        if not part_key:
            logger.error(f"Plex download_track: no Part.key for track {track_id}")
            return None

        ext = '.tmp'
        if container and isinstance(container, str) and container.strip():
            safe = container.strip().replace('/', '').replace('\\', '')
            if safe:
                ext = f'.{safe}'

        url = f"{_plex_base_url()}{part_key}"
        local_filename = os.path.join(temp_dir, f"{track_id}{ext}")
        with requests.get(
            url,
            headers=_plex_headers(),
            params={'download': 1},
            stream=True,
            timeout=REQUESTS_TIMEOUT,
        ) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        logger.info(f"Downloaded '{item.get('Name', track_id)}' to '{local_filename}'")
        return local_filename
    except Exception as e:
        logger.error(f"Plex download_track failed for {item.get('Name', 'Unknown')}: {e}", exc_info=True)
        return None


# -----------------------------------------------------------------------------
# Playlists
# -----------------------------------------------------------------------------


def get_all_playlists():
    """Return all audio playlists, normalized to {Id, Name, ...}."""
    container = _plex_get_json('/playlists', params={'playlistType': 'audio'})
    if not container:
        return []
    items = container.get('Metadata') or []
    return [
        {
            'Id': str(p.get('ratingKey')) if p.get('ratingKey') is not None else None,
            'Name': p.get('title'),
            'PlaylistType': p.get('playlistType'),
        }
        for p in items
    ]


def get_playlist_by_name(playlist_name):
    """Find a Plex audio playlist by exact name."""
    for p in get_all_playlists():
        if p.get('Name') == playlist_name:
            return p
    return None


def delete_playlist(playlist_id):
    """Delete a Plex playlist by ratingKey."""
    url = f"{_plex_base_url()}/playlists/{playlist_id}"
    try:
        r = requests.delete(url, headers=_plex_headers(), timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        return True
    except Exception as e:
        logger.error(f"Plex delete_playlist failed for {playlist_id}: {e}", exc_info=True)
        return False


def _playlist_uri(item_ids, user_creds=None):
    """Build the `uri=server://{machineId}/com.plexapp.plugins.library/library/metadata/{rk1},...` value."""
    machine_id = _machine_identifier(user_creds=user_creds)
    if not machine_id:
        raise RuntimeError("Could not resolve Plex machineIdentifier; cannot build playlist uri.")
    joined = ','.join(str(i) for i in item_ids)
    return f"server://{machine_id}/com.plexapp.plugins.library/library/metadata/{joined}"


def _create_plex_playlist(name, item_ids, user_creds=None):
    """POST a fresh Plex audio playlist with the given track ratingKeys.

    Plex caps the URI in a single request fairly aggressively for very long
    track lists, so the first batch goes via POST and the rest are appended
    via PUT in PLEX_PLAYLIST_BATCH_SIZE chunks.
    """
    if not item_ids:
        return None

    first_batch = item_ids[:PLEX_PLAYLIST_BATCH_SIZE]
    rest = item_ids[PLEX_PLAYLIST_BATCH_SIZE:]

    try:
        params = {
            'type': 'audio',
            'title': name,
            'smart': 0,
            'uri': _playlist_uri(first_batch, user_creds=user_creds),
        }
        r = requests.post(
            f"{_plex_base_url(user_creds)}/playlists",
            headers=_plex_headers(user_creds),
            params=params,
            timeout=REQUESTS_TIMEOUT,
        )
        r.raise_for_status()
        body = r.json().get('MediaContainer') or {}
        created = (body.get('Metadata') or [{}])[0]
        playlist_id = str(created.get('ratingKey')) if created.get('ratingKey') is not None else None
        if not playlist_id:
            logger.error(f"Plex create playlist '{name}': server returned no ratingKey")
            return None
    except Exception as e:
        logger.error(f"Plex create playlist '{name}' failed: {e}", exc_info=True)
        return None

    if rest and not _add_items_to_playlist(playlist_id, rest, user_creds=user_creds):
        logger.error(f"Plex create playlist '{name}': failed to append overflow tracks")

    logger.info(f"✅ Plex: created playlist '{name}' (Id={playlist_id}) with {len(item_ids)} tracks")
    return {'Id': playlist_id, 'Name': name}


def _add_items_to_playlist(playlist_id, item_ids, user_creds=None):
    """Append tracks to an existing Plex playlist (PLEX_PLAYLIST_BATCH_SIZE per request)."""
    if not item_ids:
        return True
    url = f"{_plex_base_url(user_creds)}/playlists/{playlist_id}/items"
    for i in range(0, len(item_ids), PLEX_PLAYLIST_BATCH_SIZE):
        batch = item_ids[i:i + PLEX_PLAYLIST_BATCH_SIZE]
        try:
            r = requests.put(
                url,
                headers=_plex_headers(user_creds),
                params={'uri': _playlist_uri(batch, user_creds=user_creds)},
                timeout=REQUESTS_TIMEOUT,
            )
            r.raise_for_status()
        except Exception as e:
            logger.error(
                f"Plex _add_items_to_playlist: batch starting at {i} failed for playlist {playlist_id}: {e}",
                exc_info=True,
            )
            return False
    return True


def _clear_playlist_items(playlist_id, user_creds=None):
    """DELETE every entry from a Plex playlist while keeping the playlist itself."""
    url = f"{_plex_base_url(user_creds)}/playlists/{playlist_id}/items"
    r = requests.delete(url, headers=_plex_headers(user_creds), timeout=REQUESTS_TIMEOUT)
    r.raise_for_status()


def create_playlist(base_name, item_ids):
    """Cron-style create. Always creates a new playlist (no upsert).

    Matches the jellyfin signature; `create_or_replace_playlist` is the upsert path.
    """
    _create_plex_playlist(base_name, item_ids)


def create_instant_playlist(playlist_name, item_ids, user_creds=None):
    """User-triggered instant playlist; suffixes name with `_instant` to match other providers."""
    final_name = f"{playlist_name.strip()}_instant"
    return _create_plex_playlist(final_name, item_ids, user_creds=user_creds)


def create_or_replace_playlist(playlist_name, item_ids, user_creds=None):
    """Cron upsert: create the playlist if missing, or wipe its contents and refill in place.

    Plex supports `DELETE /playlists/{id}/items` (clear all entries) without dropping the
    playlist itself, so the ratingKey stays stable across cron runs. If the clear fails,
    fall back to delete+recreate (which yields a new ratingKey).
    """
    if not item_ids:
        return None

    existing = get_playlist_by_name(playlist_name)
    if not existing:
        return _create_plex_playlist(playlist_name, item_ids, user_creds=user_creds)

    playlist_id = existing.get('Id')
    if not playlist_id:
        logger.error(f"Plex create_or_replace_playlist: existing playlist '{playlist_name}' has no Id")
        return None

    try:
        _clear_playlist_items(playlist_id, user_creds=user_creds)
    except Exception as e:
        logger.info(
            f"Plex: in-place clear of '{playlist_name}' failed ({e}); falling back to delete+recreate."
        )
        if not delete_playlist(playlist_id):
            logger.error(f"Plex: failed to delete playlist '{playlist_name}' (Id={playlist_id}) for fallback")
            return None
        return _create_plex_playlist(playlist_name, item_ids, user_creds=user_creds)

    if not _add_items_to_playlist(playlist_id, item_ids, user_creds=user_creds):
        logger.error(f"Plex create_or_replace_playlist: failed to add tracks to playlist {playlist_id}")
        return None

    logger.info(
        f"✅ Plex: replaced contents of playlist '{playlist_name}' (Id={playlist_id}, tracks={len(item_ids)})"
    )
    return {'Id': playlist_id, 'Name': playlist_name}


# -----------------------------------------------------------------------------
# User-specific reads
# -----------------------------------------------------------------------------


def get_top_played_songs(limit, user_creds=None):
    """Return the top-played tracks across music libraries, sorted by viewCount desc."""
    target_section_ids = _music_section_ids(user_creds=user_creds)
    out = []
    for section_id in target_section_ids:
        params = {
            'type': _PLEX_TYPE_TRACK,
            'sort': 'viewCount:desc',
        }
        items = _paginate(f'/library/sections/{section_id}/all', params, user_creds=user_creds)
        for item in items:
            if not item.get('viewCount'):
                # Once we hit zero-viewCount tracks the rest are unplayed; skip them.
                continue
            out.append(_normalize_track(item))

    # Sort across sections and trim
    out.sort(key=lambda t: t.get('PlayCount') or 0, reverse=True)
    if limit and limit > 0:
        return out[:limit]
    return out


def get_last_played_time(item_id, user_creds=None):
    """Return Plex `lastViewedAt` (epoch seconds) for a track ratingKey, or None."""
    container = _plex_get_json(f'/library/metadata/{item_id}', user_creds=user_creds)
    if not container:
        return None
    items = container.get('Metadata') or []
    if not items:
        return None
    return items[0].get('lastViewedAt')


def get_lyrics(track_id: str, timeout: float = 2.5):
    """Fetch embedded lyrics for a Plex track via its lyrics-stream key.

    Plex stores synced/unsynced lyrics as a stream with `streamType=4` on the
    track's Media.Part.Stream array. Fetching that stream's `key` returns the
    LRC/plain-text payload.
    """
    try:
        container = _plex_get_json(f'/library/metadata/{track_id}', timeout=timeout)
        if not container:
            return None
        items = container.get('Metadata') or []
        if not items:
            return None
        track = items[0]

        lyrics_key = None
        for media in track.get('Media') or []:
            for part in media.get('Part') or []:
                for stream in part.get('Stream') or []:
                    if stream.get('streamType') == _PLEX_LYRICS_STREAM_TYPE and stream.get('key'):
                        lyrics_key = stream.get('key')
                        break
                if lyrics_key:
                    break
            if lyrics_key:
                break

        if not lyrics_key:
            return None

        url = f"{_plex_base_url()}{lyrics_key}"
        r = requests.get(url, headers=_plex_headers(), timeout=timeout)
        r.raise_for_status()
        text = r.text
        if not text:
            return None
        # Strip LRC timestamp prefixes like "[00:12.34]" so the rest of the
        # pipeline gets plain lyrics text. Skip metadata lines like "[ar: ...]".
        lines = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            while stripped.startswith('['):
                end = stripped.find(']')
                if end == -1:
                    break
                stripped = stripped[end + 1:].strip()
            if stripped:
                lines.append(stripped)
        out = '\n'.join(lines).strip()
        return out or None
    except Exception as exc:
        logger.debug(f"Plex get_lyrics failed for {track_id}: {exc}")
        return None
