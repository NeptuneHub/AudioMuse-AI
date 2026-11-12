# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""VLC media player backend for the AudioMuse-AI media-server abstraction.

Implements the provider interface against the Lua HTTP interface VLC opens with
``vlc --extraintf http --http-port 8080 --http-password <secret>``. Dispatched by
tasks/mediaserver/__init__.py when config.MEDIASERVER_TYPE == 'vlc'.

Main Features:
* The catalogue is the configured music folder on the VLC host, walked through
  requests/browse.json. The folder may be a Windows, Linux or macOS path or a
  file:// URI; it is always sent as a file:// URI (VLC lists nothing for a bare
  path) and every sub-folder is reached through the URI VLC itself returns.
* VLC has no track ids: a track id is the SHA-256 of the file's path relative
  to the music folder ('/' separators, NFC), so the same file in the same
  folder keeps its id on every scan and on any VLC host. An album is a folder
  and its id is the reversible base64 of that relative folder, because an album
  job only receives the album id.
* The HTTP interface exposes no tags, so title, artist, album and year come from
  the Artist/Album (Year)/NN - Title folder layout.
* The HTTP interface has no download route and a VLM broadcast is paced in real
  time, so a track is downloaded in three steps: a VLM job copies it at disk
  speed into Folder.jpg in VLC's temp folder (C:/Windows/Temp or /tmp; MP3
  stays raw MPEG audio, anything else is remuxed to mkv, no re-encoding), that
  file is queued in VLC's play queue for a moment so VLC's folder art finder
  makes it the item's artwork, and the /art route serves it whole. The copy job
  named audiomuse_copy is also the lock that keeps two downloads off the one
  scratch file; a job older than COPY_STALE_SECONDS is a dead download's.
* VLC keeps no play counts, lyrics or saved playlists, so those reads are empty.
  An instant playlist goes to VLC's play queue; any other playlist is logged as
  not stored and returned as created, so the task that asked for it carries on.
* Query strings are percent-encoded by hand: VLC's Lua parser does not turn '+'
  back into a space.
"""

import base64
import hashlib
import html
import logging
import os
import re
import tempfile
import time
import unicodedata
import uuid
import xml.etree.ElementTree as ET
from urllib.parse import quote, unquote, urlparse

import config

from . import context
from . import http as requests
from .helper import detect_path_format, is_auth_error

logger = logging.getLogger(__name__)

REQUESTS_TIMEOUT = 30
DOWNLOAD_READ_TIMEOUT = 120
VLC_MAX_FOLDER_DEPTH = 32
SEARCH_RESULTS_LIMIT = 10
TEST_SAMPLE_SIZE = 100

COPY_MEDIA = 'audiomuse_copy'
COPY_SINCE_OPTION = 'audiomuse-since'
COPY_TIMEOUT = 120
COPY_STALE_SECONDS = 300
COPY_POLL_SECONDS = 0.05
COPY_RETRY_SECONDS = 0.25
ART_TIMEOUT = 15
SCRATCH_NAME = 'Folder.jpg'
WINDOWS_SCRATCH_DIR = 'C:/Windows/Temp'
POSIX_SCRATCH_DIR = '/tmp'
PLAY_QUEUE_ID = 'vlc-play-queue'

_ALBUM_ID_PREFIX = 'd-'
_AUDIO_EXTENSIONS = frozenset({
    '.mp3', '.flac', '.ogg', '.oga', '.opus', '.m4a', '.m4b', '.aac', '.alac',
    '.wav', '.wma', '.aif', '.aiff', '.ape', '.wv', '.mpc', '.mka', '.dsf', '.dff',
})
_YEAR_SUFFIX = re.compile(r'^(?P<name>.*?)\s*[(\[](?P<year>\d{4})[)\]]\s*$')
_YEAR_PREFIX = re.compile(r'^(?P<year>\d{4})\s*-\s*(?P<name>.+)$')
_DISC_FOLDER = re.compile(r'^(?:cd|disc|disk)\s*\d{1,2}$', re.IGNORECASE)
_TRACK_NUMBER = re.compile(r'^\d{1,3}$')
_TRACK_PREFIX = re.compile(r'^(?:\d{1,2}[-.]\d{1,3}\s+|\d{1,3}\s*[-._)]\s+|0\d\s+)')
_DRIVE_ONLY = re.compile(r'^/?[A-Za-z]:$')
_WINDOWS_ROOT = re.compile(r'^file://(?:/[A-Za-z]:|[^/])')
_ERROR_PRE = re.compile(r'<pre>(.*?)</pre>', re.DOTALL)
_RAW_COPY_EXTENSIONS = frozenset({'.mp3'})


class VlcAPIError(Exception):
    pass


def folder_uri(folder):
    text = (folder or '').strip()
    if not text:
        return ''
    if text.lower().startswith('file:'):
        parsed = urlparse(text.replace('\\', '/'))
        host, path = parsed.netloc, unquote(parsed.path)
        if _DRIVE_ONLY.match(host):
            host, path = '', '/' + host + path
    else:
        text = text.replace('\\', '/')
        if text.startswith('//'):
            host, _, rest = text[2:].partition('/')
            path = '/' + rest
        else:
            host, path = '', text if text.startswith('/') else '/' + text
    path = path.rstrip('/') or '/'
    if _DRIVE_ONLY.match(path):
        path += '/'
    return 'file://' + host + quote(path, safe='/:')


def _child_uri(root, parts):
    if not parts:
        return root
    return root.rstrip('/') + '/' + '/'.join(quote(part, safe='') for part in parts)


def _settings(user_creds=None):
    creds = context.active_creds(user_creds) or {}
    if not creds.get('url'):
        creds = {
            'url': config.VLC_URL,
            'user': config.VLC_USER,
            'password': config.VLC_PASSWORD,
            'music_folder': config.VLC_MUSIC_FOLDER,
        }
    return {
        'url': (creds.get('url') or '').strip().rstrip('/'),
        'user': creds.get('user') or '',
        'password': creds.get('password') or '',
        'root': folder_uri(creds.get('music_folder')),
    }


def _query(params):
    return '&'.join(f"{key}={quote(str(value), safe='')}" for key, value in params.items())


def _get(settings, path, params=None, timeout=REQUESTS_TIMEOUT, stream=False):
    if not settings['url']:
        raise VlcAPIError("The VLC URL is not configured.")
    url = settings['url'] + path + ('?' + _query(params) if params else '')
    response = requests.get(
        url, auth=(settings['user'], settings['password']), timeout=timeout, stream=stream
    )
    response.raise_for_status()
    return response


def _error_text(body):
    match = _ERROR_PRE.search(body or '')
    return html.unescape(match.group(1)).strip()[:300] if match else 'unexpected response'


def _get_json(settings, path, params=None):
    response = _get(settings, path, params)
    try:
        return response.json()
    except ValueError as exc:
        raise VlcAPIError(f"VLC failed on {path}: {_error_text(response.text)}") from exc


def _get_xml(settings, path, params=None):
    response = _get(settings, path, params)
    try:
        return ET.fromstring(response.content)
    except ET.ParseError as exc:
        raise VlcAPIError(f"VLC failed on {path}: {_error_text(response.text)}") from exc


def _browse(settings, uri):
    payload = _get_json(settings, '/requests/browse.json', {'uri': uri})
    elements = payload.get('element') if isinstance(payload, dict) else None
    if not isinstance(elements, list):
        raise VlcAPIError(f"VLC returned no folder listing for {uri}")
    return [
        e for e in elements
        if isinstance(e, dict) and str(e.get('name') or '') not in ('', '.', '..')
    ]


def _is_audio(name):
    return os.path.splitext(name)[1].lower() in _AUDIO_EXTENSIONS


def _audio_files(elements):
    return [
        e for e in elements
        if e.get('type') != 'dir' and e.get('uri')
        and not str(e['name']).startswith('.') and _is_audio(str(e['name']))
    ]


def _walk(settings):
    if not settings['root']:
        raise VlcAPIError("The VLC music folder is not configured.")
    pending = [(settings['root'], ())]
    seen = set()
    while pending:
        uri, parts = pending.pop()
        if uri in seen or len(parts) > VLC_MAX_FOLDER_DEPTH:
            continue
        seen.add(uri)
        elements = _browse(settings, uri)
        for element in reversed(elements):
            name = str(element['name'])
            if element.get('type') == 'dir' and element.get('uri') and not name.startswith('.'):
                pending.append((element['uri'], parts + (name,)))
        files = _audio_files(elements)
        if files:
            yield parts, files


def _rel_path(parts, name):
    return unicodedata.normalize('NFC', '/'.join(parts + (name,)))


def track_id(rel_path):
    normalized = unicodedata.normalize('NFC', rel_path.replace('\\', '/').strip('/'))
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:32]


def album_id(parts):
    rel = unicodedata.normalize('NFC', '/'.join(parts))
    encoded = base64.urlsafe_b64encode(rel.encode('utf-8')).decode('ascii').rstrip('=')
    return _ALBUM_ID_PREFIX + encoded


def album_parts(value):
    text = str(value or '')
    if not text.startswith(_ALBUM_ID_PREFIX):
        raise ValueError(f"{text!r} is not a VLC album id")
    encoded = text[len(_ALBUM_ID_PREFIX):]
    rel = base64.urlsafe_b64decode(encoded + '=' * (-len(encoded) % 4)).decode('utf-8')
    parts = tuple(part for part in rel.split('/') if part)
    if any(part in ('.', '..') for part in parts):
        raise ValueError(f"{text!r} points outside the music folder")
    return parts


def _split_year(name):
    match = _YEAR_SUFFIX.match(name) or _YEAR_PREFIX.match(name)
    if match and match.group('name').strip():
        return match.group('name').strip(), int(match.group('year'))
    return name, None


def _parse_stem(stem):
    pieces = [piece.strip() for piece in stem.split(' - ')]
    for index in range(len(pieces) - 2, -1, -1):
        if _TRACK_NUMBER.match(pieces[index]):
            return (
                ' - '.join(pieces[index + 1:]),
                pieces[0] if index >= 1 else None,
                pieces[1] if index >= 2 else None,
            )
    stem = _TRACK_PREFIX.sub('', stem, count=1).strip()
    artist, separator, title = stem.partition(' - ')
    if separator and artist.strip() and title.strip():
        return title.strip(), artist.strip(), None
    return stem, None, None


def _album_folders(parts):
    folders = list(parts)
    disc = None
    if len(folders) >= 2 and _DISC_FOLDER.match(folders[-1]):
        disc = folders.pop()
    return folders, disc


def _describe(parts, file_name):
    folders, _disc = _album_folders(parts)
    artist_folder = folders[-2] if len(folders) >= 2 else ''
    album, year = _split_year(folders[-1]) if folders else (None, None)
    if album and not artist_folder and ' - ' in album:
        artist_folder, _, album = album.partition(' - ')
    title, file_artist, file_album = _parse_stem(os.path.splitext(file_name)[0])
    return (
        title or file_name,
        file_artist or artist_folder or 'Unknown Artist',
        artist_folder or file_artist or 'Unknown Artist',
        album or file_album,
        year,
    )


def _album_name(parts):
    folders, disc = _album_folders(parts)
    if not folders:
        return 'Unknown Album'
    name = _split_year(folders[-1])[0]
    if len(folders) == 1 and ' - ' in name:
        name = name.partition(' - ')[2]
    return f"{name} ({disc})" if disc else name


def _track(parts, element):
    name = str(element['name'])
    rel = _rel_path(parts, name)
    title, artist, album_artist, album, year = _describe(parts, name)
    path = element.get('path') or unquote(urlparse(element['uri']).path)
    return {
        'Id': track_id(rel),
        'Name': title,
        'AlbumArtist': artist,
        'ArtistId': None,
        'OriginalAlbumArtist': album_artist,
        'Album': album,
        'Path': path,
        'FilePath': path,
        'url': element['uri'],
        'RelativePath': rel,
        'Year': year,
        'Rating': None,
        'DurationSeconds': None,
    }


def _album_tracks(parts, files):
    ordered = sorted(files, key=lambda e: str(e['name']).casefold())
    return [_track(parts, element) for element in ordered]


def _modified(element):
    try:
        return float(element.get('modification_time') or 0)
    except (TypeError, ValueError):
        return 0.0


def list_libraries(user_creds=None):
    return []


def get_recent_albums(limit):
    settings = _settings()
    albums = []
    for parts, files in _walk(settings):
        newest = max(_modified(element) for element in files)
        albums.append((newest, {'Id': album_id(parts), 'Name': _album_name(parts)}))
    albums.sort(key=lambda pair: pair[0], reverse=True)
    result = [album for _newest, album in albums]
    return result[:limit] if limit else result


def get_tracks_from_album(album_id_value, user_creds=None):
    settings = _settings(user_creds)
    try:
        parts = album_parts(album_id_value)
        files = _audio_files(_browse(settings, _child_uri(settings['root'], parts)))
    except Exception:
        logger.exception("VLC could not list the tracks of album %s", album_id_value)
        return []
    return _album_tracks(parts, files)


def get_all_songs(user_creds=None, apply_filter=True):
    settings = _settings(user_creds)
    songs = []
    for parts, files in _walk(settings):
        songs.extend(_album_tracks(parts, files))
    logger.info("Found %d songs under the VLC music folder %s", len(songs), settings['root'])
    return songs


def search_albums(query, user_creds=None):
    settings = _settings(user_creds)
    needle = (query or '').strip().casefold()
    results = []
    try:
        for parts, files in _walk(settings):
            name = _album_name(parts)
            _title, _artist, artist, _album, year = _describe(parts, str(files[0]['name']))
            if needle and needle not in name.casefold() and needle not in artist.casefold():
                continue
            results.append({
                'id': album_id(parts),
                'name': name,
                'artist': artist,
                'year': year,
                'track_count': len(files),
            })
            if len(results) >= SEARCH_RESULTS_LIMIT:
                break
    except Exception:
        logger.exception("VLC album search for %r failed", query)
        return []
    return results


def _failed_test(error, auth_failed=False):
    return {
        'ok': False,
        'error': error,
        'auth_failed': auth_failed,
        'sample_count': 0,
        'path_format': 'none',
        'warnings': [],
    }


def test_connection(user_creds=None):
    settings = _settings(user_creds)
    missing = [
        label for key, label in (('url', 'URL'), ('password', 'password'), ('root', 'music folder'))
        if not settings[key]
    ]
    if missing:
        return _failed_test(f"VLC {', '.join(missing)} not configured")
    sample = []
    smallest = None
    try:
        _get_json(settings, '/requests/status.json')
        for parts, files in _walk(settings):
            for element in files[:TEST_SAMPLE_SIZE - len(sample)]:
                sample.append(_track(parts, element))
                if smallest is None or _size(element) < _size(smallest):
                    smallest = element
            if len(sample) >= TEST_SAMPLE_SIZE:
                break
        if smallest is not None:
            _probe_download(settings, smallest['uri'])
    except Exception as exc:
        logger.warning("VLC test_connection failed", exc_info=True)
        return _failed_test(str(exc), auth_failed=is_auth_error(exc))
    if not sample:
        return _failed_test(f"VLC lists no audio files under {settings['root']}")
    return {
        'ok': True,
        'error': None,
        'sample_count': len(sample),
        'path_format': detect_path_format(sample),
        'warnings': [],
    }


def _vlm(settings, command):
    root = _get_xml(settings, '/requests/vlm_cmd.xml', {'command': command})
    error = (root.findtext('error') or '').strip()
    if error:
        raise VlcAPIError(f"VLC rejected the VLM command {command.split(' ', 1)[0]!r}: {error}")
    return root


def _size(element):
    try:
        return int(element.get('size') or 0)
    except (TypeError, ValueError):
        return 0


def _scratch_path(settings):
    folder = WINDOWS_SCRATCH_DIR if _WINDOWS_ROOT.match(settings['root']) else POSIX_SCRATCH_DIR
    return f"{folder}/{SCRATCH_NAME}"


def _copy_format(uri):
    extension = os.path.splitext(unquote(urlparse(uri).path))[1].lower()
    return ('raw', extension) if extension in _RAW_COPY_EXTENSIONS else ('mkv', '.mkv')


def _copy_media(settings):
    for node in _get_xml(settings, '/requests/vlm.xml').iter('broadcast'):
        if node.get('name') == COPY_MEDIA:
            return node
    return None


def _copy_started_at(node):
    for option in node.iter('option'):
        name, _, value = (option.text or '').partition('=')
        if name == COPY_SINCE_OPTION and value.isdigit():
            return int(value)
    return None


def _clear_stale_copy(settings):
    node = _copy_media(settings)
    if node is None:
        return
    started = _copy_started_at(node)
    if started is not None and time.time() - started < COPY_STALE_SECONDS:
        return
    _vlm(settings, f'del {COPY_MEDIA}')
    logger.warning("Deleted the VLC copy job a stopped download left behind on %s", settings['url'])


def _claim_copy(settings, uri, output):
    deadline = time.monotonic() + COPY_STALE_SECONDS + COPY_TIMEOUT
    while True:
        try:
            _vlm(settings, (
                f'new {COPY_MEDIA} broadcast enabled input "{uri}" output "{output}" '
                f'option {COPY_SINCE_OPTION}={int(time.time())}'
            ))
            return
        except VlcAPIError as exc:
            if 'already in use' not in str(exc) or time.monotonic() > deadline:
                raise
        _clear_stale_copy(settings)
        time.sleep(COPY_RETRY_SECONDS)


def _wait_copied(settings):
    deadline = time.monotonic() + COPY_TIMEOUT
    while True:
        node = _copy_media(settings)
        if node is None:
            raise VlcAPIError("the VLC copy job was deleted before it finished")
        if not list(node.iter('instance')):
            return
        if time.monotonic() > deadline:
            raise VlcAPIError(f"VLC did not finish copying the track in {COPY_TIMEOUT} s")
        time.sleep(COPY_POLL_SECONDS)


def _queue_leaves(settings):
    pending = [_get_json(settings, '/requests/playlist.json')]
    leaves = []
    while pending:
        node = pending.pop()
        if not isinstance(node, dict):
            continue
        if node.get('type') == 'leaf':
            leaves.append(node)
        pending.extend(node.get('children') or [])
    return leaves


def _open_art(settings, name):
    deadline = time.monotonic() + ART_TIMEOUT
    item = None
    while True:
        if item is None:
            item = next((leaf.get('id') for leaf in _queue_leaves(settings) if leaf.get('name') == name), None)
        if item is not None:
            try:
                return _get(
                    settings, '/art', {'item': item},
                    timeout=(10, DOWNLOAD_READ_TIMEOUT), stream=True,
                )
            except requests.exceptions.HTTPError as exc:
                if getattr(exc.response, 'status_code', None) != 404:
                    raise
        if time.monotonic() > deadline:
            raise VlcAPIError(
                "VLC did not serve the copied track; keep 'Automatically preparse items' enabled in VLC"
            )
        time.sleep(COPY_POLL_SECONDS)


def _drop_queued_copies(settings):
    try:
        for leaf in _queue_leaves(settings):
            if str(leaf.get('name') or '').startswith(COPY_MEDIA + '_'):
                _get(settings, '/requests/status.json', {'command': 'pl_delete', 'id': leaf.get('id')})
    except Exception:
        logger.exception("Could not remove the copied track from VLC's play queue; the next download removes it")


def _release_copy(settings):
    try:
        _vlm(settings, f'del {COPY_MEDIA}')
    except Exception:
        logger.exception("Could not delete the VLC copy job; the next download clears it once it is stale")


def _serve_copy(settings, uri, mux):
    scratch = _scratch_path(settings)
    _claim_copy(settings, uri, f'#std{{access=file,mux={mux},dst=\\"{scratch}\\"}}')
    try:
        _vlm(settings, f'control {COPY_MEDIA} play')
        _wait_copied(settings)
        name = f"{COPY_MEDIA}_{uuid.uuid4().hex[:12]}"
        _get(settings, '/requests/status.json', {
            'command': 'in_enqueue', 'input': folder_uri(scratch), 'name': name,
        })
        try:
            return _open_art(settings, name)
        finally:
            _drop_queued_copies(settings)
    finally:
        _release_copy(settings)


def _download(settings, uri, target):
    partial = target + '.part'
    try:
        response = _serve_copy(settings, uri, _copy_format(uri)[0])
        with response, open(partial, 'wb') as handle:
            for chunk in response.iter_content(chunk_size=65536):
                handle.write(chunk)
        size = os.path.getsize(partial)
        expected = str(response.headers.get('Content-Length') or '')
        if not size:
            raise VlcAPIError("VLC copied no audio")
        if expected.isdigit() and int(expected) != size:
            raise VlcAPIError(f"VLC sent {size} of {expected} bytes")
        os.replace(partial, target)
        return target
    finally:
        if os.path.exists(partial):
            os.remove(partial)


def _probe_download(settings, uri):
    with tempfile.TemporaryDirectory(prefix='audiomuse-vlc-') as folder:
        _download(settings, uri, os.path.join(folder, 'probe' + _copy_format(uri)[1]))


def download_track(temp_dir, item):
    track = item.get('Id')
    uri = item.get('url')
    if not track or not uri:
        logger.error("VLC track %r has no id or file URI, so it cannot be downloaded", item.get('Name'))
        return None
    target = os.path.join(temp_dir, f"{track}{_copy_format(uri)[1]}")
    try:
        _download(_settings(), uri, target)
    except Exception:
        logger.exception("Failed to download '%s' from VLC", item.get('Name', track))
        return None
    logger.info("Downloaded '%s' from VLC to '%s'", item.get('Name', track), target)
    return target


def _queue_uris(settings, item_ids):
    wanted = {str(i) for i in item_ids}
    uris = {}
    for parts, files in _walk(settings):
        for element in files:
            found = track_id(_rel_path(parts, str(element['name'])))
            if found in wanted:
                uris[found] = element['uri']
        if len(uris) == len(wanted):
            break
    return uris


def create_instant_playlist(playlist_name, item_ids, user_creds=None):
    settings = _settings(user_creds)
    name = f"{playlist_name.strip()}_instant"
    try:
        uris = _queue_uris(settings, item_ids)
        queued = 0
        for item in item_ids:
            uri = uris.get(str(item))
            if uri:
                _get(settings, '/requests/status.json', {'command': 'in_enqueue', 'input': uri})
                queued += 1
    except Exception:
        logger.exception("Could not queue playlist '%s' in VLC", name)
        return None
    if not queued:
        logger.error("None of the %d tracks of '%s' were found under the VLC music folder", len(item_ids), name)
        return None
    logger.info("Queued %d of %d tracks of '%s' in VLC's play queue", queued, len(item_ids), name)
    return {'Id': PLAY_QUEUE_ID, 'Name': name}


def _not_stored(playlist_name, item_ids):
    logger.warning(
        "VLC keeps no saved playlists, so '%s' (%d tracks) was not stored on the VLC server",
        playlist_name, len(item_ids or []),
    )
    return {'Id': '', 'Name': playlist_name}


def create_playlist(base_name, item_ids):
    return _not_stored(base_name, item_ids)


def create_or_replace_playlist(playlist_name, item_ids, user_creds=None):
    if not item_ids:
        return None
    return _not_stored(playlist_name, item_ids)


def get_all_playlists():
    return []


def get_playlist_by_name(playlist_name, user_creds=None):
    return None


def delete_playlist(playlist_id):
    return False


def get_playlist_track_ids(playlist_id, user_creds=None):
    return []


def get_top_played_songs(limit, user_creds=None):
    return []


def get_last_played_time(item_id, user_creds=None):
    return None


def get_lyrics(track_id_value, timeout=2.5):
    return None
