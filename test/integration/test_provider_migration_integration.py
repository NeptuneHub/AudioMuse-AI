# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Provider-migration tests against a real Postgres database.

Drives tasks.provider_migration_matcher and provider_migration_tasks over a
live database to migrate item ids between providers and rewrite the stored
segmented id map end to end.

Main Features:
* Real cross-provider migration for each source/target pairing.
* Segmented id-map rewrite and relabel-overflow soft-failure handling.
"""

import importlib.util
import json
import os
import re
import sys
import tempfile
from urllib.parse import quote
from unittest.mock import MagicMock

import pytest

try:
    import psycopg2
except Exception:  # pragma: no cover - psycopg2 is in test/requirements.txt
    psycopg2 = None


def _load_module(mod_name, *rel_parts):
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    repo_root = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
    )
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    mod_path = os.path.join(repo_root, *rel_parts)
    spec = importlib.util.spec_from_file_location(mod_name, mod_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


matcher = _load_module('tasks.provider_migration_matcher', 'tasks', 'provider_migration_matcher.py')
mig = _load_module('tasks.provider_migration_tasks', 'tasks', 'provider_migration_tasks.py')


PROVIDERS = ('jellyfin', 'emby', 'navidrome', 'lyrion', 'plex')

_ID_BASE = {
    'jellyfin': 0x10000,
    'emby': 5000,
    'navidrome': 0xABCD00,
    'lyrion': 90000,
    'plex': 70000,
}

_CROSS_TARGET_SHIFT = 1_000_000

_ORPHAN_OFFSET = 900

_EXPECTED_CONFIG_KEYS = {
    'jellyfin': ['JELLYFIN_URL', 'JELLYFIN_USER_ID', 'JELLYFIN_TOKEN'],
    'emby': ['EMBY_URL', 'EMBY_USER_ID', 'EMBY_TOKEN'],
    'navidrome': ['NAVIDROME_URL', 'NAVIDROME_USER', 'NAVIDROME_PASSWORD'],
    'lyrion': ['LYRION_URL'],
    'plex': ['PLEX_URL', 'PLEX_TOKEN'],
}

_TARGET_CREDS = {
    'jellyfin': {'url': 'http://jf.test:8096', 'user_id': 'jfuser', 'token': 'jftoken'},
    'emby': {'url': 'http://emby.test:8096', 'user_id': 'embyuser', 'token': 'embytoken'},
    'navidrome': {'url': 'http://nav.test:4533', 'user': 'navuser', 'password': 'navpass'},
    'lyrion': {'url': 'http://lms.test:9000'},
    'plex': {'url': 'http://plex.test:32400', 'token': 'plextoken'},
}


SHARED_TRACKS = [
    {
        'artist': 'Daft Punk',
        'album': 'Discovery',
        'album_artist': 'Daft Punk',
        'title': 'One More Time',
        'disc': 1,
        'track': 1,
        'ext': 'flac',
    },
    {
        'artist': 'Green Day',
        'album': 'American Idiot',
        'album_artist': 'Green Day',
        'title': 'Boulevard of Broken Dreams',
        'disc': 1,
        'track': 4,
        'ext': 'flac',
    },
    {
        'artist': 'Eagles',
        'album': 'Ultimate Rock Hits',
        'album_artist': 'Various Artists',
        'title': 'Hotel California',
        'disc': 1,
        'track': 3,
        'ext': 'mp3',
    },
]

ORPHAN_TRACK = {
    'artist': 'Nobody',
    'album': 'Orphan Album',
    'album_artist': 'Nobody',
    'title': 'Orphan Track',
    'disc': 1,
    'track': 1,
    'ext': 'flac',
}


def _relative_path(track):
    folder_artist = track['album_artist'] or track['artist']
    filename = f"{track['disc']}-{track['track']:02d} - {track['title']}.{track['ext']}"
    return f"{folder_artist}/{track['album']}/{filename}"


def _provider_id(provider, shift, index):
    n = _ID_BASE[provider] + shift + index
    if provider == 'jellyfin':
        return format(n, '032x')
    if provider == 'navidrome':
        return format(n, '016x')
    return str(n)


def _provider_path(provider, rel):
    if provider == 'jellyfin':
        return '/media/music/MyTunes/' + rel
    if provider == 'emby':
        return '/mnt/media/MyTunes/' + rel
    if provider == 'lyrion':
        return 'file:///media/music/MyTunes/' + quote(rel)
    if provider == 'plex':
        return '/data/music/MyTunes/' + rel
    return rel


_SCHEMA_DDL = [
    "CREATE TABLE score (item_id TEXT PRIMARY KEY, title TEXT, author TEXT, "
    "album TEXT, album_artist TEXT, tempo REAL, key TEXT, scale TEXT, "
    "mood_vector TEXT, energy REAL, other_features TEXT, year INTEGER, "
    "rating INTEGER, file_path TEXT)",
    "CREATE TABLE playlist (id SERIAL PRIMARY KEY, playlist_name TEXT, "
    "item_id TEXT, title TEXT, author TEXT, UNIQUE (playlist_name, item_id))",
    "CREATE TABLE embedding (item_id TEXT PRIMARY KEY, embedding BYTEA, "
    "FOREIGN KEY (item_id) REFERENCES score (item_id) ON DELETE CASCADE)",
    "CREATE TABLE clap_embedding (item_id TEXT PRIMARY KEY, embedding BYTEA, "
    "FOREIGN KEY (item_id) REFERENCES score (item_id) ON DELETE CASCADE)",
    "CREATE TABLE lyrics_embedding (item_id TEXT PRIMARY KEY, embedding BYTEA, "
    "axis_vector BYTEA, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
    "FOREIGN KEY (item_id) REFERENCES score (item_id) ON DELETE CASCADE)",
    "CREATE TABLE voyager_index_data (index_name VARCHAR(255) PRIMARY KEY, "
    "index_data BYTEA NOT NULL, id_map_json TEXT NOT NULL, "
    "embedding_dimension INTEGER NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)",
    "CREATE TABLE map_projection_data (index_name VARCHAR(255) PRIMARY KEY, "
    "projection_data BYTEA NOT NULL, id_map_json TEXT NOT NULL, "
    "embedding_dimension INTEGER NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)",
    "CREATE TABLE artist_index_data (index_name VARCHAR(255) PRIMARY KEY, "
    "index_data BYTEA NOT NULL, artist_map_json TEXT NOT NULL, "
    "gmm_params_json TEXT NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)",
    "CREATE TABLE artist_metadata_data (name VARCHAR(255) PRIMARY KEY, "
    "blob_data BYTEA NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)",
    "CREATE TABLE artist_component_projection (index_name VARCHAR(255) PRIMARY KEY, "
    "projection_data BYTEA NOT NULL, artist_component_map_json TEXT NOT NULL, "
    "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)",
    "CREATE TABLE artist_mapping (artist_name TEXT PRIMARY KEY, artist_id TEXT)",
    "CREATE TABLE app_config (key TEXT PRIMARY KEY, value TEXT NOT NULL, "
    "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)",
    "CREATE TABLE migration_session (id SERIAL PRIMARY KEY, "
    "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, completed_at TIMESTAMP, "
    "status TEXT NOT NULL DEFAULT 'in_progress', source_type TEXT NOT NULL, "
    "target_type TEXT NOT NULL, target_creds TEXT NOT NULL, "
    "state JSONB NOT NULL DEFAULT '{}')",
    "CREATE TABLE migration_target_meta (session_id INTEGER NOT NULL "
    "REFERENCES migration_session(id) ON DELETE CASCADE, new_id TEXT NOT NULL, "
    "path TEXT, title TEXT, artist TEXT, album TEXT, album_artist TEXT, "
    "year INTEGER, PRIMARY KEY (session_id, new_id))",
    "CREATE TABLE music_servers (server_id TEXT PRIMARY KEY, name TEXT, "
    "server_type TEXT, creds JSONB DEFAULT '{}', music_libraries TEXT DEFAULT '', "
    "is_default BOOLEAN NOT NULL DEFAULT FALSE, track_count INTEGER, "
    "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)",
    "CREATE UNIQUE INDEX idx_music_servers_single_default "
    "ON music_servers (is_default) WHERE is_default",
    "CREATE TABLE track_server_map ("
    "item_id TEXT NOT NULL REFERENCES score (item_id) ON UPDATE CASCADE ON DELETE CASCADE, "
    "server_id TEXT NOT NULL REFERENCES music_servers (server_id) ON DELETE CASCADE, "
    "provider_track_id TEXT NOT NULL, match_tier TEXT, file_path TEXT, "
    "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY (item_id, server_id))",
    "CREATE UNIQUE INDEX idx_track_server_map_provider_unique "
    "ON track_server_map (server_id, provider_track_id)",
    "CREATE TABLE artist_server_map (artist_name TEXT NOT NULL, "
    "server_id TEXT NOT NULL REFERENCES music_servers (server_id) ON DELETE CASCADE, "
    "provider_artist_id TEXT NOT NULL, "
    "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
    "PRIMARY KEY (artist_name, server_id), UNIQUE (server_id, provider_artist_id))",
]

_DEFAULT_SERVER_ID = 'srv-default'


@pytest.fixture(scope='session')
def pg_dsn():
    if psycopg2 is None:
        pytest.skip("psycopg2 not importable")

    dsn = os.environ.get('AUDIOMUSE_TEST_DATABASE_URL')
    if dsn:
        try:
            psycopg2.connect(dsn).close()
        except Exception as e:
            pytest.skip(f"AUDIOMUSE_TEST_DATABASE_URL not reachable: {e}")
        yield dsn
        return

    try:
        import pgserver
    except Exception:
        pytest.skip(
            "No test database. Set AUDIOMUSE_TEST_DATABASE_URL to a disposable "
            "DB, or `pip install pgserver` for an ephemeral local instance."
        )

    data_dir = tempfile.mkdtemp(prefix='audiomuse_pg_')
    server = pgserver.get_server(data_dir)
    try:
        yield server.get_uri()
    finally:
        server.cleanup()


@pytest.fixture
def migration_db(pg_dsn):
    setup = psycopg2.connect(pg_dsn)
    setup.autocommit = True
    with setup.cursor() as cur:
        cur.execute("DROP SCHEMA IF EXISTS public CASCADE")
        cur.execute("CREATE SCHEMA public")
        for ddl in _SCHEMA_DDL:
            cur.execute(ddl)
    setup.close()

    opened = []

    def _connect():
        conn = psycopg2.connect(pg_dsn)
        opened.append(conn)
        return conn

    mig._get_dedicated_conn = _connect
    mig._get_redis = lambda: MagicMock()
    mig._drain_workers_or_timeout = lambda *a, **k: None
    mig._post_commit_reload = lambda *a, **k: None

    yield {'dsn': pg_dsn, 'connect': _connect}

    for conn in opened:
        try:
            conn.close()
        except Exception:
            pass


def _insert_segmented_index(cur, table, binary_col, base, id_map_json, dim, n_parts=3):
    step = max(1, -(-len(id_map_json) // n_parts))
    frags = [id_map_json[i : i + step] for i in range(0, len(id_map_json), step)]
    while len(frags) < n_parts:
        frags.append('')
    for k in range(1, n_parts + 1):
        cur.execute(
            f"INSERT INTO {table} (index_name, {binary_col}, id_map_json, "
            f"embedding_dimension) VALUES (%s, %s, %s, %s)",
            (
                f"{base}_{k}_{n_parts}",
                psycopg2.Binary(b'\x00' if k == 1 else b''),
                frags[k - 1],
                dim,
            ),
        )


def _reassemble_id_map(parts):
    ordered = sorted(parts, key=lambda p: int(re.match(r'^.*_(\d+)_\d+$', p[0]).group(1)))
    return ''.join((frag or '') for _, frag in ordered)


def _seed_library(conn, source_rendered, segmented=False, source_type='jellyfin'):
    """Seed a canonicalized, single-server install.

    The path lives on the SERVER's map row, never on the shared score row: score
    is the union catalogue and a path belongs to a file on a server. Seeding it
    the old way (score.file_path) would test a schema that no longer exists.
    """
    src_ids = [r['id'] for r in source_rendered]
    ivf_map = json.dumps({str(i): sid for i, sid in enumerate(src_ids)})
    projection_map = json.dumps(src_ids)
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO music_servers (server_id, name, server_type, is_default) "
            "VALUES (%s, %s, %s, TRUE)",
            (_DEFAULT_SERVER_ID, 'Default', source_type),
        )
        for index, r in enumerate(source_rendered):
            cur.execute(
                "INSERT INTO score (item_id, title, author, album, album_artist, year) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (
                    r['id'],
                    r['title'],
                    r['artist'],
                    r['album'],
                    r['album_artist'],
                    1900 + index,
                ),
            )
            cur.execute(
                "INSERT INTO track_server_map "
                "(item_id, server_id, provider_track_id, match_tier, file_path) "
                "VALUES (%s, %s, %s, %s, %s)",
                (r['id'], _DEFAULT_SERVER_ID, r['id'], 'default', r['path']),
            )
            for table in ('embedding', 'clap_embedding', 'lyrics_embedding'):
                cur.execute(f"INSERT INTO {table} (item_id) VALUES (%s)", (r['id'],))
        if segmented:
            _insert_segmented_index(
                cur, 'voyager_index_data', 'index_data', 'ivf_main', ivf_map, 128
            )
            _insert_segmented_index(
                cur, 'map_projection_data', 'projection_data', 'map_main', projection_map, 2
            )
        else:
            cur.execute(
                "INSERT INTO voyager_index_data (index_name, index_data, id_map_json, "
                "embedding_dimension) VALUES (%s, %s, %s, %s)",
                ('ivf_main', psycopg2.Binary(b'\x00'), ivf_map, 128),
            )
            cur.execute(
                "INSERT INTO map_projection_data (index_name, projection_data, "
                "id_map_json, embedding_dimension) VALUES (%s, %s, %s, %s)",
                ('map_main', psycopg2.Binary(b'\x00'), projection_map, 2),
            )
        cur.execute(
            "INSERT INTO artist_index_data (index_name, index_data, artist_map_json, "
            "gmm_params_json) VALUES (%s, %s, %s, %s)",
            ('artist_main', psycopg2.Binary(b'\x00'), '{}', '{}'),
        )
        cur.execute(
            "INSERT INTO artist_metadata_data (name, blob_data) VALUES (%s, %s)",
            ('artist_main', psycopg2.Binary(b'\x00')),
        )
        cur.execute(
            "INSERT INTO artist_component_projection (index_name, projection_data, "
            "artist_component_map_json) VALUES (%s, %s, %s)",
            ('artist_main', psycopg2.Binary(b'\x00'), '{}'),
        )
        cur.execute(
            "INSERT INTO artist_mapping (artist_name, artist_id) VALUES (%s, %s)",
            ('Daft Punk', 'old-artist-id'),
        )
    conn.commit()


def _insert_session(conn, source, target, matches, new_meta):
    state = {
        'dry_run': {'matches': matches},
        'manual_matches': {},
        'manual_unmatches': [],
        'selected_libraries': None,
    }
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO migration_session (source_type, target_type, target_creds, "
            "state, status) VALUES (%s, %s, %s, %s, 'dry_run_ready') RETURNING id",
            (source, target, json.dumps(_TARGET_CREDS[target]), json.dumps(state)),
        )
        session_id = cur.fetchone()[0]
        for new_id, meta in (new_meta or {}).items():
            cur.execute(
                "INSERT INTO migration_target_meta (session_id, new_id, path, title, "
                "artist, album, album_artist, year) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    session_id,
                    new_id,
                    meta.get('path'),
                    meta.get('title'),
                    meta.get('artist'),
                    meta.get('album'),
                    meta.get('album_artist'),
                    meta.get('year'),
                ),
            )
    conn.commit()
    return session_id


@pytest.mark.integration
@pytest.mark.parametrize('target', PROVIDERS)
@pytest.mark.parametrize('source', PROVIDERS)
def test_real_provider_migration(source, target, migration_db):
    self_migration = source == target
    target_shift = 1 if self_migration else _CROSS_TARGET_SHIFT

    source_rendered = []
    for index, track in enumerate(SHARED_TRACKS):
        rel = _relative_path(track)
        source_rendered.append(
            {
                'id': _provider_id(source, 0, index),
                'path': _provider_path(source, rel),
                'title': track['title'],
                'artist': track['artist'],
                'album': track['album'],
                'album_artist': track['album_artist'],
            }
        )
    orphan_rel = _relative_path(ORPHAN_TRACK)
    source_rendered.append(
        {
            'id': _provider_id(source, 0, _ORPHAN_OFFSET),
            'path': _provider_path(source, orphan_rel),
            'title': ORPHAN_TRACK['title'],
            'artist': ORPHAN_TRACK['artist'],
            'album': ORPHAN_TRACK['album'],
            'album_artist': ORPHAN_TRACK['album_artist'],
        }
    )
    target_rendered = []
    for index, track in enumerate(SHARED_TRACKS):
        rel = _relative_path(track)
        target_rendered.append(
            {
                'id': _provider_id(target, target_shift, index),
                'path': _provider_path(target, rel),
                'title': track['title'],
                'artist': track['artist'],
                'album': track['album'],
                'album_artist': track['album_artist'],
            }
        )

    old_rows = [
        {
            'item_id': r['id'],
            'file_path': r['path'],
            'title': r['title'],
            'author': r['artist'],
            'album': r['album'],
            'album_artist': r['album_artist'],
        }
        for r in source_rendered
    ]
    new_tracks = [
        {
            'id': r['id'],
            'path': r['path'],
            'title': r['title'],
            'artist': r['artist'],
            'album': r['album'],
            'album_artist': r['album_artist'],
        }
        for r in target_rendered
    ]

    match_result = matcher.match_tracks(old_rows, new_tracks)
    matches = match_result['matches']

    orphan_id = source_rendered[-1]['id']
    expected_map = {
        source_rendered[i]['id']: target_rendered[i]['id'] for i in range(len(SHARED_TRACKS))
    }
    expected_map_inv = {new: old for old, new in expected_map.items()}
    assert matches == expected_map, (
        f"{source}->{target}: matcher mapping wrong\n  expected {expected_map}\n  got {matches}"
    )
    assert orphan_id not in matches, "the source-only track must stay unmatched"

    new_meta = {
        r['id']: {
            'path': r['path'],
            'title': r['title'],
            'artist': r['artist'],
            'album': r['album'],
            'album_artist': r['album_artist'],
            'year': 2000 + i,
        }
        for i, r in enumerate(target_rendered)
    }

    conn = migration_db['connect']()
    _seed_library(conn, source_rendered)
    session_id = _insert_session(conn, source, target, matches, new_meta)

    print(f"\n=== Migrating {source} -> {target} (session {session_id}) ===")
    print(f"  source ids: {[r['id'] for r in source_rendered]}")
    print(f"  target ids: {[r['id'] for r in target_rendered]}")
    if self_migration:
        overlap = set(expected_map.values()) & set(r['id'] for r in source_rendered)
        print(f"  self-migration id overlap (exercises two-pass rewrite): {sorted(overlap)}")
        assert overlap, "self-migration should produce overlapping new/old ids"

    result = mig.execute_provider_migration(session_id)
    assert result['ok'] is True
    assert result['matched'] == len(SHARED_TRACKS)

    new_provider_ids = set(expected_map.values())
    catalogue_ids = {r['id'] for r in source_rendered}
    matched_item_ids = set(expected_map.keys())
    verify = migration_db['connect']()
    with verify.cursor() as cur:
        # THE CATALOGUE IS NEVER TOUCHED. item_id is the fp_2 hash of the audio, so a
        # provider swap cannot change it and must never delete it: the analysis behind
        # it is expensive and irreplaceable. Only the MAPPING moves.
        cur.execute("SELECT item_id, file_path, title, album_artist, year FROM score")
        score = {row[0]: row for row in cur.fetchall()}
        assert set(score.keys()) == catalogue_ids, (
            f"{source}->{target}: score item_ids must NOT move\n"
            f"  want {catalogue_ids}\n  got {set(score.keys())}"
        )
        assert orphan_id in score, "the orphan's catalogue row and analysis must survive"
        assert all(row[1] is None for row in score.values()), (
            "no path may ever be written to the shared score row"
        )
        for i, r in enumerate(target_rendered):
            row = score[expected_map_inv[r['id']]]
            assert row[2] == r['title']
            assert row[3] == r['album_artist']
            assert row[4] == 2000 + i, "year not refreshed from new_meta"

        # The mapping is what migrated: matched songs are now reachable by the TARGET
        # provider's id, and the unmatched one is unbound from the server entirely.
        cur.execute(
            "SELECT item_id, provider_track_id, file_path FROM track_server_map "
            "WHERE server_id = %s",
            (_DEFAULT_SERVER_ID,),
        )
        rows = cur.fetchall()
        assert {r[0] for r in rows} == matched_item_ids, (
            "only matched songs stay bound to the migrated server"
        )
        assert {r[1] for r in rows} == new_provider_ids, (
            f"{source}->{target}: provider ids not repointed at the target"
        )
        by_item = {r[0]: r for r in rows}
        for old_id, new_id in expected_map.items():
            assert by_item[old_id][1] == new_id
            assert by_item[old_id][2] == new_meta[new_id]['path'], (
                "the target's path must land on the server's own map row"
            )

        # Embeddings hang off score, so if the catalogue survived, so did they.
        for table in ('embedding', 'clap_embedding', 'lyrics_embedding'):
            cur.execute(f"SELECT item_id FROM {table}")
            ids = {row[0] for row in cur.fetchall()}
            assert ids == catalogue_ids, (
                f"{source}->{target}: {table} must survive a provider swap intact"
            )

        # No item_id moved, so every similarity index still points at the right songs
        # and needs no rebuild. This is the whole reason the canonical id exists.
        assert result['index_rebuild_needed'] is False
        cur.execute("SELECT id_map_json FROM voyager_index_data WHERE index_name = 'ivf_main'")
        ivf_map = json.loads(cur.fetchone()[0])
        assert set(ivf_map.values()) == catalogue_ids, "the IVF id map must be untouched"

        cur.execute("SELECT id_map_json FROM map_projection_data WHERE index_name = 'map_main'")
        proj_map = json.loads(cur.fetchone()[0])
        assert proj_map == [r['id'] for r in source_rendered], (
            "the projection id map must be untouched"
        )

        # Artist IDs belong to the old provider and cannot be repointed (the matcher
        # produces an id per TRACK), so they are dropped and the next analysis rebuilds
        # them. Artist ANALYSIS is keyed by artist NAME, which a provider swap does not
        # change, so the artist indexes survive untouched like the track ones.
        for table in ('artist_mapping', 'artist_server_map'):
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            assert cur.fetchone()[0] == 0, f"{table} holds dead provider ids; must be cleared"
        for table in (
            'artist_index_data',
            'artist_metadata_data',
            'artist_component_projection',
        ):
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            assert cur.fetchone()[0] == 1, (
                f"{table} is keyed by artist NAME and must survive a provider swap"
            )

        cur.execute("SELECT key, value FROM app_config")
        config_rows = dict(cur.fetchall())
        assert 'MEDIASERVER_TYPE' not in config_rows, (
            "media keys live in the registry only; app_config must be purged"
        )

        cur.execute(
            "SELECT server_type, music_libraries FROM music_servers WHERE is_default"
        )
        server_row = cur.fetchone()
        assert server_row[0] == target, "the default registry row must point at the target"
        assert not server_row[1], "null selection should clear music_libraries"

        cur.execute("SELECT status FROM migration_session WHERE id = %s", (session_id,))
        assert cur.fetchone()[0] == 'completed'
    verify.close()
    print(
        f"  ok: {len(matched_item_ids)} mappings repointed, orphan unbound but kept, "
        f"catalogue + indexes intact, default server -> {target}"
    )


@pytest.mark.integration
def test_real_provider_migration_rewrites_segmented_id_map(migration_db):
    source, target = 'jellyfin', 'navidrome'

    source_rendered = []
    for index, track in enumerate(SHARED_TRACKS):
        rel = _relative_path(track)
        source_rendered.append(
            {
                'id': _provider_id(source, 0, index),
                'path': _provider_path(source, rel),
                'title': track['title'],
                'artist': track['artist'],
                'album': track['album'],
                'album_artist': track['album_artist'],
            }
        )
    source_rendered.append(
        {
            'id': _provider_id(source, 0, _ORPHAN_OFFSET),
            'path': _provider_path(source, _relative_path(ORPHAN_TRACK)),
            'title': ORPHAN_TRACK['title'],
            'artist': ORPHAN_TRACK['artist'],
            'album': ORPHAN_TRACK['album'],
            'album_artist': ORPHAN_TRACK['album_artist'],
        }
    )
    target_rendered = [
        {
            'id': _provider_id(target, _CROSS_TARGET_SHIFT, index),
            'path': _provider_path(target, _relative_path(track)),
            'title': track['title'],
            'artist': track['artist'],
            'album': track['album'],
            'album_artist': track['album_artist'],
        }
        for index, track in enumerate(SHARED_TRACKS)
    ]

    old_rows = [
        {
            'item_id': r['id'],
            'file_path': r['path'],
            'title': r['title'],
            'author': r['artist'],
            'album': r['album'],
            'album_artist': r['album_artist'],
        }
        for r in source_rendered
    ]
    new_tracks = [
        {
            'id': r['id'],
            'path': r['path'],
            'title': r['title'],
            'artist': r['artist'],
            'album': r['album'],
            'album_artist': r['album_artist'],
        }
        for r in target_rendered
    ]
    matches = matcher.match_tracks(old_rows, new_tracks)['matches']
    expected_map = {
        source_rendered[i]['id']: target_rendered[i]['id'] for i in range(len(SHARED_TRACKS))
    }
    assert matches == expected_map
    orphan_id = source_rendered[-1]['id']
    new_meta = {
        r['id']: {
            'path': r['path'],
            'title': r['title'],
            'artist': r['artist'],
            'album': r['album'],
            'album_artist': r['album_artist'],
            'year': 2000 + i,
        }
        for i, r in enumerate(target_rendered)
    }

    conn = migration_db['connect']()
    _seed_library(conn, source_rendered, segmented=True)
    session_id = _insert_session(conn, source, target, matches, new_meta)

    result = mig.execute_provider_migration(session_id)
    assert result['ok'] is True
    assert result['matched'] == len(SHARED_TRACKS)

    catalogue_ids = {r['id'] for r in source_rendered}
    verify = migration_db['connect']()
    with verify.cursor() as cur:
        assert result['index_rebuild_needed'] is False

        cur.execute("SELECT index_name, id_map_json FROM voyager_index_data")
        vparts = [(n, j) for n, j in cur.fetchall() if re.match(r'^ivf_main_\d+_\d+$', n)]
        assert len(vparts) >= 2, "ivf index must actually be segmented in this test"
        ivf_map = json.loads(_reassemble_id_map(vparts))
        assert set(ivf_map.values()) == catalogue_ids, (
            "a segmented IVF id map must survive a provider swap byte-for-byte"
        )
        assert orphan_id in ivf_map.values(), (
            "the unbound song keeps its analysis and its index slot"
        )

        cur.execute("SELECT index_name, id_map_json FROM map_projection_data")
        pparts = [(n, j) for n, j in cur.fetchall() if re.match(r'^map_main_\d+_\d+$', n)]
        assert len(pparts) >= 2, "map projection must actually be segmented in this test"
        proj_map = json.loads(_reassemble_id_map(pparts))
        assert proj_map == [r['id'] for r in source_rendered], (
            "a segmented projection id map must survive a provider swap untouched"
        )
    verify.close()
    print(
        f"  ok (segmented): {len(vparts)} ivf parts reassembled, id maps untouched -> {target}"
    )


def test_segmented_id_map_relabel_overflow_is_soft_failure(migration_db):
    import config

    source, target = 'jellyfin', 'navidrome'

    source_rendered = []
    for index, track in enumerate(SHARED_TRACKS):
        source_rendered.append(
            {
                'id': _provider_id(source, 0, index),
                'path': _provider_path(source, _relative_path(track)),
                'title': track['title'],
                'artist': track['artist'],
                'album': track['album'],
                'album_artist': track['album_artist'],
            }
        )
    source_rendered.append(
        {
            'id': _provider_id(source, 0, _ORPHAN_OFFSET),
            'path': _provider_path(source, _relative_path(ORPHAN_TRACK)),
            'title': ORPHAN_TRACK['title'],
            'artist': ORPHAN_TRACK['artist'],
            'album': ORPHAN_TRACK['album'],
            'album_artist': ORPHAN_TRACK['album_artist'],
        }
    )
    target_rendered = [
        {
            'id': _provider_id(target, _CROSS_TARGET_SHIFT, index),
            'path': _provider_path(target, _relative_path(track)),
            'title': track['title'],
            'artist': track['artist'],
            'album': track['album'],
            'album_artist': track['album_artist'],
        }
        for index, track in enumerate(SHARED_TRACKS)
    ]

    old_rows = [
        {
            'item_id': r['id'],
            'file_path': r['path'],
            'title': r['title'],
            'author': r['artist'],
            'album': r['album'],
            'album_artist': r['album_artist'],
        }
        for r in source_rendered
    ]
    new_tracks = [
        {
            'id': r['id'],
            'path': r['path'],
            'title': r['title'],
            'artist': r['artist'],
            'album': r['album'],
            'album_artist': r['album_artist'],
        }
        for r in target_rendered
    ]
    matches = matcher.match_tracks(old_rows, new_tracks)['matches']
    new_meta = {
        r['id']: {
            'path': r['path'],
            'title': r['title'],
            'artist': r['artist'],
            'album': r['album'],
            'album_artist': r['album_artist'],
            'year': 2000 + i,
        }
        for i, r in enumerate(target_rendered)
    }

    conn = migration_db['connect']()
    _seed_library(conn, source_rendered, segmented=True)
    session_id = _insert_session(conn, source, target, matches, new_meta)

    # IVF_MAX_PART_SIZE_MB=0 used to make the id-map REWRITE overflow, forcing the
    # index to be dropped and a full rebuild flagged. There is no rewrite any more, so
    # there is nothing to overflow: the part-size limit is irrelevant to a migration.
    # This guards against ever reintroducing an item_id rewrite here, which would take
    # the embeddings with it and cost a full rebuild on every provider swap.
    saved_max_part = config.IVF_MAX_PART_SIZE_MB
    config.IVF_MAX_PART_SIZE_MB = 0
    try:
        result = mig.execute_provider_migration(session_id)
    finally:
        config.IVF_MAX_PART_SIZE_MB = saved_max_part

    assert result['ok'] is True
    assert result['matched'] == len(SHARED_TRACKS)
    assert result['index_rebuild_needed'] is False, (
        "a provider swap moves no item_id, so it can never need an index rebuild"
    )

    catalogue_ids = {r['id'] for r in source_rendered}
    verify = migration_db['connect']()
    with verify.cursor() as cur:
        cur.execute("SELECT count(*) FROM voyager_index_data")
        assert cur.fetchone()[0] > 0, "the ivf index must NOT be dropped by a migration"
        cur.execute("SELECT count(*) FROM map_projection_data")
        assert cur.fetchone()[0] > 0, "the map projection must NOT be dropped by a migration"

        cur.execute("SELECT item_id FROM score")
        score_ids = {row[0] for row in cur.fetchall()}
        assert score_ids == catalogue_ids, "the catalogue is never touched by a migration"

        cur.execute("SELECT status FROM migration_session WHERE id = %s", (session_id,))
        assert cur.fetchone()[0] == 'completed'
    verify.close()
    print(
        f"  ok: part-size limit is moot, indexes kept, {len(score_ids)} catalogue rows "
        f"intact -> {target}"
    )
