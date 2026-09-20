# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Album Creation end to end against a real Postgres and a real similarity index.

Seeds three sonic families of songs, builds the real disk-paged audio index over
them and runs tasks.album_creation_manager with nothing mocked but get_db, so the
SQL, the index queries, the per-server availability mask and the selection all
run for real.

Main Features:
* A song seed gives a full, duplicate-free album that keeps the seed, honours the
  per-artist cap and stays inside the seed's own sonic family
* An album seed leaves its own tracks out; an artist seed features that artist
  beyond the cap while every other artist stays capped
* Bound to a secondary server, every track of the album exists on that server
* The weekly seed sample only offers songs of the bound server
* The album search groups by album and album artist on the selected server
"""

import os
import sys

import numpy as np
import pytest

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    import psycopg2
except Exception:  # pragma: no cover - psycopg2 is in test/requirements.txt
    psycopg2 = None

import config

pytestmark = pytest.mark.integration

_DIM = config.EMBEDDING_DIMENSION
_FAMILIES = 3
_PER_FAMILY = 70
_CLAP_DIM = config.CLAP_EMBEDDING_DIMENSION
_LYRIC_DIM = config.LYRICS_EMBEDDING_DIMENSION
_HIDDEN = 8
_TABLES = ("track_server_map, music_servers, lyrics_embedding, clap_embedding, embedding, "
           "score, ivf_cell, ivf_dir")

_SCHEMA = (
    "CREATE TABLE score (item_id TEXT PRIMARY KEY, title TEXT, author TEXT, "
    "album TEXT, album_artist TEXT, tempo REAL, key TEXT, scale TEXT, "
    "mood_vector TEXT, energy REAL, other_features TEXT, year INTEGER, "
    "rating INTEGER, file_path TEXT, duration DOUBLE PRECISION, search_u TEXT)",
    "CREATE TABLE embedding (item_id TEXT PRIMARY KEY REFERENCES score (item_id) "
    "ON DELETE CASCADE, embedding BYTEA)",
    "CREATE TABLE lyrics_embedding (item_id TEXT PRIMARY KEY REFERENCES score (item_id) "
    "ON DELETE CASCADE, embedding BYTEA, axis_vector BYTEA)",
    "CREATE TABLE clap_embedding (item_id TEXT PRIMARY KEY REFERENCES score (item_id) "
    "ON DELETE CASCADE, embedding BYTEA)",
    "CREATE TABLE music_servers (server_id TEXT PRIMARY KEY, name TEXT, "
    "server_type TEXT, creds JSONB DEFAULT '{}', music_libraries TEXT DEFAULT '', "
    "is_default BOOLEAN NOT NULL DEFAULT FALSE, track_count INTEGER, "
    "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
    "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)",
    "CREATE TABLE track_server_map ("
    "item_id TEXT NOT NULL REFERENCES score (item_id) ON UPDATE CASCADE ON DELETE CASCADE, "
    "server_id TEXT NOT NULL REFERENCES music_servers (server_id) ON DELETE CASCADE, "
    "provider_track_id TEXT NOT NULL, match_tier TEXT, file_path TEXT, "
    "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
    "PRIMARY KEY (server_id, provider_track_id))",
    "CREATE TABLE ivf_dir (name VARCHAR(255) PRIMARY KEY, blob_data BYTEA NOT NULL, "
    "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)",
    "CREATE TABLE ivf_cell (index_name VARCHAR(255) NOT NULL, cell_id INTEGER NOT NULL, "
    "cell_data BYTEA NOT NULL, PRIMARY KEY (index_name, cell_id))",
)


def _family_of(item_id):
    return int(item_id.split('_')[1])


def _add_lyrics(cur, item_id, subject, axis_rng):
    from lyrics import axis_columns

    axes = axis_rng.random(len(axis_columns())).astype(np.float32)
    cur.execute(
        "INSERT INTO lyrics_embedding (item_id, embedding, axis_vector) VALUES (%s, %s, %s)",
        (item_id, psycopg2.Binary(subject.astype(np.float32).tobytes()),
         psycopg2.Binary(axes.tobytes())),
    )


def _add_track(cur, item_id, title, author, album, vector, clap, year, duration, servers):
    cur.execute(
        "INSERT INTO score (item_id, title, author, album, album_artist, tempo, energy, "
        "mood_vector, other_features, year, duration, search_u) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
        (item_id, title, author, album, author, 110.0, 0.5,
         'rock:0.8,pop:0.4', 'danceable:0.6,aggressive:0.6,happy:0.6,party:0.6,relaxed:0.6,sad:0.6',
         year, duration, f"{title} {author} {album}".lower()),
    )
    cur.execute(
        "INSERT INTO embedding (item_id, embedding) VALUES (%s, %s)",
        (item_id, psycopg2.Binary(vector.astype(np.float32).tobytes())),
    )
    cur.execute(
        "INSERT INTO clap_embedding (item_id, embedding) VALUES (%s, %s)",
        (item_id, psycopg2.Binary(clap.astype(np.float32).tobytes())),
    )
    for server in servers:
        cur.execute(
            "INSERT INTO track_server_map (item_id, server_id, provider_track_id, match_tier) "
            "VALUES (%s, %s, %s, 'fingerprint')",
            (item_id, server, f"{server}-{item_id}"),
        )


def _seed_library(cur):
    rng = np.random.default_rng(17)
    centres = np.eye(_DIM, dtype=np.float32)[:_FAMILIES] * 4.0
    clap_centres = np.eye(_CLAP_DIM, dtype=np.float32)[:_FAMILIES] * 4.0
    basis = np.eye(_LYRIC_DIM, dtype=np.float32)
    cur.execute(
        "INSERT INTO music_servers (server_id, name, server_type, is_default) VALUES "
        "('srv-a', 'Main', 'navidrome', TRUE), ('srv-b', 'Second', 'navidrome', FALSE)"
    )
    for number in range(_HIDDEN):
        _add_track(
            cur, f"fp_9_{number:03d}", f"Hidden {number}", f"Hidden Artist {number}", "Hidden Album",
            centres[0] + 0.9 * rng.standard_normal(_DIM).astype(np.float32),
            clap_centres[1] + 0.2 * rng.standard_normal(_CLAP_DIM).astype(np.float32),
            1991, 200.0, ('srv-a',) if number < _HIDDEN - 2 else ('srv-b',),
        )
        _add_lyrics(
            cur, f"fp_9_{number:03d}",
            4.0 * basis[1] + 3.0 * basis[_FAMILIES] + 0.2 * rng.standard_normal(_LYRIC_DIM).astype(np.float32),
            rng,
        )
    for family in range(_FAMILIES):
        for number in range(_PER_FAMILY):
            item_id = f"fp_{family}_{number:03d}"
            author = f"Artist {family}-{number % 14}"
            album = f"Album {family}-{number % 7}"
            title = f"Song {family} {number}"
            vector = centres[family] + 0.9 * rng.standard_normal(_DIM).astype(np.float32)
            cur.execute(
                "INSERT INTO score (item_id, title, author, album, album_artist, tempo, energy, "
                "mood_vector, other_features, year, duration, search_u) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (item_id, title, author, album, author, 90.0 + number, 0.3 + 0.005 * number,
                 'rock:0.8,pop:0.4', 'danceable:0.6,aggressive:0.6,happy:0.6,party:0.6,relaxed:0.6,sad:0.6',
                 1990 + family, 180.0 + number, f"{title} {author} {album}".lower()),
            )
            cur.execute(
                "INSERT INTO embedding (item_id, embedding) VALUES (%s, %s)",
                (item_id, psycopg2.Binary(vector.astype(np.float32).tobytes())),
            )
            cur.execute(
                "INSERT INTO clap_embedding (item_id, embedding) VALUES (%s, %s)",
                (item_id, psycopg2.Binary(
                    (clap_centres[family] + 0.9 * rng.standard_normal(_CLAP_DIM).astype(np.float32)).tobytes()
                )),
            )
            _add_lyrics(
                cur, item_id,
                4.0 * basis[family] + 3.0 * basis[_FAMILIES + number % 2]
                + 0.2 * rng.standard_normal(_LYRIC_DIM).astype(np.float32), rng,
            )
            cur.execute(
                "INSERT INTO track_server_map (item_id, server_id, provider_track_id, match_tier) "
                "VALUES (%s, 'srv-a', %s, 'fingerprint')",
                (item_id, f"a-{item_id}"),
            )
            if family == 0:
                cur.execute(
                    "INSERT INTO track_server_map (item_id, server_id, provider_track_id, match_tier) "
                    "VALUES (%s, 'srv-b', %s, 'fingerprint')",
                    (item_id, f"b-{item_id}"),
                )


@pytest.fixture(scope='module')
def album_library(shared_pg_dsn):
    from tasks import ivf_manager
    from tasks.mediaserver import registry

    conn = psycopg2.connect(shared_pg_dsn)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {_TABLES} CASCADE")
        for ddl in _SCHEMA:
            cur.execute(ddl)
        _seed_library(cur)

    patcher = pytest.MonkeyPatch()
    patcher.setattr("database.get_db", lambda: conn)
    patcher.setattr("tasks.mediaserver.registry.get_db", lambda: conn)
    patcher.setattr(config, "DATABASE_URL", shared_pg_dsn)
    patcher.setattr(config, "LYRICS_ENABLED", True)
    patcher.setattr(config, "ALBUM_CREATION_TRACKS", 12)
    patcher.setattr(config, "MAX_SONGS_PER_ARTIST", 3)
    from tasks import clap_text_search

    patcher.setattr(config, "ALBUM_CREATION_MUSICNN_SHARE", 0.5)
    registry.invalidate_server_cache()
    ivf_manager.build_and_store_ivf_index(conn)
    ivf_manager.load_ivf_index_for_querying(force_reload=True)
    assert ivf_manager.ivf_index is not None, "the audio index did not load from the test database"
    clap_text_search.build_and_store_clap_index(conn)
    assert clap_text_search._load_clap_index_from_db(), "the DCLAP index did not load"
    try:
        yield conn
    finally:
        clap_text_search._CLAP_INDEX_CACHE.update(
            {'index': None, 'id_map': None, 'reverse_id_map': None, 'loaded': False}
        )
        clap_text_search._CLAP_CACHE['loaded'] = False
        patcher.undo()
        ivf_manager.ivf_index = None
        ivf_manager.id_map = None
        ivf_manager.reverse_id_map = None
        registry.invalidate_server_cache()
        with conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {_TABLES} CASCADE")
        conn.close()


def _create(seed_type, **seed):
    from tasks import album_creation_manager as acm

    return acm.create_album(seed_type, rng=np.random.default_rng(5), **seed)


def test_a_song_seed_gives_a_full_album_inside_its_own_sonic_family(album_library):
    from tasks import album_creation_manager as acm

    album = _create('song', item_id='fp_1_010')
    ids = [track['item_id'] for track in album['tracks']]
    assert len(ids) == 12 == len(set(ids))
    assert 'fp_1_010' in ids
    assert {_family_of(item_id) for item_id in ids} <= {1, 9}
    authors = [track['author'] for track in album['tracks']]
    assert max(authors.count(author) for author in set(authors)) <= 3
    assert [track['slot'] for track in album['tracks']] == list(range(1, 13))
    assert album['tracks'][0]['role'] == acm.ROLE_OPENER
    assert album['tracks'][-1]['role'] == acm.ROLE_CLOSER
    assert album['stats']['tracks'] == 12 and album['stats']['minutes'] > 30


def test_dclap_puts_tracks_in_the_album_that_musicnn_alone_would_never_pick(album_library, monkeypatch):
    from tasks import album_creation_manager as acm

    seed = acm.load_tracks(['fp_1_010'])[0]
    assert seed['clap'] is not None and seed['clap'].shape == (_CLAP_DIM,)
    hidden = {f"fp_9_{number:03d}" for number in range(_HIDDEN - 2)}
    assert hidden <= {track['item_id'] for track in acm.candidate_pool(seed['vector'], {}, seed['clap'])}
    assert hidden & {track['item_id'] for track in _create('song', item_id='fp_1_010')['tracks']}
    monkeypatch.setattr(config, 'ALBUM_CREATION_MUSICNN_SHARE', 1.0)
    audio_only = _create('song', item_id='fp_1_010')
    assert not hidden & {track['item_id'] for track in audio_only['tracks']}


def test_the_dclap_half_of_the_pool_is_scoped_to_the_bound_server(album_library):
    from tasks import album_creation_manager as acm
    from tasks.mediaserver import registry

    seed = acm.load_tracks(['fp_1_010'])[0]
    elsewhere = {f"fp_9_{number:03d}" for number in range(_HIDDEN - 2, _HIDDEN)}
    with registry.bind(registry.get_server('srv-a')):
        on_default = {track['item_id'] for track in acm.candidate_pool(seed['vector'], {}, seed['clap'])}
    assert not elsewhere & on_default
    with registry.bind(registry.get_server('srv-b')):
        on_second = {track['item_id'] for track in acm.candidate_pool(seed['vector'], {}, seed['clap'])}
    assert on_second == set(registry.translate_ids(list(on_second), 'srv-b'))


def test_the_album_prefers_songs_about_what_the_seed_is_about(album_library, monkeypatch):
    from tasks import album_creation_manager as acm

    seed = acm.load_tracks(['fp_1_010'])[0]
    assert seed['lyrics'] is not None and seed['lyrics'].shape == (_LYRIC_DIM,)

    def same_subject(share, item_id):
        monkeypatch.setattr(config, 'ALBUM_CREATION_LYRIC_SHARE', share)
        tracks = _create('song', item_id=item_id)['tracks']
        assert len(tracks) == 12
        wanted = int(item_id.split('_')[2]) % 2
        return sum(1 for track in tracks if int(track['item_id'].split('_')[2]) % 2 == wanted)

    over = ['fp_1_010', 'fp_1_021', 'fp_2_014', 'fp_0_033']
    with_rule = sum(same_subject(0.25, item_id) for item_id in over)
    without_rule = sum(same_subject(1.0, item_id) for item_id in over)
    slots = 12 * len(over)
    assert with_rule > without_rule, (with_rule, without_rule)
    assert with_rule >= 0.6 * slots > without_rule - 1, (with_rule, without_rule)


def test_an_album_is_still_built_when_the_dclap_index_is_away(album_library, monkeypatch):
    from tasks import clap_text_search

    monkeypatch.setattr(clap_text_search, 'is_clap_cache_loaded', lambda: False)
    album = _create('song', item_id='fp_1_010')
    assert len({track['item_id'] for track in album['tracks']}) == 12
    monkeypatch.setattr(config, 'ALBUM_CREATION_MUSICNN_SHARE', 1.0)
    album = _create('song', item_id='fp_1_010')
    assert len({track['item_id'] for track in album['tracks']}) == 12


def test_bound_to_a_secondary_server_every_track_exists_there(album_library):
    from tasks.mediaserver import registry

    with registry.bind(registry.get_server('srv-b')):
        album = _create('song', item_id='fp_0_005')
    ids = [track['item_id'] for track in album['tracks']]
    assert len(ids) == 12
    assert set(registry.translate_ids(ids, 'srv-b')) == set(ids)


def test_the_weekly_seed_sample_only_offers_songs_of_the_bound_server(album_library):
    from tasks import album_creation_manager as acm
    from tasks.mediaserver import registry

    with registry.bind(registry.get_server('srv-b')):
        seeds = acm.weekly_seed_ids()
    assert seeds and set(seeds) == set(registry.translate_ids(seeds, 'srv-b'))
    assert {_family_of(item_id) for item_id in seeds} <= {0, 9}
    assert {_family_of(item_id) for item_id in acm.weekly_seed_ids()} <= {0, 1, 2, 9}


def test_a_description_seed_builds_an_album_from_the_dclap_index(album_library, monkeypatch):
    from tasks import album_creation_manager as acm, clap_text_search

    wanted = [track['item_id'] for track in acm.load_tracks(
        [f"fp_1_{number:03d}" for number in range(40)]
    )]
    monkeypatch.setattr(acm, '_text_embedding', lambda query, steering=None: np.zeros(_CLAP_DIM, dtype=np.float32))
    monkeypatch.setattr(
        clap_text_search, 'search_by_embedding',
        lambda embedding, limit=None: [{'item_id': item_id} for item_id in wanted[:limit]],
    )
    monkeypatch.setattr(config, 'MOOD_LABELS', ['rock'])
    monkeypatch.setattr(acm, 'concept_vocabulary', lambda: [])
    album = acm.create_album('text', query='rock album', rng=np.random.default_rng(3))
    ids = [track['item_id'] for track in album['tracks']]
    assert len(ids) == 12 == len(set(ids))
    assert set(ids) <= set(wanted)
    assert album['seed'] == {'type': 'text', 'label': 'rock album'}
    assert album['suggested_name'] == 'Rock album'
    assert album['tracks'][0]['role'] == acm.ROLE_OPENER


def test_a_description_seed_only_draws_from_the_bound_server(album_library, monkeypatch):
    from tasks import album_creation_manager as acm, clap_text_search
    from tasks.mediaserver import registry

    everything = [f"fp_{family}_{number:03d}" for family in range(_FAMILIES) for number in range(40)]
    monkeypatch.setattr(acm, '_text_embedding', lambda query, steering=None: np.zeros(_CLAP_DIM, dtype=np.float32))
    monkeypatch.setattr(
        clap_text_search, 'search_by_embedding',
        lambda embedding, limit=None: [{'item_id': item_id} for item_id in everything[:limit]],
    )
    monkeypatch.setattr(acm, 'concept_vocabulary', lambda: [])
    with registry.bind(registry.get_server('srv-b')):
        album = acm.create_album('text', query='any music', rng=np.random.default_rng(3))
    ids = [track['item_id'] for track in album['tracks']]
    assert len(ids) == 12
    assert {_family_of(item_id) for item_id in ids} == {0}
