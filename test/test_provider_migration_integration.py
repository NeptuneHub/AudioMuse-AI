"""Real provider-migration integration test (runs against a live Postgres).

Unlike a mocked-cursor unit test, this drives the *actual* migration end to
end against a real PostgreSQL database for every combination of supported
providers (including migrating a provider to itself):

  1. build a small library exactly as the SOURCE provider's score rows look
     (provider-specific item_id + file_path), seed it into a real ``score``
     table plus the embedding / voyager / map-projection / artist tables the
     migration touches,
  2. build the same library as the TARGET provider's probe would return it,
  3. run the real ``tasks.provider_migration_matcher.match_tracks`` to produce
     the old_id -> new_id mapping (one source-only track stays unmatched so we
     also exercise orphan deletion),
  4. persist a ``migration_session`` in ``dry_run_ready`` and run the real
     ``tasks.provider_migration_tasks.execute_provider_migration`` job (the
     transactional id rewrite),
  5. assert the database ended up correct: item_ids rewritten, embeddings
     followed through the FK cascade, the orphan deleted, voyager / map
     id_maps rewritten, provider-specific artist tables cleared, ``app_config``
     updated with the target provider's credentials, and the session marked
     completed.

The only thing faked is the provider HTTP layer (we hand the matcher the
probe-shaped dicts directly) and the embedding bytes — everything that touches
the database is the production code path.

Migrating a provider to itself is the most demanding case: it models a library
re-scan that re-issues ids so the new ids overlap the existing ones, which is
exactly what the two-pass ``_MIG_TMP_PREFIX`` rewrite in
``_run_migration_transaction`` exists to survive. A mocked cursor can never
prove that works; a real Postgres can.

Database selection (in priority order):
  * ``AUDIOMUSE_TEST_DATABASE_URL`` — a throwaway DB the test fully owns. The
    test DROPs and recreates the ``public`` schema, so this MUST point at a
    disposable database (the GitHub workflow points it at a postgres service).
    The generic ``DATABASE_URL`` is intentionally NOT used, to avoid ever
    dropping a real database.
  * otherwise, an ephemeral instance via the optional ``pgserver`` package
    (``pip install pgserver``) for zero-infra local runs.
  * otherwise the whole module is skipped.

Run locally:
    pip install pgserver
    pytest test/test_provider_migration_integration.py -s -v --tb=short
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
    """Load a ``tasks.*`` module straight from its file (same loader the
    provider-migration unit tests use)."""
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    repo_root = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    )
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    mod_path = os.path.join(repo_root, *rel_parts)
    spec = importlib.util.spec_from_file_location(mod_name, mod_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


matcher = _load_module(
    'tasks.provider_migration_matcher', 'tasks', 'provider_migration_matcher.py'
)
mig = _load_module(
    'tasks.provider_migration_tasks', 'tasks', 'provider_migration_tasks.py'
)


PROVIDERS = ('jellyfin', 'emby', 'navidrome', 'lyrion')

_ID_BASE = {
    'jellyfin': 0x10000,
    'emby': 5000,
    'navidrome': 0xABCD00,
    'lyrion': 90000,
}

_CROSS_TARGET_SHIFT = 1_000_000

_ORPHAN_OFFSET = 900

_EXPECTED_CONFIG_KEYS = {
    'jellyfin': ['JELLYFIN_URL', 'JELLYFIN_USER_ID', 'JELLYFIN_TOKEN'],
    'emby': ['EMBY_URL', 'EMBY_USER_ID', 'EMBY_TOKEN'],
    'navidrome': ['NAVIDROME_URL', 'NAVIDROME_USER', 'NAVIDROME_PASSWORD'],
    'lyrion': ['LYRION_URL'],
}

_TARGET_CREDS = {
    'jellyfin': {'url': 'http://jf.test:8096', 'user_id': 'jfuser', 'token': 'jftoken'},
    'emby': {'url': 'http://emby.test:8096', 'user_id': 'embyuser', 'token': 'embytoken'},
    'navidrome': {'url': 'http://nav.test:4533', 'user': 'navuser', 'password': 'navpass'},
    'lyrion': {'url': 'http://lms.test:9000'},
}


SHARED_TRACKS = [
    {'artist': 'Daft Punk', 'album': 'Discovery', 'album_artist': 'Daft Punk',
     'title': 'One More Time', 'disc': 1, 'track': 1, 'ext': 'flac'},
    {'artist': 'Green Day', 'album': 'American Idiot', 'album_artist': 'Green Day',
     'title': 'Boulevard of Broken Dreams', 'disc': 1, 'track': 4, 'ext': 'flac'},
    {'artist': 'Eagles', 'album': 'Ultimate Rock Hits', 'album_artist': 'Various Artists',
     'title': 'Hotel California', 'disc': 1, 'track': 3, 'ext': 'mp3'},
]

ORPHAN_TRACK = {
    'artist': 'Nobody', 'album': 'Orphan Album', 'album_artist': 'Nobody',
    'title': 'Orphan Track', 'disc': 1, 'track': 1, 'ext': 'flac',
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
    return rel


# ---------------------------------------------------------------------------
# Schema — mirrors the tables tasks.provider_migration_tasks touches, copied
# from app_helper.init_db so the transaction runs against a real layout.
# ---------------------------------------------------------------------------

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
]


@pytest.fixture(scope='session')
def pg_dsn():
    """A libpq DSN for a throwaway Postgres the test fully owns."""
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
    """Reset the schema, wire the execute job at the real DB, yield helpers."""
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
    """Write ``id_map_json`` split across ``n_parts`` rows, mimicking the
    large-library layout produced by ``store_voyager_index_segmented`` (each
    row holds a partial-JSON fragment; the binary lives on part 1)."""
    step = max(1, -(-len(id_map_json) // n_parts))
    frags = [id_map_json[i:i + step] for i in range(0, len(id_map_json), step)]
    while len(frags) < n_parts:
        frags.append('')
    for k in range(1, n_parts + 1):
        cur.execute(
            f"INSERT INTO {table} (index_name, {binary_col}, id_map_json, "
            f"embedding_dimension) VALUES (%s, %s, %s, %s)",
            (f"{base}_{k}_{n_parts}",
             psycopg2.Binary(b'\x00' if k == 1 else b''), frags[k - 1], dim),
        )


def _reassemble_id_map(parts):
    """Concatenate ``(index_name, id_map_json)`` fragments in part order."""
    ordered = sorted(parts, key=lambda p: int(re.match(r'^.*_(\d+)_\d+$', p[0]).group(1)))
    return ''.join((frag or '') for _, frag in ordered)


def _seed_library(conn, source_rendered, segmented=False):
    """Insert the source library into every table the migration rewrites.

    With ``segmented=True`` the voyager / map-projection id_maps are written
    split across part rows (the >~1M-song layout) instead of as a single row.
    """
    src_ids = [r['id'] for r in source_rendered]
    voyager_map = json.dumps({str(i): sid for i, sid in enumerate(src_ids)})
    projection_map = json.dumps(src_ids)
    with conn.cursor() as cur:
        for index, r in enumerate(source_rendered):
            cur.execute(
                "INSERT INTO score (item_id, title, author, album, album_artist, "
                "file_path, year) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (r['id'], r['title'], r['artist'], r['album'], r['album_artist'],
                 r['path'], 1900 + index),
            )
            for table in ('embedding', 'clap_embedding', 'lyrics_embedding'):
                cur.execute(f"INSERT INTO {table} (item_id) VALUES (%s)", (r['id'],))
        if segmented:
            _insert_segmented_index(cur, 'voyager_index_data', 'index_data',
                                    'voyager_main', voyager_map, 128)
            _insert_segmented_index(cur, 'map_projection_data', 'projection_data',
                                    'map_main', projection_map, 2)
        else:
            cur.execute(
                "INSERT INTO voyager_index_data (index_name, index_data, id_map_json, "
                "embedding_dimension) VALUES (%s, %s, %s, %s)",
                ('voyager_main', psycopg2.Binary(b'\x00'), voyager_map, 128),
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
        'new_meta': new_meta,
        'selected_libraries': None,
    }
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO migration_session (source_type, target_type, target_creds, "
            "state, status) VALUES (%s, %s, %s, %s, 'dry_run_ready') RETURNING id",
            (source, target, json.dumps(_TARGET_CREDS[target]), json.dumps(state)),
        )
        session_id = cur.fetchone()[0]
    conn.commit()
    return session_id


@pytest.mark.integration
@pytest.mark.parametrize('target', PROVIDERS)
@pytest.mark.parametrize('source', PROVIDERS)
def test_real_provider_migration(source, target, migration_db):
    """Run the real execute transaction for one (source -> target) pair."""
    self_migration = source == target
    target_shift = 1 if self_migration else _CROSS_TARGET_SHIFT

    source_rendered = []
    for index, track in enumerate(SHARED_TRACKS):
        rel = _relative_path(track)
        source_rendered.append({
            'id': _provider_id(source, 0, index),
            'path': _provider_path(source, rel),
            'title': track['title'], 'artist': track['artist'],
            'album': track['album'], 'album_artist': track['album_artist'],
        })
    orphan_rel = _relative_path(ORPHAN_TRACK)
    source_rendered.append({
        'id': _provider_id(source, 0, _ORPHAN_OFFSET),
        'path': _provider_path(source, orphan_rel),
        'title': ORPHAN_TRACK['title'], 'artist': ORPHAN_TRACK['artist'],
        'album': ORPHAN_TRACK['album'], 'album_artist': ORPHAN_TRACK['album_artist'],
    })
    target_rendered = []
    for index, track in enumerate(SHARED_TRACKS):
        rel = _relative_path(track)
        target_rendered.append({
            'id': _provider_id(target, target_shift, index),
            'path': _provider_path(target, rel),
            'title': track['title'], 'artist': track['artist'],
            'album': track['album'], 'album_artist': track['album_artist'],
        })

    old_rows = [
        {'item_id': r['id'], 'file_path': r['path'], 'title': r['title'],
         'author': r['artist'], 'album': r['album'], 'album_artist': r['album_artist']}
        for r in source_rendered
    ]
    new_tracks = [
        {'id': r['id'], 'path': r['path'], 'title': r['title'], 'artist': r['artist'],
         'album': r['album'], 'album_artist': r['album_artist']}
        for r in target_rendered
    ]

    match_result = matcher.match_tracks(old_rows, new_tracks)
    matches = match_result['matches']

    orphan_id = source_rendered[-1]['id']
    expected_map = {
        source_rendered[i]['id']: target_rendered[i]['id']
        for i in range(len(SHARED_TRACKS))
    }
    assert matches == expected_map, (
        f"{source}->{target}: matcher mapping wrong\n  expected {expected_map}\n  got {matches}"
    )
    assert orphan_id not in matches, "the source-only track must stay unmatched"

    new_meta = {
        r['id']: {'path': r['path'], 'title': r['title'], 'artist': r['artist'],
                  'album': r['album'], 'album_artist': r['album_artist'],
                  'year': 2000 + i}
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

    new_ids = set(expected_map.values())
    verify = migration_db['connect']()
    with verify.cursor() as cur:
        cur.execute("SELECT item_id, file_path, title, album_artist, year FROM score")
        score = {row[0]: row for row in cur.fetchall()}
        assert set(score.keys()) == new_ids, (
            f"{source}->{target}: score item_ids not rewritten\n  want {new_ids}\n  got {set(score.keys())}"
        )
        assert orphan_id not in score, "orphan score row must be deleted"
        for i, r in enumerate(target_rendered):
            row = score[r['id']]
            assert row[1] == r['path'], f"file_path not refreshed for {r['id']}"
            assert row[2] == r['title']
            assert row[3] == r['album_artist']
            assert row[4] == 2000 + i, "year not refreshed from new_meta"

        for table in ('embedding', 'clap_embedding', 'lyrics_embedding'):
            cur.execute(f"SELECT item_id FROM {table}")
            ids = {row[0] for row in cur.fetchall()}
            assert ids == new_ids, (
                f"{source}->{target}: {table} did not follow the rewrite "
                f"(orphan cascade-delete + id remap)\n  want {new_ids}\n  got {ids}"
            )

        cur.execute("SELECT id_map_json FROM voyager_index_data WHERE index_name = 'voyager_main'")
        voyager_map = json.loads(cur.fetchone()[0])
        assert set(voyager_map.values()) == new_ids
        assert orphan_id not in voyager_map.values()

        cur.execute("SELECT id_map_json FROM map_projection_data WHERE index_name = 'map_main'")
        proj_map = json.loads(cur.fetchone()[0])
        assert len(proj_map) == len(SHARED_TRACKS) + 1, "list length must be preserved"
        assert proj_map[-1] is None, "orphan slot in the projection list must become None"
        assert set(v for v in proj_map if v is not None) == new_ids

        for table in ('artist_index_data', 'artist_metadata_data',
                      'artist_component_projection', 'artist_mapping'):
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            assert cur.fetchone()[0] == 0, f"{table} should be cleared on migration"

        cur.execute("SELECT key, value FROM app_config")
        config_rows = dict(cur.fetchall())
        assert config_rows.get('MEDIASERVER_TYPE') == target
        for key in _EXPECTED_CONFIG_KEYS[target]:
            assert key in config_rows, f"{target}: app_config missing {key}"
        assert 'MUSIC_LIBRARIES' not in config_rows, "null selection should clear MUSIC_LIBRARIES"

        cur.execute("SELECT status FROM migration_session WHERE id = %s", (session_id,))
        assert cur.fetchone()[0] == 'completed'
    verify.close()
    print(f"  ok: {len(new_ids)} tracks rewritten, orphan deleted, app_config -> {target}")


@pytest.mark.integration
def test_real_provider_migration_rewrites_segmented_id_map(migration_db):
    """Regression: a SEGMENTED voyager / map id_map must be rewritten end to end.

    ``store_voyager_index_segmented`` splits ``id_map_json`` across part rows
    once a library is large enough (~1M+ songs), so every segmented row holds a
    partial-JSON fragment. The migration must reassemble + rewrite + re-split;
    the previous per-row ``json.loads`` left every fragment untouched, leaving
    the index pointing at deleted old-provider ids. This drives the real
    execute job against a real Postgres with a segmented index seeded.
    """
    source, target = 'jellyfin', 'navidrome'

    source_rendered = []
    for index, track in enumerate(SHARED_TRACKS):
        rel = _relative_path(track)
        source_rendered.append({
            'id': _provider_id(source, 0, index),
            'path': _provider_path(source, rel),
            'title': track['title'], 'artist': track['artist'],
            'album': track['album'], 'album_artist': track['album_artist'],
        })
    source_rendered.append({
        'id': _provider_id(source, 0, _ORPHAN_OFFSET),
        'path': _provider_path(source, _relative_path(ORPHAN_TRACK)),
        'title': ORPHAN_TRACK['title'], 'artist': ORPHAN_TRACK['artist'],
        'album': ORPHAN_TRACK['album'], 'album_artist': ORPHAN_TRACK['album_artist'],
    })
    target_rendered = [{
        'id': _provider_id(target, _CROSS_TARGET_SHIFT, index),
        'path': _provider_path(target, _relative_path(track)),
        'title': track['title'], 'artist': track['artist'],
        'album': track['album'], 'album_artist': track['album_artist'],
    } for index, track in enumerate(SHARED_TRACKS)]

    old_rows = [
        {'item_id': r['id'], 'file_path': r['path'], 'title': r['title'],
         'author': r['artist'], 'album': r['album'], 'album_artist': r['album_artist']}
        for r in source_rendered
    ]
    new_tracks = [
        {'id': r['id'], 'path': r['path'], 'title': r['title'], 'artist': r['artist'],
         'album': r['album'], 'album_artist': r['album_artist']}
        for r in target_rendered
    ]
    matches = matcher.match_tracks(old_rows, new_tracks)['matches']
    expected_map = {
        source_rendered[i]['id']: target_rendered[i]['id']
        for i in range(len(SHARED_TRACKS))
    }
    assert matches == expected_map
    orphan_id = source_rendered[-1]['id']
    new_meta = {
        r['id']: {'path': r['path'], 'title': r['title'], 'artist': r['artist'],
                  'album': r['album'], 'album_artist': r['album_artist'],
                  'year': 2000 + i}
        for i, r in enumerate(target_rendered)
    }

    conn = migration_db['connect']()
    _seed_library(conn, source_rendered, segmented=True)
    session_id = _insert_session(conn, source, target, matches, new_meta)

    result = mig.execute_provider_migration(session_id)
    assert result['ok'] is True
    assert result['matched'] == len(SHARED_TRACKS)

    new_ids = set(expected_map.values())
    old_ids = set(expected_map.keys()) | {orphan_id}
    verify = migration_db['connect']()
    with verify.cursor() as cur:
        cur.execute("SELECT index_name, id_map_json FROM voyager_index_data")
        vparts = [(n, j) for n, j in cur.fetchall() if re.match(r'^voyager_main_\d+_\d+$', n)]
        assert len(vparts) >= 2, "voyager index must actually be segmented in this test"
        voyager_map = json.loads(_reassemble_id_map(vparts))
        assert set(voyager_map.values()) == new_ids
        assert not (set(voyager_map.values()) & old_ids), "no stale old-provider ids may remain"
        assert orphan_id not in voyager_map.values()

        cur.execute("SELECT index_name, id_map_json FROM map_projection_data")
        pparts = [(n, j) for n, j in cur.fetchall() if re.match(r'^map_main_\d+_\d+$', n)]
        assert len(pparts) >= 2, "map projection must actually be segmented in this test"
        proj_map = json.loads(_reassemble_id_map(pparts))
        assert len(proj_map) == len(SHARED_TRACKS) + 1, "list length must be preserved"
        assert proj_map[-1] is None, "orphan slot must become None"
        assert set(v for v in proj_map if v is not None) == new_ids
    verify.close()
    print(f"  ok (segmented): {len(vparts)} voyager parts reassembled + rewritten -> {target}")


def test_segmented_id_map_relabel_overflow_is_soft_failure(migration_db):
    """A relabel that cannot fit back into the existing part rows is a SOFT
    failure: the migration still commits, the stale index is dropped, and
    ``index_rebuild_needed`` is flagged so the UI can ask for a re-analysis.

    Forced by shrinking ``VOYAGER_MAX_PART_SIZE_MB`` to 0 so the rewritten
    id_map needs far more part rows than the seeded index has, which is exactly
    the condition ``rewrite_segmented_id_map`` raises ``ValueError`` for.
    """
    import config

    source, target = 'jellyfin', 'navidrome'

    source_rendered = []
    for index, track in enumerate(SHARED_TRACKS):
        source_rendered.append({
            'id': _provider_id(source, 0, index),
            'path': _provider_path(source, _relative_path(track)),
            'title': track['title'], 'artist': track['artist'],
            'album': track['album'], 'album_artist': track['album_artist'],
        })
    source_rendered.append({
        'id': _provider_id(source, 0, _ORPHAN_OFFSET),
        'path': _provider_path(source, _relative_path(ORPHAN_TRACK)),
        'title': ORPHAN_TRACK['title'], 'artist': ORPHAN_TRACK['artist'],
        'album': ORPHAN_TRACK['album'], 'album_artist': ORPHAN_TRACK['album_artist'],
    })
    target_rendered = [{
        'id': _provider_id(target, _CROSS_TARGET_SHIFT, index),
        'path': _provider_path(target, _relative_path(track)),
        'title': track['title'], 'artist': track['artist'],
        'album': track['album'], 'album_artist': track['album_artist'],
    } for index, track in enumerate(SHARED_TRACKS)]

    old_rows = [
        {'item_id': r['id'], 'file_path': r['path'], 'title': r['title'],
         'author': r['artist'], 'album': r['album'], 'album_artist': r['album_artist']}
        for r in source_rendered
    ]
    new_tracks = [
        {'id': r['id'], 'path': r['path'], 'title': r['title'], 'artist': r['artist'],
         'album': r['album'], 'album_artist': r['album_artist']}
        for r in target_rendered
    ]
    matches = matcher.match_tracks(old_rows, new_tracks)['matches']
    expected_map = {
        source_rendered[i]['id']: target_rendered[i]['id']
        for i in range(len(SHARED_TRACKS))
    }
    new_ids = set(expected_map.values())
    new_meta = {
        r['id']: {'path': r['path'], 'title': r['title'], 'artist': r['artist'],
                  'album': r['album'], 'album_artist': r['album_artist'],
                  'year': 2000 + i}
        for i, r in enumerate(target_rendered)
    }

    conn = migration_db['connect']()
    _seed_library(conn, source_rendered, segmented=True)
    session_id = _insert_session(conn, source, target, matches, new_meta)

    saved_max_part = config.VOYAGER_MAX_PART_SIZE_MB
    config.VOYAGER_MAX_PART_SIZE_MB = 0
    try:
        result = mig.execute_provider_migration(session_id)
    finally:
        config.VOYAGER_MAX_PART_SIZE_MB = saved_max_part

    assert result['ok'] is True
    assert result['matched'] == len(SHARED_TRACKS)
    assert result['index_rebuild_needed'] is True

    verify = migration_db['connect']()
    with verify.cursor() as cur:
        cur.execute("SELECT count(*) FROM voyager_index_data")
        assert cur.fetchone()[0] == 0, "stale voyager index must be dropped, not left corrupt"
        cur.execute("SELECT count(*) FROM map_projection_data")
        assert cur.fetchone()[0] == 0, "stale map projection must be dropped, not left corrupt"

        cur.execute("SELECT item_id FROM score")
        score_ids = {row[0] for row in cur.fetchall()}
        assert score_ids == new_ids, "every other table must still migrate and commit"

        cur.execute("SELECT status FROM migration_session WHERE id = %s", (session_id,))
        assert cur.fetchone()[0] == 'completed'
    verify.close()
    print(f"  ok (soft-fail): stale index dropped + flagged, {len(score_ids)} tracks migrated -> {target}")
