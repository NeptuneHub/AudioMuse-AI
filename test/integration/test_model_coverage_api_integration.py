# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only

"""Model coverage against real PostgreSQL directories and source mappings.

Main Features:
* Verifies global header counts and local directory/mapping intersections.
* Checks legacy default IDs, lyrics percentages and empty secondary sources.
* Uses only the shared disposable test database and an isolated schema.
"""

from types import SimpleNamespace
import uuid

import numpy as np
import psycopg2
from psycopg2 import sql
import pytest
from flask import Flask

import app_models
import config
import database
from tasks import neural_fingerprint_index as nfi
from tasks import paged_ivf
from tasks.mediaserver import registry

pytestmark = pytest.mark.integration


def test_global_and_local_model_coverage(shared_pg_dsn, monkeypatch):
    conn = psycopg2.connect(shared_pg_dsn)
    conn.autocommit = True
    schema = 'model_coverage_' + uuid.uuid4().hex
    with conn.cursor() as cur:
        cur.execute(sql.SQL('CREATE SCHEMA {}').format(sql.Identifier(schema)))
        cur.execute(sql.SQL('SET search_path TO {}').format(sql.Identifier(schema)))
    try:
        with conn.cursor() as cur:
            cur.execute('CREATE TABLE score (item_id TEXT PRIMARY KEY)')
            cur.execute('CREATE TABLE lyrics_embedding (item_id TEXT PRIMARY KEY, embedding BYTEA)')
            cur.execute('CREATE TABLE music_servers (server_id TEXT PRIMARY KEY, is_default BOOLEAN, updated_at TIMESTAMPTZ)')
            cur.execute('CREATE TABLE track_server_map (server_id TEXT, item_id TEXT)')
            cur.execute('CREATE TABLE ivf_dir (name TEXT PRIMARY KEY, blob_data BYTEA)')
            cur.execute("INSERT INTO score VALUES ('fp_a'), ('fp_b'), ('legacy'), ('fp_unanalysed')")
            cur.execute("INSERT INTO lyrics_embedding VALUES ('fp_a', %s)", (b'lyrics',))
            cur.execute("INSERT INTO music_servers VALUES ('primary', TRUE, NOW()), ('secondary', FALSE, NOW())")
            cur.execute("INSERT INTO track_server_map VALUES ('primary', 'fp_a'), ('secondary', 'fp_b')")
            for name, ids in ((config.INDEX_NAME, ['fp_a', 'fp_b', 'legacy']),
                              ('clap_index', ['fp_b']), ('lyrics_index', ['fp_a'])):
                blob = paged_ivf.pack_directory(np.zeros((1, 2), dtype=np.float32),
                                               np.zeros(len(ids), dtype=np.uint32), ids, 2, 'angular')
                cur.execute('INSERT INTO ivf_dir VALUES (%s, %s)', (name + '__ivf_dir', blob))
        monkeypatch.setattr(database, 'get_db', lambda: conn)
        monkeypatch.setattr(registry, 'get_default_server_id', lambda: 'primary')
        monkeypatch.setattr(registry, 'get_server', lambda sid: {'server_id': sid})
        monkeypatch.setattr(registry, 'has_secondary_servers', lambda: True)
        pack = SimpleNamespace(build_id=schema, ids=np.array(['fp_a', 'fp_b']), live_tracks=2)
        monkeypatch.setattr(nfi, '_STATE', {'pack': pack, 'building': False, 'error': None})
        paged_ivf.invalidate_availability_cache()
        nfi.invalidate_availability_cache()
        app = Flask(__name__)
        app.register_blueprint(app_models.models_bp)
        client = app.test_client()
        global_result = client.get('/api/models')
        assert global_result.status_code == 200
        assert set(global_result.json) == {'models'}
        assert global_result.json['models']['lyrics']['global'] == {'count': 1, 'total': 4, 'percentage': 25.0}
        primary = client.get('/api/models?server_id=primary').json['models']
        assert primary['musicnn']['local'] == {'count': 2, 'total': 2, 'percentage': 100.0}
        assert primary['clap']['local']['count'] == 0
        assert primary['neural-fingerprint']['local']['count'] == 1
        secondary = client.get('/api/models?server_id=secondary').json['models']
        assert secondary['musicnn']['local'] == {'count': 1, 'total': 1, 'percentage': 100.0}
        assert secondary['lyrics']['local']['count'] == 0
        with conn.cursor() as cur:
            cur.execute("DELETE FROM track_server_map WHERE server_id = 'secondary'")
        paged_ivf.invalidate_availability_cache('secondary')
        nfi.invalidate_availability_cache('secondary')
        empty = client.get('/api/models?server_id=secondary').json['models']
        assert all(model['local'] == {'count': 0, 'total': 0, 'percentage': 0.0} for model in empty.values())
    finally:
        paged_ivf.invalidate_availability_cache()
        nfi.invalidate_availability_cache()
        with conn.cursor() as cur:
            cur.execute('SET search_path TO public')
            cur.execute(sql.SQL('DROP SCHEMA {} CASCADE').format(sql.Identifier(schema)))
        conn.close()
