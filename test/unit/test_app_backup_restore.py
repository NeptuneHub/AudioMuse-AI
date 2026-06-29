"""Unit tests for app_backup.restore_backup() chunk and lock validation.

Exercises the /api/backup/restore route through a Flask test client with
multipart/form-data uploads. The module-level Redis lock helpers are patched
(the functions, not redis), BACKUP_DIR is redirected to a pytest tmp dir, and
no test ever completes a chunk set, so the detached restore subprocess is
never spawned.
"""
import io
import os
from unittest.mock import MagicMock

import pytest
from flask import Flask

import app_backup

CONFIRMATION = "I want to restore the database from the backup. This action is not reversible"


@pytest.fixture
def client():
    app = Flask(__name__)
    app.config['TESTING'] = True
    app.register_blueprint(app_backup.backup_bp)
    return app.test_client()


def _form(confirmation=CONFIRMATION, chunk_num=None, total_chunks=None, with_file=True):
    data = {'confirmation': confirmation}
    if chunk_num is not None:
        data['chunk_num'] = str(chunk_num)
    if total_chunks is not None:
        data['total_chunks'] = str(total_chunks)
    if with_file:
        data['file'] = (io.BytesIO(b'SELECT 1;\n'), 'backup.sql')
    return data


def _post(client, **kwargs):
    return client.post(
        '/api/backup/restore',
        data=_form(**kwargs),
        content_type='multipart/form-data',
    )


class TestRestoreValidation:
    def test_wrong_confirmation_is_400(self, client):
        resp = _post(client, confirmation='nope')
        assert resp.status_code == 400
        assert 'Confirmation' in resp.get_json()['error']

    def test_missing_confirmation_is_400(self, client):
        resp = _post(client, confirmation='')
        assert resp.status_code == 400

    def test_missing_file_is_400(self, client):
        resp = _post(client, with_file=False)
        assert resp.status_code == 400
        assert resp.get_json()['error'] == 'No file uploaded.'

    def test_non_integer_chunk_fields_are_400(self, client):
        resp = _post(client, chunk_num='abc', total_chunks='3')
        assert resp.status_code == 400
        assert 'must be integers' in resp.get_json()['error']

    @pytest.mark.parametrize('chunk_num,total_chunks', [
        (0, 3),
        (4, 3),
        (2, 1),
        (-1, 3),
        (0, 0),
    ])
    def test_chunk_num_out_of_range_is_400(self, client, chunk_num, total_chunks):
        resp = _post(client, chunk_num=chunk_num, total_chunks=total_chunks)
        assert resp.status_code == 400
        assert 'Invalid chunk numbers' in resp.get_json()['error']


class TestRestoreLock:
    def test_first_chunk_lock_already_held_is_409(self, client, monkeypatch, tmp_path):
        monkeypatch.setattr(app_backup, 'BACKUP_DIR', str(tmp_path))
        monkeypatch.setattr(app_backup, '_acquire_restore_lock', lambda: False)
        resp = _post(client, chunk_num=1, total_chunks=3)
        assert resp.status_code == 409
        assert 'already in progress' in resp.get_json()['error']

    def test_later_chunk_lock_not_held_is_409(self, client, monkeypatch, tmp_path):
        monkeypatch.setattr(app_backup, 'BACKUP_DIR', str(tmp_path))
        monkeypatch.setattr(app_backup, '_restore_lock_held', lambda: False)
        resp = _post(client, chunk_num=2, total_chunks=3)
        assert resp.status_code == 409
        assert 'Restart the upload from chunk 1' in resp.get_json()['error']

    def test_later_chunk_never_tries_to_acquire(self, client, monkeypatch, tmp_path):
        monkeypatch.setattr(app_backup, 'BACKUP_DIR', str(tmp_path))
        acquire = MagicMock(return_value=True)
        monkeypatch.setattr(app_backup, '_acquire_restore_lock', acquire)
        monkeypatch.setattr(app_backup, '_restore_lock_held', lambda: False)
        resp = _post(client, chunk_num=2, total_chunks=3)
        assert resp.status_code == 409
        acquire.assert_not_called()

    def test_single_file_upload_lock_held_is_409(self, client, monkeypatch):
        monkeypatch.setattr(app_backup, '_acquire_restore_lock', lambda: False)
        resp = _post(client)
        assert resp.status_code == 409
        assert 'already in progress' in resp.get_json()['error']


class _FakeStdin:
    def __init__(self):
        self.buf = bytearray()
        self.closed = False

    def write(self, b):
        self.buf += b
        return len(b)

    def close(self):
        self.closed = True


class TestFeedDumpStrip:
    def test_strips_transaction_timeout_and_prepends_schema(self, tmp_path):
        dump = tmp_path / 'd.sql'
        dump.write_bytes(
            b"SET statement_timeout = 0;\n"
            b"SET transaction_timeout = 0;\n"
            b"SET client_encoding = 'UTF8';\n"
            b"COPY t (a) FROM stdin;\n1\n\\.\n"
        )
        fake = _FakeStdin()
        app_backup._feed_dump(fake, str(dump))
        out = bytes(fake.buf)
        assert out.startswith(b"DROP SCHEMA IF EXISTS public CASCADE; CREATE SCHEMA public;\n")
        assert b"transaction_timeout" not in out
        assert b"SET statement_timeout = 0;\n" in out
        assert b"SET client_encoding = 'UTF8';\n" in out
        assert b"COPY t (a) FROM stdin;" in out
        assert fake.closed is True


class TestRestoreChunkProgress:
    def test_intermediate_chunk_is_acknowledged(self, client, monkeypatch, tmp_path):
        monkeypatch.setattr(app_backup, 'BACKUP_DIR', str(tmp_path))
        monkeypatch.setattr(app_backup, '_acquire_restore_lock', lambda: True)
        resp = _post(client, chunk_num=1, total_chunks=3)
        assert resp.status_code == 200
        body = resp.get_json()
        assert body['success'] is True
        assert body['all_chunks_received'] is False
        assert body['chunk_num'] == 1
        assert body['total_chunks'] == 3
        assert body['received_chunks'] == [1]
        assert body['missing_chunks'] == [2, 3]
        assert os.path.exists(os.path.join(str(tmp_path), 'chunks', 'backup_1_of_3.sql'))

    def test_first_chunk_wipes_leftover_chunks(self, client, monkeypatch, tmp_path):
        monkeypatch.setattr(app_backup, 'BACKUP_DIR', str(tmp_path))
        monkeypatch.setattr(app_backup, '_acquire_restore_lock', lambda: True)
        chunks_dir = tmp_path / 'chunks'
        chunks_dir.mkdir()
        leftover = chunks_dir / 'backup_2_of_3.sql'
        leftover.write_bytes(b'stale data')
        resp = _post(client, chunk_num=1, total_chunks=3)
        assert resp.status_code == 200
        body = resp.get_json()
        assert body['received_chunks'] == [1]
        assert body['missing_chunks'] == [2, 3]
        assert not leftover.exists()

    def test_second_chunk_keeps_existing_chunks(self, client, monkeypatch, tmp_path):
        monkeypatch.setattr(app_backup, 'BACKUP_DIR', str(tmp_path))
        monkeypatch.setattr(app_backup, '_restore_lock_held', lambda: True)
        chunks_dir = tmp_path / 'chunks'
        chunks_dir.mkdir()
        (chunks_dir / 'backup_1_of_3.sql').write_bytes(b'first chunk')
        resp = _post(client, chunk_num=2, total_chunks=3)
        assert resp.status_code == 200
        body = resp.get_json()
        assert body['all_chunks_received'] is False
        assert body['received_chunks'] == [1, 2]
        assert body['missing_chunks'] == [3]
        assert (chunks_dir / 'backup_1_of_3.sql').exists()
        assert (chunks_dir / 'backup_2_of_3.sql').exists()
