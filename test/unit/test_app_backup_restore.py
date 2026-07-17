# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Unit tests for the app_backup create, download, and chunked-restore endpoints.

Posts restore chunks to confirm confirmation, file, and chunk-field
validation plus the cross-chunk restore locking behavior, and exercises the
two-step create-then-download backup flow.

Main Features:
* Confirmation, missing-file, and chunk-field/range validation returning 400.
* First-chunk lock-held and later-chunk lock-missing returning 409.
* Create returning the zipped backup file name as JSON; download serving only
  filenames matching the backup pattern and 404ing everything else.
* Zip-or-sql detection by magic bytes with in-zip .sql extraction for restore.
"""

import io
import os
import zipfile
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

    @pytest.mark.parametrize(
        'chunk_num,total_chunks',
        [
            (0, 3),
            (4, 3),
            (2, 1),
            (-1, 3),
            (0, 0),
        ],
    )
    def test_chunk_num_out_of_range_is_400(self, client, chunk_num, total_chunks):
        resp = _post(client, chunk_num=chunk_num, total_chunks=total_chunks)
        assert resp.status_code == 400
        assert 'Invalid chunk numbers' in resp.get_json()['error']


class TestCreateAndDownload:
    def test_create_returns_zipped_backup_filename_json(self, client, monkeypatch, tmp_path):
        monkeypatch.setattr(app_backup, 'BACKUP_DIR', str(tmp_path))

        def fake_run(cmd, **kwargs):
            kwargs['stdout'].write('-- dump\n')
            return MagicMock(returncode=0, stderr='')

        monkeypatch.setattr(app_backup.subprocess, 'run', fake_run)
        resp = client.post('/api/backup/create')
        assert resp.status_code == 200
        body = resp.get_json()
        assert body['success'] is True
        assert body['filename'].endswith('.zip')
        assert app_backup._BACKUP_FILENAME_RE.fullmatch(body['filename'])
        assert body['size_bytes'] > 0
        with zipfile.ZipFile(tmp_path / body['filename']) as zf:
            member = zf.namelist()[0]
            assert member.endswith('.sql')
            assert zf.read(member) == b'-- dump\n'
        assert not (tmp_path / member).exists()

    def test_download_serves_backup_as_attachment(self, client, monkeypatch, tmp_path):
        monkeypatch.setattr(app_backup, 'BACKUP_DIR', str(tmp_path))
        name = 'audiomuse_backup_20260717_120000.sql'
        (tmp_path / name).write_bytes(b'-- dump\n')
        resp = client.get(f'/api/backup/download/{name}')
        assert resp.status_code == 200
        assert resp.data == b'-- dump\n'
        assert 'attachment' in resp.headers['Content-Disposition']

    def test_download_rejects_filename_outside_backup_pattern(self, client, monkeypatch, tmp_path):
        monkeypatch.setattr(app_backup, 'BACKUP_DIR', str(tmp_path))
        (tmp_path / 'secret.txt').write_bytes(b'nope')
        resp = client.get('/api/backup/download/secret.txt')
        assert resp.status_code == 404

    def test_download_missing_backup_file_is_404(self, client, monkeypatch, tmp_path):
        monkeypatch.setattr(app_backup, 'BACKUP_DIR', str(tmp_path))
        resp = client.get('/api/backup/download/audiomuse_backup_20260717_120000.sql')
        assert resp.status_code == 404


class TestExtractSqlIfZip:
    def test_plain_sql_file_passes_through_unchanged(self, tmp_path):
        dump = tmp_path / 'dump.sql'
        dump.write_bytes(b'SELECT 1;\n')
        source, extracted = app_backup._extract_sql_if_zip(str(dump), io.StringIO())
        assert source == str(dump)
        assert extracted is None

    def test_zip_upload_extracts_inner_sql_to_temp_file(self, tmp_path):
        inner = b'SELECT 42;\n'
        zpath = tmp_path / 'dump.zip'
        with zipfile.ZipFile(zpath, 'w') as zf:
            zf.writestr('audiomuse_backup_20260717_120000.sql', inner)
        source, extracted = app_backup._extract_sql_if_zip(str(zpath), io.StringIO())
        assert source == extracted
        assert source != str(zpath)
        with open(source, 'rb') as fh:
            assert fh.read() == inner
        os.unlink(source)

    def test_zip_without_sql_member_aborts_restore(self, tmp_path):
        zpath = tmp_path / 'dump.zip'
        with zipfile.ZipFile(zpath, 'w') as zf:
            zf.writestr('readme.txt', 'not sql')
        source, extracted = app_backup._extract_sql_if_zip(str(zpath), io.StringIO())
        assert source is None
        assert extracted is None


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
        result = {}
        app_backup._feed_dump(fake, str(dump), result)
        out = bytes(fake.buf)
        assert out.startswith(b"DROP SCHEMA IF EXISTS public CASCADE; CREATE SCHEMA public;\n")
        assert b"transaction_timeout" not in out
        assert b"SET statement_timeout = 0;\n" in out
        assert b"SET client_encoding = 'UTF8';\n" in out
        assert b"COPY t (a) FROM stdin;" in out
        assert fake.closed is True
        assert result.get('ok') is True

    def test_missing_dump_file_is_not_reported_ok(self, tmp_path):
        fake = _FakeStdin()
        result = {}
        app_backup._feed_dump(fake, str(tmp_path / 'does_not_exist.sql'), result)
        assert result.get('ok') is not True
        assert 'error' in result
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
