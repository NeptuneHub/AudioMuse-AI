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
* pg_dump/psql connection args come from config.DATABASE_URL, with the password
  moved into PGPASSWORD so it never appears in argv.
"""

import io
import os
import sys
import types
import zipfile
from unittest.mock import MagicMock

import pytest
from flask import Flask
from psycopg2.extensions import parse_dsn

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
            assert zf.read(member).replace(b'\r\n', b'\n') == b'-- dump\n'
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


class TestPgConnectionArgs:
    def test_pg_dump_targets_the_database_url(self, monkeypatch):
        monkeypatch.setattr(
            app_backup.config, 'DATABASE_URL', 'postgresql://audiomuse:pw@postgres:5432/audiomusedb'
        )
        cmd = app_backup._pg_cmd('pg_dump', '--clean')
        assert cmd == [
            'pg_dump', '-d', 'postgresql://audiomuse@postgres:5432/audiomusedb', '--clean',
        ]

    def test_password_goes_to_pgpassword_and_never_into_argv(self, monkeypatch):
        monkeypatch.setattr(
            app_backup.config, 'DATABASE_URL', 'postgresql://audiomuse:s3cr%40t@db:5432/audiomusedb'
        )
        assert 's3cr' not in ' '.join(app_backup._pg_cmd('pg_dump'))
        assert app_backup._pg_env()['PGPASSWORD'] == 's3cr@t'

    def test_database_url_query_options_are_preserved_for_pg_dump(self, monkeypatch):
        monkeypatch.setattr(
            app_backup.config,
            'DATABASE_URL',
            'postgresql://user:pw@db:5432/music?sslmode=require&application_name=AudioMuse',
        )
        cmd = app_backup._pg_cmd('pg_dump')
        assert cmd[2] == (
            'postgresql://user@db:5432/music?sslmode=require&application_name=AudioMuse'
        )
        assert app_backup._pg_env()['PGPASSWORD'] == 'pw'

    def test_unix_socket_url_still_resolves_to_the_socket_directory(self, monkeypatch):
        monkeypatch.setattr(
            app_backup.config,
            'DATABASE_URL',
            'postgresql://postgres:@%2Fvar%2Flib%2Fpgdata:5432/postgres',
        )
        cmd = app_backup._pg_cmd('psql', '-v', 'ON_ERROR_STOP=1')
        assert cmd == [
            'psql',
            '-d',
            'postgresql://postgres@%2Fvar%2Flib%2Fpgdata:5432/postgres',
            '-v',
            'ON_ERROR_STOP=1',
        ]
        assert parse_dsn(cmd[2])['host'] == '/var/lib/pgdata'
        assert app_backup._pg_env()['PGPASSWORD'] == ''

    def test_unusable_connection_releases_the_restore_lock_and_logs(self, monkeypatch, tmp_path):
        monkeypatch.setattr(app_backup.config, 'DATABASE_URL', 'postgresql://u:p@[::1:5432/db')
        released = []
        monkeypatch.setattr(app_backup, '_release_restore_lock', lambda: released.append(True))
        monkeypatch.setattr(
            app_backup.restart_manager, 'publish_start_request', lambda **_kwargs: True
        )
        log_file = tmp_path / 'restore.log'
        assert app_backup._run_restore_runner(str(tmp_path / 'dump.sql'), str(log_file)) == 1
        assert released == [True]
        assert 'unusable' in log_file.read_text()

    def test_runner_aborts_before_psql_when_flask_stop_is_not_acked(
        self, monkeypatch, tmp_path
    ):
        dump_file = tmp_path / 'dump.sql'
        dump_file.write_text('SELECT 1;')
        log_file = tmp_path / 'restore.log'
        released = []
        worker_start = MagicMock(return_value=True)
        psql = MagicMock(side_effect=AssertionError('psql must not run'))
        monkeypatch.setattr(
            app_backup.config, 'DATABASE_URL', 'postgresql://u:p@db:5432/music'
        )
        monkeypatch.setattr(
            app_backup.restart_manager, 'stop_local_flask_service_detail',
            lambda: (False, 'exit 7: flask: ERROR (abnormal termination)'),
        )
        monkeypatch.setattr(app_backup.restart_manager, 'start_local_flask_service', lambda: True)
        monkeypatch.setattr(app_backup.restart_manager, 'publish_start_request', worker_start)
        monkeypatch.setattr(app_backup.subprocess, 'Popen', psql)
        monkeypatch.setattr(app_backup, '_release_restore_lock', lambda: released.append(True))

        assert app_backup._run_restore_runner(str(dump_file), str(log_file)) == 1
        psql.assert_not_called()
        worker_start.assert_called_once_with(
            timeout_seconds=app_backup.config.QUEUE_CONTROL_ACTION_WINDOW_SECONDS
        )
        assert released == [True]
        assert dump_file.exists()
        log_text = log_file.read_text()
        assert 'Restore ABORTED' in log_text
        assert 'THE DATABASE WAS NOT TOUCHED' in log_text
        assert 'abnormal termination' in log_text
        marker = next(
            line for line in log_text.splitlines()
            if line.startswith(app_backup.RESTORE_RESULT_MARKER)
        )
        assert marker.split()[1] == app_backup.RESTORE_RESULT_ABORTED

    def test_successful_database_restore_returns_nonzero_when_services_do_not_recover(
        self, monkeypatch, tmp_path
    ):
        dump_file = tmp_path / 'dump.sql'
        dump_file.write_text('SELECT 1;')
        log_file = tmp_path / 'restore.log'

        class _PsqlProcess:
            def __init__(self):
                self.stdin = _FakeStdin()

            def wait(self, timeout=None):
                return 0

        monkeypatch.setattr(
            app_backup.config, 'DATABASE_URL', 'postgresql://u:p@db:5432/music'
        )
        monkeypatch.setattr(
            app_backup.restart_manager, 'stop_local_flask_service_detail', lambda: (True, '')
        )
        monkeypatch.setattr(
            app_backup.restart_manager, 'publish_start_request', lambda **_kwargs: False
        )
        monkeypatch.setattr(app_backup.restart_manager, 'start_local_flask_service', lambda: True)
        monkeypatch.setattr(app_backup.subprocess, 'Popen', lambda *_a, **_k: _PsqlProcess())
        monkeypatch.setattr(
            app_backup.subprocess, 'run', lambda *_a, **_k: MagicMock(returncode=0)
        )
        monkeypatch.setattr(app_backup, '_release_restore_lock', lambda: None)
        monkeypatch.setitem(
            sys.modules,
            'tasks.mcp_helper',
            types.SimpleNamespace(_ensure_ai_chat_db_user=lambda: None),
        )
        monkeypatch.setitem(
            sys.modules,
            'database',
            types.SimpleNamespace(USERS_PASSWORD_CHANGED_AT_DDL='ALTER TABLE users ADD x int'),
        )

        assert app_backup._run_restore_runner(str(dump_file), str(log_file)) == 2
        log_text = log_file.read_text()
        assert 'Restore command finished with return code 0' in log_text
        assert 'service recovery FAILED' in log_text
        assert 'restored database was committed' in log_text


class TestRestoreWorkerStopSafety:
    def test_endpoint_does_not_spawn_runner_without_worker_stop_ack(
        self, client, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(app_backup, 'BACKUP_DIR', str(tmp_path))
        monkeypatch.setattr(app_backup, 'RESTORE_LOG_DIR', str(tmp_path))
        monkeypatch.setattr(app_backup, '_acquire_restore_lock', lambda: True)
        released = MagicMock()
        worker_start = MagicMock(return_value=True)
        runner = MagicMock(side_effect=AssertionError('restore runner must not start'))
        monkeypatch.setattr(app_backup, '_release_restore_lock', released)
        monkeypatch.setattr(
            app_backup.restart_manager, 'publish_stop_request', lambda **_kwargs: False
        )
        monkeypatch.setattr(app_backup.restart_manager, 'publish_start_request', worker_start)
        monkeypatch.setattr(app_backup.subprocess, 'Popen', runner)

        resp = _post(client)

        assert resp.status_code == 503
        assert 'did not confirm' in resp.get_json()['error']
        runner.assert_not_called()
        worker_start.assert_called_once_with(
            timeout_seconds=app_backup.config.QUEUE_CONTROL_ACTION_WINDOW_SECONDS
        )
        released.assert_called_once_with()

    def test_create_backup_runs_pg_dump_against_the_database_url(
        self, client, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(app_backup, 'BACKUP_DIR', str(tmp_path))
        monkeypatch.setattr(
            app_backup.config, 'DATABASE_URL', 'postgresql://audiomuse:pw@postgres:5432/audiomusedb'
        )
        seen = {}

        def fake_run(cmd, **kwargs):
            seen['cmd'] = cmd
            seen['env'] = kwargs['env']
            kwargs['stdout'].write('-- dump\n')
            return MagicMock(returncode=0, stderr='')

        monkeypatch.setattr(app_backup.subprocess, 'run', fake_run)
        assert client.post('/api/backup/create').status_code == 200
        assert 'postgresql://audiomuse@postgres:5432/audiomusedb' in seen['cmd']
        assert '-h' not in seen['cmd']
        assert seen['env']['PGPASSWORD'] == 'pw'


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
