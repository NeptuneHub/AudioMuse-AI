# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Session fixtures that boot the real AudioMuse-AI stack for the end-to-end tests.

`stack` starts Postgres, Navidrome, gunicorn and two queue workers once per
session (see stack/boot.py) and every other fixture is a view on it: `api` for
HTTP, `db` for read-only SQL assertions, `navidrome` for provider-side
assertions and seeding, `library` for the committed fixture files, and
`analyzed_library` for the one full analysis most modules depend on. A stack
failure is a fixture failure that names the step and quotes the log tail; the
only skip is the absence of any database on a developer machine.

Main Features:
* stack / api / db / navidrome / library session fixtures
* seeded_catalogue inserts the seed rows (test/e2e/seed) bound to the
  placeholder files, before the analysis
* analyzed_library runs POST /api/analysis/start on every album once and waits
  for the web process to have loaded every index it builds
* an autouse liveness check fails the next test when a stack process died
"""

import os
import sys
import time

import pytest

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from test.e2e.golden import RECORD_ENV, Golden, Resolver  # noqa: E402
from test.e2e.stack import postgres  # noqa: E402
from test.e2e.stack.api import TaskOutcomeError  # noqa: E402
from test.e2e.stack.boot import Stack, require_linux  # noqa: E402
from test.e2e.stack.errors import StackError  # noqa: E402

_CURRENT = {'stack': None}

ANALYSIS_TIMEOUT_SECONDS = 2400
INDEX_LOAD_TIMEOUT_SECONDS = 120
ATTACH_ENV = 'AUDIOMUSE_E2E_BASE_URL'


class AnalysisPreconditionFailed(Exception):
    pass


class AnalyzedLibrary:
    def __init__(self, task_id, final, library, fresh):
        self.task_id = task_id
        self.final = final
        self.library = library
        self.fresh = fresh
        self.ready_at = time.time()


@pytest.fixture(scope='session')
def stack():
    try:
        require_linux()
    except StackError as exc:
        pytest.fail(str(exc), pytrace=False)
    instance = Stack()
    try:
        booted = instance.boot()
    except StackError as exc:
        instance.shutdown()
        pytest.fail(str(exc), pytrace=False)
    if not booted:
        instance.shutdown()
        pytest.skip(postgres.INSTALL_HINT)
    _CURRENT['stack'] = instance
    yield instance
    _CURRENT['stack'] = None
    instance.shutdown()


@pytest.fixture(scope='session')
def api(stack):
    return stack.api


@pytest.fixture(scope='session')
def navidrome(stack):
    return stack.subsonic


@pytest.fixture(scope='session')
def library(stack):
    return stack.library


@pytest.fixture
def db(stack):
    conn = postgres.connect(stack.dsn)
    yield conn
    conn.close()


def _indexes_loaded(api):
    checks = (
        ('/api/clap/stats', 'loaded'),
        ('/api/lyrics/stats', 'loaded'),
        ('/api/sem_grove/stats', 'loaded'),
        ('/api/map_cache_status', 'ok'),
    )
    for path, key in checks:
        response = api.get(path)
        if response.status_code != 200 or not response.json().get(key):
            return False
    return True


@pytest.fixture(scope='session')
def seeded_catalogue(stack):
    if stack.seed.count == 0:
        return stack.seed
    conn = postgres.connect(stack.dsn, autocommit=False)
    try:
        server_id = postgres.scalar(stack.dsn, 'SELECT server_id FROM music_servers WHERE is_default')
        stack.seed.apply(conn, server_id, stack.subsonic.all_songs())
    except StackError as exc:
        pytest.fail(str(exc), pytrace=False)
    finally:
        conn.close()
    return stack.seed


@pytest.fixture(scope='session')
def analyzed_library(stack, seeded_catalogue):
    api = stack.api
    api.wait_idle(300)
    fresh = postgres.scalar(stack.dsn, 'SELECT count(*) FROM score') <= stack.seed_count
    task_id = api.start_task('/api/analysis/start', {'num_recent_albums': 0, 'top_n_moods': 5})
    try:
        final = api.wait_for_task(task_id, timeout=ANALYSIS_TIMEOUT_SECONDS)
    except TaskOutcomeError as exc:
        raise AnalysisPreconditionFailed(
            f'the initial library analysis failed, every dependent test is void: {exc}'
        ) from exc
    api.wait_idle(120)
    deadline = time.monotonic() + INDEX_LOAD_TIMEOUT_SECONDS
    while not _indexes_loaded(api):
        if time.monotonic() >= deadline:
            raise AnalysisPreconditionFailed(
                'analysis succeeded but the web process did not load its indexes within '
                f'{INDEX_LOAD_TIMEOUT_SECONDS}s. ' + stack.flask.process.describe()
            )
        time.sleep(2)
    return AnalyzedLibrary(task_id, final, stack.library, fresh)


@pytest.fixture(scope='session')
def golden_resolver(stack, analyzed_library):
    resolver = Resolver.build(stack)
    conn = postgres.connect(stack.dsn)
    try:
        with conn.cursor() as cur:
            cur.execute('SELECT provider_artist_id, artist_name FROM artist_server_map')
            resolver.add_artists(cur.fetchall())
            cur.execute('SELECT server_id, name FROM music_servers')
            resolver.add_servers(cur.fetchall())
    finally:
        conn.close()
    return resolver


@pytest.fixture(scope='module')
def golden(request, golden_resolver):
    recorder = Golden(request.module.__name__.rsplit('.', 1)[-1], golden_resolver, bool(os.environ.get(RECORD_ENV, '').strip()))
    yield recorder
    recorder.flush()


@pytest.fixture(scope='session')
def page_base_url(request):
    attached = os.environ.get(ATTACH_ENV, '').strip()
    if attached:
        return attached.rstrip('/')
    instance = request.getfixturevalue('stack')
    request.getfixturevalue('analyzed_library')
    return instance.base_url


@pytest.fixture(autouse=True)
def _stack_alive():
    instance = _CURRENT['stack']
    if instance is None:
        return
    problems = instance.alive_problems()
    if problems:
        pytest.fail('a stack process is gone:\n' + '\n'.join(problems), pytrace=False)


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    setattr(item, f'rep_{report.when}', report)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    instance = _CURRENT['stack']
    if instance is None:
        return
    terminalreporter.section('end-to-end stack')
    terminalreporter.write_line(instance.summary())
