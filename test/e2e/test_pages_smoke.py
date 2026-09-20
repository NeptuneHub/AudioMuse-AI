# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Every HTML page of the app opens in a real browser without a JavaScript error.

Each page is loaded headless at a desktop and two phone viewports against the
live stack (or against AUDIOMUSE_E2E_BASE_URL when attached to a running
instance), its key element must attach, and no console error, uncaught
exception, failed same-origin request or 5xx response may occur while it
settles. A failing page leaves a screenshot and a Playwright trace under the
run directory that CI uploads.

Main Features:
* one test per page and viewport, key selector taken from the template
* console, pageerror, requestfailed and 5xx collectors turn front-end
  breakage into a readable assertion
* screenshot and trace on failure, never networkidle (pages poll on a timer)
"""

import os
import re
import urllib.parse

import pytest

from test.e2e.stack import paths

pytestmark = [pytest.mark.e2e, pytest.mark.browser]

PAGES = [
    ('/', '#key-numbers'),
    ('/browse', '#browse-title'),
    ('/analysis', '#basic-view-btn'),
    ('/cleaning', '#start-clean-btn'),
    ('/chat/', '#playlistForm'),
    ('/similarity', '#similarity-form'),
    ('/artist_similarity', '#artist-similarity-form'),
    ('/album_creation', '#album-creation-form'),
    ('/path', '#path-form'),
    ('/alchemy', '#alchemy-form'),
    ('/clap_search', '#search-form'),
    ('/lyrics_search', '#axis-form'),
    ('/recording_search', '#recording-panel'),
    ('/map', '#map_size'),
    ('/sonic_fingerprint', '#fingerprint-form'),
    ('/hyperbolic', '#hyper-similar-form'),
    ('/cron', '#analysis-cron'),
    ('/backup', '#create-backup-btn'),
    ('/provider-migration', '#mig-step-1'),
    ('/setup', '#setup-form'),
    ('/users', '.page-header'),
    ('/plugins', '#plugins-apply-btn'),
    ('/login', '#key-numbers'),
]

VIEWPORTS = {
    'desktop': {'viewport': {'width': 1366, 'height': 768}},
    'phone-393': {'viewport': {'width': 393, 'height': 852}, 'is_mobile': True, 'has_touch': True},
    'phone-440': {'viewport': {'width': 440, 'height': 956}, 'is_mobile': True, 'has_touch': True},
}

IGNORED_FAILURES = ('net::ERR_ABORTED',)
SETTLE_MS = 1000
SELECTOR_TIMEOUT_MS = 10000


def _safe(name):
    return re.sub(r'[^A-Za-z0-9._-]+', '_', name)[:120]


@pytest.fixture(scope='session')
def artifact_dir(page_base_url):
    os.makedirs(paths.PLAYWRIGHT_DIR, exist_ok=True)
    return paths.PLAYWRIGHT_DIR


@pytest.fixture
def smoke_page(browser, page_base_url, artifact_dir, request):
    viewport_name = request.node.callspec.params['viewport']
    context = browser.new_context(base_url=page_base_url, **VIEWPORTS[viewport_name])
    context.tracing.start(screenshots=True, snapshots=True)
    page = context.new_page()
    origin = urllib.parse.urlsplit(page_base_url).netloc
    problems = []

    def same_origin(url):
        return urllib.parse.urlsplit(url).netloc == origin

    page.on('console', lambda msg: problems.append(f'console.{msg.type}: {msg.text}') if msg.type == 'error' else None)
    page.on('pageerror', lambda exc: problems.append(f'pageerror: {exc}'))
    page.on(
        'requestfailed',
        lambda req: problems.append(f'requestfailed: {req.url} {req.failure}')
        if same_origin(req.url) and not any(code in str(req.failure) for code in IGNORED_FAILURES)
        else None,
    )
    page.on(
        'response',
        lambda res: problems.append(f'http {res.status}: {res.url}')
        if same_origin(res.url) and res.status >= 500
        else None,
    )
    yield page, problems
    failed = getattr(request.node, 'rep_call', None) is not None and request.node.rep_call.failed
    stem = os.path.join(artifact_dir, _safe(request.node.name))
    if failed:
        try:
            page.screenshot(path=stem + '.png', full_page=True)
        except Exception:
            pass
        context.tracing.stop(path=stem + '.zip')
    else:
        context.tracing.stop()
    context.close()


@pytest.mark.parametrize('viewport', list(VIEWPORTS))
@pytest.mark.parametrize('path,selector', PAGES, ids=[p[0] for p in PAGES])
def test_page_renders_without_errors(smoke_page, path, selector, viewport):
    page, problems = smoke_page
    response = page.goto(path, wait_until='domcontentloaded')
    assert response is not None, f'{path}: {response and response.status}'
    assert response.ok, f'{path}: {response and response.status}'
    page.locator(selector).first.wait_for(state='attached', timeout=SELECTOR_TIMEOUT_MS)
    page.locator('#sidebar').first.wait_for(state='attached', timeout=SELECTOR_TIMEOUT_MS)
    page.wait_for_timeout(SETTLE_MS)
    assert page.title(), 'empty title'
    if path == '/login':
        assert urllib.parse.urlsplit(page.url).path == '/', page.url
    assert not problems, f'{path} at {viewport}:\n' + '\n'.join(problems)
