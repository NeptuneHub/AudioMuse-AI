"""Regression tests for the reverse-proxy subpath prefix handling (issue #668).

The bug: behind a reverse proxy that forwards the FULL path (e.g. nginx
``proxy_pass`` without a trailing slash) while ALSO sending
``X-Forwarded-Prefix``, ProxyFix puts the prefix in ``SCRIPT_NAME`` while
``PATH_INFO`` still carries it. ``request.path`` then becomes
``/audiomuseai/setup`` instead of ``/setup``; the setup barrier's literal
comparison fails and it redirects to ``url_for('setup_page')`` ==
``/audiomuseai/setup`` -- the same URL -- forever.

``StripDuplicatedScriptName`` collapses the duplicated prefix so both the
correctly-configured and the misconfigured proxy work, and the loop is
impossible.
"""
from flask import Flask, request, redirect, url_for, jsonify
from werkzeug.middleware.proxy_fix import ProxyFix

from proxy_prefix import StripDuplicatedScriptName


# --- the middleware in isolation -------------------------------------------

def _run(environ):
    captured = {}

    def downstream(env, start_response):
        captured['SCRIPT_NAME'] = env.get('SCRIPT_NAME', '')
        captured['PATH_INFO'] = env.get('PATH_INFO', '')
        return []

    StripDuplicatedScriptName(downstream)(environ, lambda *a, **k: None)
    return captured


class TestStripDuplicatedScriptName:
    def test_no_prefix_is_noop(self):
        out = _run({'SCRIPT_NAME': '', 'PATH_INFO': '/setup'})
        assert out['PATH_INFO'] == '/setup'

    def test_correctly_stripped_path_is_noop(self):
        # nginx already stripped the prefix: PATH_INFO does not start with it.
        out = _run({'SCRIPT_NAME': '/audiomuseai', 'PATH_INFO': '/setup'})
        assert out['PATH_INFO'] == '/setup'

    def test_duplicated_prefix_is_collapsed(self):
        out = _run({'SCRIPT_NAME': '/audiomuseai', 'PATH_INFO': '/audiomuseai/setup'})
        assert out['PATH_INFO'] == '/setup'

    def test_duplicated_prefix_bare_root(self):
        out = _run({'SCRIPT_NAME': '/audiomuseai', 'PATH_INFO': '/audiomuseai/'})
        assert out['PATH_INFO'] == '/'

    def test_path_equal_to_prefix_becomes_root(self):
        out = _run({'SCRIPT_NAME': '/audiomuseai', 'PATH_INFO': '/audiomuseai'})
        assert out['PATH_INFO'] == '/'

    def test_prefix_with_trailing_slash_normalized(self):
        out = _run({'SCRIPT_NAME': '/audiomuseai/', 'PATH_INFO': '/audiomuseai/setup'})
        assert out['PATH_INFO'] == '/setup'

    def test_similar_but_unrelated_path_untouched(self):
        # /amazing must NOT be treated as carrying the /am prefix.
        out = _run({'SCRIPT_NAME': '/am', 'PATH_INFO': '/amazing'})
        assert out['PATH_INFO'] == '/amazing'


# --- end-to-end: the real setup barrier behind the real middleware ----------

def _barrier_app(with_fix):
    app = Flask(__name__)

    @app.route('/')
    def dashboard_page():
        return 'DASH'

    @app.route('/setup')
    def setup_page():
        return 'SETUP'

    @app.before_request
    def barrier():
        # Mirror app_auth.auth_setup_barrier's setup-needed branch with setup
        # unconditionally required.
        if request.path in ('/setup', '/api/setup'):
            return
        if request.path.startswith('/api/'):
            return jsonify(error='Setup required'), 403
        return redirect(url_for('setup_page'))

    inner = StripDuplicatedScriptName(app.wsgi_app) if with_fix else app.wsgi_app
    app.wsgi_app = ProxyFix(inner, x_for=1, x_proto=1, x_host=1, x_prefix=1)
    return app


_PROXY_HEADERS = {
    'X-Forwarded-Proto': 'https',
    'X-Forwarded-Host': 'host',
    'X-Forwarded-Prefix': '/audiomuseai',
}


class TestBarrierUnderMisconfiguredProxy:
    def test_loop_without_fix(self):
        # Reproduce the bug: full path forwarded + X-Forwarded-Prefix -> the
        # barrier redirects /audiomuseai/setup back to /audiomuseai/setup.
        app = _barrier_app(with_fix=False)
        rv = app.test_client().get('/audiomuseai/setup', headers=_PROXY_HEADERS)
        assert rv.status_code == 302
        assert rv.headers['Location'] == '/audiomuseai/setup'  # points at itself

    def test_no_loop_with_fix(self):
        app = _barrier_app(with_fix=True)
        rv = app.test_client().get('/audiomuseai/setup', headers=_PROXY_HEADERS)
        assert rv.status_code == 200
        assert rv.get_data(as_text=True) == 'SETUP'

    def test_recommended_config_still_works_with_fix(self):
        # nginx strips the prefix: app receives /setup; must keep working.
        app = _barrier_app(with_fix=True)
        rv = app.test_client().get('/setup', headers=_PROXY_HEADERS)
        assert rv.status_code == 200
        assert rv.get_data(as_text=True) == 'SETUP'

    def test_root_redirect_target_is_prefixed_with_fix(self):
        app = _barrier_app(with_fix=True)
        rv = app.test_client().get('/audiomuseai/', headers=_PROXY_HEADERS)
        assert rv.status_code == 302
        assert rv.headers['Location'] == '/audiomuseai/setup'
