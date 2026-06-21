"""WSGI middleware that collapses a duplicated reverse-proxy subpath prefix.

When AudioMuse-AI is served under a subpath (e.g. ``/audiomuseai``) the proxy
is expected to STRIP that prefix before forwarding (nginx ``location
/audiomuseai/ { proxy_pass http://upstream/; }`` with both trailing slashes)
while announcing it via ``X-Forwarded-Prefix``. ProxyFix then puts the prefix
in ``SCRIPT_NAME`` and ``PATH_INFO`` holds the real route (``/setup``).

A very common misconfiguration forwards the FULL path (``proxy_pass`` without a
trailing slash, or a regex ``location``) *and still* sends
``X-Forwarded-Prefix``. The prefix is then counted twice: it lands in
``SCRIPT_NAME`` (via ProxyFix) while ``PATH_INFO`` also keeps it
(``/audiomuseai/setup``). Every absolute-path check and ``url_for()`` redirect
then carries a doubled prefix, producing an infinite redirect loop to
``/<prefix>/setup`` (issue #668).

This middleware removes a leading ``SCRIPT_NAME`` from ``PATH_INFO`` so the
duplication collapses to the single, correct form. It must run AFTER ProxyFix
(i.e. be wrapped as the inner app: ``ProxyFix(StripDuplicatedScriptName(app))``)
so ``SCRIPT_NAME`` is already populated. It is a no-op for correctly configured
proxies (``PATH_INFO`` does not start with the prefix) and when no prefix is set.
"""


class StripDuplicatedScriptName:
    def __init__(self, app):
        self.app = app

    def __call__(self, environ, start_response):
        prefix = environ.get('SCRIPT_NAME', '').rstrip('/')
        if prefix:
            path_info = environ.get('PATH_INFO', '')
            if path_info == prefix or path_info.startswith(prefix + '/'):
                environ['PATH_INFO'] = path_info[len(prefix):] or '/'
        return self.app(environ, start_response)
