# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""HTTPS on the HTTP port through the dual-protocol listener.

Main Features:
* the self-signed certificate is created once, names localhost and the host's
  addresses, keeps its private key readable by the owner only, and is reused
  untouched on the next call
* one bound port answers a plain HTTP request and an HTTPS request alike,
  the app sees the real client address, and a client that sends its first
  byte late is still served as HTTP
* with built-in HTTPS disabled the listener hands every connection through
  untouched and the status says why; a certificate failure is reported, not
  raised
* the gunicorn worker hook adopts the sockets the worker inherited without
  changing their descriptor, and the waitress entry point serves on a
  dual-protocol socket bound to the HTTP host and port
"""

import http.client
import importlib.util
import os
import socket
import ssl
import sys
import threading
import time
import types
import urllib.error
import urllib.request

import pytest
from werkzeug.serving import make_server

import config
import tls_listener


def _ok_app(environ, start_response):
    start_response('200 OK', [('Content-Type', 'text/plain')])
    return [f"ok from {environ.get('REMOTE_ADDR')}".encode()]


@pytest.fixture(autouse=True)
def fresh_tls(monkeypatch, tmp_path):
    monkeypatch.setattr(config, 'FLASK_HTTPS_CERT_DIR', str(tmp_path))
    monkeypatch.setattr(config, 'FLASK_BUILTIN_HTTPS', True)
    tls_listener.reset_tls()
    yield
    tls_listener.reset_tls()


@pytest.fixture
def dual_server():
    server = make_server('127.0.0.1', 0, _ok_app, threaded=True)
    server.socket = tls_listener.adopt_listener(server.socket)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server.server_port
    server.shutdown()
    server.server_close()


def _insecure():
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    return context


def test_certificate_is_created_once_names_the_host_and_keeps_its_key_private(tmp_path):
    cert_path, key_path = tls_listener.ensure_certificate(str(tmp_path))
    assert os.path.isfile(cert_path)
    assert os.path.isfile(key_path)
    from cryptography import x509

    certificate = x509.load_pem_x509_certificate(open(cert_path, 'rb').read())
    names = certificate.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    assert 'localhost' in names.get_values_for_type(x509.DNSName)
    assert any(str(a) == '127.0.0.1' for a in names.get_values_for_type(x509.IPAddress))
    assert (certificate.not_valid_after_utc - certificate.not_valid_before_utc).days > 3000
    if os.name != 'nt':
        assert oct(os.stat(key_path).st_mode & 0o777) == '0o600'
    before = open(cert_path, 'rb').read()
    assert tls_listener.ensure_certificate(str(tmp_path)) == (cert_path, key_path)
    assert open(cert_path, 'rb').read() == before


def test_one_port_answers_http_and_https_and_reports_the_real_client(dual_server):
    assert tls_listener.prepare_tls() is True
    port = dual_server
    with urllib.request.urlopen(f'http://127.0.0.1:{port}/', timeout=10) as response:
        assert response.read() == b'ok from 127.0.0.1'
    with urllib.request.urlopen(f'https://127.0.0.1:{port}/', context=_insecure(), timeout=10) as response:
        assert response.read() == b'ok from 127.0.0.1'
    with pytest.raises(urllib.error.URLError):
        urllib.request.urlopen(f'https://127.0.0.1:{port}/', timeout=10)
    connection = http.client.HTTPSConnection('127.0.0.1', port, context=_insecure(), timeout=10)
    for _ in range(3):
        connection.request('GET', '/')
        assert connection.getresponse().read() == b'ok from 127.0.0.1'
    connection.close()
    assert tls_listener.https_status() == {'enabled': True, 'running': True, 'port': int(config.FLASK_BIND_PORT), 'error': None}


def test_a_client_whose_first_byte_arrives_late_is_still_served_as_http(dual_server):
    tls_listener.prepare_tls()
    raw = socket.create_connection(('127.0.0.1', dual_server), timeout=10)
    time.sleep(tls_listener.SNIFF_TIMEOUT * 2)
    raw.sendall(b'GET / HTTP/1.0\r\nHost: x\r\n\r\n')
    reply = b''
    while True:
        part = raw.recv(4096)
        if not part:
            break
        reply += part
    raw.close()
    assert reply.startswith(b'HTTP/1.')
    assert b' 200 ' in reply.split(b'\r\n')[0]
    assert b'ok from 127.0.0.1' in reply


def test_disabled_https_passes_everything_through_and_a_bad_certificate_is_reported(monkeypatch, dual_server, tmp_path):
    monkeypatch.setattr(config, 'FLASK_BUILTIN_HTTPS', False)
    assert tls_listener.prepare_tls() is False
    assert tls_listener.https_status() == {'enabled': False, 'running': False, 'port': 0, 'error': 'disabled by FLASK_BUILTIN_HTTPS'}
    with urllib.request.urlopen(f'http://127.0.0.1:{dual_server}/', timeout=10) as response:
        assert response.read() == b'ok from 127.0.0.1'
    tls_listener.reset_tls()
    monkeypatch.setattr(config, 'FLASK_BUILTIN_HTTPS', True)
    broken = tmp_path / 'broken'
    broken.mkdir()
    (broken / tls_listener.CERT_FILE).write_text('not a certificate')
    (broken / tls_listener.KEY_FILE).write_text('not a key')
    monkeypatch.setattr(config, 'FLASK_HTTPS_CERT_DIR', str(broken))
    assert tls_listener.prepare_tls() is False
    status = tls_listener.https_status()
    assert status['enabled'] is True
    assert status['running'] is False
    assert status['port'] == 0
    assert status['error']
    with urllib.request.urlopen(f'http://127.0.0.1:{dual_server}/', timeout=10) as response:
        assert response.read() == b'ok from 127.0.0.1'


def test_the_relay_pair_is_a_tcp_pair_that_accepts_the_options_servers_set():
    inner, outer = tls_listener.loopback_pair()
    try:
        assert inner.family == socket.AF_INET
        assert outer.family == socket.AF_INET
        inner.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        inner.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        outer.sendall(b'GET / HTTP/1.1\r\n')
        assert inner.recv(64) == b'GET / HTTP/1.1\r\n'
        inner.sendall(b'HTTP/1.1 200 OK\r\n')
        assert outer.recv(64) == b'HTTP/1.1 200 OK\r\n'
    finally:
        inner.close()
        outer.close()


def test_the_gunicorn_hook_adopts_the_inherited_sockets_without_moving_the_port():
    repo_root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
    spec = importlib.util.spec_from_file_location('audiomuse_gunicorn_conf', os.path.join(repo_root, 'gunicorn.conf.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules['audiomuse_gunicorn_conf'] = module
    spec.loader.exec_module(module)
    bound = socket.socket()
    bound.bind(('127.0.0.1', 0))
    bound.listen(1)
    fd, port = bound.fileno(), bound.getsockname()[1]
    listener = types.SimpleNamespace(sock=bound)
    module.post_worker_init(types.SimpleNamespace(sockets=[listener]))
    try:
        assert isinstance(listener.sock, tls_listener.DualProtocolListener)
        assert listener.sock.fileno() == fd
        assert listener.sock.getsockname()[1] == port
        assert tls_listener.https_status()['running'] is True
        assert tls_listener.adopt_listener(listener.sock) is listener.sock
    finally:
        listener.sock.close()


def test_waitress_serves_on_a_dual_protocol_socket_bound_to_the_http_port(monkeypatch):
    import service_roles

    seen = {}
    fake_waitress = types.SimpleNamespace(serve=lambda app, **kw: seen.update(kw))
    monkeypatch.setitem(sys.modules, 'waitress', fake_waitress)
    monkeypatch.setitem(sys.modules, 'app', types.SimpleNamespace(app=_ok_app))
    bound = tls_listener.dual_listener('127.0.0.1', 0)
    monkeypatch.setattr(tls_listener, 'dual_listener', lambda host, port: bound)
    try:
        service_roles.serve_flask()
        assert seen['sockets'] == [bound]
        assert 'host' not in seen
        assert 'port' not in seen
        assert seen['threads'] == service_roles.FLASK_THREADS
        assert isinstance(bound, tls_listener.DualProtocolListener)
        assert bound.getsockname()[0] == '127.0.0.1'
        assert tls_listener.https_status()['running'] is True
    finally:
        bound.close()
