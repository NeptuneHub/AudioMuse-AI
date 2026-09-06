# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""HTTPS on the HTTP port, so the microphone works on a LAN address with no deployment change.

Browsers hand the microphone only to pages on HTTPS or on localhost, and a
self-hosted AudioMuse-AI is reached as http://<ip>:8000. No script can lift
that rule, and a second port would have to be published in every container
deployment, so the one port the app already binds answers both protocols. The
first byte of a new connection tells a TLS handshake (0x16) from an HTTP
request line; a TLS connection is terminated here, with a self-signed
certificate the app generates once, through a memory BIO on a single thread,
and its plaintext reaches the HTTP server (gunicorn, waitress or werkzeug)
over a local socket pair, so none of them has to know about TLS.

Main Features:
* ensure_certificate writes a 10-year RSA certificate whose subject
  alternative names cover localhost, the host name and every address the host
  answers on, through the cryptography package or, failing that, the openssl
  binary; an existing pair is reused so the browser exception stays valid
* prepare_tls loads that certificate once per process; a failure is logged and
  reported through https_status, and HTTP keeps serving
* DualProtocolListener is a listening socket whose accept() peeks the first
  byte for at most SNIFF_TIMEOUT and, for TLS, returns the app's end of a
  loopback TCP pair (a Unix pair refuses the TCP options waitress sets on
  every connection) with the real client address, while a daemon thread
  relays bytes between the client and the pair; everything else is returned
  untouched
* dual_listener binds one for waitress, adopt_listener turns the socket
  gunicorn or werkzeug already bound into one (the descriptor moves, the port
  does not), and https_status tells the page whether HTTPS answers on the
  app's port or why not
"""

import datetime
import ipaddress
import logging
import os
import select
import shutil
import socket
import ssl
import subprocess
import threading

import config
from service_roles import FLASK_BIND_HOST

logger = logging.getLogger(__name__)

CERT_FILE = 'audiomuse-https.crt'
KEY_FILE = 'audiomuse-https.key'
SNIFF_TIMEOUT = 0.25
RELAY_IDLE_SECONDS = 300.0
_VALID_DAYS = 3650
_COMMON_NAME = 'AudioMuse-AI'
_TLS_HANDSHAKE = 0x16
_CHUNK = 65536
_LOCK = threading.Lock()
_STATE = {'prepared': False, 'context': None, 'error': None}


def _host_addresses():
    names = {'localhost'}
    addresses = {'127.0.0.1', '::1'}
    try:
        host = socket.gethostname()
        if host:
            names.add(host)
            fqdn = socket.getfqdn(host)
            if fqdn:
                names.add(fqdn)
            for info in socket.getaddrinfo(host, None):
                addresses.add(info[4][0].split('%')[0])
    except OSError:
        pass
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            probe.connect(('10.255.255.255', 1))
            addresses.add(probe.getsockname()[0])
        finally:
            probe.close()
    except OSError:
        pass
    parsed = []
    for address in addresses:
        try:
            parsed.append(ipaddress.ip_address(address))
        except ValueError:
            continue
    return sorted(names), parsed


def _write_with_cryptography(cert_path, key_path, names, addresses):
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, _COMMON_NAME)])
    now = datetime.datetime.now(datetime.timezone.utc)
    alternatives = [x509.DNSName(name) for name in names] + [x509.IPAddress(address) for address in addresses]
    certificate = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=_VALID_DAYS))
        .add_extension(x509.SubjectAlternativeName(alternatives), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    _write_private(key_path, key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL, serialization.NoEncryption()
    ))
    with open(cert_path, 'wb') as handle:
        handle.write(certificate.public_bytes(serialization.Encoding.PEM))


def _write_with_openssl(cert_path, key_path, names, addresses):
    binary = shutil.which('openssl')
    if not binary:
        raise RuntimeError('neither the cryptography package nor an openssl binary is available to create the certificate')
    alternatives = ','.join([f'DNS:{name}' for name in names] + [f'IP:{address}' for address in addresses])
    subprocess.run(
        [
            binary, 'req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-sha256', '-days', str(_VALID_DAYS),
            '-subj', f'/CN={_COMMON_NAME}', '-addext', f'subjectAltName={alternatives}',
            '-keyout', key_path, '-out', cert_path,
        ],
        check=True, capture_output=True, timeout=120,
    )
    _restrict(key_path)


def _write_private(path, data):
    with open(path, 'wb') as handle:
        handle.write(data)
    _restrict(path)


def _restrict(path):
    try:
        os.chmod(path, 0o600)
    except OSError:
        logger.warning('Could not restrict the permissions of %s', path)


def ensure_certificate(directory):
    cert_path = os.path.join(directory, CERT_FILE)
    key_path = os.path.join(directory, KEY_FILE)
    if os.path.isfile(cert_path) and os.path.isfile(key_path):
        return cert_path, key_path
    os.makedirs(directory, exist_ok=True)
    names, addresses = _host_addresses()
    try:
        _write_with_cryptography(cert_path, key_path, names, addresses)
    except ImportError:
        _write_with_openssl(cert_path, key_path, names, addresses)
    logger.info('Created the self-signed HTTPS certificate in %s for %s', directory, ', '.join(names + [str(a) for a in addresses]))
    return cert_path, key_path


def prepare_tls():
    with _LOCK:
        if _STATE['prepared']:
            return _STATE['context'] is not None
        _STATE['prepared'] = True
    if not config.FLASK_BUILTIN_HTTPS:
        _STATE['error'] = 'disabled by FLASK_BUILTIN_HTTPS'
        logger.info('Built-in HTTPS disabled (FLASK_BUILTIN_HTTPS=false)')
        return False
    try:
        cert_path, key_path = ensure_certificate(config.FLASK_HTTPS_CERT_DIR)
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.load_cert_chain(cert_path, key_path)
    except Exception as exc:
        _STATE['error'] = str(exc)
        logger.exception('Built-in HTTPS could not load its certificate; the record button will explain')
        return False
    with _LOCK:
        _STATE['context'] = context
    logger.info('HTTPS answers on the HTTP port %s with the self-signed certificate in %s', config.FLASK_BIND_PORT, config.FLASK_HTTPS_CERT_DIR)
    return True


def reset_tls():
    with _LOCK:
        _STATE.update({'prepared': False, 'context': None, 'error': None})


def https_status():
    with _LOCK:
        running = _STATE['context'] is not None
        return {
            'enabled': bool(config.FLASK_BUILTIN_HTTPS),
            'running': running,
            'port': int(config.FLASK_BIND_PORT) if running else 0,
            'error': _STATE['error'] if _STATE['prepared'] and not running else None,
        }


class _TlsRelay:
    def __init__(self, context, raw, inner, addr):
        self.raw, self.inner, self.addr = raw, inner, addr
        self.incoming, self.outgoing = ssl.MemoryBIO(), ssl.MemoryBIO()
        self.tls = context.wrap_bio(self.incoming, self.outgoing, server_side=True)
        self.to_client, self.to_app = bytearray(), bytearray()
        self.handshaken = self.client_done = self.app_done = False
        self.inner_write_closed = self.raw_write_closed = False

    def _handshake(self):
        if self.handshaken:
            return
        try:
            self.tls.do_handshake()
            self.handshaken = True
        except ssl.SSLWantReadError:
            pass

    def _decrypt_client_bytes(self):
        if not self.handshaken or self.client_done:
            return
        while True:
            try:
                chunk = self.tls.read(_CHUNK)
            except ssl.SSLWantReadError:
                return
            except ssl.SSLZeroReturnError:
                self.client_done = True
                return
            if not chunk:
                self.client_done = True
                return
            self.to_app += chunk

    def _close_finished_directions(self):
        if self.client_done and not self.to_app and not self.inner_write_closed:
            self.inner.shutdown(socket.SHUT_WR)
            self.inner_write_closed = True
        if self.app_done and not self.to_client and not self.raw_write_closed:
            self.raw.shutdown(socket.SHUT_WR)
            self.raw_write_closed = True

    def _wait(self):
        readers = [sock for sock, wanted in (
            (self.raw, not self.client_done), (self.inner, self.handshaken and not self.app_done),
        ) if wanted]
        writers = [sock for sock, wanted in ((self.raw, bool(self.to_client)), (self.inner, bool(self.to_app))) if wanted]
        if not readers and not writers:
            return None, None
        ready_r, ready_w, _ = select.select(readers, writers, [], RELAY_IDLE_SECONDS)
        if not ready_r and not ready_w:
            return None, None
        return ready_r, ready_w

    def _read_client(self):
        data = self.raw.recv(_CHUNK)
        if data:
            self.incoming.write(data)
            return
        self.incoming.write_eof()
        self.client_done = True

    def _read_app(self):
        data = self.inner.recv(_CHUNK)
        if data:
            self.tls.write(data)
            return
        self.app_done = True
        try:
            self.tls.unwrap()
        except ssl.SSLError:
            pass

    def _step(self):
        self._handshake()
        self._decrypt_client_bytes()
        self.to_client += self.outgoing.read()
        self._close_finished_directions()
        ready_r, ready_w = self._wait()
        if ready_r is None:
            return False
        if self.raw in ready_r:
            self._read_client()
        if self.inner in ready_r:
            self._read_app()
        if self.raw in ready_w and self.to_client:
            del self.to_client[:self.raw.send(bytes(self.to_client[:_CHUNK]))]
        if self.inner in ready_w and self.to_app:
            del self.to_app[:self.inner.send(bytes(self.to_app[:_CHUNK]))]
        return True

    def run(self):
        self.raw.setblocking(False)
        self.inner.setblocking(False)
        try:
            while self._step():
                pass
        except OSError as exc:
            logger.debug('TLS relay with %s ended: %s', self.addr, exc)
        finally:
            for sock in (self.raw, self.inner):
                try:
                    sock.close()
                except OSError:
                    pass


def _relay(context, raw, inner, addr):
    _TlsRelay(context, raw, inner, addr).run()


def loopback_pair():
    gate = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        gate.bind(('127.0.0.1', 0))
        gate.listen(1)
        outer = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        outer.connect(gate.getsockname())
        inner, _ = gate.accept()
    finally:
        gate.close()
    for end in (inner, outer):
        end.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return inner, outer


class DualProtocolListener(socket.socket):
    def accept(self):
        conn, addr = super().accept()
        context = _STATE['context']
        if context is None:
            return conn, addr
        conn.settimeout(SNIFF_TIMEOUT)
        try:
            head = conn.recv(1, socket.MSG_PEEK)
        except OSError:
            head = b''
        conn.settimeout(None)
        if head[:1] != bytes([_TLS_HANDSHAKE]):
            return conn, addr
        inner, outer = loopback_pair()
        threading.Thread(target=_relay, args=(context, conn, outer, addr), name='tls-relay', daemon=True).start()
        return inner, addr


def adopt_listener(sock):
    if isinstance(sock, DualProtocolListener):
        return sock
    family, kind, proto = sock.family, sock.type, sock.proto
    return DualProtocolListener(family, kind, proto, fileno=sock.detach())


def dual_listener(host=FLASK_BIND_HOST, port=None, backlog=1024):
    port = int(config.FLASK_BIND_PORT if port is None else port)
    family = socket.AF_INET6 if ':' in host else socket.AF_INET
    sock = DualProtocolListener(family, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.listen(backlog)
    return sock
