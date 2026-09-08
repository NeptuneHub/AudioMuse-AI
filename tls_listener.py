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
* DualProtocolListener is a listening socket with its own sniffing thread:
  that thread accepts every connection, waits up to SNIFF_TIMEOUT for its
  first byte on a selector (never one connection at a time, so an idle
  browser preconnect delays nobody), hands an HTTP connection to the server
  untouched and, for TLS, hands over the app's end of a loopback TCP pair (a
  Unix pair refuses the TCP options waitress sets on every connection) with
  the real client address while a daemon thread relays bytes between the
  client and the pair; the server's accept() only pops the next sniffed
  connection, and fileno() is a wake-up descriptor that becomes readable
  exactly when one is ready, so gunicorn, waitress and werkzeug keep their
  own select loops unchanged
* the relay keeps at most _HIGH_WATER bytes in flight per direction: a client
  sending faster than the app reads is simply not read from until the app
  catches up, so a large upload costs no memory beyond that
* dual_listener binds one for waitress, adopt_listener turns the socket
  gunicorn or werkzeug already bound into one (the descriptor moves, the port
  does not), and https_status tells the page whether HTTPS answers on the
  app's port or why not
"""

import datetime
import errno
import ipaddress
import logging
import os
import queue
import select
import selectors
import shutil
import socket
import ssl
import subprocess
import tempfile
import threading
import time

import config
from service_roles import FLASK_BIND_HOST

logger = logging.getLogger(__name__)

CERT_FILE = 'audiomuse-https.crt'
KEY_FILE = 'audiomuse-https.key'
SNIFF_TIMEOUT = 2.0
RELAY_IDLE_SECONDS = 300.0
_VALID_DAYS = 3650
_COMMON_NAME = 'AudioMuse-AI'
_TLS_HANDSHAKE = 0x16
_CHUNK = 65536
_HIGH_WATER = 1 << 20
_LOCK = threading.Lock()
_STATE = {'prepared': False, 'context': None, 'error': None, 'adopted': 0, 'cert_dir': None}
_NOT_PREPARED = (
    'the web server never ran the HTTPS hook: gunicorn did not load gunicorn.conf.py '
    '(start it from /app or set GUNICORN_CMD_ARGS="--config /app/gunicorn.conf.py"), '
    'or the server entry point is not one of gunicorn, waitress or app.py'
)


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


def writable_cert_dir():
    wanted = config.FLASK_HTTPS_CERT_DIR
    try:
        os.makedirs(wanted, exist_ok=True)
        if os.access(wanted, os.W_OK):
            return wanted
        raise PermissionError(f'{wanted} is not writable')
    except OSError as exc:
        fallback = os.path.join(tempfile.gettempdir(), 'audiomuse_tls')
        logger.warning(
            'The HTTPS certificate cannot be stored in %s (%s); using %s instead, so the browser warning '
            'returns after every restart. Set FLASK_HTTPS_CERT_DIR to a writable directory to keep it.',
            wanted, exc, fallback,
        )
        os.makedirs(fallback, exist_ok=True)
        return fallback


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
        directory = writable_cert_dir()
        cert_path, key_path = ensure_certificate(directory)
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.load_cert_chain(cert_path, key_path)
    except Exception as exc:
        _STATE['error'] = str(exc)
        logger.exception('Built-in HTTPS could not load its certificate; the record button will explain')
        return False
    with _LOCK:
        _STATE['context'] = context
        _STATE['cert_dir'] = directory
    logger.info('HTTPS answers on the HTTP port %s with the self-signed certificate in %s', config.FLASK_BIND_PORT, directory)
    return True


def reset_tls():
    with _LOCK:
        _STATE.update({'prepared': False, 'context': None, 'error': None, 'cert_dir': None})


def https_status():
    with _LOCK:
        running = _STATE['context'] is not None
        if running or not config.FLASK_BUILTIN_HTTPS:
            reason = None if running else 'disabled by FLASK_BUILTIN_HTTPS'
        elif not _STATE['prepared']:
            reason = _NOT_PREPARED
        else:
            reason = _STATE['error']
        return {
            'enabled': bool(config.FLASK_BUILTIN_HTTPS),
            'running': running,
            'port': int(config.FLASK_BIND_PORT) if running else 0,
            'error': reason,
            'prepared': bool(_STATE['prepared']),
            'adopted': int(_STATE['adopted']),
            'cert_dir': _STATE['cert_dir'],
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

    def _interest(self):
        readers = [sock for sock, wanted in (
            (self.raw, not self.client_done and len(self.to_app) < _HIGH_WATER),
            (self.inner, self.handshaken and not self.app_done and len(self.to_client) < _HIGH_WATER),
        ) if wanted]
        writers = [sock for sock, wanted in ((self.raw, bool(self.to_client)), (self.inner, bool(self.to_app))) if wanted]
        return readers, writers

    def _wait(self):
        readers, writers = self._interest()
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
            running = True
            while running:
                running = self._step()
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


def _is_tls(head):
    return head[:1] == bytes([_TLS_HANDSHAKE])


def _sniff_wait(pending):
    if not pending:
        return None
    return max(0.0, min(deadline for deadline, _addr in pending.values()) - time.monotonic())


def _not_ready():
    return BlockingIOError(errno.EWOULDBLOCK, 'no sniffed connection is ready yet')


class DualProtocolListener(socket.socket):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ready = queue.SimpleQueue()
        self._wake_r, self._wake_w = socket.socketpair()
        self._wake_w.setblocking(False)
        self._sniffer = None

    def fileno(self):
        return self._wake_r.fileno()

    def start(self):
        if self._sniffer is None:
            socket.socket.setblocking(self, False)
            self._sniffer = threading.Thread(target=self._sniff_loop, name='tls-sniff', daemon=True)
            self._sniffer.start()
        return self

    def accept(self):
        self._wake_r.settimeout(self.gettimeout())
        try:
            self._wake_r.recv(1)
        except (BlockingIOError, InterruptedError):
            raise _not_ready() from None
        try:
            return self._ready.get_nowait()
        except queue.Empty:
            raise _not_ready() from None

    def close(self):
        super().close()
        for end in (self._wake_r, self._wake_w):
            try:
                end.close()
            except OSError:
                pass

    def _sniff_loop(self):
        selector = selectors.DefaultSelector()
        pending = {}
        try:
            selector.register(socket.socket.fileno(self), selectors.EVENT_READ, data=None)
            while True:
                events = selector.select(_sniff_wait(pending))
                now = time.monotonic()
                for key, _mask in events:
                    self._service(selector, pending, key, now)
                self._expire(selector, pending, now)
        except (OSError, ValueError):
            logger.debug('The dual-protocol listener stopped sniffing', exc_info=True)
        finally:
            for conn in pending:
                try:
                    conn.close()
                except OSError:
                    pass
            selector.close()

    def _service(self, selector, pending, key, now):
        if key.data is None:
            self._take_one(selector, pending, now)
            return
        conn = key.fileobj
        try:
            head = conn.recv(1, socket.MSG_PEEK)
        except (BlockingIOError, InterruptedError):
            return
        except OSError:
            head = b''
        self._settle(selector, pending, conn, head)

    def _expire(self, selector, pending, now):
        expired = [conn for conn, (deadline, _addr) in pending.items() if deadline <= now]
        for conn in expired:
            self._settle(selector, pending, conn, b'')

    def _take_one(self, selector, pending, now):
        try:
            conn, addr = socket.socket.accept(self)
        except (BlockingIOError, InterruptedError, ConnectionAbortedError):
            return
        conn.setblocking(False)
        pending[conn] = (now + SNIFF_TIMEOUT, addr)
        selector.register(conn, selectors.EVENT_READ, data=addr)

    def _settle(self, selector, pending, conn, head):
        _deadline, addr = pending.pop(conn)
        selector.unregister(conn)
        conn.setblocking(True)
        if not _is_tls(head):
            self._hand(conn, addr)
            return
        context = _STATE['context']
        inner, outer = loopback_pair()
        if context is None:
            logger.warning('Refused a TLS connection from %s: built-in HTTPS is not running (%s)', addr, https_status()['error'])
            conn.close()
            outer.close()
            self._hand(inner, addr)
            return
        threading.Thread(target=_relay, args=(context, conn, outer, addr), name='tls-relay', daemon=True).start()
        self._hand(inner, addr)

    def _hand(self, sock, addr):
        self._ready.put((sock, addr))
        try:
            self._wake_w.send(b'x')
        except OSError:
            logger.debug('The dual-protocol listener could not signal a ready connection', exc_info=True)


def _count_adopted():
    with _LOCK:
        _STATE['adopted'] += 1


def adopt_listener(sock):
    if isinstance(sock, DualProtocolListener):
        return sock
    family, kind, proto = sock.family, sock.type, sock.proto
    _count_adopted()
    return DualProtocolListener(family, kind, proto, fileno=sock.detach()).start()


def dual_listener(host=FLASK_BIND_HOST, port=None, backlog=1024):
    port = int(config.FLASK_BIND_PORT if port is None else port)
    family = socket.AF_INET6 if ':' in host else socket.AF_INET
    sock = DualProtocolListener(family, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.listen(backlog)
    _count_adopted()
    return sock.start()
