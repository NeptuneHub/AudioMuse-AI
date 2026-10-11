# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The Tor side of the node: run Tor, publish the onion service, dial through Tor.

Every AudioMuse-AI server is a Tor onion service. The node starts its own Tor
process (the official prebuilt binary, TOR_BINARY) and drives it with stem, the
Tor Project's Python controller library. Tor listens only on 127.0.0.1 (a SOCKS
port and a control port) and the node's server socket listens only on
127.0.0.1; Tor publishes that socket as an onion service. Other servers reach it
through the Tor network with no open port, no public IP address and no router
setting, whatever NAT sits in front of either side. Connections are end-to-end
encrypted by Tor and neither side learns the other's IP address.

Main Features:
* start: Tor with its own data directory under TEMP_DIR (kept between starts so
  the next bootstrap is quick), bound to this process through
  __OwningControllerProcess so it never outlives the node; the bootstrap
  percentage is reported as it goes; Tor warnings go to the container log
* publish: the onion service on ONION_PORT for the local server port, with the
  stored ED25519-V3 key or a new one (returned so the caller stores it once)
* connect: a SOCKS5 CONNECT through Tor to an onion service or a host name (the
  directory relays), as a trio stream; Tor's refusals raise TorConnectError
* StreamIO adapts a trio stream to the read/write calls of the protocol
* valid_onion: a v3 onion service id (56 base32 characters, no ".onion")
"""

import os
import re
import socket
import struct
import subprocess
import tempfile

import stem
import trio
from stem.connection import AuthenticationFailure
from stem.control import Controller

import config
from p2p.errors import P2PError

ONION_PORT = 4101
BOOTSTRAP_SECONDS = 300.0
CONTROL_WAIT_SECONDS = 30.0
POLL_SECONDS = 1.0
PUBLISH_SECONDS = 180
CIRCUIT_DIRTINESS_SECONDS = 3600
STOP_SECONDS = 5.0
ONION_SUFFIX = '.onion'
_ONION_RE = re.compile(r'^[a-z2-7]{56}$')
_PROGRESS_RE = re.compile(r'PROGRESS=(\d+)')
SOCKS_REPLIES = {
    1: 'general failure', 2: 'not allowed', 3: 'network unreachable', 4: 'onion service unreachable',
    5: 'connection refused', 6: 'timed out', 7: 'command not supported', 8: 'address type not supported',
}


class TorError(P2PError):
    pass


class TorConnectError(OSError):
    pass


def valid_onion(text):
    return isinstance(text, str) and bool(_ONION_RE.match(text))


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def data_directory():
    base = str(config.TEMP_DIR or '').strip() or tempfile.gettempdir()
    return os.path.join(base, 'p2p-tor')


def torrc_lines(data_dir, socks_port, control_port):
    return [
        f'DataDirectory {data_dir}',
        f'SocksPort 127.0.0.1:{socks_port}',
        f'ControlPort 127.0.0.1:{control_port}',
        'CookieAuthentication 1',
        f'MaxCircuitDirtiness {CIRCUIT_DIRTINESS_SECONDS}',
        'Log warn stderr',
    ]


class Tor:
    def __init__(self, process, socks_port, control_port):
        self.process = process
        self.socks_port = socks_port
        self.control_port = control_port
        self.controller = None

    def running(self):
        return self.process.poll() is None

    def close(self):
        if self.controller is not None:
            try:
                self.controller.close()
            except Exception:
                pass
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=STOP_SECONDS)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=STOP_SECONDS)


def spawn(binary):
    data_dir = data_directory()
    os.makedirs(data_dir, mode=0o700, exist_ok=True)
    socks_port, control_port = free_port(), free_port()
    torrc = os.path.join(data_dir, 'torrc')
    with open(torrc, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(torrc_lines(data_dir, socks_port, control_port)) + '\n')
    process = subprocess.Popen(
        [binary, '-f', torrc, '__OwningControllerProcess', str(os.getpid())],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
    )
    return Tor(process, socks_port, control_port)


def _attach(tor):
    controller = Controller.from_port(address='127.0.0.1', port=tor.control_port)
    controller.authenticate()
    return controller


def _bootstrap_percent(controller):
    match = _PROGRESS_RE.search(controller.get_info('status/bootstrap-phase', ''))
    return int(match.group(1)) if match else 0


async def start(binary, on_progress=None):
    tor = await trio.to_thread.run_sync(spawn, binary)
    try:
        with trio.fail_after(CONTROL_WAIT_SECONDS):
            while tor.controller is None:
                if not tor.running():
                    raise TorError(f'Tor exited with code {tor.process.returncode} while starting')
                try:
                    tor.controller = await trio.to_thread.run_sync(_attach, tor)
                except (OSError, ValueError, stem.SocketError, AuthenticationFailure) as exc:
                    if not tor.running():
                        raise TorError(f'Tor exited with code {tor.process.returncode} while starting') from exc
                    await trio.sleep(POLL_SECONDS / 4)
        with trio.fail_after(BOOTSTRAP_SECONDS):
            while True:
                percent = await trio.to_thread.run_sync(_bootstrap_percent, tor.controller)
                if on_progress is not None:
                    on_progress(percent)
                if percent >= 100:
                    return tor
                if not tor.running():
                    raise TorError(f'Tor exited with code {tor.process.returncode} while connecting')
                await trio.sleep(POLL_SECONDS)
    except trio.TooSlowError as exc:
        await trio.to_thread.run_sync(tor.close)
        raise TorError('Tor could not connect to the Tor network in time') from exc
    except BaseException:
        with trio.CancelScope(shield=True):
            await trio.to_thread.run_sync(tor.close)
        raise


def _publish(controller, local_port, key_content):
    if key_content:
        response = controller.create_ephemeral_hidden_service(
            {ONION_PORT: local_port}, key_type='ED25519-V3', key_content=key_content,
            await_publication=True, timeout=PUBLISH_SECONDS,
        )
        return response.service_id, key_content
    response = controller.create_ephemeral_hidden_service(
        {ONION_PORT: local_port}, key_type='NEW', key_content='ED25519-V3',
        await_publication=True, timeout=PUBLISH_SECONDS,
    )
    return response.service_id, response.private_key


async def publish(tor, local_port, key_content):
    return await trio.to_thread.run_sync(_publish, tor.controller, local_port, key_content)


async def _receive_exactly(stream, size):
    data = b''
    while len(data) < size:
        chunk = await stream.receive_some(size - len(data))
        if not chunk:
            raise TorConnectError('Tor closed the SOCKS connection')
        data += chunk
    return data


async def connect(socks_port, host, port):
    stream = await trio.open_tcp_stream('127.0.0.1', socks_port)
    try:
        await stream.send_all(b'\x05\x01\x00')
        if await _receive_exactly(stream, 2) != b'\x05\x00':
            raise TorConnectError('Tor refused the SOCKS greeting')
        name = host.encode('ascii')
        await stream.send_all(b'\x05\x01\x00\x03' + bytes([len(name)]) + name + struct.pack('>H', int(port)))
        head = await _receive_exactly(stream, 4)
        if head[1] != 0:
            raise TorConnectError(f'Tor could not reach {host}: {SOCKS_REPLIES.get(head[1], head[1])}')
        if head[3] == 3:
            skip = (await _receive_exactly(stream, 1))[0]
        else:
            skip = 16 if head[3] == 4 else 4
        await _receive_exactly(stream, skip + 2)
        return stream
    except BaseException:
        await trio.aclose_forcefully(stream)
        raise


async def connect_onion(socks_port, onion):
    return await connect(socks_port, onion + ONION_SUFFIX, ONION_PORT)


class StreamIO:
    def __init__(self, stream):
        self.stream = stream

    async def read(self, size):
        return await self.stream.receive_some(size)

    async def write(self, data):
        await self.stream.send_all(data)
