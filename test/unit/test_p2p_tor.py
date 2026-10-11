# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The Tor side of the P2P node (p2p.tor), with no Tor network involved.

A tiny SOCKS5 server on 127.0.0.1 plays Tor's SOCKS port, fake executables
play the Tor program, and fake stem controllers play its control port, so
every path of the module runs offline and fast.

Main Features:
* valid_onion accepts exactly 56 base32 characters, nothing with ".onion",
  capitals, the digits 0, 1, 8, 9 or another length
* torrc_lines binds the SOCKS and control ports to 127.0.0.1 only, never
  opens an ORPort or DirPort, and sets CookieAuthentication,
  MaxCircuitDirtiness and a warning log
* connect sends a SOCKS5 domain CONNECT and consumes exactly the reply for an
  IPv4, IPv6 or domain bound address; every SOCKS error code raises
  TorConnectError with the readable reason; a refused greeting or a closed
  connection raises TorConnectError; connect_onion dials the onion service
  port of "<id>.onion"
* StreamIO reads and writes a trio stream
* start: a Tor program that exits at once is a TorError naming its exit code,
  after it was started on the torrc with __OwningControllerProcess; one that
  never opens its control port is stopped and reported; the bootstrap
  percentage is reported until 100; a Tor that dies while connecting is a
  TorError and its controller is closed
* publish uses the stored ED25519-V3 key as it is, or asks Tor for a NEW key
  and returns it for the caller to store; both wait for publication
* Tor.close terminates the process and kills it when it does not stop
"""

import os
import socket
import stat
import struct
import subprocess
from functools import partial
from types import SimpleNamespace

import pytest
import trio
import trio.testing

import config
from p2p import tor as p2p_tor
from p2p.errors import P2PError

ONION = 'torservice' + 'z' * 46
POSIX_ONLY = pytest.mark.skipif(os.name == 'nt', reason='the fake Tor program is a POSIX shell script')


async def _exactly(stream, size):
    data = b''
    while len(data) < size:
        chunk = await stream.receive_some(size - len(data))
        if not chunk:
            return data
        data += chunk
    return data


class FakeSocks:
    def __init__(self, code=0, atyp=1, bound=b'\x7f\x00\x00\x01', greeting=b'\x05\x00', after=b'',
                 close_after_greeting=False, reply=None):
        self.code = code
        self.atyp = atyp
        self.bound = bound
        self.greeting = greeting
        self.after = after
        self.close_after_greeting = close_after_greeting
        self.reply = reply
        self.greetings = []
        self.requests = []

    async def handler(self, stream):
        self.greetings.append(await _exactly(stream, 3))
        if self.close_after_greeting:
            await stream.aclose()
            return
        await stream.send_all(self.greeting)
        if self.greeting != b'\x05\x00':
            await stream.receive_some(1)
            return
        head = await _exactly(stream, 5)
        name = await _exactly(stream, head[4])
        port = struct.unpack('>H', await _exactly(stream, 2))[0]
        self.requests.append((head[:4], name.decode('ascii'), port))
        if self.reply is not None:
            await stream.send_all(self.reply)
            await stream.aclose()
            return
        await stream.send_all(
            b'\x05' + bytes([self.code]) + b'\x00' + bytes([self.atyp]) + self.bound + b'\x1f\x90' + self.after
        )
        await stream.receive_some(1)


def _dial(server, host='relay.example.org', port=443, onion=None):
    outcome = {}

    async def run():
        async with trio.open_nursery() as nursery:
            listeners = await nursery.start(partial(trio.serve_tcp, server.handler, 0, host='127.0.0.1'))
            socks_port = listeners[0].socket.getsockname()[1]
            try:
                if onion is None:
                    stream = await p2p_tor.connect(socks_port, host, port)
                else:
                    stream = await p2p_tor.connect_onion(socks_port, onion)
                outcome['after'] = await _exactly(stream, len(server.after))
                await stream.aclose()
            except p2p_tor.TorConnectError as exc:
                outcome['error'] = exc
            nursery.cancel_scope.cancel()

    trio.run(run)
    return outcome


def _script(path, body):
    path.write_text('#!/bin/sh\n' + body + '\n', encoding='utf-8')
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


class FakeProcess:
    def __init__(self):
        self.returncode = None
        self.terminated = False
        self.killed = False
        self.wait_error = None

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None):
        if self.wait_error is not None:
            error, self.wait_error = self.wait_error, None
            raise error
        if self.returncode is None:
            self.returncode = -15
        return self.returncode


class FakeController:
    def __init__(self, phases, process=None, die_after=None):
        self.phases = list(phases)
        self.process = process
        self.die_after = die_after
        self.asked = 0
        self.closed = False

    def get_info(self, key, default=None):
        assert key == 'status/bootstrap-phase'
        self.asked += 1
        if self.die_after is not None and self.asked >= self.die_after:
            self.process.returncode = 1
        return self.phases.pop(0) if len(self.phases) > 1 else self.phases[0]

    def close(self):
        self.closed = True


class TestOnionIds:
    def test_a_v3_onion_id_is_valid(self):
        assert p2p_tor.valid_onion(ONION)
        assert p2p_tor.valid_onion('abcdefghijklmnopqrstuvwxyz234567' + 'a' * 24)

    @pytest.mark.parametrize('text', [
        ONION + '.onion', ONION.upper(), ONION[:-1], ONION + 'a', 'a' * 55 + '1', 'a' * 55 + '8', '',
        None, ONION.encode('ascii'), 56,
    ])
    def test_anything_else_is_not(self, text):
        assert not p2p_tor.valid_onion(text)

    def test_the_onion_service_port(self):
        assert p2p_tor.ONION_PORT == 4101


class TestTorrc:
    def test_every_port_listens_on_the_loopback_only(self):
        lines = p2p_tor.torrc_lines('/data/p2p-tor', 9050, 9051)

        assert 'DataDirectory /data/p2p-tor' in lines
        assert 'SocksPort 127.0.0.1:9050' in lines
        assert 'ControlPort 127.0.0.1:9051' in lines
        ports = [line for line in lines if line.split()[0].endswith('Port')]
        assert len(ports) == 2 and all(line.split()[1].startswith('127.0.0.1:') for line in ports)
        assert not [line for line in lines if line.split()[0] in ('ORPort', 'DirPort', 'HiddenServiceDir')]

    def test_cookie_authentication_circuit_reuse_and_the_warning_log(self):
        lines = p2p_tor.torrc_lines('/data/p2p-tor', 9050, 9051)

        assert 'CookieAuthentication 1' in lines
        assert f'MaxCircuitDirtiness {p2p_tor.CIRCUIT_DIRTINESS_SECONDS}' in lines
        assert 'Log warn stderr' in lines

    def test_the_data_directory_lives_under_the_temp_dir(self, monkeypatch, tmp_path):
        monkeypatch.setattr(config, 'TEMP_DIR', str(tmp_path))

        assert p2p_tor.data_directory() == os.path.join(str(tmp_path), 'p2p-tor')

    def test_a_free_port_is_a_loopback_port_nobody_holds(self):
        port = p2p_tor.free_port()

        with socket.socket() as sock:
            sock.bind(('127.0.0.1', port))
        assert 0 < port < 65536


class TestConnect:
    @pytest.mark.parametrize('atyp, bound', [
        (1, b'\x0a\x00\x00\x01'),
        (4, bytes(range(16))),
        (3, bytes([11]) + b'example.org'),
    ])
    def test_a_domain_connect_consumes_exactly_the_reply(self, atyp, bound):
        server = FakeSocks(atyp=atyp, bound=bound, after=b'first bytes of the stream')

        outcome = _dial(server, 'relay.example.org', 443)

        assert 'error' not in outcome
        assert outcome['after'] == b'first bytes of the stream'
        assert server.greetings == [b'\x05\x01\x00']
        assert server.requests == [(b'\x05\x01\x00\x03', 'relay.example.org', 443)]

    def test_connect_onion_dials_the_onion_service_port(self):
        server = FakeSocks()

        outcome = _dial(server, onion=ONION)

        assert 'error' not in outcome
        assert server.requests == [(b'\x05\x01\x00\x03', ONION + '.onion', p2p_tor.ONION_PORT)]

    @pytest.mark.parametrize('code', sorted(p2p_tor.SOCKS_REPLIES))
    def test_every_socks_error_is_a_tor_connect_error_with_its_reason(self, code):
        outcome = _dial(FakeSocks(code=code), 'relay.example.org', 443)

        error = outcome['error']
        assert isinstance(error, OSError)
        assert str(error) == f'Tor could not reach relay.example.org: {p2p_tor.SOCKS_REPLIES[code]}'

    def test_an_unknown_socks_error_names_its_code(self):
        outcome = _dial(FakeSocks(code=0x42), 'relay.example.org', 443)

        assert str(outcome['error']) == 'Tor could not reach relay.example.org: 66'

    def test_the_onion_service_unreachable_reason(self):
        assert p2p_tor.SOCKS_REPLIES[4] == 'onion service unreachable'

    def test_a_refused_greeting_is_a_tor_connect_error(self):
        server = FakeSocks(greeting=b'\x05\xff')

        outcome = _dial(server)

        assert str(outcome['error']) == 'Tor refused the SOCKS greeting'
        assert server.requests == []

    def test_a_connection_closed_after_the_greeting_is_a_tor_connect_error(self):
        outcome = _dial(FakeSocks(close_after_greeting=True))

        assert str(outcome['error']) == 'Tor closed the SOCKS connection'

    def test_a_reply_cut_short_is_a_tor_connect_error(self):
        outcome = _dial(FakeSocks(reply=b'\x05\x00\x00\x01\x7f\x00'))

        assert str(outcome['error']) == 'Tor closed the SOCKS connection'

    def test_a_connect_error_is_not_a_p2p_failure_of_the_node(self):
        assert issubclass(p2p_tor.TorConnectError, OSError)
        assert not issubclass(p2p_tor.TorConnectError, P2PError)
        assert issubclass(p2p_tor.TorError, P2PError)


class TestStreamIO:
    def test_reads_and_writes_go_through_the_trio_stream(self):
        async def run():
            left, right = trio.testing.memory_stream_pair()
            io = p2p_tor.StreamIO(left)
            await io.write(b'abc')
            sent = await right.receive_some(10)
            await right.send_all(b'xyz')
            first = await io.read(2)
            rest = await io.read(10)
            return sent, first, rest

        assert trio.run(run) == (b'abc', b'xy', b'z')


class TestStart:
    @POSIX_ONLY
    def test_a_tor_that_exits_at_once_is_a_tor_error(self, monkeypatch, tmp_path):
        monkeypatch.setattr(config, 'TEMP_DIR', str(tmp_path / 'temp'))
        args_file = tmp_path / 'args.txt'
        fake = _script(tmp_path / 'tor', f'printf "%s\\n" "$@" > "{args_file}"\nexit 3')

        with pytest.raises(p2p_tor.TorError) as raised:
            trio.run(p2p_tor.start, str(fake))

        torrc = tmp_path / 'temp' / 'p2p-tor' / 'torrc'
        assert str(raised.value) == 'Tor exited with code 3 while starting'
        assert args_file.read_text(encoding='utf-8').splitlines() == [
            '-f', str(torrc), '__OwningControllerProcess', str(os.getpid()),
        ]
        lines = torrc.read_text(encoding='utf-8').splitlines()
        assert lines[0] == 'DataDirectory ' + str(tmp_path / 'temp' / 'p2p-tor')
        ports = [line.split() for line in lines if line.split()[0].endswith('Port')]
        assert [name for name, _address in ports] == ['SocksPort', 'ControlPort']
        assert all(address.startswith('127.0.0.1:') for _name, address in ports)

    @POSIX_ONLY
    def test_a_tor_that_never_opens_its_control_port_is_stopped(self, monkeypatch, tmp_path):
        monkeypatch.setattr(config, 'TEMP_DIR', str(tmp_path))
        monkeypatch.setattr(p2p_tor, 'CONTROL_WAIT_SECONDS', 0.5)
        spawned = []
        original = p2p_tor.spawn

        def spawn(binary):
            spawned.append(original(binary))
            return spawned[-1]

        monkeypatch.setattr(p2p_tor, 'spawn', spawn)
        fake = _script(tmp_path / 'tor', 'exec sleep 30')

        with pytest.raises(p2p_tor.TorError) as raised:
            trio.run(p2p_tor.start, str(fake))

        assert str(raised.value) == 'Tor could not connect to the Tor network in time'
        assert spawned[0].process.poll() is not None

    def _start(self, monkeypatch, controller, process):
        monkeypatch.setattr(p2p_tor, 'spawn', lambda binary: p2p_tor.Tor(process, 9050, 9051))
        monkeypatch.setattr(p2p_tor, '_attach', lambda tor: controller)
        monkeypatch.setattr(p2p_tor, 'POLL_SECONDS', 0.001)
        progress = []

        async def run():
            return await p2p_tor.start('tor', progress.append)

        return trio.run(run), progress

    def test_the_bootstrap_percentage_is_reported_until_100(self, monkeypatch):
        process = FakeProcess()
        controller = FakeController([
            'NOTICE BOOTSTRAP PROGRESS=0 TAG=starting SUMMARY="Starting"',
            'NOTICE BOOTSTRAP PROGRESS=45 TAG=requesting_descriptors',
            '',
            'NOTICE BOOTSTRAP PROGRESS=100 TAG=done SUMMARY="Done"',
        ])

        tor, progress = self._start(monkeypatch, controller, process)

        assert progress == [0, 45, 0, 100]
        assert tor.controller is controller and (tor.socks_port, tor.control_port) == (9050, 9051)
        assert not controller.closed and not process.terminated

    def test_a_tor_that_dies_while_connecting_is_a_tor_error_and_is_closed(self, monkeypatch):
        process = FakeProcess()
        controller = FakeController(['NOTICE BOOTSTRAP PROGRESS=10 TAG=conn'], process=process, die_after=2)

        with pytest.raises(p2p_tor.TorError) as raised:
            self._start(monkeypatch, controller, process)

        assert str(raised.value) == 'Tor exited with code 1 while connecting'
        assert controller.closed


class TestPublish:
    class Controller:
        def __init__(self):
            self.calls = []

        def create_ephemeral_hidden_service(self, ports, **kwargs):
            self.calls.append((ports, kwargs))
            private = 'NEWKEYCONTENT' if kwargs['key_type'] == 'NEW' else None
            return SimpleNamespace(service_id=ONION, private_key=private)

    def _publish(self, key):
        controller = self.Controller()
        tor = SimpleNamespace(controller=controller)
        result = trio.run(p2p_tor.publish, tor, 5555, key)
        return result, controller.calls

    def test_without_a_stored_key_tor_makes_a_new_one_to_store(self):
        result, calls = self._publish(None)

        assert result == (ONION, 'NEWKEYCONTENT')
        assert calls == [({p2p_tor.ONION_PORT: 5555}, {
            'key_type': 'NEW', 'key_content': 'ED25519-V3', 'await_publication': True,
            'timeout': p2p_tor.PUBLISH_SECONDS,
        })]

    def test_the_stored_key_is_used_as_it_is(self):
        result, calls = self._publish('STOREDKEYCONTENT')

        assert result == (ONION, 'STOREDKEYCONTENT')
        assert calls == [({p2p_tor.ONION_PORT: 5555}, {
            'key_type': 'ED25519-V3', 'key_content': 'STOREDKEYCONTENT', 'await_publication': True,
            'timeout': p2p_tor.PUBLISH_SECONDS,
        })]


class TestClose:
    def test_close_stops_the_controller_and_the_process(self):
        process = FakeProcess()
        tor = p2p_tor.Tor(process, 1, 2)
        tor.controller = FakeController(['x'])

        tor.close()

        assert tor.controller.closed and process.terminated and not process.killed
        assert not tor.running()

    def test_a_process_that_does_not_stop_is_killed(self):
        process = FakeProcess()
        process.wait_error = subprocess.TimeoutExpired('tor', p2p_tor.STOP_SECONDS)
        tor = p2p_tor.Tor(process, 1, 2)

        tor.close()

        assert process.terminated and process.killed

    def test_a_process_that_already_exited_is_left_alone(self):
        process = FakeProcess()
        process.returncode = 0
        p2p_tor.Tor(process, 1, 2).close()

        assert not process.terminated and not process.killed
