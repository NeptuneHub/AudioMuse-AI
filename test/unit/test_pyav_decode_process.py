# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Unit tests for the separate-process PyAV decode in tasks.analysis.song.

The PyAV fallback decodes in a child process (tasks.pyav_decode) so a
native crash on a damaged file cannot end the worker. These tests start real
child processes, including ones that really crash or get killed, and check the
verdict each outcome gets.

Main Features:
* A clean file decodes in the child to exactly the in-process PyAV result
* Bytes that are not audio come back empty, so robust_load reports the file as
  unreadable and the album marks it as not analyzable; the child's own log is
  relayed line by line into the parent's log, keeping its level
* A child that crashes or hangs twice marks the file, and every file of a fully
  broken album is marked, however many fail in a row, unless a clean test clip
  fails too (a broken decoder marks nothing); a crash that does not repeat
  returns the second attempt's audio
* A killed child, an error exit, a garbled or unknown reply and a temporary
  error raise AudioDecodeInterrupted, so nothing is marked
* Exit codes are read per platform: negative signals on POSIX, NTSTATUS codes
  on Windows, where a console close and an illegal instruction are not crashes
  the file causes
* The child module imports neither config nor database, so a decode never opens
  a database connection
* The album stage lets AudioDecodeInterrupted through instead of turning it into
  a not-analyzable mark
"""

import logging
import signal
import subprocess
import sys
from unittest.mock import Mock

import av
import numpy as np
import pytest

from tasks import pyav_decode
from tasks.analysis import album as album_mod
from tasks.analysis import song
from tasks.analysis.helper import TrackPlan

CRASH = 'import faulthandler; faulthandler._sigsegv()'


def _write_mp3(path, seconds=3.0, rate=44100):
    with av.open(str(path), 'w') as container:
        stream = container.add_stream('mp3', rate=rate)
        stream.layout = 'stereo'
        t = np.arange(int(rate * seconds), dtype=np.float32) / rate
        data = np.vstack([np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 660 * t)])
        data = (data * 0.5 * 32767).astype(np.int16)
        frame = av.AudioFrame.from_ndarray(data.T.reshape(1, -1), format='s16', layout='stereo')
        frame.sample_rate = rate
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)


@pytest.fixture(autouse=True)
def unproven_decoder(monkeypatch):
    monkeypatch.setattr(song, '_decoder_proven', False)


def _child_runs(monkeypatch, code, clean_clip_too=False):
    real = song._decode_command

    def command(file_path, target_sr):
        if file_path.endswith('.wav') and not clean_clip_too:
            return real(file_path, target_sr)
        return [sys.executable, '-c', code], None

    monkeypatch.setattr(song, '_decode_command', command)


def _librosa_fails(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError('librosa cannot open this container')

    monkeypatch.setattr(song.librosa, 'load', fail)


@pytest.mark.parametrize('target_sr', [None, 16000])
def test_a_clean_file_decodes_in_the_child_exactly_like_in_process(tmp_path, target_sr):
    path = tmp_path / 'clean.mp3'
    _write_mp3(path)
    expected, expected_sr = song._decode_audio_with_pyav(str(path), target_sr)

    audio, sr = song._decode_audio_isolated(str(path), target_sr)

    assert sr == expected_sr
    assert audio.dtype == np.float32
    assert audio.flags.writeable
    np.testing.assert_array_equal(audio, expected)


def test_bytes_that_are_not_audio_are_reported_unreadable(tmp_path, monkeypatch):
    path = tmp_path / 'garbage.m4a'
    path.write_bytes(b'this is not audio at all' * 200)

    audio, _sr = song._decode_audio_isolated(str(path), None)
    assert audio.size == 0

    _librosa_fails(monkeypatch)
    assert song.robust_load_audio_with_fallback(str(path), target_sr=None) == (None, None)


def test_the_child_log_reaches_the_parent_log(tmp_path, caplog):
    path = tmp_path / 'garbage.m4a'
    path.write_bytes(b'this is not audio at all' * 200)

    with caplog.at_level(logging.WARNING, logger='tasks.analysis.song'):
        song._decode_audio_isolated(str(path), None)

    relayed = [r for r in caplog.records if r.getMessage().startswith('PyAV decode process for')]
    assert any(
        r.levelno == logging.ERROR and 'PyAV could not read garbage.m4a' in r.getMessage()
        for r in relayed
    )
    assert any('Traceback' in r.getMessage() for r in relayed)


def test_a_child_that_crashes_twice_marks_the_file(tmp_path, monkeypatch, caplog):
    _child_runs(monkeypatch, CRASH)

    with caplog.at_level(logging.ERROR, logger='tasks.analysis.song'):
        audio, sr = song._decode_audio_isolated(str(tmp_path / 'broken.m4a'), None)

    assert audio.size == 0
    assert sr is None
    assert 'attempt 1 of 2' in caplog.text
    assert 'attempt 2 of 2' in caplog.text
    assert list(tmp_path.iterdir()) == []


def test_every_file_of_a_fully_broken_album_is_marked(tmp_path, monkeypatch):
    _child_runs(monkeypatch, CRASH)

    verdicts = [
        song._decode_audio_isolated(str(tmp_path / f'broken-rip-{n}.m4a'), None)[0].size
        for n in range(6)
    ]

    assert verdicts == [0] * 6


def test_a_decoder_that_also_crashes_on_a_clean_clip_marks_nothing(tmp_path, monkeypatch):
    _child_runs(monkeypatch, CRASH, clean_clip_too=True)

    with pytest.raises(song.AudioDecodeInterrupted, match='clean test clip'):
        song._decode_audio_isolated(str(tmp_path / 'any.m4a'), None)


def test_a_decode_that_hangs_twice_marks_the_file(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(song, 'wall_seconds', lambda load_timeout: 1)
    monkeypatch.setattr(song, '_decoder_proven', True)
    _child_runs(monkeypatch, 'import time; time.sleep(30)')

    with caplog.at_level(logging.ERROR, logger='tasks.analysis.song'):
        audio, _sr = song._decode_audio_isolated(str(tmp_path / 'hang.m4a'), None)

    assert audio.size == 0
    assert caplog.text.count('ran past 1s') == 2


def test_an_unknown_reply_status_marks_nothing(monkeypatch):
    _child_runs(
        monkeypatch,
        "import sys; sys.stdout.buffer.write(b'{\"status\": \"later\", \"sr\": 1, \"samples\": 0}\\n')",
    )

    with pytest.raises(song.AudioDecodeInterrupted, match="status 'later'"):
        song._decode_audio_isolated('/music/x.m4a', None)


def test_the_wall_clock_limit_never_drops_below_its_floor():
    assert pyav_decode.wall_seconds(30) == 300
    assert pyav_decode.wall_seconds(900) == 900
    assert pyav_decode.wall_seconds(0) is None


def test_a_crash_that_does_not_repeat_returns_the_second_attempt(tmp_path, monkeypatch):
    marker = tmp_path / 'crashed-once'
    code = (
        "import faulthandler, os, sys\n"
        "import numpy\n"
        f"marker = {str(marker)!r}\n"
        "if not os.path.exists(marker):\n"
        "    open(marker, 'w').close()\n"
        "    faulthandler._sigsegv()\n"
        "head = b'{\"status\": \"ok\", \"sr\": 16000, \"samples\": 2}' + bytes([10])\n"
        "sys.stdout.buffer.write(head + numpy.ones(2, numpy.float32).tobytes())\n"
    )
    _child_runs(monkeypatch, code)

    audio, sr = song._decode_audio_isolated('/music/flaky.m4a', None)

    assert sr == 16000
    np.testing.assert_array_equal(audio, np.ones(2, dtype=np.float32))


@pytest.mark.skipif(not hasattr(signal, 'SIGKILL'), reason='POSIX kill signal')
def test_a_killed_child_is_not_marked(monkeypatch):
    _child_runs(monkeypatch, 'import os, signal; os.kill(os.getpid(), signal.SIGKILL)')
    _librosa_fails(monkeypatch)

    with pytest.raises(song.AudioDecodeInterrupted, match='SIGKILL'):
        song.robust_load_audio_with_fallback('/music/x.m4a', target_sr=None)


def test_a_child_that_exits_with_an_error_is_not_marked(monkeypatch):
    _child_runs(monkeypatch, 'raise SystemExit(1)')

    with pytest.raises(song.AudioDecodeInterrupted, match='exit code 1'):
        song._decode_audio_isolated('/music/x.m4a', None)


def test_a_garbled_reply_is_not_marked(monkeypatch):
    _child_runs(monkeypatch, 'print("hello")')

    with pytest.raises(song.AudioDecodeInterrupted, match='unreadable reply'):
        song._decode_audio_isolated('/music/x.m4a', None)


def test_a_short_reply_is_not_marked(monkeypatch):
    code = (
        "import sys\n"
        "sys.stdout.buffer.write(b'{\"status\": \"ok\", \"sr\": 16000, \"samples\": 9}'"
        " + bytes([10]) + bytes(8))\n"
    )
    _child_runs(monkeypatch, code)

    with pytest.raises(song.AudioDecodeInterrupted, match='8 bytes for 9 samples'):
        song._decode_audio_isolated('/music/x.m4a', None)


def test_a_temporary_error_in_the_child_is_not_marked(tmp_path):
    missing = tmp_path / 'gone.m4a'

    with pytest.raises(song.AudioDecodeInterrupted, match='temporary error'):
        song._decode_audio_isolated(str(missing), None)


@pytest.mark.skipif(sys.platform == 'win32', reason='POSIX signal numbers')
@pytest.mark.parametrize('returncode,crashed', [
    (-int(signal.SIGSEGV), True),
    (-int(signal.SIGABRT), True),
    (-int(signal.SIGFPE), True),
    (-int(getattr(signal, 'SIGTRAP', 5)), True),
    (-9, False),
    (-4, False),
    (-15, False),
    (1, False),
])
def test_posix_exit_codes_are_read_as_crashes_only_for_native_faults(returncode, crashed):
    assert song._decode_child_crashed(returncode) is crashed


@pytest.mark.parametrize('returncode,crashed', [
    (0xC0000005, True),
    (0xC0000005 - (1 << 32), True),
    (0xC0000409, True),
    (0xC0000374, True),
    (0xC0000094, True),
    (0xC00000FD, True),
    (0x80000003, True),
    (3, True),
    (0xC0000006, False),
    (0xC000013A, False),
    (0xC000001D, False),
    (1, False),
])
def test_windows_exit_codes_are_read_as_crashes_only_for_native_faults(
    monkeypatch, returncode, crashed
):
    monkeypatch.setattr(song.sys, 'platform', 'win32')
    assert song._decode_child_crashed(returncode) is crashed


def test_the_command_starts_the_frozen_app_itself_with_the_decode_flag(monkeypatch):
    monkeypatch.setattr(song.sys, 'frozen', True, raising=False)
    monkeypatch.setattr(song, 'AUDIO_LOAD_TIMEOUT', 600)
    monkeypatch.setattr(song, 'AUDIO_MIN_DECODED_FRACTION', 0.5)

    command, cwd = song._decode_command('song.m4a', 16000)

    assert command[:2] == [sys.executable, pyav_decode.CHILD_FLAG]
    assert command[3:] == ['16000', '600', '0.5']
    assert cwd is None


def test_the_command_runs_the_child_module_from_the_project_root(monkeypatch):
    monkeypatch.delattr(song.sys, 'frozen', raising=False)

    command, cwd = song._decode_command('song.m4a', None)

    assert command[:3] == [sys.executable, '-m', 'tasks.pyav_decode']
    assert command[4] == '0'
    assert cwd == song._PROJECT_ROOT


def test_the_child_module_never_imports_config_or_the_database():
    code = (
        "import sys, tasks.pyav_decode; "
        "print('config' in sys.modules, 'database' in sys.modules)"
    )
    child = subprocess.run(
        [sys.executable, '-c', code], cwd=song._PROJECT_ROOT,
        capture_output=True, text=True, check=True,
    )

    assert child.stdout.split() == ['False', 'False']


def test_the_album_stage_lets_an_interrupted_decode_through(monkeypatch):
    def interrupted(path):
        raise song.AudioDecodeInterrupted('the decode process was killed')

    monkeypatch.setattr(album_mod, 'download_track', lambda temp_dir, item: '/nonexistent/x.m4a')
    monkeypatch.setattr(album_mod, 'decode_audio_once', interrupted)
    monkeypatch.setattr(album_mod, '_stage_collect_chromaprint', lambda *a, **k: None)
    monkeypatch.setattr(album_mod._ah, 'catalog_item_id', lambda item: 'track-1')

    with pytest.raises(song.AudioDecodeInterrupted):
        album_mod._analyze_single_track(
            {'Id': 'x', 'Name': 'x'}, TrackPlan(musicnn=True, clap=False, lyrics=False, base=False),
            'Artist - Track', 'album-1', 'Album', 'parent-1', 3, {}, Mock(), None, None, None,
            {}, {},
        )
