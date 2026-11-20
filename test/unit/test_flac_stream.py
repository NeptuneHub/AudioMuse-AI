# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""tasks.flac_stream: which songs the player gets as FLAC, and that nothing is lost.

Main Features:
* kind and mimetype from the first bytes: FLAC, MP3, ADTS AAC and anything after
  an ID3 tag are native, mp4 waits for the whole file, the rest is converted;
  needed_bytes reads past an ID3 tag (and its footer)
* A real child converts 16 and 24-bit WAV from a pipe and ALAC from a file to FLAC
  with every sample unchanged; an AAC mp4 is sent as it is; a file PyAV cannot
  open is sent as it is
* A piece, from a file or a pipe, at a whole or a fractional second, is exactly
  those samples: FLAC from 24-bit audio, float WAV from float audio
* The child is started as python -m tasks.flac_stream, the frozen app with
  CHILD_FLAG, and the frozen launcher routes that flag to run
* A pipe that claims to seek (Windows reports pipes as seekable) still gives the
  exact piece: PyAV never sees a seek on stdin
* A child that exits a moment after its reply keeps its piece and its stream
  (Windows ends a frozen child some 10 ms after its last byte); one that never
  exits is killed after EXIT_WAIT_SECONDS
"""

import io
import os
import sys

import numpy as np
import pytest

from tasks import flac_stream


def _tag(size, footer=False):
    syncsafe = bytes([(size >> 21) & 127, (size >> 14) & 127, (size >> 7) & 127, size & 127])
    return b'ID3\x04\x00' + (b'\x10' if footer else b'\x00') + syncsafe + b'\x00' * size + (b'3DI' + b'\x00' * 7 if footer else b'')


@pytest.mark.parametrize('head, kind, mimetype', [
    (b'fLaC\x00\x00\x00\x22' + b'\x00' * 8, flac_stream.KIND_NATIVE, 'audio/flac'),
    (b'\xff\xfb\x90\x64' + b'\x00' * 8, flac_stream.KIND_NATIVE, 'audio/mpeg'),
    (b'\xff\xf1\x50\x80' + b'\x00' * 8, flac_stream.KIND_NATIVE, 'audio/aac'),
    (_tag(20) + b'fLaC' + b'\x00' * 8, flac_stream.KIND_NATIVE, 'audio/flac'),
    (_tag(20) + b'\xff\xfb\x90\x64' + b'\x00' * 8, flac_stream.KIND_NATIVE, 'audio/mpeg'),
    (b'ID3audio', flac_stream.KIND_NATIVE, 'audio/mpeg'),
    (b'\x00\x00\x00\x20ftypM4A ', flac_stream.KIND_MP4, 'audio/mp4'),
    (b'RIFF\x00\x00\x00\x00WAVE', flac_stream.KIND_CONVERT, 'application/octet-stream'),
    (b'FORM\x00\x00\x00\x00AIFF', flac_stream.KIND_CONVERT, 'application/octet-stream'),
    (b'OggS\x00\x02' + b'\x00' * 6, flac_stream.KIND_CONVERT, 'application/octet-stream'),
    (b'wvpk' + b'\x00' * 8, flac_stream.KIND_CONVERT, 'application/octet-stream'),
])
def test_the_first_bytes_say_what_is_sent_as_it_is(head, kind, mimetype):
    assert flac_stream.kind(head) == kind
    assert flac_stream.mimetype(head) == mimetype


def test_needed_bytes_reads_past_an_id3_tag_and_its_footer():
    assert flac_stream.needed_bytes(b'fLaC') == flac_stream.SNIFF_BYTES
    assert flac_stream.needed_bytes(_tag(300)) == 10 + 300 + flac_stream.SNIFF_BYTES
    assert flac_stream.needed_bytes(_tag(300, footer=True)) == 10 + 300 + 10 + flac_stream.SNIFF_BYTES


def _write(path, samples, rate, subtype, fmt='WAV'):
    import soundfile

    soundfile.write(path, samples, rate, format=fmt, subtype=subtype)
    return path


def _ints(path_or_bytes, bits):
    import av

    source = io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, bytes) else path_or_bytes
    with av.open(source) as container:
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format='s16' if bits == 16 else 's32', layout=stream.layout.name, rate=stream.rate)
        parts = [f.to_ndarray().reshape(-1) for frame in container.decode(stream) for f in resampler.resample(frame)]
    values = np.concatenate(parts).astype(np.int64)
    return values >> 8 if bits == 24 else values


def _read_all(chunks):
    return b''.join(chunks)


def _file_chunks(path):
    with open(path, 'rb') as handle:
        while chunk := handle.read(4096):
            yield chunk


@pytest.mark.parametrize('subtype, bits, rate', [('PCM_16', 16, 44100), ('PCM_24', 24, 96000)])
def test_a_wav_from_a_pipe_becomes_flac_with_every_sample_unchanged(tmp_path, subtype, bits, rate):
    samples = np.random.default_rng(5).uniform(-0.8, 0.8, size=(rate // 2, 2))
    source = _write(str(tmp_path / 'a.wav'), samples, rate, subtype)
    chunks = flac_stream.flac_chunks(_file_chunks(source), str(tmp_path / 'flac.log'), 'a')
    assert chunks is not None
    data = _read_all(chunks)
    assert data[:4] == b'fLaC'
    assert np.array_equal(_ints(data, bits), _ints(source, bits))


def test_an_alac_file_becomes_flac_and_an_aac_file_is_sent_as_it_is(tmp_path):
    import av

    rate = 44100
    samples = (np.random.default_rng(7).uniform(-0.5, 0.5, size=(2, rate)) * 32767).astype(np.int16)
    paths = {}
    for codec, fmt in (('alac', 's16p'), ('aac', 'fltp')):
        paths[codec] = str(tmp_path / f'{codec}.m4a')
        with av.open(paths[codec], 'w', format='ipod') as out:
            stream = out.add_stream(codec, rate=rate)
            stream.layout = 'stereo'
            frame = av.AudioFrame.from_ndarray(samples if fmt == 's16p' else samples.astype(np.float32) / 32768, format=fmt, layout='stereo')
            frame.rate = rate
            for packet in stream.encode(frame):
                out.mux(packet)
            for packet in stream.encode(None):
                out.mux(packet)
    data = _read_all(flac_stream.flac_chunks(paths['alac'], str(tmp_path / 'alac.log'), 'alac'))
    assert data[:4] == b'fLaC'
    assert np.array_equal(_ints(data, 16), _ints(paths['alac'], 16))
    assert flac_stream.flac_chunks(paths['aac'], str(tmp_path / 'aac.log'), 'aac') is None


def test_a_file_pyav_cannot_open_is_sent_as_it_is_and_the_reason_is_logged(tmp_path, caplog):
    junk = tmp_path / 'junk.bin'
    junk.write_bytes(b'RIFF' + b'\x07' * 600)
    assert flac_stream.flac_chunks(str(junk), str(tmp_path / 'junk.log'), 'junk') is None
    assert 'did not start' in caplog.text and 'PyAV could not open' in caplog.text


@pytest.mark.parametrize('subtype, bits, start', [('PCM_24', 24, 0.0), ('PCM_24', 24, 2.3), ('FLOAT', 0, 1.7)])
def test_a_piece_is_the_songs_own_samples_from_a_file_or_a_pipe(tmp_path, subtype, bits, start):
    import soundfile

    rate = 48000
    samples = np.random.default_rng(9).uniform(-0.9, 0.9, size=(rate * 4, 2))
    source = _write(str(tmp_path / 'a.wav'), samples, rate, subtype)
    whole, _ = soundfile.read(source, dtype='float32' if subtype == 'FLOAT' else 'int32', always_2d=True)
    first, last = int(round(start * rate)), int(round((start + 1.5) * rate))
    for given in (source, _file_chunks(source)):
        header, body = flac_stream.cut_piece(given, start, 1.5, str(tmp_path / 'p.log'), 'a')
        piece, piece_rate = soundfile.read(io.BytesIO(body), dtype='float32' if subtype == 'FLOAT' else 'int32', always_2d=True)
        assert header['format'] == ('wav' if subtype == 'FLOAT' else 'flac') and piece_rate == rate
        assert np.array_equal(piece, whole[first:last])
        assert header['seconds'] is None or header['seconds'] == pytest.approx(4.0)


def test_the_child_command_is_the_module_or_the_frozen_app_with_its_flag(monkeypatch):
    command, cwd = flac_stream._command('/m/a.wav')
    assert command == [sys.executable, '-m', 'tasks.flac_stream', '/m/a.wav'] and os.path.isdir(cwd)
    monkeypatch.setattr(sys, 'frozen', True, raising=False)
    assert flac_stream._command('-') == ([sys.executable, flac_stream.CHILD_FLAG, '-'], None)


class _WindowsPipe:
    def __init__(self, data):
        self._data = io.BytesIO(data)

    def read(self, size=-1):
        return self._data.read(size)

    def seekable(self):
        return True

    def tell(self):
        return 0

    def seek(self, offset, whence=0):
        return 0


def _padded(flac, size):
    position = 4
    while not flac[position] & 0x80:
        position += 4 + int.from_bytes(flac[position + 1:position + 4], 'big')
    end = position + 4 + int.from_bytes(flac[position + 1:position + 4], 'big')
    data = bytearray(flac)
    data[position] &= 0x7F
    return bytes(data[:end]) + bytes([0x81]) + size.to_bytes(3, 'big') + bytes(size) + bytes(data[end:])


def test_a_piece_from_a_pipe_that_claims_to_seek_is_still_the_exact_samples(tmp_path, monkeypatch):
    import soundfile

    rate = 44100
    samples = np.random.default_rng(11).uniform(-0.9, 0.9, size=(rate * 4, 2))
    source = _write(str(tmp_path / 'a.flac'), samples, rate, 'PCM_24', fmt='FLAC')
    whole, _ = soundfile.read(source, dtype='int32', always_2d=True)
    with open(source, 'rb') as handle:
        song = _padded(handle.read(), 300000)
    assert np.array_equal(soundfile.read(io.BytesIO(song), dtype='int32', always_2d=True)[0], whole)
    monkeypatch.setattr(sys, 'stdin', type('Stdin', (), {'buffer': _WindowsPipe(song)})())
    reply = io.BytesIO()
    assert flac_stream.piece(flac_stream.FROM_STDIN, 1.0, 1.5, reply) == 0
    header, body = reply.getvalue().split(b'\n', 1)
    piece, _ = soundfile.read(io.BytesIO(body), dtype='int32', always_2d=True)
    assert b'"piece"' in header and np.array_equal(piece, whole[rate:rate + int(1.5 * rate)])


_LATE_EXIT = """import os, sys, time
out = os.fdopen(os.dup(1), 'wb')
os.dup2(2, 1)
out.write(sys.argv[1].encode() + b'\\n' + b'BODY' * 2000)
out.close()
time.sleep(float(sys.argv[2]))
"""


def _late_child(monkeypatch, status, seconds):
    header = '{"status": "%s", "format": "flac", "seconds": 1.0}' % status
    monkeypatch.setattr(flac_stream, '_command', lambda source, *extra: ([sys.executable, '-c', _LATE_EXIT, header, str(seconds)], None))


def test_a_child_that_exits_a_moment_after_its_reply_keeps_its_piece_and_its_stream(tmp_path, monkeypatch, caplog):
    _late_child(monkeypatch, flac_stream.STATUS_PIECE, 0.5)
    header, body = flac_stream.cut_piece(str(tmp_path / 'a.wav'), 0.0, 1.0, str(tmp_path / 'p.log'), 'a')
    assert header['status'] == flac_stream.STATUS_PIECE and body == b'BODY' * 2000
    _late_child(monkeypatch, flac_stream.STATUS_FLAC, 0.5)
    assert _read_all(flac_stream.flac_chunks(str(tmp_path / 'a.wav'), str(tmp_path / 'f.log'), 'a')) == b'BODY' * 2000
    assert 'Could not cut' not in caplog.text and 'failed' not in caplog.text


def test_a_child_that_never_exits_after_its_reply_is_killed_after_the_wait(tmp_path, monkeypatch):
    import time

    monkeypatch.setattr(flac_stream, 'EXIT_WAIT_SECONDS', 0.3)
    _late_child(monkeypatch, flac_stream.STATUS_PIECE, 60)
    began = time.monotonic()
    assert flac_stream.cut_piece(str(tmp_path / 'a.wav'), 0.0, 1.0, str(tmp_path / 'p.log'), 'a') is None
    assert time.monotonic() - began < 20
