# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""FLAC for the page's player: a song in any format, whole or in exact pieces.

A piece (piece(): start and seconds) is the song's own samples from that second
on, cut on the sample, at its own rate and channels: FLAC (16 or 24-bit) from
integer audio, 32-bit float WAV from float audio (MP3, AAC, Ogg, float WAV), so
nothing is rounded or clipped; with the song's length. A file is sought to the
piece (an MP3 PREROLL_SECONDS before, its first frame after a seek is not
whole); a song still downloading is read from its start. The page plays songs
in pieces, so the browser holds only a few seconds of each decoded song.

The whole song: the play route sends FLAC, MP3 and AAC as they are (every browser decodes them;
as FLAC an MP3 is 5 times larger and loses the peaks over full scale its decoder
gives, since FLAC holds integers). Every other format (WAV, AIFF, ALAC, WavPack,
APE, TTA, WMA, Ogg, DSD, ...) is decoded with PyAV and encoded as FLAC at
FLAC_LEVEL while it is sent: the child reads the song as the media server
delivers it (stdin) and writes each encoded block to stdout, which Flask relays
in CHUNK_BYTES chunks, so download, conversion and transfer overlap. A 15
minute 192 kHz 24-bit WAV (989 MB) becomes 405 MB; its first chunk leaves after
about a second.

Main Features:
* Lossless: 8 and 16-bit sources stay 16-bit, everything else is 24-bit, at the
  song's own rate and channels (more than 8 channels, FLAC's limit, become stereo)
* kind() reads the first bytes: native (FLAC, MP3, ADTS AAC, anything after an
  ID3 tag: sent as they are), mp4 (its index may sit at the end, so the child
  gets the whole file and sends AAC as it is, ALAC as FLAC) or convert
* The conversion runs in a child process (python -m tasks.flac_stream, or the
  frozen app with CHILD_FLAG), so a native crash in FFmpeg on a damaged file
  ends only the child, never Flask. Its log goes to a file, never a pipe
  (no stderr deadlock), and is relayed when the conversion fails
* The child answers one JSON line first: flac (FLAC bytes follow), piece (with
  its format and the song's seconds), original (send the file as it is) or
  unreadable; a packet that fails to decode is
  skipped. Flask sends the original whenever the child cannot convert, so the
  page decides (it skips a song it cannot decode)
* The child ends itself past WALL_SECONDS; Flask kills it when the download is
  abandoned. It imports only PyAV and the logging setup, to start fast
"""

import ctypes
import json
import logging
import os
import subprocess  # nosec B404
import sys
import threading

from app_logging import configure_logging

logger = logging.getLogger(__name__)

CHILD_FLAG = '--transcode-flac'
FROM_STDIN = '-'
FLAC_LEVEL = 1
CHUNK_BYTES = 256 * 1024
WALL_SECONDS = 1800
LOG_TAIL_CHARS = 4000
FLAC_MAX_CHANNELS = 8
KIND_NATIVE = 'native'
KIND_MP4 = 'mp4'
KIND_CONVERT = 'convert'
SNIFF_BYTES = 12
STATUS_FLAC = 'flac'
STATUS_PIECE = 'piece'
STATUS_ORIGINAL = 'original'
STATUS_UNREADABLE = 'unreadable'
PREROLL_SECONDS = 1.0
PIECE_BLOCK = 4096
FLOAT_FORMATS = frozenset({'flt', 'fltp', 'dbl', 'dblp'})
NATIVE = frozenset({
    ('flac', 'flac'),
    ('mp3', 'mp3'),
    ('mp3', 'mp3float'),
    ('aac', 'aac'),
    ('mov,mp4,m4a,3gp,3g2,mj2', 'aac'),
})
SHORT_FORMATS = frozenset({'u8', 'u8p', 's16', 's16p'})
_SEM_FAILCRITICALERRORS = 0x0001
_SEM_NOGPFAULTERRORBOX = 0x0002
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def needed_bytes(head):
    if head[:3] == b'ID3' and len(head) >= 10:
        size = (head[6] & 127) << 21 | (head[7] & 127) << 14 | (head[8] & 127) << 7 | (head[9] & 127)
        return 10 + size + (10 if head[5] & 0x10 else 0) + SNIFF_BYTES
    return SNIFF_BYTES


def _body(head):
    if head[:3] != b'ID3':
        return head
    return head[needed_bytes(head) - SNIFF_BYTES:] if len(head) >= 10 else b''


def _synced(body):
    return len(body) >= 2 and body[0] == 0xFF and (body[1] & 0xE0) == 0xE0


def kind(head):
    body = _body(head)
    if body[4:8] == b'ftyp':
        return KIND_MP4
    if body[:4] == b'fLaC' or _synced(body) or head[:3] == b'ID3':
        return KIND_NATIVE
    return KIND_CONVERT


def mimetype(head):
    body = _body(head)
    if body[:4] == b'fLaC':
        return 'audio/flac'
    if body[4:8] == b'ftyp':
        return 'audio/mp4'
    if _synced(body) and not body[1] & 0x06:
        return 'audio/aac'
    if _synced(body) or head[:3] == b'ID3':
        return 'audio/mpeg'
    return 'application/octet-stream'


def _target(container, stream, from_stdin):
    if not from_stdin and (container.format.name, stream.codec_context.name) in NATIVE:
        return None
    sample_format = 's16' if stream.format.name in SHORT_FORMATS else 's32'
    layout = stream.layout.name if stream.channels <= FLAC_MAX_CHANNELS else 'stereo'
    return sample_format, layout


def _frames(container, stream):
    import av

    for packet in container.demux(stream):
        try:
            yield from packet.decode()
        except av.error.FFmpegError:
            logger.warning("Skipped a packet that does not decode in %s", container.name)


def _status(reply, status):
    reply.write(json.dumps({'status': status}).encode('ascii') + b'\n')


def transcode(source_path, reply):
    import av

    from_stdin = source_path == FROM_STDIN
    try:
        source = av.open(sys.stdin.buffer if from_stdin else source_path)
        stream = source.streams.audio[0]
        target = _target(source, stream, from_stdin)
    except Exception:
        logger.exception("PyAV could not open %s", 'the song' if from_stdin else os.path.basename(source_path))
        _status(reply, STATUS_UNREADABLE)
        return 1
    with source:
        if target is None:
            _status(reply, STATUS_ORIGINAL)
            return 0
        sample_format, layout = target
        _status(reply, STATUS_FLAC)
        reply.flush()
        resampler = av.AudioResampler(format=sample_format, layout=layout, rate=stream.rate)
        with av.open(reply, 'w', format='flac') as out:
            flac = out.add_stream('flac', rate=stream.rate, options={'compression_level': str(FLAC_LEVEL)})
            flac.layout = layout
            flac.format = sample_format
            for frame in _frames(source, stream):
                for block in resampler.resample(frame):
                    out.mux(flac.encode(block))
            for block in resampler.resample(None):
                out.mux(flac.encode(block))
            out.mux(flac.encode(None))
    return 0


def _song_seconds(container, stream):
    if stream.duration and stream.time_base:
        return float(stream.duration * stream.time_base)
    return container.duration / 1e6 if container.duration else None


def _cut(container, stream, layout, sample_format, first, last):
    import av
    import numpy as np

    channels = len(av.AudioLayout(layout).channels)
    resampler = av.AudioResampler(format=sample_format, layout=layout, rate=stream.rate)
    parts, position = [], None
    zero = stream.start_time or 0
    for frame in _frames(container, stream):
        if frame.pts is not None and stream.time_base is not None:
            position = int(round(float((frame.pts - zero) * stream.time_base) * stream.rate))
        elif position is None:
            position = 0
        for block in resampler.resample(frame):
            data = block.to_ndarray().reshape(-1, channels)
            low, high = max(first - position, 0), min(last - position, data.shape[0])
            if high > low:
                parts.append(data[low:high])
            position += data.shape[0]
        if position >= last:
            break
    dtype = {'s16': np.int16, 's32': np.int32, 'flt': np.float32}[sample_format]
    return np.concatenate(parts) if parts else np.zeros((0, channels), dtype=dtype)


def _wav_float(audio, rate):
    import struct

    channels = audio.shape[1]
    data = audio.astype('<f4').tobytes()
    header = struct.pack('<4sI4s4sIHHIIHH4sI', b'RIFF', 36 + len(data), b'WAVE', b'fmt ', 16, 3, channels, rate,
                         rate * channels * 4, channels * 4, 32, b'data', len(data))
    return header + data


def _flac(audio, rate, layout, sample_format):
    import io

    import av

    out = io.BytesIO()
    with av.open(out, 'w', format='flac') as container:
        flac = container.add_stream('flac', rate=rate, options={'compression_level': str(FLAC_LEVEL)})
        flac.layout = layout
        flac.format = sample_format
        for start in range(0, audio.shape[0], PIECE_BLOCK):
            frame = av.AudioFrame.from_ndarray(audio[start:start + PIECE_BLOCK].reshape(1, -1), format=sample_format, layout=layout)
            frame.rate = rate
            container.mux(flac.encode(frame))
        container.mux(flac.encode(None))
    return out.getvalue()


def piece(source_path, start, seconds, reply):
    import av

    from_stdin = source_path == FROM_STDIN
    try:
        source = av.open(sys.stdin.buffer if from_stdin else source_path)
        stream = source.streams.audio[0]
    except Exception:
        logger.exception("PyAV could not open %s", 'the song' if from_stdin else os.path.basename(source_path))
        _status(reply, STATUS_UNREADABLE)
        return 1
    with source:
        rate = stream.rate
        layout = stream.layout.name if stream.channels <= FLAC_MAX_CHANNELS else 'stereo'
        floating = stream.format.name in FLOAT_FORMATS
        sample_format = 'flt' if floating else 's16' if stream.format.name in SHORT_FORMATS else 's32'
        total = _song_seconds(source, stream)
        if not from_stdin and start > PREROLL_SECONDS:
            source.seek(int((start - PREROLL_SECONDS) / stream.time_base), stream=stream)
        audio = _cut(source, stream, layout, sample_format, int(round(start * rate)), int(round((start + seconds) * rate)))
    body = _wav_float(audio, rate) if floating else _flac(audio, rate, layout, sample_format)
    reply.write(json.dumps({'status': STATUS_PIECE, 'format': 'wav' if floating else 'flac', 'seconds': total}).encode('ascii') + b'\n')
    reply.write(body)
    return 0


def _disable_crash_dialog():
    if sys.platform == 'win32':
        ctypes.windll.kernel32.SetErrorMode(_SEM_FAILCRITICALERRORS | _SEM_NOGPFAULTERRORBOX)


def run(argv):
    reply = os.fdopen(os.dup(1), 'wb')
    os.dup2(2, 1)
    _disable_crash_dialog()
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='backslashreplace')
    configure_logging()
    watchdog = threading.Timer(WALL_SECONDS, os._exit, args=(1,))
    watchdog.daemon = True
    watchdog.start()
    try:
        with reply:
            code = piece(argv[0], float(argv[1]), float(argv[2]), reply) if len(argv) == 3 else transcode(argv[0], reply)
    except Exception:
        logger.exception("The FLAC conversion stopped")
        code = 1
    sys.stderr.flush()
    os._exit(code)


def _command(source, *extra):
    if getattr(sys, 'frozen', False):
        return [sys.executable, CHILD_FLAG, source, *extra], None
    return [sys.executable, '-m', __name__, source, *extra], _PROJECT_ROOT


def _log_tail(log_path):
    try:
        with open(log_path, 'rb') as handle:
            return handle.read().decode('utf-8', 'replace').strip()[-LOG_TAIL_CHARS:]
    except OSError:
        return ''


def _feed(child, chunks):
    try:
        for chunk in chunks:
            child.stdin.write(chunk)
    except (BrokenPipeError, OSError, ValueError):
        pass
    finally:
        chunks.close()
        try:
            child.stdin.close()
        except (BrokenPipeError, OSError):
            pass


def _relay(child, log_path, name):
    finished = False
    try:
        while chunk := child.stdout.read1(CHUNK_BYTES):
            yield chunk
        finished = True
    finally:
        if child.poll() is None:
            child.kill()
        code = child.wait()
        child.stdout.close()
        if finished and code != 0:
            logger.error("The FLAC conversion of %s failed (exit code %s); the page got a cut song. Child log: %s",
                         name, code, _log_tail(log_path))


def _spawn(source, log_path, *extra):
    from_stdin = not isinstance(source, str)
    command, cwd = _command(FROM_STDIN if from_stdin else os.path.abspath(source), *extra)
    with open(log_path, 'wb') as log:
        child = subprocess.Popen(  # nosec B603
            command, cwd=cwd, stdin=subprocess.PIPE if from_stdin else subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=log, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
        )
    if from_stdin:
        threading.Thread(target=_feed, args=(child, source), name='flac-feed', daemon=True).start()
    try:
        header = json.loads(child.stdout.readline() or b'{}')
    except ValueError:
        header = {}
    return child, header


def cut_piece(source, start, seconds, log_path, name):
    child, header = _spawn(source, log_path, repr(float(start)), repr(float(seconds)))
    try:
        body = child.stdout.read() if header.get('status') == STATUS_PIECE else b''
    finally:
        if child.poll() is None:
            child.kill()
        code = child.wait()
        child.stdout.close()
    if header.get('status') != STATUS_PIECE or code != 0:
        logger.error("Could not cut %s from %.1f s for %.1f s (exit code %s). Child log: %s",
                     name, start, seconds, code, _log_tail(log_path))
        return None
    return header, body


def flac_chunks(source, log_path, name):
    child, header = _spawn(source, log_path)
    status = header.get('status')
    if status == STATUS_FLAC:
        logger.info("Sending %s to the page as FLAC (level %s), converted while it downloads", name, FLAC_LEVEL)
        return _relay(child, log_path, name)
    if child.poll() is None:
        child.kill()
    code = child.wait()
    child.stdout.close()
    if status != STATUS_ORIGINAL:
        logger.error("The FLAC conversion of %s did not start (exit code %s); sending the file as it is. Child log: %s",
                     name, code, _log_tail(log_path))
    return None


if __name__ == '__main__':
    sys.exit(run(sys.argv[1:]))
