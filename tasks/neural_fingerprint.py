# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Neural audio fingerprint: one 32-byte code per half second of a track.

The encoder is the neural music fingerprinter of Araz, Serra and Bogdanov
("Enhancing Neural Audio Fingerprint Robustness to Audio Degradation for
Music Identification", ISMIR 2025; AGPLv3 code and weights, the triplet
checkpoint), the NAFP architecture of Chang et al. (ICASSP 2021) trained with
real room impulse responses, microphone responses and background noise so
that a second of a recording heard through a phone lands where the same
second of the clean track lands. It was exported once from the TensorFlow
checkpoint to neural_fingerprint.onnx at the repository root (17.2 million
parameters, 71 MB; scripts/onnx_export/export_neural_fingerprint_to_onnx.py).
The analysis stores the whole sequence for every track and the Search by
Recording page finds which track, and where in it, a clip aligns with.

Main Features:
* mel_patches: the audio resampled once to 8 kHz (the same helper the other
  stages use, so the already decoded track is reused), cut into one-second
  segments every HOP_SECONDS, each turned into the model's 256-band mel patch
  (n_fft 1024, hop 256, 160 to 4000 Hz, Slaney mel, magnitude, dB relative to
  the segment's own maximum, floored at -80 dB and scaled to [-1, 1]) that
  matches the reference essentia front end to 2e-4 on all 33 frames
* embed_patches runs the ONNX model in batches through the same provider
  chain as MusiCNN and CLAP (CUDA where the image has it, the CPU otherwise);
  fingerprint_audio chains both one batch of segments at a time, so a
  one-hour recording costs the same transient memory as a three-minute song,
  and is what the analysis stage and the search share
* product quantisation: every 128-vector is stored as 32 bytes, one byte per
  slice of four numbers, against a codebook of 256 centroids per slice that
  ships next to the model (neural_fingerprint_pq.npz, trained once on library
  fingerprints by scripts/onnx_export/train_neural_fingerprint_codebook.py
  through train_codebook); encode_codes and decode_codes convert both ways,
  the decoded vectors renormalised, and the codebook's checksum travels in
  every blob so a blob and a codebook that do not belong together are refused
* encode_blob / decode_blob: the code sequence behind a small header (magic
  NFP2, dimension, count, slices, codebook id), 14 KB per average track; a
  blob with another magic, dimension, slice count or codebook is refused
* the session is created once per process and released by unload_session,
  so the analysis worker can drop it between songs when
  PER_SONG_MODEL_RELOAD is set
"""

import logging
import os
import struct
import threading
import zlib

import numpy as np
import librosa
from numpy.lib.stride_tricks import sliding_window_view

import config

logger = logging.getLogger(__name__)

SAMPLE_RATE = 8000
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 256
F_MIN = 160.0
F_MAX = 4000.0
SEGMENT_SAMPLES = 8000
SEGMENT_FRAMES = 33
HOP_SAMPLES = 4000
HOP_SECONDS = HOP_SAMPLES / SAMPLE_RATE
DIM = 128
DYNAMIC_RANGE_DB = 80.0
LABEL = 'neural fingerprint'
PQ_SUBSPACES = 32
PQ_CENTROIDS = 256
PQ_SUBDIM = DIM // PQ_SUBSPACES
CODE_BYTES = PQ_SUBSPACES
BLOB_MAGIC = b'NFP2'
_AMIN = 1e-5
_HEADER = struct.Struct('<4sHIBI')
HEADER_BYTES = _HEADER.size
_BATCH = 128
_ENCODE_BATCH = 2048
_LOCK = threading.Lock()
_STATE = {'session': None, 'input': None, 'filterbank': None, 'codebook': None, 'codebook_id': None, 'codebook_bias': None}


DISABLED_MESSAGE = 'Neural fingerprint search is disabled. Set NEURAL_FINGERPRINT_ENABLED=true in config.'


def is_enabled():
    return bool(config.NEURAL_FINGERPRINT_ENABLED)


def is_available():
    if not is_enabled():
        return False
    paths = (config.NEURAL_FINGERPRINT_MODEL_PATH, config.NEURAL_FINGERPRINT_CODEBOOK_PATH)
    return all(bool(path) and os.path.isfile(path) for path in paths)


def _filterbank():
    if _STATE['filterbank'] is None:
        _STATE['filterbank'] = librosa.filters.mel(
            sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS, fmin=F_MIN, fmax=F_MAX, htk=False, norm='slaney'
        ).astype(np.float32)
    return _STATE['filterbank']


def _session():
    from tasks.onnx_utils import create_onnx_session, resolve_providers

    with _LOCK:
        if _STATE['session'] is None:
            if not is_available():
                raise RuntimeError(
                    f'The neural fingerprint model is missing at {config.NEURAL_FINGERPRINT_MODEL_PATH}.'
                )
            session = create_onnx_session(
                config.NEURAL_FINGERPRINT_MODEL_PATH,
                provider_options=resolve_providers(label=LABEL),
                label=LABEL,
            )
            _STATE['session'] = session
            _STATE['input'] = session.get_inputs()[0].name
        return _STATE['session'], _STATE['input']


def warm_session():
    if not is_available():
        return False
    _session()
    return True


def unload_session():
    with _LOCK:
        session = _STATE['session']
        _STATE['session'] = None
        _STATE['input'] = None
    if session is None:
        return False
    from tasks.memory_utils import cleanup_onnx_session

    cleanup_onnx_session(session, LABEL)
    return True


def segment_starts(n_samples, hop=HOP_SAMPLES):
    if n_samples < SEGMENT_SAMPLES:
        return np.zeros(0, dtype=np.int64)
    return np.arange(0, n_samples - SEGMENT_SAMPLES + 1, hop, dtype=np.int64)


def _model_signal(audio, sr):
    from tasks.analysis import resample_audio

    signal = np.asarray(audio, dtype=np.float32)
    if signal.ndim > 1:
        signal = signal.mean(axis=0)
    if int(sr) != SAMPLE_RATE:
        signal = np.asarray(resample_audio(signal, int(sr), SAMPLE_RATE), dtype=np.float32)
    return signal


def _patches_at(padded, starts):
    windows = sliding_window_view(padded, SEGMENT_SAMPLES + HOP_LENGTH)[starts]
    spectrum = librosa.stft(
        windows, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=N_FFT, window='hann',
        center=True, pad_mode='constant',
    )
    mel = np.einsum('mf,bft->bmt', _filterbank(), np.abs(spectrum).astype(np.float32))[:, :, :SEGMENT_FRAMES]
    mel = np.maximum(mel, _AMIN)
    peak = mel.reshape(mel.shape[0], -1).max(axis=1)[:, None, None]
    db = np.maximum(20.0 * np.log10(mel / peak), -DYNAMIC_RANGE_DB)
    patches = (1.0 + db / (DYNAMIC_RANGE_DB / 2.0)).astype(np.float32)
    return patches[:, :, :, None]


def _padded(signal):
    return np.concatenate([signal, np.zeros(HOP_LENGTH, dtype=np.float32)])


def mel_patches(audio, sr, hop=HOP_SAMPLES):
    signal = _model_signal(audio, sr)
    starts = segment_starts(signal.size, hop)
    if not starts.size:
        return None
    return _patches_at(_padded(signal), starts)


def embed_patches(patches):
    session, input_name = _session()
    out = []
    for start in range(0, patches.shape[0], _BATCH):
        out.append(session.run(None, {input_name: np.ascontiguousarray(patches[start:start + _BATCH])})[0])
    vectors = np.concatenate(out, axis=0).astype(np.float32)
    return vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9)


def fingerprint_audio(audio, sr, hop=HOP_SAMPLES):
    signal = _model_signal(audio, sr)
    starts = segment_starts(signal.size, hop)
    if not starts.size:
        return None
    padded = _padded(signal)
    return np.concatenate([
        embed_patches(_patches_at(padded, starts[block:block + _BATCH])) for block in range(0, starts.size, _BATCH)
    ])


def codebook_id(book):
    return zlib.crc32(np.ascontiguousarray(book, dtype=np.float32).tobytes()) & 0xFFFFFFFF


def load_codebook(path):
    with np.load(path, allow_pickle=False) as data:
        book = np.ascontiguousarray(data['codebook'], dtype=np.float32)
    expected = (PQ_SUBSPACES, PQ_CENTROIDS, PQ_SUBDIM)
    if book.shape != expected:
        raise ValueError(f'The codebook {path} has shape {book.shape}, expected {expected}')
    return book


def _codebook_bias(book):
    return 0.5 * np.einsum('skd,skd->sk', book, book)


def codebook():
    with _LOCK:
        if _STATE['codebook'] is None:
            path = config.NEURAL_FINGERPRINT_CODEBOOK_PATH
            if not path or not os.path.isfile(path):
                raise RuntimeError(f'The neural fingerprint codebook is missing at {path}.')
            book = load_codebook(path)
            _STATE['codebook'] = book
            _STATE['codebook_id'] = codebook_id(book)
            _STATE['codebook_bias'] = _codebook_bias(book)
        return _STATE['codebook'], _STATE['codebook_id'], _STATE['codebook_bias']


def _check_matrix(vectors):
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != DIM:
        raise ValueError(f'expected a (n, {DIM}) matrix, got {matrix.shape}')
    return matrix


def encode_codes(vectors, book=None):
    matrix = _check_matrix(vectors)
    if book is None:
        book, _, bias = codebook()
    else:
        bias = _codebook_bias(book)
    codes = np.empty((matrix.shape[0], PQ_SUBSPACES), dtype=np.uint8)
    for start in range(0, matrix.shape[0], _ENCODE_BATCH):
        block = matrix[start:start + _ENCODE_BATCH].reshape(-1, PQ_SUBSPACES, PQ_SUBDIM)
        scores = np.einsum('nsd,skd->nsk', block, book) - bias[None]
        codes[start:start + _ENCODE_BATCH] = np.argmax(scores, axis=2)
    return codes


def decode_codes(codes, book=None):
    if book is None:
        book = codebook()[0]
    matrix = np.asarray(codes, dtype=np.uint8)
    if matrix.ndim != 2 or matrix.shape[1] != PQ_SUBSPACES:
        raise ValueError(f'expected a (n, {PQ_SUBSPACES}) code matrix, got {matrix.shape}')
    vectors = book[np.arange(PQ_SUBSPACES)[None, :], matrix].reshape(matrix.shape[0], DIM)
    return vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9)


def _nearest_centroid(points, centroids):
    bias = 0.5 * np.einsum('kd,kd->k', centroids, centroids)
    labels = np.empty(points.shape[0], dtype=np.int64)
    for start in range(0, points.shape[0], 65536):
        block = points[start:start + 65536]
        labels[start:start + 65536] = np.argmax(block @ centroids.T - bias[None], axis=1)
    return labels


def train_codebook(vectors, iterations=25, seed=0):
    matrix = _check_matrix(vectors)
    if matrix.shape[0] < PQ_CENTROIDS:
        raise ValueError(f'need at least {PQ_CENTROIDS} vectors to train a codebook, got {matrix.shape[0]}')
    rng = np.random.default_rng(seed)
    parts = matrix.reshape(matrix.shape[0], PQ_SUBSPACES, PQ_SUBDIM)
    book = np.empty((PQ_SUBSPACES, PQ_CENTROIDS, PQ_SUBDIM), dtype=np.float32)
    for s in range(PQ_SUBSPACES):
        points = np.ascontiguousarray(parts[:, s, :])
        centroids = points[rng.choice(points.shape[0], PQ_CENTROIDS, replace=False)].copy()
        for _ in range(iterations):
            labels = _nearest_centroid(points, centroids)
            counts = np.bincount(labels, minlength=PQ_CENTROIDS).astype(np.float32)
            sums = np.stack(
                [np.bincount(labels, weights=points[:, d], minlength=PQ_CENTROIDS) for d in range(PQ_SUBDIM)], axis=1
            ).astype(np.float32)
            filled = counts > 0
            centroids[filled] = sums[filled] / counts[filled, None]
            empty = np.flatnonzero(~filled)
            if empty.size:
                centroids[empty] = points[rng.choice(points.shape[0], empty.size, replace=False)]
        book[s] = centroids
    return book


def encode_blob(vectors):
    matrix = _check_matrix(vectors)
    book, book_id, _ = codebook()
    codes = encode_codes(matrix, book)
    return _HEADER.pack(BLOB_MAGIC, DIM, int(codes.shape[0]), PQ_SUBSPACES, book_id) + codes.tobytes()


def decode_blob(blob):
    raw = bytes(blob)
    if len(raw) < _HEADER.size:
        return None
    magic, dim, count, subspaces, book_id = _HEADER.unpack_from(raw)
    if magic != BLOB_MAGIC or dim != DIM or subspaces != PQ_SUBSPACES:
        return None
    loaded_id = codebook()[1]
    if book_id != loaded_id:
        logger.error(
            'Neural fingerprint blob encoded with codebook %08x but codebook %08x is loaded; '
            'restore the matching neural_fingerprint_pq.npz or re-analyse the library',
            book_id, loaded_id,
        )
        return None
    body = np.frombuffer(raw, dtype=np.uint8, offset=_HEADER.size)
    if body.size != count * subspaces:
        return None
    return body.reshape(count, subspaces)


def decode_blob_f32(blob):
    codes = decode_blob(blob)
    if codes is None:
        return None
    return decode_codes(codes)


def fingerprint_track(audio, sr, track_name=''):
    vectors = fingerprint_audio(audio, sr)
    if vectors is None:
        logger.warning('  - Neural fingerprint skipped for %s: shorter than one second', track_name)
        return None
    if config.PER_SONG_MODEL_RELOAD:
        unload_session()
    return encode_blob(vectors)
