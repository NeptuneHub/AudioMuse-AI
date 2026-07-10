# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Content-based catalogue id from an acoustic fingerprint.

Turns a song's audio into one stable 64-bit id so that the same recording -
even re-encoded, retagged, or with tiny differences - collapses to the SAME id
across every media server. The pipeline is: audio -> acoustic fingerprint
(Chromaprint when the library is available, otherwise a librosa chroma
fingerprint using the identical downstream steps) -> aggregate the per-frame
fingerprint tokens into a 64-bit SimHash -> store as a signed BIGINT and look it
up by exact equality. No external network service is ever contacted.

Main Features:
* Chromaprint fingerprint via pyacoustid when present, with a pure-librosa
  chroma fallback so every container and native build can fingerprint audio.
* Deterministic 64-bit SimHash (stable across platforms via blake2b) that maps
  near-identical audio to the same id, stored as a Postgres-friendly BIGINT.
"""

import hashlib
import logging

logger = logging.getLogger(__name__)

_SIMHASH_BITS = 64
_UINT64_MASK = (1 << 64) - 1

_chromaprint_checked = False
_chromaprint_module = None


def _get_chromaprint():
    global _chromaprint_checked, _chromaprint_module
    if not _chromaprint_checked:
        _chromaprint_checked = True
        try:
            import acoustid.chromaprint as _cp
            _chromaprint_module = _cp
        except Exception:
            try:
                import chromaprint as _cp
                _chromaprint_module = _cp
            except Exception:
                _chromaprint_module = None
    return _chromaprint_module


def _to_int16_mono(samples):
    import numpy as np

    arr = np.asarray(samples)
    if arr.ndim > 1:
        arr = arr.mean(axis=tuple(range(1, arr.ndim)))
    if arr.dtype.kind == 'f':
        arr = np.clip(arr, -1.0, 1.0)
        arr = (arr * 32767.0).astype(np.int16)
    elif arr.dtype != np.int16:
        arr = arr.astype(np.int16)
    return arr


def _chromaprint_tokens(samples, sample_rate):
    cp = _get_chromaprint()
    if cp is None:
        return None
    try:
        pcm = _to_int16_mono(samples)
        fingerprinter = cp.Fingerprinter()
        fingerprinter.start(int(sample_rate), 1)
        fingerprinter.feed(pcm.tobytes())
        fingerprinter.finish()
        raw = fingerprinter.get_fingerprint()
        if not raw:
            return None
        return [int(v) & _UINT64_MASK for v in raw]
    except Exception:
        logger.debug("Chromaprint fingerprinting failed; using chroma fallback", exc_info=True)
        return None


def _chroma_tokens(samples, sample_rate):
    try:
        import numpy as np
        import librosa

        arr = np.asarray(samples, dtype=np.float32)
        if arr.ndim > 1:
            arr = arr.mean(axis=tuple(range(1, arr.ndim)))
        if arr.size < sample_rate:
            return None
        chroma = librosa.feature.chroma_cqt(y=arr, sr=int(sample_rate))
        n_frames = chroma.shape[1]
        if n_frames < 4:
            return None
        tokens = []
        prev = None
        frame_mean = chroma.mean(axis=0)
        for i in range(n_frames):
            col = chroma[:, i]
            bits = 0
            thr = frame_mean[i]
            for b in range(12):
                if col[b] > thr:
                    bits |= (1 << b)
            if prev is not None:
                for b in range(12):
                    if col[b] > prev[b]:
                        bits |= (1 << (12 + b))
            bits |= (int(np.argmax(col)) & 0xF) << 24
            tokens.append(bits & _UINT64_MASK)
            prev = col
        return tokens or None
    except Exception:
        logger.debug("Chroma fallback fingerprinting failed", exc_info=True)
        return None


def _token_hash(token):
    digest = hashlib.blake2b(int(token).to_bytes(8, 'little'), digest_size=8).digest()
    return int.from_bytes(digest, 'little')


def simhash64(tokens):
    """Aggregate per-frame fingerprint tokens into a stable unsigned 64-bit SimHash."""
    if not tokens:
        return None
    acc = [0] * _SIMHASH_BITS
    for token in tokens:
        h = _token_hash(token)
        for bit in range(_SIMHASH_BITS):
            if (h >> bit) & 1:
                acc[bit] += 1
            else:
                acc[bit] -= 1
    out = 0
    for bit in range(_SIMHASH_BITS):
        if acc[bit] > 0:
            out |= (1 << bit)
    return out


def to_signed_bigint(value):
    """Map an unsigned 64-bit value into the signed range Postgres BIGINT accepts."""
    if value is None:
        return None
    value &= _UINT64_MASK
    return value - (1 << 64) if value >= (1 << 63) else value


def from_signed_bigint(value):
    if value is None:
        return None
    return value + (1 << 64) if value < 0 else value


def fingerprint_tokens(samples, sample_rate):
    """Per-frame fingerprint tokens for the audio, Chromaprint first then chroma."""
    tokens = _chromaprint_tokens(samples, sample_rate)
    if tokens:
        return tokens
    return _chroma_tokens(samples, sample_rate)


def canonical_fingerprint(samples, sample_rate):
    """Signed 64-bit BIGINT catalogue id for the audio, or None if unavailable."""
    tokens = fingerprint_tokens(samples, sample_rate)
    if not tokens:
        return None
    return to_signed_bigint(simhash64(tokens))


_ID_PREFIX = "fp_"


def canonical_id_str(fingerprint):
    """The catalogue item_id string for a signed-BIGINT fingerprint, or None."""
    if fingerprint is None:
        return None
    return _ID_PREFIX + ("%016x" % from_signed_bigint(fingerprint))


def is_fingerprint_id(item_id):
    return isinstance(item_id, str) and item_id.startswith(_ID_PREFIX)


def canonical_fingerprint_file(path, target_sr=16000):
    """Load an audio file and return its signed 64-bit catalogue id, or None."""
    try:
        import librosa

        samples, sr = librosa.load(str(path), sr=target_sr, mono=True)
        if samples is None or samples.size == 0:
            return None
        return canonical_fingerprint(samples, sr)
    except Exception:
        logger.debug("Could not fingerprint audio file %s", path, exc_info=True)
        return None
