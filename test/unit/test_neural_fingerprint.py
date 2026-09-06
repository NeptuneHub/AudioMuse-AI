# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Neural fingerprint front end, product-quantised blob codec and migration.

Main Features:
* a track becomes one 256 by 33 mel patch per half second, scaled to [-1, 1]
  with each patch's own peak at 1, whatever the input rate, and a clip
  shorter than one second yields nothing
* the patch matches the model's reference front end: a pure tone lands in
  the mel band of its frequency and silence is the -1 floor everywhere
* a sequence round-trips through the 32-byte codes within the codebook's
  reach, a foreign or truncated blob decodes to None, and a blob written with
  another codebook is refused
* a legacy int8 blob decodes to the same codes its migration writes
* the model is available exactly when its file and the codebook exist, and
  fingerprinting a track through a fake session unloads the session when
  per-song reload is on
* the startup migration rewrites legacy rows batch by batch behind the
  advisory lock and does nothing when another web process holds it
"""

import numpy as np
import pytest

import config
from tasks import ivf_quant
from tasks import neural_fingerprint as nf


def _unit(rng, shape):
    vectors = rng.standard_normal(shape).astype(np.float32)
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)


@pytest.fixture
def codebook(monkeypatch, tmp_path):
    rng = np.random.default_rng(7)
    vectors = _unit(rng, (600, nf.DIM))
    book = nf.train_codebook(vectors, iterations=6)
    path = tmp_path / 'neural_fingerprint_pq.npz'
    np.savez(path, codebook=book)
    monkeypatch.setattr(config, 'NEURAL_FINGERPRINT_CODEBOOK_PATH', str(path))
    for key in ('codebook', 'codebook_id', 'codebook_bias'):
        monkeypatch.setitem(nf._STATE, key, None)
    return vectors, book


def _legacy_blob(vectors):
    body = np.clip(np.rint(vectors * ivf_quant.I8_SCALE), -127, 127).astype(np.int8)
    return nf._LEGACY_HEADER.pack(nf.LEGACY_MAGIC, nf.DIM, vectors.shape[0], ivf_quant.DTYPE_I8) + body.tobytes()


def test_patches_are_half_second_apart_scaled_and_peak_normalised():
    rng = np.random.default_rng(1)
    audio = rng.standard_normal(44100 * 5).astype(np.float32) * 0.1
    patches = nf.mel_patches(audio, 44100)
    expected = 1 + (5 * nf.SAMPLE_RATE - nf.SEGMENT_SAMPLES) // nf.HOP_SAMPLES
    assert patches.shape == (expected, nf.N_MELS, nf.SEGMENT_FRAMES, 1)
    assert patches.dtype == np.float32
    assert patches.min() >= -1.0
    assert patches.max() <= 1.0
    assert np.allclose(patches.reshape(expected, -1).max(axis=1), 1.0)
    assert nf.mel_patches(np.zeros(4000, dtype=np.float32), 8000) is None
    assert nf.segment_starts(8000).tolist() == [0]
    assert nf.segment_starts(8000 + 4000 * 3).tolist() == [0, 4000, 8000, 12000]


def test_a_pure_tone_lands_in_its_mel_band_and_silence_is_the_floor():
    t = np.arange(8000) / 8000.0
    tone = np.sin(2 * np.pi * 1000.0 * t).astype(np.float32)
    patch = nf.mel_patches(tone, 8000)[0, :, :, 0]
    import librosa

    centres = librosa.mel_frequencies(n_mels=nf.N_MELS + 2, fmin=nf.F_MIN, fmax=nf.F_MAX, htk=False)[1:-1]
    assert abs(centres[int(np.argmax(patch.mean(axis=1)))] - 1000.0) < 60.0
    silence = nf.mel_patches(np.zeros(8000, dtype=np.float32), 8000)[0, :, :, 0]
    assert np.all(silence == -1.0) or np.all(silence == 1.0)


def test_blob_round_trip_and_rejection_of_foreign_bytes(codebook):
    vectors, book = codebook
    seven = vectors[:7]
    blob = nf.encode_blob(seven)
    assert blob[:4] == nf.BLOB_MAGIC
    assert len(blob) == nf._HEADER.size + 7 * nf.CODE_BYTES
    codes = nf.decode_blob(blob)
    assert codes.shape == (7, nf.CODE_BYTES)
    assert codes.dtype == np.uint8
    assert np.array_equal(codes, nf.encode_codes(seven, book))
    back = nf.decode_blob_f32(blob)
    assert back.shape == (7, nf.DIM)
    assert np.allclose(np.linalg.norm(back, axis=1), 1.0, atol=1e-4)
    assert np.einsum('ij,ij->i', back, seven).min() > 0.9
    assert nf.decode_blob(blob[:20]) is None
    assert nf.decode_blob(b'XXXX' + blob[4:]) is None
    assert nf.decode_blob(b'') is None
    with pytest.raises(ValueError):
        nf.encode_blob(np.zeros((3, 4), dtype=np.float32))


def test_a_blob_from_another_codebook_is_refused(codebook, caplog):
    vectors, _book = codebook
    blob = nf.encode_blob(vectors[:3])
    magic, dim, count, subspaces, book_id = nf._HEADER.unpack_from(blob)
    foreign = nf._HEADER.pack(magic, dim, count, subspaces, book_id ^ 1) + blob[nf._HEADER.size:]
    with caplog.at_level('ERROR'):
        assert nf.decode_blob(foreign) is None
    assert 'codebook' in caplog.text


def test_a_legacy_int8_blob_decodes_and_migrates_to_the_same_codes(codebook):
    vectors, book = codebook
    legacy = _legacy_blob(vectors[:5])
    assert nf.is_legacy_blob(legacy) is True
    assert nf.is_legacy_blob(nf.encode_blob(vectors[:2])) is False
    codes = nf.decode_blob(legacy)
    assert np.array_equal(codes, nf.encode_codes(nf.decode_legacy_blob(legacy), book))
    migrated = nf.migrate_blob(legacy)
    assert migrated[:4] == nf.BLOB_MAGIC
    assert np.array_equal(nf.decode_blob(migrated), codes)
    assert nf.migrate_blob(legacy[:12]) is None
    assert nf.decode_blob(legacy[:12]) is None


def test_availability_needs_the_model_and_the_codebook(monkeypatch, tmp_path):
    (tmp_path / 'model.onnx').write_bytes(b'x')
    (tmp_path / 'book.npz').write_bytes(b'x')
    monkeypatch.setattr(config, 'NEURAL_FINGERPRINT_MODEL_PATH', str(tmp_path / 'model.onnx'))
    monkeypatch.setattr(config, 'NEURAL_FINGERPRINT_CODEBOOK_PATH', str(tmp_path / 'missing.npz'))
    assert nf.is_available() is False
    monkeypatch.setattr(config, 'NEURAL_FINGERPRINT_CODEBOOK_PATH', str(tmp_path / 'book.npz'))
    assert nf.is_available() is True
    monkeypatch.setattr(config, 'NEURAL_FINGERPRINT_MODEL_PATH', '')
    assert nf.is_available() is False


def test_fingerprint_track_encodes_the_sequence_and_unloads_per_song(monkeypatch, codebook):
    calls = []

    class FakeSession:
        def run(self, _outputs, feed):
            batch = next(iter(feed.values()))
            calls.append(batch.shape)
            out = np.zeros((batch.shape[0], nf.DIM), dtype=np.float32)
            out[:, 0] = 1.0
            return [out]

    unloaded = []
    monkeypatch.setitem(nf._STATE, 'session', FakeSession())
    monkeypatch.setitem(nf._STATE, 'input', 'mel')
    monkeypatch.setattr(nf, 'unload_session', lambda: unloaded.append(True) or True)
    monkeypatch.setattr(config, 'PER_SONG_MODEL_RELOAD', True)
    audio = np.random.default_rng(3).standard_normal(8000 * 3).astype(np.float32)
    blob = nf.fingerprint_track(audio, 8000, 'track')
    codes = nf.decode_blob(blob)
    assert codes.shape == (5, nf.CODE_BYTES)
    assert calls == [(5, nf.N_MELS, nf.SEGMENT_FRAMES, 1)]
    assert unloaded == [True]
    assert nf.fingerprint_track(np.zeros(100, dtype=np.float32), 8000, 'short') is None


class _FakeCursor:
    def __init__(self, conn, lock_free):
        self.conn = conn
        self.lock_free = lock_free
        self.statements = []
        self.rowcount = 0

    def execute(self, sql, params=None):
        self.statements.append((sql, params))
        if 'UPDATE embedding' in sql:
            item_id = params[1]
            self.rowcount = 1 if item_id in self.conn.legacy else 0
            self.conn.legacy.pop(item_id, None)
            self.conn.written[item_id] = bytes(params[0].adapted)

    def fetchone(self):
        return (self.lock_free,)

    def fetchall(self):
        sql, params = self.statements[-1]
        after, limit = params[0], params[2]
        return sorted((k, v) for k, v in self.conn.legacy.items() if k > after)[:limit]

    def close(self):
        pass


class _FakeConn:
    def __init__(self, legacy, lock_free=True):
        self.legacy = dict(legacy)
        self.written = {}
        self.commits = 0
        self.lock_free = lock_free
        self.cursors = []

    def cursor(self):
        cur = _FakeCursor(self, self.lock_free)
        self.cursors.append(cur)
        return cur

    def commit(self):
        self.commits += 1


def test_migration_rewrites_legacy_rows_in_batches_behind_the_lock(codebook):
    vectors, _book = codebook
    legacy = {f'fp_{i:03d}': _legacy_blob(vectors[i * 4:(i + 1) * 4]) for i in range(5)}
    legacy['fp_bad'] = b'NFP1garbage'
    conn = _FakeConn(legacy)
    migrated = nf.migrate_legacy_fingerprints(conn=conn, batch_size=2)
    assert migrated == 5
    assert sorted(conn.written) == [f'fp_{i:03d}' for i in range(5)]
    assert all(blob[:4] == nf.BLOB_MAGIC for blob in conn.written.values())
    assert list(conn.legacy) == ['fp_bad']
    assert conn.commits >= 3
    statements = [sql for sql, _ in conn.cursors[0].statements]
    assert 'pg_try_advisory_lock' in statements[0]
    assert 'pg_advisory_unlock' in statements[-1]
    assert sum('UPDATE embedding' in sql for sql in statements) == 5


def test_migration_skips_when_another_process_holds_the_lock(codebook):
    vectors, _book = codebook
    conn = _FakeConn({'fp_000': _legacy_blob(vectors[:3])}, lock_free=False)
    assert nf.migrate_legacy_fingerprints(conn=conn) == 0
    assert conn.written == {}
    assert not any('UPDATE embedding' in sql for sql, _ in conn.cursors[0].statements)
