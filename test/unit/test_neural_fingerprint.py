# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Neural fingerprint front end and product-quantised blob codec.

Main Features:
* a track becomes one 256 by 33 mel patch per half second, scaled to [-1, 1]
  with each patch's own peak at 1, whatever the input rate, and a clip
  shorter than one second yields nothing
* the patch matches the model's reference front end: a pure tone lands in
  the mel band of its frequency and silence is the -1 floor everywhere
* a sequence round-trips through the 32-byte codes within the codebook's
  reach, a foreign or truncated blob decodes to None, and a blob written with
  another codebook is refused
* the model is available exactly when its file and the codebook exist, and
  fingerprinting a track through a fake session unloads the session when
  per-song reload is on
"""

import numpy as np
import pytest

import config
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


def test_the_enabled_flag_switches_the_feature_off_even_with_both_files_present(monkeypatch, tmp_path):
    (tmp_path / 'model.onnx').write_bytes(b'x')
    (tmp_path / 'book.npz').write_bytes(b'x')
    monkeypatch.setattr(config, 'NEURAL_FINGERPRINT_MODEL_PATH', str(tmp_path / 'model.onnx'))
    monkeypatch.setattr(config, 'NEURAL_FINGERPRINT_CODEBOOK_PATH', str(tmp_path / 'book.npz'))
    assert nf.is_available() is True
    monkeypatch.setattr(config, 'NEURAL_FINGERPRINT_ENABLED', False)
    assert nf.is_enabled() is False
    assert nf.is_available() is False
    assert nf.warm_session() is False


def test_the_session_comes_from_the_shared_provider_chain(monkeypatch, tmp_path):
    from tasks import onnx_utils

    seen = {}

    class FakeSession:
        def get_inputs(self):
            return [type('Input', (), {'name': 'mel'})()]

    monkeypatch.setattr(onnx_utils, 'resolve_providers', lambda **kw: seen.update(chain=kw) or [('FakeProvider', {})])
    monkeypatch.setattr(
        onnx_utils, 'create_onnx_session',
        lambda path, provider_options=None, label=None, **kw: seen.update(path=path, providers=provider_options) or FakeSession(),
    )
    (tmp_path / 'model.onnx').write_bytes(b'x')
    (tmp_path / 'book.npz').write_bytes(b'x')
    monkeypatch.setattr(config, 'NEURAL_FINGERPRINT_MODEL_PATH', str(tmp_path / 'model.onnx'))
    monkeypatch.setattr(config, 'NEURAL_FINGERPRINT_CODEBOOK_PATH', str(tmp_path / 'book.npz'))
    for key in ('session', 'input'):
        monkeypatch.setitem(nf._STATE, key, None)
    session, input_name = nf._session()
    assert isinstance(session, FakeSession)
    assert input_name == 'mel'
    assert seen['path'] == str(tmp_path / 'model.onnx')
    assert seen['providers'] == [('FakeProvider', {})]
    assert seen['chain'] == {'label': 'neural fingerprint'}


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
