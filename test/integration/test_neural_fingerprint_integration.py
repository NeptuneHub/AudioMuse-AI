# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""End-to-end test of the real neural fingerprint encoder, codebook and search.

Runs tasks.neural_fingerprint over the three test songs with the actual ONNX
encoder and the shipped codebook, checks the sequences against values recorded
from the same files, and identifies a noisy slice of one song through the real
pack, cells and vote-and-align search.

Main Features:
* Per song the segment count is exact, the mean cosine between neighbouring
  and two-seconds-apart segments and the norm of the mean vector are within
  0.01 of the recorded values, and the recorded 32-byte codes of the first and
  the middle segment match on at least 13 of 16 bytes.
* The three songs' mean vectors keep their recorded pairwise cosines.
* A 10-second slice of one song under 10 dB of white noise comes back at rank
  one, at its offset, flagged identified with the recorded score, and the
  other songs are not flagged.
"""

from pathlib import Path

import numpy as np
import pytest

TOLERANCE = 0.01
CODE_BYTES_CHECKED = 16
CODE_BYTES_MATCHING = 13
EXPECTED = {
    'Aaron Dunn - Minuet - Notebook for Anna Magdalena.mp3': {
        'segments': 99, 'consecutive_cosine': 0.7176, 'two_second_cosine': 0.3489, 'mean_vector_norm': 0.5689,
        'first_codes': '59252da96de7281312b0aa5016a0fcf5', 'middle_codes': 'f84f0239bb5a813bd26433033327079d',
    },
    'Art Flower - Art Flower - Creamy Snowflakes.mp3': {
        'segments': 610, 'consecutive_cosine': 0.8027, 'two_second_cosine': 0.3867, 'mean_vector_norm': 0.4448,
        'first_codes': '6f018a574c76cd40f735f1df096cb938', 'middle_codes': 'de53f964cbaef5e4f50835c95ead69ff',
    },
    "Michael Hawley - Sonata 'Waldstein', Op. 53 - II. Introduzione-Adagio molto.mp3": {
        'segments': 410, 'consecutive_cosine': 0.8695, 'two_second_cosine': 0.5426, 'mean_vector_norm': 0.4783,
        'first_codes': 'c743b9c5cfc72f7fa567f856c19fe56d', 'middle_codes': '03da03530fc22bfba08b911646949857',
    },
}
EXPECTED_PAIR_COSINES = {(0, 1): -0.1193, (0, 2): 0.1464, (1, 2): -0.0840}
QUERY_SONG = "Michael Hawley - Sonata 'Waldstein', Op. 53 - II. Introduzione-Adagio molto.mp3"
QUERY_START_SECONDS = 20.0
QUERY_SECONDS = 10.0
NOISE_SNR_DB = 10.0
EXPECTED_QUERY_SCORE = 0.775
QUERY_SCORE_TOLERANCE = 0.03


def _noisy_slice(audio, sr):
    start = int(QUERY_START_SECONDS * sr)
    clip = np.array(audio[start:start + int(QUERY_SECONDS * sr)], dtype=np.float32)
    noise = np.random.default_rng(0).standard_normal(clip.size).astype(np.float32)
    clip_rms = float(np.sqrt(np.mean(clip ** 2)) + 1e-9)
    noise *= clip_rms / (float(np.sqrt(np.mean(noise ** 2))) + 1e-9) / 10 ** (NOISE_SNR_DB / 20)
    return clip + noise


def _matching_bytes(codes_row, expected_hex):
    expected = np.frombuffer(bytes.fromhex(expected_hex), dtype=np.uint8)
    return int(np.sum(codes_row[:CODE_BYTES_CHECKED] == expected))


def _model_file(configured, name, project_root):
    for candidate in (configured, project_root / 'model' / name, project_root / 'test' / 'models' / name):
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    return None


@pytest.mark.integration
def test_real_neural_fingerprint_matches_recorded_values_and_identifies_a_noisy_slice(monkeypatch, tmp_path):
    project_root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(project_root))
    try:
        import onnxruntime
    except Exception as exc:  # pragma: no cover
        pytest.skip(f'onnxruntime not importable: {exc}')
    try:
        import librosa
    except Exception as exc:  # pragma: no cover
        pytest.skip(f'librosa not importable: {exc}')

    import config
    from tasks import neural_fingerprint as nf
    from tasks import neural_fingerprint_index as nfi

    model_path = _model_file(config.NEURAL_FINGERPRINT_MODEL_PATH, 'neural_fingerprint.onnx', project_root)
    codebook_path = _model_file(config.NEURAL_FINGERPRINT_CODEBOOK_PATH, 'neural_fingerprint_pq.npz', project_root)
    if model_path is None or codebook_path is None:
        pytest.skip(
            'neural fingerprint model or codebook missing: download both from the model release into model/ '
            f'or test/models/ ({config.NEURAL_FINGERPRINT_MODEL_PATH}, {config.NEURAL_FINGERPRINT_CODEBOOK_PATH})'
        )
    monkeypatch.setattr(config, 'NEURAL_FINGERPRINT_MODEL_PATH', str(model_path))
    monkeypatch.setattr(config, 'NEURAL_FINGERPRINT_CODEBOOK_PATH', str(codebook_path))
    songs = [project_root / 'test' / 'songs' / name for name in EXPECTED]
    missing = [path.name for path in songs if not path.is_file()]
    if missing:
        pytest.skip(f'test songs missing under test/songs: {missing}')

    print(
        f'\n[neural-fingerprint-test] codebook id {nf.codebook()[1]:08x}, model {model_path.name}, '
        f'onnxruntime {onnxruntime.__version__}'
    )
    failures = []
    blobs = []
    audios = {}
    mean_vectors = []
    for path in songs:
        expected = EXPECTED[path.name]
        audio, sr = librosa.load(str(path), sr=None, mono=True)
        audios[path.name] = (audio, sr)
        vectors = nf.fingerprint_audio(audio, sr)
        assert vectors is not None, f'{path.name}: no fingerprint produced'
        codes = nf.encode_codes(vectors)
        blobs.append((path.name, nf.encode_blob(vectors)))
        mean_vector = vectors.mean(axis=0)
        mean_vectors.append(mean_vector / np.linalg.norm(mean_vector))
        measured = {
            'segments': int(vectors.shape[0]),
            'consecutive_cosine': float(np.einsum('ij,ij->i', vectors[:-1], vectors[1:]).mean()),
            'two_second_cosine': float(np.einsum('ij,ij->i', vectors[:-4], vectors[4:]).mean()),
            'mean_vector_norm': float(np.linalg.norm(mean_vector)),
        }
        first = _matching_bytes(codes[0], expected['first_codes'])
        middle = _matching_bytes(codes[vectors.shape[0] // 2], expected['middle_codes'])
        print(f'\n=== {path.name}')
        for key, value in measured.items():
            print(f'  {key:19s}: {value:.4f} (expected {expected[key]})')
        print(f'  code bytes matching: first {first}/{CODE_BYTES_CHECKED}, middle {middle}/{CODE_BYTES_CHECKED}')
        if measured['segments'] != expected['segments']:
            failures.append(f'{path.name}: {measured["segments"]} segments, expected {expected["segments"]}')
        for key in ('consecutive_cosine', 'two_second_cosine', 'mean_vector_norm'):
            if abs(measured[key] - expected[key]) > TOLERANCE:
                failures.append(f'{path.name}: {key} {measured[key]:.4f}, expected {expected[key]}')
        if min(first, middle) < CODE_BYTES_MATCHING:
            failures.append(f'{path.name}: codes match on {first} and {middle} of {CODE_BYTES_CHECKED} bytes')

    for (i, j), expected_cosine in EXPECTED_PAIR_COSINES.items():
        cosine = float(mean_vectors[i] @ mean_vectors[j])
        print(f'  mean-vector cosine {i}-{j}: {cosine:.4f} (expected {expected_cosine})')
        if abs(cosine - expected_cosine) > TOLERANCE:
            failures.append(f'mean-vector cosine {i}-{j} {cosine:.4f}, expected {expected_cosine}')

    codes = {name: nf.decode_blob(blob) for name, blob in blobs}
    sample = nfi.training_sample(iter(codes.items()), 20000, np.random.default_rng(0))
    quantizer = nfi.train_quantizer(sample, nfi._cell_count(sum(int(c.shape[0]) for c in codes.values())))
    ids, lengths, labels, part_codes = next(iter(nfi.label_tracks(iter(codes.items()), quantizer)))
    tracks_of, offsets = nfi.track_offset_arrays(0, lengths)
    cells = nfi.part_cells(0, labels, part_codes, tracks_of, offsets, quantizer.n_cells)
    store = {}
    while True:
        try:
            name, cell, blob = next(cells)
        except StopIteration as done:
            counts = done.value
            break
        store[(name, cell)] = blob
    directory = nfi.unpack_directory(
        nfi.pack_directory('test-build', nf.codebook()[1], quantizer, ids, lengths, counts, 1, len(ids))
    )
    nfi.unload()
    nfi._swap_pack(nfi._pack_from(directory))
    monkeypatch.setattr(nfi, 'ensure_loaded', lambda: True)
    monkeypatch.setattr(
        nfi, '_read_cell_rows',
        lambda wanted: [(name, cell, blob) for (name, cell), blob in store.items() if cell in set(wanted)],
    )
    monkeypatch.setattr(nfi, '_candidate_codes', lambda wanted: {i: codes[i] for i in wanted if i in codes})
    try:
        audio, sr = audios[QUERY_SONG]
        rows = nfi.identify(_noisy_slice(audio, sr), sr, len(EXPECTED))
    finally:
        nfi.unload()
    print(f'\n=== identify {QUERY_SECONDS:.0f} s of {QUERY_SONG} at {QUERY_START_SECONDS:.0f} s under {NOISE_SNR_DB:.0f} dB SNR')
    for row in rows:
        print(f'  {row["item_id"][:40]:40s} score {row["score"]:.3f} at {row["offset_seconds"]:6.1f} s identified {row["identified"]}')
    assert rows[0]['item_id'] == QUERY_SONG
    assert abs(rows[0]['offset_seconds'] - QUERY_START_SECONDS) <= 0.5
    assert rows[0]['identified'] is True
    assert abs(rows[0]['score'] - EXPECTED_QUERY_SCORE) <= QUERY_SCORE_TOLERANCE
    assert not any(row['identified'] for row in rows[1:])
    assert not failures, '\n'.join(failures)
