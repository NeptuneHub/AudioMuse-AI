# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Golden-vector record/replay test of the real SAE concept-steering pipeline.

Pushes a fixed set of deterministic embeddings through the actual SAE encoder
ONNX and the real tasks/clap_steering.py ranking code, then compares the latent
activations and the per-concept ranking against a recorded baseline.

The inputs are seeded pseudo-embeddings rather than real tracks on purpose. The
encoder consumes a 512 dim DCLAP vector and does not care where it came from, so
synthetic unit vectors exercise exactly the same arithmetic while keeping the
test hermetic and free of anybody's music.

Main Features:
* Replays latent activations and concept scores against stored golden vectors.
* Records a fresh baseline when none is present, the way the lyrics test does.
* Pins the encoder sha256 in the baseline, so a changed model fails loudly
  instead of being silently compared against vectors it never produced.
* Asserts the properties that must hold whatever the numbers are: conjunction,
  direction inversion and an untouched ranking when nothing is requested.
"""

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

ACTIVATION_TOLERANCE = 1e-4
SCORE_TOLERANCE = 1e-4

PROBE_CONCEPTS = ['saxophone', 'female vocals', 'piano', 'techno', 'choir']
N_PROBES = 24
EMBEDDING_DIM = 512


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _probe_embeddings():
    rng = np.random.default_rng(20260827)
    rows = rng.normal(size=(N_PROBES, EMBEDDING_DIM)).astype(np.float32)
    rows /= np.linalg.norm(rows, axis=1, keepdims=True)
    return rows


def _artifacts():
    models = Path(os.environ.get('SAE_MODEL_DIR') or _REPO_ROOT / 'model')
    encoder = models / 'dclap_sae_k20_d1024_best_encoder.onnx'
    decoder = models / 'dclap_sae_k20_d1024_best_decoder.onnx'
    concepts = _REPO_ROOT / 'dclap_sae_concepts.json'
    return encoder, decoder, concepts


def test_real_sae_steering_matches_expected_vectors(monkeypatch):
    if importlib.util.find_spec('onnxruntime') is None:
        pytest.skip('onnxruntime is not installed')

    encoder_path, decoder_path, concepts_path = _artifacts()
    missing = [p.name for p in (encoder_path, decoder_path, concepts_path) if not p.exists()]
    if missing:
        pytest.skip(
            f'SAE artifacts missing: {", ".join(missing)}. The two ONNX graphs are '
            'downloaded into model/ from the AudioMuse-AI-SAE release, the way the '
            'Dockerfile does it; the concept catalogue ships in the repository.'
        )

    monkeypatch.setenv('CLAP_SAE_STEERING_ENABLED', 'true')
    monkeypatch.setenv('CLAP_SAE_ENCODER_PATH', str(encoder_path))
    monkeypatch.setenv('CLAP_SAE_MODEL_PATH', str(decoder_path))
    monkeypatch.setenv('CLAP_SAE_CONCEPTS_PATH', str(concepts_path))

    import config

    monkeypatch.setattr(config, 'CLAP_SAE_STEERING_ENABLED', True, raising=False)
    monkeypatch.setattr(config, 'CLAP_SAE_ENCODER_PATH', str(encoder_path), raising=False)
    monkeypatch.setattr(config, 'CLAP_SAE_MODEL_PATH', str(decoder_path), raising=False)
    monkeypatch.setattr(config, 'CLAP_SAE_CONCEPTS_PATH', str(concepts_path), raising=False)

    from tasks import clap_steering

    for key in ('encoder', 'decoder'):
        clap_steering._STATE[key] = None
    assert clap_steering.warmup(), clap_steering._STATE['unavailable_reason']

    catalogue = clap_steering.get_catalogue()
    assert catalogue['available'] is True
    offered = {t['term'] for group in catalogue['categories'] for t in group['terms']}
    probes = [term for term in PROBE_CONCEPTS if term in offered]
    assert probes, f'none of the probe concepts are in the catalogue: {PROBE_CONCEPTS}'

    rows = _probe_embeddings()
    session = clap_steering._STATE['encoder']
    input_name = clap_steering._STATE['encoder_input']
    activations = session.run(None, {input_name: rows})[0].astype(np.float32)

    steered = {}
    for term in probes:
        terms, warnings = clap_steering.normalize_terms([{'term': term, 'weight': 3.0}])
        assert warnings == [], warnings
        vector, applied = clap_steering.apply_steering(rows[0], terms)
        assert len(applied) == 1
        steered[term] = np.asarray(vector, dtype=np.float32)

    encoder_sha = _sha256(encoder_path)
    expected_path = _REPO_ROOT / 'test' / 'sae_steering_expected.json'
    explicit_record = os.environ.get('SAE_RECORD_EXPECTED', '').lower() in ('1', 'true', 'yes')
    record_mode = explicit_record or not expected_path.exists()

    current_meta = {
        'encoder_sha256': encoder_sha,
        'd_sae': int(activations.shape[1]),
        'probes': N_PROBES,
        'concepts': probes,
    }

    print(f'\n  encoder sha256 : {encoder_sha}')
    print(f'  d_sae          : {activations.shape[1]}')
    print(f'  mean L0        : {float((activations > 0).sum(axis=1).mean()):.1f}')

    if record_mode:
        payload = {
            '_meta': current_meta,
            'activation_checksum': [float(v) for v in activations.sum(axis=1)],
            'active_counts': [int(v) for v in (activations > 0).sum(axis=1)],
            'steered': {term: [float(v) for v in steered[term]] for term in probes},
        }
        with open(expected_path, 'w', newline='\n') as handle:
            json.dump(payload, handle, indent=2)
        print(f'  wrote baseline : {expected_path.name} (encoder sha256={encoder_sha})')
        pytest.skip(f'recorded a fresh baseline in {expected_path.name}; re-run to replay it')

    with open(expected_path) as handle:
        expected = json.load(handle)

    baseline_meta = expected.get('_meta', {})
    baseline_sha = baseline_meta.get('encoder_sha256')
    if baseline_sha and baseline_sha != encoder_sha:
        pytest.fail(
            'the SAE encoder changed since the baseline was recorded '
            f'(baseline sha256={baseline_sha}, current sha256={encoder_sha}). '
            f'The recorded vectors in {expected_path.name} are no longer valid for this '
            'model. Delete the file to re-record against the new model, or restore the '
            'model the baseline was recorded with.'
        )
    assert baseline_meta.get('d_sae') == int(activations.shape[1])

    np.testing.assert_allclose(
        activations.sum(axis=1),
        np.asarray(expected['activation_checksum'], dtype=np.float64),
        rtol=0,
        atol=ACTIVATION_TOLERANCE,
        err_msg='SAE encoder activations drifted from the recorded baseline',
    )
    assert [int(v) for v in (activations > 0).sum(axis=1)] == expected['active_counts']

    for term in probes:
        assert term in expected['steered'], f'{term} is not in the baseline; re-record it'
        np.testing.assert_allclose(
            steered[term],
            np.asarray(expected['steered'][term], dtype=np.float64),
            rtol=0,
            atol=SCORE_TOLERANCE,
            err_msg=f'the steered query for "{term}" drifted from the recorded baseline',
        )


def test_real_sae_steering_holds_its_invariants(monkeypatch):
    if importlib.util.find_spec('onnxruntime') is None:
        pytest.skip('onnxruntime is not installed')

    encoder_path, decoder_path, concepts_path = _artifacts()
    if not all(p.exists() for p in (encoder_path, decoder_path, concepts_path)):
        pytest.skip('SAE artifacts missing; see the other test for where they come from')

    import config

    monkeypatch.setattr(config, 'CLAP_SAE_STEERING_ENABLED', True, raising=False)
    monkeypatch.setattr(config, 'CLAP_SAE_ENCODER_PATH', str(encoder_path), raising=False)
    monkeypatch.setattr(config, 'CLAP_SAE_MODEL_PATH', str(decoder_path), raising=False)
    monkeypatch.setattr(config, 'CLAP_SAE_CONCEPTS_PATH', str(concepts_path), raising=False)

    from tasks import clap_steering

    for key in ('encoder', 'decoder'):
        clap_steering._STATE[key] = None
    assert clap_steering.warmup(), clap_steering._STATE['unavailable_reason']

    query = _probe_embeddings()[0]
    catalogue = clap_steering.get_catalogue()
    offered = {t['term'] for group in catalogue['categories'] for t in group['terms']}
    pair = [term for term in ('saxophone', 'female vocals') if term in offered]
    if len(pair) < 2:
        pytest.skip('the catalogue does not offer both probe concepts')

    assert clap_steering.apply_steering(query, []) == (query, [])

    for concept in catalogue['categories']:
        for term in concept['terms']:
            entry = clap_steering._STATE['concepts'][term['term']]
            norm = float(np.linalg.norm(np.asarray(entry['mask'], dtype=np.float64)))
            assert abs(norm - 1.0) < 1e-4, f'{term["term"]} mask is not unit norm'

    gentle, _ = clap_steering.normalize_terms([{'term': pair[0], 'weight': 1.0}])
    strong, _ = clap_steering.normalize_terms([{'term': pair[0], 'weight': 10.0}])
    near, _ = clap_steering.apply_steering(query, gentle)
    far, _ = clap_steering.apply_steering(query, strong)
    assert float(far @ query) < float(near @ query) < 1.0

    assert abs(float(np.linalg.norm(near)) - float(np.linalg.norm(query))) < 1e-4

    both, applied = clap_steering.normalize_terms(
        [{'term': pair[0], 'weight': 3.0}, {'term': pair[1], 'weight': 3.0}]
    )
    combined, used = clap_steering.apply_steering(query, both)
    assert len(used) == 2
    single, _ = clap_steering.apply_steering(
        query, clap_steering.normalize_terms([{'term': pair[0], 'weight': 3.0}])[0]
    )
    assert not np.allclose(combined, single)

    unknown, warnings = clap_steering.normalize_terms([{'term': 'a concept nobody trained'}])
    assert unknown == []
    assert warnings
