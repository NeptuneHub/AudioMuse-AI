# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Concept steering request validation and conjunctive candidate ranking.

Covers the two halves of tasks/clap_steering.py that hold the contract with the
search API: what a client is allowed to ask for, and how the requested concepts
turn into an ordering. The ONNX encoder is replaced by a fake whose activations
are chosen by hand, so the ranking maths is checked against known answers rather
than against whatever the real dictionary happens to do.

Main Features:
* Rejects unknown, duplicated and malformed concepts, caps the list length and
  snaps a free-form weight onto the strength grid
* An empty or absent refinement returns no terms and never warms a graph, which
  is what keeps the plain text search untouched
* Steering moves the query and preserves its magnitude; "more" and "less" move it
  in opposite directions and a stronger setting moves it further
* Two concepts each contribute, and an encoder failure returns the query intact
"""

import os
import sys

import numpy as np
import pytest

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tasks import clap_steering


D_SAE = 8


class _FakeEncoder:
    def __init__(self, activations):
        self._activations = np.asarray(activations, dtype=np.float32)

    def get_inputs(self):
        raise AssertionError('the fake encoder is installed directly, never introspected')

    def run(self, _outputs, feed):
        rows = next(iter(feed.values()))
        return [self._activations[: rows.shape[0]]]


def _concept(term, category, support):
    weights = np.ones(len(support), dtype=np.float32)
    weights /= np.linalg.norm(weights)
    return {
        'term': term,
        'category': category,
        'label': category.title(),
        'grounding': 0.5,
        'support': list(support),
        'mask': [float(v) for v in weights],
    }


@pytest.fixture
def steering(monkeypatch):
    concepts = [
        _concept('saxophone', 'instrument', [0, 1]),
        _concept('female vocals', 'vocals', [2, 3]),
        _concept('techno', 'genre', [4]),
    ]
    catalogue = {
        'd_sae': D_SAE,
        'category_order': ['genre', 'instrument', 'vocals'],
        'concepts': concepts,
    }
    monkeypatch.setattr(clap_steering, '_warmup_locked', lambda: True)
    monkeypatch.setitem(clap_steering._STATE, 'catalogue', catalogue)
    monkeypatch.setitem(clap_steering._STATE, 'concepts', {c['term']: c for c in concepts})
    monkeypatch.setitem(clap_steering._STATE, 'd_sae', D_SAE)
    monkeypatch.setitem(clap_steering._STATE, 'encoder_input', 'embedding')
    return clap_steering


class _LinearSAE:
    """encode: project 512 -> 8 ; decode: project 8 -> 512, both fixed and linear."""

    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        self.enc = rng.normal(size=(512, D_SAE)).astype(np.float32)
        self.dec = rng.normal(size=(D_SAE, 512)).astype(np.float32)

    def encode(self, rows):
        return np.maximum(0.0, rows @ self.enc)

    def decode(self, code):
        return code @ self.dec


class _FakeSession:
    def __init__(self, fn):
        self._fn = fn

    def run(self, _outputs, feed):
        return [self._fn(next(iter(feed.values())).astype(np.float32))]


def _install_decoder(monkeypatch):
    sae = _LinearSAE()
    monkeypatch.setitem(clap_steering._STATE, 'encoder', _FakeSession(sae.encode))
    monkeypatch.setitem(clap_steering._STATE, 'decoder', _FakeSession(sae.decode))
    monkeypatch.setitem(clap_steering._STATE, 'decoder_input', 'latents')


def _install_encoder(monkeypatch, activations):
    monkeypatch.setitem(clap_steering._STATE, 'encoder', _FakeEncoder(activations))


def test_absent_refinement_returns_no_terms_and_no_warnings(steering):
    assert steering.normalize_terms(None) == ([], [])
    assert steering.normalize_terms([]) == ([], [])


def test_absent_refinement_never_warms_a_graph(monkeypatch):
    monkeypatch.setattr(
        clap_steering,
        '_warmup_locked',
        lambda: pytest.fail('warmup must not run when no concept was requested'),
    )
    assert clap_steering.normalize_terms(None) == ([], [])
    query = np.ones(512, dtype=np.float32)
    assert clap_steering.apply_steering(query, []) == (query, [])


def test_unknown_concept_is_rejected_with_a_named_warning(steering):
    terms, warnings = steering.normalize_terms([{'term': 'bulgarian throat singing'}])
    assert terms == []
    assert 'bulgarian throat singing' in warnings[0]


def test_duplicate_concept_is_kept_once(steering):
    terms, warnings = steering.normalize_terms([{'term': 'techno'}, {'term': 'techno'}])
    assert [t['term'] for t in terms] == ['techno']
    assert any('more than once' in w for w in warnings)


def test_unknown_direction_is_rejected(steering):
    terms, warnings = steering.normalize_terms([{'term': 'techno', 'direction': 'sideways'}])
    assert terms == []
    assert 'more or less' in warnings[0]


def test_non_numeric_weight_is_rejected(steering):
    terms, warnings = steering.normalize_terms([{'term': 'techno', 'weight': 'loud'}])
    assert terms == []
    assert 'non numeric weight' in warnings[0]


def test_a_bare_string_is_accepted_as_a_concept(steering):
    terms, warnings = steering.normalize_terms(['techno'])
    assert [t['term'] for t in terms] == ['techno']
    assert warnings == []


def test_weight_snaps_onto_the_strength_grid(steering):
    from config import CLAP_SAE_ALPHA_STEPS

    terms, _ = steering.normalize_terms([{'term': 'techno', 'weight': 3.4}])
    assert terms[0]['weight'] in CLAP_SAE_ALPHA_STEPS
    assert terms[0]['weight'] == 3.0


def test_concept_list_is_capped_and_the_cap_is_reported(steering):
    from config import CLAP_SAE_MAX_TERMS

    requested = [{'term': 'techno'}] + [{'term': f'unknown-{i}'} for i in range(20)]
    terms, warnings = steering.normalize_terms(requested)
    assert len(terms) <= CLAP_SAE_MAX_TERMS
    assert warnings


def test_catalogue_groups_terms_in_the_declared_category_order(steering):
    catalogue = steering.get_catalogue()
    assert catalogue['available'] is True
    assert [c['category'] for c in catalogue['categories']] == ['genre', 'instrument', 'vocals']
    assert catalogue['max_terms'] >= 1


def test_catalogue_never_exposes_track_names(steering):
    catalogue = steering.get_catalogue()
    for group in catalogue['categories']:
        for term in group['terms']:
            assert set(term) == {'term', 'grounding'}


def test_no_concept_leaves_the_query_untouched(steering, monkeypatch):
    _install_decoder(monkeypatch)
    query = np.arange(512, dtype=np.float32) / 512.0
    out, applied = steering.apply_steering(query, [])
    assert applied == []
    assert out is query


def test_amplifying_moves_the_query(steering, monkeypatch):
    _install_decoder(monkeypatch)
    query = np.ones(512, dtype=np.float32) / np.sqrt(512)
    terms, _ = steering.normalize_terms([{'term': 'saxophone', 'weight': 1.0}])
    out, applied = steering.apply_steering(query, terms)

    assert len(applied) == 1
    assert out is not query
    assert abs(float(np.linalg.norm(out)) - float(np.linalg.norm(query))) < 1e-5
    assert float(out @ query) < 0.9999


def test_more_and_less_do_not_produce_the_same_query(steering, monkeypatch):
    _install_decoder(monkeypatch)
    query = np.ones(512, dtype=np.float32) / np.sqrt(512)
    more, _ = steering.normalize_terms([{'term': 'saxophone', 'direction': 'more'}])
    less, _ = steering.normalize_terms([{'term': 'saxophone', 'direction': 'less'}])
    up, _ = steering.apply_steering(query, more)
    down, _ = steering.apply_steering(query, less)

    assert not np.allclose(up, down)


def test_suppression_never_drives_a_latent_below_zero(steering, monkeypatch):
    seen = {}

    class _Recorder:
        def run(self, _outputs, feed):
            code = next(iter(feed.values())).astype(np.float32)
            seen['code'] = code
            return [code @ np.ones((D_SAE, 512), dtype=np.float32)]

    monkeypatch.setitem(clap_steering._STATE, 'encoder',
                        _FakeSession(lambda rows: np.zeros((rows.shape[0], D_SAE), np.float32)))
    monkeypatch.setitem(clap_steering._STATE, 'decoder', _Recorder())
    monkeypatch.setitem(clap_steering._STATE, 'decoder_input', 'latents')

    terms, _ = steering.normalize_terms(
        [{'term': 'saxophone', 'weight': 10.0, 'direction': 'less'}]
    )
    steering.apply_steering(np.ones(512, dtype=np.float32), terms)

    assert float(seen['code'].min()) >= 0.0


def test_a_stronger_setting_moves_the_query_further(steering, monkeypatch):
    _install_decoder(monkeypatch)
    query = np.ones(512, dtype=np.float32) / np.sqrt(512)
    gentle, _ = steering.normalize_terms([{'term': 'saxophone', 'weight': 1.0}])
    strong, _ = steering.normalize_terms([{'term': 'saxophone', 'weight': 10.0}])
    near, _ = steering.apply_steering(query, gentle)
    far, _ = steering.apply_steering(query, strong)

    assert float(far @ query) < float(near @ query)


def test_two_concepts_both_contribute(steering, monkeypatch):
    _install_decoder(monkeypatch)
    query = np.ones(512, dtype=np.float32) / np.sqrt(512)
    one, _ = steering.normalize_terms([{'term': 'saxophone', 'weight': 3.0}])
    both, _ = steering.normalize_terms(
        [{'term': 'saxophone', 'weight': 3.0}, {'term': 'female vocals', 'weight': 3.0}]
    )
    single, _ = steering.apply_steering(query, one)
    pair, applied = steering.apply_steering(query, both)

    assert len(applied) == 2
    assert not np.allclose(single, pair)


def test_steering_survives_an_encoder_failure(steering, monkeypatch):
    class _Broken:
        def run(self, *_args, **_kwargs):
            raise RuntimeError('onnx exploded')

    _install_decoder(monkeypatch)
    monkeypatch.setitem(clap_steering._STATE, 'encoder', _Broken())
    query = np.ones(512, dtype=np.float32)
    terms, _ = steering.normalize_terms([{'term': 'techno'}])
    out, applied = steering.apply_steering(query, terms)
    assert out is query
    assert applied == []
