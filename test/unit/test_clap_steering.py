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
* Ranking is a conjunction: the weakest concept decides a track's score, so a
  track missing one concept cannot be rescued by excelling at another
* A "less" direction inverts a concept's contribution, and a lower strength
  pulls a concept's influence toward neutral
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
    return {
        'term': term,
        'category': category,
        'label': category.title(),
        'grounding': 0.5,
        'support': list(support),
        'unit_activation': [1.0 for _ in support],
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
    assert clap_steering.rank_candidates(np.zeros((3, D_SAE), dtype=np.float32), []) == (None, [])


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

    terms, _ = steering.normalize_terms([{'term': 'techno', 'weight': 0.37}])
    assert terms[0]['weight'] in CLAP_SAE_ALPHA_STEPS
    assert terms[0]['weight'] == 0.5


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


def test_ranking_is_a_conjunction_not_an_average(steering, monkeypatch):
    # Every row sums to 20, so a concept's share is proportional to its raw sum
    # and the percentile each row lands on is fixed by construction:
    #   saxophone shares ascending -> row2 row3 row1 row4 row0  -> .00 .25 .50 .75 1.0
    #   female shares   ascending -> row2 row0 row1 row3 row4  -> .00 .25 .50 .75 1.0
    # row0 is lopsided (1.00 / 0.25), row1 is balanced (0.50 / 0.50).
    # A conjunction scores row1 above row0; an average scores row0 above row1.
    activations = np.array(
        [
            [5.0, 5.0, 1.0, 1.0, 0.0, 8.0, 0.0, 0.0],
            [3.0, 3.0, 2.0, 2.0, 0.0, 10.0, 0.0, 0.0],
            [0.5, 0.5, 0.5, 0.5, 0.0, 18.0, 0.0, 0.0],
            [1.5, 1.5, 3.0, 3.0, 0.0, 11.0, 0.0, 0.0],
            [4.0, 4.0, 4.5, 4.5, 0.0, 3.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    _install_encoder(monkeypatch, activations)
    terms, _ = steering.normalize_terms(
        [{'term': 'saxophone', 'weight': 1.0}, {'term': 'female vocals', 'weight': 1.0}]
    )
    scores, applied = steering.rank_candidates(np.zeros((5, 512), dtype=np.float32), terms)

    assert len(applied) == 2
    assert scores[1] > scores[0]
    assert scores.argmax() == 4
    assert scores.argmin() == 2


def test_a_concept_is_scored_as_a_share_not_as_a_raw_sum(steering, monkeypatch):
    # row0 is a globally loud track: the largest raw saxophone sum, but it is a
    # fifth of everything else it activates. row1 is quiet and mostly saxophone.
    # Ranking on raw sums puts row0 first; ranking on share puts row1 first.
    activations = np.array(
        [
            [10.0, 10.0, 0.0, 0.0, 0.0, 80.0, 0.0, 0.0],
            [3.0, 3.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    _install_encoder(monkeypatch, activations)
    terms, _ = steering.normalize_terms([{'term': 'saxophone', 'weight': 1.0}])
    scores, _ = steering.rank_candidates(np.zeros((2, 512), dtype=np.float32), terms)

    assert scores.argmax() == 1


def test_less_direction_inverts_a_concept(steering, monkeypatch):
    activations = np.array(
        [
            [9.0, 9.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    _install_encoder(monkeypatch, activations)
    rows = np.zeros((2, 512), dtype=np.float32)

    more, _ = steering.normalize_terms([{'term': 'saxophone', 'direction': 'more'}])
    less, _ = steering.normalize_terms([{'term': 'saxophone', 'direction': 'less'}])
    high, _ = steering.rank_candidates(rows, more)
    low, _ = steering.rank_candidates(rows, less)

    assert high.argmax() == 0
    assert low.argmax() == 1


def test_lower_strength_moves_a_concept_toward_neutral(steering, monkeypatch):
    activations = np.array(
        [
            [9.0, 9.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    _install_encoder(monkeypatch, activations)
    rows = np.zeros((2, 512), dtype=np.float32)

    strong, _ = steering.normalize_terms([{'term': 'saxophone', 'weight': 1.0}])
    gentle, _ = steering.normalize_terms([{'term': 'saxophone', 'weight': 0.1}])
    strong_scores, _ = steering.rank_candidates(rows, strong)
    gentle_scores, _ = steering.rank_candidates(rows, gentle)

    assert strong_scores.min() < gentle_scores.min()


def test_ranking_survives_an_encoder_failure(steering, monkeypatch):
    class _Broken:
        def run(self, *_args, **_kwargs):
            raise RuntimeError('onnx exploded')

    monkeypatch.setitem(clap_steering._STATE, 'encoder', _Broken())
    terms, _ = steering.normalize_terms([{'term': 'techno'}])
    scores, applied = steering.rank_candidates(np.zeros((2, 512), dtype=np.float32), terms)
    assert scores is None
    assert applied == []


def test_ranking_ignores_an_empty_candidate_set(steering, monkeypatch):
    _install_encoder(monkeypatch, np.zeros((1, D_SAE), dtype=np.float32))
    terms, _ = steering.normalize_terms([{'term': 'techno'}])
    assert steering.rank_candidates(np.zeros((0, 512), dtype=np.float32), terms) == (None, [])
    assert steering.rank_candidates(None, terms) == (None, [])
