# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Tests for confidence-aware automatic-playlist naming context.

Main Features:
* Only decisive per-track and per-cluster axis votes reach the naming context;
  low coverage or tied votes disable the axes
* build_naming_context maps grounded evidence to one editorial naming dimension
  (bittersweet, contrast, theme, mood, or listener function)
* Instrumental detection and fallbacks stay grounded when lyric axes are absent
"""

import numpy as np

from config import LYRICS_INSTRUMENTAL_AXIS_FILL
from tasks.ai.playlist_namer import build_naming_context, confident_axis_labels


COLUMNS = [
    ('AXIS_1_SETTING', 'URBAN'),
    ('AXIS_1_SETTING', 'INTERIOR'),
    ('AXIS_2_SOCIAL_DYNAMIC', 'SOLITARY'),
    ('AXIS_2_SOCIAL_DYNAMIC', 'ROMANTIC'),
    ('AXIS_3_EMOTIONAL_VALENCE', 'RADIANT'),
    ('AXIS_3_EMOTIONAL_VALENCE', 'MELANCHOLIC'),
    ('AXIS_4_NARRATIVE_TEMPORALITY', 'CHRONICLE'),
    ('AXIS_4_NARRATIVE_TEMPORALITY', 'STORYTELLING'),
    ('AXIS_5_THEMATIC_WEIGHT', 'MORTAL'),
    ('AXIS_5_THEMATIC_WEIGHT', 'SENSORIAL'),
]


def _axis_blob(
    setting=(0.1, 0.8),
    social=(0.8, 0.1),
    valence=(0.1, 0.8),
    temporality=(0.51, 0.49),
    weight=(0.8, 0.1),
):
    return np.array(
        setting + social + valence + temporality + weight,
        dtype=np.float32,
    ).tobytes()


def test_only_decisive_axis_votes_are_returned():
    labels = confident_axis_labels([_axis_blob()] * 10, 10, COLUMNS)

    assert labels == {
        'AXIS_1_SETTING': 'INTERIOR',
        'AXIS_2_SOCIAL_DYNAMIC': 'SOLITARY',
        'AXIS_3_EMOTIONAL_VALENCE': 'MELANCHOLIC',
        'AXIS_5_THEMATIC_WEIGHT': 'MORTAL',
    }
    assert 'AXIS_4_NARRATIVE_TEMPORALITY' not in labels


def test_low_playlist_lyrics_coverage_disables_all_axes():
    assert confident_axis_labels([_axis_blob()] * 4, 10, COLUMNS) == {}


def test_instrumental_axis_sentinel_counts_as_no_lyrics():
    sentinel = np.full(
        len(COLUMNS), LYRICS_INSTRUMENTAL_AXIS_FILL, dtype=np.float32
    ).tobytes()

    assert confident_axis_labels([sentinel] * 10, 10, COLUMNS) == {}


def test_a_tied_cluster_vote_is_not_treated_as_a_theme():
    blobs = [
        _axis_blob(social=(0.8, 0.1)) for _ in range(5)
    ] + [
        _axis_blob(social=(0.1, 0.8)) for _ in range(5)
    ]

    labels = confident_axis_labels(blobs, 10, COLUMNS)

    assert 'AXIS_2_SOCIAL_DYNAMIC' not in labels


def test_lyric_setting_is_diagnostic_but_never_used_as_a_listening_context():
    blobs = [
        _axis_blob(
            setting=(0.1, 0.8),
            social=(0.5, 0.5),
            valence=(0.5, 0.5),
            temporality=(0.5, 0.5),
            weight=(0.5, 0.5),
        )
        for _ in range(10)
    ]
    rows = [{'mood_vector': 'indie:0.8', 'other_features': ''} for _ in range(10)]

    context = build_naming_context(rows, {}, blobs, 10, COLUMNS)

    assert context['axis_labels'] == {'AXIS_1_SETTING': 'INTERIOR'}
    assert context['naming_brief'] == 'general-purpose listening'
    assert context['naming_dimension'] == 'mood'
    assert context['naming_evidence'] == 'general-purpose listening'
    assert context['fallback_name'] == 'Indie'


def test_bright_sound_plus_melancholic_lyrics_becomes_bittersweet():
    rows = [
        {
            'mood_vector': 'indie:0.8,rock:0.4,instrumental:0.1',
            'other_features': 'party:0.8,happy:0.7,relaxed:0.2',
        }
        for _ in range(10)
    ]

    context = build_naming_context(rows, {}, [_axis_blob()] * 10, 10, COLUMNS)

    assert context['genre'] == 'Indie'
    assert context['ideas'] == ['bittersweet', 'solitude', 'serious']
    assert context['naming_brief'] == (
        'melancholic lyrics over upbeat, energetic music; '
        'introspective, solitary lyrics; serious themes about human struggle'
    )
    assert context['fallback_name'] == 'Bittersweet Indie'
    assert context['naming_dimension'] == 'contrast'
    assert context['naming_evidence'] == (
        'melancholic lyrics contrasted with upbeat energetic music'
    )


def test_primary_genre_overrides_the_centroid_genre_for_naming():
    rows = [
        {
            'mood_vector': 'pop:0.9,rnb:0.7',
            'other_features': '',
        }
        for _ in range(10)
    ]

    context = build_naming_context(
        rows,
        {'pop': 0.9, 'rnb': 0.7},
        [],
        10,
        COLUMNS,
        primary_genre='rnb',
    )

    assert context['genre'] == 'R&B'
    assert context['fallback_name'] == 'R&B'


def test_diversified_context_can_choose_the_grounded_party_focus(monkeypatch):
    from tasks.ai import playlist_namer

    monkeypatch.setattr(playlist_namer.secrets, 'choice', lambda choices: choices[-1])
    rows = [
        {
            'mood_vector': 'pop:0.9',
            'other_features': 'party:0.9,happy:0.8',
        }
        for _ in range(10)
    ]
    blobs = [
        _axis_blob(
            social=(0.1, 0.8),
            valence=(0.1, 0.8),
            weight=(0.1, 0.8),
        )
        for _ in range(10)
    ]

    context = build_naming_context(
        rows,
        {},
        blobs,
        10,
        COLUMNS,
        primary_genre='pop',
        diversify=True,
    )

    assert context['naming_dimension'] == 'function'
    assert context['naming_evidence'] == 'energetic party music'


def test_diversify_never_downgrades_to_general_purpose_when_grounded_evidence_exists(
    monkeypatch,
):
    from tasks.ai import playlist_namer

    captured = []

    def choose_last(choices):
        captured.append(list(choices))
        return choices[-1]

    monkeypatch.setattr(playlist_namer.secrets, 'choice', choose_last)

    dimension, evidence = playlist_namer._naming_target(
        {'AXIS_5_THEMATIC_WEIGHT': 'SENSORIAL'}, None, False, diversify=True
    )

    assert (dimension, evidence) == ('function', 'dance-focused themes')
    assert ('mood', 'general-purpose listening') not in captured[0]


def test_fallback_candidates_rotate_ideas_then_synonyms():
    from tasks.ai.playlist_namer import fallback_candidates

    assert fallback_candidates('Pop', ['bittersweet'], False) == [
        'Bittersweet Pop', 'Wistful Pop', 'Nostalgic Pop'
    ]
    assert fallback_candidates('Rock', ['calm', 'solitude'], False) == [
        'Chill Rock', 'Solo Rock', 'Mellow Rock', 'Serene Rock'
    ]
    assert fallback_candidates('Ambient', [], True) == ['Ambient Instrumentals']


def test_naming_context_exposes_the_fallback_candidates():
    rows = [
        {'mood_vector': 'pop:0.9', 'other_features': ''}
        for _ in range(10)
    ]

    context = build_naming_context(rows, {}, [], 10, COLUMNS, primary_genre='pop')

    assert context['fallback_candidates'][0] == context['fallback_name']


def test_instrumental_playlist_gets_a_safe_fallback_without_lyrics_axes():
    rows = [
        {
            'mood_vector': 'ambient:0.7,electronic:0.5,instrumental:0.8',
            'other_features': 'relaxed:0.9,danceable:0.2',
        }
        for _ in range(10)
    ]

    context = build_naming_context(rows, {}, [], 10, COLUMNS)

    assert context['axis_labels'] == {}
    assert context['ideas'] == ['calm', 'instrumental']
    assert context['naming_brief'] == 'calm, relaxed instrumental music'
    assert context['instrumental'] is True
    assert context['fallback_name'] == 'Chill Ambient Instrumentals'
    assert context['naming_dimension'] == 'function'
    assert context['naming_evidence'] == 'calm relaxed instrumental listening'


def test_romantic_lyrics_choose_a_theme_instead_of_a_generic_mood():
    rows = [
        {
            'mood_vector': 'soul:0.8',
            'other_features': 'happy:0.8,party:0.7',
        }
        for _ in range(10)
    ]
    blobs = [
        _axis_blob(
            social=(0.1, 0.8),
            valence=(0.1, 0.8),
            weight=(0.5, 0.5),
        )
        for _ in range(10)
    ]

    context = build_naming_context(rows, {}, blobs, 10, COLUMNS)

    assert context['naming_dimension'] == 'relationship'
    assert context['naming_evidence'] == (
        'melancholic lyrics about sadness and longing; romantic lyrics'
    )


def test_melancholic_lyrics_and_bright_sound_choose_emotional_contrast():
    rows = [
        {
            'mood_vector': 'rock:0.8',
            'other_features': 'happy:0.8,party:0.7',
        }
        for _ in range(10)
    ]
    blobs = [
        _axis_blob(
            social=(0.5, 0.5),
            valence=(0.1, 0.8),
            weight=(0.5, 0.5),
        )
        for _ in range(10)
    ]

    context = build_naming_context(rows, {}, blobs, 10, COLUMNS)

    assert context['naming_dimension'] == 'contrast'
    assert context['naming_evidence'] == (
        'melancholic lyrics contrasted with upbeat energetic music'
    )


def test_sensorial_lyrics_choose_a_listener_function():
    rows = [
        {
            'mood_vector': 'electronic:0.8',
            'other_features': 'danceable:0.9',
        }
        for _ in range(10)
    ]
    blobs = [
        _axis_blob(
            social=(0.5, 0.5),
            valence=(0.5, 0.5),
            weight=(0.1, 0.8),
        )
        for _ in range(10)
    ]

    context = build_naming_context(rows, {}, blobs, 10, COLUMNS)

    assert context['naming_dimension'] == 'function'
    assert context['naming_evidence'] == (
        'dance-focused themes; energetic, danceable music'
    )


def test_mortal_theme_with_valence_uses_the_more_specific_mood():
    rows = [
        {'mood_vector': 'acoustic:0.8', 'other_features': 'sad:0.8'}
        for _ in range(10)
    ]

    context = build_naming_context(rows, {}, [_axis_blob()] * 10, 10, COLUMNS)

    assert context['naming_dimension'] == 'mood'
    assert context['naming_evidence'] == (
        'melancholic lyrics about sadness and longing; somber music'
    )


def test_mortal_theme_without_valence_remains_a_plain_lyrical_topic():
    rows = [
        {'mood_vector': 'metal:0.8', 'other_features': 'aggressive:0.8'}
        for _ in range(10)
    ]
    blobs = [
        _axis_blob(valence=(0.5, 0.5))
        for _ in range(10)
    ]

    context = build_naming_context(rows, {}, blobs, 10, COLUMNS)

    assert context['naming_dimension'] == 'theme'
    assert context['naming_evidence'] == (
        'lyrics about life-or-death struggles and perseverance; '
        'introspective, solitary lyrics'
    )


def test_chronicle_axis_does_not_invent_a_lyrical_topic():
    rows = [{'mood_vector': 'country:0.8', 'other_features': ''} for _ in range(10)]
    blobs = [
        _axis_blob(
            setting=(0.5, 0.5),
            social=(0.5, 0.5),
            valence=(0.5, 0.5),
            temporality=(0.8, 0.1),
            weight=(0.5, 0.5),
        )
        for _ in range(10)
    ]

    context = build_naming_context(rows, {}, blobs, 10, COLUMNS)

    assert context['axis_labels'] == {'AXIS_4_NARRATIVE_TEMPORALITY': 'CHRONICLE'}
    assert context['naming_dimension'] == 'mood'
    assert context['naming_evidence'] == 'general-purpose listening'
    assert context['fallback_name'] == 'Country'


def test_storytelling_axis_does_not_invent_the_story_subject():
    rows = [{'mood_vector': 'folk:0.8', 'other_features': ''} for _ in range(10)]
    blobs = [
        _axis_blob(
            setting=(0.5, 0.5),
            social=(0.5, 0.5),
            valence=(0.5, 0.5),
            temporality=(0.1, 0.8),
            weight=(0.5, 0.5),
        )
        for _ in range(10)
    ]

    context = build_naming_context(rows, {}, blobs, 10, COLUMNS)

    assert context['axis_labels'] == {
        'AXIS_4_NARRATIVE_TEMPORALITY': 'STORYTELLING'
    }
    assert context['naming_evidence'] == 'general-purpose listening'
    assert context['fallback_name'] == 'Folk'


def test_axis_sentinels_do_not_fake_lyrics_coverage_for_ambient_tracks():
    rows = [
        {'mood_vector': 'ambient:0.7', 'other_features': 'relaxed:0.9'}
        for _ in range(10)
    ]
    sentinel = np.full(
        len(COLUMNS), LYRICS_INSTRUMENTAL_AXIS_FILL, dtype=np.float32
    ).tobytes()

    context = build_naming_context(rows, {}, [sentinel] * 10, 10, COLUMNS)

    assert context['axis_labels'] == {}
    assert context['fallback_name'] == 'Chill Ambient Instrumentals'


def test_a_rare_instrumental_tag_does_not_make_the_whole_cluster_instrumental():
    rows = [
        {'mood_vector': 'funk:0.8,instrumental:0.9', 'other_features': 'party:0.8'}
    ] + [
        {'mood_vector': 'funk:0.8', 'other_features': 'party:0.8'}
        for _ in range(9)
    ]

    context = build_naming_context(rows, {}, [_axis_blob()] * 10, 10, COLUMNS)

    assert 'instrumental' not in context['ideas']
    assert not context['fallback_name'].endswith('Instrumentals')


def test_ambient_score_does_not_override_full_lyrics_coverage():
    rows = [
        {'mood_vector': 'ambient:0.8,instrumental:0.1', 'other_features': 'relaxed:0.8'}
        for _ in range(10)
    ]

    context = build_naming_context(rows, {}, [_axis_blob()] * 10, 10, COLUMNS)

    assert 'instrumental' not in context['ideas']
