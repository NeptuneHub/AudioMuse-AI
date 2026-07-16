# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only

"""Build compact, grounded inputs for automatic-playlist AI naming.

The clustering itself is musical, so lyric axes are used only when their
playlist-level vote is decisive.  Broad axis labels are converted to safe title
concepts instead of concrete scenes; this keeps small language models useful
without asking them to infer hospitals, characters, or stories that are not in
the data.
"""

from collections import Counter, defaultdict
from math import ceil
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from config import OTHER_FEATURE_LABELS, STRATIFIED_GENRES


MIN_LYRICS_COVERAGE = 0.50
MIN_TRACK_AXIS_MARGIN = 0.03
MIN_CONFIDENT_AXIS_VOTES = 5
MIN_AXIS_VOTE_COVERAGE = 0.35
MIN_AXIS_WIN_SHARE = 0.55
MIN_AXIS_LEAD = 0.15

GENRE_DISPLAY = {
    'rock': 'Rock', 'pop': 'Pop', 'alternative': 'Alternative', 'indie': 'Indie',
    'electronic': 'Electronic', 'jazz': 'Jazz', 'metal': 'Metal',
    'classic rock': 'Classic Rock', 'soul': 'Soul', 'indie rock': 'Indie Rock',
    'electronica': 'Electronica', 'folk': 'Folk', 'punk': 'Punk', 'blues': 'Blues',
    'hard rock': 'Hard Rock', 'ambient': 'Ambient', 'acoustic': 'Acoustic',
    'experimental': 'Experimental', 'Hip-Hop': 'Hip-Hop', 'country': 'Country',
    'funk': 'Funk', 'electro': 'Electro', 'heavy metal': 'Heavy Metal',
    'Progressive rock': 'Prog Rock', 'rnb': 'R&B', 'indie pop': 'Indie Pop',
    'House': 'House',
}

SOUND_IDEAS = {
    'danceable': 'energetic',
    'aggressive': 'intense',
    'happy': 'joyful',
    'party': 'energetic',
    'relaxed': 'calm',
    'sad': 'melancholy',
}

SOUND_BRIEFS = {
    'danceable': 'energetic, danceable music',
    'aggressive': 'intense, forceful music',
    'happy': 'upbeat, joyful music',
    'party': 'energetic party music',
    'relaxed': 'calm, relaxed music',
    'sad': 'somber music',
}

VALENCE_BRIEFS = {
    'RADIANT': 'radiant, celebratory lyrics',
    'MELANCHOLIC': 'melancholic lyrics about sadness and longing',
    'VOLATILE': 'angry, restless lyrics',
    'VULNERABLE': 'vulnerable, emotionally tense lyrics',
    'SERENE': 'peaceful, emotionally still lyrics',
    'NUMB': 'emotionally detached lyrics',
}

SOCIAL_BRIEFS = {
    'SOLITARY': 'introspective, solitary lyrics',
    'ROMANTIC': 'romantic lyrics',
    'KINSHIP': 'lyrics about family bonds',
    'COLLECTIVE': 'lyrics addressed to a group or community',
    'ADVERSARIAL': 'defiant lyrics confronting an opponent',
    'DIVINE': 'spiritual lyrics addressed to a higher power',
}

THEME_BRIEFS = {
    'TRIVIAL': 'lighthearted themes',
    'MORTAL': 'serious themes about human struggle',
    'POLITICAL': 'political and social themes',
    'SENSORIAL': 'dance-focused themes',
}

NARRATIVE_BRIEFS = {
    'RETROSPECTIVE': 'lyrics looking back on memories',
    'CHRONICLE': 'lyrics describing events in the present',
    'EXISTENTIAL': 'philosophical, reflective lyrics',
    'STORYTELLING': 'storytelling lyrics',
    'DIRECT_PLEA': 'lyrics directly pleading with someone',
}

AXIS_IDEAS = {
    'AXIS_1_SETTING': {
        'URBAN': 'urban', 'WILDERNESS': 'wild', 'INTERIOR': 'indoor',
        'TRANSIT': 'travel', 'EXTRATERRESTRIAL': 'cosmic',
        'SURREAL_ABSTRACT': 'dreamlike',
    },
    'AXIS_2_SOCIAL_DYNAMIC': {
        'SOLITARY': 'solitude', 'ROMANTIC': 'romance', 'KINSHIP': 'family',
        'COLLECTIVE': 'togetherness', 'ADVERSARIAL': 'defiance',
        'DIVINE': 'spirituality',
    },
    'AXIS_3_EMOTIONAL_VALENCE': {
        'RADIANT': 'joyful', 'MELANCHOLIC': 'melancholy', 'VOLATILE': 'intense',
        'VULNERABLE': 'uneasy', 'SERENE': 'calm', 'NUMB': 'detached',
    },
    'AXIS_4_NARRATIVE_TEMPORALITY': {
        'RETROSPECTIVE': 'memories', 'CHRONICLE': 'present',
        'EXISTENTIAL': 'reflection', 'STORYTELLING': 'stories',
        'DIRECT_PLEA': 'longing',
    },
    'AXIS_5_THEMATIC_WEIGHT': {
        'TRIVIAL': 'lighthearted', 'MORTAL': 'serious',
        'POLITICAL': 'political', 'SENSORIAL': 'physical energy',
    },
}

_FALLBACK_ADJECTIVES = {
    'bittersweet': 'Bittersweet', 'joyful': 'Feel Good', 'melancholy': 'Sad',
    'intense': 'Intense', 'uneasy': 'Moody', 'calm': 'Chill',
    'detached': 'Low-Key', 'solitude': 'Solo', 'romance': 'Romantic',
    'family': 'Family', 'togetherness': 'Collective', 'defiance': 'Defiant',
    'spirituality': 'Spiritual', 'lighthearted': 'Lighthearted',
    'serious': 'Serious', 'political': 'Political',
    'physical energy': 'High Energy', 'energetic': 'High Energy',
    'urban': 'Urban', 'wild': 'Wild', 'indoor': 'Indoor', 'travel': 'Travel',
    'cosmic': 'Cosmic', 'dreamlike': 'Dreamlike', 'memories': 'Nostalgic',
    'present': 'Current', 'reflection': 'Reflective', 'stories': 'Storytelling',
    'longing': 'Longing', 'instrumental': 'Instrumental',
}


def _parse_pairs(text: Optional[str]) -> Dict[str, float]:
    parsed = {}
    for part in (text or '').split(','):
        key, separator, raw_value = part.partition(':')
        if not separator:
            continue
        try:
            parsed[key.strip()] = float(raw_value)
        except ValueError:
            continue
    return parsed


def _average_column(rows: Iterable[Dict], column: str) -> Dict[str, float]:
    rows = list(rows)
    if not rows:
        return {}
    sums = defaultdict(float)
    for row in rows:
        for key, value in _parse_pairs(row.get(column)).items():
            sums[key] += value
    return {key: sums[key] / len(rows) for key in sums}


def _with_centroid_fallback(
    averages: Dict[str, float], centroid_scores: Optional[Dict]
) -> Dict[str, float]:
    if averages or not centroid_scores:
        return averages
    return {
        key: float(value)
        for key, value in centroid_scores.items()
        if isinstance(value, (int, float))
    }


def pick_genre(mood_scores: Dict[str, float]) -> str:
    candidates = {key: mood_scores[key] for key in STRATIFIED_GENRES if key in mood_scores}
    if not candidates:
        return 'Pop'
    winner = max(candidates, key=candidates.get)
    return GENRE_DISPLAY.get(winner, winner.title())


def _pick_sound(other_scores: Dict[str, float]) -> Optional[str]:
    candidates = {key: other_scores[key] for key in OTHER_FEATURE_LABELS if key in other_scores}
    return max(candidates, key=candidates.get) if candidates else None


def confident_axis_labels(
    axis_blobs: Iterable[bytes],
    total_tracks: int,
    columns: Sequence[Tuple[str, str]],
) -> Dict[str, str]:
    """Return only axis winners supported by decisive per-track and cluster votes."""
    if not total_tracks or not columns:
        return {}
    expected_bytes = len(columns) * np.dtype(np.float32).itemsize
    vectors = [
        vector
        for blob in axis_blobs
        if blob and len(blob) == expected_bytes
        for vector in (np.frombuffer(blob, dtype=np.float32),)
        if float(vector.max() - vector.min()) > 1e-6
    ]
    if len(vectors) < ceil(total_tracks * MIN_LYRICS_COVERAGE):
        return {}

    winners = {}
    axis_names = dict.fromkeys(axis for axis, _label in columns)
    for axis_name in axis_names:
        indices = [index for index, (axis, _label) in enumerate(columns) if axis == axis_name]
        labels = [columns[index][1] for index in indices]
        votes = []
        for vector in vectors:
            scores = vector[indices]
            order = np.argsort(scores)
            if len(order) < 2 or scores[order[-1]] - scores[order[-2]] < MIN_TRACK_AXIS_MARGIN:
                continue
            votes.append(labels[int(order[-1])])

        required_votes = max(
            MIN_CONFIDENT_AXIS_VOTES,
            ceil(len(vectors) * MIN_AXIS_VOTE_COVERAGE),
        )
        if len(votes) < required_votes:
            continue
        counts = Counter(votes)
        ranked = counts.most_common(2)
        top_label, top_count = ranked[0]
        second_count = ranked[1][1] if len(ranked) > 1 else 0
        if top_count / len(votes) < MIN_AXIS_WIN_SHARE:
            continue
        if (top_count - second_count) / len(votes) < MIN_AXIS_LEAD:
            continue
        winners[axis_name] = top_label
    return winners


def _usable_axis_vector_count(
    axis_blobs: Iterable[bytes], columns: Sequence[Tuple[str, str]]
) -> int:
    expected_bytes = len(columns) * np.dtype(np.float32).itemsize
    count = 0
    for blob in axis_blobs:
        if not blob or len(blob) != expected_bytes:
            continue
        vector = np.frombuffer(blob, dtype=np.float32)
        if float(vector.max() - vector.min()) > 1e-6:
            count += 1
    return count


def _title_ideas(
    axis_labels: Dict[str, str],
    sound_key: Optional[str],
    instrumental: bool,
) -> List[str]:
    valence = axis_labels.get('AXIS_3_EMOTIONAL_VALENCE')
    bright_sound = sound_key in {'danceable', 'happy', 'party'}
    bittersweet = valence == 'MELANCHOLIC' and bright_sound

    ideas = ['bittersweet'] if bittersweet else []
    if not ideas and valence:
        ideas.append(AXIS_IDEAS['AXIS_3_EMOTIONAL_VALENCE'][valence])
    if sound_key and not bittersweet:
        ideas.append(SOUND_IDEAS[sound_key])

    for axis_name in ('AXIS_2_SOCIAL_DYNAMIC', 'AXIS_5_THEMATIC_WEIGHT'):
        label = axis_labels.get(axis_name)
        if label:
            ideas.append(AXIS_IDEAS[axis_name][label])
        if len(ideas) >= 3:
            break
    if len(ideas) < 2 and not any(
        axis_labels.get(axis)
        for axis in (
            'AXIS_3_EMOTIONAL_VALENCE',
            'AXIS_2_SOCIAL_DYNAMIC',
            'AXIS_5_THEMATIC_WEIGHT',
        )
    ):
        narrative = axis_labels.get('AXIS_4_NARRATIVE_TEMPORALITY')
        if narrative in {'RETROSPECTIVE', 'EXISTENTIAL'}:
            ideas.append(
                AXIS_IDEAS['AXIS_4_NARRATIVE_TEMPORALITY'][narrative]
            )
    if instrumental:
        ideas.append('instrumental')

    unique = []
    for idea in ideas:
        if idea not in unique:
            unique.append(idea)
    return unique[:3]


def _naming_brief(
    axis_labels: Dict[str, str],
    sound_key: Optional[str],
    instrumental: bool,
) -> str:
    """Combine reliable evidence into prose without suggesting title vocabulary."""
    parts = []
    valence = axis_labels.get('AXIS_3_EMOTIONAL_VALENCE')
    bright_sound = sound_key in {'danceable', 'happy', 'party'}
    if valence == 'MELANCHOLIC' and bright_sound:
        parts.append('melancholic lyrics over upbeat, energetic music')
    else:
        if valence:
            parts.append(VALENCE_BRIEFS[valence])
        if sound_key:
            parts.append(SOUND_BRIEFS[sound_key])

    social = axis_labels.get('AXIS_2_SOCIAL_DYNAMIC')
    if social:
        parts.append(SOCIAL_BRIEFS[social])

    theme = axis_labels.get('AXIS_5_THEMATIC_WEIGHT')
    if theme and not (theme == 'SENSORIAL' and social == 'ROMANTIC'):
        parts.append(THEME_BRIEFS[theme])

    if len(parts) < 2:
        narrative = axis_labels.get('AXIS_4_NARRATIVE_TEMPORALITY')
        if narrative:
            parts.append(NARRATIVE_BRIEFS[narrative])

    if instrumental:
        if parts:
            parts[0] = parts[0].replace(' music', ' instrumental music')
            if 'instrumental' not in parts[0]:
                parts.insert(0, 'instrumental music')
        else:
            parts.append('instrumental music')

    return '; '.join(parts[:3]) or 'general-purpose listening'


def _naming_target(
    axis_labels: Dict[str, str],
    sound_key: Optional[str],
    instrumental: bool,
) -> Tuple[str, str]:
    """Choose one editorial dimension before asking the model for any wording."""
    valence = axis_labels.get('AXIS_3_EMOTIONAL_VALENCE')
    social = axis_labels.get('AXIS_2_SOCIAL_DYNAMIC')
    theme = axis_labels.get('AXIS_5_THEMATIC_WEIGHT')
    narrative = axis_labels.get('AXIS_4_NARRATIVE_TEMPORALITY')
    bright_sound = sound_key in {'danceable', 'happy', 'party'}

    if social == 'ROMANTIC':
        evidence = []
        if valence:
            evidence.append(VALENCE_BRIEFS[valence])
        evidence.append('romantic lyrics')
        return 'relationship', '; '.join(evidence)

    if valence == 'MELANCHOLIC' and bright_sound:
        return 'contrast', 'melancholic lyrics contrasted with upbeat energetic music'

    if theme == 'POLITICAL':
        evidence = [THEME_BRIEFS[theme]]
        if social:
            evidence.append(SOCIAL_BRIEFS[social])
        if valence:
            evidence.append(VALENCE_BRIEFS[valence])
        return 'theme', '; '.join(evidence)

    if theme == 'MORTAL':
        if valence:
            evidence = [VALENCE_BRIEFS[valence]]
            if sound_key:
                evidence.append(SOUND_BRIEFS[sound_key])
            return 'mood', '; '.join(evidence)
        evidence = ['lyrics about life-or-death struggles and perseverance']
        if social:
            evidence.append(SOCIAL_BRIEFS[social])
        return 'theme', '; '.join(evidence)

    if theme == 'SENSORIAL':
        evidence = ['dance-focused themes']
        if sound_key:
            evidence.append(SOUND_BRIEFS[sound_key])
        return 'function', '; '.join(evidence)

    if instrumental and sound_key == 'relaxed':
        return 'function', 'calm relaxed instrumental listening'

    if sound_key in {'danceable', 'party'}:
        return 'function', SOUND_BRIEFS[sound_key]

    if social:
        evidence = [SOCIAL_BRIEFS[social]]
        if valence:
            evidence.append(VALENCE_BRIEFS[valence])
        return 'theme', '; '.join(evidence)

    if narrative in {'RETROSPECTIVE', 'EXISTENTIAL'}:
        return 'theme', NARRATIVE_BRIEFS[narrative]

    evidence = []
    if valence:
        evidence.append(VALENCE_BRIEFS[valence])
    if sound_key:
        evidence.append(SOUND_BRIEFS[sound_key])
    if instrumental:
        evidence.append('instrumental music')
    return 'mood', '; '.join(evidence) or 'general-purpose listening'


def _fallback_name(genre: str, ideas: Sequence[str], instrumental: bool) -> str:
    if not ideas:
        return f"{genre}{' Instrumentals' if instrumental else ''}"
    adjective = _FALLBACK_ADJECTIVES.get(ideas[0], ideas[0].title())
    suffix = ' Instrumentals' if instrumental else ''
    return f'{adjective} {genre}{suffix}'


def build_naming_context(
    score_rows: Sequence[Dict],
    centroid_scores: Optional[Dict],
    axis_blobs: Iterable[bytes],
    total_tracks: int,
    axis_columns: Sequence[Tuple[str, str]],
) -> Dict:
    axis_blobs = list(axis_blobs)
    mood_scores = _with_centroid_fallback(
        _average_column(score_rows, 'mood_vector'), centroid_scores
    )
    other_scores = _with_centroid_fallback(
        _average_column(score_rows, 'other_features'), centroid_scores
    )
    genre = pick_genre(mood_scores)
    sound_key = _pick_sound(other_scores)
    has_lyrics_coverage = _usable_axis_vector_count(
        axis_blobs, axis_columns
    ) >= ceil(total_tracks * MIN_LYRICS_COVERAGE)
    instrumental = mood_scores.get('instrumental', 0.0) > 0.5 or (
        not has_lyrics_coverage
        and (
            genre == 'Ambient'
            or mood_scores.get('instrumental', 0.0) > 0.25
        )
    )
    axis_labels = confident_axis_labels(axis_blobs, total_tracks, axis_columns)
    ideas = _title_ideas(axis_labels, sound_key, instrumental)
    naming_dimension, naming_evidence = _naming_target(
        axis_labels, sound_key, instrumental
    )
    return {
        'genre': genre,
        'ideas': ideas,
        'naming_brief': _naming_brief(axis_labels, sound_key, instrumental),
        'naming_dimension': naming_dimension,
        'naming_evidence': naming_evidence,
        'instrumental': instrumental,
        'axis_labels': axis_labels,
        'fallback_name': _fallback_name(genre, ideas, instrumental),
    }
