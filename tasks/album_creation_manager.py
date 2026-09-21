# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Assemble a CD-format album from one seed and sequence it like a real album.

app_album_creation only parses the request; the cron entry point
run_album_of_the_week_task runs per server through tasks.task_run. The page and
API need LYRICS_ENABLED and CLAP_ENABLED; the schedule needs neither, being
switched only from the Scheduled Tasks page.

Main Features:
* The seed is a SONG kept in the album, or a DESCRIPTION turned into a DCLAP
  point, refinable with the search page's concepts and refused without an index.
* A description is also read for the attributes it NAMES, each judged by the
  index that knows it (a genre by the analysis tags, an instrument by the DCLAP
  concepts) and ranked by the PRODUCT of those ranks, because one text point for
  a compound query satisfies neither half. Each also queries DCLAP alone and
  narrows the pool by the same factor, taking turns over ATTRIBUTE_PASSES rounds
  so whichever narrows first does not win outright.
* TWO indexes feed the pool: the similar-song engine over MusiCNN (server-scoped,
  deduped) and DCLAP, which hears what merely measures alike. The pool steps one
  hop out and widens to POOL_MAX_QUERY until three albums of
  candidates survive from enough artists for the per-artist cap. Both embeddings
  then mix into ONE vector (ALBUM_CREATION_MUSICNN_SHARE) so every later step
  works in one space; without DCLAP the album is MusiCNN alone.
* Selection is a guided greedy towards a CALIBRATED cohesion target (0.86 mixed:
  the 0.80 real albums show within ONE artist is, across artists, a change of
  genre), with no pair below PAIR_FLOOR.
* Preferences hold only while enough candidates remain: tagged artist, album
  length, ERA_WINDOW_YEARS of the seed, and the ALBUM_CREATION_LYRIC_SHARE
  nearest in the lyrics space.
* At most OTHER_VOICE_CAP tracks break the album's sung/instrumental character,
  voted by each track's DCLAP neighbours (0.97 AUC) rather than its own lyrics
  flag, which is missing on 4 in 10 sung tracks.
* Dedup beyond the engine's: one version per song, no live/demo/remix/skit or
  rendition named in a title suffix, no holiday song outside December, at most
  MAX_SONGS_PER_ARTIST per artist.
* Sequencing follows what real albums measure: the calmest, least typical track
  closes, the two most intense take slots two and three, the opener is an intro
  or a bang by genre, and neither two calm tracks nor two of one artist sit
  together. Intensity is led by per-song centred mood scores.
"""

import logging
import math
import re
from collections import Counter
from datetime import date

import numpy as np

import config
from .ivf_manager import ensure_ivf_index_loaded, find_nearest_neighbors_by_vector
from .provider_migration_matcher import normalize_meta

logger = logging.getLogger(__name__)

SEED_SONG = 'song'
SEED_TEXT = 'text'
SEED_TYPES = (SEED_SONG, SEED_TEXT)

ROLE_OPENER = 'opener'
ROLE_SINGLE = 'single'
ROLE_TRACK = 'track'
ROLE_CLOSER = 'closer'

OPENER_INTRO = 'intro'
OPENER_BANG = 'bang'
INTRO_OPENER_GENRES = frozenset({'Hip-Hop', 'rnb', 'metal', 'electronic'})

MOOD_LABELS = ('danceable', 'aggressive', 'happy', 'party', 'relaxed', 'sad')
INWARD_LYRIC_LABELS = ('MELANCHOLIC', 'MORTAL', 'SOLITARY', 'SERENE')
OUTWARD_LYRIC_LABELS = ('VOLATILE', 'ADVERSARIAL', 'SENSORIAL', 'RADIANT')

AROUSING_MOODS = ('party', 'aggressive', 'danceable')
CALMING_MOODS = ('relaxed', 'sad')
INTENSITY_WEIGHTS = {'arousal': 0.6, 'energy': 0.3, 'tempo': 0.1}
TEMPO_OCTAVE = (70.0, 140.0)

CLOSER_WEIGHTS = {
    'intensity': -0.45, 'log_duration': 0.27, 'typicality': -0.26,
    'no_lyrics': 0.26, 'inward': 0.25,
}
SINGLE_WEIGHTS = {'intensity': 0.35, 'typicality': 0.25, 'happy': 0.15}
OPENER_WEIGHTS = {
    OPENER_BANG: {'intensity': 0.30, 'sad': -0.20, 'inward': -0.15, 'log_duration': -0.14},
    OPENER_INTRO: {'log_duration': -0.40, 'intensity': -0.25, 'no_lyrics': 0.30},
}
MIDDLE_INWARD_WEIGHT = 0.30
CALM_INTENSITY_Z = -0.8

POOL_FIRST_QUERY = 150
POOL_CLAP_QUERY = 150
POOL_MAX_QUERY = 1200
POOL_GROWTH = 4
TEXT_POOL_QUERY = 2000
TEXT_POOL_MAX = 4000
ATTRIBUTE_KEEP = 60
ATTRIBUTE_PASSES = 2
ARTIST_HEADROOM = 2
TEXT_QUERY_MAX_WORDS = 12
POOL_HOPS = 1
POOL_FRONTIER = 4
POOL_HOP_QUERY = 100
SELECTION_SAMPLE = 48
SELECTION_FLOOR = 0.15
PAIR_FLOOR = 0.75
ANCHOR_NEAREST = 20
MIN_ALBUM_TRACKS = 4
MIN_TRACK_SECONDS = 100.0
MAX_TRACK_SECONDS = 600.0
ERA_WINDOW_YEARS = 15
ERA_SAMPLE = 50
OTHER_VOICE_CAP = 2
VOICE_NEIGHBOURS = 10
VOICE_BLOCK = 256
PREFERENCE_HEADROOM = 3
WEEKLY_SEED_SAMPLE = 25

_BRACKETED = re.compile(r"[\(\[][^\)\]]*[\)\]]")
_DASH_SUFFIX = re.compile(r"(?<!\s)\s+-\s+[^\n]*$")
_NOT_WORD = re.compile(r"[^\w\s]", re.UNICODE)
_SPACES = re.compile(r"\s+")
_YEAR_TAG = re.compile(
    r"(?<!\s)\s+(?:['\N{RIGHT SINGLE QUOTATION MARK}]\d{2}|(?:19|20)\d{2})\s*$"
)
_VERSION_TITLE = re.compile(
    r"(?:\b(?:live|demo|remix|rmx|karaoke|rehearsal|skit|interlude|intro|outro|"
    r"reprise|alternate|dal vivo)\b|\btake \d)",
    re.IGNORECASE,
)
_TITLE_SUFFIX = re.compile(r"(?:[\(\[]([^\)\]]*)[\)\]])|(?:(?<!\s)\s+-\s+([^\n]*)$)")
_ALTERNATE_SUFFIX = re.compile(
    r"\b(mix|mixed|edit|version|ver|instrumental|acoustic|a ?capp?ella|sessions?|outtake|"
    r"commentary|interview|show|concert|stripped|orchestral|symphonic|extended|dub|unplugged|"
    r"choir|voiceless|work in progress)\b",
    re.IGNORECASE,
)
_CANONICAL_SUFFIX = re.compile(
    r"\b(remaster\w*|mono|stereo|album|original|single|radio)\b", re.IGNORECASE
)
_HOLIDAY_TEXT = re.compile(
    r"christmas|xmas|x-mas|natale|\bnoel\b|santa claus|jingle bell|silent night|navidad|"
    r"weihnacht|let it snow|sleigh ride|holy night|winter wonderland|rudolph|silver bells|"
    r"deck the hall|little drummer boy|ye faithful|merry gentlemen|what child is this|"
    r"herald angels|away in a manger",
    re.IGNORECASE,
)

_TRACK_SQL = (
    "SELECT s.item_id, s.title, s.author, s.album, s.album_artist, s.year, s.duration, "
    "s.tempo, s.energy, s.mood_vector, s.other_features, e.embedding, l.axis_vector, "
    "l.embedding AS lyrics_embedding, c.embedding AS clap_embedding "
    "FROM score s JOIN embedding e ON e.item_id = s.item_id "
    "LEFT JOIN lyrics_embedding l ON l.item_id = s.item_id "
    "LEFT JOIN clap_embedding c ON c.item_id = s.item_id "
    "WHERE e.embedding IS NOT NULL AND s.item_id = ANY(%s)"
)


class AlbumSeedError(ValueError):
    pass


class AlbumSeedNotFound(AlbumSeedError):
    pass


def song_key(title, author):
    base = _BRACKETED.sub(' ', (title or '').lower())
    base = _DASH_SUFFIX.sub(' ', base)
    base = _YEAR_TAG.sub('', base.rstrip()) or base
    base = _SPACES.sub(' ', _NOT_WORD.sub(' ', base)).strip()
    if not base:
        return None
    return base, (author or '').strip().lower()


def is_holiday_text(*texts):
    return any(text and _HOLIDAY_TEXT.search(text) for text in texts)


def is_alternate_take(title):
    suffixes = (first or second or '' for first, second in _TITLE_SUFFIX.findall(title or ''))
    return any(
        _ALTERNATE_SUFFIX.search(suffix) and not _CANONICAL_SUFFIX.search(suffix)
        for suffix in suffixes
    )


def has_clean_title(track, holiday_allowed):
    title = track['title'] or ''
    if _VERSION_TITLE.search(title) or is_alternate_take(title):
        return False
    return holiday_allowed or not is_holiday_text(track['title'], track['album'])


def has_known_artist(track):
    return bool(normalize_meta(track['author']))


def has_album_length(duration):
    return duration is None or MIN_TRACK_SECONDS <= duration <= MAX_TRACK_SECONDS


def is_album_candidate(track, holiday_allowed):
    return has_clean_title(track, holiday_allowed) and has_album_length(track['duration'])


def parse_scores(packed):
    scores = {}
    for part in (packed or '').split(','):
        label, _, value = part.partition(':')
        try:
            scores[label.strip()] = float(value)
        except ValueError:
            continue
    scores.pop('', None)
    return scores


def centered_moods(other_features):
    scores = parse_scores(other_features)
    present = [scores[label] for label in MOOD_LABELS if label in scores]
    if not present:
        return dict.fromkeys(MOOD_LABELS, 0.0)
    mean = sum(present) / len(present)
    return {label: scores.get(label, mean) - mean for label in MOOD_LABELS}


def top_genre(mood_vector):
    scores = parse_scores(mood_vector)
    return max(scores, key=scores.get) if scores else ''


def _axis_index():
    try:
        from lyrics import axis_columns

        return {label: index for index, (_axis, label) in enumerate(axis_columns())}
    except Exception:
        logger.exception("Lyric themes are unavailable; album sequencing runs on audio alone")
        return {}


def lyric_traits(axis_blob, axis_index):
    if not axis_blob or not axis_index:
        return False, 0.0
    vector = np.frombuffer(bytes(axis_blob), dtype=np.float32)
    if vector.shape[0] != len(axis_index) or float(vector.max() - vector.min()) <= 1e-6:
        return False, 0.0
    inward = float(np.mean([vector[axis_index[label]] for label in INWARD_LYRIC_LABELS]))
    outward = float(np.mean([vector[axis_index[label]] for label in OUTWARD_LYRIC_LABELS]))
    return True, inward - outward


def unit_rows(vectors):
    matrix = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-9)


def musicnn_share():
    return min(max(float(config.ALBUM_CREATION_MUSICNN_SHARE), 0.0), 1.0)


def mixed_rows(audio_vectors, clap_vectors):
    audio = unit_rows(audio_vectors)
    if clap_vectors is None:
        return audio
    share = musicnn_share()
    return np.concatenate(
        [math.sqrt(share) * audio, math.sqrt(1.0 - share) * unit_rows(clap_vectors)], axis=1
    )


def with_clap(tracks, required_count, needed):
    if musicnn_share() >= 1.0 or any(track['clap'] is None for track in tracks[:required_count]):
        return tracks, None
    kept = tracks[:required_count] + [
        track for track in tracks[required_count:] if track['clap'] is not None
    ]
    if len(kept) < needed:
        return tracks, None
    return kept, [track['clap'] for track in kept]


def cohesion_of(units):
    count = len(units)
    if count < 2:
        return 1.0
    similarity = units @ units.T
    return float((similarity.sum() - np.trace(similarity)) / (count * (count - 1)))


def zscores(values):
    data = np.array([np.nan if value is None else value for value in values], dtype=np.float64)
    known = ~np.isnan(data)
    if not known.any():
        return np.zeros(len(data))
    mean = data[known].mean()
    spread = data[known].std()
    data[~known] = mean
    if spread <= 1e-9:
        return np.zeros(len(data))
    return (data - mean) / spread


def _artist_has_room(author, artist_counts, max_per_artist, exempt_author):
    if not author or author == exempt_author or not max_per_artist or max_per_artist <= 0:
        return True
    return artist_counts[author] < max_per_artist


def _open_candidates(total, taken, keys, used_keys, authors, artist_counts,
                     max_per_artist, exempt_author, other_voice, other_voices):
    open_indices = []
    for index in range(total):
        if taken[index]:
            continue
        if keys[index] is not None and keys[index] in used_keys:
            continue
        if other_voice is not None and other_voice[index] and other_voices[0] >= OTHER_VOICE_CAP:
            continue
        if _artist_has_room(authors[index], artist_counts, max_per_artist, exempt_author):
            open_indices.append(index)
    return np.array(open_indices, dtype=np.int64)


def _commit_pick(pick, chosen, taken, authors, artist_counts, keys, used_keys,
                 other_voice, other_voices):
    chosen.append(pick)
    taken[pick] = True
    if authors[pick]:
        artist_counts[authors[pick]] += 1
    if keys[pick] is not None:
        used_keys.add(keys[pick])
    if other_voice is not None and other_voice[pick]:
        other_voices[0] += 1


def _narrow_to_the_best_candidates(open_indices, weakest_link, similarity_sum,
                                   count, target, rng):
    linked = open_indices[weakest_link[open_indices] >= PAIR_FLOOR]
    if linked.size:
        open_indices = linked
    near = open_indices[similarity_sum[open_indices] / count >= target - SELECTION_FLOOR]
    if near.size:
        open_indices = near
    if open_indices.size > SELECTION_SAMPLE:
        open_indices = rng.choice(open_indices, SELECTION_SAMPLE, replace=False)
    return open_indices


def _first_anchor(units, query_unit, rng):
    nearest = np.argsort(-(units @ query_unit))[:ANCHOR_NEAREST]
    return [int(rng.choice(nearest))]


def select_album_tracks(units, query_unit, authors, keys, required, size, target,
                        max_per_artist, exempt_author, rng, other_voice=None):
    chosen = list(required) or _first_anchor(units, query_unit, rng)
    artist_counts = Counter(authors[index] for index in chosen if authors[index])
    used_keys = {keys[index] for index in chosen if keys[index] is not None}
    taken = np.zeros(len(units), dtype=bool)
    taken[chosen] = True
    similarity_sum = units @ units[chosen].sum(axis=0)
    weakest_link = (units @ units[chosen].T).min(axis=1)
    pair_sum = float(sum(units[a] @ units[b] for i, a in enumerate(chosen) for b in chosen[i + 1:]))

    other_voices = [sum(1 for index in chosen if other_voice is not None and other_voice[index])]

    while len(chosen) < size:
        count = len(chosen)
        open_indices = _open_candidates(
            len(units), taken, keys, used_keys, authors, artist_counts,
            max_per_artist, exempt_author, other_voice, other_voices,
        )
        if open_indices.size == 0:
            break
        open_indices = _narrow_to_the_best_candidates(
            open_indices, weakest_link, similarity_sum, count, target, rng
        )
        after = (pair_sum + similarity_sum[open_indices]) / ((count + 1) * count / 2.0)
        pick = int(open_indices[int(np.argmin(np.abs(after - target)))])
        pair_sum += float(similarity_sum[pick])
        to_pick = units @ units[pick]
        similarity_sum = similarity_sum + to_pick
        weakest_link = np.minimum(weakest_link, to_pick)
        _commit_pick(pick, chosen, taken, authors, artist_counts, keys, used_keys,
                     other_voice, other_voices)
    return chosen


def opener_style(genres):
    known = [genre for genre in genres if genre]
    if not known:
        return OPENER_BANG
    dominant = Counter(known).most_common(1)[0][0]
    return OPENER_INTRO if dominant in INTRO_OPENER_GENRES else OPENER_BANG


def _weighted(features, weights):
    return sum(weight * features[name] for name, weight in weights.items())


def spread_calm_tracks(steady, calm, closer_is_calm):
    if not calm:
        return list(steady)
    last_gap = len(steady) - (1 if closer_is_calm else 0)
    gaps = list(range(last_gap, 0, -1))
    placed = {}
    leftover = []
    for offset, track in enumerate(reversed(calm)):
        if offset < len(gaps):
            placed[gaps[offset]] = track
        else:
            leftover.append(track)
    ordered = []
    for position, track in enumerate(steady, start=1):
        ordered.append(track)
        if position in placed:
            ordered.append(placed[position])
    return ordered + list(reversed(leftover))


def _may_follow(index, authors, calm, previous, after_calm, keep_calm_apart):
    if authors[index] and authors[index] == previous:
        return False
    return not (keep_calm_apart and after_calm and index in calm)


def artist_clashes(ordered, authors, author_before, author_after):
    names = [author_before] + [authors[index] for index in ordered] + [author_after]
    return sum(1 for first, second in zip(names, names[1:]) if first and first == second)


def calm_clashes(ordered, calm, calm_after):
    flags = [index in calm for index in ordered] + [bool(calm_after)]
    return sum(1 for first, second in zip(flags, flags[1:]) if first and second)


def _spread_artists(sequence, authors, calm, author_before, calm_before):
    remaining = list(sequence)
    ordered = []
    previous = author_before
    while remaining:
        counts = Counter(authors[index] for index in remaining if authors[index])
        crowded = [author for author, count in counts.items() if count > (len(remaining) + 1) // 2]
        after_calm = ordered[-1] in calm if ordered else calm_before
        crowd = [index for index in remaining if crowded and authors[index] == crowded[0]]
        pick = next(
            (
                index
                for keep_calm_apart, pool in ((True, crowd), (True, remaining), (False, remaining))
                for index in pool
                if _may_follow(index, authors, calm, previous, after_calm, keep_calm_apart)
            ),
            remaining[0],
        )
        remaining.remove(pick)
        ordered.append(pick)
        previous = authors[pick]
    return ordered


def separate_artists(middle, authors, calm, author_before, author_after=None, calm_after=False):
    from_the_start = _spread_artists(middle, authors, calm, author_before, False)
    from_the_end = _spread_artists(list(reversed(middle)), authors, calm, author_after, calm_after)
    from_the_end.reverse()
    return min(
        (from_the_start, from_the_end),
        key=lambda ordered: (
            artist_clashes(ordered, authors, author_before, author_after),
            calm_clashes(ordered, calm, calm_after),
        ),
    )


def sequence_album(features, style, authors=None):
    count = len(features['intensity'])
    remaining = list(range(count))
    if count < MIN_ALBUM_TRACKS:
        return [(index, ROLE_TRACK) for index in remaining]

    closer_scores = _weighted(features, CLOSER_WEIGHTS)
    closer = max(remaining, key=lambda index: closer_scores[index])
    remaining.remove(closer)

    single_scores = _weighted(features, SINGLE_WEIGHTS)
    singles = sorted(remaining, key=lambda index: -single_scores[index])[:2]
    for index in singles:
        remaining.remove(index)

    opener_scores = _weighted(features, OPENER_WEIGHTS[style])
    opener = max(remaining, key=lambda index: opener_scores[index])
    remaining.remove(opener)

    intensity = features['intensity']
    decline = intensity - MIDDLE_INWARD_WEIGHT * features['inward']
    remaining.sort(key=lambda index: -decline[index])
    calm = [index for index in remaining if intensity[index] < CALM_INTENSITY_Z]
    steady = [index for index in remaining if intensity[index] >= CALM_INTENSITY_Z]
    closer_is_calm = bool(intensity[closer] < CALM_INTENSITY_Z)
    middle = spread_calm_tracks(steady, calm, closer_is_calm)
    if authors:
        middle = separate_artists(
            middle, authors, set(calm), authors[singles[-1]], authors[closer], closer_is_calm
        )

    return (
        [(opener, ROLE_OPENER)]
        + [(index, ROLE_SINGLE) for index in singles]
        + [(index, ROLE_TRACK) for index in middle]
        + [(closer, ROLE_CLOSER)]
    )


def folded_tempo(tempo):
    if not tempo or tempo <= 0:
        return None
    low, high = TEMPO_OCTAVE
    while tempo >= high:
        tempo /= 2.0
    while tempo < low:
        tempo *= 2.0
    return tempo


def arousal_of(moods):
    return sum(moods[label] for label in AROUSING_MOODS) - sum(
        moods[label] for label in CALMING_MOODS
    )


def album_features(tracks, units):
    centroid = units.mean(axis=0)
    centroid = centroid / max(float(np.linalg.norm(centroid)), 1e-9)
    moods = [track['moods'] for track in tracks]
    signals = {
        'arousal': zscores([arousal_of(mood) for mood in moods]),
        'energy': zscores([track['energy'] for track in tracks]),
        'tempo': zscores([folded_tempo(track['tempo']) for track in tracks]),
    }
    return {
        'intensity': zscores(list(_weighted(signals, INTENSITY_WEIGHTS))),
        'log_duration': zscores(
            [math.log(track['duration']) if track['duration'] else None for track in tracks]
        ),
        'typicality': zscores(list(units @ centroid)),
        'happy': zscores([mood['happy'] for mood in moods]),
        'sad': zscores([mood['sad'] for mood in moods]),
        'inward': zscores([track['inward'] for track in tracks]),
        'no_lyrics': np.array([0.0 if track['has_lyrics'] else 1.0 for track in tracks]),
    }


def _vector_from_row(blob, dimension):
    if not blob or len(blob) != dimension * 4:
        return None
    return np.frombuffer(bytes(blob), dtype=np.float32)


def _clap_from_row(row):
    return _vector_from_row(row['clap_embedding'], config.CLAP_EMBEDDING_DIMENSION)


def _lyrics_from_row(row, has_lyrics):
    if not has_lyrics:
        return None
    vector = _vector_from_row(row['lyrics_embedding'], config.LYRICS_EMBEDDING_DIMENSION)
    if vector is None or float(np.linalg.norm(vector)) <= 1e-9:
        return None
    return vector


def _track_from_row(row, axis_index):
    has_lyrics, inward = lyric_traits(row['axis_vector'], axis_index)
    duration = row['duration']
    return {
        'item_id': row['item_id'],
        'title': row['title'],
        'author': row['author'],
        'album': row['album'],
        'album_artist': row['album_artist'],
        'year': row['year'],
        'duration': float(duration) if duration else None,
        'tempo': row['tempo'],
        'energy': row['energy'],
        'mood_vector': row['mood_vector'],
        'other_features': row['other_features'],
        'top_genre': top_genre(row['mood_vector']),
        'moods': centered_moods(row['other_features']),
        'has_lyrics': has_lyrics,
        'inward': inward if has_lyrics else None,
        'vector': np.frombuffer(bytes(row['embedding']), dtype=np.float32),
        'clap': _clap_from_row(row),
        'lyrics': _lyrics_from_row(row, has_lyrics),
    }


def load_tracks(item_ids, axis_index=None):
    ids = [str(item_id) for item_id in dict.fromkeys(item_ids) if item_id]
    if not ids:
        return []
    from database import get_db
    from psycopg2.extras import DictCursor

    if axis_index is None:
        axis_index = _axis_index()
    cur = get_db().cursor(cursor_factory=DictCursor)
    try:
        cur.execute(_TRACK_SQL, (ids,))
        rows = cur.fetchall()
    finally:
        cur.close()
    by_id = {
        row['item_id']: _track_from_row(row, axis_index)
        for row in rows
        if len(row['embedding']) == config.EMBEDDING_DIMENSION * 4
    }
    return [by_id[item_id] for item_id in ids if item_id in by_id]


def _ids_for(sql, params):
    from database import get_db

    cur = get_db().cursor()
    try:
        cur.execute(sql, params)
        return [row[0] for row in cur.fetchall()]
    finally:
        cur.close()


def _clean_author(author):
    return (author or '').strip().lower() if normalize_meta(author) else ''


def known_year(year):
    return year if year and 1900 < year < 2100 else None


def median_year(tracks):
    years = [known_year(track['year']) for track in tracks if known_year(track['year'])]
    return float(np.median(years)) if years else None


def other_voices(units, query_unit, has_lyrics):
    flags = np.asarray(has_lyrics, dtype=np.float32)
    total = len(units)
    wanted = min(VOICE_NEIGHBOURS, total - 1)
    if wanted < 1:
        return [False] * total
    sung = np.empty(total, dtype=bool)
    for start in range(0, total, VOICE_BLOCK):
        block = -(units[start:start + VOICE_BLOCK] @ units.T)
        rows = np.arange(len(block))
        block[rows, start + rows] = np.inf
        nearest = np.argpartition(block, wanted - 1, axis=1)[:, :wanted]
        sung[start:start + len(block)] = flags[nearest].mean(axis=1) >= 0.5
    around_seed = np.argsort(-(units @ query_unit))[:ANCHOR_NEAREST]
    seed_is_sung = 2 * float(flags[around_seed].sum()) >= len(around_seed)
    return [bool(differs) for differs in sung != seed_is_sung]


def in_era(track, year):
    own = known_year(track['year'])
    return year is None or own is None or abs(own - year) <= ERA_WINDOW_YEARS


def keep_preferred(tracks, needed, tests):
    for test in tests:
        kept = [track for track in tracks if test(track)]
        if len(kept) >= needed:
            tracks = kept
    return tracks


def keep_lyric_neighbours(tracks, lyric_query, needed):
    share = min(max(float(config.ALBUM_CREATION_LYRIC_SHARE), 0.0), 1.0)
    if lyric_query is None or share >= 1.0:
        return tracks
    sung = [track for track in tracks if track['lyrics'] is not None]
    if len(sung) < needed:
        return tracks
    scores = unit_rows([track['lyrics'] for track in sung]) @ unit_rows([lyric_query])[0]
    floor = float(np.quantile(scores, 1.0 - share))
    near = {track['item_id'] for track, score in zip(sung, scores) if score >= floor}
    silent = [track['item_id'] for track in tracks if track['lyrics'] is None]
    near.update(silent[:int(round(share * len(silent)))])
    kept = [track for track in tracks if track['item_id'] in near]
    return kept if len(kept) >= needed else tracks


def _song_seed(item_id, axis_index):
    if not item_id:
        raise AlbumSeedError("A song seed needs the item_id of the song.")
    tracks = load_tracks([item_id], axis_index)
    if not tracks:
        raise AlbumSeedNotFound("The selected song has not been analysed yet.")
    song = tracks[0]
    return {
        'type': SEED_SONG,
        'label': f"{song['title']} - {song['author']}",
        'name': f"{song['title']} - The Album",
        'query': song['vector'],
        'clap_query': song['clap'],
        'lyric_query': song['lyrics'],
        'required': [song],
        'excluded_ids': set(),
        'exempt_author': None,
        'target': config.ALBUM_CREATION_COHESION,
        'holiday': is_holiday_text(song['title'], song['album']),
        'year': known_year(song['year']),
    }


def _as_words(text):
    return _SPACES.sub(' ', _NOT_WORD.sub(' ', str(text or '').lower())).strip()


def tag_vocabulary():
    labels = {}
    for label in config.MOOD_LABELS:
        spelled = _as_words(label)
        if spelled:
            labels.setdefault(spelled, str(label).strip())
    return sorted(labels.items(), key=lambda pair: len(pair[0]), reverse=True)


def concept_vocabulary():
    try:
        from .clap_steering import concept_terms

        return sorted(concept_terms(), key=len, reverse=True)
    except Exception:
        logger.exception("The DCLAP concept catalogue could not be read")
        return []


def named_attributes(query):
    taken = ' ' + _as_words(query) + ' '
    tags, concepts = [], []
    for spelled, label in tag_vocabulary():
        if f' {spelled} ' in taken:
            tags.append(label)
            taken = taken.replace(f' {spelled} ', ' ')
    for term in concept_vocabulary():
        spelled = _as_words(term)
        if spelled and f' {spelled} ' in taken:
            concepts.append(term)
            taken = taken.replace(f' {spelled} ', ' ')
    return tags, concepts


def _within_pool_rank(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size < 2:
        return np.ones(values.size)
    order = np.argsort(np.argsort(values))
    return order / float(values.size - 1)


def attribute_columns(tracks, tags, concepts):
    columns = []
    present = [index for index, track in enumerate(tracks) if track['clap'] is not None]
    if concepts and len(present) >= 2:
        from .clap_steering import concept_scores

        fired = concept_scores([tracks[index]['clap'] for index in present], concepts)
        for values in fired.values():
            column = np.zeros(len(tracks))
            column[present] = _within_pool_rank(values)
            columns.append(column)
    for tag in tags:
        columns.append(_within_pool_rank(
            [parse_scores(track['mood_vector']).get(tag, 0.0) for track in tracks]
        ))
    return columns


def _rank_product(columns, size):
    product = np.ones(size)
    for column in columns:
        product = product * np.maximum(column, 1e-6)
    return product


def enough_artists(tracks, indexes):
    cap = config.MAX_SONGS_PER_ARTIST
    if not cap or cap <= 0:
        return True
    names = {_clean_author(tracks[index]['author']) for index in indexes}
    names.discard('')
    return len(names) * cap >= config.ALBUM_CREATION_TRACKS * ARTIST_HEADROOM


def keep_named_attributes(tracks, tags, concepts, needed):
    columns = attribute_columns(tracks, tags, concepts)
    if not columns or len(tracks) <= needed:
        return tracks
    kept = list(range(len(tracks)))
    target = max(needed, ATTRIBUTE_KEEP)
    step = (target / float(len(kept))) ** (1.0 / (len(columns) * ATTRIBUTE_PASSES))
    for _pass in range(ATTRIBUTE_PASSES):
        for column in columns:
            wanted = max(target, int(round(len(kept) * step)))
            narrowed = sorted(kept, key=lambda index, scores=column: -scores[index])[:wanted]
            if len(narrowed) >= needed and enough_artists(tracks, narrowed):
                kept = narrowed
    product = _rank_product(columns, len(tracks))
    keep = max(needed, ATTRIBUTE_KEEP)
    best = sorted(kept, key=lambda index: -product[index])[:keep]
    return [tracks[index] for index in sorted(best)]


def text_pool_ids(embedding, count):
    from .clap_text_search import is_clap_cache_loaded, search_by_embedding

    if not is_clap_cache_loaded():
        raise AlbumSeedError(
            "The DCLAP index is not loaded, so an album cannot be built from a description."
        )
    found = search_by_embedding(np.asarray(embedding, dtype=np.float32), limit=count)
    return available_ids([row['item_id'] for row in found])


def _text_embedding(query, steering=None):
    from .clap_analyzer import get_text_embedding
    from .clap_text_search import is_clap_cache_loaded, warmup_text_search_model

    if not config.CLAP_ENABLED:
        raise AlbumSeedError("DCLAP is turned off, so an album cannot be built from a description.")
    if not is_clap_cache_loaded():
        raise AlbumSeedError(
            "The DCLAP index is not loaded, so an album cannot be built from a description."
        )
    warmup_text_search_model()
    embedding = get_text_embedding(query)
    if embedding is None:
        raise AlbumSeedError("That description could not be understood; try other words.")
    if steering:
        from .clap_steering import apply_steering

        embedding, _applied = apply_steering(embedding, steering)
    return np.asarray(embedding, dtype=np.float32).reshape(-1)


def _text_seed(query, steering=None):
    words = (query or '').split()
    if not words:
        raise AlbumSeedError("Describe the album you want in a few words.")
    if len(words) > TEXT_QUERY_MAX_WORDS:
        raise AlbumSeedError(f"Use at most {TEXT_QUERY_MAX_WORDS} words to describe the album.")
    clean_query = ' '.join(words)
    embedding = _text_embedding(clean_query, steering)
    tags, concepts = named_attributes(clean_query)
    named = [_text_embedding(word) for word in tags + concepts] if len(tags + concepts) > 1 else []
    return {
        'type': SEED_TEXT,
        'label': clean_query,
        'name': clean_query[:1].upper() + clean_query[1:],
        'query': None,
        'clap_query': embedding,
        'lyric_query': None,
        'required': [],
        'excluded_ids': set(),
        'target': config.ALBUM_CREATION_COHESION,
        'holiday': is_holiday_text(clean_query),
        'year': None,
        'tags': tags,
        'concepts': concepts + [term['term'] for term in (steering or []) if term.get('direction') != 'less'],
        'named_queries': named,
    }


def resolve_seed(seed_type, item_id=None, query=None, axis_index=None, steering=None):
    if seed_type == SEED_SONG:
        return _song_seed(item_id, axis_index)
    if seed_type == SEED_TEXT:
        return _text_seed(query, steering)
    raise AlbumSeedError("Unknown seed type; expected one of: " + ", ".join(SEED_TYPES) + ".")


def is_enabled():
    return bool(config.LYRICS_ENABLED and config.CLAP_ENABLED)


def similar_song_ids(vector, count):
    neighbours = find_nearest_neighbors_by_vector(vector, n=count, eliminate_duplicates=False)
    return [row['item_id'] for row in neighbours]


def available_ids(item_ids):
    ids = [item_id for item_id in dict.fromkeys(item_ids) if item_id]
    if not ids:
        return []
    from .index_availability import active_availability_scope
    from .mediaserver import registry

    server_id = active_availability_scope()
    if not server_id:
        return ids
    kept = set(_ids_for(
        "SELECT s.item_id FROM score s WHERE s.item_id = ANY(%s) AND "
        + registry.availability_sql('s'),
        (ids, server_id, server_id == registry.get_default_server_id()),
    ))
    return [item_id for item_id in ids if item_id in kept]


def clap_song_ids(vector, count):
    if vector is None or musicnn_share() >= 1.0:
        return []
    try:
        from .clap_text_search import is_clap_cache_loaded, search_by_embedding

        if not is_clap_cache_loaded():
            logger.info("The DCLAP index is not loaded; the album pool uses MusiCNN alone.")
            return []
        found = search_by_embedding(np.asarray(vector, dtype=np.float32), limit=count)
    except Exception:
        logger.exception("The DCLAP index could not be queried; the album pool uses MusiCNN alone")
        return []
    return available_ids([row['item_id'] for row in found])


def candidate_pool(query_vector, axis_index=None, clap_query=None, first=POOL_FIRST_QUERY,
                   carried=None):
    if not ensure_ivf_index_loaded():
        raise RuntimeError("The similarity index is not loaded; run the analysis first.")
    seen, tracks = (carried if carried is not None else ({}, []))
    fresh = [item_id for item_id in similar_song_ids(query_vector, first) if item_id not in seen]
    fresh += [
        item_id for item_id in clap_song_ids(clap_query, min(first, POOL_CLAP_QUERY))
        if item_id not in seen and item_id not in set(fresh)
    ]
    seen.update(dict.fromkeys(fresh))
    tracks = tracks + load_tracks(fresh, axis_index)
    query_unit = unit_rows([query_vector])[0]
    frontier = tracks
    for _ in range(POOL_HOPS):
        if not frontier:
            break
        similarity = unit_rows([track['vector'] for track in frontier]) @ query_unit
        farthest = np.argsort(similarity)[:POOL_FRONTIER]
        fresh = list(dict.fromkeys(
            item_id
            for index in farthest
            for item_id in similar_song_ids(frontier[index]['vector'], POOL_HOP_QUERY)
            if item_id not in seen
        ))
        if not fresh:
            break
        seen.update(dict.fromkeys(fresh))
        frontier = load_tracks(fresh, axis_index)
        tracks.extend(frontier)
    return tracks, (seen, tracks)


def _public_track(track, slot, role):
    return {
        'slot': slot,
        'role': role,
        'item_id': track['item_id'],
        'title': track['title'],
        'author': track['author'],
        'album': track['album'],
        'album_artist': track['album_artist'],
        'year': track['year'],
        'duration': track['duration'],
        'mood_vector': track['mood_vector'],
        'other_features': track['other_features'],
    }


def text_pool_ids_for(seed, first, seen=None):
    named = seed.get('named_queries') or []
    ids = list(text_pool_ids(seed['clap_query'], first))
    share = max(first // 2, POOL_FIRST_QUERY)
    for embedding in named:
        ids.extend(text_pool_ids(embedding, share))
    known = seen if seen is not None else set()
    return [item_id for item_id in dict.fromkeys(ids) if item_id not in known]


def seed_unit(seed, units, clap_vectors):
    if seed['query'] is not None:
        return mixed_rows(
            [seed['query']], None if clap_vectors is None else [seed['clap_query']]
        )[0]
    if clap_vectors is None or seed['clap_query'] is None:
        return units[0]
    nearest = unit_rows(clap_vectors) @ unit_rows([seed['clap_query']])[0]
    return units[int(np.argmax(nearest))]


def _voice_query(seed, clap_vectors):
    if seed['clap_query'] is not None:
        return unit_rows([seed['clap_query']])[0]
    return unit_rows(clap_vectors)[0]


def gather_candidates(seed, axis_index, holiday_allowed, needed):
    skipped = seed['excluded_ids'] | {track['item_id'] for track in seed['required']}
    first = POOL_FIRST_QUERY if seed['type'] == SEED_SONG else TEXT_POOL_QUERY
    ceiling = POOL_MAX_QUERY if seed['type'] == SEED_SONG else TEXT_POOL_MAX
    carried, text_seen, found = None, set(), []
    while True:
        if seed['type'] == SEED_TEXT:
            wanted = text_pool_ids_for(seed, first, text_seen)
            text_seen.update(wanted)
            found = found + load_tracks(wanted, axis_index)
        else:
            found, carried = candidate_pool(
                seed['query'], axis_index, seed['clap_query'], first, carried
            )
        clean = [
            track for track in found
            if track['item_id'] not in skipped and has_clean_title(track, holiday_allowed)
        ]
        varied = len(clean) >= needed and enough_artists(clean, range(len(clean)))
        if varied or first >= ceiling:
            if not varied:
                logger.info(
                    "Only %d candidates by %d artists after widening the pool to %d.",
                    len(clean), len({_clean_author(track['author']) for track in clean}), first,
                )
            return clean
        first = min(first * POOL_GROWTH, ceiling)


def create_album(seed_type, item_id=None, query=None, steering=None, rng=None, today=None):
    if rng is None:
        rng = np.random.default_rng()
    axis_index = _axis_index()
    seed = resolve_seed(seed_type, item_id, query, axis_index, steering)
    holiday_allowed = seed['holiday'] or (today or date.today()).month == 12
    required = seed['required']
    needed = config.ALBUM_CREATION_TRACKS * PREFERENCE_HEADROOM
    clean = gather_candidates(seed, axis_index, holiday_allowed, needed)
    if seed['type'] == SEED_TEXT:
        seed['year'] = median_year(clean[:ERA_SAMPLE])
        clean = keep_named_attributes(clean, seed['tags'], seed['concepts'], needed)
    tracks = required + keep_lyric_neighbours(
        keep_preferred(
            clean,
            needed,
            (
                has_known_artist,
                lambda track: has_album_length(track['duration']),
                lambda track: in_era(track, seed['year']),
            ),
        ),
        seed['lyric_query'],
        needed,
    )
    if len(tracks) < MIN_ALBUM_TRACKS:
        raise AlbumSeedNotFound("Not enough analysed songs around this seed to build an album.")

    tracks, clap_vectors = with_clap(tracks, len(required), config.ALBUM_CREATION_TRACKS)
    units = mixed_rows([track['vector'] for track in tracks], clap_vectors)
    query_unit = seed_unit(seed, units, clap_vectors)
    voice_units = units if clap_vectors is None else unit_rows(clap_vectors)
    voice_query = query_unit if clap_vectors is None else _voice_query(seed, clap_vectors)
    chosen = select_album_tracks(
        units,
        query_unit,
        [_clean_author(track['author']) for track in tracks],
        [song_key(track['title'], track['author']) for track in tracks],
        list(range(len(required))),
        config.ALBUM_CREATION_TRACKS,
        seed['target'],
        config.MAX_SONGS_PER_ARTIST,
        None,
        rng,
        other_voices(voice_units, voice_query, [track['has_lyrics'] for track in tracks]),
    )
    picked = [tracks[index] for index in chosen]
    picked_units = units[chosen]
    style = opener_style([track['top_genre'] for track in picked])
    order = sequence_album(
        album_features(picked, picked_units), style,
        [_clean_author(track['author']) for track in picked],
    )
    seconds = sum(track['duration'] or 0.0 for track in picked)
    return {
        'seed': {'type': seed['type'], 'label': seed['label']},
        'suggested_name': seed['name'],
        'stats': {
            'tracks': len(picked),
            'minutes': int(round(seconds / 60.0)),
            'artists': len({_clean_author(track['author']) for track in picked}),
            'cohesion': round(cohesion_of(picked_units), 3),
            'target_cohesion': round(seed['target'], 3),
            'opener_style': style,
        },
        'tracks': [
            _public_track(picked[index], slot, role)
            for slot, (index, role) in enumerate(order, start=1)
        ],
    }


def weekly_seed_ids(today=None):
    from database import get_db
    from .mediaserver import context as ms_context, registry

    default_id = registry.get_default_server_id()
    server_id = ms_context.active_server_id() or default_id
    sql = (
        "SELECT s.item_id, s.title, s.album, s.duration FROM score s "
        "JOIN embedding e ON e.item_id = s.item_id WHERE e.embedding IS NOT NULL"
    )
    params = []
    if server_id:
        sql += " AND " + registry.availability_sql('s')
        params += [server_id, server_id == default_id]
    cur = get_db().cursor()
    try:
        cur.execute(sql + " ORDER BY random() LIMIT %s", tuple(params + [WEEKLY_SEED_SAMPLE]))
        rows = cur.fetchall()
    finally:
        cur.close()
    holiday_allowed = (today or date.today()).month == 12
    clean = [
        (item_id, duration) for item_id, title, album, duration in rows
        if has_clean_title({'title': title, 'album': album}, holiday_allowed)
    ]
    clean.sort(key=lambda row: not has_album_length(row[1]))
    return [item_id for item_id, _duration in clean]


def create_album_of_the_week():
    for item_id in weekly_seed_ids():
        try:
            return create_album(SEED_SONG, item_id=item_id)
        except AlbumSeedError:
            logger.info("Weekly seed %s could not grow into an album; trying another.", item_id)
    return None


def run_album_of_the_week_task(server_scope="all"):
    from config import ALBUM_OF_THE_WEEK_PLAYLIST_NAME
    from .task_run import run_playlist_task_per_server

    def build_ids():
        album = create_album_of_the_week()
        if not album:
            return []
        logger.info("The album of the week grew from seed %s.", album['seed']['label'])
        return [track['item_id'] for track in album['tracks']]

    return run_playlist_task_per_server(
        'album_of_the_week', 'album of the week',
        ALBUM_OF_THE_WEEK_PLAYLIST_NAME, 'Album of the Week',
        build_ids, server_scope,
    )
