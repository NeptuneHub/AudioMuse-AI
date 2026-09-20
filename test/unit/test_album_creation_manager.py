# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Album Creation manager: which songs make the album and in which order.

Main Features:
* Dedup keys (a year tag is the same song), the live/demo/skit filter, the alternate
  rendition suffixes (remaster, mono, single and radio cuts stay), the holiday guard
  and the duration bounds
* Mood scores are centred per song and the tempo is folded into one octave, so a
  doubled tempo never reads as a more intense song
* The selector keeps the seed, honours the per-artist cap, keeps one version per
  song, lands on the target cohesion
  and never links a track below the pair floor while an alternative exists
* The sequencer puts one opener, two singles, the middle and one closer in order,
  closes on the calmest long track and keeps calm tracks apart
* create_album keeps a song seed, filters the pool and returns JSON-ready rows;
  the pool walks outward through the similar-song engine without its artist cap
  and widens itself when too few candidates survive
* A description seed reads its genre from the analysis tags and its instrument from
  the DCLAP concept dictionary, and ranks candidates by the product of the two
* Tagged artists, the seed's era and album length are preferred while enough remain,
  and an untagged library is not held to the artist cap
* Sung or instrumental is a vote of a track's nearest pool neighbours, so a missing
  or hallucinated lyrics flag changes nothing; at most OTHER_VOICE_CAP tracks of the
  other voice get in
* The weekly task upserts ONE fixed playlist name per server and keeps the previous
  playlist when nothing could be built. No flag switches it off: only its row on the
  Scheduled Tasks page does
"""

from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import config
from tasks import album_creation_manager as acm

DIM = 16


def _unit(vector):
    vector = np.asarray(vector, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def _cluster(rng, count, spread, axis=0):
    base = np.zeros(DIM, dtype=np.float32)
    base[axis] = 1.0
    return [_unit(base + spread * rng.standard_normal(DIM)) for _ in range(count)]


def _leaning_cluster(rng, count, spread, lean):
    base = np.zeros(DIM, dtype=np.float32)
    base[0], base[1] = 1.0, lean
    return [_unit(base + spread * rng.standard_normal(DIM)) for _ in range(count)]


def _track(item_id, vector, title=None, author=None, **extra):
    vector = np.asarray(vector, dtype=np.float32)
    track = {
        'item_id': item_id,
        'title': title or f'Song {item_id}',
        'author': author or f'Artist {item_id}',
        'album': 'Album',
        'album_artist': author or f'Artist {item_id}',
        'year': 2001,
        'duration': 220.0,
        'tempo': 120.0,
        'energy': 0.5,
        'mood_vector': 'rock:0.6,pop:0.4',
        'other_features': 'danceable:0.6,aggressive:0.6,happy:0.6,party:0.6,relaxed:0.6,sad:0.6',
        'top_genre': 'rock',
        'moods': dict.fromkeys(acm.MOOD_LABELS, 0.0),
        'has_lyrics': True,
        'inward': 0.0,
        'vector': vector,
        'clap': vector.copy(),
        'lyrics': np.ones(8, dtype=np.float32),
    }
    track.update(extra)
    return track


class TestDedupKeysAndHygiene:
    def test_a_bracketed_or_dashed_suffix_is_the_same_song(self):
        plain = acm.song_key('Heroes', 'David Bowie')
        assert acm.song_key('Heroes (2017 Remaster)', 'david bowie ') == plain
        assert acm.song_key('Heroes - Single Version', 'David Bowie') == plain
        assert acm.song_key('Heroes', 'Another Artist') != plain

    def test_a_track_without_a_title_has_no_key_and_is_never_folded(self):
        assert acm.song_key('', 'Someone') is None
        assert acm.song_key('(Live)', 'Someone') is None

    @pytest.mark.parametrize('title', [
        'Song (Live at Wembley)', 'Song - Demo', 'Song (Club Remix)', 'Skit', 'Interlude II',
        'Intro', 'Song (Alternate Take 3)', 'Song (take 2)', 'Canzone (dal vivo)',
    ])
    def test_versions_and_non_songs_are_not_album_material(self, title):
        assert not acm.is_album_candidate(_track('x', np.ones(DIM), title=title), False)

    @pytest.mark.parametrize('title', ['Alive', 'Demolition Man', 'Delivery', 'Introspection'])
    def test_the_version_filter_matches_whole_words_only(self, title):
        assert acm.is_album_candidate(_track('x', np.ones(DIM), title=title), False)

    @pytest.mark.parametrize('title', [
        'Song (instrumental)', 'Song (Acoustic Version)', 'Song (extended mix)', 'Song [Morales dub mix]',
        'Song (edit)', 'Song (no choir)', 'Song (a cappella)', 'Song (Acappella)', 'Song (outtake)',
        'Song (Work in Progress)', 'Song (BBC session)', 'Song (August 12, Dinner Show)',
        'Song (Japanese ver.)', 'Song - Acoustic', 'Song - iTunes Session',
    ])
    def test_an_alternate_rendition_named_in_a_suffix_is_not_album_material(self, title):
        assert not acm.has_clean_title(_track('x', np.ones(DIM), title=title), False)

    @pytest.mark.parametrize('title', [
        'Song (2011 Remaster)', 'Song (remastered 1998)', 'Song (mono)', 'Song (Album Version)',
        'Song (Single Version)', 'Song (single edit)', 'Song (radio edit)', 'Song (Original Mix)',
        'Song (mono version)', 'Song, Pt. 2', 'Song (feat. Someone)', 'Song - Allegro',
    ])
    def test_the_song_itself_under_another_label_stays(self, title):
        assert acm.has_clean_title(_track('x', np.ones(DIM), title=title), False)

    @pytest.mark.parametrize('title', ['Acoustic Dreams', 'Mix It Up', 'The Show Must Go On', 'Edit the Sad Parts'])
    def test_a_rendition_word_outside_a_suffix_is_only_a_title(self, title):
        assert acm.has_clean_title(_track('x', np.ones(DIM), title=title), False)

    def test_a_year_tag_is_the_same_song(self):
        plain = acm.song_key('Star People', 'George Michael')
        assert acm.song_key("Star People '97 (radio version)", 'George Michael') == plain
        assert acm.song_key('Star People \N{RIGHT SINGLE QUOTATION MARK}97', 'George Michael') == plain
        assert acm.song_key('Star People 1997', 'George Michael') == plain

    def test_a_title_that_is_only_a_year_or_a_number_keeps_it(self):
        assert acm.song_key('1999', 'Prince') == ('1999', 'prince')
        assert acm.song_key('Sonata No. 14', 'X') != acm.song_key('Sonata No. 21', 'X')

    @pytest.mark.parametrize('author, known', [
        ('Nick Drake', True), ('Unknown Artist', False), ('[Unknown Artist]', False), ('', False), (None, False),
    ])
    def test_a_placeholder_is_not_an_artist(self, author, known):
        assert acm.has_known_artist({'author': author}) is known
        assert bool(acm._clean_author(author)) is known

    @pytest.mark.parametrize('duration, kept', [(99.0, False), (100.0, True), (600.0, True), (601.0, False), (None, True)])
    def test_the_duration_bounds_drop_skits_and_epics_but_keep_unknowns(self, duration, kept):
        assert acm.is_album_candidate(_track('x', np.ones(DIM), duration=duration), False) is kept

    def test_holiday_songs_stay_out_unless_allowed(self):
        carol = _track('x', np.ones(DIM), title='White Christmas')
        by_album = _track('y', np.ones(DIM), title='Track 4', album='A Very Special Xmas')
        assert not acm.is_album_candidate(carol, False)
        assert not acm.is_album_candidate(by_album, False)
        assert acm.is_album_candidate(carol, True)
        assert acm.is_holiday_text('What Child Is This?') and acm.is_holiday_text('Hark! The Herald Angels Sing')


class TestSignals:
    def test_mood_scores_are_centred_per_song(self):
        moods = acm.centered_moods('danceable:0.7,aggressive:0.6,happy:0.8,party:0.7,relaxed:0.5,sad:0.6')
        assert abs(sum(moods.values())) < 1e-9
        assert moods['happy'] > 0 > moods['relaxed']

    def test_missing_mood_scores_are_neutral(self):
        assert acm.centered_moods(None) == dict.fromkeys(acm.MOOD_LABELS, 0.0)
        assert acm.centered_moods('happy:0.9')['sad'] == 0.0

    @pytest.mark.parametrize('tempo, folded', [(85.0, 85.0), (170.0, 85.0), (42.5, 85.0), (140.0, 70.0), (0, None), (None, None)])
    def test_the_tempo_is_folded_into_one_octave(self, tempo, folded):
        assert acm.folded_tempo(tempo) == folded

    def test_a_doubled_tempo_does_not_make_a_song_more_intense(self):
        rng = np.random.default_rng(0)
        vectors = _cluster(rng, 6, 0.05)
        tracks = [_track(str(i), v) for i, v in enumerate(vectors)]
        tracks[0]['tempo'] = 85.0
        tracks[1]['tempo'] = 170.0
        features = acm.album_features(tracks, acm.unit_rows(vectors))
        assert features['intensity'][0] == pytest.approx(features['intensity'][1])

    def test_intensity_follows_the_moods_before_the_energy(self):
        rng = np.random.default_rng(1)
        vectors = _cluster(rng, 6, 0.05)
        tracks = [_track(str(i), v) for i, v in enumerate(vectors)]
        tracks[0]['moods'] = dict(tracks[0]['moods'], party=0.2, aggressive=0.2, relaxed=-0.2)
        tracks[0]['energy'] = 0.45
        tracks[1]['moods'] = dict(tracks[1]['moods'], relaxed=0.2, sad=0.2, party=-0.2)
        tracks[1]['energy'] = 0.55
        features = acm.album_features(tracks, acm.unit_rows(vectors))
        assert features['intensity'][0] > features['intensity'][1]

    def test_the_instrumental_sentinel_and_a_wrong_size_mean_no_lyrics(self):
        axis_index = {label: i for i, label in enumerate(acm.INWARD_LYRIC_LABELS + acm.OUTWARD_LYRIC_LABELS)}
        flat = np.full(len(axis_index), -0.19, dtype=np.float32).tobytes()
        assert acm.lyric_traits(flat, axis_index) == (False, 0.0)
        assert acm.lyric_traits(np.ones(3, dtype=np.float32).tobytes(), axis_index) == (False, 0.0)
        assert acm.lyric_traits(None, axis_index) == (False, 0.0)

    def test_inward_lyrics_score_above_outward_ones(self):
        labels = acm.INWARD_LYRIC_LABELS + acm.OUTWARD_LYRIC_LABELS
        axis_index = {label: i for i, label in enumerate(labels)}
        vector = np.array([0.9] * 4 + [0.1] * 4, dtype=np.float32)
        has_lyrics, inward = acm.lyric_traits(vector.tobytes(), axis_index)
        assert has_lyrics and inward == pytest.approx(0.8)

    def test_zscores_treat_unknowns_as_average_and_survive_a_flat_column(self):
        assert list(acm.zscores([1.0, None, 3.0])) == pytest.approx([-1.0, 0.0, 1.0])
        assert list(acm.zscores([5.0, 5.0])) == [0.0, 0.0]
        assert list(acm.zscores([None, None])) == [0.0, 0.0]


class TestTheSelector:
    def _pool(self, seed=3, near=40, far=160):
        rng = np.random.default_rng(seed)
        vectors = _cluster(rng, near, 0.05) + _cluster(rng, far, 0.13)
        return acm.unit_rows(vectors)

    def _select(self, units, **kw):
        count = len(units)
        args = {
            'authors': [f'a{i}' for i in range(count)],
            'keys': [(f't{i}', f'a{i}') for i in range(count)],
            'required': [0],
            'size': 12,
            'target': 0.90,
            'max_per_artist': 3,
            'rng': np.random.default_rng(11),
            'other_voice': None,
        }
        args.update(kw)
        return acm.select_album_tracks(
            units, units[0], args['authors'], args['keys'], args['required'], args['size'],
            args['target'], args['max_per_artist'], None, args['rng'],
            args['other_voice'],
        )

    def test_the_seed_stays_first_and_the_album_is_full_and_unique(self):
        chosen = self._select(self._pool())
        assert chosen[0] == 0
        assert len(chosen) == 12 and len(set(chosen)) == 12

    def test_the_album_lands_on_the_target_cohesion_not_on_nearest_neighbours(self):
        units = self._pool()
        chosen = self._select(units)
        nearest = list(np.argsort(-(units @ units[0]))[:12])
        assert acm.cohesion_of(units[chosen]) == pytest.approx(0.90, abs=0.03)
        assert acm.cohesion_of(units[nearest]) > acm.cohesion_of(units[chosen]) + 0.03

    def _two_voices(self):
        rng = np.random.default_rng(8)
        return acm.unit_rows(_cluster(rng, 40, 0.05, axis=0) + _cluster(rng, 40, 0.05, axis=1))

    def test_a_sung_song_whose_lyrics_are_missing_still_counts_as_sung(self):
        units = self._two_voices()
        has_lyrics = [True] * 40 + [False] * 40
        has_lyrics[0] = has_lyrics[7] = False
        other = acm.other_voices(units, units[0], has_lyrics)
        assert not any(other[:40]) and all(other[40:])

    def test_an_instrumental_with_hallucinated_lyrics_still_counts_as_instrumental(self):
        units = self._two_voices()
        has_lyrics = [True] * 40 + [False] * 40
        has_lyrics[40] = has_lyrics[52] = True
        other = acm.other_voices(units, units[40], has_lyrics)
        assert all(other[:40]) and not any(other[40:])

    def test_a_library_without_lyrics_has_no_other_voice(self):
        units = self._two_voices()
        assert not any(acm.other_voices(units, units[0], [False] * 80))

    def test_at_most_a_couple_of_tracks_differ_from_the_seed_in_having_lyrics(self):
        units = self._pool()
        other_voice = [False] + [True] * 120 + [False] * (len(units) - 121)
        chosen = self._select(units, other_voice=other_voice)
        assert len(chosen) == 12
        assert sum(1 for index in chosen if other_voice[index]) == acm.OTHER_VOICE_CAP

    def test_an_all_instrumental_neighbourhood_still_fills_up_to_the_cap_and_stops(self):
        units = self._pool(near=10, far=10)
        other_voice = [False] + [True] * (len(units) - 1)
        assert len(self._select(units, other_voice=other_voice)) == 1 + acm.OTHER_VOICE_CAP

    def test_no_artist_gets_more_than_the_cap(self):
        units = self._pool()
        authors = ['same'] * 60 + [f'a{i}' for i in range(len(units) - 60)]
        chosen = self._select(units, authors=authors)
        assert sum(1 for i in chosen if authors[i] == 'same') == 3
        assert len(chosen) == 12

    def test_a_zero_cap_means_no_cap(self):
        units = self._pool()
        assert len(self._select(units, authors=['same'] * len(units), max_per_artist=0)) == 12

    def test_a_track_without_an_artist_is_never_counted_against_a_cap(self):
        units = self._pool()
        assert len(self._select(units, authors=[''] * len(units))) == 12

    def test_only_one_version_of_a_song_gets_in(self):
        units = self._pool()
        keys = [('same song', 'x')] * 50 + [(f't{i}', 'x') for i in range(len(units) - 50)]
        chosen = self._select(units, keys=keys)
        assert sum(1 for i in chosen if keys[i] == ('same song', 'x')) == 1

    def test_untitled_tracks_are_never_folded_together(self):
        units = self._pool()
        assert len(self._select(units, keys=[None] * len(units))) == 12

    def test_every_pair_stays_above_the_floor_while_an_alternative_exists(self):
        units = self._pool()
        chosen = self._select(units)
        similarity = units[chosen] @ units[chosen].T
        assert similarity[np.triu_indices(len(chosen), 1)].min() >= acm.PAIR_FLOOR

    def test_a_pool_that_cannot_honour_the_floor_still_fills_the_album(self):
        rng = np.random.default_rng(5)
        units = acm.unit_rows([_unit(rng.standard_normal(DIM)) for _ in range(40)])
        assert len(self._select(units)) == 12

    def test_a_small_pool_gives_a_short_album_not_an_error(self):
        assert len(self._select(self._pool(near=3, far=2))) == 5

    def test_without_a_required_track_the_anchor_is_near_the_query(self):
        units = self._pool()
        chosen = self._select(units, required=[])
        nearest = set(np.argsort(-(units @ units[0]))[:acm.ANCHOR_NEAREST])
        assert chosen[0] in nearest and len(chosen) == 12

    def test_the_same_random_state_gives_the_same_album(self):
        units = self._pool()
        first = self._select(units, rng=np.random.default_rng(99))
        assert first == self._select(units, rng=np.random.default_rng(99))
        assert first != self._select(units, rng=np.random.default_rng(100))


class TestTheSequencer:
    def _features(self, intensity, **columns):
        count = len(intensity)
        features = {name: np.zeros(count) for name in (
            'log_duration', 'typicality', 'happy', 'sad', 'inward', 'no_lyrics')}
        features['intensity'] = np.array(intensity, dtype=float)
        for name, values in columns.items():
            features[name] = np.array(values, dtype=float)
        return features

    def test_the_running_order_is_opener_two_singles_middle_closer(self):
        features = self._features([0.2, 1.5, 1.2, 0.4, 0.1, -0.2, -0.4, 0.0, 0.3, -0.6, -2.0, 0.6])
        order = acm.sequence_album(features, acm.OPENER_BANG)
        roles = [role for _index, role in order]
        assert roles == [acm.ROLE_OPENER, acm.ROLE_SINGLE, acm.ROLE_SINGLE] + [acm.ROLE_TRACK] * 8 + [acm.ROLE_CLOSER]
        assert sorted(index for index, _role in order) == list(range(12))

    def test_the_calmest_long_track_closes_and_the_two_most_intense_are_the_singles(self):
        features = self._features(
            [0.2, 1.5, 1.2, 0.4, 0.1, -0.2, -0.4, 0.0, 0.3, -0.6, -2.0, 0.6],
            log_duration=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.5, 0],
        )
        order = acm.sequence_album(features, acm.OPENER_BANG)
        assert order[-1] == (10, acm.ROLE_CLOSER)
        assert {index for index, role in order if role == acm.ROLE_SINGLE} == {1, 2}
        assert order[0] == (11, acm.ROLE_OPENER)

    def test_an_intro_style_album_opens_on_a_short_quiet_instrumental(self):
        features = self._features(
            [0.2, 1.5, 1.2, 0.4, 0.1, -0.2, -0.3, 0.0, 0.3, -0.5, -2.0, 0.6],
            log_duration=[0, 0, 0, 0, 0, 0, -2.0, 0, 0, 0, 1.5, 0],
            no_lyrics=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        )
        assert acm.sequence_album(features, acm.OPENER_INTRO)[0] == (6, acm.ROLE_OPENER)

    def test_the_middle_declines_and_inward_lyrics_sink_toward_the_end(self):
        features = self._features(
            [0.0, 2.0, 1.8, 0.5, 0.5, 0.2, 0.1, -0.1, -0.2, -0.3, -2.5, 1.6],
            inward=[0, 0, 0, -1.0, 1.0, 0, 0, 0, 0, 0, 0, 0],
        )
        middle = [index for index, role in acm.sequence_album(features, acm.OPENER_BANG) if role == acm.ROLE_TRACK]
        assert middle.index(3) < middle.index(4)
        assert middle.index(5) < middle.index(9)

    def test_two_calm_tracks_never_sit_together_and_none_touches_a_calm_closer(self):
        features = self._features([0.3, 1.5, 1.2, 0.4, 0.2, 0.1, -0.9, -1.0, -1.1, 0.0, -2.0, 0.6])
        order = [index for index, _role in acm.sequence_album(features, acm.OPENER_BANG)]
        calm = {6, 7, 8, 10}
        assert all(not (a in calm and b in calm) for a, b in zip(order, order[1:]))

    def test_calm_tracks_with_nowhere_to_go_are_kept_not_dropped(self):
        assert acm.spread_calm_tracks(['s1'], ['c1', 'c2', 'c3'], True) == ['s1', 'c1', 'c2', 'c3']
        assert acm.spread_calm_tracks(['s1', 's2', 's3'], ['c1'], True) == ['s1', 's2', 'c1', 's3']
        assert acm.spread_calm_tracks(['s1', 's2'], ['c1'], False) == ['s1', 's2', 'c1']
        assert acm.spread_calm_tracks(['s1', 's2'], [], True) == ['s1', 's2']

    def test_two_tracks_of_one_artist_are_pulled_apart_in_the_middle(self):
        features = self._features([0.0, 2.0, 1.8, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, -2.5, 1.6])
        authors = ['x', 's1', 's2', 'same', 'same', 'same', 'a', 'b', 'c', 'd', 'z', 'o']
        order = [index for index, _role in acm.sequence_album(features, acm.OPENER_BANG, authors)]
        names = [authors[index] for index in order]
        assert all(not (a == b == 'same') for a, b in zip(names, names[1:])), names
        assert sorted(order) == list(range(12))

    def test_a_dominant_artist_is_interleaved_not_left_in_one_block(self):
        authors = ['x', 'y', 'z', 'n', 'n', 'n', 'n', 'n']
        ordered = acm.separate_artists(list(range(8)), authors, set(), 'q', 'w')
        assert sorted(ordered) == list(range(8))
        assert acm.artist_clashes(ordered, authors, 'q', 'w') <= 1, [authors[i] for i in ordered]

    def test_the_middle_never_ends_on_the_closers_artist_nor_starts_on_the_singles(self):
        authors = ['b', 'j', 'd', 'g', 'n', 'k', 'n', 'n']
        ordered = acm.separate_artists(list(range(8)), authors, set(), 'n', 'n')
        assert sorted(ordered) == list(range(8))
        assert acm.artist_clashes(ordered, authors, 'n', 'n') == 0, [authors[i] for i in ordered]
        assert authors[ordered[0]] != 'n' != authors[ordered[-1]]

    def test_artist_separation_never_pushes_two_calm_tracks_together(self):
        authors = ['n', 'n', 'x', 'y']
        ordered = acm.separate_artists([0, 1, 2, 3], authors, {0, 2}, 'q')
        assert sorted(ordered) == [0, 1, 2, 3]
        assert acm.artist_clashes(ordered, authors, 'q', None) == 0
        assert all(not (a in {0, 2} and b in {0, 2}) for a, b in zip(ordered, ordered[1:])), ordered
        assert acm.separate_artists([0, 1], ['n', 'n'], set(), 'n') == [0, 1]
        assert acm.artist_clashes(acm.separate_artists([0, 1, 2], ['n', 'x', 'y'], set(), 'n'), ['n', 'x', 'y'], 'n', None) == 0
        assert acm.separate_artists([0, 1], ['', ''], set(), '') == [0, 1]

    def test_clashes_count_the_boundaries_too(self):
        assert acm.artist_clashes([0, 1], ['n', 'n'], 'n', 'n') == 3
        assert acm.artist_clashes([0, 1], ['a', 'b'], 'n', 'n') == 0
        assert acm.artist_clashes([0, 1], ['', ''], '', None) == 0

    def test_a_handful_of_tracks_gets_no_roles(self):
        order = acm.sequence_album(self._features([0.1, 0.2, 0.3]), acm.OPENER_BANG)
        assert [role for _index, role in order] == [acm.ROLE_TRACK] * 3

    @pytest.mark.parametrize('genres, style', [
        (['Hip-Hop', 'Hip-Hop', 'rock'], acm.OPENER_INTRO),
        (['metal'] * 3, acm.OPENER_INTRO),
        (['rock', 'pop', 'rock'], acm.OPENER_BANG),
        (['', ''], acm.OPENER_BANG),
    ])
    def test_the_dominant_genre_decides_the_opener_style(self, genres, style):
        assert acm.opener_style(genres) == style


@pytest.fixture
def library(monkeypatch):
    rng = np.random.default_rng(21)
    vectors = _cluster(rng, 30, 0.05) + _cluster(rng, 90, 0.13)
    tracks = [_track(f'fp_{i:03d}', v) for i, v in enumerate(vectors)]
    by_id = {track['item_id']: track for track in tracks}
    monkeypatch.setattr(acm, '_axis_index', lambda: {})
    monkeypatch.setattr(acm, 'load_tracks', lambda ids, axis_index=None: [by_id[i] for i in ids if i in by_id])
    monkeypatch.setattr(acm, 'candidate_pool', lambda query, axis_index=None, clap_query=None, first=0: list(tracks))
    monkeypatch.setattr(config, 'ALBUM_CREATION_TRACKS', 12)
    monkeypatch.setattr(config, 'ALBUM_CREATION_COHESION', 0.90)
    monkeypatch.setattr(config, 'MAX_SONGS_PER_ARTIST', 3)
    return tracks


class TestCreateAlbum:
    def _create(self, seed_type=acm.SEED_SONG, **kw):
        kw.setdefault('rng', np.random.default_rng(4))
        kw.setdefault('today', date(2026, 6, 1))
        return acm.create_album(seed_type, **kw)

    def test_a_song_seed_stays_in_its_album(self, library):
        album = self._create(item_id='fp_000')
        ids = [track['item_id'] for track in album['tracks']]
        assert 'fp_000' in ids and len(ids) == 12 == len(set(ids))
        assert album['seed'] == {'type': 'song', 'label': 'Song fp_000 - Artist fp_000'}
        assert album['suggested_name'] == 'Song fp_000 - The Album'

    def test_the_rows_are_json_ready_and_numbered_in_running_order(self, library):
        album = self._create(item_id='fp_000')
        assert [track['slot'] for track in album['tracks']] == list(range(1, 13))
        assert album['tracks'][0]['role'] == acm.ROLE_OPENER
        assert album['tracks'][-1]['role'] == acm.ROLE_CLOSER
        assert all('vector' not in track and 'moods' not in track for track in album['tracks'])
        stats = album['stats']
        assert stats['tracks'] == 12 and stats['minutes'] == 44 and stats['artists'] == 12
        assert stats['cohesion'] == pytest.approx(0.90, abs=0.04)
        assert stats['target_cohesion'] == 0.9

    def test_tracks_of_the_seeds_era_are_preferred_while_enough_remain(self, library):
        library[0]['year'] = 1972
        for position, track in enumerate(library[1:], start=1):
            track['year'] = 1975 if position % 2 else 2020
        album = self._create(item_id='fp_000')
        assert len(album['tracks']) == 12
        assert all(abs(track['year'] - 1972) <= acm.ERA_WINDOW_YEARS for track in album['tracks'])

    def test_a_lonely_era_falls_back_to_every_era_instead_of_a_short_album(self, library):
        library[0]['year'] = 1950
        for track in library[1:]:
            track['year'] = 2020
        assert len(self._create(item_id='fp_000')['tracks']) == 12

    def test_an_unknown_year_never_excludes_a_track(self, library):
        library[0]['year'] = 1972
        for track in library[1:]:
            track['year'] = None
        assert len(self._create(item_id='fp_000')['tracks']) == 12
        assert acm.known_year(0) is None and acm.known_year(1888) is None and acm.known_year(1999) == 1999

    def _other_voice_next_door(self, library, seed_is_sung):
        rng = np.random.default_rng(33)
        for track in library[:60]:
            track['has_lyrics'] = seed_is_sung
        for track, vector in zip(library[60:], _leaning_cluster(rng, 60, 0.05, 0.45)):
            track['vector'] = vector
            track['clap'] = vector.copy()
            track['has_lyrics'] = not seed_is_sung
        return {track['item_id'] for track in library[60:]}

    def _from_next_door(self, next_door):
        album = self._create(item_id='fp_000')
        assert len(album['tracks']) == 12
        return sum(1 for track in album['tracks'] if track['item_id'] in next_door)

    def test_a_sung_album_lets_in_only_a_couple_of_instrumentals(self, library, monkeypatch):
        next_door = self._other_voice_next_door(library, True)
        assert self._from_next_door(next_door) <= acm.OTHER_VOICE_CAP
        monkeypatch.setattr(acm, 'OTHER_VOICE_CAP', 12)
        assert self._from_next_door(next_door) > 2

    def test_an_instrumental_album_lets_in_only_a_couple_of_sung_tracks(self, library, monkeypatch):
        next_door = self._other_voice_next_door(library, False)
        assert self._from_next_door(next_door) <= acm.OTHER_VOICE_CAP
        monkeypatch.setattr(acm, 'OTHER_VOICE_CAP', 12)
        assert self._from_next_door(next_door) > 2

    def test_a_sung_seed_whose_lyrics_are_missing_still_gets_a_sung_album(self, library):
        next_door = self._other_voice_next_door(library, True)
        library[0]['has_lyrics'] = False
        assert self._from_next_door(next_door) <= acm.OTHER_VOICE_CAP

    def test_the_voice_vote_follows_dclap_and_not_musicnn(self, library, monkeypatch):
        monkeypatch.setattr(config, 'ALBUM_CREATION_MUSICNN_SHARE', 0.5)
        rng = np.random.default_rng(17)
        sung = _leaning_cluster(rng, 60, 0.05, 0.45)
        for position, track in enumerate(library):
            track['has_lyrics'] = position >= 60
            track['clap'] = sung[position - 60].copy() if position >= 60 else track['vector'].copy()
        library[0]['has_lyrics'] = False
        album = self._create(item_id='fp_000')
        sung_tracks = [track for track in album['tracks'] if track['item_id'] >= 'fp_060']
        assert len(album['tracks']) == 12 and len(sung_tracks) <= acm.OTHER_VOICE_CAP

    def test_untagged_files_stay_out_while_tagged_tracks_remain(self, library):
        for track in library[1:60]:
            track['author'] = 'Unknown Artist'
        album = self._create(item_id='fp_000')
        assert len(album['tracks']) == 12
        assert all(track['author'] != 'Unknown Artist' for track in album['tracks'])

    def test_a_library_of_untagged_files_is_not_held_to_the_artist_cap(self, library):
        for track in library:
            track['author'] = 'Unknown Artist'
        assert len(self._create(item_id='fp_000')['tracks']) == 12

    def _two_subjects(self, library, monkeypatch, share=0.5):
        monkeypatch.setattr(config, 'ALBUM_CREATION_LYRIC_SHARE', share)
        rng = np.random.default_rng(12)
        half = len(library) // 2
        about = _cluster(rng, half, 0.05, axis=0) + _cluster(rng, len(library) - half, 0.05, axis=1)
        for track, subject in zip(library, about):
            track['lyrics'] = np.asarray(subject[:8], dtype=np.float32)
        return half

    def _other_subject(self, library, album):
        half = len(library) // 2
        return [track for track in album['tracks'] if track['item_id'] >= f'fp_{half:03d}']

    def test_songs_about_the_seeds_subject_are_preferred(self, library, monkeypatch):
        self._two_subjects(library, monkeypatch)
        album = self._create(item_id='fp_000')
        assert len(album['tracks']) == 12
        assert not self._other_subject(library, album)

    def test_the_lyric_preference_gives_way_when_too_few_songs_have_lyrics(self, library, monkeypatch):
        self._two_subjects(library, monkeypatch)
        for track in library[20:]:
            track['lyrics'] = None
        album = self._create(item_id='fp_000')
        assert len(album['tracks']) == 12
        assert self._other_subject(library, album)

    def test_a_seed_without_lyrics_leaves_the_album_to_the_sound(self, library, monkeypatch):
        self._two_subjects(library, monkeypatch)
        library[0]['lyrics'] = None
        library[0]['has_lyrics'] = False
        assert self._other_subject(library, self._create(item_id='fp_000'))

    def test_the_lyric_share_can_be_turned_off(self, library, monkeypatch):
        self._two_subjects(library, monkeypatch, share=1.0)
        assert self._other_subject(library, self._create(item_id='fp_000'))

    def test_preferences_apply_in_turn_and_only_while_enough_tracks_remain(self):
        tracks = [{'n': n} for n in range(10)]
        assert acm.keep_preferred(tracks, 3, (lambda t: t['n'] < 5, lambda t: t['n'] < 1)) == tracks[:5]
        assert acm.keep_preferred(tracks, 3, (lambda t: t['n'] > 99,)) == tracks

    @pytest.mark.parametrize('share, kept', [(1.0, 20), (0.5, 10), (0.25, 5)])
    def test_the_lyric_share_keeps_that_many_of_the_candidates(self, share, kept, monkeypatch):
        monkeypatch.setattr(config, 'ALBUM_CREATION_LYRIC_SHARE', share)
        rng = np.random.default_rng(9)
        about = [_unit(rng.standard_normal(8)) for _ in range(20)]
        tracks = [_track(f'id{i}', np.ones(DIM), lyrics=about[i]) for i in range(20)]
        assert len(acm.keep_lyric_neighbours(tracks, about[0], 5)) == kept

    def test_the_lyric_share_never_shortens_the_album(self, monkeypatch):
        monkeypatch.setattr(config, 'ALBUM_CREATION_LYRIC_SHARE', 0.25)
        rng = np.random.default_rng(10)
        tracks = [_track(f'id{i}', np.ones(DIM), lyrics=_unit(rng.standard_normal(8))) for i in range(20)]
        assert acm.keep_lyric_neighbours(tracks, tracks[0]['lyrics'], 30) == tracks
        assert acm.keep_lyric_neighbours(tracks, None, 5) == tracks
        silent = [_track(f'id{i}', np.ones(DIM), lyrics=None) for i in range(20)]
        assert acm.keep_lyric_neighbours(silent, tracks[0]['lyrics'], 5) == silent

    def test_the_same_share_of_the_tracks_without_lyrics_is_kept(self, monkeypatch):
        monkeypatch.setattr(config, 'ALBUM_CREATION_LYRIC_SHARE', 0.5)
        rng = np.random.default_rng(11)
        sung = [_track(f'sung{i}', np.ones(DIM), lyrics=_unit(rng.standard_normal(8))) for i in range(20)]
        silent = [_track(f'silent{i}', np.ones(DIM), lyrics=None) for i in range(20)]
        kept = acm.keep_lyric_neighbours(sung + silent, sung[0]['lyrics'], 5)
        assert sum(1 for track in kept if track['lyrics'] is None) == 10
        assert sum(1 for track in kept if track['lyrics'] is not None) == 10
        assert [track['item_id'] for track in kept if track['lyrics'] is None] == [f'silent{i}' for i in range(10)]

    def test_without_an_artist_seed_the_cap_holds_even_for_the_seed_song_artist(self, library):
        for track in library:
            track['author'] = 'Nina Simone'
        assert len(self._create(item_id='fp_000')['tracks']) == 3

    def test_live_cuts_and_holiday_songs_never_reach_the_album(self, library):
        for track in library[1:60]:
            track['title'] = track['title'] + ' (Live)'
        for track in library[60:100]:
            track['album'] = 'Christmas Classics'
        album = self._create(item_id='fp_000')
        assert all('(Live)' not in track['title'] for track in album['tracks'])
        assert all(track['album'] != 'Christmas Classics' for track in album['tracks'])

    def test_december_and_a_holiday_seed_both_let_holiday_songs_in(self, library):
        for track in library[1:]:
            track['album'] = 'Christmas Classics'
        assert len(self._create(item_id='fp_000', today=date(2026, 12, 5))['tracks']) == 12
        library[0]['title'] = 'Jingle Bells'
        assert len(self._create(item_id='fp_000')['tracks']) == 12

    def test_a_library_of_short_songs_still_gets_its_album(self, library):
        for track in library:
            track['duration'] = 45.0
        assert len(self._create(item_id='fp_000')['tracks']) == 12

    def test_short_and_epic_tracks_stay_out_while_enough_others_remain(self, library):
        for track in library[1:40]:
            track['duration'] = 45.0
        for track in library[40:60]:
            track['duration'] = 1900.0
        album = self._create(item_id='fp_000')
        assert all(acm.has_album_length(track['duration']) for track in album['tracks'])

    def test_the_seed_song_is_never_filtered_out(self, library):
        library[0]['title'] = 'Seed (Live)'
        assert 'fp_000' in [track['item_id'] for track in self._create(item_id='fp_000')['tracks']]

    def test_an_unknown_seed_or_a_desert_around_it_is_a_not_found(self, library, monkeypatch):
        with pytest.raises(acm.AlbumSeedNotFound):
            self._create(item_id='fp_missing')
        monkeypatch.setattr(acm, 'candidate_pool', lambda query, axis_index=None, clap_query=None, first=0: [])
        with pytest.raises(acm.AlbumSeedNotFound):
            self._create(item_id='fp_000')

    @pytest.mark.parametrize('seed_type, kw', [
        (acm.SEED_SONG, {}), (acm.SEED_TEXT, {}), ('playlist', {'item_id': 'x'}),
    ])
    def test_a_missing_or_unknown_seed_is_a_seed_error(self, library, seed_type, kw):
        with pytest.raises(acm.AlbumSeedError):
            self._create(seed_type, **kw)


class TestThePool:
    def test_it_walks_outward_through_the_similar_song_engine_without_its_artist_cap(self, monkeypatch):
        rng = np.random.default_rng(8)
        vectors = {f'id{i}': _unit(rng.standard_normal(DIM)) for i in range(120)}
        calls = []

        def engine(vector, n, eliminate_duplicates):
            calls.append((n, eliminate_duplicates))
            start = len(calls) * 5
            return [{'item_id': f'id{i}', 'distance': 0.1} for i in range(start - 5, start + 3)]

        monkeypatch.setattr(acm, 'ensure_ivf_index_loaded', lambda: True)
        monkeypatch.setattr(acm, 'find_nearest_neighbors_by_vector', engine)
        monkeypatch.setattr(acm, 'load_tracks', lambda ids, axis_index=None: [_track(i, vectors[i]) for i in ids])
        pool = acm.candidate_pool(vectors['id0'], {})
        assert calls[0] == (acm.POOL_FIRST_QUERY, False)
        assert len(calls) == 1 + acm.POOL_HOPS * acm.POOL_FRONTIER
        assert all(call == (acm.POOL_HOP_QUERY, False) for call in calls[1:])
        ids = [track['item_id'] for track in pool]
        assert len(ids) == len(set(ids)) > 8

    def test_a_missing_index_is_a_runtime_error_not_an_empty_album(self, monkeypatch):
        monkeypatch.setattr(acm, 'ensure_ivf_index_loaded', lambda: False)
        with pytest.raises(RuntimeError):
            acm.candidate_pool(np.ones(DIM), {})

    def _engine(self, monkeypatch, vectors):
        monkeypatch.setattr(acm, 'ensure_ivf_index_loaded', lambda: True)
        monkeypatch.setattr(
            acm, 'find_nearest_neighbors_by_vector',
            lambda vector, n, eliminate_duplicates: [{'item_id': f'audio{i}'} for i in range(6)],
        )
        monkeypatch.setattr(
            acm, 'load_tracks',
            lambda ids, axis_index=None: [_track(i, vectors.get(i, np.ones(DIM))) for i in ids],
        )

    def test_the_dclap_index_adds_its_own_neighbours_to_the_pool(self, monkeypatch):
        self._engine(monkeypatch, {})
        asked = []

        def clap_ids(vector, count):
            asked.append(count)
            return ['clap1', 'clap2']

        monkeypatch.setattr(acm, 'clap_song_ids', clap_ids)
        ids = [track['item_id'] for track in acm.candidate_pool(np.ones(DIM), {}, np.ones(4))]
        assert asked == [acm.POOL_CLAP_QUERY]
        assert {'clap1', 'clap2'} <= set(ids) and 'audio0' in ids

    def test_a_pool_without_a_dclap_query_asks_only_the_similar_song_engine(self, monkeypatch):
        self._engine(monkeypatch, {})
        calls = []
        monkeypatch.setattr(acm, 'clap_song_ids', lambda vector, count: calls.append(vector) or [])
        ids = [track['item_id'] for track in acm.candidate_pool(np.ones(DIM), {}, None)]
        assert calls == [None] and 'audio0' in ids

    def test_dclap_ids_are_scoped_to_the_server_before_they_join_the_pool(self, monkeypatch):
        monkeypatch.setattr(config, 'ALBUM_CREATION_MUSICNN_SHARE', 0.5)
        monkeypatch.setattr(acm, 'available_ids', lambda ids: [i for i in ids if i != 'elsewhere'])
        module = MagicMock()
        module.is_clap_cache_loaded.return_value = True
        module.search_by_embedding.return_value = [{'item_id': 'here'}, {'item_id': 'elsewhere'}]
        with patch.dict('sys.modules', {'tasks.clap_text_search': module}):
            assert acm.clap_song_ids(np.ones(4), 5) == ['here']
        assert module.search_by_embedding.call_args.kwargs == {'limit': 5}

    @pytest.mark.parametrize('loaded, fails', [(False, False), (True, True)])
    def test_an_unavailable_dclap_index_leaves_the_musicnn_pool_alone(self, monkeypatch, loaded, fails):
        monkeypatch.setattr(config, 'ALBUM_CREATION_MUSICNN_SHARE', 0.5)
        module = MagicMock()
        module.is_clap_cache_loaded.return_value = loaded
        module.search_by_embedding.side_effect = RuntimeError('index gone') if fails else None
        with patch.dict('sys.modules', {'tasks.clap_text_search': module}):
            assert acm.clap_song_ids(np.ones(4), 5) == []

    def test_dclap_is_not_asked_at_all_when_its_share_is_zero(self, monkeypatch):
        monkeypatch.setattr(config, 'ALBUM_CREATION_MUSICNN_SHARE', 1.0)
        with patch.dict('sys.modules', {'tasks.clap_text_search': MagicMock()}):
            assert acm.clap_song_ids(np.ones(4), 5) == []


class TestTheMixedSpace:
    def _rows(self, share, audio, clap):
        with patch.object(config, 'ALBUM_CREATION_MUSICNN_SHARE', share):
            return acm.mixed_rows(audio, clap)

    def test_a_cosine_in_the_mixed_space_is_the_weighted_average_of_both(self):
        rng = np.random.default_rng(1)
        audio = [_unit(rng.standard_normal(DIM)) for _ in range(2)]
        clap = [_unit(rng.standard_normal(DIM)) for _ in range(2)]
        for share in (0.0, 0.3, 0.5, 1.0):
            rows = self._rows(share, audio, clap)
            expected = share * float(audio[0] @ audio[1]) + (1 - share) * float(clap[0] @ clap[1])
            assert float(rows[0] @ rows[1]) == pytest.approx(expected, abs=1e-6)
            assert float(np.linalg.norm(rows[0])) == pytest.approx(1.0, abs=1e-6)

    def test_without_dclap_vectors_the_space_is_musicnn_alone(self):
        rng = np.random.default_rng(2)
        audio = [_unit(rng.standard_normal(DIM)) for _ in range(3)]
        rows = self._rows(0.5, audio, None)
        assert rows.shape == (3, DIM)
        assert float(rows[0] @ rows[1]) == pytest.approx(float(audio[0] @ audio[1]), abs=1e-6)

    def test_a_share_outside_zero_to_one_is_clamped(self, monkeypatch):
        monkeypatch.setattr(config, 'ALBUM_CREATION_MUSICNN_SHARE', 4.0)
        assert acm.musicnn_share() == 1.0
        monkeypatch.setattr(config, 'ALBUM_CREATION_MUSICNN_SHARE', -2.0)
        assert acm.musicnn_share() == 0.0

    def _tracks(self, count, missing=()):
        rng = np.random.default_rng(3)
        tracks = [_track(f'id{i}', _unit(rng.standard_normal(DIM))) for i in range(count)]
        for i in missing:
            tracks[i]['clap'] = None
        return tracks

    def test_candidates_without_a_dclap_vector_drop_out_while_enough_remain(self, monkeypatch):
        monkeypatch.setattr(config, 'ALBUM_CREATION_MUSICNN_SHARE', 0.5)
        kept, clap = acm.with_clap(self._tracks(20, missing=(5, 7)), 1, 12)
        assert [track['item_id'] for track in kept] == [f'id{i}' for i in range(20) if i not in (5, 7)]
        assert len(clap) == len(kept)

    def test_too_few_dclap_vectors_build_the_album_on_musicnn_alone(self, monkeypatch):
        monkeypatch.setattr(config, 'ALBUM_CREATION_MUSICNN_SHARE', 0.5)
        tracks = self._tracks(20, missing=range(1, 15))
        kept, clap = acm.with_clap(tracks, 1, 12)
        assert kept == tracks and clap is None

    def test_a_seed_without_a_dclap_vector_builds_on_musicnn_alone(self, monkeypatch):
        monkeypatch.setattr(config, 'ALBUM_CREATION_MUSICNN_SHARE', 0.5)
        tracks = self._tracks(20, missing=(0,))
        kept, clap = acm.with_clap(tracks, 1, 12)
        assert kept == tracks and clap is None

    def test_the_dclap_half_can_veto_a_track_the_musicnn_half_likes(self, monkeypatch):
        monkeypatch.setattr(config, 'ALBUM_CREATION_MUSICNN_SHARE', 0.5)
        rng = np.random.default_rng(4)
        tracks = [_track(f'id{i}', v) for i, v in enumerate(_cluster(rng, 30, 0.02))]
        twin, stranger = tracks[1], tracks[2]
        stranger['clap'] = _unit(-stranger['clap'] + 0.01 * rng.standard_normal(DIM))
        _kept, clap = acm.with_clap(tracks, 1, 12)
        units = acm.mixed_rows([track['vector'] for track in tracks], clap)
        assert float(units[0] @ units[1]) > float(units[0] @ units[2])
        assert twin['clap'] is not None and stranger['clap'] is not None


class TestDatabaseReads:
    def _db(self, rows):
        cur = MagicMock()
        cur.fetchall.return_value = rows
        db = MagicMock()
        db.cursor.return_value = cur
        return db, cur

    def test_tracks_come_back_in_the_asked_order_and_a_foreign_embedding_is_dropped(self, monkeypatch):
        monkeypatch.setattr(config, 'EMBEDDING_DIMENSION', DIM)
        good = np.arange(DIM, dtype=np.float32).tobytes()
        row = {'title': 'T', 'author': 'A', 'album': 'Al', 'album_artist': 'A', 'year': 1999, 'duration': 200,
               'tempo': 100.0, 'energy': 0.4, 'mood_vector': 'jazz:0.9,rock:0.1', 'other_features': '',
               'axis_vector': None, 'clap_embedding': None, 'lyrics_embedding': None}
        db, _cur = self._db([
            dict(row, item_id='b', embedding=good),
            dict(row, item_id='a', embedding=good),
            dict(row, item_id='c', embedding=b'\x00' * 12),
        ])
        with patch('database.get_db', return_value=db):
            tracks = acm.load_tracks(['a', 'b', 'c', 'a'], {})
        assert [track['item_id'] for track in tracks] == ['a', 'b']
        assert tracks[0]['top_genre'] == 'jazz' and tracks[0]['has_lyrics'] is False
        assert tracks[0]['vector'].shape == (DIM,)
        assert tracks[0]['clap'] is None

    def test_a_dclap_vector_is_read_when_it_has_the_right_size(self, monkeypatch):
        monkeypatch.setattr(config, 'EMBEDDING_DIMENSION', DIM)
        monkeypatch.setattr(config, 'CLAP_EMBEDDING_DIMENSION', 4)
        row = {'title': 'T', 'author': 'A', 'album': 'Al', 'album_artist': 'A', 'year': 1999, 'duration': 200,
               'tempo': 100.0, 'energy': 0.4, 'mood_vector': '', 'other_features': '', 'axis_vector': None,
               'lyrics_embedding': None, 'embedding': np.arange(DIM, dtype=np.float32).tobytes()}
        db, _cur = self._db([
            dict(row, item_id='a', clap_embedding=np.arange(4, dtype=np.float32).tobytes()),
            dict(row, item_id='b', clap_embedding=b'\x00' * 8),
            dict(row, item_id='c', clap_embedding=None),
        ])
        with patch('database.get_db', return_value=db):
            tracks = acm.load_tracks(['a', 'b', 'c'], {})
        assert tracks[0]['clap'].shape == (4,)
        assert tracks[1]['clap'] is None and tracks[2]['clap'] is None

    def test_no_ids_means_no_query(self):
        with patch('database.get_db') as get_db:
            assert acm.load_tracks([]) == []
        get_db.assert_not_called()

    def test_the_weekly_seeds_are_scoped_to_the_bound_server_and_skip_live_and_holiday_titles(self):
        db, cur = self._db([
            ('short', 'Short Song', 'Album', 30.0), ('a', 'Song', 'Album', 200.0),
            ('b', 'Song (Live)', 'Album', 200.0), ('c', 'Silent Night', 'Carols', 200.0),
        ])
        with (
            patch('database.get_db', return_value=db),
            patch('tasks.mediaserver.context.active_server_id', return_value='srv-2'),
            patch('tasks.mediaserver.registry.get_default_server_id', return_value='srv-1'),
        ):
            assert acm.weekly_seed_ids(today=date(2026, 6, 1)) == ['a', 'short']
            assert acm.weekly_seed_ids(today=date(2026, 12, 1)) == ['a', 'c', 'short']
        sql, params = cur.execute.call_args[0]
        assert 'ORDER BY random()' in sql and 'track_server_map' in sql
        assert params == ('srv-2', False, acm.WEEKLY_SEED_SAMPLE)


class TestAlbumOfTheWeek:
    def _run(self):
        with patch('tasks.mediaserver.registry.servers_for_scope', return_value=[None]):
            return acm.run_album_of_the_week_task(server_scope='all')

    def _album(self):
        return {'seed': {'type': 'song', 'label': 'S - A'}, 'tracks': [{'item_id': 'c'}, {'item_id': 'a'}, {'item_id': 'b'}]}

    def test_no_flag_switches_the_scheduled_run_off(self, monkeypatch):
        monkeypatch.setattr(config, 'LYRICS_ENABLED', False)
        with (
            patch.object(acm, 'weekly_seed_ids', return_value=['seed']) as seeds,
            patch.object(acm, 'create_album', return_value=self._album()) as create,
        ):
            assert acm.create_album_of_the_week() == self._album()
        seeds.assert_called_once()
        create.assert_called_once_with(acm.SEED_SONG, item_id='seed')

    def test_without_lyric_themes_the_album_is_still_built_on_audio_alone(self, library):
        album = acm.create_album(acm.SEED_SONG, item_id='fp_000', rng=np.random.default_rng(4), today=date(2026, 6, 1))
        assert len(album['tracks']) == 12 and album['tracks'][-1]['role'] == acm.ROLE_CLOSER

    def test_a_seed_that_cannot_grow_into_an_album_gives_way_to_the_next(self, monkeypatch):
        monkeypatch.setattr(config, 'LYRICS_ENABLED', True)
        outcomes = {'bad': acm.AlbumSeedNotFound('no'), 'good': self._album()}

        def create(seed_type, item_id=None):
            if isinstance(outcomes[item_id], Exception):
                raise outcomes[item_id]
            return outcomes[item_id]

        with patch.object(acm, 'weekly_seed_ids', return_value=['bad', 'good']), patch.object(acm, 'create_album', create):
            assert acm.create_album_of_the_week() == self._album()
        with patch.object(acm, 'weekly_seed_ids', return_value=[]):
            assert acm.create_album_of_the_week() is None

    def test_the_run_upserts_one_fixed_playlist_name_in_running_order(self):
        with (
            patch.object(acm, 'create_album_of_the_week', return_value=self._album()),
            patch('tasks.mediaserver.create_or_replace_playlist', return_value={'Id': 'pl'}) as upsert,
            patch('tasks.ivf_manager.create_playlist_from_ids') as legacy,
        ):
            summary = self._run()
        upsert.assert_called_once_with(config.ALBUM_OF_THE_WEEK_PLAYLIST_NAME, ['c', 'a', 'b'])
        legacy.assert_not_called()
        assert summary['playlists_created'] == 1 and summary['failed'] == []

    def test_nothing_built_keeps_the_previous_playlist(self):
        with (
            patch.object(acm, 'create_album_of_the_week', return_value=None),
            patch('tasks.mediaserver.create_or_replace_playlist') as upsert,
            patch('tasks.ivf_manager.create_playlist_from_ids') as legacy,
        ):
            summary = self._run()
        upsert.assert_not_called()
        legacy.assert_not_called()
        assert summary['playlists_created'] == 0

    def test_a_backend_without_upsert_falls_back_to_a_dated_playlist(self):
        with (
            patch.object(acm, 'create_album_of_the_week', return_value=self._album()),
            patch('tasks.mediaserver.create_or_replace_playlist', side_effect=NotImplementedError),
            patch('tasks.ivf_manager.create_playlist_from_ids', return_value='legacy') as legacy,
        ):
            self._run()
        assert legacy.call_args[0][0].startswith('Album of the Week (Cron ')
        assert legacy.call_args[0][1] == ['c', 'a', 'b']

    def test_it_raises_only_when_every_server_failed(self):
        with patch.object(acm, 'create_album_of_the_week', side_effect=ValueError('boom')):
            with pytest.raises(RuntimeError):
                self._run()

    def test_a_cancelled_claim_does_no_work(self):
        import taskqueue
        from taskqueue import TaskCancelled

        with (
            patch.object(taskqueue, 'current_task_id', return_value='album-cancelled'),
            patch('tasks.task_run._read_task_statuses', return_value={}),
            patch('tasks.task_run.save_task_status') as save,
            patch('tasks.mediaserver.registry.servers_for_scope') as servers,
        ):
            with pytest.raises(TaskCancelled):
                acm.run_album_of_the_week_task(server_scope='all')
        save.assert_not_called()
        servers.assert_not_called()


class TestTheDescriptionSeed:
    def test_a_genre_word_is_read_from_the_analysis_tags(self, monkeypatch):
        monkeypatch.setattr(config, 'MOOD_LABELS', ['jazz', 'heavy metal', 'rock'])
        monkeypatch.setattr(acm, 'concept_vocabulary', lambda: ['trumpet', 'piano'])
        assert acm.named_attributes('jazz with trumpet') == (['jazz'], ['trumpet'])
        assert acm.named_attributes('heavy metal with piano') == (['heavy metal'], ['piano'])
        assert acm.named_attributes('something entirely else') == ([], [])

    def test_a_longer_tag_wins_over_the_word_inside_it(self, monkeypatch):
        monkeypatch.setattr(config, 'MOOD_LABELS', ['metal', 'heavy metal'])
        monkeypatch.setattr(acm, 'concept_vocabulary', lambda: [])
        assert acm.named_attributes('heavy metal album') == (['heavy metal'], [])

    def test_the_ranking_is_the_product_of_every_named_attribute(self, monkeypatch):
        monkeypatch.setattr(acm, 'concept_vocabulary', lambda: [])
        tracks = [
            _track('a', np.ones(DIM), mood_vector='jazz:0.9,rock:0.1'),
            _track('b', np.ones(DIM), mood_vector='jazz:0.2,rock:0.9'),
            _track('c', np.ones(DIM), mood_vector='jazz:0.5,rock:0.5'),
        ]
        product = acm.attribute_scores(tracks, ['jazz'], [])
        assert list(np.argsort(-product)) == [0, 2, 1]
        assert acm.attribute_scores(tracks, [], []) is None

    def test_an_instrument_is_scored_by_the_dclap_concepts(self):
        tracks = [_track(name, np.ones(DIM)) for name in ('a', 'b', 'c')]
        module = MagicMock()
        module.concept_scores.return_value = {'trumpet': np.array([0.1, 0.9, 0.5])}
        with patch.dict('sys.modules', {'tasks.clap_steering': module}):
            product = acm.attribute_scores(tracks, [], ['trumpet'])
        assert list(np.argsort(-product)) == [1, 2, 0]
        assert module.concept_scores.call_args[0][1] == ['trumpet']

    def test_a_track_without_a_dclap_vector_turns_the_instrument_off(self):
        tracks = [_track('a', np.ones(DIM)), _track('b', np.ones(DIM), clap=None)]
        assert acm.attribute_scores(tracks, [], ['trumpet']) is None

    def test_only_the_best_ranked_candidates_stay(self, monkeypatch):
        monkeypatch.setattr(acm, 'concept_vocabulary', lambda: [])
        monkeypatch.setattr(acm, 'ATTRIBUTE_KEEP', 4)
        tracks = [_track('id%02d' % n, np.ones(DIM), mood_vector='jazz:%.2f' % (n / 20.0))
                  for n in range(20)]
        kept = acm.keep_named_attributes(tracks, ['jazz'], [], 2)
        assert [track['item_id'] for track in kept] == ['id16', 'id17', 'id18', 'id19']
        assert acm.keep_named_attributes(tracks, [], [], 2) == tracks

    def test_nothing_is_dropped_when_the_pool_is_already_short(self, monkeypatch):
        monkeypatch.setattr(acm, 'concept_vocabulary', lambda: [])
        tracks = [_track('id%d' % n, np.ones(DIM)) for n in range(5)]
        assert acm.keep_named_attributes(tracks, ['jazz'], [], 12) == tracks

    @pytest.mark.parametrize('query, problem', [
        ('', 'a few words'), ('a b c d e f g h i j k l m', 'at most'),
    ])
    def test_an_empty_or_rambling_description_is_refused(self, query, problem):
        with pytest.raises(acm.AlbumSeedError) as error:
            acm.resolve_seed(acm.SEED_TEXT, query=query)
        assert problem in str(error.value)

    def test_the_description_becomes_the_seed_label_and_the_album_name(self, monkeypatch):
        monkeypatch.setattr(acm, '_text_embedding', lambda query, steering=None: np.ones(4, dtype=np.float32))
        monkeypatch.setattr(config, 'MOOD_LABELS', ['jazz'])
        monkeypatch.setattr(acm, 'concept_vocabulary', lambda: ['trumpet'])
        seed = acm.resolve_seed(acm.SEED_TEXT, query='  jazz   with trumpet ')
        assert seed['label'] == 'jazz with trumpet'
        assert seed['name'] == 'Jazz with trumpet'
        assert seed['query'] is None and seed['required'] == []
        assert seed['tags'] == ['jazz'] and seed['concepts'] == ['trumpet']

    def test_an_unloaded_dclap_index_is_a_seed_error_not_a_crash(self):
        module = MagicMock()
        module.is_clap_cache_loaded.return_value = False
        with patch.dict('sys.modules', {'tasks.clap_text_search': module}):
            with pytest.raises(acm.AlbumSeedError):
                acm.text_pool_ids(np.ones(4), 10)


class TestTheAdaptivePool:
    def _seed(self, kind=acm.SEED_SONG):
        return {
            'type': kind, 'query': np.ones(DIM), 'clap_query': np.ones(4),
            'required': [], 'excluded_ids': set(),
        }

    def test_the_pool_widens_until_enough_candidates_survive(self, monkeypatch):
        asked = []

        def pool(query, axis_index=None, clap_query=None, first=0):
            asked.append(first)
            return [_track('id%d' % n, np.ones(DIM)) for n in range(first // 10)]

        monkeypatch.setattr(acm, 'candidate_pool', pool)
        found = acm.gather_candidates(self._seed(), {}, False, 36)
        assert asked == [acm.POOL_FIRST_QUERY, acm.POOL_FIRST_QUERY * acm.POOL_GROWTH]
        assert len(found) == 60

    def test_it_stops_at_the_ceiling_instead_of_looping(self, monkeypatch):
        asked = []

        def pool(query, axis_index=None, clap_query=None, first=0):
            asked.append(first)
            return [_track('id0', np.ones(DIM))]

        monkeypatch.setattr(acm, 'candidate_pool', pool)
        found = acm.gather_candidates(self._seed(), {}, False, 36)
        assert asked[-1] == acm.POOL_MAX_QUERY and len(found) == 1
        assert asked == sorted(set(asked))

    def test_a_description_pool_comes_from_dclap_and_widens_too(self, monkeypatch):
        asked = []

        def ids(embedding, count):
            asked.append(count)
            return ['id%d' % n for n in range(count // 100)]

        monkeypatch.setattr(acm, 'text_pool_ids', ids)
        monkeypatch.setattr(
            acm, 'load_tracks',
            lambda item_ids, axis_index=None: [_track(i, np.ones(DIM)) for i in item_ids],
        )
        found = acm.gather_candidates(self._seed(acm.SEED_TEXT), {}, False, 36)
        assert asked[0] == acm.TEXT_POOL_QUERY and len(found) >= 36


class TestConceptRefinement:
    def test_the_chosen_concepts_steer_the_query_and_are_checked_on_the_candidates(self, monkeypatch):
        monkeypatch.setattr(config, 'MOOD_LABELS', ['pop'])
        monkeypatch.setattr(acm, 'concept_vocabulary', lambda: ['viola'])
        module = MagicMock()
        module.apply_steering.return_value = (np.full(4, 2.0, dtype=np.float32), ['viola'])
        monkeypatch.setattr(config, 'CLAP_ENABLED', True)
        search = MagicMock()
        search.get_text_embedding = None
        with patch.dict('sys.modules', {'tasks.clap_steering': module}):
            monkeypatch.setattr(acm, '_text_embedding', acm._text_embedding)
            seed = acm._text_seed.__wrapped__ if hasattr(acm._text_seed, '__wrapped__') else acm._text_seed
            monkeypatch.setattr(
                acm, '_text_embedding',
                lambda query, steering=None: np.full(4, 2.0 if steering else 1.0, dtype=np.float32),
            )
            built = seed('pop with viola', {}, [{'term': 'harp', 'direction': 'more', 'weight': 3.0}])
        assert built['concepts'] == ['viola', 'harp']
        assert float(built['clap_query'][0]) == 2.0

    def test_a_concept_pushed_away_is_not_required_of_the_candidates(self, monkeypatch):
        monkeypatch.setattr(config, 'MOOD_LABELS', ['pop'])
        monkeypatch.setattr(acm, 'concept_vocabulary', lambda: [])
        monkeypatch.setattr(acm, '_text_embedding', lambda query, steering=None: np.ones(4, dtype=np.float32))
        built = acm._text_seed('pop album', {}, [{'term': 'choir', 'direction': 'less', 'weight': 3.0}])
        assert built['concepts'] == []


class TestADescriptionWithoutDclap:
    def test_it_refuses_before_loading_the_text_model(self, monkeypatch):
        monkeypatch.setattr(config, 'CLAP_ENABLED', True)
        module = MagicMock()
        module.is_clap_cache_loaded.return_value = False
        analyzer = MagicMock()
        with patch.dict('sys.modules', {'tasks.clap_text_search': module, 'tasks.clap_analyzer': analyzer}):
            with pytest.raises(acm.AlbumSeedError):
                acm._text_embedding('jazz with trumpet')
        module.warmup_text_search_model.assert_not_called()
        analyzer.get_text_embedding.assert_not_called()

    def test_it_refuses_when_dclap_is_switched_off(self, monkeypatch):
        monkeypatch.setattr(config, 'CLAP_ENABLED', False)
        with patch.dict('sys.modules', {'tasks.clap_text_search': MagicMock(), 'tasks.clap_analyzer': MagicMock()}):
            with pytest.raises(acm.AlbumSeedError):
                acm._text_embedding('jazz with trumpet')


class TestTheArtistHeadroom:
    def _pool(self, artists):
        return [
            _track('id%02d' % n, np.ones(DIM), author=artists[n % len(artists)],
                   mood_vector='jazz:%.3f' % (1.0 - n / 200.0))
            for n in range(120)
        ]

    def test_a_narrowing_that_leaves_too_few_artists_is_refused(self, monkeypatch):
        monkeypatch.setattr(config, 'ALBUM_CREATION_TRACKS', 12)
        monkeypatch.setattr(config, 'MAX_SONGS_PER_ARTIST', 3)
        monkeypatch.setattr(acm, 'concept_vocabulary', lambda: [])
        crowded = self._pool(['One Band', 'Other Band'])
        assert not acm.enough_artists(crowded, list(range(len(crowded))))
        kept = acm.keep_named_attributes(crowded, ['jazz'], [], 12)
        assert len(kept) == acm.ATTRIBUTE_KEEP
        assert {track['author'] for track in kept} == {'One Band', 'Other Band'}

    def test_a_wide_enough_field_still_narrows(self, monkeypatch):
        monkeypatch.setattr(config, 'ALBUM_CREATION_TRACKS', 12)
        monkeypatch.setattr(config, 'MAX_SONGS_PER_ARTIST', 3)
        monkeypatch.setattr(acm, 'concept_vocabulary', lambda: [])
        many = self._pool(['Artist %d' % n for n in range(30)])
        kept = acm.keep_named_attributes(many, ['jazz'], [], 12)
        assert len(kept) < len(many)
        assert acm.enough_artists(kept, list(range(len(kept))))

    def test_the_cap_being_off_asks_nothing_of_the_artists(self, monkeypatch):
        monkeypatch.setattr(config, 'MAX_SONGS_PER_ARTIST', 0)
        assert acm.enough_artists(self._pool(['Only Band']), [0, 1, 2])
