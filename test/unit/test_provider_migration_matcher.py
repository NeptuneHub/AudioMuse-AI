# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Track matching between source and target providers during migration.

Covers the matcher that pairs tracks across providers by path and metadata,
including the normalization it applies before comparing.

Main Features:
* Path normalization strips common prefixes, file:// scheme and lowercases
* Metadata normalization drops leading "the", remaster/feat/live tags
* Tiered matching prefers path over exact then normalized metadata
* Collisions resolve to the higher tier and multi-disc disambiguates by disc/track
* Chunked CandidateIndex.match_chunk with a shared claimed set is equivalent to
  the one-shot match_tracks result for any chunk size
"""

import os
import sys
import importlib.util
import pytest


def _load_matcher():
    mod_name = 'tasks.provider_migration_matcher'
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    repo_root = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
    )
    mod_path = os.path.join(repo_root, 'tasks', 'provider_migration_matcher.py')
    spec = importlib.util.spec_from_file_location(mod_name, mod_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope='module')
def matcher():
    return _load_matcher()


class TestNormalizePath:
    def test_empty_returns_none(self, matcher):
        assert matcher.normalize_path('') is None
        assert matcher.normalize_path(None) is None

    def test_strips_media_music_prefix(self, matcher):
        assert (
            matcher.normalize_path('/media/music/Artist/Album/Track.flac')
            == 'artist/album/track.flac'
        )

    def test_strips_mnt_data_prefix(self, matcher):
        assert (
            matcher.normalize_path('/mnt/data/music/Artist/Album/Track.mp3')
            == 'artist/album/track.mp3'
        )

    def test_strips_music_prefix(self, matcher):
        assert matcher.normalize_path('/music/Artist/Album/Track.flac') == 'artist/album/track.flac'

    def test_strips_volume1_prefix(self, matcher):
        assert (
            matcher.normalize_path('/volume1/music/Artist/Album/Song.mp3')
            == 'artist/album/song.mp3'
        )

    def test_strips_file_uri_scheme(self, matcher):
        assert (
            matcher.normalize_path('file:///mnt/data/music/Artist/Track.flac')
            == 'artist/track.flac'
        )

    def test_url_decodes_file_uri(self, matcher):
        assert (
            matcher.normalize_path('file:///music/The%20Beatles/Abbey%20Road/Come%20Together.flac')
            == 'the beatles/abbey road/come together.flac'
        )

    def test_backslash_to_forward_slash(self, matcher):
        assert (
            matcher.normalize_path('C:\\Music\\Artist\\Track.flac') == 'c:/music/artist/track.flac'
        )

    def test_lowercases_result(self, matcher):
        assert matcher.normalize_path('/MEDIA/MUSIC/ARTIST/TRACK.FLAC') == 'artist/track.flac'

    def test_no_matching_prefix_returns_lowercased_lstripped(self, matcher):
        assert matcher.normalize_path('/weird/path/song.mp3') == 'weird/path/song.mp3'

    def test_relative_path_unchanged_except_lowercase(self, matcher):
        assert matcher.normalize_path('Artist/Album/Song.flac') == 'artist/album/song.flac'

    def test_home_music_prefix(self, matcher):
        assert matcher.normalize_path('/home/music/Artist/Song.mp3') == 'artist/song.mp3'


class TestPathTailKey:
    def test_three_components(self, matcher):
        assert matcher.path_tail_key('a/b/c/d/e.flac') == 'c/d/e.flac'

    def test_exactly_three(self, matcher):
        assert matcher.path_tail_key('x/y/z.mp3') == 'x/y/z.mp3'

    def test_two_components(self, matcher):
        assert matcher.path_tail_key('album/song.flac') == 'album/song.flac'

    def test_single_component_returns_none(self, matcher):
        assert matcher.path_tail_key('file.flac') is None

    def test_empty_returns_none(self, matcher):
        assert matcher.path_tail_key('') is None

    def test_none_returns_none(self, matcher):
        assert matcher.path_tail_key(None) is None

    def test_leading_slash_stripped(self, matcher):
        assert matcher.path_tail_key('/a/b/c/d.flac') == 'b/c/d.flac'

    def test_lowercases(self, matcher):
        assert matcher.path_tail_key('Artist/ALBUM/Song.FLAC') == 'artist/album/song.flac'


class TestNormalizeMeta:
    def test_empty_returns_empty(self, matcher):
        assert matcher.normalize_meta('') == ''
        assert matcher.normalize_meta(None) == ''

    def test_lowercase(self, matcher):
        assert matcher.normalize_meta('HELLO WORLD') == 'hello world'

    def test_leading_the_stripped_for_artist(self, matcher):
        assert matcher.normalize_meta('The Beatles') == 'beatles'

    def test_leading_the_stripped_case_insensitive(self, matcher):
        assert matcher.normalize_meta('THE Beatles') == 'beatles'

    def test_remastered_paren_stripped(self, matcher):
        assert matcher.normalize_meta('Hey Jude (Remastered 2009)') == 'hey jude'

    def test_remastered_bracket_stripped(self, matcher):
        assert matcher.normalize_meta('Hey Jude [Remastered]') == 'hey jude'

    def test_feat_paren_stripped(self, matcher):
        assert matcher.normalize_meta('Love Me Do (feat. Ringo)') == 'love me do'

    def test_featuring_bracket_stripped(self, matcher):
        assert matcher.normalize_meta('Song [featuring Someone]') == 'song'

    def test_explicit_stripped(self, matcher):
        assert matcher.normalize_meta('Bad Song (Explicit)') == 'bad song'

    def test_clean_stripped(self, matcher):
        assert matcher.normalize_meta('Bad Song [Clean]') == 'bad song'

    def test_radio_edit_stripped(self, matcher):
        assert matcher.normalize_meta('Hit Single (Radio Edit)') == 'hit single'

    def test_live_version_stripped(self, matcher):
        assert matcher.normalize_meta('Song (Live)') == 'song'

    def test_multiple_whitespace_collapsed(self, matcher):
        assert matcher.normalize_meta('Too   Much  Space') == 'too much space'

    def test_keeps_core_title(self, matcher):
        assert matcher.normalize_meta('Mixed Feelings') == 'mixed feelings'


def _old(item_id, file_path=None, title=None, author=None, album=None, album_artist=None):
    return {
        'item_id': item_id,
        'file_path': file_path,
        'title': title,
        'author': author,
        'album': album,
        'album_artist': album_artist,
    }


def _new(new_id, path=None, title=None, artist=None, album=None, album_artist=None):
    return {
        'id': new_id,
        'path': path,
        'title': title,
        'artist': artist,
        'album': album,
        'album_artist': album_artist,
    }


class TestMatchTracks:
    def test_matches_by_normalized_path(self, matcher):
        old_rows = [
            _old(
                'old1',
                file_path='/media/music/Artist/Album/Track.flac',
                title='Track',
                author='Artist',
                album='Album',
            )
        ]
        new_tracks = [
            _new(
                'new1',
                path='/music/Artist/Album/Track.flac',
                title='Track',
                artist='Artist',
                album='Album',
            )
        ]
        result = matcher.match_tracks(old_rows, new_tracks)
        assert result['matches'] == {'old1': 'new1'}
        assert result['tier_counts']['path'] == 1
        assert result['unmatched'] == []

    def test_matches_by_path_tail_when_prefixes_differ(self, matcher):
        old_rows = [
            _old(
                'old1',
                file_path='/unknown/prefix/Artist/Album/Track.flac',
                title='Track',
                author='Artist',
                album='Album',
            )
        ]
        new_tracks = [
            _new(
                'new1',
                path='/other/weird/Artist/Album/Track.flac',
                title='Track',
                artist='Artist',
                album='Album',
            )
        ]
        result = matcher.match_tracks(old_rows, new_tracks)
        assert result['matches'] == {'old1': 'new1'}
        assert result['tier_counts']['tail'] == 1

    def test_matches_by_exact_metadata_when_no_path(self, matcher):
        old_rows = [
            _old(
                'old1',
                file_path='/media/music/a/b/c.flac',
                title='Yesterday',
                author='The Beatles',
                album='Help!',
                album_artist='The Beatles',
            )
        ]
        new_tracks = [
            _new(
                'new1',
                path=None,
                title='Yesterday',
                artist='The Beatles',
                album='Help!',
                album_artist='The Beatles',
            )
        ]
        result = matcher.match_tracks(old_rows, new_tracks)
        assert result['matches'] == {'old1': 'new1'}
        assert result['tier_counts']['exact_meta'] == 1

    def test_matches_by_normalized_metadata(self, matcher):
        old_rows = [
            _old(
                'old1',
                file_path=None,
                title='Hey Jude (Remastered 2015)',
                author='The Beatles',
                album='Past Masters',
                album_artist='The Beatles',
            )
        ]
        new_tracks = [
            _new(
                'new1',
                path=None,
                title='Hey Jude',
                artist='Beatles',
                album='Past Masters',
                album_artist='Beatles',
            )
        ]
        result = matcher.match_tracks(old_rows, new_tracks)
        assert result['matches'] == {'old1': 'new1'}
        assert result['tier_counts']['norm_meta'] == 1

    def test_orphan_when_no_match_any_tier(self, matcher):
        old_rows = [
            _old('old1', file_path='/a/b/c.flac', title='Nothing', author='Noone', album='Missing')
        ]
        new_tracks = [
            _new('new1', path='/x/y/z.flac', title='Else', artist='Other', album='Different')
        ]
        result = matcher.match_tracks(old_rows, new_tracks)
        assert result['matches'] == {}
        assert len(result['unmatched']) == 1
        assert result['unmatched'][0]['item_id'] == 'old1'

    def test_tier_priority_path_beats_exact_meta(self, matcher):
        old_rows = [
            _old(
                'old1',
                file_path='/media/music/A/B/C.flac',
                title='Song',
                author='Artist',
                album='Album',
            )
        ]
        new_tracks = [
            _new(
                'new_path',
                path='/music/A/B/C.flac',
                title='DifferentTitle',
                artist='DifferentArtist',
                album='DifferentAlbum',
            ),
            _new('new_meta', path='/other/x/y.flac', title='Song', artist='Artist', album='Album'),
        ]
        result = matcher.match_tracks(old_rows, new_tracks)
        assert result['matches']['old1'] == 'new_path'

    def test_collision_higher_tier_wins(self, matcher):
        old_rows = [
            _old(
                'old_by_meta',
                file_path='/impossible/path/no_match.flac',
                title='Song',
                author='Artist',
                album='Album',
            ),
            _old(
                'old_by_path',
                file_path='/media/music/X/Y/Z.flac',
                title='Other',
                author='OtherArtist',
                album='OtherAlbum',
            ),
        ]
        new_tracks = [
            _new('shared', path='/music/X/Y/Z.flac', title='Song', artist='Artist', album='Album'),
        ]
        result = matcher.match_tracks(old_rows, new_tracks)
        assert result['matches'] == {'old_by_path': 'shared'}
        assert len(result['unmatched']) == 1
        assert result['unmatched'][0]['item_id'] == 'old_by_meta'

    def test_multidisc_disambiguates_by_disc_track(self, matcher):
        old_rows = [
            _old(
                'nav_d1',
                file_path='Green Day/American Idiot (Japanese Edition)/01-05 - Are We The Waiting.flac',
                title='Are We The Waiting',
                author='Green Day',
                album='American Idiot (Japanese Edition)',
                album_artist='Green Day',
            ),
            _old(
                'nav_d2',
                file_path='Green Day/American Idiot (Japanese Edition)/02-04 - Are We The Waiting.flac',
                title='Are We The Waiting',
                author='Green Day',
                album='American Idiot (Japanese Edition)',
                album_artist='Green Day',
            ),
        ]
        new_tracks = [
            _new(
                'emby_d1',
                path='/media/music/American Idiot (Japanese Edition) (2004) {CD}/1-5 Are We The Waiting.flac',
                title='Are We The Waiting',
                artist='Green Day',
                album='American Idiot (Japanese Edition)',
                album_artist='Green Day',
            ),
            _new(
                'emby_d2',
                path='/media/music/American Idiot (Japanese Edition) (2004) {CD}/2-4 Are We The Waiting.flac',
                title='Are We The Waiting',
                artist='Green Day',
                album='American Idiot (Japanese Edition)',
                album_artist='Green Day',
            ),
        ]
        result = matcher.match_tracks(old_rows, new_tracks)
        assert result['matches'] == {'nav_d1': 'emby_d1', 'nav_d2': 'emby_d2'}
        assert len(result['unmatched']) == 0

    def test_extract_disc_track_various_formats(self, matcher):
        assert matcher.extract_disc_track('01-05 - Are We The Waiting.flac') == (1, 5)
        assert matcher.extract_disc_track('1-5 Are We The Waiting.flac') == (1, 5)
        assert matcher.extract_disc_track('2.4 Song.mp3') == (2, 4)
        assert matcher.extract_disc_track('2 4 Song.mp3') == (2, 4)
        assert matcher.extract_disc_track('/music/Album/02-07 Song.flac') == (2, 7)
        assert matcher.extract_disc_track('Song.flac') is None
        assert matcher.extract_disc_track('07 Song.flac') is None
        assert matcher.extract_disc_track('') is None
        assert matcher.extract_disc_track(None) is None

    def test_unmatched_grouped_by_album(self, matcher):
        old_rows = [
            _old('o1', album='Abbey Road', album_artist='Beatles', title='T1'),
            _old('o2', album='Abbey Road', album_artist='Beatles', title='T2'),
            _old('o3', album='Rumours', album_artist='Fleetwood Mac', title='T3'),
        ]
        new_tracks = []
        result = matcher.match_tracks(old_rows, new_tracks)
        assert result['matches'] == {}
        assert len(result['unmatched']) == 3
        by_album = result['unmatched_by_album']
        assert ('Beatles', 'Abbey Road') in by_album
        assert ('Fleetwood Mac', 'Rumours') in by_album
        assert len(by_album[('Beatles', 'Abbey Road')]) == 2
        assert len(by_album[('Fleetwood Mac', 'Rumours')]) == 1


class TestTitleArtistTier:
    def test_disabled_by_default_different_album_stays_unmatched(self, matcher):
        old_rows = [
            _old(
                'old1',
                file_path=None,
                title='Yesterday',
                author='Beatles',
                album='Help!',
                album_artist='Beatles',
            )
        ]
        new_tracks = [
            _new(
                'new1',
                path=None,
                title='Yesterday',
                artist='Beatles',
                album='1967-1970',
                album_artist='Beatles',
            )
        ]
        result = matcher.match_tracks(old_rows, new_tracks)
        assert result['matches'] == {}
        assert len(result['unmatched']) == 1

    def test_enabled_matches_across_albums(self, matcher):
        old_rows = [
            _old(
                'old1',
                file_path=None,
                title='Yesterday',
                author='Beatles',
                album='Help!',
                album_artist='Beatles',
            )
        ]
        new_tracks = [
            _new(
                'new1',
                path=None,
                title='Yesterday',
                artist='Beatles',
                album='1967-1970',
                album_artist='Beatles',
            )
        ]
        result = matcher.match_tracks(old_rows, new_tracks, allow_title_artist_only=True)
        assert result['matches'] == {'old1': 'new1'}
        assert result['tier_counts']['title_artist'] == 1

    def test_lower_priority_than_norm_meta(self, matcher):
        old_rows = [
            _old(
                'old1',
                file_path=None,
                title='Yesterday',
                author='The Beatles',
                album='Help!',
                album_artist='The Beatles',
            )
        ]
        new_tracks = [
            _new(
                'new_compilation',
                path=None,
                title='Yesterday',
                artist='Beatles',
                album='1967-1970',
                album_artist='Beatles',
            ),
            _new(
                'new_studio',
                path=None,
                title='Yesterday',
                artist='Beatles',
                album='Help!',
                album_artist='Beatles',
            ),
        ]
        result = matcher.match_tracks(old_rows, new_tracks, allow_title_artist_only=True)
        assert result['matches'] == {'old1': 'new_studio'}
        assert result['tier_counts']['norm_meta'] == 1
        assert result['tier_counts']['title_artist'] == 0

    def test_tier_counts_include_title_artist_only_when_enabled(self, matcher):
        result_off = matcher.match_tracks([], [])
        assert 'title_artist' not in result_off['tier_counts']

        result_on = matcher.match_tracks([], [], allow_title_artist_only=True)
        assert 'title_artist' in result_on['tier_counts']
        assert result_on['tier_counts']['title_artist'] == 0


class TestArtistHierarchy:
    def test_source_prefers_author_over_various_artists_album_artist(self, matcher):
        old_rows = [
            _old(
                'old1',
                file_path=None,
                title='Hotel California',
                author='Eagles',
                album='Ultimate Rock Hits',
                album_artist='Various Artists',
            )
        ]
        new_tracks = [
            _new(
                'new1',
                path=None,
                title='Hotel California',
                artist='Eagles',
                album='Ultimate Rock Hits',
                album_artist='Various Artists',
            )
        ]
        result = matcher.match_tracks(old_rows, new_tracks)
        assert result['matches'] == {'old1': 'new1'}
        assert result['tier_counts']['exact_meta'] == 1

    def test_title_artist_tier_uses_author_not_various_artists(self, matcher):
        old_rows = [
            _old(
                'old1',
                file_path=None,
                title='Hotel California',
                author='Eagles',
                album='Ultimate Rock Hits',
                album_artist='Various Artists',
            )
        ]
        new_tracks = [
            _new(
                'new1',
                path=None,
                title='Hotel California',
                artist='Eagles',
                album='Hotel California',
                album_artist='Eagles',
            )
        ]
        result = matcher.match_tracks(old_rows, new_tracks, allow_title_artist_only=True)
        assert result['matches'] == {'old1': 'new1'}
        assert result['tier_counts']['title_artist'] == 1

    def test_target_prefers_artist_over_various_artists_album_artist(self, matcher):
        old_rows = [
            _old(
                'old1',
                file_path=None,
                title='Stairway to Heaven',
                author='Led Zeppelin',
                album='Classic Rock Anthems',
                album_artist='Led Zeppelin',
            )
        ]
        new_tracks = [
            _new(
                'new1',
                path=None,
                title='Stairway to Heaven',
                artist='Led Zeppelin',
                album='Classic Rock Anthems',
                album_artist='Various Artists',
            )
        ]
        result = matcher.match_tracks(old_rows, new_tracks)
        assert result['matches'] == {'old1': 'new1'}
        assert result['tier_counts']['exact_meta'] == 1

    def test_source_falls_back_to_album_artist_when_author_missing(self, matcher):
        old_rows = [
            _old(
                'old1',
                file_path=None,
                title='Bohemian Rhapsody',
                author=None,
                album='A Night at the Opera',
                album_artist='Queen',
            )
        ]
        new_tracks = [
            _new(
                'new1',
                path=None,
                title='Bohemian Rhapsody',
                artist='Queen',
                album='A Night at the Opera',
                album_artist='Queen',
            )
        ]
        result = matcher.match_tracks(old_rows, new_tracks)
        assert result['matches'] == {'old1': 'new1'}
        assert result['tier_counts']['exact_meta'] == 1

    def test_target_falls_back_to_album_artist_when_artist_missing(self, matcher):
        old_rows = [
            _old(
                'old1',
                file_path=None,
                title='Riders on the Storm',
                author='The Doors',
                album='L.A. Woman',
                album_artist='The Doors',
            )
        ]
        new_tracks = [
            _new(
                'new1',
                path=None,
                title='Riders on the Storm',
                artist=None,
                album='L.A. Woman',
                album_artist='The Doors',
            )
        ]
        result = matcher.match_tracks(old_rows, new_tracks)
        assert result['matches'] == {'old1': 'new1'}
        assert result['tier_counts']['exact_meta'] == 1

    def test_best_artist_old_helper_precedence(self, matcher):
        fn = matcher._best_artist_old
        assert fn({'author': 'A', 'artist': 'B', 'album_artist': 'C'}) == 'A'
        assert fn({'author': None, 'artist': 'B', 'album_artist': 'C'}) == 'B'
        assert fn({'author': None, 'artist': None, 'album_artist': 'C'}) == 'C'
        assert fn({'author': '', 'artist': '', 'album_artist': 'C'}) == 'C'
        assert fn({}) is None

    def test_best_artist_new_helper_precedence(self, matcher):
        fn = matcher._best_artist_new
        assert fn({'artist': 'A', 'album_artist': 'B'}) == 'A'
        assert fn({'artist': None, 'album_artist': 'B'}) == 'B'
        assert fn({'artist': '', 'album_artist': 'B'}) == 'B'
        assert fn({}) is None


def _tiered_catalogue():
    old_rows = []
    new_tracks = []
    expected_matches = {}
    expected_tiers = {}

    for i in range(10):
        old_rows.append(_old(
            f'p{i}',
            file_path=f'/media/music/PathArt{i}/PathAlb{i}/PathTrack{i}.flac',
            title=f'Path Track {i}', author=f'Path Artist {i}', album=f'Path Album {i}',
        ))
        new_tracks.append(_new(
            f'npath{i}',
            path=f'/music/PathArt{i}/PathAlb{i}/PathTrack{i}.flac',
            title=f'Path Track {i}', artist=f'Path Artist {i}', album=f'Path Album {i}',
        ))
        expected_matches[f'p{i}'] = f'npath{i}'
        expected_tiers[f'p{i}'] = 'path'

    for i in range(10):
        old_rows.append(_old(
            f't{i}',
            file_path=f'/unknownroot/one{i}/TailArt{i}/TailAlb{i}/TailTrack{i}.flac',
            title=f'Tail Track {i}', author=f'Tail Artist {i}', album=f'Tail Album {i}',
        ))
        new_tracks.append(_new(
            f'ntail{i}',
            path=f'/anotherroot/two{i}/TailArt{i}/TailAlb{i}/TailTrack{i}.flac',
            title=f'Tail Track {i}', artist=f'Tail Artist {i}', album=f'Tail Album {i}',
        ))
        expected_matches[f't{i}'] = f'ntail{i}'
        expected_tiers[f't{i}'] = 'tail'

    for i in range(10):
        old_rows.append(_old(
            f'e{i}',
            title=f'Exact Song {i}', author=f'Exact Artist {i}', album=f'Exact Album {i}',
        ))
        new_tracks.append(_new(
            f'nexact{i}',
            title=f'Exact Song {i}', artist=f'Exact Artist {i}', album=f'Exact Album {i}',
        ))
        expected_matches[f'e{i}'] = f'nexact{i}'
        expected_tiers[f'e{i}'] = 'exact_meta'

    for i in range(10):
        old_rows.append(_old(
            f'm{i}',
            title=f'Norm Song {i} (Remastered 2011)',
            author=f'The Norm Artist {i}', album=f'Norm Album {i}',
        ))
        new_tracks.append(_new(
            f'nnorm{i}',
            title=f'Norm Song {i}', artist=f'Norm Artist {i}', album=f'Norm Album {i}',
        ))
        expected_matches[f'm{i}'] = f'nnorm{i}'
        expected_tiers[f'm{i}'] = 'norm_meta'

    for i in range(10):
        old_rows.append(_old(
            f'x{i}',
            file_path=f'lonely{i}.flac',
            title=f'Ghost {i}', author=f'Ghost Artist {i}',
        ))

    for j in range(5):
        new_tracks.append(_new(
            f'ncomp{j}',
            path=f'/music/CompArt{j}/CompAlb{j}/CompTrack{j}.flac',
            title=f'Comp Song {j}', artist=f'Comp Artist {j}', album=f'Comp Album {j}',
        ))
        old_rows.append(_old(
            f'cA{j}',
            file_path=f'/media/music/CompArt{j}/CompAlb{j}/CompTrack{j}.flac',
            title=f'Unrelated {j}', author=f'Unrelated Artist {j}',
            album=f'Unrelated Album {j}',
        ))
        old_rows.append(_old(
            f'cB{j}',
            title=f'Comp Song {j}', author=f'Comp Artist {j}', album=f'Comp Album {j}',
        ))
        expected_matches[f'cA{j}'] = f'ncomp{j}'
        expected_tiers[f'cA{j}'] = 'path'

    for j in range(5):
        new_tracks.append(_new(
            f'nx{j}',
            path=f'/music/Distract{j}/Nowhere{j}/Void{j}.flac',
            title=f'Void {j}', artist=f'Void Artist {j}', album=f'Void Album {j}',
        ))

    expected_unmatched = {f'x{i}' for i in range(10)} | {f'cB{j}' for j in range(5)}
    return old_rows, new_tracks, expected_matches, expected_tiers, expected_unmatched


def _run_chunked(matcher, old_rows, new_tracks, chunk_size):
    index = matcher.CandidateIndex(new_tracks)
    claimed = {}
    matches = {}
    tiers = {}
    for start in range(0, len(old_rows), chunk_size):
        result = index.match_chunk(old_rows[start:start + chunk_size], claimed)
        matches.update(result['matches'])
        tiers.update(result['match_tiers'])
    return matches, tiers


class TestChunkedEquivalence:
    def test_baseline_covers_every_tier_and_competition(self, matcher):
        old_rows, new_tracks, expected_matches, expected_tiers, expected_unmatched = (
            _tiered_catalogue()
        )
        result = matcher.match_tracks(old_rows, new_tracks)
        assert result['matches'] == expected_matches
        assert result['match_tiers'] == expected_tiers
        assert {row['item_id'] for row in result['unmatched']} == expected_unmatched

    @pytest.mark.parametrize('chunk_size', [1, 7, 65])
    def test_chunked_matching_equals_one_shot(self, matcher, chunk_size):
        old_rows, new_tracks, expected_matches, expected_tiers, _unmatched = (
            _tiered_catalogue()
        )
        baseline = matcher.match_tracks(old_rows, new_tracks)
        matches, tiers = _run_chunked(matcher, old_rows, new_tracks, chunk_size)
        assert matches == baseline['matches'] == expected_matches
        assert tiers == baseline['match_tiers'] == expected_tiers


class TestClaimStealingAcrossChunks:
    """A later chunk with a STRONGER tier must be able to take a provider track.

    The `claimed` map spans chunks while the best-match tie-break only sees one
    chunk, so without a tier rank the first chunk to touch a provider track owned
    it forever - a normalized-metadata guess would permanently outrank an exact
    path match that arrives later, and nothing ever re-matched it.
    """

    def test_stronger_later_tier_takes_the_track(self, matcher):
        new_tracks = [{
            'id': 'p1', 'path': '/music/song.flac',
            'title': 'Song', 'artist': 'Artist', 'album': 'Album',
        }]
        weak = {
            'item_id': 'fp_weak', 'file_path': '/elsewhere/other.flac',
            'title': 'Song', 'author': 'Artist', 'album': 'Album', 'album_artist': 'Artist',
        }
        strong = {
            'item_id': 'fp_strong', 'file_path': '/music/song.flac',
            'title': 'Different', 'author': 'Other', 'album': 'Other', 'album_artist': 'Other',
        }

        index = matcher.CandidateIndex(new_tracks)
        claimed = {}
        first = index.match_chunk([weak], claimed)
        second = index.match_chunk([strong], claimed)

        assert first['matches'] == {'fp_weak': 'p1'}
        # The exact-path row wins it in the next chunk...
        assert second['matches'] == {'fp_strong': 'p1'}
        assert second['match_tiers']['fp_strong'] == 'path'
        # ...and the weak owner is left for a later sweep (the upsert moves the
        # provider id, so the database never holds both).
        assert claimed['p1'] == index._tier_rank['path']

    def test_weaker_later_tier_does_not_steal(self, matcher):
        new_tracks = [{
            'id': 'p1', 'path': '/music/song.flac',
            'title': 'Song', 'artist': 'Artist', 'album': 'Album',
        }]
        strong = {
            'item_id': 'fp_strong', 'file_path': '/music/song.flac',
            'title': 'Song', 'author': 'Artist', 'album': 'Album', 'album_artist': 'Artist',
        }
        weak = {
            'item_id': 'fp_weak', 'file_path': '/elsewhere/other.flac',
            'title': 'Song', 'author': 'Artist', 'album': 'Album', 'album_artist': 'Artist',
        }

        index = matcher.CandidateIndex(new_tracks)
        claimed = {}
        first = index.match_chunk([strong], claimed)
        second = index.match_chunk([weak], claimed)

        assert first['matches'] == {'fp_strong': 'p1'}
        assert second['matches'] == {}
        assert [row['item_id'] for row in second['unmatched']] == ['fp_weak']
