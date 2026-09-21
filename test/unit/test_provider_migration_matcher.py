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
        assert second['matches'] == {'fp_strong': 'p1'}
        assert second['match_tiers']['fp_strong'] == 'path'
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


class TestDuplicateFilesOfOneSong:
    def _tracks(self):
        return [
            {'id': 'n-1', 'path': '/music/Queen/II/01 Procession.flac', 'title': 'Procession', 'artist': 'Queen', 'album': 'II', 'album_artist': 'Queen'},
            {'id': 'n-2', 'path': '/music/Queen/II copy/01 Procession.flac', 'title': 'Procession', 'artist': 'Queen', 'album': 'II', 'album_artist': 'Queen'},
            {'id': 'n-3', 'path': '/other/Queen/Greatest/01 Procession.mp3', 'title': 'Procession', 'artist': 'Queen', 'album': 'Greatest', 'album_artist': 'Queen'},
            {'id': 'n-4', 'path': '/music/Other/Song.flac', 'title': 'Song', 'artist': 'Other', 'album': 'X', 'album_artist': 'Other'},
        ]

    def test_every_known_file_of_the_song_is_mapped(self):
        CandidateIndex = _load_matcher().CandidateIndex

        old = {
            'item_id': 'fp_1', 'title': 'Procession', 'author': 'Queen', 'album': 'II', 'album_artist': 'Queen',
            'file_path': '/media/Queen/II/01 Procession.flac',
            'file_paths': ['/media/Queen/II/01 Procession.flac', '/media/Queen/II copy/01 Procession.flac',
                           '/srv/music/Queen/Greatest/01 Procession.mp3'],
        }
        result = CandidateIndex(self._tracks()).match_chunk([old])
        assert result['matches'] == {'fp_1': 'n-1'}
        assert result['extra_matches'] == {'n-2': 'fp_1', 'n-3': 'fp_1'}
        assert result['extra_match_tiers'] == {'n-2': 'path', 'n-3': 'tail'}

    def test_a_single_file_song_has_no_extras(self):
        CandidateIndex = _load_matcher().CandidateIndex

        old = {'item_id': 'fp_1', 'title': 'Procession', 'author': 'Queen', 'album': 'II', 'album_artist': 'Queen',
               'file_path': '/media/Queen/II/01 Procession.flac'}
        result = CandidateIndex(self._tracks()).match_chunk([old])
        assert result['matches'] == {'fp_1': 'n-1'}
        assert result['extra_matches'] == {}

    def test_a_file_another_song_owns_is_never_taken_as_an_extra(self):
        CandidateIndex = _load_matcher().CandidateIndex

        owner = {'item_id': 'fp_2', 'title': 'Song', 'author': 'Other', 'album': 'X', 'album_artist': 'Other',
                 'file_path': '/media/Other/Song.flac'}
        greedy = {'item_id': 'fp_1', 'title': 'Procession', 'author': 'Queen', 'album': 'II', 'album_artist': 'Queen',
                  'file_path': '/media/Queen/II/01 Procession.flac',
                  'file_paths': ['/media/Queen/II/01 Procession.flac', '/media/Other/Song.flac']}
        claimed = {}
        index = CandidateIndex(self._tracks())
        first = index.match_chunk([owner], claimed)
        second = index.match_chunk([greedy], claimed)
        assert first['matches'] == {'fp_2': 'n-4'}
        assert 'n-4' not in second['extra_matches']
        assert second['matches'] == {'fp_1': 'n-1'}

    def test_an_unmatched_song_contributes_no_extras(self):
        CandidateIndex = _load_matcher().CandidateIndex

        old = {'item_id': 'fp_9', 'title': 'Missing', 'author': 'Nobody', 'album': 'None', 'album_artist': 'Nobody',
               'file_path': '/media/nowhere.flac', 'file_paths': ['/media/nowhere.flac']}
        result = CandidateIndex(self._tracks()).match_chunk([old])
        assert result['matches'] == {}
        assert result['extra_matches'] == {}


class TestExtrasNeverStealAnotherSong:
    def test_an_ambiguous_tail_is_never_used_for_an_extra(self):
        CandidateIndex = _load_matcher().CandidateIndex

        tracks = [
            {'id': 'abba', 'path': '/music/ABBA/Greatest Hits/CD1/01.mp3', 'title': 'Waterloo', 'artist': 'ABBA', 'album': 'Greatest Hits', 'album_artist': 'ABBA'},
            {'id': 'queen', 'path': '/volume2/Queen/Greatest Hits/CD1/01.mp3', 'title': 'Bohemian', 'artist': 'Queen', 'album': 'Greatest Hits', 'album_artist': 'Queen'},
        ]
        old = {'item_id': 'fp_q', 'title': 'Bohemian', 'author': 'Queen', 'album': 'Greatest Hits', 'album_artist': 'Queen',
               'file_path': '/volume2/Queen/Greatest Hits/CD1/01.mp3',
               'file_paths': ['/volume2/Queen/Greatest Hits/CD1/01.mp3', '/nas2/other/Greatest Hits/CD1/01.mp3']}
        result = CandidateIndex(tracks).match_chunk([old])
        assert result['matches'] == {'fp_q': 'queen'}
        assert result['extra_matches'] == {}, 'a shared tail proves nothing, so ABBA is never bound to Queen'

    def test_a_later_songs_own_match_takes_back_a_file_claimed_as_an_extra(self):
        CandidateIndex = _load_matcher().CandidateIndex

        tracks = [
            {'id': 'n-1', 'path': '/music/A/Album/01 Song.flac', 'title': 'Song', 'artist': 'A', 'album': 'Album', 'album_artist': 'A'},
            {'id': 'n-4', 'path': '/music/B/Other/02 Tune.flac', 'title': 'Tune', 'artist': 'B', 'album': 'Other', 'album_artist': 'B'},
        ]
        greedy = {'item_id': 'fp_1', 'title': 'Song', 'author': 'A', 'album': 'Album', 'album_artist': 'A',
                  'file_path': '/media/A/Album/01 Song.flac',
                  'file_paths': ['/media/A/Album/01 Song.flac', '/media/B/Other/02 Tune.flac']}
        owner = {'item_id': 'fp_9', 'title': 'Tune', 'author': 'B', 'album': 'Other', 'album_artist': 'B',
                 'file_path': '/media/B/Other/02 Tune.flac'}
        claimed = {}
        index = CandidateIndex(tracks)
        first = index.match_chunk([greedy], claimed)
        second = index.match_chunk([owner], claimed)
        assert first['extra_matches'] == {'n-4': 'fp_1'}
        assert second['matches'] == {'fp_9': 'n-4'}, (
            'an extra claims below every tier, so the file goes back to the song it really belongs to'
        )


class TestPathKeysSharedBySeveralFiles:
    def test_copies_under_two_roots_keep_both_on_the_same_server(self):
        CandidateIndex = _load_matcher().CandidateIndex

        tracks = [
            {'id': 'a', 'path': '/music/Queen/II/01 Procession.flac', 'title': 'Procession', 'artist': 'Queen', 'album': 'II'},
            {'id': 'b', 'path': '/data/Queen/II/01 Procession.flac', 'title': 'Procession', 'artist': 'Queen', 'album': 'II'},
        ]
        old = {'item_id': 'fp_s', 'title': 'Procession', 'author': 'Queen', 'album': 'II',
               'file_paths': ['/music/Queen/II/01 Procession.flac', '/data/Queen/II/01 Procession.flac']}
        result = CandidateIndex(tracks).match_chunk([old])
        assert result['matches'] == {'fp_s': 'a'}
        assert result['extra_matches'] == {'b': 'fp_s'}

    def test_two_libraries_with_the_same_layout_keep_both_copies(self):
        CandidateIndex = _load_matcher().CandidateIndex

        tracks = [
            {'id': 'nA', 'path': '/music/A/Queen/II/01.flac', 'title': 'Procession', 'artist': 'Queen', 'album': 'II'},
            {'id': 'nB', 'path': '/music/B/Queen/II/01.flac', 'title': 'Procession', 'artist': 'Queen', 'album': 'II'},
        ]
        old = {'item_id': 'fp_s', 'title': 'Procession', 'author': 'Queen', 'album': 'II',
               'file_paths': ['/mnt/user/Music/A/Queen/II/01.flac', '/mnt/user/Music/B/Queen/II/01.flac']}
        result = CandidateIndex(tracks).match_chunk([old])
        assert result['matches'] == {'fp_s': 'nA'}
        assert result['match_tiers'] == {'fp_s': 'tail'}
        assert result['extra_matches'] == {'nB': 'fp_s'}

    def test_a_tail_two_songs_share_falls_through_to_metadata(self):
        CandidateIndex = _load_matcher().CandidateIndex

        tracks = [
            {'id': 'abba', 'path': '/music/ABBA/Greatest Hits/CD1/01.mp3', 'title': 'Waterloo', 'artist': 'ABBA', 'album': 'Greatest Hits'},
            {'id': 'queen', 'path': '/music/Queen/Greatest Hits/CD1/01.mp3', 'title': 'Bohemian', 'artist': 'Queen', 'album': 'Greatest Hits'},
        ]
        old = {'item_id': 'fp_q', 'title': 'Bohemian', 'author': 'Queen', 'album': 'Greatest Hits',
               'file_paths': ['/nas2/Queen Collection/Greatest Hits/CD1/01.mp3']}
        result = CandidateIndex(tracks).match_chunk([old])
        assert result['matches'] == {'fp_q': 'queen'}
        assert result['match_tiers'] == {'fp_q': 'exact_meta'}, 'a tied tail never binds ABBA to Queen'

    def _live_tracks(self):
        return [
            {'id': 'F1', 'path': '/music/Queen/II/01 Procession.flac', 'title': 'Procession', 'artist': 'Queen', 'album': 'II'},
            {'id': 'F2', 'path': '/music/Queen/II Live/01 Procession.flac', 'title': 'Procession', 'artist': 'Queen', 'album': 'II Live'},
        ]

    def _live_rows(self):
        song = {'item_id': 'fp_a', 'title': 'Procession', 'author': 'Queen', 'album': 'II',
                'file_paths': ['/media/Queen/II/01 Procession.flac', '/media/Queen/II Live/01 Procession.flac']}
        guess = {'item_id': 'fp_b', 'title': 'Procession (Live)', 'author': 'Queen', 'album': 'II Live'}
        return song, guess

    def test_a_path_duplicate_never_takes_another_songs_metadata_match(self):
        CandidateIndex = _load_matcher().CandidateIndex

        song, guess = self._live_rows()
        result = CandidateIndex(self._live_tracks()).match_chunk([song, guess])
        assert result['matches'] == {'fp_a': 'F1', 'fp_b': 'F2'}
        assert result['extra_matches'] == {}

    def test_a_later_chunk_keeps_a_file_an_earlier_song_matched(self):
        CandidateIndex = _load_matcher().CandidateIndex

        song, guess = self._live_rows()
        index = CandidateIndex(self._live_tracks())
        claimed = {}
        first = index.match_chunk([guess], claimed)
        second = index.match_chunk([song], claimed)
        assert first['matches'] == {'fp_b': 'F2'}
        assert second['matches'] == {'fp_a': 'F1'}
        assert second['extra_matches'] == {}

    def test_the_same_title_twice_on_one_album_is_told_apart_by_track_number(self):
        CandidateIndex = _load_matcher().CandidateIndex

        tracks = [
            {'id': 'n3', 'path': '/music/X/Album/03 - Interlude.flac', 'title': 'Interlude', 'artist': 'X', 'album': 'Album'},
            {'id': 'n9', 'path': '/music/X/Album/09 - Interlude.flac', 'title': 'Interlude', 'artist': 'X', 'album': 'Album'},
        ]
        rows = [
            {'item_id': 'fp_3', 'title': 'Interlude', 'author': 'X', 'album': 'Album', 'file_paths': ['/other/root/X-Album/03 - Interlude.flac']},
            {'item_id': 'fp_9', 'title': 'Interlude', 'author': 'X', 'album': 'Album', 'file_paths': ['/other/root/X-Album/09 - Interlude.flac']},
        ]
        result = CandidateIndex(tracks).match_chunk(rows)
        assert result['matches'] == {'fp_3': 'n3', 'fp_9': 'n9'}


class TestTakeoverEdges:
    def test_a_path_that_ends_with_another_path_still_maps_both_files_in_any_order(self):
        CandidateIndex = _load_matcher().CandidateIndex

        short = {'id': 't_short', 'path': '/media/Artist/Album/01.flac', 'title': 'S', 'artist': 'A', 'album': 'B'}
        long_ = {'id': 't_long', 'path': '/media/media/Artist/Album/01.flac', 'title': 'S', 'artist': 'A', 'album': 'B'}
        old = {'item_id': 'fp_A', 'title': 'S', 'author': 'A', 'album': 'B',
               'file_paths': ['/media/Artist/Album/01.flac', '/media/media/Artist/Album/01.flac']}
        for tracks in ([long_, short], [short, long_]):
            result = CandidateIndex(tracks).match_chunk([dict(old)])
            assert result['matches'] == {'fp_A': 't_short'}
            assert result['extra_matches'] == {'t_long': 'fp_A'}

    def test_an_extra_never_unbinds_the_song_that_owns_the_file(self):
        CandidateIndex = _load_matcher().CandidateIndex

        tracks = [
            {'id': 'queen', 'path': '/music/Queen/Greatest Hits/CD1/01.mp3', 'title': 'Bohemian', 'artist': 'Queen', 'album': 'Greatest Hits'},
            {'id': 'abba_own', 'path': '/music/ABBA/Gold/Disc 1/01.mp3', 'title': 'Waterloo', 'artist': 'ABBA', 'album': 'Gold'},
        ]
        queen = {'item_id': 'fp_q', 'title': 'Bohemian', 'author': 'Queen', 'album': 'Greatest Hits'}
        abba = {'item_id': 'fp_a', 'title': 'Waterloo', 'author': 'ABBA', 'album': 'Gold',
                'file_paths': ['/music/ABBA/Gold/Disc 1/01.mp3', '/nas2/ABBA/Greatest Hits/CD1/01.mp3']}
        for rows in ([queen, abba], [abba, queen]):
            result = CandidateIndex(tracks).match_chunk([dict(r) for r in rows])
            assert result['matches'] == {'fp_q': 'queen', 'fp_a': 'abba_own'}
            assert result['extra_matches'] == {}
            assert result['unmatched'] == []

    def test_a_songs_own_later_tail_or_metadata_match_takes_back_an_extra(self):
        CandidateIndex = _load_matcher().CandidateIndex

        tracks = [
            {'id': 'n-1', 'path': '/music/X/Album/01 X.flac', 'title': 'X', 'artist': 'X', 'album': 'Album'},
            {'id': 'n-4', 'path': '/music/Y/Other/04 Y.flac', 'title': 'Y', 'artist': 'Y', 'album': 'Other'},
        ]
        greedy = {'item_id': 'fp_x', 'title': 'X', 'author': 'X', 'album': 'Album',
                  'file_paths': ['/music/X/Album/01 X.flac', '/music/Y/Other/04 Y.flac']}
        owner = {'item_id': 'fp_y', 'title': 'Y', 'author': 'Y', 'album': 'Other'}
        index = CandidateIndex(tracks)
        claimed = {}
        first = index.match_chunk([greedy], claimed)
        second = index.match_chunk([owner], claimed)
        assert first['extra_matches'] == {'n-4': 'fp_x'}
        assert second['matches'] == {'fp_y': 'n-4'}

    def test_a_tail_tied_between_copies_of_the_same_song_still_matches(self):
        CandidateIndex = _load_matcher().CandidateIndex

        tracks = [
            {'id': 'nA', 'path': '/music/A/Queen/II/01.flac', 'title': 'Procession', 'artist': 'Queen', 'album': 'II'},
            {'id': 'nB', 'path': '/music/B/Queen/II/01.flac', 'title': 'Procession', 'artist': 'Queen', 'album': 'II (Remaster 2011)'},
        ]
        old = {'item_id': 'fp_s', 'title': 'Procession', 'author': 'Queen', 'album': 'Queen II',
               'file_paths': ['/mnt/user/Music/Queen/II/01.flac']}
        result = CandidateIndex(tracks).match_chunk([old])
        assert result['matches'] == {'fp_s': 'nA'}

    def test_folder_depth_is_not_evidence_for_a_tied_tail(self):
        CandidateIndex = _load_matcher().CandidateIndex

        tracks = [
            {'id': 'abba', 'path': '/music/pop/ABBA/Greatest Hits/CD1/01.mp3', 'title': 'Waterloo', 'artist': 'ABBA', 'album': 'Greatest Hits'},
            {'id': 'queen', 'path': '/music/Queen/Greatest Hits/CD1/01.mp3', 'title': 'Bohemian', 'artist': 'Queen', 'album': 'Greatest Hits'},
        ]
        old = {'item_id': 'fp_q', 'title': 'Bohemian', 'author': 'Queen', 'album': 'Greatest Hits',
               'file_paths': ['/nas2/x/Queen Collection/Greatest Hits/CD1/01.mp3']}
        result = CandidateIndex(tracks).match_chunk([old])
        assert result['matches'] == {'fp_q': 'queen'}
        assert result['match_tiers'] == {'fp_q': 'exact_meta'}


class TestCatalogueOnlyMatchingForSongsWithoutPaths:
    def test_metadata_folding_matches_the_spellings_servers_disagree_on(self):
        normalize_meta = _load_matcher().normalize_meta

        assert normalize_meta('The Fall\u2010Off') == normalize_meta('The Fall-Off')
        assert normalize_meta('We Gotta Groove: The Brother Studio Years') == normalize_meta('We Gotta Groove - The Brother Studio Years')
        assert normalize_meta('That\u2019s Right Baby') == normalize_meta("That's Right Baby")
        assert normalize_meta('[Unknown Artist]') == '' and normalize_meta('Unknown Album') == ''

    def test_a_song_whose_album_tag_differs_matches_by_title_artist_and_duration(self):
        CandidateIndex = _load_matcher().CandidateIndex

        tracks = [
            {'id': 'j1', 'path': '/m/Kacey/Deeper Well - Deeper into the Well - 18 - Superbloom.flac', 'title': 'Superbloom',
             'artist': 'Kacey Musgraves', 'album': 'Deeper Well: Deeper into the Well', 'duration': 182.9},
            {'id': 'j2', 'path': '/m/MisterWives/SUPERBLOOM - 19 - SUPERBLOOM.mp3', 'title': 'SUPERBLOOM',
             'artist': 'MisterWives', 'album': 'SUPERBLOOM', 'duration': 213.6},
        ]
        old = {'item_id': 'fp_s', 'title': 'Superbloom', 'author': 'Kacey Musgraves', 'album': 'Deeper Well', 'duration': 182.946667}
        result = CandidateIndex(tracks, duration_tolerance=1.0).match_chunk([old])
        assert result['matches'] == {'fp_s': 'j1'}
        assert result['match_tiers'] == {'fp_s': 'title_duration'}

    def test_unknown_artist_and_album_placeholders_still_match_by_title_and_duration(self):
        CandidateIndex = _load_matcher().CandidateIndex

        tracks = [{'id': 'j1', 'path': '/m/Walk/04 - Walk To Work.mp3', 'title': '04 - Walk To Work', 'artist': None, 'album': None, 'duration': 210.7}]
        old = {'item_id': 'fp_w', 'title': '04 - Walk To Work', 'author': '[Unknown Artist]', 'album': '[Unknown Album]', 'duration': 210.703673}
        assert CandidateIndex(tracks, duration_tolerance=1.0).match_chunk([old])['matches'] == {'fp_w': 'j1'}

    def test_a_different_version_with_the_same_title_is_told_apart_by_duration(self):
        CandidateIndex = _load_matcher().CandidateIndex

        tracks = [
            {'id': 's5', 'path': '/m/Gorillaz/Plastic Beach - 05 - Stylo.flac', 'title': 'Stylo', 'artist': 'Gorillaz', 'album': 'Plastic Beach', 'duration': 262.6},
            {'id': 's4', 'path': '/m/Gorillaz/Plastic Beach - 04 - Stylo.flac', 'title': 'Stylo', 'artist': 'Gorillaz', 'album': 'Plastic Beach', 'duration': 267.1},
        ]
        old = {'item_id': 'fp_st', 'title': 'Stylo', 'author': 'Gorillaz', 'album': 'Plastic Beach', 'duration': 267.119751}
        result = CandidateIndex(tracks, duration_tolerance=1.0).match_chunk([old])
        assert result['matches'] == {'fp_st': 's4'}
        assert result['extra_matches'] == {}, 'a version 4.5 s longer is another recording, never a duplicate'

    def test_the_same_title_and_duration_by_two_different_artists_is_refused(self):
        CandidateIndex = _load_matcher().CandidateIndex

        tracks = [
            {'id': 'a', 'path': '/m/A/Intro.mp3', 'title': 'Intro', 'artist': 'Artist A', 'album': 'X', 'duration': 60.0},
            {'id': 'b', 'path': '/m/B/Intro.mp3', 'title': 'Intro', 'artist': 'Artist B', 'album': 'Y', 'duration': 60.2},
        ]
        old = {'item_id': 'fp_i', 'title': 'Intro', 'author': '[Unknown Artist]', 'album': '[Unknown Album]', 'duration': 60.1}
        result = CandidateIndex(tracks, duration_tolerance=1.0).match_chunk([old])
        assert result['matches'] == {}

    def test_every_copy_with_the_same_title_artist_and_duration_is_kept_as_a_duplicate(self):
        CandidateIndex = _load_matcher().CandidateIndex

        tracks = [
            {'id': 'flac', 'path': '/m/Brad/After Bach II - 08 - Between Bach.flac', 'title': 'Between Bach', 'artist': 'Brad Mehldau', 'album': 'After Bach II', 'duration': 301.2},
            {'id': 'mp3', 'path': '/m/Brad (1)/After Bach II - 08 - Between Bach.mp3', 'title': 'Between Bach', 'artist': 'Brad Mehldau', 'album': 'After Bach II (Deluxe)', 'duration': 301.4},
        ]
        old = {'item_id': 'fp_b', 'title': 'Between Bach', 'author': 'Brad Mehldau', 'album': 'After Bach II', 'duration': 301.25}
        result = CandidateIndex(tracks, duration_tolerance=1.0).match_chunk([old])
        assert result['matches'] == {'fp_b': 'flac'}
        assert result['extra_matches'] == {'mp3': 'fp_b'}
        assert result['extra_match_tiers'] == {'mp3': 'title_duration'}

    def test_duplicate_files_of_already_mapped_songs_only_take_unclaimed_files(self):
        CandidateIndex = _load_matcher().CandidateIndex

        tracks = [
            {'id': 'copy', 'path': '/m/x/Song.mp3', 'title': 'Song', 'artist': 'A', 'album': 'Al', 'duration': 200.0},
            {'id': 'taken', 'path': '/m/y/Song.flac', 'title': 'Song', 'artist': 'A', 'album': 'Al', 'duration': 200.3},
        ]
        mapped = {'item_id': 'fp_m', 'title': 'Song', 'author': 'A', 'album': 'Al', 'duration': 200.1}
        claimed = {'taken': 2}
        result = CandidateIndex(tracks, duration_tolerance=1.0).duplicate_files([mapped], claimed)
        assert result['matches'] == {}
        assert result['extra_matches'] == {'copy': 'fp_m'}
        assert claimed['taken'] == 2 and 'copy' in claimed

    def test_a_file_name_title_and_a_longer_artist_credit_still_match(self):
        CandidateIndex = _load_matcher().CandidateIndex
        normalize_meta = _load_matcher().normalize_meta

        assert normalize_meta('24 HoliznaCC0 - Ramen.mp3') == normalize_meta('24 HoliznaCC0 - Ramen')
        tracks = [{'id': 'j1', 'path': '/m/Slash/Apocalyptic Love - 03 - Anastasia.flac', 'title': 'Anastasia',
                   'artist': 'Slash', 'album': 'Apocalyptic Love (Deluxe)', 'duration': 367.5}]
        old = {'item_id': 'fp_a', 'title': 'Anastasia', 'author': 'Slash featuring Myles Kennedy and the Conspirators',
               'album': 'Apocalyptic Love', 'duration': 367.28}
        assert CandidateIndex(tracks, duration_tolerance=1.0).match_chunk([old])['matches'] == {'fp_a': 'j1'}

    def test_a_different_artist_credit_on_the_same_album_and_duration_is_the_same_song(self):
        CandidateIndex = _load_matcher().CandidateIndex

        tracks = [{'id': 'j1', 'path': '/m/Galantis/Church - 05 - I Found U.flac', 'title': 'I Found U',
                   'artist': 'Galantis', 'album': 'Church', 'duration': 207.386122}]
        old = {'item_id': 'fp_f', 'title': 'I Found U', 'author': 'Passion Pit', 'album': 'Church', 'duration': 207.386122}
        assert CandidateIndex(tracks, duration_tolerance=1.0).match_chunk([old])['matches'] == {'fp_f': 'j1'}

    def test_a_title_that_is_the_file_name_matches_the_target_file_name(self):
        CandidateIndex = _load_matcher().CandidateIndex

        tracks = [{'id': 'j1', 'path': '/m/Patrick Davies/12 Patrick Davies - 20 nos da.mp3', 'title': '20 nos da',
                   'artist': 'Patrick Davies', 'album': 'Sampler', 'duration': 210.8}]
        old = {'item_id': 'fp_p', 'title': '12 Patrick Davies - 20 nos da.mp3', 'author': '[Unknown Artist]',
               'album': '[Unknown Album]', 'duration': 210.8129}
        assert CandidateIndex(tracks, duration_tolerance=1.0).match_chunk([old])['matches'] == {'fp_p': 'j1'}
