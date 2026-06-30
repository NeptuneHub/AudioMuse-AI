from unittest.mock import patch

import pytest
from flask import Flask

from app_helper import serialize_neighbor_results
import app_ivf


def score(item_id, title='T', author='A', album='Album', album_artist='AlbArt',
          mood_vector='rock:0.9,happy:0.5', other_features='danceable:0.4'):
    return {
        'item_id': item_id,
        'title': title,
        'author': author,
        'album': album,
        'album_artist': album_artist,
        'mood_vector': mood_vector,
        'other_features': other_features,
    }


def neighbors(*pairs):
    return [{'item_id': iid, 'distance': dist} for iid, dist in pairs]


class TestSerializeShortCircuits:
    def test_empty_list_returns_empty_without_lookup(self):
        with patch('app_helper.get_score_data_by_ids') as mock_lookup:
            assert serialize_neighbor_results([]) == []
            mock_lookup.assert_not_called()

    def test_none_returns_empty_without_lookup(self):
        with patch('app_helper.get_score_data_by_ids') as mock_lookup:
            assert serialize_neighbor_results(None) == []
            mock_lookup.assert_not_called()


class TestMissingAlbumDefault:
    def test_none_album_substituted_with_default_unknown(self):
        with patch('app_helper.get_score_data_by_ids',
                   return_value=[score('1', album=None)]):
            out = serialize_neighbor_results(neighbors(('1', 0.1)))
        assert out[0]['album'] == 'unknown'

    def test_empty_album_substituted_with_default_unknown(self):
        with patch('app_helper.get_score_data_by_ids',
                   return_value=[score('1', album='')]):
            out = serialize_neighbor_results(neighbors(('1', 0.1)))
        assert out[0]['album'] == 'unknown'

    def test_present_album_preserved_with_default(self):
        with patch('app_helper.get_score_data_by_ids',
                   return_value=[score('1', album='Real Album')]):
            out = serialize_neighbor_results(neighbors(('1', 0.1)))
        assert out[0]['album'] == 'Real Album'

    def test_custom_sentinel_substitutes(self):
        with patch('app_helper.get_score_data_by_ids',
                   return_value=[score('1', album=None)]):
            out = serialize_neighbor_results(
                neighbors(('1', 0.1)), missing_album='N/A')
        assert out[0]['album'] == 'N/A'


class TestMissingAlbumNoneContract:
    def test_none_album_kept_as_none(self):
        with patch('app_helper.get_score_data_by_ids',
                   return_value=[score('1', album=None)]):
            out = serialize_neighbor_results(
                neighbors(('1', 0.1)), missing_album=None)
        assert out[0]['album'] is None

    def test_empty_album_kept_as_empty(self):
        with patch('app_helper.get_score_data_by_ids',
                   return_value=[score('1', album='')]):
            out = serialize_neighbor_results(
                neighbors(('1', 0.1)), missing_album=None)
        assert out[0]['album'] == ''

    def test_present_album_kept_with_none_sentinel(self):
        with patch('app_helper.get_score_data_by_ids',
                   return_value=[score('1', album='Real Album')]):
            out = serialize_neighbor_results(
                neighbors(('1', 0.1)), missing_album=None)
        assert out[0]['album'] == 'Real Album'


class TestMissingDetailsSkipped:
    def test_id_absent_from_details_map_is_skipped(self):
        with patch('app_helper.get_score_data_by_ids',
                   return_value=[score('1')]):
            out = serialize_neighbor_results(neighbors(('1', 0.1), ('2', 0.2)))
        assert len(out) == 1
        assert out[0]['item_id'] == '1'

    def test_all_ids_absent_yields_empty(self):
        with patch('app_helper.get_score_data_by_ids', return_value=[]):
            out = serialize_neighbor_results(neighbors(('1', 0.1), ('2', 0.2)))
        assert out == []


class TestDistancePassthrough:
    def test_distance_comes_from_neighbor_map(self):
        with patch('app_helper.get_score_data_by_ids',
                   return_value=[score('1'), score('2')]):
            out = serialize_neighbor_results(neighbors(('1', 0.25), ('2', 0.75)))
        by_id = {r['item_id']: r['distance'] for r in out}
        assert by_id == {'1': 0.25, '2': 0.75}

    def test_distance_zero_preserved(self):
        with patch('app_helper.get_score_data_by_ids',
                   return_value=[score('1')]):
            out = serialize_neighbor_results(neighbors(('1', 0.0)))
        assert out[0]['distance'] == pytest.approx(0.0)


class TestFeatureFlagsPassthrough:
    def test_mood_and_other_features_present(self):
        with patch('app_helper.get_score_data_by_ids',
                   return_value=[score('1', mood_vector='metal:0.8',
                                       other_features='danceable:0.9')]):
            out = serialize_neighbor_results(neighbors(('1', 0.1)))
        assert out[0]['mood_vector'] == 'metal:0.8'
        assert out[0]['other_features'] == 'danceable:0.9'
        assert out[0]['top_genre'] == 'metal'

    def test_missing_features_become_none(self):
        with patch('app_helper.get_score_data_by_ids',
                   return_value=[score('1', mood_vector=None,
                                       other_features=None)]):
            out = serialize_neighbor_results(neighbors(('1', 0.1)))
        assert out[0]['mood_vector'] is None
        assert out[0]['other_features'] is None
        assert out[0]['top_genre'] is None


class TestAlbumArtistFlag:
    def test_album_artist_included_when_flag_true(self):
        with patch('app_helper.get_score_data_by_ids',
                   return_value=[score('1', album_artist='Band')]):
            out = serialize_neighbor_results(
                neighbors(('1', 0.1)), include_album_artist=True)
        assert out[0]['album_artist'] == 'Band'

    def test_album_artist_falls_back_to_unknown(self):
        with patch('app_helper.get_score_data_by_ids',
                   return_value=[score('1', album_artist=None)]):
            out = serialize_neighbor_results(
                neighbors(('1', 0.1)), include_album_artist=True)
        assert out[0]['album_artist'] == 'unknown'

    def test_album_artist_omitted_when_flag_false(self):
        with patch('app_helper.get_score_data_by_ids',
                   return_value=[score('1', album_artist='Band')]):
            out = serialize_neighbor_results(
                neighbors(('1', 0.1)), include_album_artist=False)
        assert 'album_artist' not in out[0]


class TestOutputOrder:
    def test_output_follows_input_neighbor_order(self):
        with patch('app_helper.get_score_data_by_ids',
                   return_value=[score('3'), score('1'), score('2')]):
            out = serialize_neighbor_results(
                neighbors(('1', 0.1), ('2', 0.2), ('3', 0.3)))
        assert [r['item_id'] for r in out] == ['1', '2', '3']

    def test_core_fields_present(self):
        with patch('app_helper.get_score_data_by_ids',
                   return_value=[score('1', title='Song', author='Artist')]):
            out = serialize_neighbor_results(neighbors(('1', 0.5)))
        row = out[0]
        assert row['item_id'] == '1'
        assert row['title'] == 'Song'
        assert row['author'] == 'Artist'


@pytest.fixture
def flask_ctx():
    app = Flask(__name__)
    with app.app_context():
        yield


class TestNeighborSearchErrorResponse:
    def test_runtime_error_maps_to_503(self, flask_ctx):
        resp, status = app_ivf._neighbor_search_error_response(
            'item-1', RuntimeError('boom'), is_runtime=True)
        assert status == 503
        body = resp.get_json()
        assert body['error'] == 'The similarity search service is currently unavailable.'

    def test_unexpected_error_maps_to_500(self, flask_ctx):
        resp, status = app_ivf._neighbor_search_error_response(
            'item-1', ValueError('boom'), is_runtime=False)
        assert status == 500
        body = resp.get_json()
        assert body['error'] == 'An unexpected error occurred.'

    def test_message_does_not_leak_exception_text(self, flask_ctx):
        secret = 'TRACEBACK_SECRET_42'
        for is_runtime in (True, False):
            resp, _status = app_ivf._neighbor_search_error_response(
                'ctx', RuntimeError(secret), is_runtime=is_runtime)
            assert secret not in resp.get_json()['error']

    def test_context_value_not_leaked_to_client(self, flask_ctx):
        resp, _status = app_ivf._neighbor_search_error_response(
            'sensitive-item-id', RuntimeError('x'), is_runtime=True)
        assert 'sensitive-item-id' not in resp.get_json()['error']
