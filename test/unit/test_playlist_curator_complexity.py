# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only

"""Behavioral characterization for Playlist Curator complexity refactors.

Main Features:
* Verifies duplicate grouping, missing-vector handling, and track ranking.
* Verifies SQL-paged smart search preserves result ordering.
* Verifies extender filtering and result metadata enrichment.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest
from flask import Flask

import app_playlist_curator


@pytest.fixture
def client():
    app = Flask(__name__)  # NOSONAR -- Isolated unit-only app; no network surface.
    app.register_blueprint(app_playlist_curator.playlist_curator_bp)
    return app.test_client()


def test_duplicate_groups_ignore_missing_vectors_and_rank_tracks():
    vectors = {
        "a": np.array([1.0, 0.0]),
        "b": np.array([0.9999, 0.01]),
        "c": np.array([0.0, 1.0]),
    }
    metadata = [
        {
            "item_id": "a",
            "title": "Best copy",
            "author": "Artist",
            "album": "Album",
            "album_artist": "Artist",
            "year": 2020,
            "rating": 5,
        },
        {
            "item_id": "b",
            "title": "Other copy",
            "author": "Artist",
            "album": "Album",
            "album_artist": None,
            "year": None,
            "rating": None,
        },
    ]

    with (
        patch.object(app_playlist_curator, "get_vectors_by_ids", return_value=vectors),
        patch.object(app_playlist_curator, "get_score_data_by_ids", return_value=metadata),
    ):
        result = app_playlist_curator._find_duplicate_groups(
            ["a", "b", "c", "missing"], threshold=0.001
        )

    assert result["total_groups"] == 1
    assert result["total_duplicate_tracks"] == 2
    assert [track["item_id"] for track in result["groups"][0]["tracks"]] == [
        "a",
        "b",
    ]
    assert result["groups"][0]["tracks"][0]["score"] == 6.0


def test_search_only_pages_in_sql_and_preserves_database_id_order(client):
    cursor = Mock()
    cursor.fetchone.return_value = {"total": 5}
    cursor.fetchall.return_value = [{"item_id": "b"}, {"item_id": "a"}]
    database = Mock()
    database.cursor.return_value = cursor
    metadata = [
        {"item_id": "a", "title": "A"},
        {"item_id": "b", "title": "B"},
    ]

    with (
        patch.object(app_playlist_curator, "get_db", return_value=database),
        patch.object(
            app_playlist_curator,
            "get_score_data_lite_by_ids",
            return_value=metadata,
        ),
    ):
        response = client.post(
            "/api/curator/search",
            json={
                "filters": [
                    {"field": "artist", "operator": "contains", "value": "Artist"}
                ],
                "search_only": True,
                "page": 2,
                "per_page": 2,
            },
        )

    assert response.status_code == 200
    assert response.get_json() == {
        "results": [
            {"distance": 0.0, "item_id": "b", "title": "B"},
            {"distance": 0.0, "item_id": "a", "title": "A"},
        ],
        "total": 5,
        "page": 2,
        "per_page": 2,
        "has_more": True,
        "playlist_song_count": 5,
        "included_count": 0,
        "excluded_count": 0,
    }
    assert cursor.execute.call_count == 2
    assert cursor.execute.call_args_list[1].args[1] == ("%Artist%", 2, 2)
    cursor.close.assert_called_once_with()


def test_extend_search_filters_neighbors_and_enriches_survivor(client):
    cursor = Mock()
    database = Mock()
    database.cursor.return_value = cursor
    neighbors = [
        {"item_id": "seed", "distance": 0.0},
        {"item_id": "low-rating", "distance": 0.1},
        {"item_id": "old", "distance": 0.1},
        {"item_id": "far", "distance": 0.8},
        {"item_id": "match", "distance": 0.2},
    ]
    candidate_metadata = [
        {"item_id": "low-rating", "rating": 2, "year": 2020},
        {"item_id": "old", "rating": 5, "year": 1990},
        {"item_id": "far", "rating": 5, "year": 2020},
        {
            "item_id": "match",
            "title": "Matched title",
            "author": "Matched artist",
            "album": "Matched album",
            "album_artist": "Matched album artist",
            "rating": 5,
            "year": 2020,
        },
    ]
    source_metadata = [{"item_id": "seed", "title": "Seed"}]

    def metadata_for(item_ids):
        return source_metadata if item_ids == ["seed"] else candidate_metadata

    with (
        patch.object(app_playlist_curator, "get_db", return_value=database),
        patch.object(
            app_playlist_curator,
            "get_vectors_by_ids",
            return_value={"seed": np.array([1.0, 0.0])},
        ),
        patch.object(
            app_playlist_curator,
            "find_nearest_neighbors_by_vector",
            return_value=neighbors,
        ) as nearest,
        patch.object(
            app_playlist_curator,
            "get_score_data_by_ids",
            side_effect=metadata_for,
        ),
    ):
        response = client.post(
            "/api/curator/search",
            json={
                "source_ids": ["seed"],
                "duplicate_threshold": 0,
                "min_rating": 4,
                "year_min": 2000,
                "similarity_threshold": 0.5,
                "max_songs": 2,
            },
        )

    assert response.status_code == 200
    assert response.get_json() == {
        "results": [
            {
                "item_id": "match",
                "distance": 0.2,
                "title": "Matched title",
                "author": "Matched artist",
                "album": "Matched album",
                "album_artist": "Matched album artist",
                "year": 2020,
            }
        ],
        "playlist_song_count": 1,
        "included_count": 0,
        "excluded_count": 0,
        "source_tracks": source_metadata,
    }
    nearest.assert_called_once()
    assert nearest.call_args.kwargs == {"n": 500, "eliminate_duplicates": True}
    cursor.close.assert_called_once_with()
