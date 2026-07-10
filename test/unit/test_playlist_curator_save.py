"""Playlist Curator save-new and replace-seeded-playlist API tests."""

from unittest.mock import patch

import pytest
from flask import Flask

import app_playlist_curator


@pytest.fixture
def client():
    app = Flask(__name__)
    app.config.update(TESTING=True)  # NOSONAR -- Isolated Flask unit app; no live CSRF surface.
    app.register_blueprint(app_playlist_curator.playlist_curator_bp)
    return app.test_client()


def test_save_playlist_create_new_remains_available(client):
    with patch(
        "tasks.ivf_manager.create_playlist_from_ids", return_value="new-123"
    ) as create:
        response = client.post(
            "/api/curator/save_playlist",
            json={"new_playlist_name": "New Mix", "track_ids": ["a", "a", "b"]},
        )

    assert response.status_code == 201
    assert response.get_json() == {
        "action": "created",
        "message": "Playlist 'New Mix' created with 2 songs!",
        "playlist_id": "new-123",
        "playlist_name": "New Mix",
        "total_songs": 2,
    }
    create.assert_called_once_with("New Mix", ["a", "b"])


def test_save_playlist_replaces_existing_name(client):
    playlists = [{"Id": "seed-1", "Name": "Road Trip"}]
    replaced = {"Id": "seed-1", "Name": "Road Trip"}
    with (
        patch.object(app_playlist_curator, "_fetch_server_playlists", return_value=playlists),
        patch(
            "tasks.mediaserver.create_or_replace_playlist", return_value=replaced
        ) as replace,
        patch("tasks.ivf_manager.create_playlist_from_ids") as create,
    ):
        response = client.post(
            "/api/curator/save_playlist",
            json={"replace_playlist_name": "Road Trip", "track_ids": [1, 1, 2]},
        )

    assert response.status_code == 200
    assert response.get_json() == {
        "action": "replaced",
        "message": "Playlist 'Road Trip' replaced with 2 songs!",
        "playlist_id": "seed-1",
        "playlist_name": "Road Trip",
        "total_songs": 2,
    }
    replace.assert_called_once_with("Road Trip", ["1", "2"])
    create.assert_not_called()


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"track_ids": ["a"]}, "Provide exactly one playlist save action"),
        (
            {
                "new_playlist_name": "New",
                "replace_playlist_name": "Old",
                "track_ids": ["a"],
            },
            "Provide exactly one playlist save action",
        ),
        (
            {"replace_playlist_name": 42, "track_ids": ["a"]},
            "Playlist name must be a non-empty string",
        ),
        (
            {"replace_playlist_name": "   ", "track_ids": ["a"]},
            "Playlist name must be a non-empty string",
        ),
        (
            {"replace_playlist_name": "Road Trip", "track_ids": "a"},
            "Track IDs must be a non-empty list",
        ),
        (
            {"replace_playlist_name": "Road Trip", "track_ids": []},
            "Track IDs must be a non-empty list",
        ),
    ],
)
def test_save_playlist_rejects_invalid_action_payloads(client, payload, message):
    response = client.post("/api/curator/save_playlist", json=payload)

    assert response.status_code == 400
    assert response.get_json() == {"error": message}


def test_save_playlist_replacement_requires_existing_exact_name(client):
    with (
        patch.object(
            app_playlist_curator,
            "_fetch_server_playlists",
            return_value=[{"Id": "seed-1", "Name": "Road Trip"}],
        ),
        patch("tasks.mediaserver.create_or_replace_playlist") as replace,
    ):
        response = client.post(
            "/api/curator/save_playlist",
            json={"replace_playlist_name": "road trip", "track_ids": ["a"]},
        )

    assert response.status_code == 404
    assert response.get_json() == {"error": "Playlist 'road trip' no longer exists"}
    replace.assert_not_called()


def test_save_playlist_reports_unsupported_replacement(client):
    with (
        patch.object(
            app_playlist_curator,
            "_fetch_server_playlists",
            return_value=[{"id": "seed-1", "name": "Road Trip"}],
        ),
        patch(
            "tasks.mediaserver.create_or_replace_playlist",
            side_effect=NotImplementedError,
        ),
    ):
        response = client.post(
            "/api/curator/save_playlist",
            json={"replace_playlist_name": "Road Trip", "track_ids": ["a"]},
        )

    assert response.status_code == 501
    assert response.get_json() == {
        "error": "Replacing playlists is not supported by this media server"
    }


def test_save_playlist_treats_falsey_provider_result_as_failure(client):
    with (
        patch.object(
            app_playlist_curator,
            "_fetch_server_playlists",
            return_value=[{"Id": "seed-1", "Name": "Road Trip"}],
        ),
        patch("tasks.mediaserver.create_or_replace_playlist", return_value=None),
    ):
        response = client.post(
            "/api/curator/save_playlist",
            json={"replace_playlist_name": "Road Trip", "track_ids": ["a"]},
        )

    assert response.status_code == 502
    assert response.get_json() == {"error": "Media server failed to replace playlist"}


@pytest.mark.parametrize("payload", [7, "not-an-object", ["not", "an", "object"]])
def test_save_playlist_rejects_non_object_json(client, payload):
    response = client.post("/api/curator/save_playlist", json=payload)

    assert response.status_code == 400
    assert response.get_json() == {"error": "JSON body must be an object"}
