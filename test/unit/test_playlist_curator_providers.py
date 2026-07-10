"""Provider-route regression tests for Playlist Curator."""

from unittest.mock import Mock, patch

import pytest
from flask import Flask

import app_playlist_curator


@pytest.fixture
def client():
    app = Flask(__name__)
    app.config.update(TESTING=True)
    app.register_blueprint(app_playlist_curator.playlist_curator_bp)
    return app.test_client()


def test_plex_playlist_discovery_uses_central_dispatcher(client):
    playlists = [{"Id": "plex-list-1", "Name": "Plex Mix"}]
    with (
        patch.object(app_playlist_curator.config, "MEDIASERVER_TYPE", "plex"),
        patch("tasks.mediaserver.get_all_playlists", return_value=playlists) as get_playlists,
    ):
        response = client.get("/api/curator/server_playlists")

    assert response.status_code == 200
    assert response.get_json() == [
        {
            "playlist_id": "plex-list-1",
            "playlist_name": "Plex Mix",
            "song_count": 0,
        }
    ]
    get_playlists.assert_called_once_with()


def test_plex_playlist_track_loading_uses_central_dispatcher(client):
    cursor = Mock()
    cursor.fetchall.return_value = [("plex-track-1",), ("plex-track-2",)]
    database = Mock()
    database.cursor.return_value = cursor
    metadata = [
        {"item_id": "plex-track-1", "title": "One"},
        {"item_id": "plex-track-2", "title": "Two"},
    ]

    with (
        patch.object(app_playlist_curator.config, "MEDIASERVER_TYPE", "plex"),
        patch(
            "tasks.mediaserver.get_playlist_track_ids",
            return_value=["plex-track-1", "plex-track-2"],
        ) as get_track_ids,
        patch.object(app_playlist_curator, "get_db", return_value=database),
        patch.object(
            app_playlist_curator,
            "get_score_data_by_ids",
            return_value=metadata,
        ),
    ):
        response = client.post(
            "/api/curator/server_playlist_tracks",
            json={"playlist_id": "plex-list-1"},
        )

    assert response.status_code == 200
    assert response.get_json() == {
        "tracks": metadata,
        "total_provider_tracks": 2,
        "resolved_tracks": 2,
        "unresolved_tracks": 0,
    }
    get_track_ids.assert_called_once_with("plex-list-1")


def test_plex_stream_resolves_media_part_and_proxies_range(client):
    upstream = Mock()
    upstream.status_code = 206
    upstream.headers = {
        "Content-Type": "audio/flac",
        "Content-Length": "5",
        "Content-Range": "bytes 0-4/5",
        "Accept-Ranges": "bytes",
    }
    upstream.iter_content.return_value = [b"audio"]

    with (
        patch.object(app_playlist_curator.config, "MEDIASERVER_TYPE", "plex"),
        patch.object(app_playlist_curator.config, "PLEX_URL", "http://plex:32400"),
        patch.object(app_playlist_curator.config, "PLEX_TOKEN", "plex-token"),
        patch(
            "tasks.mediaserver.plex._resolve_part",
            return_value=("/library/parts/7/file.flac", "flac"),
        ) as resolve_part,
        patch.object(
            app_playlist_curator.http_requests,
            "get",
            return_value=upstream,
        ) as get_stream,
    ):
        response = client.get(
            "/api/curator/stream/plex-track-1",
            headers={"Range": "bytes=0-4"},
        )

    assert response.status_code == 206
    assert response.data == b"audio"
    assert response.headers["Content-Type"] == "audio/flac"
    resolve_part.assert_called_once_with("plex-track-1")
    get_stream.assert_called_once_with(
        "http://plex:32400/library/parts/7/file.flac",
        params=None,
        headers={"X-Plex-Token": "plex-token", "Range": "bytes=0-4"},
        stream=True,
        timeout=(10, 60),
        allow_redirects=True,
    )


def test_mpd_playlist_track_loading_remains_unsupported(client):
    with (
        patch.object(app_playlist_curator.config, "MEDIASERVER_TYPE", "mpd"),
        patch("tasks.mediaserver.get_playlist_track_ids") as get_track_ids,
    ):
        response = client.post(
            "/api/curator/server_playlist_tracks",
            json={"playlist_id": "mpd-list-1"},
        )

    assert response.status_code == 501
    assert response.get_json() == {
        "error": "MPD is not supported by the playlist curator"
    }
    get_track_ids.assert_not_called()
