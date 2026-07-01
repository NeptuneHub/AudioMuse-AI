# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Live HTTP smoke tests that drive a running AudioMuse-AI server.

Posts to the real REST API over the network, polling task status to
completion, and asserts each response shape and the follow-on playlist
creation for the core end-user flows.

Main Features:
* Analysis and clustering task start plus poll-to-SUCCESS.
* Instant chat playlist, sonic fingerprint, song alchemy, map, similar
  tracks, and song-path flows each creating a playlist.
"""

import pytest
import warnings
import requests
import time

BASE_URL = 'http://192.168.3.97:8000'
RETRIES = 3
RETRY_DELAY = 2


def get_with_retries(url, **kwargs):
    for attempt in range(1, RETRIES + 1):
        try:
            resp = requests.get(url, **kwargs)
            return resp
        except requests.RequestException:
            if attempt == RETRIES:
                raise
            time.sleep(RETRY_DELAY)


def wait_for_success(task_id, timeout=1200):
    start = time.time()
    while time.time() - start < timeout:
        act_resp = get_with_retries(f'{BASE_URL}/api/active_tasks')
        act_resp.raise_for_status()
        active = act_resp.json()
        if active and active.get('task_id') == task_id:
            time.sleep(1)
            continue
        last_resp = get_with_retries(f'{BASE_URL}/api/last_task')
        last_resp.raise_for_status()
        final = last_resp.json()
        final_id = final.get('task_id')
        final_state = (final.get('status') or final.get('state') or '').upper()
        if final_id == task_id and final_state == 'SUCCESS':
            return final
        pytest.fail(f'Task {task_id} final state is {final_state}, expected SUCCESS')
    pytest.fail(f'Task {task_id} did not reach SUCCESS within {timeout} seconds')


def test_analysis_smoke_flow():
    start_time = time.time()
    resp = requests.post(
        f'{BASE_URL}/api/analysis/start', json={'num_recent_albums': 1, 'top_n_moods': 5}
    )
    assert resp.status_code == 202
    data = resp.json()
    assert data.get('task_type') == 'main_analysis'
    task_id = data.get('task_id')
    assert task_id

    final = wait_for_success(task_id, timeout=1200)
    assert final.get('task_type') == 'main_analysis'
    assert final.get('status', final.get('state')) == 'SUCCESS'

    elapsed = time.time() - start_time
    print(f"[TIMING] Analysis completed in {elapsed:.2f} seconds")
    time.sleep(10)


def test_instant_playlist_functionality():
    chat_base_url = BASE_URL
    chat_payload = {
        "userInput": "High energy song",
        "ai_provider": "GEMINI",
        "ai_model": "gemini-2.5-pro",
    }
    chat_resp = requests.post(
        f'{chat_base_url}/chat/api/chatPlaylist', json=chat_payload, timeout=120
    )
    assert chat_resp.status_code == 200, f"Status: {chat_resp.status_code}, Body: {chat_resp.text}"
    chat_data = chat_resp.json()
    assert "response" in chat_data, f"Missing 'response' key: {chat_data}"
    resp_obj = chat_data["response"]
    expected_keys = {
        "ai_model_selected",
        "ai_provider_used",
        "executed_query",
        "message",
        "original_request",
        "query_results",
    }
    missing_keys = expected_keys - set(resp_obj.keys())
    if missing_keys:
        warnings.warn(
            f"Shape warning: chatPlaylist response missing keys: {missing_keys} in {resp_obj}"
        )
    results = resp_obj.get("query_results", [])
    assert isinstance(results, list) and results, (
        f"No query_results in chatPlaylist response: {resp_obj}"
    )
    for track in results:
        assert all(k in track for k in ("item_id", "title", "artist")), (
            f"Track missing keys: {track}"
        )
    item_ids = [track["item_id"] for track in results]
    pl_payload = {"playlist_name": "functional_test", "item_ids": item_ids}
    pl_resp = requests.post(
        f'{chat_base_url}/chat/api/create_playlist', json=pl_payload, timeout=120
    )
    assert pl_resp.status_code == 200, f"Status: {pl_resp.status_code}, Body: {pl_resp.text}"
    pl_data = pl_resp.json()
    assert "message" in pl_data, f"Missing 'message' in playlist creation response: {pl_data}"
    assert "created playlist" in pl_data["message"].lower(), (
        f"Unexpected playlist creation message: {pl_data['message']}"
    )
    print(
        f"[TIMING] Instant Playlist test completed successfully. Playlist message: {pl_data['message']}"
    )


def test_sonic_fingerprint_and_playlist():
    start_time = time.time()
    payload = {'n': 1, 'jellyfin_user_identifier': 'admin', 'jellyfin_token': ''}
    resp = requests.post(f'{BASE_URL}/api/sonic_fingerprint/generate', json=payload, timeout=120)
    assert resp.status_code == 200, f"Status: {resp.status_code}, Body: {resp.text}"
    data = resp.json()
    if not (isinstance(data, list) and data and isinstance(data[0], dict) and 'item_id' in data[0]):
        warnings.warn(f"Shape warning: sonic_fingerprint response shape unexpected: {data}")
    track_ids = [track['item_id'] for track in data if 'item_id' in track]
    assert track_ids

    pl_resp = requests.post(
        f'{BASE_URL}/api/create_playlist',
        json={'playlist_name': 'TestSonicFingerprint', 'track_ids': track_ids},
        timeout=120,
    )
    assert pl_resp.status_code == 201, f"Status: {pl_resp.status_code}, Body: {pl_resp.text}"
    pl_data = pl_resp.json()
    if 'playlist_id' not in pl_data:
        warnings.warn(f"Shape warning: create_playlist response shape unexpected: {pl_data}")
    assert 'playlist_id' in pl_data

    elapsed = time.time() - start_time
    print(f"[TIMING] Sonic Fingerprint test completed in {elapsed:.2f} seconds")


def test_song_alchemy_and_playlist():
    start_time = time.time()

    def find_song_id(artist, title):
        resp = get_with_retries(
            f"{BASE_URL}/api/search_tracks", params={"artist": artist, "title": title}
        )
        assert resp.status_code == 200, (
            f"Search failed: Status: {resp.status_code}, Body: {resp.text}"
        )
        results = resp.json()
        for song in results:
            song_artist = song.get("author") or song.get("artist") or ""
            if (
                song_artist.lower() == artist.lower()
                and song.get("title", "").lower() == title.lower()
            ):
                return song["item_id"]
        if results and "item_id" in results[0]:
            return results[0]["item_id"]
        pytest.fail(f"Could not find song id for {artist} - {title}. Response: {results}")

    add_id = find_song_id('Red Hot Chili Peppers', 'By the Way')
    sub_id = find_song_id('System of a Down', 'Attack')

    payload = {
        "items": [{"id": add_id, "op": "ADD"}, {"id": sub_id, "op": "SUBTRACT"}],
        "n": 10,
        "temperature": 1,
        "subtract_distance": 0.2,
    }
    resp = requests.post(f'{BASE_URL}/api/alchemy', json=payload, timeout=120)
    assert resp.status_code == 200, f"Status: {resp.status_code}, Body: {resp.text}"
    data = resp.json()
    expected_keys = {
        "add_centroid_2d",
        "add_points",
        "centroid_2d",
        "filtered_out",
        "projection",
        "results",
        "sub_points",
        "subtract_centroid_2d",
    }
    if not (isinstance(data, dict) and expected_keys.issubset(data.keys())):
        warnings.warn(f"Shape warning: /api/alchemy response shape unexpected: {data}")
    results = data.get('results', [])
    assert isinstance(results, list) and results, f"No results in alchemy response: {data}"
    track_ids = [track['item_id'] for track in results if 'item_id' in track]
    assert track_ids

    pl_resp = requests.post(
        f'{BASE_URL}/api/create_playlist',
        json={'playlist_name': 'TestSongAlchemy', 'track_ids': track_ids},
        timeout=120,
    )
    assert pl_resp.status_code == 201, f"Status: {pl_resp.status_code}, Body: {pl_resp.text}"
    pl_data = pl_resp.json()
    assert 'playlist_id' in pl_data

    elapsed = time.time() - start_time
    print(f"[TIMING] Song Alchemy test completed in {elapsed:.2f} seconds")


def test_map_visualization():
    start_time = time.time()
    resp = requests.get(f'{BASE_URL}/api/map?percent=25', timeout=120)
    assert resp.status_code == 200, f"Status: {resp.status_code}, Body: {resp.text}"
    data = resp.json()
    assert 'items' in data and isinstance(data['items'], list), f"Response: {data}"

    sim_resp = get_with_retries(
        f'{BASE_URL}/api/similar_tracks',
        params={'title': 'By the Way', 'artist': 'Red Hot Chili Peppers', 'n': 1},
    )
    assert sim_resp.status_code == 200, f"Status: {sim_resp.status_code}, Body: {sim_resp.text}"
    sim_data = sim_resp.json()
    assert isinstance(sim_data, list) and sim_data, f"Response: {sim_data}"
    item_id = sim_data[0].get('item_id')
    assert item_id

    pl_resp = requests.post(
        f'{BASE_URL}/api/create_playlist',
        json={'playlist_name': 'TestPlaylist', 'track_ids': [item_id]},
    )
    assert pl_resp.status_code == 201, f"Status: {pl_resp.status_code}, Body: {pl_resp.text}"
    pl_data = pl_resp.json()
    assert 'playlist_id' in pl_data

    elapsed = time.time() - start_time
    print(f"[TIMING] Map Visualization test completed in {elapsed:.2f} seconds")


def test_annoy_similarity_and_playlist():
    start_time = time.time()
    sim_resp = get_with_retries(
        f'{BASE_URL}/api/similar_tracks',
        params={'title': 'By the Way', 'artist': 'Red Hot Chili Peppers', 'n': 1},
    )
    assert sim_resp.status_code == 200, f"Status: {sim_resp.status_code}, Body: {sim_resp.text}"
    sim_data = sim_resp.json()
    assert isinstance(sim_data, list) and sim_data, f"Response: {sim_data}"
    item_id = sim_data[0].get('item_id')
    assert item_id

    pl_resp = requests.post(
        f'{BASE_URL}/api/create_playlist',
        json={'playlist_name': 'TestPlaylist', 'track_ids': [item_id]},
    )
    assert pl_resp.status_code == 201, f"Status: {pl_resp.status_code}, Body: {pl_resp.text}"
    pl_data = pl_resp.json()
    assert 'playlist_id' in pl_data

    elapsed = time.time() - start_time
    print(f"[TIMING] Similar Song test completed in {elapsed:.2f} seconds")


def test_song_path_and_playlist():
    start_time = time.time()

    def find_song_id(artist, title):
        resp = get_with_retries(
            f"{BASE_URL}/api/search_tracks", params={"artist": artist, "title": title}
        )
        assert resp.status_code == 200, (
            f"Search failed: Status: {resp.status_code}, Body: {resp.text}"
        )
        results = resp.json()
        for song in results:
            song_artist = song.get("author") or song.get("artist") or ""
            if (
                song_artist.lower() == artist.lower()
                and song.get("title", "").lower() == title.lower()
            ):
                return song["item_id"]
        if results and "item_id" in results[0]:
            return results[0]["item_id"]
        pytest.fail(f"Could not find song id for {artist} - {title}. Response: {results}")

    start_artist = 'Red Hot Chili Peppers'
    start_title = 'By the Way'
    end_artist = 'System of a Down'
    end_title = 'Attack'

    start_song_id = find_song_id(start_artist, start_title)
    end_song_id = find_song_id(end_artist, end_title)

    params = {'start_song_id': start_song_id, 'end_song_id': end_song_id, 'max_steps': 25}
    resp = requests.get(f'{BASE_URL}/api/find_path', params=params, timeout=120)
    assert resp.status_code == 200, f"Status: {resp.status_code}, Body: {resp.text}"
    data = resp.json()
    if isinstance(data, dict) and 'path' in data:
        path = data['path']
    else:
        path = data
    assert isinstance(path, list) and path, f"Response: {data}"
    track_ids = [track['item_id'] for track in path if 'item_id' in track]
    assert track_ids

    pl_resp = requests.post(
        f'{BASE_URL}/api/create_playlist',
        json={'playlist_name': 'TestSongPath', 'track_ids': track_ids},
        timeout=120,
    )
    assert pl_resp.status_code == 201, f"Status: {pl_resp.status_code}, Body: {pl_resp.text}"
    pl_data = pl_resp.json()
    assert 'playlist_id' in pl_data

    elapsed = time.time() - start_time
    print(f"[TIMING] Song Path test completed in {elapsed:.2f} seconds")


@pytest.mark.parametrize(
    'algorithm,use_embedding,pca_max',
    [
        ('kmeans', True, 199),
        ('gmm', True, 199),
        ('spectral', True, 199),
        ('dbscan', True, 199),
    ],
)
def test_clustering_smoke_flow(algorithm, use_embedding, pca_max):
    start_time = time.time()
    payload = {
        'clustering_method': algorithm,
        'enable_clustering_embeddings': use_embedding,
        'clustering_runs': 20,
        'stratified_sampling_target_percentile': 1,
    }
    if use_embedding:
        payload['pca_components_max'] = pca_max

    resp = requests.post(f'{BASE_URL}/api/clustering/start', json=payload)
    assert resp.status_code == 202
    data = resp.json()
    assert data.get('task_type') == 'main_clustering'
    task_id = data.get('task_id')
    assert task_id

    final = wait_for_success(task_id, timeout=4800)
    assert final.get('task_type') == 'main_clustering'
    assert final.get('status', final.get('state')) == 'SUCCESS'

    details = final.get('details', {})
    best_score = details.get('best_score')
    num_playlists = details.get('num_playlists_created')

    assert best_score is not None, "Best score not found in details"
    assert num_playlists is not None, "Number of playlists created not found in details"

    elapsed = time.time() - start_time
    print(
        f"[RESULT] Algorithm={algorithm} | BestScore={best_score} | PlaylistsCreated={num_playlists} | Time={elapsed:.2f}s"
    )
    time.sleep(10)


if __name__ == '__main__':
    import sys

    sys.exit(pytest.main(['-v', '-s', __file__]))
