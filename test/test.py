#
# Setup instructions (Ubuntu CLI):
#
# 1. Create a virtual environment:
#      python3 -m venv .venv
#
# 2. Activate the virtual environment:
#      source .venv/bin/activate
#
# 3. Install requirements:
#      pip install -r requirements.txt
#
# 4. Set the BASE_URL variable below to point to Media Server
#
# Example commands to run tests:
#
# Run all tests:
#   pytest -v -s test.py
#
# Run a single test by name (replace <test_name> with the function name):
#   pytest -v -s test.py -k <test_name>
#
# Examples for each test:
#   pytest -v -s test.py -k test_analysis_smoke_flow
#   pytest -v -s test.py -k test_instant_playlist_functionality
#   pytest -v -s test.py -k test_sonic_fingerprint_and_playlist
#   pytest -v -s test.py -k test_song_alchemy_and_playlist
#   pytest -v -s test.py -k test_map_visualization
#   pytest -v -s test.py -k test_annoy_similarity_and_playlist
#   pytest -v -s test.py -k test_song_path_and_playlist
#   pytest -v -s test.py -k test_clustering_smoke_flow
#
import pytest
import warnings
import requests
import time
import re

# Update the BASE_URL to point to your running API server
BASE_URL = 'http://192.168.3.97:8000'
RETRIES = 3
RETRY_DELAY = 2  # seconds between retries


def get_with_retries(url, **kwargs):
    """Make an HTTP GET request with retries on connection errors."""
    for attempt in range(1, RETRIES + 1):
        try:
            resp = requests.get(url, **kwargs)
            return resp
        except requests.RequestException:
            if attempt == RETRIES:
                raise
            time.sleep(RETRY_DELAY)


def wait_for_success(task_id, timeout=1200):  # timeout extended to 20 minutes (1200s)
    """Poll the active_tasks endpoint until the task is no longer active, then verify final status via last_task."""
    start = time.time()
    while time.time() - start < timeout:
        # Check if task is still active
        act_resp = get_with_retries(f'{BASE_URL}/api/active_tasks')
        act_resp.raise_for_status()
        active = act_resp.json()
        # If still active, wait a moment
        if active and active.get('task_id') == task_id:
            time.sleep(1)
            continue
        # No longer active; fetch the final status
        last_resp = get_with_retries(f'{BASE_URL}/api/last_task')
        last_resp.raise_for_status()
        final = last_resp.json()
        final_id = final.get('task_id')
        final_state = (final.get('status') or final.get('state') or '').upper()
        if final_id == task_id and final_state == 'SUCCESS':
            return final
        pytest.fail(f'Task {task_id} final state is {final_state}, expected SUCCESS')
    pytest.fail(f'Task {task_id} did not reach SUCCESS within {timeout} seconds')


#pytest -v -s test.py -k test_analysis_smoke_flow
def test_analysis_smoke_flow():
    start_time = time.time()
    resp = requests.post(
        f'{BASE_URL}/api/analysis/start',
        json={'num_recent_albums': 1, 'top_n_moods': 5}
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


# pytest -v -s test.py -k test_instant_playlist_functionality
def test_instant_playlist_functionality():
    """Test the Instant Playlist (chatPlaylist) and playlist creation endpoints."""
    # Use the same BASE_URL as other tests, but allow override for chat endpoints if needed
    chat_base_url = BASE_URL  # Change if chat API is on a different host/port
    # 1. Call chatPlaylist endpoint
    chat_payload = {
        "userInput": "High energy song",
        "ai_provider": "GEMINI",
        "ai_model": "gemini-2.5-pro"
    }
    chat_resp = requests.post(f'{chat_base_url}/chat/api/chatPlaylist', json=chat_payload, timeout=120)
    assert chat_resp.status_code == 200, f"Status: {chat_resp.status_code}, Body: {chat_resp.text}"
    chat_data = chat_resp.json()
    # Check response shape
    assert "response" in chat_data, f"Missing 'response' key: {chat_data}"
    resp_obj = chat_data["response"]
    expected_keys = {"ai_model_selected", "ai_provider_used", "executed_query", "message", "original_request", "query_results"}
    missing_keys = expected_keys - set(resp_obj.keys())
    if missing_keys:
        warnings.warn(f"Shape warning: chatPlaylist response missing keys: {missing_keys} in {resp_obj}")
    # Check query_results is a non-empty list of dicts with item_id, title, artist
    results = resp_obj.get("query_results", [])
    assert isinstance(results, list) and results, f"No query_results in chatPlaylist response: {resp_obj}"
    for track in results:
        assert all(k in track for k in ("item_id", "title", "artist")), f"Track missing keys: {track}"
    # 2. Create playlist with returned item_ids
    item_ids = [track["item_id"] for track in results]
    pl_payload = {"playlist_name": "functional_test", "item_ids": item_ids}
    pl_resp = requests.post(f'{chat_base_url}/chat/api/create_playlist', json=pl_payload, timeout=120)
    assert pl_resp.status_code == 200, f"Status: {pl_resp.status_code}, Body: {pl_resp.text}"
    pl_data = pl_resp.json()
    # Check playlist creation response shape
    assert "message" in pl_data, f"Missing 'message' in playlist creation response: {pl_data}"
    assert "created playlist" in pl_data["message"].lower(), f"Unexpected playlist creation message: {pl_data['message']}"
    print(f"[TIMING] Instant Playlist test completed successfully. Playlist message: {pl_data['message']}")



# pytest -v -s test.py -k test_sonic_fingerprint_and_playlist
def test_sonic_fingerprint_and_playlist():
    """Test /api/sonic_fingerprint/generate and playlist creation."""
    start_time = time.time()
    payload = {'n': 1, 'jellyfin_user_identifier': 'admin', 'jellyfin_token': ''}
    resp = requests.post(f'{BASE_URL}/api/sonic_fingerprint/generate', json=payload, timeout=120)
    assert resp.status_code == 200, f"Status: {resp.status_code}, Body: {resp.text}"
    data = resp.json()
    # Example shape: [{'item_id': ..., ...}]
    if not (isinstance(data, list) and data and isinstance(data[0], dict) and 'item_id' in data[0]):
        warnings.warn(f"Shape warning: sonic_fingerprint response shape unexpected: {data}")
    track_ids = [track['item_id'] for track in data if 'item_id' in track]
    assert track_ids

    pl_resp = requests.post(
        f'{BASE_URL}/api/create_playlist',
        json={'playlist_name': 'TestSonicFingerprint', 'track_ids': track_ids},
        timeout=120
    )
    assert pl_resp.status_code == 201, f"Status: {pl_resp.status_code}, Body: {pl_resp.text}"
    pl_data = pl_resp.json()
    if 'playlist_id' not in pl_data:
        warnings.warn(f"Shape warning: create_playlist response shape unexpected: {pl_data}")
    assert 'playlist_id' in pl_data

    elapsed = time.time() - start_time
    print(f"[TIMING] Sonic Fingerprint test completed in {elapsed:.2f} seconds")

#pytest -v -s test.py -k test_song_alchemy_and_playlist
def test_song_alchemy_and_playlist():
    """Test /api/alchemy and playlist creation."""
    start_time = time.time()
    # Helper to find song id
    def find_song_id(artist, title):
        resp = get_with_retries(
            f"{BASE_URL}/api/search_tracks",
            params={"artist": artist, "title": title}
        )
        assert resp.status_code == 200, f"Search failed: Status: {resp.status_code}, Body: {resp.text}"
        results = resp.json()
        for song in results:
            song_artist = song.get("author") or song.get("artist") or ""
            if song_artist.lower() == artist.lower() and song.get("title", "").lower() == title.lower():
                return song["item_id"]
        if results and "item_id" in results[0]:
            return results[0]["item_id"]
        pytest.fail(f"Could not find song id for {artist} - {title}. Response: {results}")

    # Find item_ids for the two example songs
    add_id = find_song_id('Red Hot Chili Peppers', 'By the Way')
    sub_id = find_song_id('System of a Down', 'Attack')

    payload = {
        "items": [
            {"id": add_id, "op": "ADD"},
            {"id": sub_id, "op": "SUBTRACT"}
        ],
        "n": 10,
        "temperature": 1,
        "subtract_distance": 0.2
    }
    resp = requests.post(f'{BASE_URL}/api/alchemy', json=payload, timeout=120)
    assert resp.status_code == 200, f"Status: {resp.status_code}, Body: {resp.text}"
    data = resp.json()
    # Check shape: must be dict with keys like 'add_centroid_2d', 'add_points', 'centroid_2d', 'filtered_out', 'projection', 'results', 'sub_points', 'subtract_centroid_2d'
    expected_keys = {"add_centroid_2d", "add_points", "centroid_2d", "filtered_out", "projection", "results", "sub_points", "subtract_centroid_2d"}
    if not (isinstance(data, dict) and expected_keys.issubset(data.keys())):
        warnings.warn(f"Shape warning: /api/alchemy response shape unexpected: {data}")
    # Use the 'results' field for playlist creation
    results = data.get('results', [])
    assert isinstance(results, list) and results, f"No results in alchemy response: {data}"
    track_ids = [track['item_id'] for track in results if 'item_id' in track]
    assert track_ids

    pl_resp = requests.post(
        f'{BASE_URL}/api/create_playlist',
        json={'playlist_name': 'TestSongAlchemy', 'track_ids': track_ids},
        timeout=120
    )
    assert pl_resp.status_code == 201, f"Status: {pl_resp.status_code}, Body: {pl_resp.text}"
    pl_data = pl_resp.json()
    assert 'playlist_id' in pl_data

    elapsed = time.time() - start_time
    print(f"[TIMING] Song Alchemy test completed in {elapsed:.2f} seconds")

#pytest -v -s test.py -k test_map_visualization
def test_map_visualization():
    """Test /api/map visualization endpoint."""
    start_time = time.time()
    resp = requests.get(f'{BASE_URL}/api/map?n=10', timeout=120)
    assert resp.status_code == 200, f"Status: {resp.status_code}, Body: {resp.text}"
    data = resp.json()
    assert 'items' in data and isinstance(data['items'], list), f"Response: {data}"

    # Also test a minimal similarity and playlist creation as in the UI
    sim_resp = get_with_retries(
        f'{BASE_URL}/api/similar_tracks',
        params={'title': 'By the Way', 'artist': 'Red Hot Chili Peppers', 'n': 1}
    )
    assert sim_resp.status_code == 200, f"Status: {sim_resp.status_code}, Body: {sim_resp.text}"
    sim_data = sim_resp.json()
    assert isinstance(sim_data, list) and sim_data, f"Response: {sim_data}"
    item_id = sim_data[0].get('item_id')
    assert item_id

    pl_resp = requests.post(
        f'{BASE_URL}/api/create_playlist',
        json={'playlist_name': 'TestPlaylist', 'track_ids': [item_id]}
    )
    assert pl_resp.status_code == 201, f"Status: {pl_resp.status_code}, Body: {pl_resp.text}"
    pl_data = pl_resp.json()
    assert 'playlist_id' in pl_data

    elapsed = time.time() - start_time
    print(f"[TIMING] Map Visualization test completed in {elapsed:.2f} seconds")

import pytest

#pytest -v -s test.py -k test_annoy_similarity_and_playlist
def test_annoy_similarity_and_playlist():
    start_time = time.time()
    sim_resp = get_with_retries(
        f'{BASE_URL}/api/similar_tracks',
        params={'title': 'By the Way', 'artist': 'Red Hot Chili Peppers', 'n': 1}
    )
    assert sim_resp.status_code == 200, f"Status: {sim_resp.status_code}, Body: {sim_resp.text}"
    sim_data = sim_resp.json()
    assert isinstance(sim_data, list) and sim_data, f"Response: {sim_data}"
    item_id = sim_data[0].get('item_id')
    assert item_id

    pl_resp = requests.post(
        f'{BASE_URL}/api/create_playlist',
        json={'playlist_name': 'TestPlaylist', 'track_ids': [item_id]}
    )
    assert pl_resp.status_code == 201, f"Status: {pl_resp.status_code}, Body: {pl_resp.text}"
    pl_data = pl_resp.json()
    assert 'playlist_id' in pl_data

    elapsed = time.time() - start_time
    print(f"[TIMING] Similar Song test completed in {elapsed:.2f} seconds")

#pytest -v -s test.py -k test_song_path_and_playlist
def test_song_path_and_playlist():
    """Test /api/path and playlist creation."""
    start_time = time.time()
    # First, search for the start and end songs to get their item_ids
    def find_song_id(artist, title):
        # Use the correct endpoint and match the real API usage
        resp = get_with_retries(
            f"{BASE_URL}/api/search_tracks",
            params={"artist": artist, "title": title}
        )
        assert resp.status_code == 200, f"Search failed: Status: {resp.status_code}, Body: {resp.text}"
        results = resp.json()
        # Try to find the best match (author or artist)
        for song in results:
            song_artist = song.get("author") or song.get("artist") or ""
            if song_artist.lower() == artist.lower() and song.get("title", "").lower() == title.lower():
                return song["item_id"]
        # fallback: return first result if available
        if results and "item_id" in results[0]:
            return results[0]["item_id"]
        pytest.fail(f"Could not find song id for {artist} - {title}. Response: {results}")

    start_artist = 'Red Hot Chili Peppers'
    start_title = 'By the Way'
    end_artist = 'System of a Down'
    end_title = 'Attack'

    start_song_id = find_song_id(start_artist, start_title)
    end_song_id = find_song_id(end_artist, end_title)

    params = {
        'start_song_id': start_song_id,
        'end_song_id': end_song_id,
        'max_steps': 25
    }
    # Use GET for /api/find_path as in the real API
    resp = requests.get(f'{BASE_URL}/api/find_path', params=params, timeout=120)
    assert resp.status_code == 200, f"Status: {resp.status_code}, Body: {resp.text}"
    data = resp.json()
    # Accept either a list or a dict with 'path' key
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
        timeout=120
    )
    assert pl_resp.status_code == 201, f"Status: {pl_resp.status_code}, Body: {pl_resp.text}"
    pl_data = pl_resp.json()
    assert 'playlist_id' in pl_data

    elapsed = time.time() - start_time
    print(f"[TIMING] Song Path test completed in {elapsed:.2f} seconds")

@pytest.mark.parametrize('algorithm,use_embedding,pca_max', [
    ('kmeans', True, 199),
    ('gmm', True, 199),
    ('spectral', True, 199),
    ('dbscan', True, 199),
])
#pytest -v -s test.py -k test_clustering_smoke_flow
def test_clustering_smoke_flow(algorithm, use_embedding, pca_max):
    start_time = time.time()
    payload = {
        'clustering_method': algorithm,
        'enable_clustering_embeddings': use_embedding,
        'clustering_runs': 20,
        'stratified_sampling_target_percentile': 1
    }
    if use_embedding:
        payload['pca_components_max'] = pca_max

    resp = requests.post(
        f'{BASE_URL}/api/clustering/start', json=payload
    )
    assert resp.status_code == 202
    data = resp.json()
    assert data.get('task_type') == 'main_clustering'
    task_id = data.get('task_id')
    assert task_id

    final = wait_for_success(task_id, timeout=4800)  # 80 minutes
    assert final.get('task_type') == 'main_clustering'
    assert final.get('status', final.get('state')) == 'SUCCESS'

    details = final.get('details', {})
    best_score = details.get('best_score')
    num_playlists = details.get('num_playlists_created')

    assert best_score is not None, "Best score not found in details"
    assert num_playlists is not None, "Number of playlists created not found in details"

    elapsed = time.time() - start_time
    print(f"[RESULT] Algorithm={algorithm} | BestScore={best_score} | PlaylistsCreated={num_playlists} | Time={elapsed:.2f}s")
    time.sleep(10)

if __name__ == '__main__':
    import sys
    # Run pytest in verbose mode with live output (-s)
    sys.exit(pytest.main(['-v', '-s', __file__]))
