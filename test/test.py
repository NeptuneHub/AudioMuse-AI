import pytest
import requests
import time
import re

# Update the BASE_URL to point to your running API server
BASE_URL = 'http://192.168.3.14:8000'
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


@pytest.mark.parametrize('algorithm,use_embedding,pca_max', [
    ('kmeans', True, 199),
    ('gmm', True, 199),
    ('spectral', True, 199),
    ('dbscan', True, 199),
    ('kmeans', False, None),
    ('gmm', False, None),
    ('spectral', False, None),
    ('dbscan', False, None),
])
def test_clustering_smoke_flow(algorithm, use_embedding, pca_max):
    start_time = time.time()
    payload = {
        'clustering_method': algorithm,
        'enable_clustering_embeddings': use_embedding,
        'clustering_runs': 100,
        'stratified_sampling_target_percentile': 10
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

    final = wait_for_success(task_id, timeout=1200)
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


def test_annoy_similarity_and_playlist():
    start_time = time.time()
    sim_resp = get_with_retries(
        f'{BASE_URL}/api/similar_tracks',
        params={'title': 'By the Way', 'artist': 'Red Hot Chili Peppers', 'n': 1}
    )
    assert sim_resp.status_code == 200
    sim_data = sim_resp.json()
    assert isinstance(sim_data, list) and sim_data
    item_id = sim_data[0].get('item_id')
    assert item_id

    pl_resp = requests.post(
        f'{BASE_URL}/api/create_playlist',
        json={'playlist_name': 'TestPlaylist', 'track_ids': [item_id]}
    )
    assert pl_resp.status_code == 201
    pl_data = pl_resp.json()
    assert 'playlist_id' in pl_data

    elapsed = time.time() - start_time
    print(f"[TIMING] Similar Song test completed in {elapsed:.2f} seconds")


if __name__ == '__main__':
    import sys
    # Run pytest in verbose mode with live output (-s)
    sys.exit(pytest.main(['-v', '-s', __file__]))
