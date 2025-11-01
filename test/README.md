# AudioMuse-AI Developer Tests

These integration tests are for developer purposes to verify the functionality of the AudioMuse-AI API endpoints. They are not included in the production Docker container and should be run from a local development machine.

## Prerequisites

- Python 3.8+
- `pip` and `venv`
- Jellyfin installed with 40+ song albums
- Have already run the analysis of 40+ albums with AudioMuse-AI (the analysis test jur run 1 album for test, it is not enough for other test)

## Setup Instructions

You can choose to clone the entire repository or just the `test` directory.

### Clone the  Repository

This is the simplest method.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/NeptuneHub/AudioMuse-AI.git
    ```

2.  **Navigate to the test directory:**
    ```bash
    cd AudioMuse-AI/test
    ```

### Environment Setup (from within the `test` directory)

Once you are inside the `test` directory, follow these steps:

1.  **Create and activate a Python virtual environment:**

    *   **On macOS/Linux:**
        ```bash
        python3 -m venv .venv
        source .venv/bin/activate
        ```
    *   **On Windows:**
        ```bash
        python -m venv .venv
        .\.venv\Scripts\activate
        ```

2.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure the API endpoint:**
    Open the `test.py` file and update the `BASE_URL` to point to your running AudioMuse-AI instance.
    ```python
    # test/test.py
    BASE_URL = 'http://YOUR_AUDIOMUSE_IP:8000'
    ```

## Running the Tests

To run all tests, execute the following command from your terminal (while in the `test` directory with the virtual environment activated):

```bash
python test.py
```

## Succesfull result example
If everything go well, you should have  result output similar to this:

```
(.venv) user@host:~/project$ python test2.py
======================================== test session starts =========================================
platform linux -- Python 3.12.3, pytest-8.4.1, pluggy-1.6.0 -- ~/project/.venv/bin/python
cachedir: .pytest_cache
rootdir: ~/project
collected 10 items

test2.py::test_analysis_smoke_flow [TIMING] Analysis completed in 268.57 seconds
PASSED
test2.py::test_clustering_smoke_flow[kmeans-True-199] [RESULT] Algorithm=kmeans | BestScore=17.439000162267043 | PlaylistsCreated=100 | Time=103.19s
PASSED
test2.py::test_clustering_smoke_flow[gmm-True-199] [RESULT] Algorithm=gmm | BestScore=17.70349717246473 | PlaylistsCreated=94 | Time=507.78s
PASSED
test2.py::test_clustering_smoke_flow[spectral-True-199] [RESULT] Algorithm=spectral | BestScore=17.58542598899186 | PlaylistsCreated=97 | Time=119.28s
PASSED
test2.py::test_clustering_smoke_flow[dbscan-True-199] [RESULT] Algorithm=dbscan | BestScore=6.556085438351011 | PlaylistsCreated=3 | Time=40.87s
PASSED
test2.py::test_clustering_smoke_flow[kmeans-False-None] [RESULT] Algorithm=kmeans | BestScore=2.2492027954091296 | PlaylistsCreated=82 | Time=72.09s
PASSED
test2.py::test_clustering_smoke_flow[gmm-False-None] [RESULT] Algorithm=gmm | BestScore=2.373805929047702 | PlaylistsCreated=93 | Time=1019.34s
PASSED
test2.py::test_clustering_smoke_flow[spectral-False-None] [RESULT] Algorithm=spectral | BestScore=2.391899050948851 | PlaylistsCreated=79 | Time=108.71s
PASSED
test2.py::test_clustering_smoke_flow[dbscan-False-None] [RESULT] Algorithm=dbscan | BestScore=-0.1758302493574464 | PlaylistsCreated=8 | Time=34.87s
PASSED
test2.py::test_annoy_similarity_and_playlist [TIMING] Annoy test completed in 0.09 seconds
PASSED

======================================================================================== 10 passed in 2364.92s (0:39:24) ========================================================================================
(.venv) user@host:~/project$
```

## Running tests through an reverse proxy

The current docker-compose.yaml starts audiomuse-ai with all dependencies, jellyfin and nginx.
It is setup to run behind a reverse proxy. The started audiomuse-ai can be accessed via the proxy `http://localhost:7777/am` or directly with `http://localhost:8000`.
The started jellyfin is not configured to allow to run the test successfull.
If you want to run the tests against an existing audiomse-ai instance you have to change the nginx configuration to point to it.
See `./nginx-confd/reverse-proxy.conf` and the line with `proxy_pass http://audiomuse-ai-flask:8000/;`.

Do run the tests against it you have to start execute: `docker compose up`.
Set the base url to `BASE_URL = 'http://localhost:7777/am'`
Your audiomuse instance should have set the following configuration to true `ENABLE_PROXY_FIX`.
And run the test by executing: `python test.py`.

