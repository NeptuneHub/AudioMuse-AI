## Quick developer pointers (where to inspect code)
- Path: `tasks/path_manager.py` — see `find_path_between_songs`, `_find_best_unique_song`, `interpolate_centroids`.
- Alchemy: `tasks/song_alchemy.py` — see `song_alchemy`, `_compute_centroid_from_ids`, `_project_*` helpers; `templates/alchemy.html` for client wiring.
- Map: `app_map.py` — see `build_map_cache`, `_sample_items`, `/api/map`; `templates/map.html` for client behavior.
- Sonic fingerprint: `tasks/sonic_fingerprint_manager.py` — see `generate_sonic_fingerprint`; `templates/sonic_fingerprint.html` for UI.

If you'd like, I can also add a minimal flow diagram or a short call-sequence per feature (function A → function B → DB → index) to this page.
# Algorithm

This document explains the runtime algorithms implemented in the repository. It focuses on the current behavior in code (file and function references are provided so you can inspect the implementation directly). Outdated historical notes (for example references to a k-means minibatch experiment or an earlier TensorFlow/ONNX mention) were removed — the sections below describe what the code actually runs today.

## **Table of Contents**

- [Front-End Quick Start: Analysis and Clustering Parameters](#front-end-quick-start-analysis-and-clustering-parameters)
- [Instant Playlist (via Chat Interface)](#instant-playlist-via-chat-interface)
- [Playlist from Similar song (via similarity Interface)](#playlist-from-similar-song-via-similarity-interface)
- [Sonic Fingerprint playlist (via sonic_fingerprint Interface)](#sonic-fingerprint-playlist-via-sonic_fingerprint-interface)
- [Song Path playlist (via path Interface)](#song-path-playlist-via-path-interface)
- [Workflow Overview](#workflow-overview)
- [Analysis Algorithm Deep Dive](#analysis-algorithm-deep-dive)
- [Clustering Algorithm Deep Dive](#clustering-algorithm-deep-dive)
 - [Song Path Deep Dive - Deep dive](#song-path-deep-dive---deep-dive)
 - [Song Alchemy - Deep dive](#song-alchemy---deep-dive)
 - [Music Map - Deep dive](#music-map---deep-dive)
 - [Sonic Fingerprint - Deep dive](#sonic-fingerprint---deep-dive)
 - [Workflow Overview](#workflow-overview)
 - [Analysis Algorithm Deep Dive](#analysis-algorithm-deep-dive)
 - [Clustering Algorithm Deep Dive](#clustering-algorithm-deep-dive)
  - [1. K-Means](#1-k-means)
  - [2. DBSCAN](#2-dbscan)
  - [3. GMM (Gaussian Mixture Models)](#3-gmm-gaussian-mixture-models)
  - [4. Spectral Clustering](#4-spectral-clustering)
  - [Montecarlo Evolutionary Approach](#montecarlo-evolutionary-approach)
  - [How Purity and Diversity Scores Are Calculated](#how-purity-and-diversity-scores-are-calculated)
  - [AI Playlist Naming](#ai-playlist-naming)
- [Concurrency Algorithm Deep Dive](#concurrency-algorithm-deep-dive)
- [Instant Chat Deep Dive](#instant-chat-deep-dive)
- [Playlist from Similar song - Deep dive](#playlist-from-similar-song---deep-dive)



## **Front-End Quick Start: Analysis and Clustering Parameters**

After deploying with the K3S Quick Start, you'll want to run an **Analysis Task** first to process your music library, followed by a **Clustering Task** to generate playlists. Here are the most important parameters to consider for your first few runs, accessible via the UI or API:

### **Analysis Task Quick Start**

1.  **`NUM_RECENT_ALBUMS`** (Default: `0`)
    *   How many of your most recently added albums to scan and analyze. Set to `0` to analyze *all* albums in your library (can take a very long time for large libraries).
    *   **Recommendation:** For a first run, you might want to set this to a smaller number (e.g., `50`, `100`) to see results quickly. For a full analysis, use `0` or a large number.

### **Clustering Task Quick Start**

1.  **`CLUSTER_ALGORITHM`** (Default: `kmeans`)
    *   **Recommendation:** For most users, especially when starting, **`kmeans`** is recommended. It's the fastest algorithm and works well when you have a general idea of the number of playlists you'd like to generate. The other algorithms (`gmm`, `dbscan`) are available for more advanced experimentation.

2.  **K-Means Specific: `NUM_CLUSTERS_MIN` & `NUM_CLUSTERS_MAX`**
    *   **`NUM_CLUSTERS_MIN`** (Default: `40`): The minimum number of playlists (clusters) the algorithm should try to create.
    *   **`NUM_CLUSTERS_MAX`** (Default: `100`): The maximum number of playlists (clusters) the algorithm should try to create. (Note: K-Means generally performs well with feature vectors).
    *   **Guidance:**
        *   Think about how many distinct playlists you'd ideally like. These parameters define the range the evolutionary algorithm will explore for the K-Means `K` value.
        *   The number of clusters cannot exceed the number of songs in the dataset being clustered for a given run. The system will automatically cap the `K` value if your `NUM_CLUSTERS_MAX` is too high for the available songs in a particular iteration's sample.
        *   For a smaller library or a quick test, you might reduce both `NUM_CLUSTERS_MIN` and `NUM_CLUSTERS_MAX` (e.g., min 10, max 30). For larger libraries, the defaults are a reasonable starting point.

3.  **`CLUSTERING_RUNS`** (Default: `5000`)
    *   This is the number of iterations the evolutionary algorithm will perform. More runs mean a more thorough search for good playlist configurations but will take longer.
    *   **Recommendation:** For a quick test, you can reduce this to `500`-`1000`. For better results, keep it high.

4.  **Scoring Weights (Primary)**:
    *   **`SCORE_WEIGHT_DIVERSITY`** (Default: `2.0`): How much to prioritize variety *between* different playlists (based on their main mood).
    *   **`SCORE_WEIGHT_PURITY`** (Default: `1.0`): How much to prioritize consistency *within* each playlist (songs matching the playlist's main mood).
    *   **Recommendation:** Start with these defaults. If you want more varied playlists, increase `SCORE_WEIGHT_DIVERSITY`. If you want playlists where songs are very similar to each other, increase `SCORE_WEIGHT_PURITY`.
    *   **Note:** Other weights like `SCORE_WEIGHT_SILHOUETTE`, `SCORE_WEIGHT_OTHER_FEATURE_DIVERSITY`, etc., default to `0.0` (disabled). They are actually there for future test and implementation

5.  **`MAX_SONGS_PER_CLUSTER`** (Default: `0` - no limit)
    *   If you want to limit the maximum number of songs in any single generated playlist, set this to a positive number (e.g., `20`, `30`). In the case of limitation is set, the algorithm will split the playlist in two or more.

6.  **AI Playlist Naming (`AI_MODEL_PROVIDER`)** (Default: `NONE`)
    *   If you've configured Ollama, Gemini or Mistral (see `GEMINI_API_KEY` or `MISTRAL_API_KEY`  in secrets, and `OLLAMA_SERVER_URL` in the ConfigMap), you can set this to `OLLAMA`,  `GEMINI` or `MISTRAL` to get AI-generated playlist names. Otherwise, playlists will have names like "Rock_Fast_Automatic". For a first run you can keep it as `NONE`.

**To run your first tasks:**
*   Go to the UI (`http://<EXTERNAL-IP>:8000`).
*   Start Analysis: adjust `NUM_RECENT_ALBUMS` if desired, and submit. Wait for it to complete (it can takes a couple of days depending on your library size).
*   Start Clustering: Adjust the clustering parameters above as desired in the form, and Submit. Wait for it to complete (it can takes between minutes and 1-2 hours depending on your library size and the number of `CLUSTERING_RUNS`).

## **Instant PLaylist (via Chat Interface)**

**IMPORTANT:** before use this function you need to run the Analysis task first from the normal (async) UI.

For a quick and interactive way to generate playlists without running the full evolutionary clustering task, you can use the "Instant Playlist" chat interface. This feature leverages AI (Ollama, Gemini or Mistral, if configured) to translate your natural language requests directly into SQL queries that are run against your analyzed music data.

**How to Use:**

1.  **Access the Chat Interface:**
    *   Navigate to `http://<EXTERNAL-IP>:8000/chat` (or `http://localhost:8000/chat` for local Docker Compose deployments).

2.  **Configure AI (Optional but Recommended):**
    *   Select your preferred AI Provider (Ollama, Gemini or Mistral).
    *   If using Ollama, ensure the "Ollama Server URL" and "Ollama Model" are correctly set (they will default to values from your server configuration).
    *   If using Gemini or Mistral, enter your "API Key" and select the "Model" (defaults are provided from server config).
    *   If you select "None" as the AI Provider, the system will not attempt to generate SQL from your text.

3.  **Make Your Request:**
    *   In the text area "What kind of music are you in the mood for?", type your playlist request in natural language.
    *   Click "Get Playlist Idea".

4.  **Review and Create:**
    *   The AI will generate a PostgreSQL query based on your request. This query is then executed against your `score` table.
    *   The results (a list of songs) will be displayed.
    *   If songs are found, a new section will appear allowing you to name the playlist and click "Let's do it" to create this playlist directly on your Jellyfin, Navidrome, or Lyrion server. The playlist name will have `_instant` appended to the name you provide.

**Example Queries (Tested with Gemini):**
*   "Create a playlist that is good for post lunch"
*   "Create a playlist for the morning POP with a good energy"
*   "Give me the tops songs of Red Hot Chili peppers"
*   "Create a mix of Metal and Hard Rock songs like AC DC, Iron Maiden"
*   "Give me some tranding songs of the radio of 2025"

**Note:** The quality and relevance of the **Instant Playlist** heavily depend on the capabilities of the configured AI model and the detail of your request. The underlying music data must have been previously analyzed using the "Analysis Task" for this feature to find songs.

## **Playlist from Similar song (via similarity Interface)**

**IMPORTANT:** before use this function you need to run the Analysis task first from the normal (async) UI.

This new functionality enable you to search the top N similar song that are similar to another one. Basically during the analysis task an Approximate Nearest Neighbors (Voyager) index is made. Then with the new similarity interface you can just search for similar song.

**How to Use:**
1.  **Access the Chat Interface:**
    *   Navigate to `http://<EXTERNAL-IP>:8000/similarity` (or `http://localhost:8000//similarity` for local Docker Compose deployments).
2.  **Input Your Song**
    *   Start writing the first 3+ letter of your favourite artist, at the third letter a search will be made helping you in finding correct Artist and song
3.  **Run the similarity search**
    *   Ask the front-end to find the similar track, it will show to you in the table
4.  **Review and Create:**
    *   Input a name for the playlist and ask the interface to create it directly on Jellyfin, Navidrome, or Lyrion. That's it!

## **Sonic Fingerprint playlist (via sonic_fingerprint Interface)**

**IMPORTANT:** before use this function you need to run the Analysis task first from the normal (async) UI.

This new functionality analyze your listening history and create your specific sonic fingerprint. With it it takes advance of Voyager Index (the sonic similarity function) to search for similar songs. With this similar song it enable you to create your personal playlist.

**How to Use:**
1.  **Access the Chat Interface:**
    *   Navigate to `http://<EXTERNAL-IP>:8000/sonic_fingerprint` (or `http://localhost:8000/sonic_fingerprint` for local Docker Compose deployments).
2.  **Input Your Username**
    *   It get by default your username configurde in env variable, but if you have multiple user you can input it in the front-end to a personalized playlist
3.  **Input Your Password or API token**
    *   On jellyfin you can still use your API Token, for Navidrome you will need to use the password of the specific user. By default it takes the value from the env variable
4.  **Select the number of song**
    *  Select the number of similar song, from a minimum of 40 to a maximum of 200
5.  **Run the Sonic Fingerprint search**
    *   Ask the front-end to generate the list of Sonic Fingerprint track, it will show to you in the table.
6.  **Review and Create:**
    *   Input a name for the playlist and ask the interface to create it directly on Jellyfin, Navidrome, or Lyrion. That's it!

**Note for Lyrion users:** Lyrion doesn't require user credentials since it doesn't have user-specific authentication like Jellyfin or Navidrome. The interface will automatically handle this for Lyrion deployments.

## **Song Path playlist (via path Interface)**

**IMPORTANT:** before use this function you need to run the Analysis task first from the normal (async) UI.

This new functionality create a sonic similar path between two song asked from the user.

**How to Use:**
1.  **Access the Chat Interface:**
    *   Navigate to `http://<EXTERNAL-IP>:8000/path` (or `http://localhost:8000/path` for local Docker Compose deployments).
3.  **Input start and end song**
    *   Insert Artist and Title of both start and end song. When you start input the first 3 char of a title or artist the front-end will give you suggestions.
4.  **Select the lenght of the path**
    *  Select the number of song to keep in the path. First and last will be the song that you insert.
5.  **Run the path search**
    *   Ask the front-end to generate the path, it will show to you in the table.
6.  **Review and Create:**
    *   Input a name for the playlist and ask the interface to create it directly on Jellyfin, Navidrome, or Lyrion. That's it!

## **Collection Sync (via collection Interface)**

**IMPORTANT:** this function sync your datatabase in a centralized one on internet. You can decide IF and WHEN using this function by explicitly accepting the [Privacy Policy](https://github.com/NeptuneHub/AudioMuse-AI/blob/main/PRIVACY.md), doing an explicit login and explicitly run the functionality.

**How to Use:**
1.  **Access the Collection Interface:**
    *   Navigate to `http://<EXTERNAL-IP>:8000/collection` (or `http://localhost:8000/collection` for local Docker Compose deployments).
2.  **Accept the Security Policy and do the Oauth Login:**
    *   To work you need first to read the Security Policy and check the checkbox, second to do an Oauth Login. Actually supported only with Github
3.  **Select the number of last album to sync**
    *   Like the offline analysis, you need to input how many last album you want to sync. If you put `0` it will sync all
4.  **Start Syncronization**
    *   The syncronization will sync Score and Embedding database online working on the last album order. Basically it pick an album and check: if **THE ANALYSIS DATA** are presents on your local database, it try to send online (to be shared with other user). If they are not present offline, it try to get from the online database.

The actual use case is to retrive the analysis from the online database without the need of an analysis AND to contribute to the database itself with the song that you already analyzed

**IMPORANT** this functionality only share analysis data **NOT** the song itself.

The login is required mainly to avoid high traffic on the central database. If you don't like to share information on internet, just don't use it.

## **Workflow Overview**

This is the main workflow of how this algorithm works. For an easy way to use it, you will have a front-end reachable at **your\_ip:8000** with buttons to start/cancel the analysis (it can take hours depending on the number of songs in your Jellyfin library) and a button to create the playlist.

*   **User Initiation:** Start analysis or clustering jobs via the Flask web UI.
*   **Task Queuing:** Jobs are sent to Redis Queue for asynchronous background processing.
*   **Parallel Worker Execution:**
    *   Multiple RQ workers (at least 2 recommended) process tasks in parallel. Main tasks (e.g., full library analysis, entire evolutionary clustering process) often spawn and manage child tasks (e.g., per-album analysis, batches of clustering iterations).
    *   **Analysis Phase:**
        *   Workers fetch metadata and download audio from Jellyfin or Navidrome, processing albums individually.
        *   Librosa and TensorFlow models analyze tracks for features (tempo, key, energy) and predictions (genres, moods, etc.).
        *   Analysis results are saved to PostgreSQL.
    *   **Clustering Phase:**
        *   The option to use embeddings for clustering is currently available in the "Advanced" section of the UI.
        *   The system can use either the traditional feature vectors (derived from tempo, moods, energy, etc.) or the richer MusiCNN embeddings directly for clustering, based on user configuration.
        *   An evolutionary algorithm performs numerous clustering runs (e.g., K-Means, DBSCAN, or GMM) on the analyzed data. It explores different parameters to find optimal playlist structures based on a comprehensive scoring system.
*   **Playlist Generation & Naming:**
    *   Playlists are formed from the best clustering solution found by the evolutionary process.
    *   Optionally, AI models (Ollama, Gemini or Mistral) can be used to generate creative, human-readable names for these playlists.
    *   Finalized playlists are created directly in your Jellyfin or Navidrome library.
*   **Advanced Task Management:**
    *   The web UI offers real-time monitoring of task progress, including main and sub-tasks.
    *   **Worker Supervision and High Availability:** In scenarios with multiple worker container instances, the system incorporates mechanisms to maintain high availability. HA is achived using **supervisord** and re-enquequing configuration in Redis Queue. For HA Redis and PostgreSQL must also be deployed in HA (deployment example in this repository don't cover this possibility ad the moment, so you need to change it)
    *  **Task Cancellation** A key feature is the ability to cancel tasks (both parent and child) even while they are actively running, offering robust control over long processes. This is more advanced than typical queue systems where cancellation might only affect pending tasks.


## Analysis Algorithm — deep dive

Purpose: extract a compact, reusable representation for every track (features + embedding) so downstream tasks (clustering, similarity, alchemy, path, fingerprint) can operate efficiently and deterministically.

Core steps (implementation highlights):

- Entrypoint(s): analysis tasks enqueued from the UI; per-album worker function is `analyze_album_task` (see `tasks/analysis.py`).
- Audio loading & preprocessing: tracks are downloaded and loaded with `librosa.load(file_path, sr=16000, mono=True)` to match the MusiCNN training configuration. A mel-spectrogram is computed with fixed parameters (n_fft=512, hop_length=256, n_mels=96, window='hann', center=False, power=2.0, norm='slaney') and scaled with `np.log10(1 + 10000 * mel_spec)`.
- Embedding generation: each track produces a 200-dim MusiCNN embedding via the `TensorflowPredictMusiCNN` wrapper (model: `msd-musicnn-1.pb`, output layer `model/dense/BiasAdd`). Embeddings are always stored in the DB so they can be reused by downstream tasks.
- Prediction heads: a set of lightweight ONNX/TensorFlow heads consume the embedding to predict `MOOD_LABELS` (genre/tag probabilities) and other binary-like features (danceable, aggressive, happy, party, relaxed, sad). These outputs become part of the track's `mood_vector` and `other_features` fields.
- Feature vector assembly: when the system uses interpretable features instead of raw embeddings, `score_vector` builds a single vector per track by concatenating normalized tempo, normalized average energy, mood probabilities, and other predicted feature scores. Tempo/energy are min/max scaled; the combined vector is then standardized with `StandardScaler` and persisted (scaler stats are saved to allow centroid inverse-transforms).

Key notes and edge cases:

- Embeddings are the canonical, high-dimensional representation — many features and tasks default to using them (controlled by `ENABLE_CLUSTERING_EMBEDDINGS`).
- Strict preprocessing parameters are crucial: mismatch produces incompatible inputs for the frozen MusiCNN graph and leads to meaningless embeddings.
- Missing timestamps or metadata are handled gracefully (defaults or smaller weights used); analysis tasks are idempotent so re-running an album updates the DB rows.

Persistence: PostgreSQL stores `score` and `embedding` rows; the Voyager ANN index is built from those embeddings in a later step.

## Clustering — deep dive

Purpose: group tracks into coherent playlists by experimenting with algorithm and parameter choices using a Monte Carlo + evolutionary search, then pick the best-scoring solution.

Core steps (implementation highlights):

- Entrypoint(s): `run_clustering_task` and its batch jobs (see `tasks/clustering.py`). The evolutionary search executes many clustering runs (default: `CLUSTERING_RUNS`).
- Input choices: clustering can run on interpretable score-vectors (`score_vector`) or on raw 200-dim MusiCNN embeddings (controlled by `ENABLE_CLUSTERING_EMBEDDINGS`). When embeddings are used, MiniBatchKMeans is preferred for speed.
- Algorithms available: K-Means (fast, default), GMM (probabilistic, flexible), DBSCAN (density-based, outlier-friendly), Spectral (rarely used — heavy). The system explores multiple algorithms/parameter sets per run.
- Stratified sampling & perturbation: each run samples a subset of songs (with configurable stratified genres to ensure underrepresented styles are included). Subsequent runs re-use most of the previous sample and swap a fraction (`SAMPLING_PERCENTAGE_CHANGE_PER_RUN`) to introduce variation.
- Evolutionary elements: the best parameter sets (elites) are retained; later runs sometimes mutate elites instead of sampling entirely new random parameters, balancing exploration and exploitation.
- Scoring: each clustering outcome is scored by a weighted composite: purity, diversity, and internal metrics (silhouette / Davies-Bouldin / Calinski-Harabasz). Weights live in `config.py` and drive the optimizer.

Key notes and edge cases:

- Start with K-Means / MiniBatchKMeans for speed and to get a working baseline. Try GMM when using embeddings for potentially richer clusters. DBSCAN is useful if you expect many outliers.
- The `TOP_PLAYLIST_NUMBER` or similar post-filter keeps only the most diverse playlists to avoid creating hundreds of similar lists.
- Parameter ranges and mutation deltas are tuned in `config.py`; raising `CLUSTERING_RUNS` yields better exploration at the cost of runtime.

### How Purity and Diversity Scores Are Calculated

The **Purity** and **Diversity** scores are custom-designed metrics used to evaluate the quality of clustered playlists based on their musical characteristics — especially mood. These scores guide the evolutionary clustering algorithm to balance two goals:

* **Purity** — how well songs within a playlist reflect its core mood identity
* **Diversity** — how different playlists are from one another in their dominant mood

These metrics apply consistently whether clustering is based on **interpretable score vectors** (like mood labels) or **non-interpretable embeddings**.

**6.1 Forming the Playlist Profile**

Each playlist (i.e., cluster) is assigned a **personality profile** — a vector that defines its musical identity.

* **If clustering used score vectors**:
  The cluster centroid is directly interpretable — e.g., `indie: 0.6`, `pop: 0.4`, `tempo: 0.8`.

* **If clustering used embeddings**:
  The system computes the **average of the original score vectors** of the songs in the playlist to form the profile.

The resulting profile determines the playlist’s **top moods** and dominant characteristics.


**6.2 Mood Purity – Intra-Playlist Consistency**

The **Mood Purity** score answers:

> *“How strongly do the songs in a playlist reflect its most defining moods?”*

**How it works:**

1. From the playlist’s profile, extract the **top K moods**.
2. For each song:

   * Check which of those moods also exist in the **`active_moods` list** — i.e., moods actually encoded in the song's feature vector.
   * Among those, take the **maximum score**.
3. Sum these values across all songs in the playlist.

 If a mood is in the top K but **not in `active_moods`**, it is **skipped**.

**Example:**

* **Playlist profile (top moods):** `pop: 0.6`, `indie: 0.4`, `vocal: 0.35`
* **Top moods:** `["pop", "indie", "vocal"]`
* **Active moods in song features:** `["indie", "rock", "vocal"]` (**pop is missing**)

| Song | Mood Scores                        | Used Moods (top ∩ active) | Max Score (used) |
| ---- | ---------------------------------- | ------------------------- | ---------------- |
| A    | indie: 0.3, rock: 0.7, vocal: 0.6  | indie, vocal              | **0.6**          |
| B    | indie: 0.4, rock: 0.45, vocal: 0.3 | indie, vocal              | **0.4**          |

`pop` is ignored — not in `active_moods`
**Raw Purity = 0.6 + 0.4 = 1.0**

This score is:

* Transformed using `np.log1p`
* Normalized using `LN_MOOD_PURITY_STATS`

A **high purity score** means that most songs strongly match the playlist’s actual mood focus.


**Mood Diversity – Inter-Playlist Variety**

The **Mood Diversity** score answers:

> *“How different are the playlists from each other in mood identity?”*

**How it works:**

1. For each playlist, find its **dominant mood** — the highest mood score in its profile.
2. Keep track of **unique dominant moods** across all playlists.
3. Sum the scores of those unique moods.

**Example:**

| Playlist | Profile                            | Dominant Mood | Score |
| -------- | ---------------------------------- | ------------- | ----- |
| P1       | indie: 0.6, rock: 0.3, vocal: 0.2  | **indie**     | 0.6   |
| P2       | pop: 0.5, indie: 0.3, vocal: 0.1   | **pop**       | 0.5   |
| P3       | vocal: 0.55, indie: 0.4, rock: 0.2 | **vocal**     | 0.55  |

**Raw Diversity = 0.6 + 0.5 + 0.55 = 1.65**

This score is:

* Transformed using `np.log1p`
* Normalized using `LN_MOOD_DIVERSITY_STATS`

A **high diversity score** means playlists explore a wide range of moods, rather than clustering around a single genre or vibe.

**Comparison: Purity & Diversity vs. Silhouette Score**

| Metric               | Measures What?                                  | Label-Aware | Cluster-Aware | Complexity   | Interpretation Strength     |
| -------------------- | ----------------------------------------------- | ----------- | ------------- | ------------ | --------------------------- |
| **Mood Purity**      | Song alignment with playlist’s top active moods | ✅ Yes       | ❌ No          | **O(N · K)** | ✅ High – mood alignment     |
| **Mood Diversity**   | Mood variety across playlists                   | ✅ Yes       | ✅ Yes         | **O(C · M)** | ✅ High – thematic spread    |
| **Silhouette Score** | Distance-based cluster separation               | ❌ No        | ✅ Yes         | **O(N²)**    | ⚠️ Medium – structural only |

> Where:
> `N` = number of songs
> `K` = number of top moods considered
> `C` = number of playlists
> `M` = number of total mood labels

 **Purity and Diversity** are fast, interpretable, and designed specifically for music data.
 **Silhouette Score** focuses only on shape/separation and ignores label meaning entirely.

**Combining Purity and Diversity**

The final score that guides the clustering algorithm is a weighted combination:

```python
final_score = (SCORE_WEIGHT_PURITY * purity_score) + (SCORE_WEIGHT_DIVERSITY * diversity_score)
```

You can tune the weights to emphasize:

* **Purity** → for tighter, more focused playlists
* **Diversity** → for broader, more eclectic playlist collections

### **AI Playlist Naming**

After the clustering algorithm has identified groups of similar songs, AudioMuse-AI can optionally use an AI model to generate creative, human-readable names for the resulting playlists. This replaces the default "Mood_Tempo" naming scheme with something more evocative.

1.  **Input to AI:** For each cluster, the system extracts key characteristics derived from the cluster's centroid (like predominant moods and tempo) and provides a sample list of songs from that cluster.
2.  **AI Model Interaction:** This information is sent to a configured AI model (either a self-hosted **Ollama** instance or a cloud-based one like **Google Gemini** or **Mistral**) along with a carefully crafted prompt.
3.  **Prompt Engineering:** The prompt guides the AI to act as a music curator and generate a concise playlist name (15-35 characters) that reflects the mood, tempo, and overall vibe of the songs, while adhering to strict formatting rules (standard ASCII characters only, no extra text).
4.  **Output Processing:** The AI's response is cleaned to ensure it meets the formatting and length constraints before being used as the final playlist name (with the `_automatic` suffix appended later by the task runner).

This step adds a layer of creativity to the purely data-driven clustering process, making the generated playlists more appealing and easier to understand at a glance. The choice of AI provider and model is configurable via environment variables and the frontend.

## Concurrency — deep dive

Purpose: scale long-running work (analysis, clustering) across multiple workers while preserving observability and cooperative cancellation.

Core steps (implementation highlights):

- Entrypoint(s): user actions in the Flask UI enqueue top-level tasks (`run_analysis_task`, `run_clustering_task`) into Redis Queue (RQ); workers pick up child jobs like `analyze_album_task` and `run_clustering_batch_task`.
- Hierarchical batching: parent tasks break large jobs into coarser-grained batches to avoid flooding the queue with tiny jobs (per-album analysis jobs; clustering iteration batches).
- Parallel execution: multiple RQ workers run batches in parallel, speeding throughput. Workers should run on separate processes/containers for scale and resilience.
- Cooperative cancellation & monitoring: tasks periodically check DB-stored status flags and will stop mid-run if marked `REVOKED`, performing cleanup and updating status. Progress and logs are written back to PostgreSQL for UI display.

Key notes and edge cases:

- Design trades off task granularity (fewer large tasks reduces queue churn; smaller tasks give finer progress visibility).
- The DB is the primary source of truth for task progress; RQ metadata is supplementary.
- For HA, supervise worker processes (e.g., `supervisord`) and use HA-ready Redis/Postgres in production.


## Instant Chat — deep dive

Purpose: translate a user's natural-language playlist request into a safe, constrained SQL query that runs against the analyzed `score` table and return results to the UI.

Core steps (implementation highlights):

- Entrypoint(s): frontend `chat.html` calls `/api/chatPlaylist` in `app_chat.py`.
- Prompt engineering: `app_chat.py` composes a strict system prompt (`base_expert_playlist_creator_prompt`) describing the `public.score` schema, allowed labels, formatting rules (SQL only), and enforcing `LIMIT 25`.
- Model call: `ai.py` adapters send the prompt to the configured provider (Ollama, Gemini, Mistral) and return the raw SQL text.
- Sanitization & validation: `clean_and_validate_sql` strips markdown, normalizes quotes/encoding, parses with `sqlglot` (Postgres dialect), and enforces the `LIMIT` clause to ensure safe execution.
- Safe execution: the service ensures a restricted DB role (`AI_CHAT_DB_USER_NAME`) exists and executes the query under that role (`SET LOCAL ROLE ...`) so AI-generated SQL has read-only, constrained permissions.
- Response: the resulting rows (`item_id`, `title`, `author`) are returned to the frontend; the UI offers playlist creation via the media-server adapters.

Key notes and edge cases:

- The system uses syntactic validation (`sqlglot`) and role-based execution to mitigate risky queries. This is not a formal proof of safety — keep the DB and AI credentials guarded.
- The quality of results depends heavily on the chosen model. The prompt includes explicit schema and examples to guide the model toward useful SQL.

## Playlist from Similar song — deep dive

Purpose: return a fast, accurate list of sonically similar tracks to a chosen seed song using a persisted ANN index (Voyager).

Core steps (implementation highlights):

- Entrypoint(s): `/similarity` UI → `/api/similar_tracks` in the backend.
- Index creation: during analysis the service builds a Voyager ANN index from stored 200-dim embeddings and saves it to PostgreSQL; the index is reloaded into memory at startup for query performance.
- Query path: `find_nearest_neighbors_by_id` (or `by_vector`) looks up the seed vector and returns the N nearest vectors using an angular metric; results are joined with `score` metadata for display.
- Playlist creation: `/api/create_playlist` takes selected IDs and calls the configured media-server adapter (Jellyfin/Navidrome/Lyrion) to create the playlist on the server.

Key notes and edge cases:

- Voyager is persisted so rebuilds are only needed when new embeddings are added.
- Loading the index into memory is essential for responsiveness; ensure enough RAM when serving large libraries.
- The chosen distance metric (angular by default) favors directional similarity, which works well with normalized embeddings.

## Song Path Deep Dive - Deep dive
Purpose: build an adaptive, fixed-length path of songs between a start and an end item that (a) sounds coherent, (b) avoids duplicate title/artist pairs, and (c) respects a realistic per-step distance estimated from the local neighborhood.

Core steps (implementation highlights):
- Entrypoint: `find_path_between_songs(start_item_id, end_item_id, Lreq)`.
- Estimate local jump size δ_avg using `_calculate_local_average_jump_distance(...)` which chains nearest neighbors around the endpoints to compute typical step distances.
- Compute direct distance D between start and end using the configured metric (`PATH_DISTANCE_METRIC` — angular or euclidean). Derive a core step count Lcore ≈ floor(D/δ_avg) (bounded and scaled by `PATH_LCORE_MULTIPLIER`) so the path length is realistic.
- Interpolate backbone centroids between start and end with `interpolate_centroids(...)` (supports angular interpolation for cosine-style metric or linear interpolation for euclidean).
- For each centroid, pick an actual track using `_find_best_unique_song(...)`. Candidates are fetched from the vector index (`find_nearest_neighbors_by_vector`) and filtered/deduplicated by:
  - ID and normalized (artist,title) signature uniqueness,
  - configurable duplicate-distance thresholds with a short lookback window,
  - availability of metadata/embedding.
- If the user-requested length Lreq > backbone length, the code will expand the backbone and repeat selection, still enforcing uniqueness and distance constraints. Final IDs are resolved to full track rows with `get_tracks_by_ids()` and returned.

Edge cases: the function gracefully handles missing embeddings, insufficient neighbors, and returns shorter paths if uniqueness constraints prevent filling every slot.

## Song Alchemy - Deep dive 
Purpose: combine one-or-more "add" seeds and optional "subtract" seeds to return a playlist of similar tracks and a 2D visualization (embedding projection + centroids).

Core steps (implementation highlights):
- Entrypoint: `song_alchemy(add_ids, subtract_ids, n_results, subtract_distance, temperature)`.
- Compute add/sub centroids by averaging item vectors (`_compute_centroid_from_ids(...)` using `get_vector_by_id`).
- Query voyager for candidates either by ID (`find_nearest_neighbors_by_id`) for single seeds or by vector (`find_nearest_neighbors_by_vector`) for centroid queries; request a superset and filter down.
- Subtract filtering: remove candidates that are too close to the subtract centroid according to the configured metric (angular or euclidean) and threshold (`subtract_distance`).
- Deduplicate and exclude seed IDs.
- Projection for visualization: try to reuse a stored projection with `load_map_projection('main_map')`. For missing points compute a local 2D projection using helpers defined in `tasks/song_alchemy.py` in this preference order:
  1. `_project_aligned_add_sub` (align X axis to add→subtract)
 2. `_project_with_discriminant` (PCA + logistic discriminant)
 3. `_project_with_umap` (if umap installed)
 4. `_project_to_2d` (simple PCA/SVD fallback)
- Sampling and selection: order candidates by distance to the add centroid; if `temperature > 0` apply a softmax over negative distances to sample stochastically; if `temperature == 0` return deterministic top-N.
- Return payload includes `results`, `filtered_out` (items excluded by subtract filter, with 2D coordinates when available), `centroid_2d`, `add_points`, `sub_points`, and projection metadata used by the UI.

UI notes: `templates/alchemy.html` trims results to exactly N on the client and plots both selected and filtered-out items (so the user sees what was excluded and why).

## Music Map - Deep dive 
Purpose: provide a fast, interactive 2D scatter of the collection for exploration and selection; server-side builds and caches deterministic sampled payloads for client efficiency.

Core steps (implementation highlights):
- Builder: `build_map_cache()` reads `score` + `embedding` rows from PostgreSQL and attempts to reuse a previously-saved projection via `load_map_projection('main_map')`.
- For missing coordinates, it computes projections using the same helper set used by Song Alchemy (UMAP/discriminant/PCA fallbacks) to keep visualization consistent.
- Deterministic downsampling: `_sample_items(items, fraction)` uses linspace indices to produce 100%/75%/50%/25% buckets so results are reproducible and memory-friendly.
- The builder stores JSON and gzipped bytes into the in-memory `MAP_JSON_CACHE` keyed by percent for fast HTTP responses.

Front-end (Plotly) behavior (`templates/map.html`):
- Client calls `/api/map?percent=<25|50|75|100>` and receives a lightweight list of items with `embedding_2d`, `item_id`, `title`, `artist`, and a compact `mood_vector` summary.
- `topGenre()` extracts the top mood/genre label from the stored `mood_vector` string; `colorPaletteFor()` deterministically maps labels to colors.
- Plot uses a single `scattergl` trace for performance, stores ids in `customdata`, and implements a manual legend with hide/show filtering, lasso selection, and client-side path overlays.

Performance notes: the heavy work (embedding projection) is done server-side and cached; the client avoids long-term in-memory retention by re-fetching on user-driven percent changes.

## Sonic Fingerprint - Deep dive 
Purpose: generate a personal playlist by averaging recently-played embeddings (a listening "fingerprint") and finding nearest neighbors to that fingerprint.

Core steps (implementation highlights):
- Entrypoint: `generate_sonic_fingerprint(num_neighbors=None, user_creds=None)`.
- Fetch the user's top-played songs from the configured media adapter (`get_top_played_songs(limit=...)`).
- Retrieve embeddings for those seeds (`get_tracks_by_ids`) and build a weighted average vector where weights reflect recency: `get_last_played_time(...)` is used to compute decayed weights (more recent plays get higher weight); a default small weight is used when timestamps are missing.
- Normalize the average fingerprint vector (sum of weighted vectors / total weight) and query the vector index (`find_nearest_neighbors_by_vector`) for neighbors to reach the requested playlist size.
- Final playlist: seed songs (the top-played items that contributed to the fingerprint) are included first, then voyager neighbors are appended, skipping duplicates until the target count is reached.

UI notes: `templates/sonic_fingerprint.html` collects optional adapter credentials (Jellyfin/Navidrome token/user) and displays results; playlist creation reuses the app's playlist creation API.