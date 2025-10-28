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


## Analysis Algorithm Deep Dive

The audio analysis in AudioMuse-AI, orchestrated by `tasks.py`, meticulously extracts a rich set of features from each track. This process is foundational for the subsequent clustering and playlist generation.
1.  **Audio Loading & Preprocessing:**
    *   Tracks are first downloaded from your Jellyfin or Navidrome library to a temporary local directory. Librosa is used then to load the audio.
    
2.  **Core Feature Extraction (Librosa):**
    *   **Tempo:** The `Tempo` algorithm analyzes the rhythmic patterns in the audio to estimate the track's tempo, expressed in Beats Per Minute (BPM).
    *   **Key & Scale:** The `Key` algorithm identifies the predominant musical key (e.g., C, G#, Bb) and scale (major or minor) of the track. This provides insights into its harmonic structure.
    *   **Energy:** The `Energy` function calculates the raw total energy of the audio signal. However, to make this feature comparable across tracks of varying lengths and overall loudness, the system computes and stores the **average energy per sample** (total energy divided by the number of samples). This normalized energy value offers a more stable representation of the track's perceived loudness or intensity.

3.  **Embedding Generation (TensorFlow & Librosa):**
    *   **MusiCNN Embeddings:** The cornerstone of the audio representation is a 200-dimensional embedding vector. This vector is generated using `TensorflowPredictMusiCNN` with the pre-trained model `msd-musicnn-1.pb` (specifically, the output from the `model/dense/BiasAdd` layer). MusiCNN is a Convolutional Neural Network (CNN) architecture that has been extensively trained on large music datasets (like the Million Song Dataset) for tasks such as music tagging. The resulting embedding is a dense, numerical summary that captures high-level semantic information and complex sonic characteristics of the track, going beyond simple acoustic features.
    *   **Important Note:** These embeddings are **always generated and saved** during the analysis phase for every track processed. This ensures they are available if you later choose to use them for clustering.

4.  **Prediction Models (TensorFlow & Librosa):**
    The rich MusiCNN embeddings serve as the input to several specialized models, each designed to predict specific characteristics of the music:
    *   **Primary Tag/Genre Prediction:**
        *   Model: `msd-msd-musicnn-1.onnx`
        *   Output: This model produces a vector of probability scores. Each score corresponds to a predefined tag or genre from a list (defined by `MOOD_LABELS` in `config.py`, including labels like 'rock', 'pop', 'electronic', 'jazz', 'chillout', '80s', 'instrumental', etc.). These scores indicate the likelihood of each tag/genre being applicable to the track.
    *   **Other Feature Predictions:** The `predict_other_models` function leverages a suite of distinct models, each targeting a specific musical attribute. These models also take the MusiCNN embedding as input and typically use a `model/Softmax` output layer:
        *   `danceable`: Predicted using `danceability-msd-musicnn-1.onnx`.
        *   `danceable`: Predicted using `danceability-msd-musicnn-1.onnx`.
        *   `aggressive`: Predicted using `mood_aggressive-msd-musicnn-1.onnx`.
        *   `happy`: Predicted using `mood_happy-msd-musicnn-1.onnx`.
        *   `party`: Predicted using `mood_party-msd-musicnn-1.onnx`.
        *   `relaxed`: Predicted using `mood_relaxed-msd-musicnn-1.onnx`.
        *   `sad`: Predicted using `mood_sad-msd-musicnn-1.onnx`.
        *   Output: Each of these models outputs a probability score (typically the probability of the positive class in a binary classification, e.g., the likelihood the track is 'danceable'). This provides a nuanced understanding of various moods and characteristics beyond the primary tags.

5.  **Feature Vector Preparation for Clustering (`score_vector` function):**
    When not using embeddings directly for clustering, all the extracted and predicted features are meticulously assembled and transformed into a single numerical vector for each track using the `score_vector` function. This is a critical step for machine learning algorithms:
    *   **Normalization:** Tempo and the calculated average energy are normalized to a 0-1 range using configured minimum/maximum values (`TEMPO_MIN_BPM`, `TEMPO_MAX_BPM`, `ENERGY_MIN`, `ENERGY_MAX`). This ensures these features have a comparable scale.
    *   **Normalization:**
        *   Tempo (BPM) and the calculated average energy per sample are normalized to a 0-1 range. This is achieved by scaling them based on predefined minimum and maximum values (e.g., `TEMPO_MIN_BPM = 40.0`, `TEMPO_MAX_BPM = 200.0`, `ENERGY_MIN = 0.01`, `ENERGY_MAX = 0.15` from `config.py`). Normalization ensures that these features, which might have vastly different original scales, contribute more equally during the initial stages of vector construction.
    *   **Vector Assembly:**
        *   The final feature vector for each track is constructed by concatenating: the normalized tempo, the normalized average energy, the vector of primary tag/genre probability scores, and the vector of other predicted feature scores (danceability, aggressive, etc.). This creates a comprehensive numerical profile of the track.
    *   **Standardization:**
        *   This complete feature vector is then standardized using `sklearn.preprocessing.StandardScaler`. Standardization transforms the data for each feature to have a zero mean and unit variance across the entire dataset. This step is particularly crucial for distance-based clustering algorithms like K-Means. It prevents features with inherently larger numerical ranges from disproportionately influencing the distance calculations, ensuring that all features contribute more equitably to the clustering process. The mean and standard deviation (scale) computed by the `StandardScaler` for each feature are saved. These saved values are used later for inverse transforming cluster centroids back to an interpretable scale, which aids in understanding the characteristics of each generated cluster.

6.  **Option to Use Embeddings Directly for Clustering:**
    *   As an alternative to the `score_vector`, AudioMuse-AI now offers the option to use the raw MusiCNN embeddings (200-dimensional vectors) directly as input for the clustering algorithms. This is controlled by the `ENABLE_CLUSTERING_EMBEDDINGS` parameter (configurable via the UI).
    *   Using embeddings directly can capture more nuanced and complex relationships between tracks, as they represent a richer, higher-dimensional summary of the audio. However, this may also require different parameter tuning for the clustering algorithms (e.g., GMM often performs well with embeddings) and can be more computationally intensive, especially with algorithms like standard K-Means (MiniBatchKMeans is used to mitigate this when embeddings are enabled). The maximum number of PCA components can also be higher when using embeddings (e.g., up to 199) compared to feature vectors (e.g., up to 8).
    *   Regardless of the K-Means variant, embeddings are standardized using `StandardScaler` before being fed into the clustering algorithms.

**Persistence:** PostgreSQL database is used for persisting analyzed track metadata, generated playlist structures, and task status.

The use of Librosa for songs preprocessing was introduced in order to improve the compatibility with other platform (eg. `ARM64`). It is configured in order to load the same MusicNN models previously used.
Librosa loads audio using `librosa.load(file_path, sr=16000, mono=True)`, ensuring the sample rate is exactly `16,000 Hz` and the audio is converted to mono—both strict requirements of the model. It then computes a Mel spectrogram using `librosa.feature.melspectrogram` with parameters precisely matching those used during model training: `n_fft=512, hop_length=256, n_mels=96, window='hann', center=False, power=2.0, norm='slaney', htk=False`. The spectrogram is scaled using `np.log10(1 + 10000 * mel_spec)`, a transformation that must be replicated exactly. These preprocessing steps are crucial: any deviation in parameters results in incompatible input and incorrect model predictions. Once prepared, the data is passed into a frozen TensorFlow model graph using v1 compatibility mode. TensorFlow maps defined input/output tensor names and executes inference with session.run(), first generating embeddings for each patch of the spectrogram, and then passing these to various classifier heads (e.g., mood, genre). The entire pipeline depends on strict adherence to the original preprocessing parameters—without this, the model will fail to produce meaningful results.

## **Clustering Algorithm Deep Dive**

AudioMuse-AI offers three main clustering algorithms (K-Means, DBSCAN, GMM). A key feature is the ability to choose the input data for these algorithms:
*   **Score-based Feature Vectors:** The traditional approach, using a vector composed of normalized tempo, energy, mood scores, and other derived features (as described in the Analysis section). This is the default.
*   **Direct Audio Embeddings:** Using the 200-dimensional MusiCNN embeddings generated during analysis. This can provide a more nuanced clustering based on deeper audio characteristics. This option is controlled by the `ENABLE_CLUSTERING_EMBEDDINGS` parameter in the UI's "Advanced" section and configuration. GMM may perform particularly well with embeddings. When embeddings are used with K-Means, MiniBatchKMeans is employed to handle the larger data size more efficiently.

Regardless of the input data chosen, the selected clustering algorithm is executed multiple times (default 5000) following an Evolutionary Monte Carlo approach. This allows the system to test multiple configurations of parameters and find the best ones.

When chose clustering algorithm consider their complexity (speed, scalability, etc) expecially if you have big song dataset:
* So K-Means -> GMM -> DBSCAN -> Spectral (from faster to slower)

About quality it really depends from how your song are distributed. K-Means because is faster is always a good choice. GMM give good result in some test with embbeding. Spectral give also good result with embbeding but is very slow.

The TOP Playlist Number parameter was added to find the top different playlist. In short after the clustering is executed, only the N most diverse playlist are keep to avoid to have hundred of playlist created. If you put this parameter to 0, it will keep all.

Here's an explanation of the pros and cons of the different algorithms:

### **1\. K-Means**

* **Best For:** Speed, simplicity, when clusters are roughly spherical and of similar size.  
* **Pros:** Very fast (especially MiniBatchKMeans for large datasets/embeddings), scalable, clear "average" cluster profiles.  
* **Cons:** Requires knowing cluster count (K), struggles with irregular shapes, sensitive to outliers, and can be slow on large datasets (complexity is O(n*k*d*i), though MiniBatchKMeans helps).

### **2\. DBSCAN**

* **Best For:** Discovering clusters of arbitrary shapes, handling outliers well, when the number of clusters is unknown.  
* **Pros:** No need to set K, finds varied shapes, robust to noise.  
* **Cons:** Sensitive to eps and min_samples parameters, can struggle with varying cluster densities, no direct "centroids," and can be slow on large datasets (complexity is O(n log n) to O(n²)).

### **3\. GMM (Gaussian Mixture Models)**

* **Best For:** Modeling more complex, elliptical cluster shapes and when a probabilistic assignment of tracks to clusters is beneficial.  
* **Pros:** Flexible cluster shapes, "soft" assignments, model-based insights.  
* **Cons:** Requires setting number of components, computationally intensive (can be slow, with complexity of O(n*k*d²) per iteration), sensitive to initialization.

### **4. Spectral Clustering**

* **Best For:** Finding clusters with complex, non-convex shapes (e.g., intertwined genres) when the number of clusters is known beforehand.
* **Pros:** Very effective for irregular cluster geometries where distance-based algorithms like K-Means fail. It does not assume clusters are spherical.
* **Cons:** Computationally very expensive (often O(n^3) due to matrix operations), which makes it impractical for large music libraries. It also requires you to specify the number of clusters, similar to K-Means.

**Recommendation:** Start with **K-Means** (which will use MiniBatchKMeans by default if embeddings are enabled) due to its speed in the evolutionary search. MiniBatchKMeans is particularly helpful for larger libraries or when using embeddings. Experiment with **GMM** for more nuanced results, especially when using direct audio embeddings. Use **DBSCAN** if you suspect many outliers or highly irregular cluster shapes. Using a high number of runs (default 5000) helps the integrated evolutionary algorithm to find a good solution.

### Montecarlo Evolutionary Approach

AudioMuse-AI doesn't just run a clustering algorithm once; it employs a sophisticated Monte Carlo evolutionary approach, managed within `tasks.py`, to discover high-quality playlist configurations. Here's a high-level overview:

1.  **Stratified Sampling:** Before each clustering run, the system selects a subset of songs from your library. A crucial part of this selection is **stratified sampling** based on predefined genres (`STRATIFIED_GENRES` in `config.py`). This ensures that the dataset used for clustering in each iteration includes a targeted number of songs from each of these important genres.
    *   **Purpose:** This is done specifically to **avoid scenarios where genres with a low number of songs in your library are poorly represented or entirely absent** from the clustering process. By explicitly sampling from these genres, even if they have few songs, the algorithm is more likely to discover clusters relevant to them.
    *   **Tuning the Target:** The target number of songs sampled per stratified genre is dynamically determined based on the `MIN_SONGS_PER_GENRE_FOR_STRATIFICATION` and `STRATIFIED_SAMPLING_TARGET_PERCENTILE` configuration values.
        *   `MIN_SONGS_PER_GENRE_FOR_STRATIFICATION` sets a minimum floor for the target.
        *   `STRATIFIED_SAMPLING_TARGET_PERCENTILE` calculates a target based on the distribution of song counts across your stratified genres (e.g., the 25th percentile).
        *   The actual target is the maximum of these two values.
    *   **Effect:** By changing these parameters, you can **increase or decrease the emphasis** on ensuring a minimum number of songs from each stratified genre are included in every clustering iteration's dataset. A higher minimum or percentile will aim for more songs per genre (if available), potentially leading to more robust clusters for those genres but also increasing the dataset size and computation time. If a genre has fewer songs than the target, all its available songs are included.

2.  **Data Perturbation:** For subsequent runs after the first, the sampled subset isn't entirely random. A percentage of songs (`SAMPLING_PERCENTAGE_CHANGE_PER_RUN`) are randomly swapped out, while the rest are kept from the previous iteration's sample. This controlled perturbation introduces variability while retaining some continuity between runs.

3.  **Multiple Iterations:** The system performs a large number of clustering runs (defined by `CLUSTERING_RUNS`, e.g., 5000 times). In each run, it experiments with different parameters for the selected clustering algorithm (K-Means, DBSCAN, or GMM) and for Principal Component Analysis (PCA) if enabled. These parameters are initially chosen randomly within pre-defined ranges.

4.  **Evolutionary Strategy:** As the runs progress, the system "learns" from good solutions:
    *   **Elite Solutions:** The parameter sets from the best-performing runs (the "elites") are remembered.
    *   **Exploitation & Mutation:** For subsequent runs, there's a chance (`EXPLOITATION_PROBABILITY_CONFIG`) that instead of purely random parameters, the system will take an elite solution and "mutate" its parameters slightly. This involves making small, random adjustments (controlled by `MUTATION_INT_ABS_DELTA`, `MUTATION_FLOAT_ABS_DELTA`, etc.) to the elite's parameters, effectively exploring the "neighborhood" of good solutions to potentially find even better ones.

5.  **Comprehensive Scoring:** Each clustering outcome is evaluated using a composite score. This score is a weighted sum of several factors, designed to balance different aspects of playlist quality:
    *   **Playlist Diversity:** Measures how varied the predominant characteristics (e.g., moods, danceability) are across all generated playlists.
    *   **Playlist Purity:** Assesses how well the songs within each individual playlist align with that playlist's central theme or characteristic.
    *   **Internal Clustering Metrics:** Standard metrics evaluate the geometric quality of the clusters:
        *   **Silhouette Score:** How distinct are the clusters?
        *   **Davies-Bouldin Index:** How well-separated are the clusters relative to their intra-cluster similarity?
        *   **Calinski-Harabasz Index:** Ratio of between-cluster to within-cluster dispersion.

6.  **Configurable Weights:** The influence of each component in the final score is determined by weights (e.g., `SCORE_WEIGHT_DIVERSITY`, `SCORE_WEIGHT_PURITY`, `SCORE_WEIGHT_SILHOUETTE`). These are defined in `config.py` and allow you to tune the algorithm to prioritize, for example, more diverse playlists over extremely pure ones, or vice-versa. 

7.  **Best Overall Solution:** After all iterations are complete, the set of parameters that yielded the highest overall composite score is chosen. The playlists generated from this top-scoring configuration are then presented and created in Jellyfin or Navidrome.

This iterative and evolutionary process allows AudioMuse-AI to automatically explore a vast parameter space and converge on a clustering solution that is well-suited to the underlying structure of your music library.

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

## Concurrency Algorithm Deep Dive

AudioMuse-AI leverages Redis Queue (RQ) to manage and execute long-running processes like audio analysis and evolutionary clustering in parallel across multiple worker nodes. This architecture, primarily orchestrated within `tasks.py`, is designed for scalability and robust task management.

1.  **Task Queuing with Redis Queue (RQ):**
    *   When a user initiates an analysis or clustering job via the web UI, the main Flask application doesn't perform the heavy lifting directly. Instead, it enqueues a task (e.g., `run_analysis_task` or `run_clustering_task`) into a Redis queue.
    *   Separate RQ worker processes, which can be scaled independently (e.g., running multiple `audiomuse-ai-worker` pods in Kubernetes), continuously monitor this queue. When a new task appears, an available worker picks it up and begins execution.

2.  **Parallel Processing on Multiple Workers:**
    *   This setup allows multiple tasks to be processed concurrently. For instance, if you have several RQ workers, they can simultaneously analyze different albums or run different batches of clustering iterations. This significantly speeds up the overall processing time, especially for large music libraries or extensive clustering runs.

3.  **Hierarchical Task Structure & Batching for Efficiency:**
    To minimize the overhead associated with frequent queue interactions (writing/reading task status, small job processing), tasks are structured hierarchically and batched:
    *   **Analysis Tasks:** The `run_analysis_task` acts as a parent task. It fetches a list of albums and then enqueues individual `analyze_album_task` jobs for each album. Each `analyze_album_task` processes all tracks within that single album. This means one "album analysis" job in the queue corresponds to the analysis of potentially many songs, reducing the number of very small, granular tasks.
    *   **Clustering Tasks:** Similarly, the main `run_clustering_task` orchestrates the evolutionary clustering. It breaks down the total number of requested clustering runs (e.g., 1000) into smaller `run_clustering_batch_task` jobs. Each batch task (e.g., `run_clustering_batch_task`) then executes a subset of these iterations (e.g., 20 iterations per batch). This strategy avoids enqueuing thousands of tiny individual clustering run tasks, again improving efficiency.

4.  **Advanced Task Monitoring and Cancellation:**
    A key feature implemented in `tasks.py` is the ability to monitor and cancel tasks *even while they are actively running*, not just when they are pending in the queue.
    *   **Cooperative Cancellation:** Both parent tasks (like `run_analysis_task` and `run_clustering_task`) and their child tasks (like `analyze_album_task` and `run_clustering_batch_task`) periodically check their own status and the status of their parent in the database.
    *   If a task sees that its own status has been set to `REVOKED` (e.g., by a user action through the UI) or if its parent task has been revoked or has failed, it will gracefully stop its current operation, perform necessary cleanup (like removing temporary files), and update its status accordingly.
    *   This is more sophisticated than typical queue systems where cancellation might only prevent a task from starting. Here, long-running iterations or album processing loops can be interrupted mid-way.

5.  **Status Tracking:**
    *   Throughout their lifecycle, tasks frequently update their progress, status (e.g., `STARTED`, `PROGRESS`, `SUCCESS`, `FAILURE`, `REVOKED`), and detailed logs into the PostgreSQL database. The Flask application reads this information to display real-time updates on the web UI.
    *   RQ's job metadata is also updated, but the primary source of truth for detailed status and logs is the application's database.

## **Instant Chat Deep Dive**

The "Instant Playlist" feature, accessible via `chat.html`, provides a direct way to generate playlists using natural language by leveraging AI models to construct and execute PostgreSQL queries against your analyzed music library.

**Core Workflow:**

1.  **User Interface (`chat.html`):**
    *   The user selects an AI provider (Ollama, Gemini or Mistral) and can customize model names, Ollama server URLs, or Gemini/Mistral API keys. These default to values from the server's `config.py` but can be overridden per session in the UI.
    *   The user types a natural language request (e.g., "sad songs for a rainy day" or "energetic pop from the 2020s").

2.  **API Call (`app_chat.py` - `/api/chatPlaylist`):**
    *   The frontend sends the user's input, selected AI provider, and any custom model/API parameters to this backend endpoint.

3.  **Prompt Engineering (`app_chat.py`):**
    *   A detailed system prompt (`base_expert_playlist_creator_prompt`) is used. This prompt instructs the AI model to act as an expert PostgreSQL query writer specializing in music. It includes:
        *   Strict rules for the output (SQL only, specific columns, `LIMIT 25`).
        *   The database schema for the `public.score` table.
        *   Detailed querying instructions for `mood_vector`, `other_features`, `tempo`, `energy`, and `author`.
        *   Examples of user requests and the corresponding desired SQL queries.
        *   A list of `MOOD_LABELS` and `OTHER_FEATURE_LABELS` available in the `mood_vector` and `other_features` columns.
        *   Specific instructions for handling requests for "top," "famous," or "trending" songs, suggesting the use of `CASE WHEN` in `ORDER BY` to prioritize known hits.
        *   Guidance on using `UNION ALL` for combining different criteria.
    *   The user's natural language input is appended to this master prompt.

4.  **AI Model Interaction (`ai.py` via `app_chat.py`):**
    *   Based on the selected provider, `app_chat.py` calls either `get_ollama_playlist_name`, `get_gemini_playlist_name` `get_mistral_playlist_name` from `ai.py`.
    *   These functions send the complete prompt to the respective AI model (Ollama, Gemini or Mistral).

5.  **SQL Query Processing (`app_chat.py` - `clean_and_validate_sql`):**
    *   The raw SQL string returned by the AI is processed:
        *   Markdown (like ```sql) is stripped.
        *   The query is normalized to start with `SELECT`.
        *   Unescaped single quotes within potential string literals (e.g., "L'amour") are escaped to `''` (e.g., "L''amour") using a regex (`re.sub(r"(\w)'(\w)", r"\1''\2", cleaned_sql)`).
        *   The string is normalized to ASCII using `unicodedata.normalize` to handle special characters.
        *   `sqlglot.parse` is used to parse the SQL (as PostgreSQL dialect). This helps validate syntax and further sanitize the query.
        *   The `LIMIT` clause is enforced to be `LIMIT 25`. If a different limit is present, it's changed; if no limit exists, it's added.
        *   The query is re-serialized by `sqlglot` to ensure a clean, valid SQL string.

6.  **Database Execution (`app_chat.py`):**
    *   Before execution, the system ensures a dedicated, restricted database user (`AI_CHAT_DB_USER_NAME` from `config.py`) exists. This user is automatically created if it doesn't exist and is granted `SELECT` ONLY permissions on the `public.score` table.
    *   The validated SQL query is executed against the PostgreSQL database using `SET LOCAL ROLE {AI_CHAT_DB_USER_NAME};`. This ensures the query runs with the restricted permissions of this AI-specific user for that transaction.

7.  **Response to Frontend (`chat.html`):**
    *   The results (list of songs: `item_id`, `title`, `author`) or any errors are sent back to `chat.html`.
    *   The frontend displays the AI's textual response (including the generated SQL and any processing messages) and the list of songs.
    *   If songs are returned, a form appears allowing the user to name the playlist. Submitting this form calls another endpoint (`/api/createJellyfinPlaylist`) in `app_chat.py` which uses the Jellyfin or Navidrome API to create the playlist with the chosen name (appended with `_instant`) and the retrieved song IDs.

## Playlist from Similar song - Deep dive

The "Playlist from Similar Song" feature provides an interactive way to discover music by finding tracks that are sonically similar to a chosen song. This process relies on a powerful combination of pre-computed audio embeddings and a specialized high-speed search index.

**Core Workflow:**

1. **Index Creation (During Analysis Task):**  
   * The foundation of this feature is an **Approximate Nearest Neighbors (ANN) index**, which is built using Spotify's **Voyager** library.  
   * During the main "Analysis Task," after the 200-dimensional MusiCNN embedding has been generated for each track, a dedicated function (build\_and\_store\voyager\_index) is triggered.  
   * This function gathers all embeddings from the database and uses them to build the Voyager index. It uses an 'angular' distance metric, which is highly effective for comparing the "direction" of high-dimensional vectors like audio embeddings, thus capturing sonic similarity well.  
   * The completed index, which is a highly optimized data structure for fast lookups, is then saved to the PostgreSQL database. This ensures it persists and only needs to be rebuilt when new music is analyzed.  
2. **User Interface (similarity.html):**  
   * The user navigates to the /similarity page.  
   * An autocomplete search box allows the user to easily find a specific "seed" song by typing its title and/or artist. This search is powered by the /api/search\_tracks endpoint.  
3. **High-Speed Similarity Search (/api/similar\_tracks):**  
   * Once the user selects a seed song and initiates the search, the frontend calls this API endpoint with the song's unique item\_id.  
   * For maximum performance, the backend loads the Voyager index from the database into memory upon starting up (load\voyager\_index\_for\_querying). This in-memory cache allows for near-instantaneous lookups.  
   * The core function find\_nearest\_neighbors\_by\_id takes the seed song's ID, finds its corresponding vector within the Voyager index, and instantly retrieves the *N* closest vectors (the most similar songs) based on the pre-calculated angular distances.  
   * The backend then fetches the metadata (title, artist) for these resulting song IDs from the main score table.  
4. **Playlist Creation (/api/create\_playlist):**  
   * The list of similar tracks, along with their distance from the seed song, is displayed to the user.  
   * The user can then enter a desired playlist name and click a button.  
   * This action calls the /api/create\_playlist endpoint, which takes the list of similar track IDs and the new name, and then uses the Jellyfin or Navidrome API to create the playlist directly on the server.

This entire workflow provides a fast and intuitive method for music discovery, moving beyond simple genre or tag-based recommendations to find songs that truly *sound* alike.

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