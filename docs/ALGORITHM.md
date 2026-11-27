# Algorithm description

This document is the high level design of AudioMuse-AI. It explains, from a
functional point of view, every main algorithm the application runs: how music
is analyzed, how songs get a stable identity across media servers, how lyrics
are turned into vectors, how the similarity indexes work, and how each
user-facing feature builds a playlist on top of that data.

Each chapter follows the same structure:

- **Functional Analysis (High-Level)**: what the user sees and what the feature
  is for.
- **Technical Analysis (Algorithm-Level)**: the steps the code actually runs,
  in order, with the decisions that matter.
- **Environment Variable Configuration**: the settings that change the
  behaviour. Most of them are also editable in the Setup Wizard, see
  [PARAMETERS](PARAMETERS.md).

## Table of Contents

0. [Architectural Design](#0-architectural-design)
1. [Song Analysis](#1-song-analysis)
2. [Catalogue Identity and Deduplication](#2-catalogue-identity-and-deduplication)
3. [Lyrics Analysis](#3-lyrics-analysis)
4. [Similarity Indexes (disk-paged IVF)](#4-similarity-indexes-disk-paged-ivf)
5. [Song Clustering](#5-song-clustering)
6. [Playlist from Similar Song](#6-playlist-from-similar-song)
7. [Song Path](#7-song-path)
8. [Song Alchemy](#8-song-alchemy)
9. [Music Map](#9-music-map)
10. [Sonic Fingerprint](#10-sonic-fingerprint)
11. [Artist Similarity](#11-artist-similarity)
12. [Text Search (DCLAP)](#12-text-search-dclap)
13. [Lyrics Search](#13-lyrics-search)
14. [Instant Playlist (Chat)](#14-instant-playlist-chat)
15. [Database Cleaning](#15-database-cleaning)
16. [Scheduled Tasks (Cron)](#16-scheduled-tasks-cron)
17. [Search by Recording](#17-search-by-recording)
18. [Album Creation](#18-album-creation)
19. [Smart Fade](#19-smart-fade)

---

## 0. Architectural Design

This chapter describes the runtime as a whole: the processes, where the data
lives, and how long jobs are controlled. See also
[ARCHITECTURE](ARCHITECTURE.md) for the deployment view and
[MULTI_SERVER](MULTI_SERVER.md) for the multi-server model.

### 0.1. Functional Analysis (High-Level)

From the point of view of a user or an operator the system offers three things:

- **A web UI and a REST API.** A Flask application serves every page (dashboard,
  analysis and clustering, similar song, artist similarity, song path, song
  alchemy, text search, lyrics search, music map, sonic fingerprint, instant
  playlist, administration) and the API behind them. The web process only
  handles short requests, status polling and static assets.
- **Background processing.** Everything heavy (analysis, clustering, cleaning,
  index rebuilds, server alignment sweeps, scheduled jobs) runs on queue workers
  through PostgreSQL. The web process only enqueues the job and then shows its
  progress from the `task_status` table.
- **Fast similarity search.** A family of disk-paged IVF indexes, built from the
  stored embeddings, answers nearest-neighbour queries in well under a second
  even on very large libraries. Similar song, path, alchemy, map, artist
  similarity, text search and lyrics search all read from them.

The main flows are:

- **Analysis**: UI -> `POST /api/analysis/start` -> `tasks.analysis.run_analysis_task`
  -> workers download audio, run the models, write `score` and the embedding
  tables -> the indexes are rebuilt and a reload message is published.
- **Clustering**: UI -> `POST /api/clustering/start` -> an evolutionary search
  spread over batch jobs -> the best result is post-processed and the playlists
  are created on the media server.
- **Instant Playlist**: UI -> `POST /chat/api/chatPlaylist` -> one tool-calling
  LLM request -> the returned tool calls run as real, grounded library queries
  -> the resulting songs can be saved as a playlist.

### 0.2. Technical Analysis (Algorithm-Level)

Components and responsibilities:

- **Web app (Flask, `app.py`)**: registers the feature blueprints (chat,
  clustering, analysis, cron, ivf, sonic fingerprint, path, external, alchemy,
  map, artist similarity, clap search, lyrics search, sem grove, backup,
  provider migration, dashboard, users, sync, music servers, plugins) and starts
  a few light background threads: the index reload listener, the cron poll, the
  map cache builder and the dashboard snapshot refresher.
- **Workers**: run the jobs defined under `tasks/`. Two queues are used, a
  high priority one for coordinator jobs (analysis, clustering, cleaning, sweep)
  and a default one for the children (album analysis, clustering batches, index
  rebuilds), so a flood of children can never starve a coordinator.
- **PostgreSQL queue**: the `task_status` table plus `LISTEN`/`NOTIFY`, and small cached
  values such as the CLAP text embeddings of the "other feature" labels.
- **PostgreSQL**: the source of truth. It holds `score` (one row per catalogue
  track), the embedding tables, `track_server_map` and `artist_server_map`,
  `music_servers`, the IVF index tables, `playlist`, `task_status`, `cron`,
  `app_config` and the dashboard snapshot.
- **Similarity indexes**: six IVF indexes (audio, CLAP text, lyrics, lyrics
  axes, SemGrove, artist) plus two 2D projections (song map, artist map). They
  are built by workers, stored in PostgreSQL, exported to a local cell file and
  read through memory mapping at query time. See
  [chapter 4](#4-similarity-indexes-disk-paged-ivf).
- **Models**: ONNX Runtime runs MusiCNN (embedding and mood prediction), DCLAP
  (audio and text), Whisper-small (speech recognition), Silero (voice activity
  detection) and gte-multilingual-base (text embedding). The Docker image
  pre-fetches the model files and pins the runtime flags so results are the same
  on different CPUs.
- **Media server adapters (`tasks/mediaserver/`)**: Navidrome, Jellyfin, Emby,
  Lyrion and Plex. They expose one common interface for listing albums,
  downloading a track, reading play history and creating playlists.

Deployment notes:

- The Docker build is multi-stage: one stage downloads the model artifacts, the
  final stage pins the OS and Python dependencies. The image sets ONNX and MKL
  flags so inference is deterministic across CPU families.
- The same image runs either role. `SERVICE_TYPE` decides whether the container
  starts the web server or a queue worker.
- Scale by adding worker containers pointed at the same PostgreSQL.
  Keep a single web process responsible for cron and index reloads, or make sure
  only one instance claims a cron row (the code already claims each row
  atomically for its wall-clock minute).

### 0.3. Environment Variable Configuration

Only a few settings are still environment-only. Everything else is stored in the
database and edited in the Setup Wizard.

Core infrastructure (environment only):

- `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_HOST`, `POSTGRES_PORT`,
  `POSTGRES_DB`: the five parts a deployment sets. `config.py` assembles them
  into the PostgreSQL connection string used by the whole application, so
  `DATABASE_URL` is derived, internal, and must not be set by hand.
- `TZ`: the timezone used for logs and for cron evaluation.

Runtime and model paths:

- `TEMP_DIR`: where audio files are downloaded before analysis.
- `EMBEDDING_MODEL_PATH`, `PREDICTION_MODEL_PATH`, `CLAP_AUDIO_MODEL_PATH`,
  `CLAP_TEXT_MODEL_PATH`, `LYRICS_MODEL_DIR`, `LYRICS_WHISPER_MODEL_DIR`:
  filesystem paths to the ONNX models.
- `PER_SONG_MODEL_RELOAD`: reload the model between songs. It costs time but it
  keeps memory flat, which matters on GPU.

Job and queue limits:

- `MAX_QUEUED_ANALYSIS_JOBS`, `MAX_CONCURRENT_BATCH_JOBS`,
  `ITERATIONS_PER_BATCH_JOB`, `REBUILD_INDEX_BATCH_SIZE`,
  `QUEUE_MAX_JOBS`, `QUEUE_MAX_JOBS_HIGH`.

AI providers:

- `AI_MODEL_PROVIDER`, `OLLAMA_SERVER_URL`, `OLLAMA_MODEL_NAME`,
  `OPENAI_SERVER_URL`, `OPENAI_MODEL_NAME`, `OPENAI_API_KEY`, `GEMINI_API_KEY`,
  `GEMINI_MODEL_NAME`, `MISTRAL_API_KEY`, `MISTRAL_MODEL_NAME`,
  `AI_REQUEST_TIMEOUT_SECONDS`, `AI_TOOLCALL_TEMPERATURE`.

Safety caps:

- `CLEANING_SAFETY_LIMIT`, `MAX_SONGS_PER_ARTIST`, `DASHBOARD_BROWSE_MAX_OFFSET`.
- `ALCHEMY_MAX_N_RESULTS` and `INSTANT_PLAYLIST_MAX_N_RESULTS` are NOT safety
  caps: they bound only their page's own input box. No API route and no internal
  routine clamps a requested result count to them.

Authentication:

- `AUTH_ENABLED`, `AUDIOMUSE_USER`, `AUDIOMUSE_PASSWORD`, `API_TOKEN`,
  `JWT_SECRET`. See [AUTH](AUTH.md).

### 0.4. Concurrency Deep Dive

This section explains the patterns shared by every long job.

**Parent and child tasks.** A long job is a *parent* task that enumerates the
work and enqueues *child* tasks: album analysis children for analysis, batch
children for clustering. The parent stays alive, drains the children and reports
progress. This keeps one readable task in the UI instead of thousands of tiny
ones, and it lets the parent apply back-pressure.

**Batch sizing.** Children are grouped so the per-task overhead stays small.
Clustering uses `ITERATIONS_PER_BATCH_JOB` iterations per child; analysis
enqueues an index rebuild every `REBUILD_INDEX_BATCH_SIZE` completed albums.
Smaller batches give faster feedback and earlier search availability, at the
cost of more queue and database traffic.

**Concurrency limits.** The parent keeps at most `MAX_QUEUED_ANALYSIS_JOBS`
album children pending, and at most `MAX_CONCURRENT_BATCH_JOBS` clustering
batches active. Without this a large library would fill the queue and exhaust
memory on the workers.

**Cooperative cancellation.** Long tasks poll their own row and their parent row
in `task_status`. A missing or revoked row is the cancellation signal at every
level: the task stops at the next check, removes its temporary files and updates
its status. This makes "Cancel Current Task" work on jobs that are already
running, not only on jobs still waiting in the queue.

**Failure ceiling.** After `CLUSTERING_MAX_FAILED_BATCHES` failures no new batch
is launched and the run finishes with what completed. There is deliberately no
time-based watchdog: slow hardware is never mistaken for a hang, and Cancel is
the one way to stop a run early.

**Observability.** Every task writes progress, a percentage and a rolling log
into `task_status`. The UI polls `/api/active_tasks` and
shows the last log lines plus the final summary. Errors are classified into the
codes documented in [ERROR_CODES](ERROR_CODES.md); the full traceback only ever
goes to the container log.

---

## 1. Song Analysis

Song Analysis is the data-gathering step. Nothing else works until it has run at
least once.

### 1.1. Functional Analysis (High-Level)

**Workflow**

1. The user opens the **Analysis and Clustering** page.
2. In the basic view the only option is **Number of Recent Albums**. Setting it
   to 0 (or a negative number) scans the whole library instead of only the
   recent additions. The advanced view adds **Top N Moods**, the number of top
   scoring mood labels stored per track.
3. The user clicks **Start Analysis**. The job runs in the background, so the
   page never blocks.
4. The Task Status panel shows the `main_analysis` task with its running time,
   state, percentage and a live log (which album is being processed, how many
   were launched, skipped or completed).
5. **Cancel Current Task** stops the main task and the album children it has
   already started.
6. When the run ends the database holds the audio features and vectors for every
   new or changed song, and all the similarity indexes have been rebuilt.

**Important behaviours**

- Analysis always covers **every configured music server**, one after the other,
  with the default server first. There is no scope selector: a narrowed scope
  would leave the other servers' exclusive songs unanalyzed and invisible to
  every other feature.
- Already analyzed tracks are skipped, so running the analysis again is cheap.
  A track that lacks one stage only (its DCLAP vector, its lyrics, or the
  **neural fingerprint** used by Search by Recording, added later) is
  re-downloaded and gets that stage alone; the album is scheduled as long as
  one of its tracks needs something.
- A song that already exists in the catalogue because another server holds the
  same recording is not downloaded again. It only gains a mapping row. See
  [chapter 2](#2-catalogue-identity-and-deduplication).

### 1.2. Technical Analysis (Algorithm-Level)

#### Stage 1: Enqueueing

`POST /api/analysis/start` (in `app_analysis.py`) accepts `num_recent_albums`
and `top_n_moods`, falling back to `NUM_RECENT_ALBUMS` and `TOP_N_MOODS`. It
generates a job id, writes a pending row in `task_status` and enqueues
`tasks.analysis.run_analysis_task` on the high priority queue.

#### Stage 2: Per-server orchestration (`run_analysis_task`)

The parent runs one phase per configured server, default first. Each phase:

1. **Pre-flight probe.** `_verify_media_server_reachable` checks the server
   before any child is created, so an unreachable or unauthenticated server
   fails fast with error 1101 or 1104 instead of failing every album job.
2. **Work map.** The albums and their tracks are loaded once, and each provider
   track id is checked against `track_server_map`. A track that already has a
   mapping is skipped without any network traffic.
3. **Dispatch.** For albums that still have work, a
   `tasks.analysis.album.analyze_album_task` child is enqueued, never more than
   `MAX_QUEUED_ANALYSIS_JOBS` at a time.
4. **Drain.** The parent polls the children, updates progress and checks for
   revocation. Database reconciliation is throttled to
   `ANALYSIS_MONITOR_DB_INTERVAL` seconds so a 1M-song library does not hammer
   PostgreSQL.
5. **Mid-run index rebuild.** Every `REBUILD_INDEX_BATCH_SIZE` completed albums a
   `rebuild_all_indexes_task` job is enqueued, so newly analyzed songs become
   searchable while a long run is still going.
6. **Final rebuild.** At the end of the run all indexes are rebuilt and a
   `index-reload` notification is published on the Postgres `audiomuse_event` channel, which
   makes the running web process swap in the new indexes without a restart.

A run only fails if it crashed or if not a single song was analyzed (error codes
2005 and 2006). Individual albums that fail are reported; a job only restarts if its worker died.

#### Stage 3: Album level (`analyze_album_task`)

For each track of the album: fetch the metadata, download the file into
`TEMP_DIR`, run the per-song pipeline, write the results, delete the temporary
file. A track that holds no analyzable audio (silent hidden track, corrupt file)
is skipped as error 2007 and never fails the album.

#### Stage 4: Per-song pipeline (`analyze_track` and friends)

1. **Decode.** `robust_load_audio_with_fallback` loads the audio with librosa and
   falls back to PyAV. `AUDIO_LOAD_TIMEOUT` prevents a corrupt file from stalling
   the worker. The PyAV fallback runs in a separate short-lived process
   (`tasks/pyav_decode.py`, the same on the container and the native apps), so a
   damaged file that crashes FFmpeg ends only that process. A file PyAV rejects, or
   one that crashes it twice in a row, is skipped as error 2007 and marked as not
   analyzable; a killed or timed-out decode marks nothing and is retried on the
   next run.
2. **Basic features.** Tempo, energy and key/scale are extracted and normalized
   with `TEMPO_MIN_BPM`/`TEMPO_MAX_BPM` and `ENERGY_MIN`/`ENERGY_MAX`.
3. **MusiCNN.** The audio is turned into mel-spectrogram patches. The embedding
   model produces one vector per patch; the patches are averaged into a single
   200-dimension track embedding. The prediction model turns the same patch
   embeddings into mood probabilities, and the top `TOP_N_MOODS` labels are
   stored in `score.mood_vector`.
4. **Catalogue identity.** The 200-dimension embedding is hashed into the
   canonical `item_id` and matched against existing catalogue rows. This is what
   makes the same recording on two servers a single row. See
   [chapter 2](#2-catalogue-identity-and-deduplication).
5. **DCLAP.** If `CLAP_ENABLED` is true the audio is resampled to 48 kHz mono,
   converted to a mel-spectrogram and passed through the DCLAP audio model,
   producing a 512-dimension embedding stored in `clap_embedding`.
6. **Other features.** The six labels `danceable`, `aggressive`, `happy`,
   `party`, `relaxed` and `sad` are not a separate model. Their CLAP *text*
   embeddings are computed once and cached on disk, and each score is the cosine
   similarity between the track's CLAP audio embedding and the label embedding.
   `score.other_features` starts as zeros and is refreshed once CLAP lands.
7. **Lyrics.** If `LYRICS_ENABLED` is true the lyrics pipeline runs, see
   [chapter 3](#3-lyrics-analysis).
8. **Chromaprint.** If `CHROMAPRINT_COLLECTION_ENABLED` is true, `fpcalc`
   computes an acoustic fingerprint for the file and stores it compressed in the
   `chromaprint` table. It is used only to confirm or refuse a duplicate merge.
9. **Neural fingerprint.** If `neural_fingerprint.onnx` and its codebook
   `neural_fingerprint_pq.npz` are present, the decoded audio is resampled to
   8 kHz and the fingerprint encoder produces one 128-vector per half second
   of the whole track, each stored as a 32-byte product-quantised code in
   `embedding.neural_fingerprint` (about 14 KB per track, 10 to 25 s of CPU).
   Search by Recording identifies a clip from any part of a song with it; see
   [chapter 17](#17-search-by-recording). A track without it is re-analyzed
   for this stage alone on the next run.
10. **Persistence and plugin hook.** The results are written under the
    canonical id, and the `song_analyzed` plugin hook fires with the server the
    song came from. See [PLUGIN](PLUGIN.md).

Each optional stage (CLAP, lyrics, chromaprint, neural fingerprint) is best effort. A failure is
recorded through the error registry and never breaks the track, with one
exception: a database outage is re-raised so the whole album is retried.

### 1.3. Environment Variable Configuration

**Core**

- The `POSTGRES_*` parts: required. The PostgreSQL connection
  string is built from those parts by `config.py`.
- `TEMP_DIR`: download directory for the audio files.

**Media server**

Media server settings live in the `music_servers` registry and are edited in the
Setup Wizard. The legacy environment variables (`MEDIASERVER_TYPE`,
`MUSIC_LIBRARIES`, `NAVIDROME_URL`, `NAVIDROME_USER`, `NAVIDROME_PASSWORD`,
`JELLYFIN_URL`, `JELLYFIN_USER_ID`, `JELLYFIN_TOKEN`, `EMBY_URL`,
`EMBY_USER_ID`, `EMBY_TOKEN`, `LYRION_URL`, `PLEX_URL`, `PLEX_TOKEN`) are only
read once, at first boot, to seed the registry.

**Task and performance tuning**

- `NUM_RECENT_ALBUMS`: default number of recent albums; 0 means the whole
  library.
- `AUDIO_LOAD_TIMEOUT`: seconds allowed to load one audio file.
- `MAX_QUEUED_ANALYSIS_JOBS`: how many album children may be pending at once.
- `REBUILD_INDEX_BATCH_SIZE`: albums between two mid-run index rebuilds.
- `ANALYSIS_MONITOR_DB_INTERVAL`: minimum seconds between database
  reconciliations in the monitor loop.
- `MUSICNN_BATCH_SIZE`, `PER_SONG_MODEL_RELOAD`: inference batch size and model
  reload policy.

**Model and feature parameters**

- `TOP_N_MOODS`: how many top moods are stored per track.
- `EMBEDDING_MODEL_PATH`, `PREDICTION_MODEL_PATH`: MusiCNN ONNX models.
- `EMBEDDING_DIMENSION`: 200, fixed by the model.
- `CLAP_ENABLED`, `CLAP_AUDIO_MODEL_PATH`, `CLAP_EMBEDDING_DIMENSION`: DCLAP
  audio side. Turning CLAP off makes the analysis clearly faster but disables
  Text Search and the six other features.
- `NEURAL_FINGERPRINT_MODEL_PATH` and `NEURAL_FINGERPRINT_CODEBOOK_PATH`: the
  fingerprint encoder for Search by Recording and the codebook that stores its
  vectors as 32 bytes; pointing either at a missing file skips the stage.
- `LYRICS_ENABLED`: master switch for the lyrics stage.
- `ENERGY_MIN`, `ENERGY_MAX`, `TEMPO_MIN_BPM`, `TEMPO_MAX_BPM`: normalization
  bounds used everywhere a score vector is built.

---

## 2. Catalogue Identity and Deduplication

This chapter explains how AudioMuse-AI decides that two audio files are the same
recording. It is the foundation of multi-server support: without it the same
album on two servers would be analyzed twice and appear twice in every playlist.

### 2.1. Functional Analysis (High-Level)

- The database stores **one row per recording**, not one row per file. That row
  is the *catalogue* entry.
- Every provider file that carries that recording, on any server, is recorded as
  its own row in `track_server_map`. Several files can point at one catalogue
  row, on one server or across servers.
- The user never sees the internal catalogue id. Every API response is
  translated back to the id of the server the request targets, so a media server
  plugin always receives ids it can play.
- The practical results: a song already analyzed on one server is not downloaded
  again for the next server; a duplicate file inside a single library does not
  produce a duplicate playlist entry; and a song removed from one server keeps
  playing from the others.

### 2.2. Technical Analysis (Algorithm-Level)

#### The content id

The catalogue `item_id` **is** the content signature (`tasks/simhash.py`). It is
a home-made similarity hash: one bit per embedding dimension, answering "is this
dimension above the song's own average". That gives a 200-bit code, written as
the scheme-versioned id `fp_<version><50 hex chars>`.

There are no random projections, no external binary and no metadata in it. The
id is simply the shape of the song's MusiCNN profile, so it is derived from the
audio itself and it is stable across re-encodes.

#### Signature proposes, three checks confirm

The signature is similarity-preserving: the same recording from two different
files lands within a few bits, while different songs differ by tens of bits. So
the signature is used only to *propose* candidates, through a banded
Hamming-tolerant lookup (`SignatureIndex`) that guarantees a match within the
allowed bit distance. The final decision needs all of these to agree:

1. **Exact cosine distance** between the raw embeddings, below
   `DUPLICATE_DISTANCE_THRESHOLD_COSINE`. This is the same rule the Similar Song
   duplicate filter has always used.
2. **Duration agreement** within `DURATION_TOLERANCE_SECONDS`. This is the
   AcoustID rule. It matters because a homogeneous library (solo piano, ambient)
   puts genuinely different recordings inside the cosine threshold, and only the
   length tells them apart. An unknown duration on either side means "cannot
   prove same recording", so identity splits instead of merging.
3. **Chromaprint agreement**, when `CHROMAPRINT_GATE_ENABLED` is true and both
   files have a stored fingerprint. The comparison aligns the two fingerprints
   within `CHROMAPRINT_MAX_ALIGN_OFFSET` frames, needs at least
   `CHROMAPRINT_MIN_OVERLAP` overlapping frames, and calls them the same
   recording at or above `CHROMAPRINT_MATCH_THRESHOLD` matching bits. If either
   fingerprint is missing the check abstains and the decision falls back to the
   first two rules, so legacy libraries roll in gradually as fingerprints are
   back-filled.

The bias is deliberate and asymmetric: a false split only creates a harmless
duplicate row, while a false merge would delete a song. When in doubt, split.

On a signature collision between two genuinely different songs, the second one
simply takes the next free id.

#### Scheme versions and the one-time migration

`CATALOGUE_ID_SCHEME_VERSION` records which rules minted the current ids. New
ids are minted at the current version, and a startup migration relabels every
older row exactly once:

- `fp_2`: embedding signature plus cosine confirmation.
- `fp_3`: adds the track duration to the confirmation.
- `fp_4`: re-verifies existing merges at the tightened duration tolerance and
  splits any group whose files now differ by more than it.

Two startup steps do this work, both directly on the Flask container and never
through the job queue:

- `tasks/fingerprint_canonicalize.py` relabels legacy rows whose `item_id` is
  still a provider id. It is a pure database operation: signatures are computed
  from the stored embeddings, duplicate candidates are read from the IVF cells
  the library already built (only tracks in the same cell are compared), and the
  rewrite reuses the transactional key-rewrite the provider migration feature
  uses. The similarity indexes are repointed at the new ids in the same
  transaction, so search keeps working across the migration without a rebuild.
- `tasks/duplicate_repair.py` gives every catalogue row its `score.duration` and
  re-checks existing merges. Durations come from **one** whole-catalogue metadata
  listing per server, never per-id or batched fetches, and never audio
  downloads. A row mapping one file gets its length stamped; a row mapping
  several files is a merge that is either confirmed (lengths agree) or unmapped
  (lengths differ), so the next analysis re-analyzes each file under its own id.
  A file whose server reports no length gets a 0 sentinel, which behaves like
  NULL for identity but stops the whole catalogue being listed again on every
  boot.

Both steps are an instant no-op on later boots. They are not once per install:
identity comes from the MusiCNN embedding, so replacing that model re-mints every
id and the rewrite runs again.

#### Alignment sweeps

The sweep (`tasks/multiserver_sync.py`) is the other way a mapping appears. It is
a pure metadata pass with no downloads and no analysis, used when a server is
added or when the user clicks Align. It matches the server's catalogue against
the analyzed database in tiers: normalized file path, path tail, exact metadata
(title, artist, album), then noise-word-normalized metadata. Confident pairs are
written to `track_server_map`; anything unmatched is left unmapped rather than
guessed.

The sweep also refreshes the server's artist links and the catalogue metadata
(album, album artist, year, rating; file path only from the default server), and
prunes mappings whose track is no longer on that server.

### 2.3. Environment Variable Configuration

- `CATALOGUE_ID_SCHEME_VERSION`: the current id scheme. Bump it only to force a
  one-time re-migration.
- `DUPLICATE_DISTANCE_THRESHOLD_COSINE`, `DUPLICATE_DISTANCE_THRESHOLD_EUCLIDEAN`:
  the vector distance below which two tracks are the same recording. The metric
  in use decides which one applies.
- `DURATION_TOLERANCE_SECONDS`: maximum length difference for two tracks to be
  the same recording.
- `CHROMAPRINT_COLLECTION_ENABLED`: compute and store a fingerprint for every
  newly analyzed track.
- `CHROMAPRINT_GATE_ENABLED`: use the fingerprints in the identity decision.
- `CHROMAPRINT_MATCH_THRESHOLD`, `CHROMAPRINT_MAX_ALIGN_OFFSET`,
  `CHROMAPRINT_MIN_OVERLAP`: the comparison parameters.
- `FPCALC`: path to the `fpcalc` binary. It is on `PATH` inside Docker and set by
  the launcher in the standalone builds.

---

## 3. Lyrics Analysis

The lyrics pipeline turns a track into a multilingual text embedding plus a set
of axis scores, or falls back to an instrumental sentinel when there are no
usable lyrics. It runs inside Song Analysis and feeds Lyrics Search, SemGrove and
the AI naming context.

### 3.1. Functional Analysis (High-Level)

- Lyrics are preferred from **text** sources: first the media server, then an
  optional external lyrics API. Speech recognition on the audio is only the last
  resort, because it is slow and it can hallucinate.
- The embedding model (`gte-multilingual-base`) is language-agnostic, so there is
  **no translation step**. Language detection is used only as metadata and as a
  quality gate.
- A track with no usable lyrics is not an error. It gets an instrumental
  sentinel, which keeps it in the index as "this song has no words" instead of
  leaving a hole.
- The user does not configure any of this per track. It simply happens during
  analysis when `LYRICS_ENABLED` is true, and the result shows up in the Lyrics
  Search page.

### 3.2. Technical Analysis (Algorithm-Level)

#### Pipeline steps

| # | Step | Control applied |
|---|------|-----------------|
| 1 | MusiCNN instrumental check | If MusiCNN flagged the track as instrumental, skip everything and emit the instrumental sentinel |
| 2 | Media server lyrics | Fetch by track id, then sanitize. Non-empty text means steps 3 to 5 are skipped |
| 3 | External lyrics API | Only if enabled and the media server missed. A hit skips steps 4 and 5, a miss falls through to speech recognition |
| 4 | Audio preparation | Load and trim the audio up to `LYRICS_MAX_AUDIO_SECONDS` (240 s) |
| 4b | Voice activity detection | Keep only the voiced parts. Too little voice means instrumental, unless MusiCNN already flagged a vocalist |
| 5 | Whisper-small transcription | Transcribe under a 300 s watchdog, sanitize, record the detected language and the average log probability |
| 6 | Language detection (text path only) | `detect_langs` gives a language and a confidence. Without CJK script, a confidence below `LYRICS_LANG_CONFIDENCE_MIN` drops the track |
| 7 | Speech recognition reliability gate | Low log probability or an unknown language drops the transcript |
| 8 | Final text gate | The content quality checks run on the final text with the resolved language |
| 9 | Embedding and axis scoring | If the text is long enough, embed it and score the axes; otherwise emit the instrumental sentinel |

Steps 4 and 5 only run when neither text source produced lyrics.

#### Voice activity detection

The Silero ONNX model finds the parts of the clip that actually contain a voice
before anything is sent to the transcriber:

- It scans with `LYRICS_VAD_THRESHOLD` (0.2) and retries once at a lower floor
  (`LYRICS_VAD_RETRY_FLOOR`, 0.15) if nothing is found.
- If even the retry finds nothing it sends the **full** clip rather than dropping
  the track.
- If the voiced audio is shorter than `VAD_VOICE_RECOGNITION` seconds the track
  is treated as instrumental, unless MusiCNN already detected a vocalist, in
  which case the gate is bypassed.
- Otherwise only the voiced segments are concatenated and sent on, so the
  transcriber is not fed long instrumental stretches.

This improves the transcription and filters instrumentals before the expensive
step.

#### Sanitizing

Sanitizing runs on **every** text source, so the embedding sees lyrics and not
formatting noise. It removes invisible and control characters, emoji and
decorative Unicode blocks, HTML-like markup (which appears when an API returns a
web page instead of lyrics), LRC timing data and metadata lines, structural
headers such as *Chorus* or *Verse 2*, and runs of blank lines. It also truncates
to 300 words so one pathological blob cannot dominate. If nothing is left, the
source counts as a miss.

#### Language and content quality

One shared function resolves the language and judges the content, and it is
called identically from the transcription path and the text path. It does two
things in order:

1. **CJK script override.** If enough of the letters are Hangul, kana or Han
   (at least `LYRICS_CJK_SCRIPT_MIN_RATIO`, 0.10), the language is forced to
   `ko`, `ja` or `zh` whatever the detector said. The script itself is a far more
   reliable signal than either detector.
2. **Content quality reject**, which drops the text when:
   - it is shorter than `LYRICS_MIN_CHARS_FOR_EMBEDDING` (250 characters), too
     little signal for a meaningful embedding;
   - its zlib compression ratio is above
     `LYRICS_TEXT_MAX_COMPRESSION_RATIO` (15), meaning it is mostly one repeated
     line, which catches ad-lib spam and hallucination loops while genuinely
     chorus-heavy songs still pass;
   - the resolved language uses a non-Latin script but the text is at least 90
     percent Latin characters, which means garbled text or the wrong text
     entirely.

#### The reliability gate and why it is asymmetric

The reliability gate is a separate signal from the content checks, and it is
deliberately **not** the same on both paths, because a low confidence score does
not mean the same thing on each source:

- **Text path.** The language detector only *classifies* text that already
  exists, it does not produce it. A low confidence therefore does not prove the
  text is bad; it may just be a language the detector handles poorly. This makes
  it a weak signal: it catches garbled text, but it can also wrongly reject valid
  lyrics.
- **Transcription path.** Whisper *generates* the text from audio, so a low
  average log probability directly means the transcript is wrong. That is a
  strong signal. The transcript is dropped when the log probability is below
  `LYRICS_ASR_MIN_AVG_LOGPROB` (-1.0), when the language is unknown, or when the
  transcript is non-English and the log probability is below
  `LYRICS_ASR_NON_ENGLISH_MIN_LOGPROB` (-0.85).

**Asymmetry 1: CJK bypasses the text gate but not the transcription gate.** On
the text path the confidence gate sits after the CJK branch, so detected CJK
script skips it. The presence of Hangul, kana or Han proves the text really is
CJK, so the detector's low score can be ignored. The transcription gate has no
CJK branch and always runs, because Whisper may have hallucinated those
characters in the first place. On both paths CJK still goes through the content
checks; the only thing it ever bypasses is the text-path confidence drop. Other
under-supported languages with no script test can still be wrongly dropped, and
that is a known limitation.

**Asymmetry 2: a stricter bar for non-English transcription.** Whisper-small is
less reliable on non-English audio, so a medium-confidence non-English transcript
is more likely to be a hallucination than an English one with the same score. The
trade-off is real: a genuine non-English song scoring between -1.0 and -0.85 is
dropped to instrumental, where an English song would survive. This only affects
the transcription path.

#### Embedding and axes

Text that passes every gate is embedded with `gte-multilingual-base` (INT8 ONNX,
CLS pooling, 768 dimensions, up to `LYRICS_GTE_MAX_TOKENS` tokens). The same
embedding is also scored against five lyrical axes, each with a small set of
labels described in plain language:

| Axis | Question it answers | Labels |
|------|--------------------|--------|
| Setting | Where the song takes place | urban, wilderness, interior, transit, extraterrestrial, surreal |
| Social dynamic | Who the narrator talks to | solitary, romantic, kinship, collective, adversarial, divine |
| Emotional valence | The psychological tone | radiant, melancholic, volatile, vulnerable, serene, numb |
| Narrative temporality | When and how the story is told | retrospective, chronicle, existential, storytelling, direct plea |
| Thematic weight | How serious the content is | trivial, mortal, political, sensorial |

The 27 axis scores are stored per track and become their own searchable index.
An instrumental track gets a fixed sentinel value on every axis, so it stays
comparable without pretending to have a theme.

#### Re-running the lyrics analysis

Lyrics results live in their own tables. Dropping them makes the next analysis
run reprocess every track through the pipeline above:

```sql
DROP TABLE IF EXISTS lyrics_embedding;
DROP TABLE IF EXISTS lyrics_index_data;
DROP TABLE IF EXISTS lyrics_axes_index_data;
```

- `lyrics_embedding`: per track text, language, embedding and axis scores.
- `lyrics_index_data`: the semantic similarity index built from those embeddings.
- `lyrics_axes_index_data`: the axis index used by the axis search.

This only affects lyrics. The audio analysis is untouched.

### 3.3. Environment Variable Configuration

**Sources and switches**

- `LYRICS_ENABLED`: master switch for the whole stage.
- `LYRICS_API_ENABLE`: allow the external lyrics API.
- `LYRICS_ASR_ENABLE`: allow Whisper transcription as the last resort.
- `LYRICS_MUSICNN_SKIP`: trust the MusiCNN instrumental flag and skip early.
- `MUSICSERVER_LYRICS_TIMEOUT`: timeout for the media server lyrics call.

**Voice activity detection**

- `LYRICS_VAD_THRESHOLD`, `LYRICS_VAD_NEG_THRESHOLD`, `LYRICS_VAD_RETRY_FLOOR`,
  `LYRICS_VAD_MIN_SILENCE_MS`, `LYRICS_VAD_MIN_SPEECH_MS`,
  `LYRICS_VAD_SPEECH_PAD_MS`, `VAD_VOICE_RECOGNITION`.

**Transcription**

- `LYRICS_MAX_AUDIO_SECONDS`, `LYRICS_ASR_BEAM_SIZE`,
  `LYRICS_ASR_MIN_AVG_LOGPROB`, `LYRICS_ASR_NON_ENGLISH_MIN_LOGPROB`,
  `LYRICS_WHISPER_MODEL_DIR`.

**Text quality and language**

- `LYRICS_MIN_CHARS_FOR_EMBEDDING`, `LYRICS_TEXT_MAX_COMPRESSION_RATIO`,
  `LYRICS_LANG_CONFIDENCE_MIN`, `LYRICS_CJK_SCRIPT_MIN_RATIO`.

**Embedding**

- `LYRICS_EMBEDDING_DIMENSION` (768), `LYRICS_GTE_MAX_TOKENS`,
  `LYRICS_MODEL_DIR`, `LYRICS_GTE_WARMUP_DURATION`.

---

## 4. Similarity Indexes (disk-paged IVF)

Every feature that answers "what sounds like this" reads from an IVF index. This
chapter explains what those indexes are and why they are built this way.

### 4.1. Functional Analysis (High-Level)

- There are **six** indexes: the audio embedding index, the DCLAP text-search
  index, the lyrics semantic index, the lyrics axes index, the SemGrove index
  (lyrics and audio fused) and the artist index. There are also two 2D
  projections, one for the song map and one for the artist map.
- They are built by the workers at the end of an analysis run, or after a
  cleaning run, and stored in PostgreSQL. The web process loads them and swaps in
  a new version when the workers publish a reload message, without a restart.
- One index covers the **union** of all servers, not one index per server. When a
  request targets a specific server, a cached availability mask filters
  candidates to the tracks that server actually has before the ranking happens.
- The design goal is that a very large library stays queryable on ordinary
  hardware: memory use is bounded both while building and while querying.

### 4.2. Technical Analysis (Algorithm-Level)

#### Build

1. Embeddings are streamed out of PostgreSQL with a server-side cursor, in
   batches, so the whole library is never in RAM at once.
2. A k-means pass over a sample of the vectors produces the coarse centroids
   (the IVF "cells"). The sample size scales with the library,
   `IVF_TRAIN_POINTS_PER_CELL` vectors per cell, and the number of cells is
   capped by `IVF_NLIST_MAX`. There is no fixed cap on the training sample:
   quality scales with the library.
3. Each vector is assigned to its nearest centroid. Cells larger than
   `IVF_MAX_CELL_MB` are split so no single cell is oversized, and every stored
   value stays under `IVF_MAX_PART_SIZE_MB`.
4. Cells are written to PostgreSQL incrementally as they complete, with
   `STORAGE EXTERNAL` so PostgreSQL does not try to compress vector data.
   Angular vectors are stored already normalized, and a header flag records it,
   so query-time scans do not renormalize.
5. Vectors are quantized to the precision in `IVF_STORAGE_DTYPE`. The default is
   `i8` (int8, angular only), with `f16` and `f32` available. Smaller means less
   RAM and less IO.
6. The per-artist GMM fits of the artist index are pure Python, so they run in
   `INDEX_BUILD_WORKERS` separate processes.

`_run_all_index_builds` runs the eight steps in order: audio IVF (fatal if it
fails), DCLAP text, lyrics, lyrics axes, SemGrove, artist similarity, song map
and artist map. Only the audio index is fatal; the others log a warning and the
run continues.

#### Query

1. At load time each index is exported to a local cell file and mapped into
   memory, so queries are served from the OS page cache instead of a PostgreSQL
   round trip per cell (`IVF_DISK_CACHE_ENABLED`).
2. A query finds the `IVF_NPROBE` nearest centroids and reads only those cells.
   This is the main recall against latency knob. Cells are fetched
   `IVF_READ_BATCH_CELLS` at a time in a single `ANY()` statement.
3. Distances are computed directly in the stored precision. Because int8 is only
   a coarse stage, the query over-fetches `IVF_RERANK_OVERFETCH` times the
   candidate pool and re-ranks it with exact float32 vectors read from the source
   embedding table, so the final ordering matches full precision.
4. Two cache layers sit in front: a per-request one and a process-wide one capped
   at `IVF_GLOBAL_CACHE_MB` and shared by every index. `IVF_PRELOAD_ALL` streams
   every cell into it at load time, which turns the index into an in-memory one
   while still respecting the cap.
5. Idle memory is given back. After `IVF_GLOBAL_CACHE_IDLE_SECONDS` the global
   cache is dropped, and after `IVF_DISK_CACHE_IDLE_SECONDS` the resident pages
   of each memory-mapped file are released (the mapping stays, the next query
   faults them back in). Repeated identical queries are served from a small
   result cache with a `IVF_RESULT_CACHE_SECONDS` lifetime.

#### Availability mask

The index holds canonical ids. When a request names a server, a small cached
mask of that server's mapped tracks is applied before ranking, and the results
are translated back to that server's provider ids on the way out. Tracks the
server does not have are dropped rather than returned with an id that would not
play.

### 4.3. Environment Variable Configuration

- `IVF_INDEX_NAME`: the key used to store the main audio index.
- `IVF_METRIC`: `angular` (cosine), `euclidean` or `dot`.
- `IVF_STORAGE_DTYPE`: `i8`, `f16` or `f32`. Applied on the next rebuild.
- `IVF_NLIST_MAX`, `IVF_TRAIN_POINTS_PER_CELL`, `IVF_MAX_CELL_MB`,
  `IVF_MAX_PART_SIZE_MB`: build-side shape and size limits.
- `IVF_NPROBE`: cells probed per query, the dominant quality knob.
- `IVF_RERANK_OVERFETCH`: how much larger the candidate pool is before the exact
  float32 re-rank.
- `IVF_QUERY_CACHE_MB`, `IVF_READ_BATCH_CELLS`,
  `IVF_QUERY_PARALLEL_MIN_VECTORS`: per-query memory, batching and threading.
- `IVF_GLOBAL_CACHE_MB`, `IVF_PRELOAD_ALL`, `IVF_GLOBAL_CACHE_IDLE_SECONDS`:
  the process-wide cell cache.
- `IVF_DISK_CACHE_ENABLED`, `IVF_DISK_CACHE_DIR`, `IVF_DISK_CACHE_IDLE_SECONDS`:
  the local cell file and its idle behaviour.
- `IVF_RESULT_CACHE_SECONDS`, `IVF_RESULT_CACHE_MAX`: the query result cache.
- `IVF_MAX_DISTANCE_NPROBE`: cells probed for the "farthest song" value shown in
  the UI.
- `INDEX_BUILD_WORKERS`: worker processes for the CPU-bound parts of a rebuild.
- `SEM_GROVE_WEIGHT_LYRICS`, `SEM_GROVE_WEIGHT_AUDIO`: the fusion weights of the
  SemGrove index. They are baked in at build time, so changing them needs a
  rebuild.

---

## 5. Song Clustering

Clustering is the main creative feature. It takes the analyzed library and groups
it into thematic playlists that are then created on the media server.

### 5.1. Functional Analysis (High-Level)

**Workflow**

1. The analysis must have run at least once.
2. The user opens the **Analysis and Clustering** page. The basic view shows
   three things: the algorithm (K-Means, fixed), **Clustering Runs** (how many
   attempts the search makes) and **Automatic Parameter Discovery**.
3. The advanced view exposes everything else: the algorithm choice (K-Means,
   DBSCAN, GMM, Spectral) with its own parameter ranges, the number of final
   playlists, whether to cluster on raw embeddings or on the readable score
   vector, the scoring weights, and the AI naming provider.
4. **Start Clustering** launches a long background job. A second clustering task
   cannot start while one is running.
5. The Task Status panel shows live progress, for example
   "Progress: 100/1000 runs. Active batches: 10. Best score: 4.52".
6. When it finishes, the old `_automatic` playlists are deleted and the new ones
   are created on the media server, then listed in the Generated Playlists
   section.

**Important behaviours**

- Clustering runs for **every** configured music server, one at a time. Each
  server clusters only the tracks it actually has, runs its own search and gets
  its own playlists. Results are never computed once and pushed to the other
  servers, because the libraries are different.
- With **Automatic Parameter Discovery** on (the recommended default), a few
  quick probe runs tune the cluster count and the sampling percentile per server
  before the real run. It overrides the manual cluster-count and percentile
  values.
- The `playlist` table always holds the last run per server. It never grows into
  a history.

### 5.2. Technical Analysis (Algorithm-Level)

Clustering is not one clustering pass. It is an **evolutionary search over
clustering configurations**: hundreds or thousands of independent iterations,
each clustering a stratified sample of the library with slightly different
parameters, each scored by a single weighted fitness number. The best scoring
iteration wins and its clusters become the playlists.

There are three layers:

- **Orchestrator** (`run_clustering_task`): prepares the data, splits the runs
  into batch jobs, monitors them, then finalizes the best result.
- **Batch worker** (`run_clustering_batch_task`): a queue job that runs a fixed
  number of iterations and reports its best one.
- **Iteration** (`_perform_single_clustering_iteration`): one attempt, from
  sample to score.

Input comes from `score` (`tempo`, `energy`, `mood_vector`, `other_features`,
`author`) and, when embedding clustering is on, from the embedding table. Output
is a set of media server playlists plus the `playlist` table.

#### Pipeline steps

| # | Step | What happens |
|---|------|--------------|
| 1 | Load lightweight data | Fetch `item_id`, `author` and `mood_vector` for every track that has a mood vector; abort if there are fewer tracks than the minimum cluster count |
| 2 | Calibrate (optional) | Quick single-iteration probes tune the parameter ranges for this server |
| 3 | Build genre map and targets | Bucket tracks by their predominant genre and compute the per-genre target |
| 4 | Plan batches | Split the requested runs into batches of `ITERATIONS_PER_BATCH_JOB`, recovering any child already recorded in the database |
| 5 | Run iterations | Each iteration re-samples, picks parameters, clusters, filters and scores; the batch keeps its best |
| 6 | Monitor and aggregate | Fold each finished batch into the global best and the elite pool, with timeout and staleness watchdogs |
| 7 | Post-process the winner | Duplicate filter, minimum size filter, then the Top-N diverse selection |
| 8 | Name and create | AI-name each surviving cluster, shuffle, split oversized playlists, delete the old `_automatic` playlists and create the new ones |

#### The feature vector

Every track is reduced to one numeric vector with a fixed layout that later steps
index into by position:

```
[ tempo_norm, energy_norm, mood_0 ... mood_n, other_0 ... other_5 ]
   index 0      index 1     index 2 ...        index 2+len(moods) ...
```

- **tempo** and **energy** are normalized to 0-1 against `TEMPO_MIN_BPM` and
  `TEMPO_MAX_BPM` (40-200) and `ENERGY_MIN` and `ENERGY_MAX` (0.01-0.15), then
  clipped.
- **moods**: one slot per active mood label, filled from the stored mood vector.
- **other features**: the six labels `danceable`, `aggressive`, `happy`, `party`,
  `relaxed`, `sad`.

This feature vector is always what **names** and **scores** a cluster. What gets
**clustered** is either this same vector or the raw 200-dimension embedding when
`enable_clustering_embeddings` is on. In the embedding case the feature vector is
still used afterwards to label and score the resulting clusters.

#### Stratified sampling

One iteration does not cluster the whole library, it clusters a representative
subset, so thousands of iterations stay affordable and each one sees a balanced
cross-section.

- **Genre buckets**: each track is given one predominant genre, the highest
  scoring label among `STRATIFIED_GENRES` in its mood vector. Everything else
  falls into an "other" bucket.
- **Per-genre target**: the target is the
  `STRATIFIED_SAMPLING_TARGET_PERCENTILE` percentile of the bucket sizes, with a
  floor of `MIN_SONGS_PER_GENRE_FOR_STRATIFICATION`. This is what stops a huge
  genre swamping a small one.
- **Subset cap**: the per-iteration sample is capped at
  `CLUSTERING_SUBSET_SONGS`. All the per-genre quotas are computed before any
  track is selected, and a smaller library simply contributes every clusterable
  song.
- **Perturbation**: every iteration churns the incoming subset by
  `SAMPLING_PERCENTAGE_CHANGE_PER_RUN` (keep about 80 percent, redraw about 20
  percent). A genre already sampled at its full capacity cannot redraw what does
  not exist. A new scheduled run starts from a fresh random sample.

#### Explore against exploit

The search has no gradient. It explores the parameter space and keeps what works.
Each iteration picks its parameters in one of two modes:

- **Explore**: generate a fresh random parameter set inside the configured ranges
  (PCA components, and the cluster count, DBSCAN `eps` and `min_samples`, GMM
  components or spectral clusters depending on the method).
- **Exploit**: take one of the best solutions so far and apply small random
  deltas (`MUTATION_INT_ABS_DELTA`, `MUTATION_FLOAT_ABS_DELTA`, and
  `MUTATION_KMEANS_COORD_FRACTION` for centroid coordinates).

The switch between them:

- Exploitation is off for the first `EXPLOITATION_START_FRACTION` of all runs, so
  the search explores broadly before it has anything worth refining.
- After that each iteration exploits with probability
  `EXPLOITATION_PROBABILITY_CONFIG`, otherwise it still explores.
- The **elite pool** is the top `TOP_N_ELITES` scoring parameter sets seen across
  all batches. The orchestrator passes the current elites into each new batch, so
  improvements spread as the run goes on.
- After `CLUSTERING_EARLY_STOP_BATCHES` consecutive batches without a better
  result, no new batch is enqueued. The batches already in flight drain and the
  best result stands.

#### One iteration

1. **Fetch and vectorize**: load the full track data for the subset and build the
   feature vectors (and the embeddings, if enabled). Tracks with missing or
   broken data are dropped.
2. **Scale**: `StandardScaler` on whichever matrix will be clustered.
3. **Pick parameters**: explore or exploit, as above.
4. **PCA** (optional): reduce the dimensionality first; the component count that
   was actually used is recorded.
5. **Cluster**: fit K-Means, DBSCAN, GMM (`GMM_COVARIANCE_TYPE`) or Spectral
   (`SPECTRAL_N_NEIGHBORS`). Degenerate configurations, for example fewer than
   two clusters or more clusters than samples, are rejected with a fitness of
   -1.0. GPU models are used when `USE_GPU_CLUSTERING` is on and the GPU module
   is available, with automatic fallback to CPU.
6. **Filter and score**: turn the clusters into candidate playlists and compute
   the fitness score.

The return value carries the fitness score, the named playlists, the per-cluster
centroids (both the feature-space version used for naming and the
clustered-space version used for the Top-N diversity step) and the parameters
that produced them.

#### Automatic parameter discovery

When `CLUSTERING_AUTO_CALIBRATION` is on, up to
`CLUSTERING_CALIBRATION_MAX_TRIES` quick single-iteration probes tune the
parameters per server against one fixed stratified sample:

- K-Means, GMM and Spectral tune their cluster or component range. A small
  library pins the range straight to `TOP_N_CLUSTERING_PLAYLIST` clusters, never
  above `subset_size / (2 * MIN_PLAYLIST_SIZE_FOR_TOP_N)` and never below
  `subset_size / CLUSTERING_MAX_PLAYLIST_SONGS`. Each probe runs at the **top** of
  the range, which is the worst case for empty playlists.
- DBSCAN has no cluster count, so its `eps` range is derived from the data with a
  k-distance heuristic. The configured 0.1 to 0.5 default is unusable in the
  200-dimension embedding space, where every point would be noise. Oversized
  components are re-split with K-Means, and the probes widen `eps` when the
  playlists come out tiny and tighten it when they come out oversized.
- A probe only passes if it produces at least `TOP_N_CLUSTERING_PLAYLIST`
  playlists with at least `MIN_PLAYLIST_SIZE_FOR_TOP_N` songs each.

#### The fitness score

Each iteration is reduced to one number: a weighted sum of seven metrics, with
weights supplied by the user. A metric is only computed when its weight is not
zero.

The three **structural** metrics are rescaled so that higher is always better:

- **silhouette**: `(silhouette_score + 1) / 2`, mapped to 0-1.
- **davies_bouldin**: `1 / (1 + davies_bouldin_score)`. Davies-Bouldin is
  lower-is-better, so this inverts it.
- **calinski_harabasz**: `1 - exp(-CH / 500)`, a saturating squash to 0-1.

These three need at least two clusters and fewer clusters than samples, otherwise
they stay at 0.

The four **content** metrics describe how musically coherent the playlists are.
Each is a raw sum, passed through `log1p`, then z-normalized against precomputed
corpus statistics so the four are comparable before weighting. There are separate
statistics for embedding-based and feature-based clustering
(`LN_*_EMBEDING_STATS` against `LN_*_STATS`):

- **mood_diversity**: sums the predominant mood score of each distinct playlist
  mood. It rewards a set of playlists that between them span many moods.
- **mood_purity**: measures how strongly the songs inside a playlist actually
  carry that playlist's top `TOP_K_MOODS_FOR_PURITY_CALCULATION` moods. It
  rewards internally consistent playlists.
- **other_feature_diversity** and **other_feature_purity**: the same two ideas
  applied to the six other features, gated by
  `OTHER_FEATURE_PREDOMINANCE_THRESHOLD_FOR_PURITY` so only features a cluster
  genuinely leans into are counted.

The final score is the weighted sum. Diversity and purity pull against each other
(more and narrower playlists against fewer and broader ones), and the weights are
how the user tunes that trade-off.

**How purity is computed, concretely.** For each cluster a profile is formed: the
centroid itself when readable score vectors were clustered, or the average of the
member score vectors when embeddings were clustered. The top K moods of that
profile are taken. For each song, the intersection between its active moods and
the playlist's top K is computed, and the highest of those mood scores is kept
(a song with no intersection is skipped). Summing over the playlist gives the raw
purity.

For example, with a playlist whose top moods are pop 0.6, indie 0.4, vocal 0.35:
a song with indie 0.3, rock 0.7, vocal 0.6 contributes `max(0.3, 0.6) = 0.6`; a
song with indie 0.4, rock 0.45, vocal 0.3 contributes 0.4. The raw purity is 1.0,
which is then transformed and normalized.

**How diversity is computed, concretely.** For each playlist the single highest
scoring mood of its profile is taken with its score. Only the unique dominant
moods across all playlists are kept, and their scores are summed. Three playlists
dominated by indie 0.6, pop 0.5 and vocal 0.55 give a raw diversity of 1.65.

**Why both, and why also the geometric metrics.** Purity and diversity are
label-aware: they measure musical meaning and they are cheap, roughly linear in
the number of songs. Silhouette, Davies-Bouldin and Calinski-Harabasz measure
geometric separation and cohesion, which matters for structure but says nothing
about what the clusters *mean*. Using both, with tunable weights, is what lets
the same engine produce either tight thematic playlists or a broad, varied set.

#### From cluster to candidate playlist

Raw cluster membership is not used directly. Each cluster is trimmed:

- **Distance gate**: every point's distance to its cluster centre is normalized
  to 0-1, and members beyond `MAX_DISTANCE` are dropped, so loose outliers do not
  dilute the playlist. DBSCAN noise (label -1) is excluded outright.
- **Closest first**: the survivors are sorted by distance to the centre, so the
  most representative tracks are kept first.
- **Per-artist cap**: at most `MAX_SONGS_PER_ARTIST` songs per artist, matching
  the similarity and path features. Set it to 0 or less to disable.
- **Per-cluster cap**: at most `MAX_SONGS_PER_CLUSTER` songs, 0 meaning
  unlimited.
- **Naming**: the centroid is inverted back to feature space and a name is built
  from the tempo band (Slow, Medium, Fast), the top moods and any strongly
  present other feature, for example `Happy_Party_Fast`. When clustering on
  embeddings the name comes from the cluster's mean feature vector instead.

#### Batch orchestration

Iterations run as queue jobs, and the orchestrator manages them defensively. The
overriding goal is that the task **always finishes**, even if individual batches
die.

- **Batching**: the requested runs are split into batches of
  `ITERATIONS_PER_BATCH_JOB`, with up to `MAX_CONCURRENT_BATCH_JOBS` active.
- **Aggregation**: `_absorb_finished_batches` reaps each finished batch's row,
  collects its best result, updates the global best and feeds the elite pool.
- **Failure ceiling**: after `CLUSTERING_MAX_FAILED_BATCHES` failures no new
  batch launches and the remaining runs are force-completed.
- **No time-based watchdog**: a batch may take as long as the hardware needs.
  A batch whose worker died is requeued by the queue itself within seconds, and
  Cancel revokes the whole run on demand.
- **State recovery**: on restart the task reloads its children from the database
  and resumes. A task already in a terminal state is skipped.

If no valid solution was found across every run, finalization raises an error
rather than creating empty playlists.

#### Post-processing the winner

The single winning result is cleaned up before any playlist is created
(`tasks/clustering_postprocessing.py`), in this order:

1. **Duplicate filtering.** Inside each playlist: sort by title so near-identical
   titles are adjacent, drop exact title and artist duplicates after normalizing
   away suffixes such as *(Remastered)*, *[Explicit]* or *- Radio Edit*, then
   drop songs whose embedding distance to a recent neighbour is below the
   duplicate threshold, using the same metric and thresholds as the similarity
   feature (`DUPLICATE_DISTANCE_CHECK_LOOKBACK`). Vectors are read straight from
   the embedding table; if none exist it falls back to title and artist matching
   only. The playlist is then shuffled.
2. **Minimum size filter.** Any playlist with fewer than
   `MIN_PLAYLIST_SIZE_FOR_TOP_N` songs is dropped.
3. **Top-N diverse selection (6 + 4).** At most `TOP_N_CLUSTERING_PLAYLIST`
   playlists are returned. The three most represented primary genres in the
   library are found, and for each of them the farthest available centroid pair
   is kept, which gives six playlists. Four more are added whose genres differ
   from those three and from each other, chosen greedily to maximize each
   candidate's minimum centroid distance from the already selected set. If
   clustering did not provide enough alternatives, the remaining slots are filled
   by global maximum-minimum centroid distance. Fewer playlists are returned only
   when there are genuinely fewer viable candidates.

#### Naming and creation

`_name_and_prepare_playlists` names the survivors. With `AI_MODEL_PROVIDER` set
to `NONE` the tag-based name produced by the clustering itself is kept. With a
provider set the flow is:

1. A compact, grounded context is built for the cluster: its most frequent
   primary genre, the average mood and other-feature scores, whether it is
   instrumental, and lyric axis labels but only when their playlist-level vote is
   decisive. Broad axis labels are turned into safe title concepts rather than
   invented scenes, which keeps small local models useful.
2. The prompt asks for a single short concept, with explicit format rules. Recent
   names are kept per server for `PLAYLIST_NAME_HISTORY_ROUNDS` rounds and passed
   as negative history, so a recurring concept is not accepted again.
3. The returned text is repaired and cleaned in code: mojibake is fixed, Unicode
   is normalized to ASCII, a character whitelist is enforced, the length is
   truncated and the `_automatic` suffix is appended.
4. If the model declines or the output cannot be sanitized into a valid name, the
   deterministic feature-based name is kept.

With `AI_NAMING_PROMPT_MODE` set to `title` the model writes the whole title
instead. The prompt is the editable `AI_NAMING_TITLE_PROMPT` followed by a sample
of the playlist songs (`MAX_SONGS_IN_AI_PROMPT`) and the last titles already
used. A title must be 5 to 40 characters; a provider error, a wrong length or a
repeat of one of the listed titles is retried up to `AI_NAMING_MAX_ATTEMPTS`
times with feedback. If only repeats come back the repeated title is kept and the
run's duplicate suffix tells the playlists apart; if nothing usable comes back
the tag-based name is kept. The setup wizard's Preview titles runs one quick
K-Means on up to 10000 songs in a worker and shows the titles either style
produces, without creating any playlist.

Finally a Fisher-Yates shuffle randomizes the order, playlists larger than
`MAX_SONGS_PER_CLUSTER` are split into numbered chunks, the existing `_automatic`
playlists are deleted and the new ones are created on the media server and
recorded in the `playlist` table.

#### Re-running the clustering

Clustering is idempotent at the output level. Every run starts by deleting the
existing `_automatic` playlists and ends by recreating them, so re-running simply
replaces the previous set. There are no clustering tables to drop. It only
**reads** the analysis tables, so a new clustering never requires re-analyzing
audio or lyrics.

### 5.3. Environment Variable Configuration

**Main configuration**

- `CLUSTER_ALGORITHM`: default algorithm (`kmeans`, `dbscan`, `gmm`,
  `spectral`).
- `ENABLE_CLUSTERING_EMBEDDINGS`: cluster on the raw embeddings (true) or on the
  readable score vector (false).
- `CLUSTERING_RUNS`: number of evolutionary iterations. Higher is slower and
  usually better.
- `TOP_N_CLUSTERING_PLAYLIST`: how many diverse playlists to keep at the end.
- `MAX_DISTANCE`: normalized distance beyond which a member is dropped from its
  cluster.
- `MAX_SONGS_PER_CLUSTER`: maximum songs per playlist, 0 for unlimited.
- `MAX_SONGS_PER_ARTIST`: maximum songs from one artist in one playlist.
- `CLUSTERING_CLEANING`: run the duplicate cleanup during post-processing.
- `USE_GPU_CLUSTERING`: use RAPIDS cuML models when available, see [GPU](GPU.md).

**Automatic calibration**

- `CLUSTERING_AUTO_CALIBRATION`, `CLUSTERING_CALIBRATION_MAX_TRIES`,
  `CLUSTERING_MAX_PLAYLIST_SONGS`, `CLUSTERING_SUBSET_SONGS`,
  `CLUSTERING_EARLY_STOP_BATCHES`.

**Algorithm ranges**

- `NUM_CLUSTERS_MIN`, `NUM_CLUSTERS_MAX`.
- `DBSCAN_EPS_MIN`, `DBSCAN_EPS_MAX`, `DBSCAN_MIN_SAMPLES_MIN`,
  `DBSCAN_MIN_SAMPLES_MAX`.
- `GMM_N_COMPONENTS_MIN`, `GMM_N_COMPONENTS_MAX`, `GMM_COVARIANCE_TYPE`.
- `SPECTRAL_N_CLUSTERS_MIN`, `SPECTRAL_N_CLUSTERS_MAX`, `SPECTRAL_N_NEIGHBORS`.
- `PCA_COMPONENTS_MIN`, `PCA_COMPONENTS_MAX` (0 disables PCA).
- `USE_MINIBATCH_KMEANS`, `MINIBATCH_KMEANS_PROCESSING_BATCH_SIZE`.

**Task tuning**

- `ITERATIONS_PER_BATCH_JOB`, `MAX_CONCURRENT_BATCH_JOBS`,
  `CLUSTERING_MAX_FAILED_BATCHES`.

**Evolutionary tuning**

- `CLUSTERING_TOP_N_ELITES`, `CLUSTERING_EXPLOITATION_START_FRACTION`,
  `CLUSTERING_EXPLOITATION_PROBABILITY`, `CLUSTERING_MUTATION_INT_ABS_DELTA`,
  `CLUSTERING_MUTATION_FLOAT_ABS_DELTA`,
  `CLUSTERING_MUTATION_KMEANS_COORD_FRACTION`.

**Fitness weights and normalization**

- `SCORE_WEIGHT_DIVERSITY`, `SCORE_WEIGHT_PURITY`,
  `SCORE_WEIGHT_OTHER_FEATURE_DIVERSITY`, `SCORE_WEIGHT_OTHER_FEATURE_PURITY`,
  `SCORE_WEIGHT_SILHOUETTE`, `SCORE_WEIGHT_DAVIES_BOULDIN`,
  `SCORE_WEIGHT_CALINSKI_HARABASZ`.
- `LN_MOOD_DIVERSITY_STATS`, `LN_MOOD_PURITY_STATS`,
  `LN_MOOD_DIVERSITY_EMBEDING_STATS`, `LN_MOOD_PURITY_EMBEDING_STATS`,
  `LN_OTHER_FEATURES_DIVERSITY_STATS`, `LN_OTHER_FEATURES_PURITY_STATS`: the
  precomputed mean and standard deviation used to normalize the raw scores.
- `TOP_K_MOODS_FOR_PURITY_CALCULATION`,
  `OTHER_FEATURE_PREDOMINANCE_THRESHOLD_FOR_PURITY`.

**Sampling**

- `STRATIFIED_GENRES`, `MIN_SONGS_PER_GENRE_FOR_STRATIFICATION`,
  `STRATIFIED_SAMPLING_TARGET_PERCENTILE`, `SAMPLING_PERCENTAGE_CHANGE_PER_RUN`.

**Post-processing and naming**

- `MIN_PLAYLIST_SIZE_FOR_TOP_N`, `DUPLICATE_DISTANCE_THRESHOLD_COSINE`,
  `DUPLICATE_DISTANCE_THRESHOLD_EUCLIDEAN`, `DUPLICATE_DISTANCE_CHECK_LOOKBACK`.
- `AI_MODEL_PROVIDER` and the provider settings, `CLUSTER_NAMING_AI_HISTORY`,
  `PLAYLIST_NAME_HISTORY_ROUNDS`, `MAX_SONGS_IN_AI_PROMPT`.
- `AI_NAMING_PROMPT_MODE` and `AI_NAMING_TITLE_PROMPT`: the naming style and the
  editable instructions of the full-title style, set in the setup wizard under AI Prompt.

---

## 6. Playlist from Similar Song

This feature builds a playlist around one seed song.

### 6.1. Functional Analysis (High-Level)

1. The user opens the **Playlist from Similar Song** page.
2. They type an artist and/or a title. An autocomplete dropdown suggests matching
   songs from the library and the user picks one. The seed can also be a mood
   centroid or a saved alchemy anchor instead of a song.
3. Options:
   - **Number of results**: how many songs the playlist should contain.
   - **Limit songs per artist**: caps how many tracks by the same artist can
     appear.
   - **Radius similarity**: switches between two ways of finding and ordering the
     results, described below.
4. **Find Similar Tracks** returns a table with title, artist and distance.
5. If there are results, a **Create Playlist** section appears with a suggested
   name, and the playlist is created on the selected media server.

This page is **per server**: results are filtered to tracks the selected server
actually has.

### 6.2. Technical Analysis (Algorithm-Level)

#### Index loading

At startup the web process loads the audio IVF index and its id map. A background
thread listens on the Postgres `audiomuse_event` channel and reloads the index in
place when the workers publish a new build, so a long analysis does not require a
restart.

#### Autocomplete

Typing calls `GET /api/search_tracks`, which runs an indexed text query against
the `score` table. The results are scoped to the selected server.

#### Finding the neighbours

`GET /api/similar_tracks` takes `item_id` (or `title` and `artist`, or a `mood`
centroid, or an `anchor_id`), `n`, `eliminate_duplicates`, `radius_similarity`
and `mood_similarity`. The backend:

1. Resolves the input id: a provider id from the selected server is translated to
   the canonical catalogue id before touching the index.
2. Looks up the seed vector and queries the IVF index for more candidates than
   the user asked for, because filtering will remove some.
3. Branches on the mode.

**Standard mode** applies filters in order and then returns the top `n` sorted by
their distance to the seed:

1. **Distance filter**: removes candidates that sit almost exactly on top of a
   song already kept, using `DUPLICATE_DISTANCE_THRESHOLD_*` and
   `DUPLICATE_DISTANCE_CHECK_LOOKBACK`. This is what removes alternate masters
   and re-releases of the same track.
2. **Name deduplication**: removes candidates with the same title and artist as
   the seed or as a song already kept.
3. **Mood similarity filter** (optional, `MOOD_SIMILARITY_ENABLE` or the request
   parameter): removes candidates whose six other features differ from the seed
   by more than `MOOD_SIMILARITY_THRESHOLD`.
4. **Artist cap**: keeps at most `MAX_SONGS_PER_ARTIST` songs per artist.

**Radius mode** produces a playlist that flows rather than one that is simply
sorted by distance. The candidate pool is prepared with the same filters, then a
bucketed greedy walk runs:

- Candidates are sorted by their distance to the seed and grouped into
  fixed-size buckets, so the walk fans out from close to far.
- It starts from the closest valid candidate and repeatedly picks the next song
  by looking only at a limited number of nearby buckets.
- The choice balances closeness to the **previously selected** song against
  closeness to the **original seed**, which is what makes consecutive tracks
  sound related instead of jumping around.
- The artist cap is enforced **during** the walk, not afterwards, so one artist
  cannot take over the early part of the playlist. A separate rule avoids three
  songs by the same artist in a row.

The final order is the order of the walk. Radius mode always returns exactly `n`
songs when the pool allows it.

#### Playlist creation

`POST /api/create_playlist` takes a name and the list of track ids. The canonical
ids are translated back to the selected server's provider ids, tracks the server
does not have are dropped, and the playlist is created through that server's
adapter. The response reports how many tracks were unavailable.

### 6.3. Environment Variable Configuration

- `IVF_INDEX_NAME`, `IVF_NPROBE`, `EMBEDDING_DIMENSION`: index selection and
  query quality, see [chapter 4](#4-similarity-indexes-disk-paged-ivf).
- `MAX_SONGS_PER_ARTIST`: the artist cap applied when "Limit songs per artist" is
  on.
- `DUPLICATE_DISTANCE_THRESHOLD_COSINE`, `DUPLICATE_DISTANCE_THRESHOLD_EUCLIDEAN`,
  `DUPLICATE_DISTANCE_CHECK_LOOKBACK`: the near-duplicate filter.
- `MOOD_SIMILARITY_ENABLE`, `MOOD_SIMILARITY_THRESHOLD`: the optional mood
  filter and how strict it is.
- `SIMILARITY_ELIMINATE_DUPLICATES_DEFAULT`, `SIMILARITY_RADIUS_DEFAULT`: the
  default state of the two checkboxes when the API call omits them.
- `RADIUS_INSTRUMENTATION`: extra per-bucket logging for the radius walk.
- `IVF_METRIC`: decides which distance function and which duplicate threshold
  apply.

---

## 7. Song Path

Song Path builds a playlist that starts at one song, ends at another, and moves
gradually between them.

### 7.1. Functional Analysis (High-Level)

1. The user opens the **Song Path** page and picks a start and an end.
   Both endpoints can be a song, a mood, a saved anchor, or a pair of songs whose
   path should follow the **lyrical** meaning instead of the sound.
2. **Songs in path** sets the total number of tracks, including the two
   endpoints. **Keep path size exact** decides whether the algorithm must reach
   that exact length.
3. **Find Path** returns the ordered list plus a chart of the progression
   (distance per step, distance to the start and to the end, a 2D view).
4. The path can then be saved as a playlist.

If **Keep path size exact** is off, the path may come out shorter when no
suitable song exists for a step. If it is on, the algorithm works harder to fill
every slot.

### 7.2. Technical Analysis (Algorithm-Level)

`GET /api/find_path` takes `start_song_id`, `end_song_id`, `max_steps` and
optionally `path_fix_size`. Defaults come from `PATH_DEFAULT_LENGTH` and
`PATH_FIX_SIZE`.

1. **Vectors.** The embeddings of both endpoints are read from the index. A
   lyrics path reads from the SemGrove index instead, so the trajectory follows
   lyrical meaning with the sound as a secondary signal.
2. **Initialization.** The used-id and used-signature sets start with the two
   endpoints so they cannot reappear, and per-artist counters are prepared for
   the `MAX_SONGS_PER_ARTIST` cap.
3. **Centroid interpolation.** The requested number of points is interpolated
   between the two vectors, linearly for `euclidean` or as a spherical
   interpolation for `angular`, following `PATH_DISTANCE_METRIC`. The endpoints
   are removed, leaving the intermediate targets.
4. **Song selection.** For each intermediate target the nearest neighbours are
   fetched and the first candidate that passes every check is taken: not already
   used, not a duplicate title and artist, under the artist cap, and not too
   close to the songs already chosen (the same duplicate distance rule used
   everywhere else).
   - With **`path_fix_size` off**, one song is picked per target with a small
     search radius. A target with no valid candidate is simply skipped, so the
     path can end up shorter than requested.
   - With **`path_fix_size` on**, the targets are first grouped into jobs, each
     job asking for as many songs as the targets it covers. Jobs are processed in
     order. When a job cannot find enough songs it is **merged** with the next
     one: a new centroid is interpolated across the combined span, the search
     radius is increased (capped), the required count is summed, and the merged
     job is retried in place. This continues until everything is found or the
     last job cannot merge further.
5. **Final path.** The end song is appended, the full details are fetched from
   the database, and the total path distance is the sum of the distances between
   consecutive songs. That is what the chart draws.

Playlist creation reuses `POST /api/create_playlist` exactly like the similarity
page.

### 7.3. Environment Variable Configuration

- `PATH_DISTANCE_METRIC`: `angular` or `euclidean`. It decides both the
  interpolation and the step distance.
- `PATH_DEFAULT_LENGTH`: default number of songs in the path.
- `PATH_FIX_SIZE`: default for "Keep path size exact".
- `PATH_CANDIDATES_PER_STEP`: neighbours sampled per step, and used by the
  heuristic that groups targets into jobs.
- `PATH_AVG_JUMP_SAMPLE_SIZE`, `PATH_LCORE_MULTIPLIER`: sampling and sizing
  helpers used when estimating a reasonable local jump distance.
- `MAX_SONGS_PER_ARTIST`, `DUPLICATE_DISTANCE_THRESHOLD_*`,
  `DUPLICATE_DISTANCE_CHECK_LOOKBACK`: the shared candidate filters.

---

## 8. Song Alchemy

Song Alchemy defines a target sound by example: add the things you want, subtract
the things you do not, and get back the tracks that match the blend.

### 8.1. Functional Analysis (High-Level)

1. The user opens the **Song Alchemy** page and adds items. An item can be a
   **song**, an **artist**, a **mood**, an existing **playlist** or a saved
   **anchor**.
2. Each item is marked **Include** or **Exclude**. At least one Include is
   required.
3. Options:
   - **Number of results**: how many songs to return.
   - **Sampling temperature**: low values stay very close to the target blend,
     high values explore further and give more variety.
   - **Subtract distance threshold**: how far the results must be from the
     excluded profile.
4. The result is a 2D scatter plot showing the input songs, the computed add and
   subtract centroids, the kept songs and the ones removed by the subtract
   filter, plus a table of the kept songs.
5. The selection can be saved as a playlist, saved as a reusable **anchor**, or
   turned into a **radio**.

**Anchors** store a blend so it can be reused as a seed anywhere else (similar
song, path, radio). Besides the add centroid, an anchor stores every Include
point of the run without averaging them, each with its weight, and every
exclusion with its radius. Re-running it later (a radio included) therefore
searches around the same seed neighbourhoods as the original run instead of only
the area around their average, and keeps working when an input song, artist or
playlist later disappears from a server or from the catalogue.

**Radios** are saved anchors that a scheduled task re-runs regularly and pushes
to the media server as a playlist that is replaced in place, so a client that
syncs "online first" keeps following the same playlist.

### 8.2. Technical Analysis (Algorithm-Level)

1. **Input processing.** `POST /api/alchemy` receives the list of items, each
   with a type, an id and an operation, plus `n`, `temperature` and optionally a
   `subtract_distance` override. Song ids are resolved from the selected server's
   provider ids to canonical ids first.
2. **Anchor points per item type.**
   - *Song*: its embedding.
   - *Artist*: the means of that artist's Gaussian mixture, weighted by the
     component weights, so a varied artist contributes several points rather than
     one blurred average. See [chapter 11](#11-artist-similarity).
   - *Mood*: a precomputed mood centroid.
   - *Playlist*: the vectors of its member tracks, capped by
     `ALCHEMY_PLAYLIST_MAX_SONGS` and reduced to at most
     `ALCHEMY_PLAYLIST_MAX_CENTROIDS` centroids.
   - *Anchor*: the stored Include points (weights normalised so the whole
     anchor counts as one item; anchors saved before these were stored fall back
     to their single centroid), on whichever side the anchor is placed, so an
     excluded anchor removes the area around each of its points like an excluded
     artist does. An included anchor also re-applies its stored exclusions at
     their saved radius, and its stored song seeds are kept out of the results,
     like the input songs of a live run. The points are stamped with a SHA-256
     prefix of the embedding model file and the dimension; an anchor saved under
     a different model, or whose centroid has the wrong size, is ignored with a
     warning (also by Similar Song and Song Path) until it is saved again.
   The number of points queried is capped by `ALCHEMY_MAX_ANCHOR_POINTS`; when
   there are more, every input keeps its heaviest point first.
3. **Centroids.** The Include points are averaged into the add centroid and the
   Exclude points into the subtract centroid.
4. **Candidate search.** The index is queried around each Include point
   separately. Every input gets an equal share of a pool of three times the
   requested count (split among its own points by weight), so one input with
   many points cannot crowd out the others. When `MAX_SONGS_PER_ARTIST` is above
   0 and `SIMILARITY_ELIMINATE_DUPLICATES_DEFAULT` is on, each point fetches five
   times its quota so the artist cap still leaves enough songs.
5. **Filtering.** The original input songs and an anchor's stored song seeds are
   removed first. With the artist cap on, it keeps each artist's songs closest
   to the Include points, then each input fills its quota from its own ranked
   neighbours. After the subtraction below, the standard near-duplicate distance
   filter and the title and artist deduplication are applied.
6. **Subtraction.** If there is a subtract centroid, every remaining candidate
   closer to it than the threshold
   (`ALCHEMY_SUBTRACT_DISTANCE_ANGULAR` or `ALCHEMY_SUBTRACT_DISTANCE_EUCLIDEAN`,
   or the request override) is removed. Those songs are returned separately so
   the plot can show what was excluded and why.
7. **Temperature sampling.** The distance of each survivor to its nearest
   Include point is turned into a similarity score, and the scores are passed
   through a softmax with the requested temperature. A low temperature sharpens the distribution and
   effectively takes the closest songs; a high temperature flattens it and mixes
   in more distant ones. A single song at temperature 0 short-circuits to a plain
   nearest-neighbour query.
8. **Projection.** The candidates, the centroids and the excluded songs are
   projected to 2D (UMAP, PCA or a discriminant projection depending on
   availability and shape) so the frontend can plot them.
9. **Response.** The kept results, the excluded ones, the 2D coordinates of the
   centroids and inputs, the projection method used, and the exclusion vectors
   with their radius so an anchor can persist them.

**Radios** (`tasks/radio_manager.py`) run the same function from a stored anchor
once per server in scope, then upsert the playlist with
`create_or_replace_playlist`, falling back to a plain create when the provider
does not support replacing. A radio that returns nothing is skipped so the
previous playlist is preserved rather than emptied. A radio is an **online**
feature: it queries the in-memory similarity index, which only the Flask process
loads, so both callers run it there directly rather than queueing it to a worker
that has no index. The scope depends on who starts the run: the `alchemy_radio`
cron tick covers every configured server (scheduled work always does), while the
*Create Radio Playlists* button on the Alchemy page targets only the server
selected in the sidebar, because that page is per server.

### 8.3. Environment Variable Configuration

- `ALCHEMY_DEFAULT_N_RESULTS`, `ALCHEMY_MAX_N_RESULTS`: the count used when the
  caller names none, and the ceiling the Alchemy page puts on its own input box.
  The maximum is frontend-only: the API and the engine enforce the default and a
  floor of 1, never the cap, so a direct API caller can ask for any number.
- `ALCHEMY_TEMPERATURE`: default sampling temperature.
- `ALCHEMY_SUBTRACT_DISTANCE_ANGULAR`, `ALCHEMY_SUBTRACT_DISTANCE_EUCLIDEAN`:
  default subtract thresholds. The metric in use decides which one applies.
- `ALCHEMY_PLAYLIST_MAX_SONGS`, `ALCHEMY_PLAYLIST_MAX_CENTROIDS`,
  `ALCHEMY_MAX_ANCHOR_POINTS`: limits on how much a single item may contribute.
- `MOOD_CENTROIDS_FILE`: the precomputed mood centroids used by mood items.
- `MAX_SONGS_PER_ARTIST`, `DUPLICATE_DISTANCE_THRESHOLD_*`,
  `DUPLICATE_DISTANCE_CHECK_LOOKBACK`: the shared candidate filters.

---

## 9. Music Map

The Music Map is a 2D picture of the whole library, where songs that sound alike
sit close together.

### 9.1. Functional Analysis (High-Level)

**How the map works.** Analysis passes the audio through a neural network that
does not output human-readable attributes such as tempo or energy (those are
stored separately). It outputs a vector of 200 numbers. That vector means nothing
to a person but a lot to the algorithm, because it captures the patterns that
make similarity search work. To draw it on a screen we need two numbers, so a
second machine learning step (UMAP) compresses 200 dimensions into 2. That
compression is an approximation, which is why a path drawn on the map does not
always look perfectly straight: the picture is a simplified view of a much richer
space.

**Workflow**

1. The page loads a subset of the library (25 percent by default) as a scatter
   plot, with points coloured by their top mood or genre.
2. Buttons switch between 25, 50, 75 and 100 percent. A clickable legend hides or
   shows individual genres.
3. Clicking a point adds it to the selection. Lasso and box selection add many at
   once. The selection list can be edited or cleared.
4. The search box highlights a song on the map, centres the view on it and adds
   it to the selection.
5. With a selection, the user can create a playlist, or (with 2 to 10 songs
   selected) compute the paths between consecutive selected songs and draw them
   on the map.

### 9.2. Technical Analysis (Algorithm-Level)

#### Building the cache

1. During the index rebuild, `build_and_store_map_projection` computes 2D
   coordinates for every embedding and stores them with their id list. This is
   the precomputed projection.
2. At web startup a background thread runs `build_map_cache`, which reads the
   catalogue (id, title, author, mood vector, embedding), loads the stored
   projection, and computes coordinates on the fly only for songs that do not
   have them yet.
3. It builds a lightweight record per song (id, title, artist, 2D coordinates and
   the single top mood), then takes deterministic samples at 100, 75, 50 and 25
   percent.
4. Each sample is serialized to JSON once and also gzipped once, and both are
   held in memory. Requests are then served with zero further work.

#### Serving

`GET /api/map?percent=50` looks the bucket up in the cache and returns the
pre-gzipped bytes when the client accepts gzip, otherwise the raw JSON. The
response sets `Cache-Control: no-store` so the browser always gets the current
data.

Multi-server adds a second cache layer. The shared cache always holds the full
union in canonical ids. A per-server cache, keyed by server and percentage,
holds each server's own pre-gzipped bucket with that server's provider ids, so a
request never has to translate the whole catalogue on the fly. All servers are
warmed when the cache is built; a server added later is filled lazily on first
use.

#### Frontend

The plot uses Plotly with the WebGL scatter type, one trace holding every point
with the item id in `customdata`. Selection, genre filtering, search highlighting
and drawn paths are all managed as Plotly shapes and re-applied when the trace is
rebuilt, so zoom, pan and overlays survive a genre filter change. The Song Path
button calls `/api/find_path` for each consecutive pair and draws the segments.

### 9.3. Environment Variable Configuration

The map has few settings of its own. What matters is the data behind it:

- The PostgreSQL connection built from the `POSTGRES_*` parts: read at startup
  to build the cache.
- The **embeddings** produced by analysis decide the layout, and the
  `mood_vector` decides the colours and the legend.
- The projection method actually used (stored UMAP projection, or an on-the-fly
  fallback) is reported in the API response and shown under the map.
- `PATH_*`, `MAX_SONGS_PER_ARTIST` and the duplicate thresholds apply when the
  Song Path button is used from the map.

---

## 10. Sonic Fingerprint

The Sonic Fingerprint turns a user's listening history into a personal playlist.

### 10.1. Functional Analysis (High-Level)

1. The analysis must have run, and the media server must track play counts and
   last-played times.
2. The user opens the **Sonic Fingerprint** page and enters the credentials of
   the server currently selected in the sidebar, because listening history is per
   user. Defaults from the server configuration may pre-fill some fields.
3. **Number of results** sets the size of the final playlist.
4. **Generate My Sonic Fingerprint** returns a mix of the user's top played songs
   and new songs that match their taste, with a distance column.
5. The result can be saved as a playlist on that user's account.
6. The results can be played in the page with the shared song player, and the
   **Smart fade** and **Demo** options ticked before the search apply to them
   (section 17.1, steps 3 to 5, and chapter 19).

A scheduled version of the same job exists. It writes to a playlist with a stable
name (`SONIC_FINGERPRINT_CRON_PLAYLIST_NAME`) and replaces it in place, so a
client that syncs keeps following the same playlist instead of collecting a new
one every run.

### 10.2. Technical Analysis (Algorithm-Level)

1. **Credentials.** `POST /api/sonic_fingerprint/generate` receives `n` and the
   user credentials. For Jellyfin and Emby a username is resolved into the user
   id the API needs. The `server` parameter selects which server the history,
   downloads and results come from.
2. **Top played songs.** The adapter returns the user's
   `SONIC_FINGERPRINT_TOP_N_SONGS` most played tracks. On Navidrome the list is
   also capped per album at `SONIC_FINGERPRINT_MAX_SONGS_PER_ALBUM` tracks, so
   one long DJ mix cannot take over the whole profile. The other providers return
   their own ranking without that cap.
3. **Canonicalization and deduplication.** Provider ids are resolved to canonical
   ids. Because two files can now resolve to the same catalogue row, the list is
   deduplicated in play-count order, keeping the highest ranked provider id. This
   stops one song being counted twice in the centroid.
4. **Embeddings.** The embeddings of those songs are read from the database.
   Songs without one are skipped.
5. **Recency weights.** For each song the last-played timestamp is fetched and
   turned into a weight with an exponential decay whose half-life is 30 days:
   `weight = exp(-decay_rate * days_since_played)`. A song with an unparseable
   date gets 0.5 and a song with no date at all gets 0.25, so an old or unknown
   play still counts but counts less.
6. **The fingerprint.** The weighted average of those embeddings is the user's
   sonic fingerprint vector.
7. **Expansion.** The index is queried around that vector for as many new songs
   as are needed to reach the requested size, with duplicate elimination on so
   the result is varied.
8. **Combination.** The final list starts with the seed songs the fingerprint was
   built from, then adds the new neighbours, skipping duplicates, until the
   target size is reached. The titles and artists are then fetched for display.
9. **Playlist creation** uses the same endpoint as the other features, and the
   user credentials are passed along so the playlist lands on the right account.

### 10.3. Environment Variable Configuration

- `SONIC_FINGERPRINT_TOP_N_SONGS`: how many top played songs form the profile.
- `SONIC_FINGERPRINT_MAX_SONGS_PER_ALBUM`: cap per album in the seed pool.
- `SONIC_FINGERPRINT_NEIGHBORS`: default total size of the generated playlist.
- `SONIC_FINGERPRINT_CRON_PLAYLIST_NAME`: the stable name used by the scheduled
  run.
- `MAX_SONGS_PER_ARTIST`, `DUPLICATE_DISTANCE_THRESHOLD_*`,
  `DUPLICATE_DISTANCE_CHECK_LOOKBACK`, `IVF_METRIC`: the shared neighbour
  filters.
- Media server credentials are read from the registry; the page can override them
  per user.

---

## 11. Artist Similarity

Artist Similarity answers "which artists sound like this one", which is a
different question from "which songs sound like this one".

### 11.1. Functional Analysis (High-Level)

1. The user opens the **Artist Similarity** page and searches for an artist.
2. The page returns a ranked list of similar artists with a divergence score
   (lower means closer). Optionally it can also show which parts of each artist
   matched, which is useful for artists who work in more than one style.
3. Selecting an artist lists their tracks, which can be turned into a playlist.

The same index also powers the artist items in Song Alchemy and the artist
similarity call that media server plugins use for their Radio feature.

### 11.2. Technical Analysis (Algorithm-Level)

An artist is not represented by the average of their songs. Averaging a band that
plays both ballads and hard rock gives a vector that matches neither. Instead:

1. **One Gaussian mixture per artist.** During the index rebuild, each artist's
   track embeddings are fitted with a diagonal-covariance Gaussian mixture. The
   number of components is chosen automatically inside a small range (2 to 10),
   so a one-style artist gets few components and a varied artist gets several.
   Each component is effectively "one of the things this artist does".
2. **Parallel fitting.** The fits are pure Python, so they run across
   `INDEX_BUILD_WORKERS` processes. The track embeddings are streamed in batches
   so the whole library is never in memory.
3. **Artist-to-artist distance.** Two artists are compared with a soft Chamfer
   distance over their component means: for each component of one artist, find
   how close the other artist gets to it, and combine those in both directions.
   The result is low when every side of artist A has something matching in artist
   B, and it does not require the two artists to have the same number of styles.
4. **Query.** `GET /api/similar_artists` accepts an artist name or an artist id,
   resolves it against the index (with a normalized fallback so punctuation and
   case differences still match), and returns the `n` closest artists with their
   divergence. With `include_component_matches=true` it also returns which
   component matched which.
5. **Server scoping.** Artist names and ids are translated through
   `artist_server_map`, so the page returns the ids of the selected server.

### 11.3. Environment Variable Configuration

- `INDEX_BUILD_WORKERS`: processes used to fit the per-artist mixtures during an
  index rebuild.
- The IVF settings of [chapter 4](#4-similarity-indexes-disk-paged-ivf) apply to
  the artist index like any other.
- The component range, covariance type and fitting limits are internal constants
  of the artist manager rather than environment variables, because they are tied
  to the shape of the embedding.

---

## 12. Text Search (DCLAP)

Text Search lets the user describe music in plain language instead of filtering
on metadata.

### 12.1. Functional Analysis (High-Level)

1. The analysis must have run with CLAP enabled, so the tracks have CLAP
   embeddings.
2. The user opens the **Text Search (DCLAP)** page. It automatically warms up the
   text model and shows a status indicator with the remaining warm time. Each
   search resets that timer.
3. The user types a query of at least three characters, or clicks one of the
   suggested queries.
4. Results are a table of title, artist and a similarity score from 0 to 1,
   ordered by relevance. The default is 100 results, up to 500.
5. Results can be saved as a playlist, with the query as the default name.

Text Search works best with short queries (two to four words) that combine a
genre, an instrument and a mood, for example "energetic rock guitar" or "calm
piano instrumental". Genre and instrument recognition is the strongest; mood is
less precise.

The page has a second tab, **By Song**. The user picks a song from the library
with the standard song picker and gets the songs whose DCLAP sound is closest to
it. It is the DCLAP counterpart of the Similar Song page, which uses the MusiCNN
index. On a set of classical seeds DCLAP kept far more results in the seed's
genre than MusiCNN, whose tag vocabulary has no classical label.

### 12.2. Technical Analysis (Algorithm-Level)

#### Split models

CLAP is used as two separate ONNX models, and that split is deliberate:

- The **audio model** is the distilled DCLAP student model. It is loaded in the
  **worker** containers during analysis and produces a 512-dimension embedding
  per track.
- The **text model** is the original LAION CLAP text encoder. It is much larger
  and it is loaded in the **web** container only when a search needs it.

So a worker never loads the text model and the web process never loads the audio
model. Both sides produce L2-normalized vectors, which makes cosine similarity a
plain dot product.

#### During analysis

The audio is resampled to 48 kHz mono, turned into a mel-spectrogram with the
`CLAP_AUDIO_*` parameters, and passed through the audio model. The embedding is
stored in the `clap_embedding` table, keyed by the catalogue id.

#### At search time

1. **Warm up.** `POST /api/clap/warmup` loads the text model and starts a
   countdown of `CLAP_TEXT_SEARCH_WARMUP_DURATION` seconds. Every search resets
   the countdown; when it expires the model is unloaded and the memory is
   returned. `GET /api/clap/warmup/status` reports the remaining time to the UI.
2. **Embed the query.** The text is tokenized with the RoBERTa tokenizer from the
   `transformers` library and passed through the text model, then normalized.
3. **Search.** The query vector is matched against the CLAP IVF index, which is
   built and served exactly like the other indexes (see
   [chapter 4](#4-similarity-indexes-disk-paged-ivf)), and the top results are
   mapped back to titles and artists.
4. **Scope and return.** Results are filtered to the selected server and returned
   with their similarity score.

#### Search by song

`POST /api/clap/search_by_song` takes the seed's `item_id`, a `limit` and the
optional `eliminate_duplicates` and `radius_similarity` switches. It reads the seed's exact stored
`clap_embedding` row, so the text model is never loaded and a song analysed after
the last index build can still be a seed, and runs it through the same query and
duplicate filters as a text search with the Similar Song rules:

- The seed is claimed before any neighbour: its id, its title and artist and its
  vector. Neither the seed nor a copy of it (same title and artist, or a
  near-identical sound under another id) is returned.
- The seed does not use one of its own artist's slots.
- `eliminate_duplicates` turns the per-artist limit (`MAX_SONGS_PER_ARTIST`) on or
  off. When it is omitted, `SIMILARITY_ELIMINATE_DUPLICATES_DEFAULT` decides, as on
  the Similar Song page, and the tab shows the same "Limit songs per artist"
  checkbox.
- `radius_similarity` runs the same radius walk as the Similar Song page: a pool of
  `limit + max(20, 3 * limit) + 1` deduplicated neighbours goes through the shared
  `radius_walk_helper`, which picks and orders the songs in distance buckets for a
  smoother flow and applies `MAX_SONGS_PER_ARTIST` itself. When it is omitted,
  `SIMILARITY_RADIUS_DEFAULT` decides (on by default). The By Song tab never sends
  it, so the page always uses the default; only an API caller can turn the walk off
  with `radius_similarity: false`. If the walk fails, the error is logged and the
  plain nearest-neighbour order is returned. The mood filter of the Similar Song
  page is not applied.

When the filters leave fewer songs than asked, the candidate pool is fetched once
more at four times the size, unless the first query already returned every song
its probed cells hold. The song picker only offers songs with a DCLAP embedding
(`/api/search_tracks` with `index=clap`). A song without a DCLAP embedding answers
404, and so does a seed with no similar song left on the selected server.

#### Suggested queries

At startup a background thread can precompute a set of inspiring queries. It
loads category-weighted terms from `tasks/query.json`, generates
`CLAP_TOP_QUERIES_COUNT` random short queries, scores them by how distinct and
how productive they are, and keeps the best ones. `GET /api/clap/top_queries`
serves them as clickable buttons.

### 12.3. Environment Variable Configuration

- `CLAP_ENABLED`: master switch. With it off, no CLAP embedding is produced
  during analysis and the page is hidden.
- `CLAP_AUDIO_MODEL_PATH`, `CLAP_TEXT_MODEL_PATH`: the two ONNX models. The audio
  model needs its companion `.onnx.data` file in the same directory.
- `CLAP_EMBEDDING_DIMENSION`: 512, fixed by the model, used for validation.
- `CLAP_AUDIO_N_MELS`, `CLAP_AUDIO_N_FFT`, `CLAP_AUDIO_HOP_LENGTH`,
  `CLAP_AUDIO_FMIN`, `CLAP_AUDIO_FMAX`, `CLAP_AUDIO_MEL_TRANSPOSE`: the
  spectrogram parameters. They must match the model that was trained on them.
- `CLAP_PYTHON_MULTITHREADS`: false lets ONNX Runtime manage its own threads
  (recommended); true makes it single-threaded and expects parallelism at the
  Python level.
- `CLAP_TEXT_SEARCH_WARMUP_DURATION`: how long the text model stays loaded after
  the last use.
- `CLAP_TOP_QUERIES_COUNT`, `CLAP_CATEGORY_WEIGHTS`: how many candidate queries
  are generated and how the categories (Genre, Mood, Energy, Tempo,
  Instrumentation, Voice type, Production, Era) are weighted.
- `CLAP_OTHER_FEATURES_CACHE_FILE`: the on-disk `.npz` caching the text embeddings of
  the six other-feature labels used during analysis.

**Operational notes**

- Because the models are split and lazily loaded, the resident memory of each
  role stays predictable: the workers hold only the audio model, the web process
  holds the text model only while it is being used, and the index itself is
  disk-paged.
- After analyzing new albums, `POST /api/clap/cache/refresh` reloads the
  embeddings without restarting the web process.
- CUDA is used when available. The text model normally runs fine on CPU; the
  audio model is the one that benefits from a GPU.

---

## 13. Lyrics Search

Lyrics Search is the text-side counterpart of Text Search. It searches what songs
are *about*, not what they sound like.

### 13.1. Functional Analysis (High-Level)

The page has three tabs, each a different way of asking the same question.

**By Axis.** Five dropdowns, one per lyrical axis (setting, social dynamic,
emotional valence, narrative temporality, thematic weight). The user picks a
value on one or more axes and leaves the rest on "none". More axes means a more
specific search. This is the most predictable mode, because the axes are a fixed
vocabulary rather than free text.

**By Text.** A free text description of a theme, for example "leaving a small
town at night". The description is embedded with the same multilingual model used
for the lyrics themselves, so it can match songs in any language without
translation.

**By Song.** Pick a song and get songs with a similar *meaning*. This mode uses
the SemGrove index, which fuses the lyrics and the audio vectors, so results are
lyrically related but still musically plausible. The page shows whether the
SemGrove index is built, how many songs it covers and the current weighting.

Results can be saved as a playlist like everywhere else. The Song Path page uses
the same SemGrove index for its lyrics path mode.

### 13.2. Technical Analysis (Algorithm-Level)

#### Axis search

Each track carries a score for the 27 axis labels described in
[chapter 3](#3-lyrics-analysis). Those scores form their own IVF index. The user's
selection becomes a target vector over the same axes, only for the axes that were
set, and the index returns the closest tracks. Instrumental tracks carry a fixed
sentinel value on every axis, so they stay comparable without pretending to have
a theme.

#### Text search

`POST /api/lyrics/search/text` embeds the query with `gte-multilingual-base` and
queries the lyrics semantic index. The model is warmed on demand and unloaded
after `LYRICS_GTE_WARMUP_DURATION` seconds of inactivity, exactly like the CLAP
text model, so the memory is only held while it is useful.

#### SemGrove

SemGrove is a separate index built from **both** modalities:

1. Each song's lyrics vector and audio vector are L2-normalized on their own, so
   neither dominates simply because it has a larger norm.
2. Each modality is whitened using statistics computed over the corpus, so the
   dimensions of both sides are on a comparable scale.
3. The two are weighted and concatenated. The weights are square-rooted so that
   their squares equal `SEM_GROVE_WEIGHT_LYRICS` and `SEM_GROVE_WEIGHT_AUDIO`,
   which means the weights behave as the intended share of the squared distance.
   The default is 75 percent lyrics and 25 percent audio.
4. The merged matrix is built through a temporary disk-backed file, so the build
   memory stays bounded, then stored as a normal IVF index.

Because the weights are baked in at build time, changing them requires an index
rebuild. `POST /api/sem_grove/search` takes a seed song and returns similar
songs with a radius walk that applies the same near-duplicate suppression and
artist caps as the audio similarity search. A song only appears in the index when
it has **both** lyrics and audio analysis.

### 13.3. Environment Variable Configuration

- `LYRICS_ENABLED`: with it off, the page is hidden and no lyrics index is built.
- `LYRICS_EMBEDDING_DIMENSION`, `LYRICS_GTE_MAX_TOKENS`, `LYRICS_MODEL_DIR`: the
  text embedding model.
- `LYRICS_GTE_WARMUP_DURATION`: how long the text model stays loaded after the
  last query.
- `SEM_GROVE_WEIGHT_LYRICS`, `SEM_GROVE_WEIGHT_AUDIO`: the fusion weights, taken
  into account at build time only.
- `DUPLICATE_DISTANCE_THRESHOLD_COSINE_LYRICS`: the near-duplicate threshold used
  on lyrics vectors, which are naturally closer together than audio vectors.

---

## 14. Instant Playlist (Chat)

Instant Playlist turns a sentence into a playlist. It is the only feature that
uses a language model at request time.

### 14.1. Functional Analysis (High-Level)

1. The user opens the **Instant Playlist** page, which offers the AI provider and
   model controls plus one text box.
2. They describe what they want, for example "upbeat workout with electronic
   funk", "sad songs about leaving home" or "no rap, something for studying", and
   click the button.
3. The page shows the progress step by step, then the resulting playlist.
4. A collapsible section shows what the AI actually decided, for transparency.
5. The playlist can be saved to the selected media server.

**What the AI does and does not do.** The model does **not** write database
queries and it does **not** invent song titles. It chooses which of the available
search tools to use and with which arguments. Everything after that is
deterministic code running real queries against the analyzed library. That is
what keeps the results grounded in music the user actually owns.

### 14.2. Technical Analysis (Algorithm-Level)

#### One call, four tools

There is a single tool-calling request per user prompt. There is no separate
classifier step. The model sees the full tool surface and emits one or more tool
calls:

- `seed_search`: find songs similar to a named song or artist. Its blend modes
  merge the seeds (union), blend them into one flavour (alchemy), remove one
  flavour (subtract), or build a **journey** that starts at the first seed and
  walks to the second through the song-path finder, keeping that order.
- `text_match`: match a description against the audio (DCLAP) or the lyrics.
  Modes are only exposed when the matching feature is enabled.
- `knowledge_lookup`: the brainstorm tool, for requests that need outside
  knowledge rather than a library lookup.
- `search_database`: metadata and feature filtering (genre, mood, tempo, energy,
  key, scale, year, rating, track length, recently added, exclusions).

Energy is spoken as 0.0 (calm) to 1.0 (intense), and that scale is a **library
percentile**: 0.35 means the calmest 35% of the analyzed songs. The percentiles
are read from the library itself, so the energy filters keep working even when
the stored `ENERGY_MIN`/`ENERGY_MAX` range no longer matches the analyzed values.

The tool descriptions carry the routing rules, so the choice of tool is driven by
the schemas rather than by a long prompt. The prompt text and the structured
output grammar are both **derived from those same schemas**, which means adding
or changing a tool updates every provider at once and nothing can drift. Enum
values for genres, moods and voice types come from the canonical vocabulary, and
array arguments carry maximum-item caps so a small model cannot loop a value
forever.

#### Making small local models reliable

This feature is designed to work with a small self-hosted model, so the
intelligence lives in the schemas and in deterministic code rather than in prompt
prose:

- **Hint pre-extraction.** Plain regular expressions read the request before the
  call:
  - years, decades (including "early/late 90s" and a few other languages' decade
    words) and relative eras ("last 5 years", "recent");
  - BPM values and bounds, tempo, energy and activity words;
  - key and scale;
  - track length ("under 3 minutes", "long songs") and a playlist time budget
    ("an hour of");
  - song count, a per-artist cap ("one song per artist") and excluded versions
    ("no live versions");
  - "recently added";
  - genres and negated genres, with negation words in several languages.

  Anything the model then leaves out is merged back into the filter afterwards.
  Relative eras and explicit instrumental wording override the model, since a
  model cannot know today's date.
- **Sound words.** Genres outside the vocabulary (for example techno or
  post-rock) and atmosphere words (dark, dreamy, cinematic) have no metadata
  field. When the plan is filter-only, an audio `text_match` on those words is
  added so the filter re-ranks a sound-matched pool.
- **Instruments.** A single DCLAP text vector lets the loudest facets of a
  request win, so "pop viola with a female voice" comes back as pop with a
  female voice and no viola. Instrument words (viola, sax, rhodes, tabla...) are
  mapped to the validated concepts of the DCLAP sparse autoencoder (section on
  concept steering), and then:
  - the sound search always runs, even when the model planned a metadata-only
    search, and its query is steered mildly (x3) toward the instrument, which
    enriches the pool without losing the genre and the voice (a strong x10 steer
    returns instrumental covers);
  - every candidate is read by the same SAE, and a song counts as "has the
    instrument" when the concept fires in the library's top 5% (the threshold is
    a sampled percentile of the library, cached for an hour);
  - the genre and the instrument form the first ranking tier, so songs with
    both come first; when too few candidates carry every requested value, an
    instrument-led search (x10) tops the pool up.
- **Copied examples and negations.** When the model's `text_match` query repeats
  the prompt's own example (three or more of its words that the request does not
  contain), the request's own words are used instead. A `text_match` query that
  is only a negation ("nothing explicit", "no vocals") is dropped: an embedding
  match reads it as its opposite.
- **Hallucination stripping.** Year, instrumental and exclusion arguments that do
  not appear in the request are removed. An exclusion only survives if the
  request actually contains a negation, and an excluded genre only if the
  request names it (directly or by an alias such as "rap").
- **Plan repairs.** A single-point tempo or energy range is widened to a window.
  An album named in the request is looked up in the library and added to the
  filter. `min_rating` is dropped when no song has a rating. "From X to Y" with
  two seeds becomes a journey. "Like X but calmer / more upbeat / faster" is measured
  against the seeds' own average energy and tempo instead of a fixed threshold.
  Track-length and recently-added arguments the request never mentions are
  stripped, like hallucinated years.
- **Deduplication and caps.** Duplicate tool calls are dropped and a plan is
  capped at four calls.
- **One replan.** If the plan returns nothing at all, exactly one replan runs with
  the failure as feedback.

#### Composing the result

When several tools return candidates, the results are merged and re-ranked:

- Songs matching the requested categorical values are ranked above songs that do
  not. These are genre, voice type, scale and instrumental, plus the **explicit
  ranges**: a year range, a BPM number, a track length and "recently added",
  where only a song inside the range counts as a match. A requested genre ranks
  above the other categorical matches. The continuous dimensions only order songs
  **within** a tier, so the categorical request is a strong preference and not a
  hard gate.
- A filter-only genre request puts the songs whose main style is that genre
  first, keeping the database's relevance order inside each group.
- Songs returned by more than one tool get an intersection boost.
- The primary tool's own similarity rank is blended in as an extra dimension.
- When the filter names a genre and the finder's pool holds fewer songs matching
  every requested value than the playlist needs, library songs that do match are
  added to the pool. They rank after the similar songs.
- A continuous range is a gate, not a maximizer: every song inside it scores the
  same, so similarity orders them ("calm" does not mean "the quietest drone"),
  and a song outside it scores by its distance to the range.
- Titles that look like intros, skits or interludes, and tracks under a minute,
  are pushed down.
- `exclude_artists` and `exclude_genres` are the one **hard** cut. Excluded
  versions (live, remix, cover...) are removed by title, unless that would leave
  nothing.
- If a filtered pool comes up short, a relax loop lowers the score threshold to
  backfill. A filter-only query that still underfills re-runs without its soft
  dimensions (tempo, energy, moods, key, scale, rating, track length, recently
  added) and then applies them as the soft re-rank over the broader pool.

`knowledge_lookup` is the one exception: its results are returned as they are. It
already grounds itself, because the model emits a *recipe* (filters, sound
descriptions and seed artists) which is then run against the real library and
fused, rather than recalling song titles that may not exist. Applying the normal
re-rank on top of that would fight the brainstorm.

#### Streaming and playlist creation

The playlist length comes from the request's `n` (the chat page's "Number of
songs" box), defaulting to `INSTANT_PLAYLIST_DEFAULT_N_RESULTS`. A count written
in the request ("15 songs") wins over the box. A time budget ("an hour of")
sizes the list and then trims it by the tracks' real durations. The pool the
tools fill is ten times the length, with a floor of 1000, so dedup and the artist
diversity cap still leave enough to select from. A per-artist cap written in the
request replaces the default cap and is kept even when the list comes out
shorter. A journey keeps its path order and is never reordered.

`POST /chat/api/chatPlaylist` returns the final result in one response.
`POST /chat/api/chatPlaylistStream` streams the same run as Server-Sent Events, so
the page can show each step as it happens. Optionally
`tasks.playlist_ordering.order_playlist` reorders the final list for a smoother
flow: a greedy nearest-neighbour walk over a combined tempo, energy and key
distance, starting from a low-energy track, with an optional energy arc (build up
then wind down) for playlists of ten tracks or more.

`POST /chat/api/create_playlist` creates the playlist on the selected server,
translating the ids and reporting anything unavailable.

#### Safety

The AI never receives database credentials and never emits SQL. Queries are
parameterized code paths that run as the application's own database user through
the standard connection helper (`database.connect_raw`, read-only variant) with
the server-side session option `default_transaction_read_only=on`, so the chat's
own library queries cannot write; no dedicated chat role is configured or
created. Installs upgraded from a release that created the retired `ai_user`
role may still carry it: at every Flask start the app tries to log in as that
role with the shipped default password and, if the login succeeds, sets it
`NOLOGIN`; otherwise it is left untouched and a warning names it. On a server
that accepts any password (trust authentication, as in the embedded database of
the standalone builds) the login always succeeds, so the role is disabled there
too. Nothing is revoked or dropped; to remove it entirely run `DROP OWNED BY
ai_user; DROP ROLE ai_user;` as a superuser (a pre-2.0.0 backup that grants to
it can then no longer be restored). Provider API keys stay server-side. Tool
failures return a generic message; the real error only reaches the container log.

### 14.3. Environment Variable Configuration

- `AI_MODEL_PROVIDER`: `OLLAMA`, `OPENAI`, `GEMINI`, `MISTRAL` or `NONE`.
- `OLLAMA_SERVER_URL`, `OLLAMA_MODEL_NAME`: the local Ollama endpoint and model.
  The page may override the URL per request.
- `OPENAI_SERVER_URL`, `OPENAI_MODEL_NAME`, `OPENAI_API_KEY`: any
  OpenAI-compatible endpoint.
- `GEMINI_API_KEY`, `GEMINI_MODEL_NAME`, `GEMINI_API_CALL_DELAY_SECONDS`.
- `MISTRAL_API_KEY`, `MISTRAL_MODEL_NAME`, `MISTRAL_API_CALL_DELAY_SECONDS`.
- `AI_REQUEST_TIMEOUT_SECONDS`: hard timeout on a provider call.
- `AI_TOOLCALL_TEMPERATURE`: sampling temperature for the tool-calling request.
  Do not set it to 0 with Qwen-family models, greedy decoding degrades their tool
  calls.
- `INSTANT_PLAYLIST_DEFAULT_N_RESULTS`, `INSTANT_PLAYLIST_MAX_N_RESULTS`: how
  many songs the playlist aims for when the caller sends no `n`, and the ceiling
  the chat page puts on its own "Number of songs" box. The maximum is
  frontend-only: the API enforces the default and a floor of 1, never the cap.
- `MAX_SONGS_PER_ARTIST_PLAYLIST`: diversity cap inside an instant playlist.
- `PLAYLIST_ENERGY_ARC`: enable the energy arc when ordering.
- `AI_BRAINSTORM_SOUND_DESCRIPTIONS_MAX`, `AI_BRAINSTORM_SEED_ARTISTS_MAX`,
  `AI_BRAINSTORM_USE_ARTIST_SEEDS`, `AI_BRAINSTORM_SIMILAR_ARTISTS_PER_SEED`,
  `AI_BRAINSTORM_LYRIC_THEMES_MAX`, `AI_BRAINSTORM_GENRE_SCORE_THRESHOLD`,
  `AI_BRAINSTORM_POOL_FLOOR`, `AI_BRAINSTORM_RELAX_YEAR_PAD`: how the brainstorm
  recipe is built and how far it relaxes when the pool is too small.
- `ALCHEMY_DEFAULT_N_RESULTS`, `ALCHEMY_MAX_N_RESULTS`: the result caps shared
  with Song Alchemy.

---

## 15. Database Cleaning

Cleaning keeps the database in step with the media servers after files are moved
or removed.

### 15.1. Functional Analysis (High-Level)

1. An admin opens **Administration > Cleaning**. The page shows a summary, a
   Start button, a per-run option and a status panel.
2. Starting the task enqueues a background job. The page shows a live log, a
   progress bar and a final summary, and the task can be cancelled.
3. The job reports which tracks each server no longer has, removes only that
   server's stale mappings, and rebuilds the similarity indexes.

**The rule: what a server returns is what it has.**

- A song that disappeared from **one** server keeps its analysis, its embeddings
  and its mappings on the other servers. It simply stops appearing in results for
  the server that lost it.
- A song found on **no** server is an orphan. By default it is only reported. It
  is deleted if `CLEANING_CATALOGUE` is on, or if the per-run checkbox is ticked,
  at most `CLEANING_SAFETY_LIMIT` albums per run. The next run deletes the next
  albums.
- A server whose fetch raised an error was not read: its own mappings stay, and
  the songs it may still hold are not treated as orphans. Every other server is
  still cleaned.

Like analysis and clustering, cleaning always covers **every** configured server.

### 15.2. Technical Analysis (Algorithm-Level)

1. **Enqueue.** `POST /api/cleaning/start` validates the request, writes a
   pending `task_status` row and enqueues
   `tasks.cleaning.identify_and_clean_orphaned_albums_task` on the high priority
   queue. There is no retry policy: a task that raises is failed outright, and
   the row is requeued only if the worker running it dies and the maintenance
   pass reclaims it.
2. **Enumerate.** For each configured server the job fetches the current track set
   through the **same** helpers the alignment sweep uses
   (`fetch_server_catalogue`), with that server's library filter applied. Reusing
   the sweep's own enumeration means the prune baseline can never disagree with
   the enumeration that created the mappings in the first place.
3. **Prune per server.** `prune_stale_mappings` removes only that server's rows
   from `track_server_map` for tracks it no longer has. No ratio guard applies:
   an id the server did not return is unbound.
4. **Orphans.** Tracks found on no server are grouped by album (album artist and
   album name). When the catalogue option is enabled for this run, the first
   `CLEANING_SAFETY_LIMIT` albums (largest first) are deleted and reported; the
   rest stay for the next run. Songs still mapped to a server whose fetch raised
   are never orphans, and with the legacy single-server fallback an unread server
   means nothing was checked. The delete removes the score row, the embeddings
   and the playlist references together.
5. **Library sizes.** Each server's stored track count is refreshed from the
   fetch that already happened, which keeps the dashboard coverage figure
   current.
6. **Duplicate repair (Path B).** Merged duplicate groups whose stored
   Chromaprints prove the files are different recordings are split. This corrects
   a false merge once the files have fingerprints. It is skip-if-missing and it
   only unmaps, it never deletes.
7. **Index rebuild.** The same full index rebuild that analysis runs happens
   **inline**, and the task is not reported complete until the indexes reflect
   the cleaned catalogue and the reload message has been published.

Database errors surface as error 4001 and fail the job. Failures are
collected and returned in the summary rather than aborting the run.

### 15.3. Environment Variable Configuration

- `CLEANING_SAFETY_LIMIT`: maximum number of orphaned albums deleted in one run.
- `CLEANING_CATALOGUE`: whether orphan catalogue rows are deleted as well as
  reported. The page has a per-run checkbox that enables it for one run without
  changing the default.
- `CHROMAPRINT_GATE_ENABLED` and the other Chromaprint settings: used by the
  duplicate repair step, see
  [chapter 2](#2-catalogue-identity-and-deduplication).
- The `POSTGRES_*` parts and the media server registry credentials.

---

## 16. Scheduled Tasks (Cron)

Scheduled Tasks run the long jobs automatically.

### 16.1. Functional Analysis (High-Level)

1. An admin opens **Administration > Scheduled Tasks**. Each supported task type
   has a cron expression field and an Enable checkbox.
2. The supported types are **analysis**, **clustering**, **sonic fingerprint**,
   **album of the week**, **alchemy radio**, and any task a plugin has registered.
3. The user enters an expression, for example `0 2 * * 0-5` for weeknights at 2
   am, enables it and saves. An expression that could never fire is rejected
   before it is stored as enabled.
4. A scheduled job starts exactly like a manual one and appears in the same task
   panel, so it can be monitored and inspected.

Scheduled batch tasks always run against **all** configured music servers, the
same as when they are started from the page.

### 16.2. Technical Analysis (Algorithm-Level)

1. **Persistence.** `GET`/`POST /api/cron` read and write the `cron` table
   (`name`, `task_type`, `cron_expr`, `enabled`, `last_run`, `options`).
2. **Matching.** A poll thread reads the enabled rows and tests each expression
   against the current time. The matcher supports `*`, single numbers,
   comma-separated lists and ranges, over minute, hour, day of month, month and
   day of week, converting Python's weekday numbering to the cron convention
   (0 = Sunday).
3. **Atomic claim.** A row that matches is claimed atomically for its wall-clock
   minute. This is what makes a restart, or a second web process, unable to
   double-fire the same schedule. A tick evaluates the current minute only: a
   minute that passed while an inline run held the poll thread is skipped,
   never replayed (replaying needs a clock the platforms do not agree on, and
   firing late is worse than not firing). The claim re-checks that the row is
   still enabled with the same expression, so a change saved on the page
   during a long online run is honoured.
4. **Batch work is enqueued, online work runs inline.** Analysis, clustering and
   plugin tasks are **enqueued** as queue jobs, so a slow media server cannot
   swallow a scheduling window or block the other schedules. The **alchemy
   radio**, the **sonic fingerprint** and the **album of the week** are online
   features that query the in-memory similarity index, which only the Flask
   process loads, so the tick runs them inline right there through one shared
   scaffold. Each still gets a task row (STARTED, then SUCCESS or FAILURE, with
   the same classified error a queue job records) and so stays visible in the
   task panel; the cost is that the poll thread waits for the run, which is the
   accepted trade for a schedule that fires once a day or once a week. Within
   one tick the batch rows are dispatched first, since they only enqueue; then
   at most ONE online row runs, in a fixed order: any other online row due in
   the same minute is skipped with a warning and is not claimed, so the page
   keeps showing it did not run, and it waits for its next occurrence. One
   online run at a time, never several together; the Scheduled Tasks page
   warns on save when schedules share a minute.
   They are self-managed: they run beside a live batch task and no batch start waits for
   them. The sonic fingerprint and the album of the week take nothing but a
   server scope, so both the dispatch entry and the task body they run are
   declared once: the registry holds the function each row runs, and
   `tasks.task_run.run_playlist_task_per_server` holds the per-server loop,
   cancel check, heartbeat, reporter and playlist upsert they share.
5. **Queue guard.** Analysis, clustering, plugin tasks (and, when started
   manually, cleaning and provider migration) are mutually exclusive: a
   scheduled batch run is skipped while any other queue-guard task is still
   queued or running, so a schedule cannot pile heavy runs on top of each
   other. The online inline runs are outside the guard.
6. **Retry on conflict.** A skipped scheduled batch run (analysis, clustering or
   a plugin task) is recorded in a `cron_retry` list instead of being dropped
   silently. The cron thread re-attempts it every
   `CRON_RETRY_INTERVAL_MINUTES` (clamped below `CRON_RETRY_MAX_MINUTES`), up to
   `CRON_RETRY_MAX_MINUTES` after the first block; once the guard clears it
   starts. Past the window it is recorded as a visible failed run and is not
   started, not even once more (fail-safe). The window is never extended: a
   scheduled run waits at most `CRON_RETRY_MAX_MINUTES`, never forever. A SUCCESS of the same task type
   that started after the first block counts as the scheduled run having
   happened and drops the retry. `GET /api/cron` exposes the pending state
   (`retry_pending`, `retry_attempts`, `retry_blocker_task_type`) so the
   Scheduled Tasks page can show that a schedule is waiting instead of looking
   like it fired normally. AudioMuse-AI goes first: while a scheduled analysis
   or clustering waits in the list, no plugin task starts, neither a plugin
   schedule nor a plugin's own `enqueue`. A plugin task already running just
   finishes, the waiting run starts at the next retry pass, and plugins resume
   after it.
7. **Error isolation.** An exception on one row is logged and the loop continues
   with the others. A failed enqueue is recorded as a failed task so it is
   visible in the UI.

### 16.3. Environment Variable Configuration

Cron reuses the defaults of the tasks it starts:

- `TOP_N_MOODS`: passed to a scheduled analysis, which always scans the whole
  library.
- `CLUSTER_ALGORITHM`, `NUM_CLUSTERS_MIN`, `NUM_CLUSTERS_MAX`, `DBSCAN_*`,
  `GMM_*`, `SPECTRAL_*`, `PCA_COMPONENTS_MIN`, `PCA_COMPONENTS_MAX`,
  `CLUSTERING_RUNS`, `MAX_SONGS_PER_CLUSTER`, `TOP_N_CLUSTERING_PLAYLIST`,
  `MIN_SONGS_PER_GENRE_FOR_STRATIFICATION`,
  `STRATIFIED_SAMPLING_TARGET_PERCENTILE`, the `SCORE_WEIGHT_*` weights and the
  AI naming settings: used to compose the scheduled clustering job.
- `SONIC_FINGERPRINT_CRON_PLAYLIST_NAME`: the stable playlist name used by the
  scheduled sonic fingerprint.
- `ALBUM_OF_THE_WEEK_PLAYLIST_NAME`: the stable playlist name used by the
  scheduled album of the week.
- `CRON_RETRY_MAX_MINUTES`: how long a scheduled run blocked by the queue guard
  waits in the retry list before it is recorded as skipped.
- `CRON_RETRY_INTERVAL_MINUTES`: how often the cron thread re-attempts blocked
  scheduled runs.
- `TZ`: the timezone the expressions are evaluated in.

---

## 17. Search by Recording

Search by Recording turns a few seconds of audio captured outside the library
(a phone recording of what is playing in a room, an uploaded file) into the
question "which song is this, and where in it": the clip's neural fingerprint
is aligned on the fingerprint sequences the analysis stores for every track.

### 17.1. Functional Analysis (High-Level)

**Workflow**

1. The user opens **Search by Recording** and picks one of two tabs. On
   **Search by Recording** they either click **Record** (the browser records
   `RECORDING_SEARCH_RECORD_SECONDS` seconds from the microphone and stops
   by itself) or upload a clip. On **Search by Song** they pick a song of
   the library with the same picker as the similar-song page, which here
   offers only the songs in the neural fingerprint index that the selected
   server has, the way the lyrics page's picker offers only SemGrove songs.
2. **Search** returns the songs of the selected server, the best match first,
   each with its match score (the badge turns green when the match is
   certain), with the same result rows and the same "create a playlist"
   button as the other search pages. The song tab leaves the chosen song
   itself out and, like the similar-song page, leaves out duplicates: a
   song with the same title and author as the chosen song or as a better
   ranked result, the same song under another title (a remaster, an
   extended version, a compilation copy), and the songs past
   `MAX_SONGS_PER_ARTIST` per artist; a playlist made from the results
   starts with the chosen song, like the similar-song page does with its
   seed. On the recording tab the best match is already the first result.
3. **Smart fade**, ticked before the search, plays the results so each song
   flows into the next, mixed on the beat where the two songs allow it.
   Chapter 19 explains it: its two endpoints, its four kinds of transition and
   what was measured. Search by Song itself is unchanged by it.
4. **Play** (or a click on a row) plays the results in the browser, each
   song fetched once from the media server through
   `GET /api/recording_search/play/<id>`, which sends the file unchanged; the
   page decodes it with the browser's Web Audio and never uses an audio
   element. Without Smart fade the songs play whole, one after the other. The
   playing row fills in light blue as the song plays, with a thin line where
   its fade starts, and the player shows the elapsed and total time. A click
   or a drag on the playing row moves in the song, never closer than one
   second to that line. **Skip to the fade** jumps to 6 seconds before the
   next transition (section 19.1).
   On iOS the page asks for the playback audio session, so the silent switch
   does not mute it; iOS still pauses the page's audio while the page is in
   the background.
5. **Demo**, ticked before the search, makes **Play** play only the demo of
   each song, its chorus or drop (`POST /api/demo`, `app_demo.py`, one song
   or a list and an optional `seconds`, at least 5, by default
   `SONG_DEMO_SECONDS` from the setup wizard's Song Player section, 30; section
   17.2 step 5), one after the other with a 4-second crossfade, the last one
   stopping at its demo end; the rows show the demo instead. Demo and Smart
   fade exclude each other: ticking one unticks the other, so a search plays
   whole songs, only demos or only smart fade.
   **Skip to the fade** is shown only with Smart fade. Play itself works
   without the neural fingerprint; the two options need it and are not shown
   without it.
6. **One player for every song list.** The player bar (Play, Skip to the
   fade, Stop) sits right above the results and the two options in the
   search form of every page that answers a list of songs: Similar Song,
   Song Path, Hyperbolic (similar songs and the path), Text Search, the three
   Lyrics searches, Sonic Fingerprint, Search by Recording, Song Alchemy and
   Album Creation (which, like a path, keeps its running order under Smart
   fade), never the map or the clustering page. Its whole interface
   is `static/song_player.js` (with `static/song_demo.js` for the demos),
   loaded by `templates/includes/layout.html`; a page only gives the list
   and, where its playlist starts with one, the seed song that plays first.

**Important behaviours**

- In-page recording uses the browser microphone, which every browser allows
  only over HTTPS or on localhost; section 17.3 explains how the record
  button gets there on a plain-HTTP LAN address. No server-side code can
  lift the policy: it is the browser's own.
- Uploads go up to `RECORDING_SEARCH_MAX_UPLOAD_MB` (1 GB) and any container
  PyAV decodes is accepted, a video included, of which the sound track is
  used. The file is streamed to disk and only its first minute is decoded,
  so a big file costs transfer time, not memory.
- The match comes from the neural fingerprint the analysis stores for the
  whole track (the **neural-fingerprint** stage; a library analysed before it
  exists is re-analysed for that stage alone, one track at a time, on the
  next analysis run). It works from any part of the song and from a short
  phone recording, because its encoder was trained on exactly that
  degradation.
- Three other modes were built, measured and removed: MusiCNN and DCLAP
  similarity (a real phone recording ranked its own song 90,803rd in those
  spaces), a chromaprint alignment over the duplicate detector's fingerprints
  (only the first two minutes of each track exist there, and it needed a
  minute of recording for a phone clip), and a Whisper transcript searched in
  the lyrics index (needs words in the clip). The neural fingerprint
  identified every real phone clip they failed on.
- The page is per server: results are filtered and id-translated to the server
  selected in the sidebar.
- Practical advice for recordings: hold the phone close to a full-range source
  in a quiet room. On a real phone recording at -45 dBFS the music sat below
  the microphone's own noise from 200 Hz up, and no model, denoiser or
  channel correction could recover the song from it; the level normalisation
  cannot rescue that either.

### 17.2. Technical Analysis (Algorithm-Level)

1. **Decode.** The upload is streamed to a temporary file (refused past
   `RECORDING_SEARCH_MAX_UPLOAD_MB` while copying, so it never sits in RAM) and
   decoded at its native rate by the analysis loader (librosa, then the PyAV
   fallback that handles the browser's webm/opus). The loader never decodes
   more than `AUDIO_LOAD_TIMEOUT` seconds; the result is then cut to
   `RECORDING_SEARCH_MAX_CLIP_SECONDS`.
2. **Level.** The clip is RMS-normalised to `RECORDING_SEARCH_TARGET_LEVEL_DB`.
   The mel front ends have no per-clip normalisation, so a recording that is
   12 dB too quiet lands far from its own song; level is the one degradation
   the query side can undo exactly.
3. **Neural fingerprint.** The encoder is a published neural music
   fingerprinter (ISMIR 2025, the NAFP architecture trained with real room
   impulse responses, microphone responses and background noise, triplet
   loss), exported once from its TensorFlow
   checkpoint to `neural_fingerprint.onnx`, published in the model release
   and downloaded into the model directory next to the MusiCNN graphs (17.2
   million parameters, 71 MB) and run through the same ONNX provider chain as
   MusiCNN and CLAP, CUDA on the GPU images and the CPU everywhere else. The
   export is
   `scripts/onnx_export/export_neural_fingerprint_to_onnx.py`, driven by
   `run_exports.sh` next to it, which clones the source, downloads the
   checkpoint from Zenodo and checks the graph against TensorFlow. The
   analysis stage
   resamples the track it already decoded to 8 kHz, cuts one-second segments
   every half second, turns each into the model's 256-band mel patch (n_fft
   1024, hop 256, 160 to 4000 Hz, magnitude in dB relative to the segment's
   own peak, floored at -80 dB, scaled to [-1, 1]; the numpy front end
   matches the reference essentia one to 2e-4 on all 33 frames) and stores
   one L2-normalised 128-vector per segment as a 32-byte product-quantised
   code in `embedding.neural_fingerprint` (14 KB for an average track, about
   3 GB for 200k tracks). The codebook `neural_fingerprint_pq.npz` ships next
   to the model: 32 slices of four numbers, 256 centroids each, trained once
   on library fingerprints by
   `scripts/onnx_export/train_neural_fingerprint_codebook.py`; every blob
   carries the codebook's checksum, so a blob and a codebook that do not
   belong together are refused with an error in the log. Measured on the
   123-track test set against the int8 rows it replaces: the four real phone
   recordings stay identified at 0.46 to 0.66 instead of 0.48 to 0.68, the
   best wrong candidate stays below 0.32, and 32 bytes is the smallest size
   that keeps that margin (16 bytes puts the hardest clip on the 0.40
   threshold). The
   index over those codes follows the lifecycle of the other similarity
   indexes: the worker builds it at the rebuild points of the analysis run
   (every `REBUILD_INDEX_BATCH_SIZE` albums and at the end), stores it in
   `ivf_dir` in one transaction it commits itself (so a web process syncing
   at any moment sees the previous build complete or the new one complete,
   never a directory whose parts are half written), publishes the
   index-reload event, and the web process loads the directory from `ivf_dir`
   and reads the cells from `ivf_cell` exactly like the other indexes. The
   directory holds the quantizer as int8, the track order and lengths and the
   cell sizes; the cells hold the codes of their rows with the track, the
   offset in the track and the inverse norm of the decoded vector beside each
   one, 40 bytes per indexed row. Everything is sized for millions of tracks,
   because one track is 450 rows rather than one. The index holds every
   `NEURAL_FINGERPRINT_INDEX_STRIDE`-th stored row of a track (1, every half
   second, by default; 2 halves the pack and the query work at a recall cost
   on degraded clips that has to be measured on real recordings first; the
   blobs keep every row and the alignment check reads them all). The cells,
   about sqrt(rows) of them and at most 65536, are placed by a two-level
   k-means trained on at least `NEURAL_FINGERPRINT_TRAIN_ROWS` rows and 20 per
   cell, sampled 100 per track from random tracks: above 64 cells they are
   grouped under sqrt(cells) coarse centroids and a row goes to the nearest
   cell of its nearest group, so assigning a million tracks costs about 300
   dot products per row instead of tens of thousands (minutes rather than
   hours), while a query still ranks the flat cell list exactly. The rows
   are labelled and stored in parts of at most 8M rows, cut at track
   boundaries, so the worker never holds more than one part: each part
   writes one `ivf_cell` row per cell under the index name
   `neural_fingerprint_index/p<part>`, in bulk inserts. A rebuild appends
   instead of starting over: fingerprints never change and the library only
   grows, so the worker keeps the quantizer, reads only the tracks
   fingerprinted since the last build, assigns their rows and adds parts; the
   quantizer is retrained from scratch when the library has grown
   `NEURAL_FINGERPRINT_RETRAIN_GROWTH` times since it was trained, when more
   than a tenth of the indexed tracks are gone, when the largest cell holds
   more than ten times the average, or when the codebook or the stride
   changed. The web process loads only the directory, at startup and on the
   index-reload event, blocking like every other index load and taking
   seconds at any library size (a few megabytes: the song ids and the
   centroids); it is held as one immutable object swapped by a single
   reference assignment, so a query that started before a swap keeps a
   consistent view. Cells stored by a build other than the loaded directory's
   are refused ("being rebuilt") until the reload arrives, so new cells are
   never paired with an old directory. Nothing is copied to local disk. A query first lists the
   `NEURAL_FINGERPRINT_NPROBE` cells each of its segments probes, fetches the
   ones it does not have in one query (every part of each cell, a few
   hundred kilobytes per cell) and keeps them in a RAM cache bounded by
   `NEURAL_FINGERPRINT_CACHE_MB`, dropped when the recording search has been
   idle like the encoder; the cells are then scored in parallel threads
   (`NEURAL_FINGERPRINT_QUERY_THREADS`, one per core by default) through the
   codebook lookup table, one lookup per byte and a multiply. Every third
   segment votes first and, when one track already leads the runner-up
   fourfold with enough votes, the rest of the voting is skipped; the best
   candidates are then verified against their own blobs, fetched in one
   query, so a track deleted since the build has no blob and never comes
   back. Like every other index it holds the union of all servers: a
   request scoped to a server votes only over that server's tracks through
   the shared availability mask, cached per server and build for 30 s and
   dropped when the mappings change. Measured on 13,043
   real tracks (6.1 million rows, 2,470 cells): a full build takes 43 s (19 s
   for the centroids, 22 s to assign the rows, which the worker does on the
   GPU through cupy on the GPU images and on the CPU elsewhere, in blocks of
   65,536 rows across tracks because per-track matrices ran five times
   slower), the table holds 28 MB (4.6 bytes per row, about 430 MB at 200k
   tracks), the web process syncs its 219 MB pack in 5 s, an append of 1,000
   tracks takes 11 s in the worker and 3 s in the web process, and centroids
   trained on the first 30 percent of the tracks with the rest appended gave
   the same ranks and scores as a full build on 28 queries, eight of them the
   real phone recordings. On the CPU the assignment costs about 1.9 s per
   65,536 rows at 8,192 cells, so a from-scratch build of a 200k library is
   about 45 minutes there, paid only when the centroids are retrained; with
   the fourfold rule the largest retrain during a backfill to 200k happens
   near 64k tracks and takes about 8 minutes. A query is embedded the same
   way and scored against the probed rows through the codebook's lookup
   tables (32 additions per row instead of a 128-wide dot product, seven
   times faster at the same result); each of its vectors reads
   `NEURAL_FINGERPRINT_NPROBE` cells, the rows they hold are decoded through
   the codebook and vote for (track, offset), votes within one hop are
   pooled, and the twenty best tracks are
   verified by the mean cosine between the whole clip and the track at that
   offset, which is the score shown. The best track is identified when its
   score clears `NEURAL_FINGERPRINT_MIN_SCORE` and leads the next track by
   `NEURAL_FINGERPRINT_MIN_LEAD`. A first candidate model, a course project
   trained without room or microphone augmentation, was exported and
   measured first: it found 40 of 40 clean clips in a 300-track index and 0
   of 40 phone-like ones, and none of the real recordings, which is why the
   published, degradation-trained weights are the ones shipped. Measured with
   those weights in a 202-track index: 30 of 30 clean 20 s clips taken from
   random positions (scores 0.81 to 0.99), and all four real phone
   recordings at rank 1 with scores 0.48 to 0.68 and leads 0.33 to 0.51,
   among them a clip taken from the end of a track, which no stored
   chromaprint could ever match; a synthetic degradation harsher than the
   phone (ten random resonances, strong early reflections, noise) still gave
   17 of 30 with no wrong top candidate above 0.26. The thresholds sit
   between those groups. The same picture through the production path (a
   throwaway database, the stored blobs, the pack and its cells, nprobe 12,
   123 tracks): 30 of 30 clean clips identified in about half a second each,
   20 of 30 of the harsh synthetic ones found first with none flagged wrong,
   and the four real recordings identified at scores 0.48 to 0.68, still
   identified when cut to their first 10 seconds (0.56 and 0.58). The offset
   reported for a song with a repeated section may point at the repeat, not
   the second the phone heard. Costs to know: about 25 ms of CPU per second of
   audio to fingerprint (10 to 25 s per track, so a 200k library is weeks of
   analysis on one worker), 14 KB per track in the database, and the pack on
   disk is the same size again, read cell by cell.
4. **Search by Song.** No audio is decoded and no model runs: the chosen
   song's stored codes are decoded through the codebook, cut into up to
   three 20-second windows (a fifth, half and four fifths of the way in, so
   an edit that shares only part of the recording still matches), each
   window is aligned on the index exactly like a clip with the source track
   excluded from the vote, and the best score per song is kept. The ranked
   list then goes down once like the similar-song filters (name, then
   artist cap, then distance; a rejected song claims nothing): a song with
   the same title and author as the chosen song or a kept result is left
   out, so is a song past `MAX_SONGS_PER_ARTIST` (the chosen song does not
   count, a song without an author is exempt), and so is a fingerprint twin
   of the chosen song or of any kept result. The copies of one song are
   ranked far apart, so every kept song is compared, not a lookback. A twin
   needs two signals: the cosine between the two whole-song fingerprints
   (the mean of every half-second row) at most
   `DUPLICATE_DISTANCE_THRESHOLD_NEURAL_FINGERPRINT` (0.25) away, and a 20 s
   piece of one song slid along the other scoring 0.5 or more, the measure
   of this very search; only the rare pairs that pass the first are
   aligned, so only those are fully decoded. Each window asks the index for
   a quarter more songs (at least 20 more) so the count still holds after
   the filter. Measured on 20 real 100-song lists (101,000 pairs): every
   same title and author copy, 10 of 18 other versions (remix, extended,
   instrumental) and no unrelated song were left out; the two
   different-title drops were the same piece under another title. At 0.30
   one unrelated song was dropped too, the average alone at 0.25 dropped 5
   songs with unrelated titles, the first and last 30 s alone caught only 7
   versions, and the alignment alone is too slow to run on every pair. The
   identified flag and the lead are computed on the kept list.
   **Smart fade** orders songs from the same stored matrices (chapter 19).
5. **Demo** (`tasks/demo_manager.py`, `demo_window`, played by the shared
   song player, section 17.1). A chorus comes back
   several times with the same sound, so its half seconds match their own
   repeats far better than a verse or a bridge does (repetition-based audio
   thumbnailing). The
   song's stored matrix is compared with itself (the cosine of every half
   second with every other one), digital silence is blanked with the same
   0.95 consecutive-row rule as the smart fade (silent rows are the most alike
   of all and would otherwise win), and the matrix is averaged along its
   diagonals over 4 seconds so a repeat reads as a path. Every half second
   then scores the mean of its 2 best matches at least 20 seconds away, the
   window of the asked length with the best mean score wins, and a 0.2 bias
   toward the middle of the song breaks near ties between the repeats. A song
   no longer than the asked length plays whole, a track longer than 10
   minutes is averaged down to 1,200 rows first so the matrices stay a few
   MB, and a song without a fingerprint gets null times. No audio is decoded
   and no model runs: about 3 ms per song. Measured on the 223 songs of a
   205k-song library that carry hand-made section labels (the public Harmonix
   Set annotations): a 30-second demo holds 68% chorus and 87% of them hold
   10 seconds of chorus or more, against 41% and 61% for the window in the
   middle of the song and 40% for a random one, and 79% hold the start of a
   chorus (58% in the middle). Longer demos gain little: 60- and 100-second
   windows hold 10 seconds of chorus 96% and 99% of the time, the middle of
   the song 95% and 99%. Moving the start to the nearest section boundary
   was measured and did not help.
6. **The chromaprint alternative, measured and removed.** Before the neural
   fingerprint, the clip was fingerprinted with the duplicate detector's
   fpcalc and slid across the stored chromaprints with learned per-bit
   weights, a noise-frame mask, playback-speed and sub-hop phase variants and
   a lead rule over the next different recording. It reached rank 1 of
   200,453 on the reference phone clip and found the song first in 50 to 70
   percent of degraded library queries, but the stored fingerprints cover
   only the first 120 seconds of each track, a phone clip needed close to a
   minute, and a clip from the end of a track could never match. The
   neural fingerprint identified all of those clips at 10 to 20 seconds, so
   the chromaprint path was retired rather than kept as a second tab.
7. **The index in the web process.** Flask syncs and maps the stored build
   at startup like the other indexes ("Neural fingerprint index loaded at
   startup", or "not found" until the analysis has built it once) and keeps
   it mapped for the life of the process; only the reload event replaces it.
   The page's warmup call preloads the encoder session so the first search
   does not pay its load, and that session alone is released after
   `RECORDING_SEARCH_WARMUP_DURATION` seconds without a query, the same
   idle-unload pattern as the text-search models.

Measured before building it, on 50 songs against a real 198k-track corpus: a
clean random 20 s slice retrieves the same neighbourhood as the whole song
(its median overlap equals a genuinely similar song's), a level error of
-12 dB alone costs two thirds of that overlap, and a phone in front of a small
speaker in a noisy room is beyond what the current models recover.

### 17.3. The Record Button on a Plain-HTTP Address

Browsers hand the microphone (`getUserMedia`, `MediaRecorder`) only to pages
on HTTPS or on localhost; on `http://192.168.x.x:8000` the API does not even
exist, in Chrome, Safari, Firefox alike, and no script can lift that. A
self-hosted app is reached exactly that way, and a second port would have to
be published in every container deployment, so the one port the app already
binds answers both protocols (`tls_listener.py`). The first byte of a new
connection tells a TLS handshake (0x16) from an HTTP request line; a sniffing
thread waits for it on a selector, up to two seconds and for every pending
connection at once, so an idle browser preconnect never holds the server's
accept loop, which only pops connections already sorted. An HTTP
connection is handed to the server untouched; a TLS one is terminated in the
web process, with a self-signed certificate it creates once into
`FLASK_HTTPS_CERT_DIR`, on one relay thread that moves bytes between the
client and a local socket pair, keeping at most a megabyte in flight per
direction (a client sending faster than the app reads is simply not read
from until the app catches up), and the server reads plain HTTP from the
pair's other end with the real client address. Gunicorn, waitress and
werkzeug therefore need no TLS support of their own: the gunicorn worker hook
in `gunicorn.conf.py` (read by gunicorn on its own) swaps the accept of the
sockets the worker inherited, the native builds bind a dual-protocol socket
for waitress, and `app.run` does the same for the development server. On an
insecure page a short notice gives the same page's HTTPS address on the same
host and port (the port the browser reached, so a container port published as
8080:8000 works); the browser warns once about the certificate and recording
works on every later visit. That one warning is the only user
step: a certificate a browser trusts silently needs a domain name and a
public or private certificate authority, which a raw LAN address cannot have.
Health probes, reverse proxies and `http://localhost:8000` are untouched;
behind the relay Flask sees the request as plain HTTP. A reverse proxy that
terminates TLS (Traefik, an ingress with a Let's Encrypt certificate) keeps
speaking plain HTTP to port 8000 as before: its connections start with a
request line, so they are passed through and the built-in certificate never
enters the picture, and the page, served on the proxy's HTTPS, records
directly. A proxy that serves plain HTTP cannot expose the same-port HTTPS
behind it; there HTTPS belongs on the proxy.

Two things make this hold outside the developer's machine. Gunicorn reads
`./gunicorn.conf.py` only when started from `/app`, so the image also sets
`GUNICORN_CMD_ARGS="--config /app/gunicorn.conf.py"`; without the hook a TLS
connection reaching a plain gunicorn hangs until its timeout, which is what a
record button that "does nothing" looked like. The certificate needs the
`cryptography` package (now a pinned requirement; the `openssl` binary is the
fallback) and a writable `FLASK_HTTPS_CERT_DIR`; when that directory cannot
be written the certificate goes to the temp directory with a warning, so
HTTPS still runs and only the browser warning returns after a restart.

The record button never fails silently. Every failure of the record flow,
from a missing API to a refused permission or a recorder that delivered
nothing, lands in a red box with the exception name and message, and when
HTTPS is not running the button raises the server's reason (the hook that
never ran, the certificate that could not be created) instead of doing
nothing. The notice lists the alternatives: uploading a clip recorded with
the phone (a video is fine, its sound track is used), Chrome's
`chrome://flags/#unsafely-treat-insecure-origin-as-secure` for that one
address, a reverse proxy, or `http://localhost:8000` on the server itself.

### 17.4. Environment Variable Configuration

- `NEURAL_FINGERPRINT_ENABLED` (false): the master switch, like `CLAP_ENABLED`,
  off by default because the stage costs 10 to 25 s of CPU per track; an
  installation turns it on from the Machine Learning Models switches of the
  setup wizard, and a choice saved there survives upgrades because it lives in
  `app_config` and wins over the default. Like every wizard parameter, the
  flag is written to `app_config` on the first web start that lacks it (with
  the environment value or the config default) and read from there
  afterwards, so a changed default never flips an installation. A library
  that already holds neural fingerprints counts as having chosen this one on:
  config infers the flag at import and that first write stores it, so an
  installation that used the feature keeps it.
  False skips the fingerprint stage of the analysis and the index build, the
  web process neither loads nor reloads the index, the Search by Recording
  entry leaves the menu the way Text Search and Lyrics Search do with their
  flags (the page itself, opened by its address, says the feature is
  disabled), and the three API routes answer 503.
- `NEURAL_FINGERPRINT_MODEL_PATH` (`/app/model/neural_fingerprint.onnx`,
  downloaded from the model release like the MusiCNN graphs; the native
  builds point it at their bundled model directory): the fingerprint encoder;
  a missing file disables the analysis stage and the tab.
- `NEURAL_FINGERPRINT_CODEBOOK_PATH` (`/app/model/neural_fingerprint_pq.npz`,
  next to the model): the 32-byte codebook every stored fingerprint is encoded with; keep
  the one the library was analysed with, a different file makes the stored
  blobs unreadable.
- `NEURAL_FINGERPRINT_NPROBE` (12): cells read per query vector.
- `NEURAL_FINGERPRINT_TRAIN_ROWS` (200000): rows the k-means that places the
  cells is trained on, sampled 100 per track from random tracks, never fewer
  than 20 per cell.
- `NEURAL_FINGERPRINT_RETRAIN_GROWTH` (4): the worker appends new tracks to
  the existing cells until the library has grown this many times since the
  centroids were trained, then rebuilds from scratch.
- `NEURAL_FINGERPRINT_MIN_SCORE` and `NEURAL_FINGERPRINT_MIN_LEAD`: the mean
  cosine the best track must reach at its alignment, and its lead over the
  next track, to count as identified.
- `NEURAL_FINGERPRINT_INDEX_STRIDE` (1): index every n-th stored half-second
  row; 2 halves the local pack and the query work and must be measured on
  real recordings first, since it costs recall on degraded clips. Changing it
  triggers a full rebuild.
- `NEURAL_FINGERPRINT_QUERY_THREADS` (0 = one per core, at most 8): threads
  scoring a clip's segments in parallel in the web process.
- `NEURAL_FINGERPRINT_CACHE_MB` (1024): RAM the web process keeps for the
  cells read from `ivf_cell` on demand; least recently used cells are dropped
  past it, and the whole cache goes when the recording search has been idle
  for `RECORDING_SEARCH_WARMUP_DURATION` seconds.
- `FLASK_BUILTIN_HTTPS` (true): answer HTTPS on the HTTP port; false switches
  the relay off and every connection passes through untouched.
- `FLASK_HTTPS_CERT_DIR` (data dir `tls/`, `/app/tls` in containers): where
  the self-signed certificate and key are kept, so the browser exception
  survives restarts.

- `RECORDING_SEARCH_DEFAULT_N_RESULTS` (100): results when the caller sends no
  count, and the value the page's count box starts on.
- `RECORDING_SEARCH_RECORD_SECONDS` (20): browser recording length.
- `RECORDING_SEARCH_MAX_CLIP_SECONDS` (60): longer uploads are cut to this.
- `RECORDING_SEARCH_MAX_UPLOAD_MB` (1024): upload ceiling.
- `RECORDING_SEARCH_TARGET_LEVEL_DB` (-14): RMS level the clip is normalised to.
- `RECORDING_SEARCH_WARMUP_DURATION` (300): idle seconds before the neural
  fingerprint pack and its encoder session unload from the web process.
- `DUPLICATE_DISTANCE_THRESHOLD_NEURAL_FINGERPRINT` (0.25): Search by Song
  leaves out a song whose whole-song fingerprint is closer than this cosine
  distance to the chosen song or a kept result, when a 20 s alignment
  confirms it; 0 turns it off.
- `MAX_SONGS_PER_ARTIST` (3): the shared per-artist cap, also applied to the
  Search by Song results; 0 turns it off.

---

## 18. Album Creation

Album Creation turns one seed into a CD-format album: a short, ordered track list
that behaves like a real album instead of a list of nearest neighbours.

### 18.1. Functional Analysis (High-Level)

1. The user opens **Album Creation** (the entry under Artist Similarity) and
   picks the kind of seed: a **song**, chosen from the search box for the
   selected music server, or a **description** of a few words such as "jazz with
   trumpet". The description box offers example queries and the same concept
   refinement the DCLAP search page has, so an instrument or a voice can be
   asked for explicitly.
2. **Create Album Proposal** returns 12 tracks (about 48 minutes) in running
   order, with the role of the key slots (Opener, the two Singles, Closer), the
   album length, the number of artists and the cohesion reached. The tracks are
   chosen with both the MusiCNN and the DCLAP analysis of every song, so the
   album agrees on its instruments and voices as well as on its genre.
3. A song seed stays in the album, in the slot that fits it. A description is
   turned into a point in the DCLAP audio space, and every genre or instrument
   it names is then checked against the candidates by the index that knows it.
4. No artist gets more than `MAX_SONGS_PER_ARTIST` tracks. Only one version of a
   song gets in. Live,
   demo, remix and skit tracks never do, and neither does an alternate rendition
   named in a title suffix (acoustic, instrumental, extended mix, session,
   outtake); a remaster, a mono, single or radio cut is the song itself and
   stays. Holiday songs stay out unless it is December or the seed itself is a
   holiday song.
5. The album stays in the seed's world: tracks within 15 years of the seed, of
   album length (100 seconds to 10 minutes), with a tagged artist and with
   lyrics about what the seed is about are preferred while enough of them
   remain, so a small or untagged library still gets its album. A description
   has no year of its own, so its era comes from the median year of its best
   matches. A sung album takes at most two instrumentals, and an instrumental
   album at most two sung tracks.
6. **Create Playlist on Media Server** sends the album, in order, to the selected
   server through the same route every other page uses.
7. The scheduled **Album of the Week** creates one album per music server from a
   random analysed song and writes it to one fixed playlist name, which every run
   cleans and refills.

The page is PER SERVER and uses the lyric themes, so the menu entry, the page and
its API are off while `LYRICS_ENABLED` is off. The Album of the Week schedule is
not behind that flag or any other: like every other schedule it is switched on
and off only from **Administration > Scheduled Tasks**, and without lyric themes
the album is sequenced on audio alone.

### 18.2. Technical Analysis (Algorithm-Level)

All of it lives in `tasks/album_creation_manager.py`; `app_album_creation.py`
only parses the request and scopes the answer to a server.

1. **Why not plain similar songs.** Measured on 12,668 real studio albums, the
   mean pairwise cosine of an album's MusiCNN embeddings is 0.80 (0.67 to 0.89
   between the 10th and the 90th percentile). The top 12 neighbours of a song sit
   at 0.95: more alike than any real album.
2. **Two indexes, one space.** MusiCNN and DCLAP disagree usefully: over 20,000
   random pairs their cosines correlate at 0.72 only. Measured against real
   albums, DCLAP is the better judge of what belongs together (given one track
   of a real album, 26.7% of the album's other tracks are in its top 100 against
   MusiCNN's 17.3%) and it hears a voice far better (0.97 AUC against 0.91 when
   telling sung tracks from instrumental ones), while MusiCNN keeps the genre
   tighter. Both vectors are therefore scaled by the square root of their share
   (`ALBUM_CREATION_MUSICNN_SHARE`, half each) and joined into one vector, so a
   cosine in that space is exactly the weighted average of the two cosines and
   every later step runs on one space. A library analysed before DCLAP, or a
   share of 1.0, builds the album on MusiCNN alone.
3. **A description, word by word.** The words are matched against the 50
   analysis tags and the DCLAP concept dictionary. A genre word is scored from
   the analysis tags, which owe nothing to DCLAP and so cannot be diluted by it;
   an instrument word from the DCLAP concepts, validated against tracks that
   name the instrument in their own title (piano 0.91, cello 0.89, trumpet 0.83,
   choir 0.83 AUC). Each named word also queries the DCLAP index on its own,
   because the text point for "rock with piano" lands among pianos and the pool
   would otherwise hold no rock at all. The attributes then take turns narrowing
   the candidates, in `ALBUM_CREATION` terms `ATTRIBUTE_PASSES` rounds, because
   whichever one narrows first otherwise wins outright: instrument first scored
   instrument 100 and genre 33, genre first the reverse, taking turns 99 and 95.
   Concepts picked in the page steer the query point as well.
4. **Candidate pool.** The songs come from the similar-song engine
   (`find_nearest_neighbors_by_vector`, without its per-artist cap), so they are
   limited to the selected server and its duplicate removal has already run. The
   first query takes 150 neighbours of the seed vector; one hop follows,
   querying 100 neighbours around the 4 tracks farthest from the seed. The DCLAP
   index adds 150 more neighbours of the seed's DCLAP vector, filtered through
   the same per-server availability rule, because that index answers to whoever
   queries it and does not scope itself. On real albums the MusiCNN pool held
   2.4 of the album's own other tracks and the two indexes together hold 3.8.
   When the DCLAP index is not loaded the pool is the MusiCNN one.
5. **The target is calibrated, not copied.** A real album is one artist. The
   same 0.80 between DIFFERENT artists is a change of genre: at that target a
   folk seed gave indie rock, a singer-songwriter seed ambient piano and an
   80s pop seed 2020s hip-hop. The target is therefore calibrated against real
   albums on
   the DCLAP concept model, which names instruments and voices: at 0.86 in the
   mixed space a created album holds its instruments together as tightly as a
   real album (0.79 against 0.79) and its voices as tightly (0.82 against 0.81),
   while 0.90 in the same space overshoots both. On MusiCNN alone the same point
   is 0.90.
6. **Selection.** A guided greedy picks the tracks: every step samples up to 48
   allowed candidates and keeps the one that brings the mean pairwise cosine
   closest to the target. A candidate below cosine 0.75 from any chosen track is
   only used when nothing else is left, so the album makes sense as a whole and
   not only on average.
7. **Preferences and the other voice.** Tagged artists, album length and the seed's era (15 years either way,
   which halves the year span at no cost in coherence) are applied in turn, each
   only while three albums' worth of candidates remain. Then what the songs are
   ABOUT: of the candidates that have lyrics, only the
   `ALBUM_CREATION_LYRIC_SHARE` nearest the seed in the lyrics embedding stay.
   Real albums are only slightly tighter in their lyrics than the same artist's
   other songs (they beat a random set of songs 98% of the time, their own
   artist's other songs only 59%), so this is a preference and never a rule: a
   track without lyrics is never judged by it. Measured on 120 seeds it lifts
   lyric cohesion from 0.16 to 0.21, where real albums sit at 0.22 and random
   songs at 0.13, and leaves every audio measure unchanged. Giving the lyrics a
   share of the mixed vector instead was tried and made the album worse on every
   count, because the target had to be loosened to make room for them. At most two tracks may
   be sung in an instrumental album or the other way round. Sung or instrumental
   is a vote of each track's 10 nearest pool neighbours in the DCLAP space,
   never its own lyrics flag: 4 in 10 tracks without lyrics are sung songs whose
   lyrics are missing, and the raw flag built albums out of exactly those.
8. **Intensity.** The arc is driven by one composite: 60% the mood scores
   (party, aggressive and danceable minus relaxed and sad), 30% energy and 10%
   tempo. The mood scores are centred per song first, because the raw scores
   share one factor that follows the release year. The tempo is folded into one
   octave (70 to 140 BPM), because the detector often doubles it.
9. **Sequencing.** The roles follow what real albums measure. The closer is the
   calmest, longest, least typical track, often without lyrics or with inward
   ones. Slots two and three take the most intense and most typical tracks. The
   opener is a short quiet intro when the dominant genre is hip-hop, R&B, metal
   or electronic, and a strong track otherwise. The middle declines in intensity
   and its lyrics turn inward. Two calm tracks are never adjacent while a gap is
   free, none sits next to a calm closer, and two tracks of one artist are pulled
   apart whenever another track can take the place.
10. **Album of the Week.** `run_album_of_the_week_task` samples random analysed
   songs available on that server and builds the album from the first one that
   works. Everything around that choice is
   `tasks.task_run.run_playlist_task_per_server`, the scaffold it shares with the
   sonic fingerprint: the per-server loop, cancel check, heartbeat and reporter,
   the `create_or_replace_playlist` call under `ALBUM_OF_THE_WEEK_PLAYLIST_NAME`,
   the dated fallback playlist on a backend without upsert, and the rule that an
   empty result keeps the previous playlist. It fails only when every server
   failed.

### 18.3. Environment Variable Configuration

- `ALBUM_CREATION_TRACKS`: tracks in a created album (default `12`).
- The pool starts at `POOL_FIRST_QUERY` and widens to `POOL_MAX_QUERY` (2000 to
  4000 for a description) until three albums' worth of candidates survive.
- `ALBUM_CREATION_MUSICNN_SHARE`: share of MusiCNN in the space the album is
  measured in, the rest being DCLAP (default `0.5`; `1.0` turns DCLAP off).
- `ALBUM_CREATION_LYRIC_SHARE`: share of the candidates kept by what their
  lyrics are about (default `0.25`; `1.0` turns it off).
- `ALBUM_CREATION_COHESION`: target mean pairwise cosine of the album in that
  space (default `0.86`, the calibrated value; lower is more eclectic and 0.80
  already changes genre).
- `ALBUM_OF_THE_WEEK_PLAYLIST_NAME`: the fixed playlist the scheduled run cleans
  and refills.
- `MAX_SONGS_PER_ARTIST`: the shared per-artist cap.
- `LYRICS_ENABLED`: must be on for the page, its API and its menu entry. It does
  not touch the schedule.

---

## 19. Smart Fade

Smart fade plays a list of songs so that each one flows into the next: the
songs are put in an order where the end of one sounds like the start of the
next, and every hand-over is mixed on the beat when the two songs allow it,
the way a DJ would. It is an option of the shared song player (section 17.1,
step 6), so it works on every page that answers a list of songs, and it needs
the neural fingerprint.

### 19.1. Functional Analysis (High-Level)

1. **Smart fade**, ticked in the search form before the search, reorders the
   results so each song flows into the next. The first song stays first (on a
   page with a seed, the seed plays first), and every row shows a small bar
   with where the song will start and end (`m:ss - m:ss`). A song path and
   Album Creation keep their running order and only get the times. A playlist
   made from the results keeps the order; the times are only used by the
   player. Smart fade and Demo exclude each other.
2. **Play** plays each song from its start time to its end time (the first
   one from the beginning, the last one to its end) and mixes it into the
   next with one of the seven transitions below. Every song plays at the same
   loudness (-14 LUFS; a quiet song is raised only as far as its peaks allow),
   and in every transition the two volume curves follow both songs' own
   levels, so a song already fading on its own or a quiet intro leaves no
   hole in the middle.
   The line under the player names the transition while it plays.
3. **One highlighted row, the louder song.** During a transition the
   highlight and the clock stay on the ending song until the middle of the mix
   or crossfade, then they jump to the starting one.
4. **Transition icons.** The page starts planning the next ten transitions
   from the current or selected song, with two server requests at a time.
   Each confirmed plan adds an icon and duration to its incoming row. Planned
   results are reused, and jumping to a different song replaces the pending
   queue. Scrubbing uses the audio stream independently of the plan requests.
5. **Skip to the fade** jumps to 6 seconds before the next transition.
   Pressed while the next song or its plan is still loading, it waits for them
   ("Preparing the mix..."), because a plan that arrives after its mix point
   can only be played as the default crossfade on the order times. A click or a drag on the playing
   row moves in the song, never closer than one second to the line where its
   fade starts, so the planned transition always happens.

**The seven transitions**

The three beat-matched mixes play only where the server is sure the beats of
both songs line up (a confirmed beat lock, or for the percussion mix two
drum-led parts whose hits line up). A pair without a beat lock keeps both
songs at their own speed and gets the transition its two parts allow: an EQ
blend when one of them has no drums (nothing can clash), a filter rise when
the next song comes in louder at a related tempo, a vocal handoff when two
voices overlap, else the crossfade. Each type has its own row icon and count.

On 80 Similar Song lists (10 seed songs from each of 8 genres, 10 songs
each, 798 transitions answered) they came out as 333 EQ blends, 268
crossfades, 76 beat mixes, 59 filter rises, 54 vocal handoffs and 3
percussion mixes; with only the beat-matched kinds and the crossfade, 40 of
those lists had played 337 crossfades of 387. The crossfades follow the
music: they are what is left where both songs have drums at tempos too far
apart to lock, with no voice over a voice, at similar loudness or with a
louder next song at an unrelated tempo.

| Genre | Crossfade | Most frequent other kinds |
|---|---|---|
| Folk | 2% | EQ blend 92% |
| Jazz | 6% | EQ blend 89% |
| Electronic | 25% | EQ blend 53%, beat mix 14% |
| Soul | 31% | EQ blend 37%, vocal handoff 14% |
| Pop | 34% | EQ blend 36%, vocal handoff 17% |
| Hip-hop | 43% | filter rise 23%, vocal handoff 19% |
| Rock | 60% | EQ blend 20%, beat mix 14% |
| Metal | 71% | filter rise 15%, beat mix 12% |

1. **Beat-matched mix with a bass swap** (`beat_mix`)
   - Label: `Beat-matched mix with a bass swap into <next song> (X to Y BPM)`
   - When: a confirmed beat lock: the tempos are within 3.5% (or half or
     double time), so the two songs end up at most 60 cents apart in pitch,
     the two songs' drum hits line up within 30 ms, and on both songs
     several beat trackers agree on where the beats are (section 19.3).
   - How it sounds: both songs play together for the whole bars nearest
     the pair's mix length (24 to 32 felt beats, 10 to 21 seconds, see "How
     long a transition lasts" below), the last bars of the span the beat
     check covered (section 19.3), so the ending song still stops on its beat nearest the fade
     point and the next song still enters on its cue. The tempo glides beat by beat
     from the first song's to the next song's. Mixed the way a DJ mixes with a
     3-band EQ, every move a smooth S-curve, one band after the other: the
     next song's highs come in first, then its mids, then the two bass lines
     swap in one move over 2 beats, on the bar nearest the middle where the
     next song's own bass is playing; after the swap the ending song's mids go
     out, its highs last. The faders only fade the two ends in and out.
2. **Beat-matched rise** (`beat_rise`)
   - Label: `Beat-matched rise into <next song> (X to Y BPM)`
   - When: the same confirmed lock, and the next song is at least 6 dB louder
     over the central 60% of the mix, measured after both whole-song gains
     with K-weighted power.
   - How it sounds: the same length and tempo glide as the blend. The next
     song comes in from the treble down (a high-pass opens from 800 Hz to
     10 Hz until a beat after the swap, so its bass arrives last); the ending
     song goes out as in the blend.
3. **Beat-matched percussion mix** (`percussion_mix`)
   - Label: `Beat-matched percussion mix into <next song> (X to Y BPM)`
   - When: both mixed parts are drum-led (above 150 Hz the percussive energy
     is at least 1.5 times the harmonic one: drum intros and outros, breaks;
     the bass lines are left out because they always swap one at a time,
     and with them house intros measured 0.07 to 0.25 against 1.4 to 61
     without, rock parts 0.11 to 0.37 either way) and their hits
     line up within 30 ms (a steady offset under 90 ms repaired), with no
     voice of one song over a voice of the other there. Either a confirmed
     beat lock whose two parts are drum-led, or, without a full lock (the
     tempos 60 to 90 cents apart, drums carry little pitch), the last 8 to 12
     seconds before the fade point.
   - How it sounds: whole bars, the tempo gliding as in the blend (the beat
     mix's length on a full lock, 8 to 12 seconds without). The faders cross equal-power;
     the two kick drums swap in one move around the middle (one bass drum at
     a time), the hi-hats cross with them; a 3-band hand-over has nothing to
     hand over in drums alone.
4. **EQ blend** (`eq_blend`)
   - Label: `EQ blend without drums to clash into <next song> (X and Y BPM)`
   - When: no beat lock, and in the mix one of the two parts has no drums
     (above 150 Hz the percussive energy at most 0.12 times the harmonic
     one: ballads, acoustic songs, beatless intros, outros and breakdowns).
     Two tempos only clash where both songs play a beat; a part without
     drums can be laid over anything (DJs mix beatless sections at any
     tempo).
   - How it sounds: the crossfade's length, points and equal-power faders,
     both songs at their own speed, with the bands handed over in order: the
     two bass lines swap in one move at the middle (one bass at a time), the
     next song's highs and mids open from -12 dB over the first half, the
     ending song's mids and highs close to -12 dB over the second half. The
     bass swaps only when both songs have a bass line (below 120 Hz, over
     the middle of the mix, the weaker within 12 dB of the stronger); when
     one has none, the only bass fades with its fader as in the crossfade,
     because taking it out at the middle with nothing to replace it left the
     mix without bass there (12 of 40 rendered genre-list blends had one
     bass line: holes deeper than 3 dB went from 3 to 1, both songs audible
     together from 2.8 to 3.7 seconds). The beat mix's deeper cuts (mids at -27 dB
     until the swap) left the next song nearly silent when it had no drums
     to carry it: 17 of 113 rendered blends dipped more than 3 dB under the
     level line with them.
5. **Filter rise** (`filter_rise`)
   - Label: `Filter rise into <next song> (X and Y BPM)`
   - When: no beat lock, both parts have drums, the next song comes in
     louder than the ending one goes out over the middle of the mix (both at
     their playing loudness), and the two tempos are related: within 200
     cents (12%) of the same tempo, half or double time, 3:2 or 4:3 (a beat
     tracker reads some songs at two thirds or three quarters of their
     tempo). DJs bring a song of another tempo in through a high-pass; a
     quieter next song stays a crossfade (the filter would hide it). The
     next song's hats and snare play over the ending song's beat until the
     middle, so two unrelated pulses clash there: on a rock list a rise of
     166 into 96 BPM (1.73 times, about 250 cents from every simple ratio)
     sounded bad by ear while 97 into 108 and 93 into 123 BPM were
     acceptable. Such a pair gets the vocal handoff or the crossfade, the kind
     that sounded acceptable between unrelated tempos in the same list. Of the
     69 genre-list rises, 59 keep it (8 unrelated, 2 without measured tempos).
   - How it sounds: both songs at their own speed, equal-power faders; the
     next song comes in through a high-pass that opens from 800 Hz down to
     the bass shelf's 180 Hz by a beat before the middle and waits there,
     then opens to 10 Hz over the 2 beats in which the ending song's bass
     goes out in one move: one kick and one bass at a time. Opening it
     straight to 10 Hz by a beat after the middle let the next song's kick
     and bass in about 2 seconds before the ending song's bass left: on the
     69 genre-list rises two bass lines played together for 1.1 seconds
     (median) and two kicks for 0.9, against 0.3 and 0.3 with the held
     filter, with the same 8 dips deeper than 3 dB and no clipping.
6. **Vocal handoff** (`vocal_handoff`)
   - Label: `Vocal handoff into <next song> (X and Y BPM)`
   - When: no beat lock, both parts have drums at similar levels, and the
     voice check finds the two songs' voices overlapping for at least 30% of
     the mix.
   - How it sounds: the crossfade, with the mids (where voices sit) moving
     from one singer to the other around the middle (up to 9 dB) and a late
     extra cut of the old singer, so the two voices are never at full level
     together.
7. **Default crossfade** (`crossfade`)
   - Label: `Default crossfade into <next song> (X and Y BPM)`, or just
     `Default crossfade into <next song>` when there is no plan and so no
     measured tempos; the line then switches to
     `Playing <next song> from m:ss until m:ss`.
   - When: no beat lock, both songs have drums at similar levels and no
     voices overlap; or no usable plan in the browser by its mix point.
   - How it sounds: a volume crossfade of 13 to 18 felt beats, 5 to 10
     seconds (see "How long a transition lasts" below), one song going
     down while the next comes up, nothing stretched; each song keeps its own
     tempo. The two curves start equal-power and follow both songs' levels,
     never sinking more than 4 dB under the louder song (section 19.3). When
     voices of both songs overlap, a gentle mid/high exchange moves the two
     singers apart; otherwise no filters.
   - Where: with a plan it starts on a beat of the ending song and the next
     song comes in on its section start, the points the plan found, over the
     fewest whole beats of the ending song that reach the pair's crossfade
     length. Without a plan it runs for 7.5 seconds between the two times
     of the order endpoint (section 19.2). The EQ blend, the filter rise and
     the vocal handoff have the same length and points.

**Switching transitions off**

The setup wizard's Smart Fade panel (at the bottom of the page, closed until
opened) has one switch per transition, all on by default; the default
crossfade has none, it is always on, and with the neural fingerprint off the
whole panel is greyed out. The switches are ordinary `app_config` rows read
by the server for every plan, so they apply to every listener of the
installation; the endpoints' inputs and answers stay the same and the page
plays whatever type comes back. A transition switched off gives its pair the
next one that fits, never a relabelled one:

- A confirmed beat lock tries the percussion mix, then the beat rise (only
  where the next song is 6 dB louder), then the beat mix. With none of them
  on, the pair is planned as a pair without a lock, at its own speed, with
  the crossfade's length and its fuller start, so the beat mix's 10 to 21
  seconds never play as a long crossfade. With the beat rise off a louder
  next song gets the beat mix; with the beat mix off a lock plays only where
  the percussion mix or the beat rise fits, and a locked percussion mix that
  meets two voices (which plays the beat mix when it is on) is planned again
  as a pair without a lock. With all three off the beat
  committee and the onset checks are skipped, but the crossfade's level
  scan costs about as much (on 20 real pairs the planning median was 4.10
  seconds with them on and 4.16 off for the locked pairs); the beat grid,
  the lengths and the voice check still run, because the crossfade starts
  on a beat and the voice check moves its cues.
- Without a lock the order is EQ blend, filter rise, vocal handoff, default
  crossfade, each tried only when switched on (with the EQ blend off, a pair
  where one part has no drums can get the filter rise). With the vocal
  handoff off, the default crossfade stays a volume crossfade even where
  voices overlap; the beat-matched kinds, the EQ blend and the filter rise
  keep their voice exchange.

Under the switches, "Test the transitions" takes two songs from the usual
search boxes and lists them; Play in the shared player plays the first, then
the second through the transition the switches choose as they stand on the
page, saved or not. The player sends them with each transition request
(`transitions`, section 19.2), so nothing is stored and no other listener
hears them; changing a switch plans the pair again. Skip to the fade jumps
to the transition. The box needs a saved configuration with the neural
fingerprint on.

**How long a transition lasts**

Every transition is counted in beats, so its length in seconds follows the
two songs: a fast pair mixes in fewer seconds than a slow one, and a dense
pair in fewer beats than a sparse one.

- The pulse. The felt beat of each part is the beat tracker's, except that a
  song read under 100 BPM with 3.9 or more onsets per beat is felt at
  double tempo (a fast song read at half speed: on a metal list the tracker
  read 71 to 86 BPM where the drums played 140 to 170, with 3.9 to 6.9
  onsets per tracked beat, against 2.0 to 3.8 on slow pop and rock songs).
  The pair's beat is the mean of the two felt beats. Listeners rate the
  speed of music from its tempo and its event density together, not the
  tempo alone (JASA 2010, ISMIR 2013), and DJs count transitions in beats
  (real DJ transitions peak at multiples of 32 beats, ISMIR 2020).
- The density. Above 150 Hz, the share of percussive against sustained
  sound in the 15 seconds of each part next to its cue: 0.55 or more is
  sparse, 0.30 or less dense, linear between; a part without drums (0.12
  or less) is sparse; the pair is as dense as its sparser part, because a
  sparse part leaves room for the other (an auto-DJ lines intros up with
  outros for the same reason). On the lists heard by ear, the pop cues
  measured a median of 0.43 (none under 0.27) and the Italian rock cues
  0.21 (none over 0.35): sustained guitars, keys and voices are what pile
  up in a long overlap of two full bands. Long crossfades blur sounds with
  many impacts (arXiv 2024), and an automatic pop DJ shortens an overlap
  where voices collide (arXiv 2026). The key fit of the two songs does
  not decide the length: the pop neighbours matched in key the least
  (median chroma correlation 0.00 against 0.54 on rock).
- The count. A beat mix plays 32 felt beats for a sparse pair down to 24
  for a dense one, kept within 10 to 21 seconds (whole bars of the ending
  song, never past the span the beat check covered); a crossfade, EQ blend,
  filter rise or vocal handoff plays 18 felt beats down to 13, kept within
  5 to 10 seconds. The percussion mix without a full lock keeps its 8 to 12
  seconds (its parts are drum-led, so sparse). The voice check's cap of 14
  seconds on mixes where two voices overlap most of the time still applies.

Fixed lengths were tried first. 15 to 21 second beat mixes with 10 second
crossfades sounded right by ear on a pop Similar Song list and too long on an
Italian rock list (too many instruments at once); about 10 second mixes with
5 second crossfades sounded too abrupt on both, and right on a fast metal
list. Replayed on the same lists (k3s), the counted beats gave:

| List | Beat mixes | Crossfade kinds |
|---|---|---|
| Pop (about 104 BPM) | 13.8 s (all capped by the voice check, as before) | vocal handoffs 10.4 s for the 104 BPM pairs, 5.2 s for 150 to 160 BPM |
| Italian rock (dense) | 12.6 s median | 7.1 s median |
| Metal | 11.4 and 16.2 s | 5.6 to 7.5 s at 140 to 170 BPM, up to 9 s at 102 to 106 BPM |

### 19.2. The Two Endpoints

Both endpoints answer numbers only. The page fetches each song from the media
server (`GET /api/recording_search/play/<id>`) and plays every cut, speed and
fade itself in the browser; another frontend can call the two endpoints the
same way. With the neural fingerprint off both answer 503.

1. **Order**: `POST /api/recording_search/smart_fade`
   - Called once, right after the search.
   - Takes `item_ids`, the songs as ids of the selected server, the first one
     (the seed) first. `keep_order: true` keeps the order sent (a path, an
     album) and only adds the times.
   - Returns the same songs in smart fade order, each with
     `fade_in_seconds` (where it starts: within its first 30 seconds, 0 for the
     first song), `fade_out_seconds` (where it ends: within its last 30
     seconds; the last song plays to its end) and `duration_seconds`. A song
     without a neural fingerprint comes last with `null` times (with
     `keep_order` it keeps its place).

   ```json
   {"count": 3, "results": [
     {"item_id": "id-A", "fade_in_seconds": 0.0,  "fade_out_seconds": 208.5, "duration_seconds": 231.0},
     {"item_id": "id-C", "fade_in_seconds": 20.0, "fade_out_seconds": 220.5, "duration_seconds": 242.3},
     {"item_id": "id-B", "fade_in_seconds": 18.0, "fade_out_seconds": 256.0, "duration_seconds": 256.0}]}
   ```

2. **Transition**:
   `GET /api/recording_search/transition?from_id=...&to_id=...&out=...&in=...`
   - Called once per pair while the current song plays, with the inputs it
     always had: `from_id` (the current song), `to_id` (the next one), `out`
     (the current song's `fade_out_seconds`) and `in` (the next song's
     `fade_in_seconds`).
   - Optional `transitions`: the kinds switched on for this one request, a
     comma list of the type names below (`crossfade` is always on; empty
     means only the crossfade), as the setup wizard's test sends its unsaved
     switches. Left out, the saved switches apply.
   - Returns the transition timing and curves for the player.
     - `type`: `beat_mix`, `beat_rise`, `percussion_mix`, `eq_blend`,
       `filter_rise`, `vocal_handoff` or `crossfade` (section 19.1).
     - `start`: the second of the current song where the transition starts.
     - `duration`: how many seconds the player schedules for the transition.
     - `middle`: seconds after `start` where the next song takes over, in a
       beat mix the bass swap (the page moves the highlight there).
     - `step`: seconds between two points of every list below (about 0.1).
       The lists are the curves measured for these two songs; their points
       are close enough that joining them with straight lines (what Web
       Audio's `setValueCurveAtTime` does) plays the curve.
     - `filters`: `high_hz` (high shelf, 4000), `mid_hz` and `mid_q` (mid
       peak, 1000 and 0.5) and `bass_hz` (low shelf, 180), the filters the
       `high_db`, `mid_db` and `bass_db` lists drive.
     - `from` (the `from_id` song) and `to` (the `to_id` song):
       - `gain`: the multiplier for the whole song, every second of it
         (-14 LUFS; a quiet song is raised only until its peak reaches
         -1 dBFS). A song gets the same gain as `from` and as `to`.
       - `speed`: the playback rate at each point (1 = normal); play the
         song's own audio at it (Web Audio `playbackRate`), `from` from
         `start` and `to` from its own `start`; 1 all along in a crossfade.
       - `volume`: the fader, a multiplier from 1 to 0 for `from` and from 0
         to 1 for `to`; in a beat mix or rise it only fades the two ends, in a
         percussion mix and a crossfade it carries the whole exchange. It
         also carries the steady-level correction and the peak protection.
       - `high_db`, `mid_db`, `bass_db`: the filter gains in dB, all 0 for a
         crossfade without vocal ducking.
       - `to` only: `start` (the second of the next song where it comes
         in), `highpass_hz` (a `beat_rise` or `filter_rise` sweeps a 2nd-order
         Butterworth high-pass instead of the low shelf; else `null`) and
         `continue_from` (the second where the next song plays on alone
         once the transition ends).
     - `info`, not needed to play: `tempo_from` and `tempo_to` (BPM,
       `null` without a steady beat), `confidence` (`high` when the beats
       lock, `low` for a crossfade from a song still at full level into one
       that comes in quieter, else `medium`), `loudness_from` and
       `loudness_to` (whole songs, LUFS).
   - A pair without a steady beat, or without room for a beat mix, still
     gets a `crossfade` at the two given points: `duration` = the pair's
     crossfade length when the beats were measured (no room for a beat mix),
     7.5 seconds without a beat (less when a song is shorter), `start` =
     `out` - `duration`, `to` from `in`, tempos `null`. Every list has one
     value per point.

   ```
   {"type": "beat_mix", "start": 147.3626, "duration": 16.0875, "middle": 8.0216, "step": 0.099922,
    "filters": {"high_hz": 4000.0, "mid_hz": 1000.0, "mid_q": 0.5, "bass_hz": 180.0},
    "from": {"gain": 0.4728, "position": [147.3626, 147.4625, ...], "volume": [1.0, 1.0009, ...],
             "high_db": [0.0, 0.0, ...], "mid_db": [0.0, 0.0, ...], "bass_db": [0.0, 0.0, ...]},
    "to": {"gain": 0.5445, "position": [23.6997, 23.8007, ...], "volume": [0.0, 0.0137, ...],
           "high_db": [-40.0, -40.0, ...], "mid_db": [-27.0, -27.0, ...], "bass_db": [-40.0, -40.0, ...],
           "highpass_hz": null, "continue_from": 39.8759},
    "info": {"tempo_from": 90.0, "tempo_to": 89.0, "confidence": "high",
             "loudness_from": -7.49, "loudness_to": -8.72}}
   ```

   - Errors: `400` "Give from_id, to_id, out and in (seconds)." (bad or
     missing parameters), `400` "transitions is a comma list of: ..." (an
     unknown name in `transitions`), `400` "These two songs have no
     room for a crossfade at these points.", `404` "This song is not in the
     library.", `502` "The media server did not deliver this song." and
     `500` "The transition could not be made. Check the container logs."
     (an unexpected error, such as a file it cannot decode).

**From the answer to the transition**

| Answer | Transition played |
|---|---|
| `type: "beat_mix"` | 1. Beat-matched mix with a bass swap |
| `type: "beat_rise"` | 2. Beat-matched rise |
| `type: "percussion_mix"` | 3. Beat-matched percussion mix |
| `type: "eq_blend"` | 4. EQ blend |
| `type: "filter_rise"` | 5. Filter rise |
| `type: "vocal_handoff"` | 6. Vocal handoff |
| `type: "crossfade"` | 7. Default crossfade, on the answer's points |
| an error, no answer, or an answer after its `start` | 7. Default crossfade, on the order times |

Without an answer the page uses an equal-power fade when both excerpts are
buffered. Channel peaks at playback gain bound the faders. Partial buffers
use complementary S-curves. The current song fades out over the 7.5 seconds before
its `fade_out_seconds`, the next song starts from its `fade_in_seconds` at
its own gain of 1, and the highlight jumps after 3.75 seconds. With the order
above, `id-A` fades out from 3:21.0 to 3:28.5 while `id-C` starts from 0:20.0
and fades in until 0:27.5, then plays on alone. A song with `null` times gets
no transition: when the next song has none, the crossfade still happens with
the next song from its beginning; when the current song has none, it plays to
the end of its file and the next song starts right after.

### 19.3. Technical Analysis (Algorithm-Level)

1. **Order** (`tasks/recording_search_manager.py`, `smart_fade_playlist`). It
   reuses the stored neural fingerprint matrices, one 128-dimension vector per
   half second. Only the first and the last 30 seconds of each song are
   decoded and compared (half of a song shorter than a minute), averaged into
   overlapping 4-second windows, so a song never starts later than 30 seconds
   in nor ends earlier than 30 seconds before its end. Every ordered pair of
   songs gets a score: the highest cosine between one of the first song's
   fade-out windows and one of the second song's fade-in windows (that pair
   of windows gives where the first ends and the second starts), minus the
   tempo penalty below, minus 0.04 per standard deviation of intensity
   between the two songs (Album Creation's intensity: centred mood scores,
   energy and folded tempo, z-scored within the list), plus 0.1 times how much
   more alike their MusiCNN embeddings are than the list's average pair. A
   score under the median pair of the list counts twice as much. The order
   is one path from the first song through all the others that maximises the
   sum of its scores: the greedy chain (each song followed by its best
   unplaced song) is the start, then moving any run of 1 to 3 songs to
   another place is tried until no move helps, which fixes the weak steps the
   greedy chain leaves for the end. The first song starts at 0 and the last
   one plays to its end. A window that holds
   silence is never used: two consecutive rows with a cosine of 0.95 or more
   are what digital silence gives (music almost never does), so a song with a
   silent tail ends before it and a silent intro is skipped (an edge silent
   from end to end keeps the old choice). On 200 random songs 7 of 199
   transitions faded through silence before this rule and 1 after. A quiet
   fade-out is not silence to the fingerprint, which brings every half second
   to its own level, so it can still be picked. The fingerprint knows nothing
   of tempo, so the stored tempo breaks near ties: a next song whose tempo
   cannot be beat-matched in tune with the current one (more than 60 cents,
   3.5%, apart, half or double time allowed; it was 8% until the page matched
   tempo by speed, which moves the pitch) loses 0.5 of cosine. That weight is set so about half of
   the hand-overs of a 50-song similar list can be beat-matched: on three such
   lists (pop, Italian rock, synthwave) neighbours with compatible stored
   tempos went from 64-70% at 0.1 to 96-100% at 0.5, for 10 to 14% less mean
   pair cosine (a weak signal, see below) and big intensity jumps of 16 to 24%
   instead of 16 to 18%; at 0.1 only 22 of 50 pop hand-overs were within 8%
   when measured on the audio, 38 at 0.5. The stored tempo is one value for
   the whole song, so it only steers the order and the beat match measures
   the tempo again where the songs meet. Measured on 200 random songs of a
   205k-song library: 0.06 s to read and decode the edges and 0.11 s to order
   them with the greedy chain alone (about 1 s on the first call after a
   restart); the start lands a median 13 s in and the end a
   median 14 s before the song's end, and the best pairs reach a cosine of
   0.12 to 0.35 between two different songs, so the order is a hint drawn from
   a fingerprint, not a musical similarity measure. On 38 real 51-song lists
   (19 seeds, each as a similar-song and a search-by-song list) the best next
   song stands about 2.4 standard deviations above a song's typical pair in
   both kinds of list, about what the best of 49 random draws gives: the
   fingerprint is built to tell recordings apart and hears only 160 Hz to
   4 kHz. What differs is intensity: in search-by-song lists the fingerprint
   chain took big intensity steps (over 1.5 standard deviations) less often
   than random pairs (16-34% against 28-33%), in similar-song lists as often
   or more (up to 38% against 31%). Measured on 40 more lists against the
   greedy fingerprint chain, the path with all terms gave 2 to 3% higher mean
   pair cosine, a worst step of 0.21 to 0.23 instead of 0.17, the same share of
   tempo-matchable steps (0.68 against 0.69, 0.67 against 0.67) and big
   intensity steps cut from 28% to 15% (similar-song) and from 25% to 10%
   (search-by-song); the MusiCNN term changes nothing in similar-song lists
   (every pair there is 0.88 to 0.95 alike) and pulls related songs together
   in search-by-song lists. Doubling the intensity weight halves the big
   steps again but starts losing tempo-matchable steps. The path alone,
   without the new terms, already raised the mean cosine by 3% and the
   tempo-matchable share by 3 to 4 points. A 200-song call answers in 0.27 to
   0.32 s.
2. **Beat match** (`tasks/smart_fade_mix.py`, `plan_transition`). Both songs
   are downloaded from the media server and only two regions are decoded
   (mono, 44.1 kHz): the 26 seconds before the ending song's fade-out time
   plus 2 after it, and from 8 seconds before the next song's fade-in time to
   28 seconds after it. A file libsndfile cannot open (an AAC `.m4a`; librosa
   1.0 no longer falls back to another decoder) goes through the analysis's
   PyAV loader instead: before, every such pair answered 500 and played the
   default crossfade without a plan, 14 of 86 transition calls in one evening
   of listening.
   librosa's beat tracker, the one the analysis uses,
   finds the beats in each (the next song's tempo only after its fade-in
   time); the tempo and the beat grid come from a line fitted through the beat
   times (a median beat gap is quantised to the tracker's 11.6 ms frames, off
   by up to 2%, and on real stretched pairs that left the beats drifting a
   median 110 ms apart over the mix, 35 ms with the fit). Fewer than 4 beats
   in either region raise `ValueError`. The stretch is the ending song's tempo
   over the next one's, half or double time allowed; past 60 cents (3.5%) the
   next song is not stretched (`tempos_match`, the same rule the order uses on
   the stored tempo). The next song enters on its first beat after its fade-in time, or
   on a section start within 8 beats of it: a novelty curve (an 8-beat
   checkerboard kernel on the beat-synchronous chroma, MFCC and loudness
   self-similarity) is computed over the region and the entry moves to the
   beat nearest its highest peak; of the two entries the one that leaves room
   for the longer mix is tried first, a tie going to the section start. The
   other entry is also checked if the first has poor beat alignment. The published
   DJ systems studied all start a mix on a phrase or section boundary; on 60 library
   songs with Harmonix Set section annotations the entry landed within
   1 second of a real section start 27% of the time (13% before; within
   0.5 s: 18% against 5%). The same snap at the ending song's fade point did
   not help (it sits deep in the outro, a median 10 seconds from any annotated
   boundary), so the mix end stays on the fingerprint point. The beat check
   (step 3) covers the fewest multiples of 8 beats of the ending song that
   reach 15 seconds (`LOCK_SECONDS`), ending on its beat nearest its
   fade-out time; when the regions have no room (the next song's first beat
   comes late) it loses 8 beats at a time down to 10 seconds, and without
   room for that it raises `NoRoom` (a `ValueError` carrying the pair's
   crossfade length), which the endpoint turns into the plain `crossfade`
   of that length at the two given points. The mix then plays the last
   whole bars of that span nearest the pair's mix length (`lengths`, see
   "How long a transition lasts" in section 19.1): the
   ending song stops on the same beat, the next song enters on the same cue,
   now over a later bar of the ending song, and the lock is the one checked
   over the whole span (a shorter span checked on its own lets weak
   onset curves pass, see "Retrying a refused beat lock" in step 7). The
   middle beat of the mix is where the bass lines swap unless the next song
   has no bass there (step 6), and the next song resumes alone the mix
   length times the stretch after its entry. A pair
   whose tempos match then goes through the beat check (step 3); a pair that
   passes gets a provisional rise or blend. The renderer makes the final
   decision using normalized levels at the selected cues.
3. **Beat check** (`_in_step`). Before a beat match is kept, the server
   cross-correlates the onset curves of the two planned parts (the next
   song's curve stretched by the plan) over half a beat either way, and keeps
   the match only if the best score within 30 ms of the plan (about where two
   drum hits stop sounding as one) is at least 60% of the best score at any
   offset. If a cue misses, both halves must agree on an offset within
   25 ms for an adjustment of up to 90 ms or 18% of a beat, whichever is
   smaller. The adjusted cue must pass the original alignment check.
   Near-ties pass: 16th-note hi-hats give peaks every 120 ms that
   cannot be told apart. A pair that fails, like a pair whose tempos do not
   match, becomes a `fade`: stretch 1 and the fewest whole beats of the ending
   song that reach the pair's crossfade length (`fade_seconds`), ending at its fade-out time. On three lists of
   10 transitions (an Italian rock similar-song list, an electronic and a
   rock text search) the 17 beat-matched mixes split cleanly: 11 at 0.97 or
   more, 6 at 0.65 or less. In the browser output of the similar-song list,
   3 of the 5 beat-matched mixes played their drums 64 to 181 ms apart
   (another 43 ms; the one with two steady grids 21 ms), and the check turns
   those 3 into fades; the tracker's beats wandered 37 to 94 ms around their
   fitted line on one side of each, against 3 to 12 ms on steady songs. Tried
   and dropped: the share of onsets on the beat against off it (off-beat
   hi-hats of electronic music put it near 1 either way), the correlation of
   the kick band alone or of each half of the mix (noisy: an outro often has
   no drums for 8 seconds) and the best offset alone (a coin flip between
   16th-note peaks).

   **Confirmed beat lock** (`_agreement`). Passing the check above is not
   enough to mix: the beats of both parts must also be trusted. A committee
   of four beat sequences is computed on each part (librosa's tracker on
   full-band, bass-band under 200 Hz and treble-band over 4 kHz onsets, plus
   its predominant-pulse curve), and their mean pairwise information gain (a
   40-bin beat error histogram, both directions, the lower kept) must reach
   1 bit (on 48 songs listeners rated the beats 2.4 of 5 at 1 bit and 3.7 at
   1.5 bits, ISMIR 2012; beat tracking errors were also the main reason
   curators rated beat-aligned transitions bad, 15% "beats do not align",
   ISMIR 2017). A pair that fails any check gets the `fade` plan, which the
   page plays as the default crossfade. The thresholds aim at about half
   crossfades and half beat-matched mixes on a 50-song similar list: the
   first strict version (a 5% stretch, the drum-hit share at 80% and
   1.5 bits) mixed 1, 9 and 9 to 18 of 50 hand-overs on the pop, Italian rock
   and synthwave lists, and was heard as "practically all crossfades". On
   the tempo-compatible pop pairs the drum-hit share splits into a group at
   0.7 or more, a few between 0.55 and 0.68 and a tail under 0.35 (clearly
   off; the earlier beat check found the audibly flammed mixes at 0.65 or
   less), so 60% is the lowest share kept. With the order's tempo weight at
   0.5, an 8% stretch, 60% and 1 bit, the three lists get 22, 22 and 26
   beat-matched mixes of 50 (28, 28 and 24 crossfades). The committee adds
   about 0.3 s per part, and only for pairs whose tempos already match.

   **Where the default crossfade starts** (`_fuller_start`). A crossfade laid
   where the ending song is already inside its own fade-out while the next
   song starts on a quiet intro leaves both quiet at once: recorded through
   the page, one such pair sat 11.7 dB under the line between the two songs
   for about 6 seconds, and transition regions with low loudness are the
   ones that "sound awkward and abrupt" (ISMIR 2017). From the two loudness
   curves (the same 100 ms K-weighted blocks, each against its own song) the
   server predicts the equal-power mix for every beat of the ending song's
   region it may start on, and takes the beat whose quietest second is
   highest, only when that lifts it by 3 dB or more and never so early that
   the ending song is cut by more than its last 30 seconds; the next song
   still enters on its section start. Over 30 transitions recorded in
   Chromium from three similar-song lists (pop, Italian rock, synthwave),
   default crossfades dipping more than 3 dB under that line went from 5 to
   3, the worst from -11.7 to -8.8 dB; the one left is a next song whose
   section start is a quiet build-up that drops as the ending song is gone
   (an intro mixed under an outro), which moving the start cannot change. No
   sample clipped (peak 0.88), no cut clicked and no silence passed 300 ms
   inside a session in either run; beat-matched mixes thin about 3 dB in
   their middle, the bass swap and mid cut by design.
4. **Confidence** (`_confidence`). For a `fade` both whole songs are also
   decoded once at 11,025 Hz (0.4 to 1 s each) for their loudness:
   K-weighted 400 ms blocks every 100 ms, the integrated loudness gated at
   -70 LUFS and 10 LU under the mean (BS.1770; a full-scale 997 Hz tone reads
   -3.01 LUFS). The planned parts are measured against their own song: a
   `fade` where the ending song is at or above its own loudness and the next
   song's part is at least 2 LU further under its own is `low`; a beat-locked
   plan is `high` without decoding anything more. The page now plays `low`
   and `medium` the same way, as the default crossfade. On 14
   transitions rated by ear on a phone, two of the three bad ones described
   the music dropping from an intense song into a slower one, and both
   measured as unlocked fades from an ending song 0.2 to 2.9 LU above its
   own level into a next song 2.3 to 6.3 LU lower: both are `low`. The third
   had no plan at all (an `.m4a`, the 500 above) and now gets a plan
   (`medium`). Of the 10
   good ones, beat-locked mixes with level steps, fades out of an ending
   already 1 to 3 LU under its own level and fades into a quiet start stayed
   `high` or `medium`; one good fade and the one rated "not so bad" came out
   `low` too. Over 48 hand-overs of 16 real lists: 13 `high`, 27 `medium`,
   8 `low`. The timbre (MFCC), chroma, onset rate and brightness of the two
   parts did not separate good from bad, and neither did the whole-song
   intensity step, so they do not decide the confidence. A fade's plan now
   takes up to about 1 s more; the page asks for it while the previous song
   plays.
5. **Voice check** (`tasks/smart_fade_voice.py`, `clear_voice`). After the
   plan, both songs' audio around the cues is decoded at 16 kHz and every
   voice is found (the MusiCNN singing tags over 3-second patches, joined
   with a speech detector). The cues may then move, the ending song's within
   `SMART_FADE_SEARCH_SECONDS` and the next song's start within
   `SMART_FADE_ENTRY_LIMIT_SECONDS`, to the pair that costs least for weak
   or silent audio, two voices at once and distance from the planned cues; a
   beat mix moves in whole bars only, and a few skipped seconds are given
   back afterwards when energy and voices allow it. A moved beat mix is
   checked again (step 3): when its hits no longer line up and no voices
   clash at the planned cues, the verified planned cues stay (two drum
   patterns that drift apart are the most common fault listeners hear in
   automatic transitions, 15% against 6.5% for voices over voices, ISMIR
   2017); with a clash the lock is dropped and the endpoint plans the pair
   again as a pair without a lock (own speed, the crossfade's length, the
   fuller start) and checks its voices, so a dropped lock plays what a pair
   that never locked would: on 8 real dropped locks the deepest dip under the
   level line from one song to the next went from 6.4 dB to 2.1 dB (mean 2.2
   to 1.0 dB), the crossfade kept at the beat cues having sat on a quiet bar.
   The share of time two voices overlap goes to
   the curves (`vocal_shapes`): an equal-power crossfade with up to 9 dB of
   mid exchange (half on the highs) around the middle and a late taper of
   the old singer, beat mixes keeping their own deeper cuts. A percussion
   mix (section 19.1) is chosen here, only when no voice overlaps another in
   it.
6. **What a player executes** (`tasks/smart_fade_levels.py`,
   `transition`, the endpoint's answer). Every number of a transition is
   computed on the server, so any frontend only schedules it; the shared page
   (`static/song_player.js`) computes nothing of its own. The page decodes both songs
   with the browser's Web Audio, in one audio context made in the Play click
   at the output device's own rate, never a whole song at once: the play
   route sends pieces cut on the sample (30 s on a grid of whole seconds, 1 s
   more on each side, where the browser's resampling of a piece differs from
   the whole song's; inside that, a piece equals the whole decode within
   -150 dBFS at every pair of song and device rates tested). Consecutive
   pieces join inside that margin through a 23 ms blend of identical audio,
   and a mix plays one piece per song that covers it whole, so two 15 minute
   192 kHz songs take about 400 MB in the browser instead of 7 GB. The page
   starts every piece as a buffer source on a whole sample of that one
   audio clock, at a whole-sample offset in the song; Skip restarts the
   ending song's source and plans the mix again. Chrome and Safari play a
   source started between two samples by interpolating between them for its
   whole length (Firefox rounds it), which took up to 5 dB off a song's highs
   at 15 kHz (6 dB on a 44.1 kHz output, 2 dB at 10 kHz) for the rest of
   every song after a transition. After a mix the next song plays on from a
   fresh source with flat filters, from the sample where the mix ends, and
   the mix's sources are muted there so no filter tail rings on.
   No audio element is ever seeked: Safari seeks MP3 files without a
   seek table several seconds off while reporting the right time, which
   played the end of a song, jumped back and played it again.
   - **Tempo glide** (`time_maps`, sent as each song's `speed`). In a blend
     or a rise the tempo glides beat by beat from the ending song's to the
     next song's, as Spotify's DJ transitions do (ISMIR 2017): both songs are
     stretched onto that gliding grid, each half the difference off in the
     middle (splitting the change between the two songs is more comfortable
     than putting it all on one, 96 listeners, ISMIR 2009), so neither song
     jumps in tempo when the mix starts or when the next song plays on alone;
     the mix then lasts the mean of the two parts (`seconds`). The default
     crossfade plays both songs as they are.
   - **Speed, no time-stretch** (sent as each song's `speed`, the slope of
     its time map). The page plays each song's own audio at that playback
     rate, as a DJ deck without key lock does: the pitch follows the tempo
     glide, and nothing is cut or re-joined. So the two songs sit apart in
     pitch by their whole tempo ratio for the 10 to 21 seconds of the mix (a
     6% gap is about a semitone: two songs in one key clash like a minor
     second). A beat mix therefore needs the tempos within `MAX_LOCK_CENTS`
     (60 cents, 3.5%); on an 11-transition pop list the mixes at 90 to 117
     cents were the ones that sounded wrong, those at 4 and 19 cents clean.
     `tempos_match`, which also steers the smart fade order, shares that 60-cent
     rule, so the order picks neighbours that can mix in tune.
     The speeds, joined by straight lines as Web Audio plays them, land
     within 2 ms of the time map; after the mix the next song goes on from a
     fresh source at `continue_from`, blended in over 23 ms, so the
     hand-over has no click. The server's simulation resamples the same way,
     so it now hears exactly what the page plays.
     Before, the page time-stretched both songs with WSOLA (46 ms frames
     re-joined every 23 ms, pitch kept). On dense, loud rock it added a
     roughness at the 43.5 Hz join rate that listeners heard as distortion
     in every beat mix: recorded through the page on three beat mixes of a
     rock similar list, the envelope modulation at 43.5 Hz sat 3.2, 7.5 and
     0.2 dB above the same mixes rendered without WSOLA, while outside the
     mixes the two matched exactly.
   - **Three bands, moved like a DJ's EQ** (`mix_shapes`, sent as
     `high_db`, `mid_db`, `bass_db` and each song's `volume`). Every song in
     the page plays through a low-pass that stays fully open, a low shelf, a
     mid peak and a high shelf that stay flat until a mix, set from
     `bass_hz` (180 Hz), `mid_hz` (1 kHz), `mid_q` (0.5, wide) and
     `high_hz` (4 kHz). In a blend every move is a raised cosine, an S-curve
     that starts and ends slowly, and the bands hand over one after the
     other the way DJ guides teach a 3-band EQ mix: the next song's highs
     come in over the first third of the time before the swap (from
     -40 dB), its mids from -27 dB until a beat before the swap, the two bass
     lines swap in one equal-power move over 2 beats (one bass line at a
     time), then the ending song's mids go down over two thirds of the time
     left and its highs over the last two thirds. The faders only fade the
     two ends in and out over 2 beats, while the bands are cut, and carry the
     steady-level correction below; real DJ mixes keep a near constant energy
     through a transition with the faders up and the EQ doing the hand-over
     (DAFx 2022, 50,742 transitions). A rise sweeps a 2nd-order Butterworth
     high-pass on the next song from 800 Hz down to 10 Hz (log scale) from the
     end of its highs move to a beat after the swap, instead of its bass. A
     filter rise (no lock) sweeps the same high-pass from 800 Hz to the bass
     shelf's 180 Hz until a beat before the swap, holds it there, and opens it
     to 10 Hz over the 2 beats in which the ending song's bass shelf closes. A
     percussion mix crosses equal-power faders over its 8 to 12 seconds, swaps
     the two kick drums with amplitude curves that add up to 1 (two kicks on
     the same beat add like one signal, so equal power would bump them, as
     DAFx 2016 derives for correlated signals) and crosses the hi-hats around
     the same moment. The default crossfade uses equal-power faders, right for
     two unrelated songs (the same derivation), with no filters unless voices
     compete (`vocal_shapes`).
   - **Each band fitted to the two songs** (`band_shifts`, folded into
     `high_db`, `mid_db` and `bass_db`). The S-curves above only fix the
     order and timing; how far each band goes depends on the songs. The
     server splits the mix it plays in advance into bass (under 180 Hz),
     mids (300 to 3000 Hz) and highs (over 4 kHz) and, at every point,
     compares each band with a target calculated by blending the two songs'
     instantaneous band powers before conversion to dB, all the first song's at the
     start and all the next song's at the end. Both songs' band gains are
     raised or lowered together by the difference (at most 6 dB, smoothed
     over 1 second, none at the two ends), so a band neither jumps nor
     doubles when it changes hands, for example a next song with a louder
     bass gets the first song's bass raised before the swap and its own
     lowered after it, while each song keeps its own breaks and drops.
     Real transitions carry gains of their own per band and per pair
     (DAFx 2022, NIME 2021), and the learned DJ models predict them from
     the two songs' audio (ICASSP 2022). On 7 real beat mixes (pop and rock
     lists), played back offline through their filters, each band's
     distance from that target fell from 2.10 to 1.26 dB for the bass and
     from 1.59 to 1.32 dB for the highs (median), the mids stayed about the
     same (1.30, then 1.40) and the whole mix too (0.32 dB off its line);
     an answer still takes 2.3 to 3.0 seconds on the live test server.
     Recorded through the page on 10 pop transitions, the beat mixes went
     from 1.06 to 0.90 dB off the level line, 0 clipped samples.
     Tried and dropped: a straight line from the first song's band level
     to the next song's as the target (it flattened each song's own bass
     and highs by up to 6 dB, which a DJ does not do), two correction
     passes instead of one (no better), and fitting only bass and highs
     (the mids got worse). Real DJ
     transitions are reproduced better by a fader plus a 3-band EQ than by a
     crossfade, and in them each knob moves one way only, in short steps,
     about one change at a time (DAFx 2022, 50,742 transitions of 5,040
     mixes); sub-band gains fitted to each real transition beat a fixed
     linear crossfade (NIME 2021, 3,930 transition regions); and in a
     listening test of generated transitions, changes that came too fast
     were the main complaint (ICASSP 2022). The curves used before crossed
     the two bass lines over the whole middle half of the mix, so both
     played together for about half of it.
   - **Where the bass swaps** (`swap_point`, sent as `middle`). The plan's
     mix starts on the next song's section start and lasts whole 8-beat
     steps, so its middle beat starts a bar of the next song. The server
     measures the next song's own bass (under 180 Hz, a 2nd-order low-pass,
     along its time map) in every 4-beat bar of the mix and swaps on the
     bar fewest beats from the middle where that bass reaches 25% of its
     loudest bar, inside 30 to 70% of the mix, the later of two equally
     near; with no such bar, on the last bar of that window. A swap into a
     bar where the next song has no bass yet (a quiet intro) leaves the mix
     with no bass line at all until the next song's comes in. Measured on
     14 real beat mixes of the pop, Italian rock and synthwave lists, each
     answer played offline through its own filters and faders: 12 keep the
     middle beat (the next song's bass already plays there), 2 rock pairs
     whose next song enters on a quiet intro move one bar (4 beats) later, and
     there the deepest dip of the mix's bass (under 180 Hz, against the line
     from one song's bass to the other's) goes from -3.8 to -0.4 dB and from
     -11.6 to -9.0 dB (more than 6 dB under it for 2.8 s, now 1.7 s). On the
     5 pop beat mixes both bass lines played within 12 dB of full for 3.7
     to 4.5 s of each mix with the straight lines, 0.3 to 0.6 s with the
     swap. Recorded through the page (10 transitions of the pop list), the
     beat mixes stay as close to the level line as before (median 1.08 dB
     off it, 1.10 with the straight lines), with 1 mix more than 3 dB under
     it instead of 2 (a crossfade, untouched) and 0 clipped samples.
   - **Peak protection** (in each song's `volume`). The server adds the
     magnitudes of the two filtered and faded tracks to bound their combined
     peak, with about 1 dB of headroom. It lowers both faders ahead of a peak
     and restores the level gradually afterward. The whole-song gain also
     limits peaks to about -1 dBFS.
   - **Loudness continuity.** A crossfade of a pair without a beat lock also
     keeps the mix within 4 dB (`CONTINUITY_FLOOR_DB`) of the louder of the
     two songs at each point, so two songs both fading leave no hole. A song
     counts with its full level while its fader carries the mix (above
     equal power: the ending song until the middle, the next one from
     there), then in proportion to its fader. Counting the ending song in
     full to the end, a next song coming in much quieter (a quiet intro 6 to
     19 dB under the ending song) kept the ending song loud to the last
     second and boosted the next one up to 5.6 dB, which then dropped back
     when the mix ended: half of 123 rendered EQ blends rose more than 3 dB
     over the line between the two songs' levels. Correction is smoothed, limited, and subordinate to peak protection.
   - **Default crossfade without a plan.** A plain crossfade over the 7.5
     seconds before the ending song's fade-out time, the next song from its
     fade-in time. With both excerpts buffered, the player measures their peaks
     and limits the equal-power faders to the available headroom. Incomplete
     buffers, joins within one recording and demo fades use complementary S-curves.
   - **Volume curves on the two songs' levels** (`fader_scale`). A fixed equal-power curve
     only adds up to a steady level when both songs stay steady. An ending
     song that is already fading out on its own, under a fade-out of its
     own, or a next song that starts with a quiet build-up, under a fade-in,
     leaves a hole in the middle of the transition. Real DJ mixes keep a
     roughly constant energy through their transitions (DAFx 2022, 50,742
     transitions of 5,040 mixes), and the power of a sum depends on how
     correlated the two signals are, so no fixed curve keeps it steady
     (DAFx 2016: equal power is exact only for uncorrelated signals). So
     for every planned transition the server measures, on both songs at
     their -14 LUFS gains, the K-weighted level of the ending song over the
     3 seconds before the mix and of the next song over the 3 seconds after
     it: the target is the straight line, in dB, from the first level to the
     second. It then plays the mix in advance at 44.1 kHz, every channel,
     sample by sample: both songs along their time maps, under the
     equal-power curves, through the mix's own bass and mid filters (and the
     rise's high-pass), summed and K-weighted, in 400 ms blocks. So the filters and
     any correlation between the two songs count, not only their two
     separate levels. Both curves are scaled together by the difference to
     the line: at most 6 dB up or down (the same twice-the-level bound the
     DAFx 2022 mixer model gives a fader), smoothed over 1 second, and faded
     to nothing over the first and last tenth of the mix, so each song still
     starts and ends exactly as it played alone. A song with a silent
     neighbour (under -60 dB) keeps the plain curves. The clipping check runs
     after it, on the shaped curves, so a boost never makes a peak. With all
     of this an answer takes 2 to 3.4 seconds on the live test server (the
     plan alone 0.7 to 2.6), asked while the song before plays. Measured on the
     same 30 real transitions of three similar-song lists, recorded through
     the page (executing these lists) before and after: the mixes that fell
     more than 3 dB under the line went from 10 to 4 and those more than 6 dB
     from 3 to 0 (deepest -7.3 to -5.3 dB), the median distance from the line
     from 1.18 to 0.61 dB (beat-matched mixes 1.52 to 0.98, crossfades 0.86
     to 0.43), with 0 clipped samples and no click before or after. Three of
     the four left are short dips in the last 6% of the mix, where the next
     song itself is quiet just before it plays on alone (the same dip was
     there before); the fourth sits on the -3 dB line. The same rule run in
     the page instead of the server gave every transition within 0.3 dB.
     Adding the two songs' separate levels instead of playing the mix in
     advance left 7 such mixes and 0.86 dB: as good on crossfades (0.45 dB),
     so two different songs barely correlate, but blind to the bass swap and
     the mid cut of a beat-matched mix.
   - **Same loudness** (`song_level`, sent as each song's `gain`).
     Each song is read once, in blocks at its own rate, every channel
     K-weighted and gated (BS.1770), with its sample peak; the gain brings
     it to -14 LUFS, as streaming players do, and raises a quieter song only
     until its peak reaches 0.891 (-1 dBFS), never limited. A file libsndfile
     cannot open (an AAC `.m4a`) is read as mono by the analysis loader, so it
     counts as two equal channels and its peak gets a 3 dB margin. The page
     applies the gain after all the mix filters; the first song of a play
     waits for its own plan, which carries its gain, and each next song takes
     the one its incoming plan gives. Plain Play and Demo keep every song as
     decoded.
   - **Measured live** in Chromium and Firefox on 13 mixes with Play, Skip
     and Skip during the mix: the recorded output was where the plan said to
     the sample, and on 10 real stretched pairs the page's stretch placed the
     next song within 12 ms of the plan for the whole mix. On 4 blends and
     4 crossfades the highlight moved once each, in the middle, and never
     showed the quieter song. Recorded through the page in Chromium 148 and
     Firefox 150 with synthetic songs, each song outside its transitions
     (before the first, and after a blend, a rise, the filtered fade the
     default crossfade replaced, and a crossfade without a plan) came out as
     decoded:
     bit for bit, or in Chromium off
     by less than -185 dBFS on under 1% of near-silent samples (its rounding
     of the start time), with 0.00 dB change at 15 kHz.
7. **Measured and not kept.**
   - **A loop of the ending song** (some streaming players repeat a bar to
     stretch a short ending; Apple Music's AutoMix "repeats sections where
     necessary"). Over 93 real transitions only 1 mix came out under
     15 seconds, and the ending song fell 10 dB or more below its own level in
     the first half of the mix in 6% of them (20 dB in 2%), half of those
     already mixed as a rise. For the pairs without a beat lock, a
     loop needs the same steady grid and shared tempo that are missing there;
     DJs use a loop only to lengthen a short intro or outro of two matching
     songs, and AutoMix itself stays off when tempos are incompatible.
   - **A stretch that separates tones from hits** and runs the tones through
     a phase vocoder. It came closer to an exact synthetic reference (1.26
     against 1.85 dB), but on real loudness-limited masters rebuilding the
     phases raised the peaks by 3 to 4 dB (1.04 to 1.50) and clipped hundreds
     of samples in every mix, and in a listening test it sounded the same as
     WSOLA (itself since replaced by a plain speed change, see Speed above).
   - **Shorter mixes.** 8 beats, an echo out and then 16 beats all sounded
     like a jump on a phone with the shapes of that time. Real DJ transitions
     peak at 32 and 64 beats, 15 and 30 seconds at 128 BPM (ISMIR 2020,
     20,765 transitions of 1,557 mixes), and streaming players cross over at
     most 12 seconds.
   - **One length for every song.** Beat mixes of 15 to 21 seconds (the
     fewest 8-beat steps reaching 15) with 10-second crossfades, EQ blends,
     filter rises and vocal handoffs sounded right on a pop Similar Song
     list and too long on an Italian rock one; about 10-second mixes with
     5-second native-speed kinds sounded right on a fast metal list and too
     abrupt on the pop and rock ones (a short transition on a slow song
     sounds like it is hurrying the song). Both gave way to the counted
     beats of section 19.1.
   - **The filtered fade** for pairs without a beat lock (about 10 seconds,
     unstretched, the ending song muffled by a low-pass closing to 400 Hz and
     the bass lines swapped): on a pop similar-song list the plain crossfade
     sounded better on average, so pairs without a confirmed lock now get the
     default crossfade.
   - **Eight more transitions for pairs without a beat lock**, reviewed and
     removed: a spectral sweep (a low-pass on the ending song and a high-pass
     on the next one closing on each other from 18 kHz to 95 Hz), a filter
     handoff (low-pass to 1.35 kHz and a bass swap), an instrumental bridge
     (the same with a 4.6 kHz low-pass when the chords matched), an opening
     pickup (the next song from its first sound under a low-passed ending), a
     natural tail (a 4-second mix across a level drop of the ending song), a
     cut into the drop (a 3.4-second filtered build and a 0.7-second switch), a
     rhythm handoff (unlocked drums traded in 0.8 seconds) and a vocal handoff
     with a mid exchange of its own (a second path for what `vocal_shapes`
     already does; the vocal handoff kept above is that voice exchange).
     On the same 45 hand-overs of three 16-song similar lists (pop, rock,
     synthwave) they took 41 of the 45: spectral sweep 33, opening pickup 4,
     one each of the others, all 7.5 seconds or shorter, and not one plain
     crossfade was left. They are the filtered fade and the short hand-overs
     above under new names, played on the pairs that do not lock, where
     overlapping drums cannot line up. Published automatic DJ systems use beat-matched
     mixes only (a drum and bass system: no filter sweeps; a streaming
     service's playlist transitions: one beat-aligned crossfade), their
     listeners' complaints were beats that do not align (15%) and voices over
     voices (6.5%), never the kind of transition (ISMIR 2017), and an
     8-bar linear crossfade was not significantly worse than rule-based EQ
     transitions or a learned one (ICASSP 2022). With the four transitions
     left the same 45 hand-overs play 39 crossfades (10.2 seconds, both songs
     audible together for 4.9 seconds against 2.0, no level hole deeper than
     3 dB, no clipped sample) and 6 beat mixes (18.3 seconds). On the
     first 15 hand-overs of 12 Similar Song lists of 12 genres (180) they
     play 135 crossfades, 37 beat mixes and 8 percussion mixes (house 6 of
     15, hip-hop and latin 1 each, none where the music is not drum-led);
     94 of the crossfades are tempos too far apart to mix in tune, mostly
     because the stored tempo the order uses is measured in steps of about
     7% (a song reads 104.2, 117.2 or 156.2 BPM), so neighbours of the same
     stored tempo are 0 to 7% apart. A rise needs a quiet ending
     against a full entry once both songs play at the same loudness: none at
     the order's cues in 150 hand-overs, two with a later fade-out point.
   - **Retrying a refused beat lock on shorter bars** (4 beats at a time down
     to 10 seconds, both entries). On 12 Similar Song lists of 12 genres (180
     hand-overs) it turned 18 crossfades into beat mixes, 12 with both halves
     checked, but played along the answer's speeds their drums were within
     30 ms in only 38% (46% with both halves) of the quarters of the mix,
     against 76% for the locks found at full length and 88% for the
     percussion mixes: checking many lengths lets weak onset curves pass by
     chance, so a refused lock stays a crossfade.
   - **Other EQ blend shapes**, rendered on the same 40 EQ blends of the
     genre lists: holding the playing bass at full power through the swap
     (a boost of up to 3 dB) moved the median dip from 1.45 to 1.3 dB with
     the same 7 holes deeper than 3 dB and 0.3 dB higher bumps; leaving out
     the mid and high exchange kept both songs audible together longer (2.7
     against 2.15 seconds) but made 9 holes. Neither is kept.
   - **Other filter rise rules and shapes**, on the 3 rises of one rock list
     rated by ear and the 69 genre-list rises: tempos related only as the
     same, half or double time would also refuse one of the acceptable rises
     (93 into 123 BPM, a 4:3 reading) and 32 of the 69; a pulse similarity
     (the correlation of the two parts' onset autocorrelations) scored an
     acceptable rise lower than the bad one. Holding the high-pass at 300 Hz
     instead of 180 Hz made 10 dips deeper than 3 dB against 8, and the EQ
     blend's shape 14.
   - **An 8-second beat mix** (5.5 to 13 seconds in 4-beat steps, both
     faders trading level around the swap) sounded too rapid on a phone; the
     10-second mixes keep the EQ-led hand-over with the faders only at the
     two ends.
   - **The bass swap on an 8-beat phrase** of the next song before any
     other bar. On the 24-beat mixes the phrases fall at a third and two
     thirds of the mix, never in the middle. Recorded through the page, the
     three such pop mixes swapped at two thirds sat 1.18, 0.96 and 1.49 dB
     off the level line against 1.09, 0.74 and 1.08 with the middle swap: the
     ending song's mids and fader then go down within the last 5 seconds,
     the fast change listeners object to.
   - **A radio-style hand-over** for `low` pairs (the ending song plays to
     where it falls 8 LU under its own loudness and fades in 2.5 seconds
     while the next one starts from its first sound): rejected after
     listening, the default crossfade is kept for them.
   - **Fading only one side** when one song is much louder at the cut, as
     radio crossfades do. In the 14 rated transitions every pair where the
     ending song was already quiet or the next started quiet sounded good
     with the existing mixes, so no extra transition was added for it.
   - **Other beat fixes, measured without gain**: forcing the tracker to the
     stored tempo, snapping the mix to a bar by onset strength, and moving the
     fade points to the nearest stretch where the tracker looks steady (worse:
     the tracker looks steady even on weak beats).

### 19.4. Environment Variable Configuration

- `NEURAL_FINGERPRINT_ENABLED` (section 17.4): Smart fade needs it. Without it
  the option is not shown and the two endpoints answer 503.
- `SMART_FADE_BEAT_MIX_ENABLED`, `SMART_FADE_BEAT_RISE_ENABLED`,
  `SMART_FADE_PERCUSSION_MIX_ENABLED`, `SMART_FADE_EQ_BLEND_ENABLED`,
  `SMART_FADE_FILTER_RISE_ENABLED` and `SMART_FADE_VOCAL_HANDOFF_ENABLED`
  (all `true`): the setup wizard's Smart Fade switches, one per transition
  ("Switching transitions off" in section 19.1). The default crossfade has
  no switch.
- `SMART_FADE_END_MARGIN_SECONDS` (2), `SMART_FADE_SEARCH_SECONDS` (32),
  `SMART_FADE_ENERGY_DROP_DB` (3), `SMART_FADE_RECOVER_SECONDS` (4),
  `SMART_FADE_END_RECOVER_SECONDS` (4), `SMART_FADE_ENTRY_LIMIT_SECONDS` (12),
  `SMART_FADE_VOCAL_MIX_SECONDS` (14), `SMART_FADE_VOCAL_DUCK_DB` (6),
  `SMART_FADE_VOCAL_TAIL_DB` (6) and `SMART_FADE_LEVEL_CACHE_SIZE` (128) in
  `config.py` tune where the voice check moves the cues and how far it
  ducks competing voices.
- The lengths and limits are constants of
  `tasks/smart_fade_mix.py` (`LOCK_SECONDS` 15, `MIX_BEATS` 32 to 24,
  `FADE_BEATS` 18 to 13, `MIN_BEAT_MIX_SECONDS` 10, `MAX_BEAT_MIX_SECONDS`
  21, `MIN_MIX_SECONDS` 5, `MAX_FADE_SECONDS` 10, `PLAIN_FADE_SECONDS` 7.5,
  `SPARSE_SHARE` 0.55, `DENSE_SHARE` 0.30, `DRUMLESS_SHARE` 0.12,
  `FOLD_BPM` 100, `FAST_ONSETS_PER_BEAT` 3.9,
  `PERCUSSION_SECONDS` 8 to 12, `PERCUSSION_LOCK_CENTS` 90,
  `MAX_LOCK_CENTS` 60, `RISE_DB` 6, `MATCH_MS` 30, `MATCH_SHARE` 0.6,
  `EDGE_SECONDS` 30, `FULL_LU` 0, `DROP_LU` 2, `MIN_AGREEMENT_BITS` 1.0,
  `FLOOR_GAIN_DB` 3), of
  `tasks/recording_search_manager.py` (`_FADE_TEMPO_PENALTY` 0.5,
  `_FADE_INTENSITY_WEIGHT` 0.04, `_FADE_SOUND_WEIGHT` 0.1, `_FADE_WEAK_WEIGHT`
  1) and of `static/song_player.js` (`SMART_CROSSFADE_SECONDS` 7.5,
  `TARGET_LUFS` -14, `PEAK_CEILING` 0.891).
