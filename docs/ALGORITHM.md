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
- `AI_CHAT_DB_USER_NAME`, `AI_CHAT_DB_USER_PASSWORD`: an optional low-privilege
  PostgreSQL role used by the Instant Playlist queries.

Safety caps:

- `ALCHEMY_MAX_N_RESULTS`, `ALCHEMY_DEFAULT_N_RESULTS`, `CLEANING_SAFETY_LIMIT`,
  `MAX_SONGS_PER_ARTIST`, `DASHBOARD_BROWSE_MAX_OFFSET`,
  `SWEEP_PRUNE_MIN_FETCH_RATIO`.

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
   the worker.
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
9. **Persistence and plugin hook.** The results are written under the canonical
   id, and the `song_analyzed` plugin hook fires with the server the song came
   from. See [PLUGIN](PLUGIN.md).

Each optional stage (CLAP, lyrics, chromaprint) is best effort. A failure is
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
prunes mappings whose track is no longer on that server. Pruning is refused when
the fetch returns fewer tracks than `SWEEP_PRUNE_MIN_FETCH_RATIO` of the
mappings already stored, so a transient provider error can never mass-delete
valid mappings.

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
- `CHROMAPRINT_BACKFILL_ALBUMS_PER_RUN`: albums per server whose already
  analyzed tracks get a fingerprint back-filled on each analysis run.
- `CHROMAPRINT_GATE_ENABLED`: use the fingerprints in the identity decision.
- `CHROMAPRINT_MATCH_THRESHOLD`, `CHROMAPRINT_MAX_ALIGN_OFFSET`,
  `CHROMAPRINT_MIN_OVERLAP`: the comparison parameters.
- `FPCALC`: path to the `fpcalc` binary. It is on `PATH` inside Docker and set by
  the launcher in the standalone builds.
- `SWEEP_PRUNE_MIN_FETCH_RATIO`: safety ratio that blocks pruning after a partial
  catalogue fetch.

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
song, path, radio). An anchor also stores its exclusions with their radius, so
re-running it later gives a comparable result.

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
   - *Anchor*: the stored vectors, plus the stored exclusions which are
     re-applied at their saved radius.
   The total number of anchor points is capped by `ALCHEMY_MAX_ANCHOR_POINTS`.
3. **Centroids.** The Include points are averaged into the add centroid and the
   Exclude points into the subtract centroid.
4. **Candidate search.** The index is queried around the add centroid (a
   multi-query when there are several anchor points), asking for clearly more
   candidates than requested so there is room to filter.
5. **Filtering.** The original input songs are removed, then the standard
   near-duplicate distance filter, the title and artist deduplication and the
   `MAX_SONGS_PER_ARTIST` cap are applied.
6. **Subtraction.** If there is a subtract centroid, every remaining candidate
   closer to it than the threshold
   (`ALCHEMY_SUBTRACT_DISTANCE_ANGULAR` or `ALCHEMY_SUBTRACT_DISTANCE_EUCLIDEAN`,
   or the request override) is removed. Those songs are returned separately so
   the plot can show what was excluded and why.
7. **Temperature sampling.** The distances of the survivors to the add centroid
   are turned into similarity scores and passed through a softmax with the
   requested temperature. A low temperature sharpens the distribution and
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

- `ALCHEMY_DEFAULT_N_RESULTS`, `ALCHEMY_MAX_N_RESULTS`: default and hard cap on
  the number of results.
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

- `seed_search`: find songs similar to a named song or artist.
- `text_match`: match a description against the audio (DCLAP) or the lyrics.
  Modes are only exposed when the matching feature is enabled.
- `knowledge_lookup`: the brainstorm tool, for requests that need outside
  knowledge rather than a library lookup.
- `search_database`: metadata and feature filtering (genre, mood, tempo, energy,
  key, scale, year, rating, exclusions).

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

- **Hint pre-extraction.** Years, decades, BPM, tempo and energy words, genres
  and negated genres are extracted from the request with plain regular
  expressions before the call. Anything the model then leaves out is merged back
  into the filter afterwards.
- **Hallucination stripping.** Year, instrumental and exclusion arguments that do
  not appear in the request are removed. An exclusion only survives if the
  request actually contains a negation.
- **Deduplication and caps.** Duplicate tool calls are dropped and a plan is
  capped at four calls.
- **One replan.** If the plan returns nothing at all, exactly one replan runs with
  the failure as feedback.
- **Unsupported constraints** (for example a duration request) are reported as a
  note instead of being silently ignored.

#### Composing the result

When several tools return candidates, the results are merged and re-ranked:

- Songs matching the requested categorical values (genre, voice type, mood) are
  ranked above songs that do not. The continuous dimensions only order songs
  **within** a tier, so the categorical request is a strong preference and not a
  hard gate.
- Songs returned by more than one tool get an intersection boost.
- The primary tool's own similarity rank is blended in as an extra dimension.
- Titles that look like intros, skits or interludes are pushed down.
- `exclude_artists` and `exclude_genres` are the one **hard** cut.
- If a filtered pool comes up short, a relax loop lowers the score threshold to
  backfill. A filter-only query that still underfills re-runs without its soft
  dimensions (tempo, energy, moods, key, scale, rating) and then applies them as
  the soft re-rank over the broader pool.

`knowledge_lookup` is the one exception: its results are returned as they are. It
already grounds itself, because the model emits a *recipe* (filters, sound
descriptions and seed artists) which is then run against the real library and
fused, rather than recalling song titles that may not exist. Applying the normal
re-rank on top of that would fight the brainstorm.

#### Streaming and playlist creation

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
parameterized code paths, and they can run as a dedicated low-privilege
PostgreSQL role (`AI_CHAT_DB_USER_NAME`) that only has read access. Provider API
keys stay server-side. Tool failures return a generic message; the real error only
reaches the container log.

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
- `AI_CHAT_DB_USER_NAME`, `AI_CHAT_DB_USER_PASSWORD`: the read-only role used for
  the library queries. The role is created or reset automatically when set.
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

**The important guarantee: cleaning never shrinks the catalogue by accident.**

- A song that disappeared from **one** server keeps its analysis, its embeddings
  and its mappings on the other servers. It simply stops appearing in results for
  the server that lost it.
- A song bound to **no** server is an orphan. By default it is only reported. It
  is deleted only if `CLEANING_CATALOGUE` is on, or if the per-run checkbox is
  ticked, and even then only when every server was read completely.
- A server whose library could not be fully read is skipped, so a transient
  provider error can never unbind valid mappings.

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
   from `track_server_map` for tracks it no longer has. The
   `SWEEP_PRUNE_MIN_FETCH_RATIO` guard applies here too: a suspiciously small
   fetch blocks the prune.
4. **Orphans.** Tracks now bound to no server at all are grouped by artist and
   album and reported, up to `CLEANING_SAFETY_LIMIT` entries. Deleting them
   requires all of: the catalogue option enabled for this run, no server failed
   or was refused, and the orphans being fewer than half the catalogue. That last
   guard exists because "half the library disappeared" is much more likely to be
   a bad view than a real deletion. When the delete does run it removes the score
   row, the embeddings and the playlist references together.
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

- `CLEANING_SAFETY_LIMIT`: maximum number of unbound albums listed in the report.
- `CLEANING_CATALOGUE`: whether orphan catalogue rows are deleted as well as
  reported. The page has a per-run checkbox that enables it for one run without
  changing the default.
- `SWEEP_PRUNE_MIN_FETCH_RATIO`: the partial-fetch guard.
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
   **alchemy radio**, and any task a plugin has registered.
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
   double-fire the same schedule.
4. **Enqueue the batch work.** Analysis, clustering, sonic fingerprint and plugin
   tasks are **enqueued** as queue jobs, so a slow media server cannot swallow a
   scheduling window or block the other schedules. The **alchemy radio** is the
   exception: it is an online feature that queries the in-memory similarity index,
   which only the Flask process loads, so the tick runs it inline right there. It
   still gets a task row (STARTED, then SUCCESS or FAILURE) and so stays visible
   in the task panel; the cost is that the poll thread waits for the run, which is
   the accepted trade for a schedule that fires once a day.
5. **Conflict check.** For analysis and clustering, a row is skipped when a task
   of that type is already active, so a schedule cannot pile runs on top of each
   other.
6. **Error isolation.** An exception on one row is logged and the loop continues
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
- `TZ`: the timezone the expressions are evaluated in.
