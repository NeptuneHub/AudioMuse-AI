# Index

- [Lyrics Workflow](#lyrics-workflow)
  - [Pipeline steps](#pipeline-steps)
  - [What `_apply_vad` means (VAD)](#what-_apply_vad-means-vad)
  - [What `_sanitize_lyrics_text` means (sanitize)](#what-_sanitize_lyrics_text-means-sanitize)
  - [What `_resolve_lang_and_quality` means](#what-_resolve_lang_and_quality-means)
  - [What `_text_quality_reject` means](#what-_text_quality_reject-means)
  - [What `_asr_should_drop` means (reliability gate & asymmetries)](#what-_asr_should_drop-means-reliability-gate--asymmetries)
  - [Restarting the lyrics analysis](#restarting-the-lyrics-analysis)
- [Clustering Workflow](#clustering-workflow)
  - [Pipeline steps](#pipeline-steps-1)
  - [What the feature vector means](#what-the-feature-vector-means)
  - [What stratified sampling means](#what-stratified-sampling-means)
  - [What the evolutionary search means (explore vs exploit)](#what-the-evolutionary-search-means-explore-vs-exploit)
  - [What a single iteration does](#what-a-single-iteration-does)
  - [What the fitness score means (the seven metrics)](#what-the-fitness-score-means-the-seven-metrics)
  - [What cluster-to-playlist filtering means](#what-cluster-to-playlist-filtering-means)
  - [What batch orchestration means (concurrency, timeouts, recovery)](#what-batch-orchestration-means-concurrency-timeouts-recovery)
  - [What post-processing means (the finalization pipeline)](#what-post-processing-means-the-finalization-pipeline)
  - [Restarting the clustering](#restarting-the-clustering)

# Lyrics Workflow

The lyrics pipeline turns a track into a multilingual text embedding (plus axis
scores), or falls back to an instrumental sentinel when no usable lyrics exist.
Lyrics are preferred from text sources (media server, external API); only if
those miss does the track go through Whisper-small ASR on the audio. The
embedding model (`gte-multilingual-base`) is language-agnostic, so there is no
translation step - language detection is used only for metadata and as a
quality gate.

## Pipeline steps

| # | Step | Control made |
|---|------|--------------|
| 1 | musicnn instrumental check | If musicnn flagged the track as instrumental → skip everything, emit instrumental sentinel |
| 2 | Media-server lyrics fetch | Fetch by `track_id`; [`_sanitize_lyrics_text`](#what-_sanitize_lyrics_text-means-sanitize); if non-empty text → use it and skip STEPS 3-5 |
| 3 | External lyrics API | If enabled + artist/track present and media-server missed → fetch & [`_sanitize_lyrics_text`](#what-_sanitize_lyrics_text-means-sanitize); HIT skips STEPS 4-5; MISS falls back to ASR |
| 4 | Audio prep | Trim/load audio up to `MAX_AUDIO_SECONDS` (240s) for ASR |
| 4b | VAD (Silero ONNX) | [`_apply_vad`](#what-_apply_vad-means-vad): keep only voiced audio; too little voice → instrumental (unless musicnn flagged a vocalist) |
| 5 | Whisper-small ASR | Transcribe (300s timeout → empty); [`_sanitize_lyrics_text`](#what-_sanitize_lyrics_text-means-sanitize); record `asr_lang`, `avg_logprob`; [`_resolve_lang_and_quality`](#what-_resolve_lang_and_quality-means) → drop to instrumental if junk |
| 6 | langdetect (text lyrics only, `whisper_raw_len==0`) | `detect_langs` → lang + confidence; [`_resolve_lang_and_quality`](#what-_resolve_lang_and_quality-means); if no CJK and **conf < 0.70 → drop** (the text-path reliability gate - see [`_asr_should_drop`](#what-_asr_should_drop-means-reliability-gate--asymmetries)) |
| 7 | ASR reliability gate | [`_asr_should_drop`](#what-_asr_should_drop-means-reliability-gate--asymmetries) (low logprob / null lang) → drop. Language was already resolved in STEP 5 |
| 8 | Final text gate | [`_text_quality_reject`](#what-_text_quality_reject-means) on final text (with the resolved language) → drop to instrumental if junk |
| 9 | Embedding + axis scoring | If `len ≥ MIN_CHARS_FOR_EMBEDDING` (250) → embed (gte-multilingual) + score axes; else / empty embedding → instrumental sentinel |

STEPS 4-5 (audio + VAD + ASR) only run when no text lyrics were found in STEP 2
or 3. Each chapter below explains one function from the table.

## What `_apply_vad` means (VAD)

VAD (Voice Activity Detection) runs the Silero ONNX model over the prepared
audio to find the segments that actually contain a voice, before sending
anything to ASR. Functionally:

- **Detect speech** - scans the clip and returns voiced timestamps using
  `LYRICS_VAD_THRESHOLD` (0.2). If nothing is found, it retries once at a lower
  floor (`LYRICS_VAD_RETRY_FLOOR`, 0.15).
- **No speech at all** - if even the retry finds nothing, it falls back to
  sending the *full* clip to ASR (rather than dropping the track outright).
- **Too little voice** - if voiced audio is below `VAD_VOICE_RECOGNITION`
  seconds, the track is treated as instrumental - **unless** musicnn already
  flagged a vocalist (`vocal_prior`), in which case the gate is bypassed and the
  full clip is sent to ASR anyway.
- **Enough voice** - keeps only the concatenated voiced segments and passes them
  to Whisper, so ASR isn't fed long instrumental stretches.

This both improves transcription quality and filters instrumentals before the
expensive ASR step.

## What `_sanitize_lyrics_text` means (sanitize)

Sanitizing is applied to **every text source** - media server, lyrics API, and
the Whisper ASR transcript - stripping everything that is not actual sung
content, so the embedding sees clean lyrics instead of formatting noise.
Functionally it removes:

- **Invisible / control characters** - BOM, zero-width spaces, and ASCII
  control codes that carry no meaning but distort the text.
- **Non-text symbols** - emoji and the various pictographic / decorative
  Unicode blocks (arrows, boxes, dingbats, regional-indicator flags, etc.).
- **Embedded markup** - `<script>`/`<style>` blocks and any other HTML-like
  tags, which appear when an API leaks a web page instead of plain lyrics.
- **LRC timing data** - inline `[mm:ss.xx]` timestamps and full LRC metadata
  lines (`[ar:]`, `[ti:]`, `[al:]`, `[length:]`, `[offset:]`, …).
- **Structural section headers** - standalone lines like *Chorus*, *Verse 2*,
  *Bridge*, *Intro*, *Hook*, *Outro*, etc., which describe structure rather than
  being lyrics.
- **Excess blank lines** - collapses runs of empty lines down to a single
  separator.
- **Runaway length** - truncates to a maximum word count (300 words) so a
  pathologically long blob can't dominate.

The result is trimmed plain text. If sanitizing leaves nothing, the source is
treated as a miss.

## What `_resolve_lang_and_quality` means

This is the **shared** language + content control, called identically from both
STEP 5 (ASR) and STEP 6 (text). Given the text and a *candidate* language
(Whisper's `asr_lang` for ASR, langdetect's result for text), it does two things
in order:

1. **CJK-script override** - if the text contains enough Hangul / kana / Han
   characters (≥ `LYRICS_CJK_SCRIPT_MIN_RATIO`, 0.10 of letters), it forces the
   language to `ko` / `ja` / `zh` regardless of the candidate. Script presence is
   a far more reliable CJK signal than either detector.
2. **Content quality reject** - runs [`_text_quality_reject`](#what-_text_quality_reject-means)
   on the resolved language and returns its verdict.

Because both paths call this one function, the CJK override and the content
checks are guaranteed to be identical no matter where the lyrics came from. It
deliberately does **not** include the reliability gate - that signal is
source-specific and lives outside this function (see
[`_asr_should_drop`](#what-_asr_should_drop-means-reliability-gate--asymmetries)).

## What `_text_quality_reject` means

After text exists (whether from an API or from ASR), this gate decides whether
the text is *good enough* to embed. It returns a reason string when the text
should be dropped to the instrumental sentinel, or nothing when the text is
accepted. The functional checks are:

- **Too short** - fewer than `MIN_CHARS_FOR_EMBEDDING` (250) characters. Short
  fragments don't carry enough signal for a meaningful embedding.
- **Too repetitive** - the zlib compression ratio of the text exceeds the
  threshold (`LYRICS_TEXT_MAX_COMPRESSION_RATIO`, default 15). A very high ratio
  means the text is mostly the same line repeated (ad-lib spam, "la la la"
  loops, ASR hallucination), so it is rejected while genuinely chorus-heavy
  songs still pass.
- **Script/language mismatch** - when the resolved language is a non-Latin-script
  language (Korean, Japanese, Chinese, Arabic, Russian, Thai, Hindi, etc.) but
  the text is ≥90% Latin characters, the content is inconsistent with the claimed
  language (garbled/mojibake or wrong text) and is rejected.

If none of these fire, the text is kept and proceeds to embedding.

## What `_asr_should_drop` means (reliability gate & asymmetries)

The **reliability gate** is a separate signal from the content checks in
`_resolve_lang_and_quality`. It exists once per source, in different steps, and
is the one control that is intentionally **not** symmetric - because *a low
confidence score does not mean the same thing on each source*:

- **Text path (STEP 6) - the language-confidence gate.** The inline
  `conf < LANG_CONFIDENCE_MIN` (0.70) drop. langdetect only *classifies* text
  that already exists; it doesn't produce it. So low confidence does **not** prove
  the text is bad - it may simply be a language langdetect handles poorly or
  doesn't support. This makes it a **weak** signal: it catches mojibake/garbage,
  but it can also wrongly flag valid lyrics in an under-supported language.
- **ASR path (STEP 7) - `_asr_should_drop`.** Whisper *generates* the text from
  audio, so its low confidence directly means the produced transcript is
  **wrong** (a hallucination). This is a **strong, trustworthy** "bad content"
  signal. It drops the transcript when:
  - logprob `< -1.0` (`LYRICS_ASR_MIN_AVG_LOGPROB`) - universal hallucination
    floor, every language;
  - `asr_lang` is null / unknown;
  - the transcript is **non-English** *and* logprob `< -0.85`
    (`LYRICS_ASR_NON_ENGLISH_MIN_LOGPROB`) - a stricter floor than English faces.

### Asymmetry 1 - CJK bypasses the text gate, but not the ASR gate

The text-path confidence gate sits *after* the CJK branch, so when CJK script is
detected it is **skipped**. `_asr_should_drop` has no CJK branch and runs
**unconditionally**. The reason is what a low score actually proves on each
source:

- **API / music server:** low langdetect confidence is unreliable - it may just
  mean the language is under-supported (Latin-script bias, code-mixed
  K-pop / J-pop), not that the text is junk. So dropping on it risks throwing
  away valid lyrics. The CJK bypass exists because CJK is the most common victim:
  the presence of Hangul / kana / Han proves the text *is* genuine CJK, so we
  ignore langdetect's untrustworthy low score. (Other under-supported languages
  with no script test can still be wrongly dropped - a known limitation of this
  gate.)
- **ASR:** low logprob is a *trustworthy* "this is wrong" signal, because Whisper
  produced the text - there is nothing to forgive. And CJK characters can't
  rescue it, since Whisper may have **hallucinated** them. So the gate runs
  regardless of script.

On both paths CJK still passes through the content checks
([`_resolve_lang_and_quality`](#what-_resolve_lang_and_quality-means)); the only
thing CJK ever bypasses is the API-path confidence drop.

### Asymmetry 2 - stricter non-English bar on ASR

The extra `-0.85` floor for non-English ASR is deliberate: it is the same
"is the transcription real?" idea, **calibrated per language**. Whisper-small is
less reliable on non-English audio, so a medium-confidence non-English transcript
is more likely to be a hallucination than an English one at the same score, and
the stricter floor demands higher confidence before trusting it. Trade-off: a
genuine non-English song scoring between `-1.0` and `-0.85` (which an English
song would survive) is dropped to instrumental. This only affects the ASR path.

## Restarting the lyrics analysis

To re-run lyrics analysis from scratch, drop the three lyrics tables. On the
next analysis run they are recreated and every track is reprocessed through the
pipeline above:

```sql
DROP TABLE IF EXISTS lyrics_embedding;
DROP TABLE IF EXISTS lyrics_index_data;
DROP TABLE IF EXISTS lyrics_axes_index_data;
```

- `lyrics_embedding` - per-track lyrics text, language, and embedding vector.
- `lyrics_index_data` - the semantic similarity index built from those embeddings.
- `lyrics_axes_index_data` - the axis-score index used for axis-based search.

Dropping these affects lyrics only; audio/musicnn analysis is untouched.

# Clustering Workflow

The clustering pipeline turns the analyzed library into a set of automatic
playlists. It does **not** run one clustering pass - it runs an *evolutionary
search* over clustering configurations: hundreds or thousands of independent
iterations, each clustering a stratified sample of the library with slightly
different parameters, each scored by a single weighted fitness number. The
best-scoring iteration wins, and its clusters are post-processed into the
playlists that are actually created.

There are three layers:

- **Orchestrator** ([`run_clustering_task`](../tasks/clustering.py)) - prepares
  the data, splits the requested number of runs into batch jobs, monitors them,
  then finalizes the single best result into playlists.
- **Batch worker** ([`run_clustering_batch_task`](../tasks/clustering.py)) - an
  RQ job that runs a fixed number of iterations and reports back its best one.
- **Iteration** ([`_perform_single_clustering_iteration`](../tasks/clustering_helper.py))
  - one clustering attempt: sample → scale → pick parameters → (PCA) → cluster →
  filter → score.

Input comes from the `score` table (per-track `tempo`, `energy`, `mood_vector`,
`other_features`, `author`) and, when embedding clustering is enabled, the
`embedding` table. Output is a set of media-server playlists plus the
`playlist` table.

## Pipeline steps

| # | Step | What happens |
|---|------|--------------|
| 1 | Load lightweight data | Fetch `item_id, author, mood_vector` for every track with a non-empty `mood_vector`; abort if fewer tracks than the minimum cluster count |
| 2 | Build genre map + targets | [Stratify](#what-stratified-sampling-means): bucket tracks by predominant `STRATIFIED_GENRES` mood; compute `target_songs_per_genre` from a percentile of bucket sizes |
| 3 | Plan batches | Split `num_clustering_runs` into batches of `ITERATIONS_PER_BATCH_JOB` (20); recover any child tasks already in the DB |
| 4 | Run iterations (per batch) | Each iteration: re-sample the subset, pick parameters [evolutionarily](#what-the-evolutionary-search-means-explore-vs-exploit), [cluster + score](#what-a-single-iteration-does); keep the batch's best |
| 5 | Monitor & aggregate | [Orchestrate batches](#what-batch-orchestration-means-concurrency-timeouts-recovery) up to `MAX_CONCURRENT_BATCH_JOBS` (10); fold each batch's best into the global best + the elite pool; timeout/staleness watchdogs prevent hangs |
| 6 | Post-process winner | On the global best: [duplicate filter → min-size filter → Top-N diverse selection](#what-post-processing-means-the-finalization-pipeline) |
| 7 | Name + create | AI-name each surviving cluster, Fisher-Yates shuffle, chunk by `MAX_SONGS_PER_CLUSTER`, delete old `_automatic` playlists, create the new ones |

## What the feature vector means

Every track is reduced to one numeric vector by
[`score_vector`](../tasks/commons.py). The layout is fixed and every later step
indexes into it positionally:

```
[ tempo_norm, energy_norm, mood_0 … mood_n, other_0 … other_5 ]
   index 0      index 1     index 2 …          index 2+len(moods) …
```

- **tempo** / **energy** - normalized to 0–1 against `TEMPO_MIN/MAX_BPM`
  (40–200) and `ENERGY_MIN/MAX` (0.01–0.15), then clipped.
- **moods** - one slot per active mood label (the top-N moods, controlled by the
  `top_n_moods` parameter), filled from the track's `mood_vector` string.
- **other features** - the six `OTHER_FEATURE_LABELS` (`danceable`,
  `aggressive`, `happy`, `party`, `relaxed`, `sad`).

This feature vector is always what *names* and *scores* a cluster. What gets
*clustered* is either this same vector (default) or the track's raw semantic
embedding when `enable_clustering_embeddings` is on - in that case the feature
vector is still used afterwards to label and score the resulting clusters.

## What stratified sampling means

A single iteration does not cluster the whole library - it clusters a
**representative subset**, so that thousands of iterations stay fast and each
sees a balanced cross-section.

- **Genre buckets** - each track is assigned a single predominant genre by
  taking the highest-scoring label among `STRATIFIED_GENRES` in its
  `mood_vector` (everything else falls into `__other__`).
- **Per-genre target** - `target_songs_per_genre` is the
  `stratified_sampling_target_percentile` percentile of the genre bucket sizes,
  floored at `min_songs_per_genre_for_stratification`. This is what keeps a huge
  genre from swamping a small one.
- **Sampling** - [`_get_stratified_song_subset`](../tasks/clustering_helper.py)
  draws up to the target from each genre.
- **Perturbation between iterations** - iteration 0 of a batch reuses the
  previous subset; every later iteration *churns* it by
  `SAMPLING_PERCENTAGE_CHANGE_PER_RUN` (0.2) - keep 80%, redraw 20%. This is the
  stochasticity that lets the search see different slices of the library over
  time.

## What the evolutionary search means (explore vs exploit)

The search has no gradient; it explores the parameter space and keeps what
works. Each iteration's parameters come from
[`_generate_evolutionary_parameters`](../tasks/clustering_helper.py), which
chooses one of two modes:

- **Explore (random)** - generate a fresh random parameter set within the
  configured ranges (PCA components, cluster count / DBSCAN `eps` & `min_samples`
  / GMM components / spectral clusters, depending on the method).
- **Exploit (mutate an elite)** - take one of the best solutions found so far
  and apply small random deltas (`MUTATION_INT_ABS_DELTA` 3,
  `MUTATION_FLOAT_ABS_DELTA` 0.05).

The switch between them:

- Exploitation is **off** for the first `EXPLOITATION_START_FRACTION` (0.2) of
  all runs - the search explores broadly before it has anything worth refining.
- After that, each iteration exploits with probability
  `EXPLOITATION_PROBABILITY_CONFIG` (0.7), otherwise still explores.
- The **elite pool** is the top `TOP_N_ELITES` (10) scoring parameter sets seen
  across all batches. The orchestrator passes the current elites into each new
  batch, so improvements propagate as the run progresses.

## What a single iteration does

[`_perform_single_clustering_iteration`](../tasks/clustering_helper.py) is the
unit of work. In order:

1. **Fetch + vectorize** - load full track data for the subset and build the
   [feature vectors](#what-the-feature-vector-means) (and embeddings, if
   enabled). Tracks with missing/garbled data are dropped.
2. **Scale** - `StandardScaler` on whichever matrix will be clustered
   (embeddings or features).
3. **Pick parameters** - [explore or exploit](#what-the-evolutionary-search-means-explore-vs-exploit).
4. **PCA** (optional) - if the chosen parameters enable it, reduce
   dimensionality before clustering; the actual component count is recorded.
5. **Cluster** - fit the chosen model in
   [`_apply_clustering_model`](../tasks/clustering_helper.py): **KMeans**,
   **DBSCAN**, **GMM** (`GMM_COVARIANCE_TYPE`, `reg_covar=1e-4`), or **Spectral**
   (`affinity='nearest_neighbors'`, `SPECTRAL_N_NEIGHBORS`). Degenerate
   configurations (e.g. `k < 2`, or `k ≥ sample size`) are rejected with a
   `fitness_score` of `-1.0`. GPU models are used when `USE_GPU_CLUSTERING` is on
   and the GPU module is available, with automatic CPU fallback.
6. **Filter + score** - turn clusters into [candidate playlists](#what-cluster-to-playlist-filtering-means)
   and compute the [fitness score](#what-the-fitness-score-means-the-seven-metrics).

The return value carries the fitness score, the named playlists, per-cluster
centroids (both the feature-space details used for naming and the
clustered-space vector used for Top-N diversity), and the parameters that
produced them.

## What the fitness score means (the seven metrics)

Each iteration is reduced to one number: a weighted sum of seven metrics, with
weights supplied by the user. A metric is only computed when its weight is
non-zero. The three structural metrics are all rescaled so **higher is always
better**:

- **silhouette** - `(silhouette_score + 1) / 2`, mapped to 0–1.
- **davies_bouldin** - `1 / (1 + davies_bouldin_score)`; Davies-Bouldin is
  lower-is-better, so this inverts it.
- **calinski_harabasz** - `1 - exp(-CH / 500)`, a saturating 0–1 squash.

These three need ≥ 2 clusters and fewer clusters than samples, or they stay 0.

The four **content** metrics describe how musically coherent the playlists are.
Each is computed as a raw sum, passed through `log1p`, then **z-normalized**
against precomputed corpus statistics (mean/sd) so the four are comparable
before weighting - and there are *separate* stats for embedding-based vs
feature-based clustering (`LN_*_EMBEDING_STATS` vs `LN_*_STATS`):

- **mood_diversity** - sums the predominant-mood score of each distinct playlist
  mood; rewards a set of playlists that *between them* span many moods.
- **mood_purity** - within each playlist, how strongly its songs actually carry
  the playlist's top `TOP_K_MOODS_FOR_PURITY_CALCULATION` (3) moods; rewards
  internally consistent playlists.
- **other_feature_diversity** / **other_feature_purity** - the same two ideas
  applied to the six "other features", gated by
  `OTHER_FEATURE_PREDOMINANCE_THRESHOLD_FOR_PURITY` (0.3) so only features a
  cluster genuinely leans into count.

`final_score = Σ weightₖ · metricₖ`. Diversity and purity pull against each
other (more, narrower playlists vs fewer, broader ones), and the weights are how
the user tunes that trade-off.

## What cluster-to-playlist filtering means

Raw cluster membership is not used directly; each cluster is trimmed into a
candidate playlist inside
[`_format_and_score_iteration_result`](../tasks/clustering_helper.py):

- **Distance gate** - every point's distance to its cluster center is normalized
  to 0–1; members beyond `MAX_DISTANCE` (0.5) are dropped, so loose outliers
  don't dilute a playlist. DBSCAN noise (label `-1`) is excluded outright.
- **Closest-first** - surviving members are sorted by distance to the center,
  so the most representative tracks are kept first.
- **Per-artist cap** - at most `MAX_SONGS_PER_ARTIST` (3) songs per artist
  (case-insensitive author key); set ≤ 0 to disable. Consistent with the path
  and voyager managers.
- **Per-cluster cap** - at most `max_songs_per_cluster` songs (0 = unlimited).
- **Naming** - [`_name_cluster`](../tasks/clustering_helper.py) inverts the
  centroid back to feature space and builds a name from the tempo band
  (Slow/Medium/Fast), the top moods, and any strongly-present other features
  (e.g. `Happy_Party_Fast`). When clustering on embeddings, the name is derived
  from the cluster's mean *feature* vector instead.

## What batch orchestration means (concurrency, timeouts, recovery)

Iterations are expensive and run as RQ jobs, so the orchestrator manages them
defensively - the overriding goal is that the task **always finishes**, even if
individual batches die.

- **Batching** - `num_clustering_runs` is split into batches of
  `ITERATIONS_PER_BATCH_JOB` (20). Up to `MAX_CONCURRENT_BATCH_JOBS` (10) run at
  once.
- **Aggregation** - [`_monitor_and_process_batches`](../tasks/clustering.py)
  collects each finished batch's best result, updates the global best, and feeds
  the elite pool (pruned to `TOP_N_ELITES`).
- **Per-batch timeout** - a batch running longer than
  `CLUSTERING_BATCH_TIMEOUT_MINUTES` (60) is declared failed, its runs are
  counted as done anyway (so the total can complete), and it's cleared from the
  active set.
- **Failure ceiling** - once `CLUSTERING_MAX_FAILED_BATCHES` (10) batches have
  failed, no new batches launch and the remaining runs are force-completed.
- **Staleness watchdog** - if `runs_completed` doesn't advance for
  `CLUSTERING_BATCH_TIMEOUT_MINUTES`, the task force-completes with the best
  result found so far, rather than hanging near the end.
- **State recovery / idempotency** - on restart the task reloads child tasks
  from the DB and resumes; a task already in a terminal state is skipped.

If no valid solution was found across every run, finalization raises rather than
creating empty playlists.

## What post-processing means (the finalization pipeline)

The single winning result is cleaned up before any playlist is created
([`tasks/clustering_postprocessing.py`](../tasks/clustering_postprocessing.py)),
in order:

1. **Duplicate filtering**
   ([`apply_duplicate_filtering_to_clustering_result`](../tasks/clustering_postprocessing.py))
   - within each playlist: sort by title (so near-identical titles are
   adjacent), drop exact title/artist duplicates (normalizing away
   *(Remastered)*, *[Explicit]*, *- Radio Edit*, etc.), then drop songs whose
   embedding distance to a recent neighbor is below the duplicate threshold,
   using the same metric and thresholds as the voyager manager
   (`DUPLICATE_DISTANCE_CHECK_LOOKBACK`, default lookback 1). Vectors are read
   straight from the `embedding` table; if none exist it falls back to
   title/artist matching only. The playlist is then shuffled.
2. **Minimum-size filter**
   ([`apply_minimum_size_filter_to_clustering_result`](../tasks/clustering_postprocessing.py))
   - drop any playlist with fewer than `MIN_PLAYLIST_SIZE_FOR_TOP_N` (20) songs.
3. **Top-N diverse selection**
   ([`select_top_n_diverse_playlists`](../tasks/clustering_postprocessing.py)) -
   only if more playlists survive than `top_n_playlists`. Greedy max-min
   selection in centroid space: start from the largest playlist, then repeatedly
   add the candidate maximizing a combined score of *distance to the
   already-selected set* (diversity) × *log size* (usefulness). This keeps a
   spread-out, non-redundant set rather than many near-identical playlists.

After post-processing, [`_name_and_prepare_playlists`](../tasks/clustering.py)
AI-names the survivors (Ollama / OpenAI / Gemini / Mistral, falling back to the
generated name on error), a final Fisher-Yates shuffle randomizes order, and
playlists larger than `MAX_SONGS_PER_CLUSTER` are split into numbered chunks.
Existing `_automatic` playlists are deleted and the new ones created on the media
server and recorded in the `playlist` table.

## Restarting the clustering

Clustering is **idempotent at the output level**: every run begins by deleting
the existing `_automatic` playlists and ends by recreating them, so re-running
simply replaces the previous set - there are no clustering tables to drop. To
get a fresh result, just start a new clustering task (optionally with different
score weights, method, or run count).

Clustering only *reads* the analysis tables (`score`, `embedding`); it never
modifies them, so re-running clustering never requires re-analyzing audio or
lyrics.
