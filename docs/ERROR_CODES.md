# Error Codes

This document is the recap of every AudioMuse-AI error code: what it means, when
it fires, where in the code it is raised, and how it can be handled.

The error subsystem lives in [error/](../error/):

- [error/error_dictionary.py](../error/error_dictionary.py) - pure data. Every code
  maps to a generic `error_class` label and a `default_message`.
- [error/error_manager.py](../error/error_manager.py) - turns a code (plus an
  optional one-line detail) into the canonical structured error the frontend renders:

  ```json
  {"error_code": 1102, "error_class": "Music Server Connection Error", "error_message": "..."}
  ```

The user-facing `error_message` is always a single line and never carries a stack
trace; the full traceback only ever reaches the container log. Unknown/unhandled
errors collapse to `9999` with a generic "check the container logs" message so no
internal detail leaks to the frontend.

## Numeric ranges

| Range | Domain |
|-------|--------|
| 1000–1099 | Configuration / Setup |
| 1100–1199 | Music Server Connection |
| 2000–2099 | Analysis / Model |
| 3000–3099 | Index / Similarity |
| 4000–4099 | Database |
| 4100–4199 | Backup / Restore |
| 5000–5099 | Lyrics |
| 6000–6099 | Task Operations (clustering, cleaning) |
| 9000–9999 | Generic / Unknown |

## Errors that actually fire (wired)

| Code | Class | Fires when… | Where | How to handle |
|------|-------|-------------|-------|---------------|
| 1002 | Configuration Error | Setup "Test connection" hits a provider `ValueError` for missing credentials (before any network I/O) | [app_setup.py](../app_setup.py) `_test_media_server_connection` | Fill in the missing user/token/URL for the selected provider. |
| 1101 | Music Server Connection Error | Setup "Test connection" can't reach the server / returns nothing; also network failures classified as `HTTPError` / `MaxRetryError` / `RetryError` / `SSLError` / `RequestException` / `LyrionAPIError` | [app_setup.py](../app_setup.py), [error/error_manager.py](../error/error_manager.py) | Check the server URL is correct and reachable from the container; for a TLS failure confirm the certificate; confirm the server is running and the network/DNS path is open. |
| 1102 | Music Server Connection Error | A `requests`/`urllib3` `ConnectionError` / `NewConnectionError` (server down / refused) | classify map → analysis / clustering / cleaning excepts | Server is down or refusing connections - start it, verify the port, check firewall rules. |
| 1103 | Music Server Connection Error | A `requests`/`urllib3` `ReadTimeout` / `ConnectTimeout` / `Timeout`, or a builtin `TimeoutError` (#523 slow server) | classify map ([error/error_manager.py](../error/error_manager.py)) | Server is too slow to respond; reduce load, raise client timeouts, or improve the network path. |
| 1104 | Music Server Authentication Error | A media-server probe fails auth, or any exception in the chain carries an HTTP 401/403 response | [tasks/analysis.py](../tasks/analysis.py), classify auth check ([error/error_manager.py](../error/error_manager.py)) | Wrong credentials - fix the configured user/token; the server accepted the connection but rejected the login. |
| 1105 | Music Server Library Error | Analysis runs but the server returns 0 tracks for every album (#552) | [tasks/analysis.py](../tasks/analysis.py) (no-tracks check) | Verify the library actually contains scannable music and that the configured user/library has read access to the tracks. |
| 2001 | Analysis Error | Main analysis task fails for any non-classified reason | [tasks/analysis.py](../tasks/analysis.py) main except | Inspect the container log for the real cause; this is the catch-all for the analysis run. |
| 2002 | Analysis Error | A per-album analysis task fails for a **real** reason (download failure, DB error, model crash, track-server map flush failure). Tracks that merely hold no analyzable audio are skipped as 2007 and do NOT fail the album | [tasks/analysis.py](../tasks/analysis.py) album except | One album failed; check the log for the album/track. RQ retries it. The parent run reports `failed_albums` and a sample of child errors, but does **not** fail unless *every* album failed (2005). |
| 2004 | Model Inference Error | An `onnxruntime` inference exception (`Fail` / `RuntimeException` / `InvalidArgument`) or a `MemoryError` escapes album analysis | classify map ([error/error_manager.py](../error/error_manager.py)) | Check the model download/integrity, GPU/VRAM headroom, or disable GPU; distinct from an ordinary album failure. |
| 2005 | Analysis Error | An analysis run reaches the end having launched albums but with **every** one of them failed, so not a single song was analyzed | [tasks/analysis.py](../tasks/analysis.py) phase end | The run is systematically broken, not merely hitting bad files: check the media server is reachable, the models loaded, and the DB is writable. |
| 2006 | Analysis Error | A multi-server (union) run finishes with **every** music server failed | [tasks/analysis.py](../tasks/analysis.py) `run_analysis_task` | Named servers all failed; check their connectivity/credentials. If only *some* servers fail the run still succeeds and lists them in `failed_servers`. |
| 2007 | Track Skipped | A single track holds no analyzable audio: a silent hidden track, a corrupt/undecodable file, or an instrumental whose lyrics produced nothing | [tasks/analysis.py](../tasks/analysis.py) `TrackNotAnalyzable` | Informational, logged at WARNING and counted as `tracks_not_analyzable`. **Never fails the album or the run** - a real library always has some of these. |
| 3001 | Index Error | Final index rebuild fails (non-empty) | [tasks/analysis.py](../tasks/analysis.py) index wrap | Inspect the log for the rebuild failure; verify disk space and that embeddings exist. |
| 3002 | Index Error | Final index rebuild raises `EmptyIndexError`; a similarity endpoint hits a not-loaded/empty index | [tasks/analysis.py](../tasks/analysis.py) index wrap, [app_ivf.py](../app_ivf.py), [app_artist_similarity.py](../app_artist_similarity.py) | Nothing was indexed - run analysis so embeddings exist before the similarity search or index step runs. |
| 4001 | Database Error | `OperationalError` in a task or endpoint (DB down / connection dropped) | classify map + `OperationalError` branches ([tasks/analysis.py](../tasks/analysis.py), [tasks/cleaning.py](../tasks/cleaning.py), data/auth endpoints) | PostgreSQL is unreachable or dropped the connection - confirm the DB is up, credentials are valid, and the connection pool isn't exhausted. |
| 4002 | Database Error | A psycopg2 `DatabaseError` subclass (query failure), or the default for a failed DB-backed endpoint ([app_sync.py](../app_sync.py), [app_external.py](../app_external.py), [app_auth.py](../app_auth.py) count/list) | classify map + endpoint defaults | A query failed rather than the connection - inspect the container log for the failing statement. |
| 4101 | Backup Error | `pg_dump` reports a server version mismatch (#540) | [app_backup.py](../app_backup.py) | Match the `pg_dump` client version to the PostgreSQL server version. |
| 4102 | Backup Error | `pg_dump` exits non-zero, is not installed, or timed out (600 s) | [app_backup.py](../app_backup.py) | Ensure `pg_dump` is installed and on PATH, the DB is reachable, and the dump fits the timeout. |
| 4103 | Restore Error | A restore chunk upload fails, the restore runner is missing, or the restore itself fails | [app_backup.py](../app_backup.py) restore path | Check the container log; verify the dump is intact and the PostgreSQL version is compatible (see #702). |
| 5001 | Lyrics Error | An HTTP lyrics endpoint (axis/text search, warmup, cache refresh) fails | [app_lyrics.py](../app_lyrics.py) | Check the log; confirm the lyrics model is available and the DB is reachable. |
| 5002 | Lyrics Transcription Error | The analysis-time lyrics pipeline (ASR transcription + embedding) fails for a track | [tasks/analysis_helper.py](../tasks/analysis_helper.py) `run_lyrics_for_track` | Per-track lyrics failure (skipped, best-effort); check the log for the model/ASR error. |
| 6001 | Clustering Error | A clustering batch / main task fails | [tasks/clustering.py](../tasks/clustering.py), [app_clustering.py](../app_clustering.py) | Check the log for the clustering failure; verify embeddings/index are present and parameters are valid. |
| 6002 | Cleaning Error | The cleaning task fails | [tasks/cleaning.py](../tasks/cleaning.py) | Check the log; if it was a DB outage it surfaces as 4001 instead. |
| 9999 | Unknown Error | Any failed task that didn't record a structured error (legacy / un-migrated jobs), or any otherwise-unhandled route exception | [app.py](../app.py) `/api/status` fallback and the global `errorhandler(Exception)` | Open the container log - the generic message intentionally hides specifics from the frontend. Migrate the call site to record a structured code. |

## Errors that are defined but not yet wired

These codes exist in the registry (so `build`/`record` and the frontend handle them
correctly) but no call site raises them yet. They are reserved for future use.

| Code | Class | Reserved for |
|------|-------|--------------|
| 1001 | Configuration Error | Invalid application configuration |
| 1003 | Startup Error | Application failed to start |
| 2003 | Analysis Error | No albums available to analyze |

## Exception → code classification

`error_manager.classify(exc, default_code)` maps an exception to a code, falling back
to `default_code`. Matching is **module-qualified**: a class name only matches when the
exception is defined under an allowed import path, so unrelated libraries that reuse a
common name (e.g. `redis.exceptions.ConnectionError`, builtin `BrokenPipeError`) do NOT
steal a media-server or database code. `classify` also walks the exception's
`__cause__`/`__context__` chain and returns 1104 when any link carries an HTTP 401/403
`response`.

| Exception (module → name) | Code |
|---------------------------|------|
| any exception whose chain has an HTTP 401/403 `response` | 1104 |
| `requests`/`urllib3` `ConnectionError`, `NewConnectionError` | 1102 |
| `requests`/`urllib3` `ConnectTimeout`, `ReadTimeout`, `Timeout` (+`*Error`); builtin `TimeoutError` | 1103 |
| `requests`/`urllib3` `SSLError`, `MaxRetryError`, `RetryError`, `HTTPError`; `requests.RequestException`; `LyrionAPIError` | 1101 |
| `psycopg2` `OperationalError`, `InterfaceError` | 4001 |
| `psycopg2` `DatabaseError` (query subclasses) | 4002 |
| `onnxruntime` `Fail`, `RuntimeException`, `InvalidArgument`; builtin `MemoryError` | 2004 |
| `EmptyIndexError` | 3002 |
| anything else | the caller's `default_code` (often the domain code, e.g. 2001 / 6001) |

An `AudioMuseError` always keeps its own code regardless of the classify map.

## HTTP status for synchronous routes

`error_manager.http_status_for_code(code)` decides the HTTP status a synchronous
route returns when it raises an `AudioMuseError` (and `error_response` pairs the
structured body with it):

| Code range | HTTP status |
|------------|-------------|
| 1100–1199 (music server connection / auth) | 502 Bad Gateway |
| 1000–1099 (configuration / setup) | 400 Bad Request |
| 3000–3099 (index / similarity) | 503 Service Unavailable |
| 4000–4099 (database) | 503 Service Unavailable |
| everything else | 500 Internal Server Error |

## How an error flows

- **Synchronous routes** raise `AudioMuseError(code, message)`. The global
  `errorhandler(AudioMuseError)` in [app.py](../app.py) renders `to_dict()` (plus a
  legacy `error` alias) as JSON with the mapped HTTP status. Any *other* unhandled
  exception hits the global `errorhandler(Exception)`, which logs the traceback and
  returns the generic `9999` body so a frontend never receives a Flask HTML 500.
- **Non-route endpoints** that already logged the failure build the response with
  `error_manager.error_response(classify(exc, <default>))`, which returns
  `(structured_body_with_error_alias, http_status)` in one call.
- **Background tasks** catch their exception, call
  `error_manager.record(classify(e, <domain code>), str(e))` (which returns the
  structured dict and logs the coded line; the surrounding handler logs the
  traceback), and store that dict on the job's `details.error`.
- **`/api/status`, `/api/last_task`, `/api/active_tasks`** all run the stored details
  through the shared `sanitize_task_details` helper ([app_helper.py](../app_helper.py)):
  it strips the traceback and heavyweight keys, truncates the log, and backfills a
  structured `9999` error on failed jobs that never recorded one.
- The traceback is **never** placed in the returned dict - it lives only in the
  container log.

## Adding a new error code

1. Add the constant and a `{error_class, default_message}` entry in
   [error/error_dictionary.py](../error/error_dictionary.py), keeping it inside the
   right numeric range.
2. If it should be derived from an exception type, add a `(name, module_prefixes, code)`
   rule to `_EXCEPTION_RULES` in [error/error_manager.py](../error/error_manager.py);
   keep `module_prefixes` tight so a name collision in another library cannot match.
3. Raise it (`AudioMuseError`) in synchronous code, record it
   (`error_manager.record` / `from_exception`) in a task, or return
   `error_manager.error_response(code, detail)` from a non-route endpoint.
4. Add a row to the wired table above (move it out of the "not yet wired" table).
