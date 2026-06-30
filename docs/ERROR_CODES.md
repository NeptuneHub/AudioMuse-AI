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
| 3000–3099 | Index / Voyager |
| 4000–4099 | Database |
| 4100–4199 | Backup / Restore |
| 5000–5099 | Lyrics / Translation |
| 6000–6099 | Task Operations (clustering, cleaning, collection) |
| 9000–9999 | Generic / Unknown |

## Errors that actually fire (wired)

| Code | Class | Fires when… | Where | How to handle |
|------|-------|-------------|-------|---------------|
| 1101 | Music Server Connection Error | Setup "Test connection" can't reach the server / returns nothing; also analysis network failures classified as `HTTPError` / `MaxRetryError` / `LyrionAPIError` | [app_setup.py](../app_setup.py), [error/error_manager.py](../error/error_manager.py) | Check the server URL is correct and reachable from the container; confirm the server is running and the network/DNS path is open. |
| 1102 | Music Server Connection Error | A task throws `ConnectionError` / `ConnectionRefusedError` / `NewConnectionError` (server down / refused) | classify map → analysis / clustering / cleaning / collection excepts | Server is down or refusing connections - start it, verify the port, check firewall rules. |
| 1103 | Music Server Connection Error | A task throws `ReadTimeout` / `ConnectTimeout` / `Timeout` (#523 slow server) | classify map ([error/error_manager.py](../error/error_manager.py)) | Server is too slow to respond; reduce load, raise client timeouts, or improve the network path. |
| 1105 | Music Server Library Error | Analysis runs but the server returns 0 tracks for every album (#552) | [tasks/analysis.py](../tasks/analysis.py) (no-tracks check) | Verify the library actually contains scannable music and that the configured user/library has read access to the tracks. |
| 2001 | Analysis Error | Main analysis task fails for any non-classified reason | [tasks/analysis.py](../tasks/analysis.py) main except | Inspect the container log for the real cause; this is the catch-all for the analysis run. |
| 2002 | Analysis Error | A per-album analysis task fails | [tasks/analysis.py](../tasks/analysis.py) album except | One album failed; check the log for the album/track and re-run analysis. |
| 3001 | Index Error | Final Voyager index rebuild fails (non-empty) | [tasks/analysis.py](../tasks/analysis.py) index wrap | Inspect the log for the rebuild failure; verify disk space and that embeddings exist. |
| 3002 | Index Error | Final index rebuild raises `EmptyIndexError` | [tasks/analysis.py](../tasks/analysis.py) index wrap | Nothing was indexed - confirm analysis produced embeddings before the index step ran. |
| 4001 | Database Error | `OperationalError` in analysis / album / cleaning (DB down mid-task) | classify map + `OperationalError` branches ([tasks/analysis.py](../tasks/analysis.py), [tasks/cleaning.py](../tasks/cleaning.py)) | PostgreSQL is unreachable or dropped the connection - confirm the DB is up, credentials are valid, and the connection pool isn't exhausted. |
| 4101 | Backup Error | `pg_dump` reports a server version mismatch (#540) | [app_backup.py](../app_backup.py) | Match the `pg_dump` client version to the PostgreSQL server version. |
| 4102 | Backup Error | `pg_dump` exits non-zero, is not installed, or timed out (600 s) | [app_backup.py](../app_backup.py) | Ensure `pg_dump` is installed and on PATH, the DB is reachable, and the dump fits the timeout. |
| 6001 | Clustering Error | A clustering batch / main task fails | [tasks/clustering.py](../tasks/clustering.py), [app_clustering.py](../app_clustering.py) | Check the log for the clustering failure; verify embeddings/index are present and parameters are valid. |
| 6002 | Cleaning Error | The cleaning task fails | [tasks/cleaning.py](../tasks/cleaning.py) | Check the log; if it was a DB outage it surfaces as 4001 instead. |
| 6003 | Collection Sync Error | A collection sync task fails | [tasks/collection_manager.py](../tasks/collection_manager.py), [app_collection.py](../app_collection.py) | Check the log; connection failures to the media server surface as 1102/1103 instead. |
| 9999 | Unknown Error | Any failed task that didn't record a structured error (legacy / un-migrated jobs) | [app.py](../app.py) `/api/status` fallback | Open the container log - the generic message intentionally hides specifics from the frontend. Migrate the call site to record a structured code. |

## Errors that are defined but not yet wired

These codes exist in the registry (so `build`/`record` and the frontend handle them
correctly) but no call site raises them yet. They are reserved for future use.

| Code | Class | Reserved for |
|------|-------|--------------|
| 1001 | Configuration Error | Invalid application configuration |
| 1002 | Configuration Error | Missing required media server credentials |
| 1003 | Startup Error | Application failed to start |
| 1104 | Music Server Authentication Error | Media server rejected the provided credentials |
| 2003 | Analysis Error | No albums available to analyze |
| 2004 | Model Inference Error | An analysis model failed to produce a result |
| 4002 | Database Error | A database query failed (vs. a connection failure, 4001) |
| 4103 | Restore Error | Database restore failed |
| 5001 | Lyrics Error | Lyrics could not be retrieved |
| 5002 | Lyrics Transcription Error | Lyrics transcription (ASR) failed |
| 5003 | Translation Error | Lyrics translation failed |

## Exception → code classification

`error_manager.classify(exc, default_code)` maps an exception by its type name to a
code, falling back to `default_code`. This is what turns raw network/DB exceptions
into the connection/database codes above.

| Exception type name | Code |
|---------------------|------|
| `ConnectionError`, `ConnectionRefusedError`, `NewConnectionError` | 1102 |
| `MaxRetryError`, `HTTPError`, `LyrionAPIError` | 1101 |
| `ConnectTimeout`, `ConnectTimeoutError`, `ReadTimeout`, `ReadTimeoutError`, `Timeout`, `timeout` | 1103 |
| `OperationalError` | 4001 |
| `EmptyIndexError` | 3002 |
| anything else | the caller's `default_code` (often the domain code, e.g. 2001 / 6001 / 6003) |

An `AudioMuseError` always keeps its own code regardless of the classify map.

## HTTP status for synchronous routes

`error_manager.http_status_for_code(code)` decides the HTTP status a synchronous
route returns when it raises an `AudioMuseError`:

| Code range | HTTP status |
|------------|-------------|
| 1100–1199 (music server connection) | 502 Bad Gateway |
| 1000–1099 (configuration / setup) | 400 Bad Request |
| 4000–4099 (database) | 503 Service Unavailable |
| everything else | 500 Internal Server Error |

## How an error flows

- **Synchronous routes** raise `AudioMuseError(code, message)`. The global handler in
  [app.py](../app.py) renders `to_dict()` as JSON with the mapped HTTP status.
- **Background tasks** catch their exception, call
  `error_manager.record(classify(e, <domain code>), str(e), exc=e)` (which logs the
  full traceback and returns the structured dict), and store that dict on the job's
  `details.error`.
- **`/api/status`** returns the stored structured error. If a job is `FAILED`/`FAILURE`
  but has no structured `error` recorded (legacy/un-migrated jobs), it backfills `9999`
  so the frontend always receives a well-formed error object.
- The traceback is **never** placed in the returned dict - it lives only in the
  container log.

## Adding a new error code

1. Add the constant and a `{error_class, default_message}` entry in
   [error/error_dictionary.py](../error/error_dictionary.py), keeping it inside the
   right numeric range.
2. If it should be derived from an exception type, add the mapping to
   `_EXCEPTION_NAME_CODES` in [error/error_manager.py](../error/error_manager.py).
3. Raise it (`AudioMuseError`) in synchronous code, or record it
   (`error_manager.record` / `from_exception`) in a task.
4. Add a row to the wired table above (move it out of the "not yet wired" table).
