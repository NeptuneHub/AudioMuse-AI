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
| 1000-1099 | Request / Configuration / Setup |
| 1100-1199 | Music Server Connection |
| 1200-1299 | Task Queue (blocked starts) |
| 2000-2099 | Analysis / Model |
| 3000-3099 | Index / Search |
| 4000-4099 | Database |
| 4100-4199 | Backup / Restore |
| 5000-5099 | Lyrics |
| 6000-6099 | Task Operations (clustering, cleaning, migration, sync, plugins, queue) |
| 9000-9999 | Task process / Generic / Unknown |

## Error codes

| Code | Class | Fires when… | Where | How to handle |
|------|-------|-------------|-------|---------------|
| 1001 | Configuration Error | Setup save fails unexpectedly, the chat pipeline has no usable AI provider, or the JWT secret is missing at login | [app_setup.py](../app_setup.py), [app_chat.py](../app_chat.py), [app_auth.py](../app_auth.py) | Re-run the setup wizard and check the AI provider / authentication settings. |
| 1002 | Configuration Error | Setup "Test connection" hits a provider `ValueError` for missing credentials (before any network I/O) | [app_setup.py](../app_setup.py) `_test_media_server_connection` | Fill in the missing user/token/URL for the selected provider. |
| 1003 | Invalid Request | Any route rejects a request as invalid (missing or malformed parameter); also an API 405 | every blueprint via `json_error` | Fix the request; the `error` text names the problem. |
| 1004 | Not Found | The requested item, session, task or route does not exist (also any unknown `/api/` path) | every blueprint via `json_error`, [app.py](../app.py) HTTP handler | Check the id or URL. |
| 1005 | Authentication Required | Login is missing or the credentials are wrong | [app_auth.py](../app_auth.py) | Log in again. |
| 1006 | Forbidden | The action needs an admin (or setup must be completed first) | [app_auth.py](../app_auth.py), [app_music_servers.py](../app_music_servers.py) | Use an admin account. |
| 1007 | Conflict | The request clashes with the current state (a backup or restore already running, a server already registered, a plugin version unavailable) | [app_backup.py](../app_backup.py), [app_provider_migration.py](../app_provider_migration.py), [plugin/blueprint.py](../plugin/blueprint.py) | Wait for the other operation or follow the message. |
| 1008 | Gone | The requested data no longer exists (an old migration without orphan snapshots) | [app_provider_migration.py](../app_provider_migration.py) | Nothing to recover; re-run the operation if needed. |
| 1009 | Payload Too Large | An upload exceeds its size cap (recording clip, request body) | [app_recording_search.py](../app_recording_search.py), [app.py](../app.py) HTTP handler | Upload a smaller file. |
| 1101 | Music Server Connection Error | Setup "Test connection" can't reach the server / returns nothing; also network failures classified as `HTTPError` / `MaxRetryError` / `RetryError` / `SSLError` / `RequestException` / `LyrionAPIError` | [app_setup.py](../app_setup.py), [error/error_manager.py](../error/error_manager.py) | Check the server URL is correct and reachable from the container; for a TLS failure confirm the certificate; confirm the server is running and the network/DNS path is open. |
| 1102 | Music Server Connection Error | A `requests`/`urllib3` `ConnectionError` / `NewConnectionError` (server down / refused) | classify map → analysis / clustering / cleaning excepts | Server is down or refusing connections - start it, verify the port, check firewall rules. |
| 1103 | Music Server Connection Error | A `requests`/`urllib3` `ReadTimeout` / `ConnectTimeout` / `Timeout`, or a builtin `TimeoutError` (#523 slow server) | classify map ([error/error_manager.py](../error/error_manager.py)) | Server is too slow to respond; reduce load, raise client timeouts, or improve the network path. |
| 1104 | Music Server Authentication Error | A media-server probe fails auth, or any exception in the chain carries an HTTP 401/403 response | [tasks/analysis/main.py](../tasks/analysis/main.py), classify auth check ([error/error_manager.py](../error/error_manager.py)) | Wrong credentials - fix the configured user/token; the server accepted the connection but rejected the login. |
| 1105 | Music Server Library Error | Analysis runs but the server returns 0 tracks for every album (#552) | [tasks/analysis/main.py](../tasks/analysis/main.py) (no-tracks check) | Verify the library actually contains scannable music and that the configured user/library has read access to the tracks. |
| 1106 | Music Server Playlist Error | The media server answered a playlist creation without creating one (it returned no playlist id). A playlist route whose exception is not classified answers 9999 instead, since nothing proves the server was at fault | [app_ivf.py](../app_ivf.py) | Check the media server is reachable and the user may create playlists. |
| 2001 | Analysis Error | Main analysis task fails for any non-classified reason | [tasks/analysis/main.py](../tasks/analysis/main.py) main except | Inspect the container log for the real cause; this is the catch-all for the analysis run. |
| 2002 | Analysis Error | A per-album analysis task fails for a **real** reason (download failure, DB error, model crash, track-server map flush failure). Tracks that merely hold no analyzable audio are skipped as 2007 and do NOT fail the album | [tasks/analysis/album.py](../tasks/analysis/album.py) album except | One album failed; check the log for the album/track. The parent run reports `failed_albums` and a sample of child errors, but does **not** fail unless *every* album failed (2005). |
| 2004 | Model Inference Error | An `onnxruntime` exception (`Fail` / `RuntimeException` / `InvalidArgument` / `NoSuchFile` / `InvalidProtobuf` / `NotImplemented`) whose text was **not** recognised as a memory allocation failure; logged per track by MusiCNN and CLAP | classify map ([error/error_manager.py](../error/error_manager.py)), [tasks/analysis/song.py](../tasks/analysis/song.py), [tasks/clap_analyzer.py](../tasks/clap_analyzer.py) | Check the model files are intact and read the model error in the container log; if it names an allocation failure in a spelling 2008 does not list, report it. |
| 2008 | Model Out Of Memory | An `onnxruntime` exception whose text reports an allocation failure (`Failed to allocate memory`, `BFCArena`, `out of memory`, `CUDA_ERROR_OUT_OF_MEMORY`, DirectML `E_OUTOFMEMORY`, `CUBLAS/CUDNN_STATUS_ALLOC_FAILED`, MIOpen `miopenStatusAllocFailed`, `std::bad_alloc`, or `OOM` as a whole word), anywhere in the cause chain. The session is freed and the inference first retries on a fresh CPU session; 2008 is recorded only when that also fails. Host RAM running out (`MemoryError`) never takes the CPU retry | classify ([error/error_manager.py](../error/error_manager.py) `is_out_of_memory`, `is_model_out_of_memory`), [tasks/onnx_utils.py](../tasks/onnx_utils.py) | Free GPU memory (fewer concurrent models, lower batch), give the worker more RAM, or run on CPU. |
| 2005 | Analysis Error | An analysis run reaches the end having launched albums but with **every** one of them failed, so not a single song was analyzed | [tasks/analysis/main.py](../tasks/analysis/main.py) phase end | The run is systematically broken, not merely hitting bad files: check the media server is reachable, the models loaded, and the DB is writable. |
| 2006 | Analysis Error | A multi-server (union) run finishes with **every** music server failed | [tasks/analysis/main.py](../tasks/analysis/main.py) `run_analysis_task` | Named servers all failed; check their connectivity/credentials. If only *some* servers fail the run still succeeds and lists them in `failed_servers`. |
| 2007 | Track Skipped | A single track holds no analyzable audio: a silent hidden track, a corrupt/undecodable file, or an instrumental whose lyrics produced nothing | [tasks/analysis/album.py](../tasks/analysis/album.py) `TrackNotAnalyzable` | Informational, logged at WARNING and counted as `tracks_not_analyzable`. **Never fails the album or the run** - a real library always has some of these. |
| 3001 | Index Error | Final index rebuild fails (non-empty) | [tasks/analysis/index.py](../tasks/analysis/index.py) index wrap | Inspect the log for the rebuild failure; verify disk space and that embeddings exist. |
| 3002 | Index Error | A similarity, CLAP or recording search hits a not-loaded/empty index (or the feature is disabled) | [app_ivf.py](../app_ivf.py), [app_artist_similarity.py](../app_artist_similarity.py), [app_clap_search.py](../app_clap_search.py), [app_recording_search.py](../app_recording_search.py) | Nothing was indexed - run analysis so embeddings exist before the search runs. |
| 3003 | Search Error | A search endpoint fails for an unclassified reason (similarity, artist, path, CLAP, hyperbolic, SemGrove, recording, sonic, alchemy, dashboard browse, external search) | the search blueprints via `json_exception` | Check the container log; a database or model failure surfaces as its own code instead. |
| 3004 | Cache Refresh Error | A warmup, cache refresh or projection rebuild fails (CLAP, lyrics, SemGrove, hyperbolic tree, map, artist projection, recording models) | the same blueprints' warmup/refresh routes | Check the container log; retry the refresh. |
| 4001 | Database Error | `OperationalError` in a task or endpoint (DB down / connection dropped) | classify map + `OperationalError` branches ([tasks/analysis/main.py](../tasks/analysis/main.py), [tasks/cleaning.py](../tasks/cleaning.py), data/auth endpoints) | PostgreSQL is unreachable or dropped the connection - confirm the DB is up, credentials are valid, and the connection pool isn't exhausted. |
| 4002 | Database Error | A psycopg2 `DatabaseError` subclass (query failure), or the default for a failed DB-backed endpoint ([app_sync.py](../app_sync.py), [app_external.py](../app_external.py), [app_auth.py](../app_auth.py) count/list) | classify map + endpoint defaults | A query failed rather than the connection - inspect the container log for the failing statement. |
| 4101 | Backup Error | `pg_dump` reports a server version mismatch (#540) | [app_backup.py](../app_backup.py) | Match the `pg_dump` client version to the PostgreSQL server version. |
| 4102 | Backup Error | `pg_dump` exits non-zero, is not installed, or timed out (3600 s) | [app_backup.py](../app_backup.py) | Ensure `pg_dump` is installed and on PATH, the DB is reachable, and the dump fits the timeout. |
| 4103 | Restore Error | A restore chunk upload fails, the restore runner is missing, or the restore itself fails | [app_backup.py](../app_backup.py) restore path | Check the container log; verify the dump is intact and the PostgreSQL version is compatible (see #702). |
| 5001 | Lyrics Error | An HTTP lyrics endpoint (axis/text search, warmup, cache refresh) fails | [app_lyrics.py](../app_lyrics.py) | Check the log; confirm the lyrics model is available and the DB is reachable. |
| 5002 | Lyrics Transcription Error | The analysis-time lyrics pipeline (ASR transcription + embedding) fails for a track | [tasks/analysis/song.py](../tasks/analysis/song.py) `run_lyrics_for_track` | Per-track lyrics failure (skipped, best-effort); check the log for the model/ASR error. |
| 6001 | Clustering Error | A clustering batch / main task fails | [tasks/clustering.py](../tasks/clustering.py), [app_clustering.py](../app_clustering.py) | Check the log for the clustering failure; verify embeddings/index are present and parameters are valid. |
| 6002 | Cleaning Error | The cleaning task fails | [tasks/cleaning.py](../tasks/cleaning.py) | Check the log; if it was a DB outage it surfaces as 4001 instead. |
| 6003 | Provider Migration Error | A provider migration task (execute, dry run, source refresh, restart resume) fails, or a migration route (album search, album tracks, report) fails | queue ([taskqueue/__init__.py](../taskqueue/__init__.py) `TASK_FUNC_ERROR_CODES`), [app_provider_migration.py](../app_provider_migration.py) | Read the message; a media-server or DB cause surfaces as its own code. |
| 6004 | Server Sync Error | The multi-server alignment sweep fails | queue (`tasks.multiserver_sync.*`) | Check the secondary servers are reachable. |
| 6005 | Sonic Fingerprint Error | The sonic fingerprint task or endpoint fails | queue, [app_sonic_fingerprint.py](../app_sonic_fingerprint.py) | Check the log; confirm analysis ran and the index is loaded. |
| 6006 | Naming Preview Error | The setup-wizard playlist naming preview fails or its status cannot be read | queue, [app_setup.py](../app_setup.py) | Check the AI provider settings and the log. |
| 6007 | Plugin Error | A plugin task, install, uninstall, enable, settings save or apply fails | queue (`plugin.manager.run_plugin_task`), [plugin/blueprint.py](../plugin/blueprint.py) | Check the plugin's log lines in the container log. |
| 6008 | Task Queue Error | A task could not be queued (analysis, clustering, cleaning, alignment, sweep, migration, naming preview) | [app_helper.py](../app_helper.py) `admit_and_enqueue_main_task`, the start routes | Check the database is reachable and retry. |
| 6009 | Task Cancel Error | A cancel could not be fully applied or confirmed (HTTP 503) | [app.py](../app.py) cancel routes | Retry the cancel; recovery tasks may still be active. |
| 1201 | Task In Progress | A manual start (analysis, clustering, cleaning, provider migration, sweep, naming preview) is refused because a queue-guard task (analysis, clustering, cleaning, provider migration, sonic fingerprint or any plugin task) is already live | [app_helper.py](../app_helper.py) `queue_busy_response` / `queue_race_response`, [app_music_servers.py](../app_music_servers.py), [app_provider_migration.py](../app_provider_migration.py), [app_setup.py](../app_setup.py) | Wait for the running task to finish, or let the scheduled retry (up to `CRON_RETRY_MAX_MINUTES`) pick it up. |
| 9001 | Worker Lost | The worker running the task died (reclaimed by maintenance) or this worker stopped the job process itself (restart, cancel, wedged-task nudge) | [taskqueue/maintenance.py](../taskqueue/maintenance.py), [taskqueue/worker.py](../taskqueue/worker.py) | Usually a restart; the task is retried within its budget. The naming preview reports it as "interrupted by a restart". |
| 9002 | Task Interrupted | A task running inside the web process was left RUNNING when that process stopped | [taskqueue/maintenance.py](../taskqueue/maintenance.py) `fail_stale_inline_rows` | Start it again. |
| 9003 | Out Of Memory | The job process was SIGKILLed by the kernel out-of-memory killer, or a builtin `MemoryError` escaped. Where the kernel log (`/dev/kmsg`) is readable from the initial pid namespace (native Linux), the victim pid must be this job's; otherwise (a normal container, or a kernel log that names no victim because the record rotated out or carries a lagging timestamp) the worker's own cgroup `oom_kill` counter must have risen while the job ran, and the summary says the container-wide counter cannot name the victim. macOS has no such counter (9005) and Windows runs jobs inline (only `MemoryError`) | [taskqueue/worker.py](../taskqueue/worker.py) `_killed_child_death`, classify | Give the worker more memory or run fewer memory-heavy jobs at once. |
| 9004 | Process Crashed | The job process died on SIGSEGV / SIGBUS / SIGABRT / SIGFPE (a native crash, most often the model runtime during inference) or SIGILL (an instruction set this CPU lacks) | [taskqueue/worker.py](../taskqueue/worker.py) `_child_death` | **Not** an out-of-memory condition. For SIGILL use the image built for CPUs without AVX2; for a segfault check the container log around the crash. |
| 9005 | Job Process Died | The job process ended without reporting back for any other reason: a non-zero exit, a SIGKILL the kernel did not count as an OOM kill (userspace killers such as systemd-oomd or a Kubernetes eviction, or a manual kill), a SIGKILL while the kernel log shows the out-of-memory killer ended a different process, an unreadable OOM counter, or the worker killing a child whose report pipe broke | [taskqueue/worker.py](../taskqueue/worker.py) | Read the summary: it states which of these happened. |
| 9999 | Unknown Error | A failed task row that carries no structured error (rows written before structured errors), or an unclassified exception in a route with no feature code | [app_helper.py](../app_helper.py) `sanitize_task_details` via `error_manager.task_error_record`, the global `errorhandler(Exception)` | Open the container log - the generic message intentionally hides specifics from the frontend. |

## Exception → code classification

`error_manager.classify(exc, default_code)` maps an exception to a code, falling back
to `default_code`. Matching is **module-qualified**: a class name only matches when the
exception is defined under an allowed import path, so unrelated libraries that reuse a
common name (e.g. `psycopg2.OperationalError`, builtin `BrokenPipeError`) do NOT
steal a media-server or database code. Every check walks the exception chain once,
outermost link first, the way a traceback prints it: the explicit `__cause__`, else the
implicit `__context__` unless the raise suppressed it (`raise ... from None`). A
`TaskFailed` or wrapper raised `from` a database or media-server error keeps that
error's code, and a deliberately detached exception is judged on its own. The checks
run in this order:

| Order | Exception (module → name) | Code |
|-------|---------------------------|------|
| 1 | an `AudioMuseError` | its own code |
| 2 | any link that carries an HTTP 401/403 `response` | 1104 |
| 3 | a memory exhaustion raised by `onnxruntime` (allocation-failure text, see 2008) | 2008 |
| 3 | a builtin `MemoryError`, or an out-of-memory raised by `cupy` / `cuml` / `numpy` | 9003 |
| 4 | `requests`/`urllib3` `ConnectionError`, `NewConnectionError` | 1102 |
| 4 | `requests`/`urllib3` `ConnectTimeout`, `ReadTimeout`, `Timeout` (+`*Error`); builtin `TimeoutError` | 1103 |
| 4 | `requests`/`urllib3` `SSLError`, `MaxRetryError`, `RetryError`, `HTTPError`; `requests.RequestException`; `LyrionAPIError` | 1101 |
| 4 | `psycopg2` `OperationalError`, `InterfaceError` (including `psycopg2.errors.OutOfMemory`, the database server's own out-of-memory) | 4001 |
| 4 | `psycopg2` `DatabaseError` (query subclasses) | 4002 |
| 4 | any other `onnxruntime` `Fail`, `RuntimeException`, `InvalidArgument`, `NoSuchFile`, `InvalidProtobuf`, `NotImplemented` | 2004 |
| 5 | anything else | the caller's `default_code` (the feature code, e.g. 2001 / 3003 / 6003) |

The out-of-memory text is only trusted on exceptions the native runtimes raise, never on
an application message, because an application message can quote a song title. The
same `error_manager.is_out_of_memory` check decides when
[tasks/onnx_utils.py](../tasks/onnx_utils.py) retries a failed GPU inference on CPU, for
`Fail` as well as `RuntimeException`.

## HTTP status for routes

`error_manager.http_status_for_code(code)` decides the HTTP status. A registry entry
may name its own `http_status`; otherwise the range decides:

| Code | HTTP status |
|------|-------------|
| 1003 / 1004 / 1005 / 1006 / 1007 / 1008 / 1009 | 400 / 404 / 401 / 403 / 409 / 410 / 413 |
| 3003, 3004 | 500 Internal Server Error |
| 6009 | 503 Service Unavailable |
| 1100-1199 (music server connection / auth / playlist) | 502 Bad Gateway |
| 1200-1299 (task queue / blocked starts) | 409 Conflict |
| 1000-1099 (other configuration / setup) | 400 Bad Request |
| 3000-3099 (other index codes) | 503 Service Unavailable |
| 4000-4099 (database) | 503 Service Unavailable |
| everything else | 500 Internal Server Error |

A route may pass `http_status` to `json_error` / `json_exception` when its contract with
the caller needs a specific status (a retryable 503, say); an explicit status wins, a
classified exception included. A Werkzeug HTTP error the route caught (the 415
`request.get_json()` raises) keeps its own status but still carries the route's extra
keys, and an `abort(response)` is passed through untouched.

A route's fallback code names the **feature** that failed (3003 search, 6003
migration, ...), never a **cause** the route cannot prove: a route whose `try` runs
more than a media-server or database call falls back to 9999, so a bug in the route
is not reported as the media server or the database being down. A real media-server
or database failure is still classified to its own code.

## How an error flows

- **Routes** answer every error through [error/responses.py](../error/responses.py):
  `json_error(code, detail, **extra)` for a rejected request and
  `json_exception(exc, feature_code, detail, **extra)` inside an `except` block (after
  `logger.exception`). The body is always `{error_code, error_class, error_message}`
  plus the legacy `error` text and any extra keys the page reads (`results: []`,
  `loaded: false`, `task_id`, ...). When the code is the route's own (every
  `json_error`, and a `json_exception` whose exception was not classified) the
  route's one-line detail is appended to `error_message` and is the `error` text;
  when the exception was classified to another code both carry that code's registry
  message only. The exception text itself never reaches the body.
  `test/unit/test_error_response_centralized.py` fails on any hand-written
  `jsonify(...)` answered with a 4xx/5xx or a computed status (tuple or
  `make_response`), and on a `jsonify({'error': ...})` answered with no status.
- **Uncaught route exceptions** hit the global `errorhandler(Exception)` in
  [app.py](../app.py), which logs the traceback and answers `json_exception(err, 9999)`,
  so a database outage still reports 4001/503. An HTTP error (404, 405, 413) on a JSON
  path (`/api/`, `/chat/api/`, `/external/`, `error.responses.JSON_ERROR_PATH_PREFIXES`)
  is answered as JSON with the request code whose registry `http_status`
  matches (`error_dictionary.request_code_for_status`, 1003 when none does) while
  keeping Werkzeug's status and headers (such as `Allow` on a 405); pages keep the
  HTML error.
- **Background tasks** need no per-task code. The queue worker
  ([taskqueue/worker.py](../taskqueue/worker.py)) builds the structured record for every
  FAIL it writes: the raised exception is classified against the task function's
  feature code from `taskqueue.TASK_FUNC_ERROR_CODES` (the allow-list is derived from
  that map, so a function cannot be allowed without a code), a job process that died
  is diagnosed from its signal (9001 / 9003 / 9004 / 9005), and a connection lost
  past the free requeues records 4001. The claim remembers the error already on the
  row: a record the task itself wrote during this attempt (the union analysis run
  records index 3001 and re-raises) wins over the queue's generic classification,
  while a record an earlier attempt left there is replaced by this attempt's. Tasks
  can also raise an `AudioMuseError` carrying their specific code (analysis 2005/2006).
  Maintenance writes 9001 for a reclaimed worker death and 9002 for an interrupted
  inline task, and a parent that gives up on a stalled album child records 2002 on it.
- **Per-track model failures** (MusiCNN, CLAP) are logged with their code (2008 or
  2004) and the track is skipped as 2007.
- **`/api/status`, `/api/last_task`, `/api/active_tasks`** run the stored details
  through the shared `sanitize_task_details` helper ([app_helper.py](../app_helper.py)),
  and the provider migration job status uses the same `error_manager.task_error_record`:
  a failed row without a structured record gets the generic `9999` one.
- **Pages** show `[code] class: message` through `apiErrorText(body, fallback)` /
  `formatErrorText` in [static/error_display.js](../static/error_display.js), which
  also holds the shared `readJsonBody(response)` (a body that is not JSON reads as
  `null`). The layout loads it before any page script, so a page's first failed fetch
  can always use it. The streamed chat error event carries the same fields, with the
  classified message as its `error` text. The
  message part is the route's own `error` detail when the body carries one (so the
  page reads "[1004] Not Found: Target track not found in index" instead of
  repeating the class sentence) and `error_message` otherwise, as for a task
  record; a body with no code falls back to the plain `error` text; the dashboard task panel renders `details.error`.
- The traceback is **never** placed in the returned dict - it lives only in the
  container log.

## Adding a new error code

1. Add the constant and a `{error_class, default_message}` entry (plus `http_status`
   when the range default does not fit) in
   [error/error_dictionary.py](../error/error_dictionary.py), inside the right range.
2. If it should be derived from an exception type, add a `(name, module_prefixes, code)`
   rule to `_EXCEPTION_RULES` in [error/error_manager.py](../error/error_manager.py);
   keep `module_prefixes` tight so a name collision in another library cannot match.
3. Use it: `json_error` / `json_exception` in a route, `AudioMuseError(code, detail)`
   in a task that knows the specific cause, or a `TASK_FUNC_ERROR_CODES` entry for a
   new queue task function.
4. Add a row to the table above.
