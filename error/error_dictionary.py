# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Canonical registry of numeric error codes and their default text.

Defines the stable integer error codes grouped by domain (request and config,
media server, analysis, index and search, database, backup, lyrics, task
operations, queue) and maps each to a human-readable class and default message
consumed by ``error_manager``.

Main Features:
* ``ERROR_REGISTRY`` maps every code to its error class and default message, and
  optionally to the one HTTP status a route answers with for it.
* Lookup helpers resolve unknown codes to ``UNKNOWN_ERROR_CODE`` safely.
* ``request_code_for_status`` reads the request code for an HTTP status out of
  the registry itself (the lowest 1000-1099 code naming that status, else
  ``ERR_INVALID_REQUEST``), so no second status table can drift from it.
"""

ERR_CONFIG_INVALID = 1001
ERR_CONFIG_MEDIASERVER_CREDENTIALS = 1002
ERR_INVALID_REQUEST = 1003
ERR_NOT_FOUND = 1004
ERR_UNAUTHORIZED = 1005
ERR_FORBIDDEN = 1006
ERR_CONFLICT = 1007
ERR_GONE = 1008
ERR_PAYLOAD_TOO_LARGE = 1009
ERR_UNKNOWN_SERVER = 1010
ERR_DEFAULT_SERVER_DELETE = 1011
ERR_PLAYLIST_REJECTED = 1012
ERR_HYPERBOLIC_REJECTED = 1013
ERR_RECORDING_REJECTED = 1014
ERR_PLUGIN_PACKAGE_REJECTED = 1015
ERR_PLUGIN_VERSION_UNAVAILABLE = 1016

ERR_MEDIASERVER_UNREACHABLE = 1101
ERR_MEDIASERVER_REFUSED = 1102
ERR_MEDIASERVER_TIMEOUT = 1103
ERR_MEDIASERVER_AUTH = 1104
ERR_MEDIASERVER_LIBRARY = 1105
ERR_MEDIASERVER_PLAYLIST = 1106
ERR_MEDIASERVER_TEST_FAILED = 1107

ERR_ANALYSIS_FAILED = 2001
ERR_ALBUM_ANALYSIS_FAILED = 2002
ERR_MODEL_INFERENCE = 2004
ERR_ANALYSIS_NO_TRACKS_ANALYZED = 2005
ERR_ANALYSIS_SERVER_FAILED = 2006
ERR_TRACK_NOT_ANALYZABLE = 2007
ERR_MODEL_OUT_OF_MEMORY = 2008

ERR_INDEX_BUILD = 3001
ERR_INDEX_EMPTY = 3002
ERR_SEARCH_FAILED = 3003
ERR_CACHE_REFRESH_FAILED = 3004
ERR_RECORDING_INDEX_UNAVAILABLE = 3005

ERR_DB_CONNECTION = 4001
ERR_DB_QUERY = 4002

ERR_BACKUP_VERSION_MISMATCH = 4101
ERR_BACKUP_FAILED = 4102
ERR_RESTORE_FAILED = 4103

ERR_LYRICS_FAILED = 5001
ERR_LYRICS_TRANSCRIPTION = 5002

ERR_CLUSTERING_FAILED = 6001
ERR_CLEANING_FAILED = 6002
ERR_PROVIDER_MIGRATION_FAILED = 6003
ERR_SERVER_SYNC_FAILED = 6004
ERR_SONIC_FINGERPRINT_FAILED = 6005
ERR_NAMING_PREVIEW_FAILED = 6006
ERR_PLUGIN_FAILED = 6007
ERR_TASK_ENQUEUE_FAILED = 6008
ERR_TASK_CANCEL_FAILED = 6009
ERR_PLUGIN_DOWNLOAD_FAILED = 6010
ERR_ALBUM_CREATION_FAILED = 6011

ERR_TASK_IN_PROGRESS = 1201

ERR_WORKER_LOST = 9001
ERR_TASK_INTERRUPTED = 9002
ERR_OUT_OF_MEMORY = 9003
ERR_PROCESS_CRASHED = 9004
ERR_JOB_PROCESS_DIED = 9005

UNKNOWN_ERROR_CODE = 9999

_REQUEST_BAND = (1000, 1100)

ERROR_REGISTRY = {
    ERR_CONFIG_INVALID: {
        "error_class": "Configuration Error",
        "default_message": "The application configuration is invalid.",
    },
    ERR_CONFIG_MEDIASERVER_CREDENTIALS: {
        "error_class": "Configuration Error",
        "default_message": "Required media server credentials are missing.",
    },
    ERR_INVALID_REQUEST: {
        "error_class": "Invalid Request",
        "default_message": "The request is invalid.",
        "http_status": 400,
    },
    ERR_NOT_FOUND: {
        "error_class": "Not Found",
        "default_message": "The requested item was not found.",
        "http_status": 404,
    },
    ERR_UNAUTHORIZED: {
        "error_class": "Authentication Required",
        "default_message": "Authentication is required.",
        "http_status": 401,
    },
    ERR_FORBIDDEN: {
        "error_class": "Forbidden",
        "default_message": "This action is not allowed for the current user.",
        "http_status": 403,
    },
    ERR_CONFLICT: {
        "error_class": "Conflict",
        "default_message": "The request conflicts with the current state.",
        "http_status": 409,
    },
    ERR_GONE: {
        "error_class": "Gone",
        "default_message": "The requested item is no longer available.",
        "http_status": 410,
    },
    ERR_PAYLOAD_TOO_LARGE: {
        "error_class": "Payload Too Large",
        "default_message": "The uploaded data is too large.",
        "http_status": 413,
    },
    ERR_UNKNOWN_SERVER: {
        "error_class": "Unknown Server",
        "default_message": (
            "Unknown server: the selected music server does not exist any more. "
            "Pick another server."
        ),
        "http_status": 400,
    },
    ERR_DEFAULT_SERVER_DELETE: {
        "error_class": "Default Server",
        "default_message": (
            "Cannot delete the default server; set another server as default first."
        ),
        "http_status": 409,
    },
    ERR_PLAYLIST_REJECTED: {
        "error_class": "Playlist Rejected",
        "default_message": (
            "The playlist was not created: none of the selected tracks are on the target "
            "server, or that server needs a user id and token."
        ),
        "http_status": 400,
    },
    ERR_HYPERBOLIC_REJECTED: {
        "error_class": "Hyperbolic Request Rejected",
        "default_message": (
            "The hyperbolic request could not be answered: check the chosen songs and values, "
            "and run the analysis if the hyperbolic index, the song's projection or this "
            "server's tree is not built yet."
        ),
        "http_status": 400,
    },
    ERR_RECORDING_REJECTED: {
        "error_class": "Recording Rejected",
        "default_message": (
            "The recording could not be searched: the clip must be decodable, non-silent "
            "audio of at least two seconds within the upload limit, and a library song needs "
            "the neural fingerprint the analysis computes."
        ),
        "http_status": 400,
    },
    ERR_PLUGIN_PACKAGE_REJECTED: {
        "error_class": "Plugin Package Rejected",
        "default_message": (
            "The plugin package was rejected: it needs a newer AudioMuse-AI, its checksum or "
            "archive is invalid, or it declares an unsafe requirement. The container log "
            "names the reason."
        ),
        "http_status": 400,
    },
    ERR_PLUGIN_VERSION_UNAVAILABLE: {
        "error_class": "Plugin Version Unavailable",
        "default_message": (
            "That plugin version is not available from any configured repository; "
            "nothing was changed."
        ),
        "http_status": 409,
    },
    ERR_MEDIASERVER_UNREACHABLE: {
        "error_class": "Music Server Connection Error",
        "default_message": "Could not reach the configured media server.",
    },
    ERR_MEDIASERVER_REFUSED: {
        "error_class": "Music Server Connection Error",
        "default_message": "The media server refused the connection.",
    },
    ERR_MEDIASERVER_TIMEOUT: {
        "error_class": "Music Server Connection Error",
        "default_message": "Timed out waiting for the media server.",
    },
    ERR_MEDIASERVER_AUTH: {
        "error_class": "Music Server Authentication Error",
        "default_message": "The media server rejected the provided credentials.",
    },
    ERR_MEDIASERVER_LIBRARY: {
        "error_class": "Music Server Library Error",
        "default_message": "No music was found to scan on the media server.",
    },
    ERR_MEDIASERVER_PLAYLIST: {
        "error_class": "Music Server Playlist Error",
        "default_message": "The media server did not create the playlist.",
    },
    ERR_MEDIASERVER_TEST_FAILED: {
        "error_class": "Music Server Connection Test Failed",
        "default_message": (
            "The media server connection test failed: the server could not be reached, or it "
            "answered without a usable music library. The container log has the server's answer."
        ),
    },
    ERR_ANALYSIS_FAILED: {
        "error_class": "Analysis Error",
        "default_message": "Audio analysis failed.",
    },
    ERR_ALBUM_ANALYSIS_FAILED: {
        "error_class": "Analysis Error",
        "default_message": "Album analysis failed.",
    },
    ERR_MODEL_INFERENCE: {
        "error_class": "Model Inference Error",
        "default_message": (
            "An analysis model failed while running inference, and the error was not "
            "recognised as an out-of-memory condition. Check that the model files are "
            "intact and read the model error in the container logs."
        ),
    },
    ERR_MODEL_OUT_OF_MEMORY: {
        "error_class": "Model Out Of Memory",
        "default_message": (
            "An analysis model ran out of memory during inference (GPU VRAM or "
            "system RAM). Free GPU memory or give the worker more memory."
        ),
    },
    ERR_ANALYSIS_NO_TRACKS_ANALYZED: {
        "error_class": "Analysis Error",
        "default_message": "The analysis ran to the end but could not analyze a single song.",
    },
    ERR_ANALYSIS_SERVER_FAILED: {
        "error_class": "Analysis Error",
        "default_message": "Analysis could not be completed for one or more music servers.",
    },
    ERR_TRACK_NOT_ANALYZABLE: {
        "error_class": "Track Skipped",
        "default_message": "The track could not be analyzed and was skipped.",
    },
    ERR_INDEX_BUILD: {
        "error_class": "Index Error",
        "default_message": "The search index could not be built.",
    },
    ERR_INDEX_EMPTY: {
        "error_class": "Index Error",
        "default_message": "The search index is empty.",
    },
    ERR_SEARCH_FAILED: {
        "error_class": "Search Error",
        "default_message": "The search could not be completed.",
        "http_status": 500,
    },
    ERR_CACHE_REFRESH_FAILED: {
        "error_class": "Cache Refresh Error",
        "default_message": "The in-memory cache could not be refreshed.",
        "http_status": 500,
    },
    ERR_RECORDING_INDEX_UNAVAILABLE: {
        "error_class": "Recording Index Unavailable",
        "default_message": (
            "The recording index is not ready: it is not built yet, is being prepared, or was "
            "built with another codebook. Run the analysis, or try again in a minute."
        ),
        "http_status": 503,
    },
    ERR_DB_CONNECTION: {
        "error_class": "Database Error",
        "default_message": "A database connection error occurred.",
    },
    ERR_DB_QUERY: {
        "error_class": "Database Error",
        "default_message": "A database query failed.",
    },
    ERR_BACKUP_VERSION_MISMATCH: {
        "error_class": "Backup Error",
        "default_message": "Backup failed due to a PostgreSQL version mismatch.",
    },
    ERR_BACKUP_FAILED: {
        "error_class": "Backup Error",
        "default_message": "The database backup failed.",
    },
    ERR_RESTORE_FAILED: {
        "error_class": "Restore Error",
        "default_message": "The database restore failed.",
    },
    ERR_LYRICS_FAILED: {
        "error_class": "Lyrics Error",
        "default_message": "Lyrics could not be retrieved.",
    },
    ERR_LYRICS_TRANSCRIPTION: {
        "error_class": "Lyrics Transcription Error",
        "default_message": "Lyrics transcription failed.",
    },
    ERR_CLUSTERING_FAILED: {
        "error_class": "Clustering Error",
        "default_message": "Playlist clustering failed.",
    },
    ERR_CLEANING_FAILED: {
        "error_class": "Cleaning Error",
        "default_message": "Database cleaning failed.",
    },
    ERR_PROVIDER_MIGRATION_FAILED: {
        "error_class": "Provider Migration Error",
        "default_message": "The provider migration failed.",
    },
    ERR_SERVER_SYNC_FAILED: {
        "error_class": "Server Sync Error",
        "default_message": "The music server alignment failed.",
    },
    ERR_SONIC_FINGERPRINT_FAILED: {
        "error_class": "Sonic Fingerprint Error",
        "default_message": "The sonic fingerprint could not be generated.",
    },
    ERR_NAMING_PREVIEW_FAILED: {
        "error_class": "Naming Preview Error",
        "default_message": "The playlist naming preview failed.",
    },
    ERR_PLUGIN_FAILED: {
        "error_class": "Plugin Error",
        "default_message": "The plugin operation failed.",
    },
    ERR_TASK_ENQUEUE_FAILED: {
        "error_class": "Task Queue Error",
        "default_message": "The task could not be queued.",
    },
    ERR_TASK_CANCEL_FAILED: {
        "error_class": "Task Cancel Error",
        "default_message": "The cancellation could not be fully applied or confirmed.",
        "http_status": 503,
    },
    ERR_PLUGIN_DOWNLOAD_FAILED: {
        "error_class": "Plugin Download Error",
        "default_message": (
            "The plugin download failed: the repository host could not be reached or answered "
            "with an error. The container log has the details."
        ),
        "http_status": 502,
    },
    ERR_ALBUM_CREATION_FAILED: {
        "error_class": "Album Creation Error",
        "default_message": "The album could not be created.",
    },
    ERR_TASK_IN_PROGRESS: {
        "error_class": "Task In Progress",
        "default_message": "Another queue job is already running.",
    },
    ERR_WORKER_LOST: {
        "error_class": "Worker Lost",
        "default_message": "The worker running this task stopped unexpectedly.",
    },
    ERR_TASK_INTERRUPTED: {
        "error_class": "Task Interrupted",
        "default_message": "The web process running this task stopped before it finished.",
    },
    ERR_OUT_OF_MEMORY: {
        "error_class": "Out Of Memory",
        "default_message": (
            "The task ran out of memory. Give the worker container more memory "
            "or run fewer memory-heavy jobs at once."
        ),
    },
    ERR_PROCESS_CRASHED: {
        "error_class": "Process Crashed",
        "default_message": "The task process crashed before it could report back.",
    },
    ERR_JOB_PROCESS_DIED: {
        "error_class": "Job Process Died",
        "default_message": "The task process ended unexpectedly.",
    },
    UNKNOWN_ERROR_CODE: {
        "error_class": "Unknown Error",
        "default_message": "An unexpected error occurred. Check the container logs for details.",
    },
}


def get_error_class(code):
    entry = ERROR_REGISTRY.get(code) or ERROR_REGISTRY[UNKNOWN_ERROR_CODE]
    return entry["error_class"]


def get_default_message(code):
    entry = ERROR_REGISTRY.get(code) or ERROR_REGISTRY[UNKNOWN_ERROR_CODE]
    return entry["default_message"]


def get_http_status(code):
    entry = ERROR_REGISTRY.get(code)
    return entry.get("http_status") if entry else None


def request_code_for_status(status):
    matches = [
        code
        for code, entry in ERROR_REGISTRY.items()
        if _REQUEST_BAND[0] <= code < _REQUEST_BAND[1]
        and "http_status" in entry
        and entry["http_status"] == status
    ]
    return min(matches) if matches else ERR_INVALID_REQUEST
