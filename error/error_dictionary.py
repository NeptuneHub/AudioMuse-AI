# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Canonical registry of numeric error codes and their default text.

Defines the stable integer error codes grouped by domain (config, media
server, analysis, index, database, backup, lyrics, clustering) and maps each to
a human-readable class and default message consumed by ``error_manager``.

Main Features:
* ``ERROR_REGISTRY`` maps every code to its error class and default message.
* Lookup helpers resolve unknown codes to ``UNKNOWN_ERROR_CODE`` safely.
"""

ERR_CONFIG_INVALID = 1001
ERR_CONFIG_MEDIASERVER_CREDENTIALS = 1002
ERR_STARTUP = 1003

ERR_MEDIASERVER_UNREACHABLE = 1101
ERR_MEDIASERVER_REFUSED = 1102
ERR_MEDIASERVER_TIMEOUT = 1103
ERR_MEDIASERVER_AUTH = 1104
ERR_MEDIASERVER_LIBRARY = 1105

ERR_ANALYSIS_FAILED = 2001
ERR_ALBUM_ANALYSIS_FAILED = 2002
ERR_ANALYSIS_NO_ALBUMS = 2003
ERR_MODEL_INFERENCE = 2004

ERR_INDEX_BUILD = 3001
ERR_INDEX_EMPTY = 3002

ERR_DB_CONNECTION = 4001
ERR_DB_QUERY = 4002

ERR_BACKUP_VERSION_MISMATCH = 4101
ERR_BACKUP_FAILED = 4102
ERR_RESTORE_FAILED = 4103

ERR_LYRICS_FAILED = 5001
ERR_LYRICS_TRANSCRIPTION = 5002
ERR_TRANSLATION_FAILED = 5003

ERR_CLUSTERING_FAILED = 6001
ERR_CLEANING_FAILED = 6002
ERR_COLLECTION_SYNC_FAILED = 6003

UNKNOWN_ERROR_CODE = 9999

ERROR_REGISTRY = {
    ERR_CONFIG_INVALID: {
        "error_class": "Configuration Error",
        "default_message": "The application configuration is invalid.",
    },
    ERR_CONFIG_MEDIASERVER_CREDENTIALS: {
        "error_class": "Configuration Error",
        "default_message": "Required media server credentials are missing.",
    },
    ERR_STARTUP: {
        "error_class": "Startup Error",
        "default_message": "The application failed to start.",
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
    ERR_ANALYSIS_FAILED: {
        "error_class": "Analysis Error",
        "default_message": "Audio analysis failed.",
    },
    ERR_ALBUM_ANALYSIS_FAILED: {
        "error_class": "Analysis Error",
        "default_message": "Album analysis failed.",
    },
    ERR_ANALYSIS_NO_ALBUMS: {
        "error_class": "Analysis Error",
        "default_message": "No albums were available to analyze.",
    },
    ERR_MODEL_INFERENCE: {
        "error_class": "Model Inference Error",
        "default_message": "An analysis model failed to produce a result.",
    },
    ERR_INDEX_BUILD: {
        "error_class": "Index Error",
        "default_message": "The search index could not be built.",
    },
    ERR_INDEX_EMPTY: {
        "error_class": "Index Error",
        "default_message": "The search index is empty.",
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
    ERR_TRANSLATION_FAILED: {
        "error_class": "Translation Error",
        "default_message": "Lyrics translation failed.",
    },
    ERR_CLUSTERING_FAILED: {
        "error_class": "Clustering Error",
        "default_message": "Playlist clustering failed.",
    },
    ERR_CLEANING_FAILED: {
        "error_class": "Cleaning Error",
        "default_message": "Database cleaning failed.",
    },
    ERR_COLLECTION_SYNC_FAILED: {
        "error_class": "Collection Sync Error",
        "default_message": "Collection synchronization failed.",
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
