# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Builds, classifies, and records structured application errors.

Turns raw exceptions and error codes into the standard error dict (code, class,
message) using ``error_dictionary``, mapping exception types to codes and
capping message detail so no raw traceback leaks to callers.

Main Features:
* ``classify`` / ``from_exception`` map exception types to registry codes using
  module-qualified name matching plus HTTP 401/403 auth detection.
* ``build`` produces one-line, length-bounded messages; ``error_response`` pairs
  the dict with an HTTP status and ``http_status_for_code`` maps codes to HTTP.
* ``record`` always logs the full trace to a logger (never to the caller).
"""

import logging

from error.error_dictionary import (
    ERROR_REGISTRY,
    UNKNOWN_ERROR_CODE,
    ERR_MEDIASERVER_REFUSED,
    ERR_MEDIASERVER_TIMEOUT,
    ERR_MEDIASERVER_UNREACHABLE,
    ERR_MEDIASERVER_AUTH,
    ERR_DB_CONNECTION,
    ERR_DB_QUERY,
    ERR_INDEX_EMPTY,
    ERR_MODEL_INFERENCE,
    get_error_class,
    get_default_message,
)

_LOGGER = logging.getLogger(__name__)

_MAX_MESSAGE_DETAIL = 400

_AUTH_STATUS_CODES = (401, 403)

# (class_name, module_prefixes, code). module_prefixes is None to match the name
# in any module (used for names unique to this app), or a tuple of import-path
# prefixes to restrict the match. The restriction stops unrelated libraries that
# reuse a common class name (e.g. redis.exceptions.ConnectionError, builtin
# BrokenPipeError) from stealing a media-server or database code.
_EXCEPTION_RULES = (
    ("LyrionAPIError", None, ERR_MEDIASERVER_UNREACHABLE),
    ("EmptyIndexError", None, ERR_INDEX_EMPTY),
    ("OperationalError", ("psycopg2",), ERR_DB_CONNECTION),
    ("InterfaceError", ("psycopg2",), ERR_DB_CONNECTION),
    ("DatabaseError", ("psycopg2",), ERR_DB_QUERY),
    ("ConnectTimeout", ("requests", "urllib3"), ERR_MEDIASERVER_TIMEOUT),
    ("ConnectTimeoutError", ("requests", "urllib3"), ERR_MEDIASERVER_TIMEOUT),
    ("ReadTimeout", ("requests", "urllib3"), ERR_MEDIASERVER_TIMEOUT),
    ("ReadTimeoutError", ("requests", "urllib3"), ERR_MEDIASERVER_TIMEOUT),
    ("Timeout", ("requests", "urllib3"), ERR_MEDIASERVER_TIMEOUT),
    ("TimeoutError", ("builtins",), ERR_MEDIASERVER_TIMEOUT),
    ("SSLError", ("requests", "urllib3"), ERR_MEDIASERVER_UNREACHABLE),
    ("NewConnectionError", ("requests", "urllib3"), ERR_MEDIASERVER_REFUSED),
    ("ConnectionError", ("requests", "urllib3"), ERR_MEDIASERVER_REFUSED),
    ("MaxRetryError", ("requests", "urllib3"), ERR_MEDIASERVER_UNREACHABLE),
    ("RetryError", ("requests", "urllib3"), ERR_MEDIASERVER_UNREACHABLE),
    ("HTTPError", ("requests", "urllib3"), ERR_MEDIASERVER_UNREACHABLE),
    ("RequestException", ("requests",), ERR_MEDIASERVER_UNREACHABLE),
    ("Fail", ("onnxruntime",), ERR_MODEL_INFERENCE),
    ("RuntimeException", ("onnxruntime",), ERR_MODEL_INFERENCE),
    ("InvalidArgument", ("onnxruntime",), ERR_MODEL_INFERENCE),
    ("MemoryError", ("builtins",), ERR_MODEL_INFERENCE),
)


def _one_line(text):
    return " ".join(str(text).split())


def build(code, message=None):
    resolved_code = code if code in ERROR_REGISTRY else UNKNOWN_ERROR_CODE
    error_class = get_error_class(resolved_code)
    base = get_default_message(resolved_code)
    detail = _one_line(message) if (message and resolved_code != UNKNOWN_ERROR_CODE) else ""
    if detail and len(detail) > _MAX_MESSAGE_DETAIL:
        detail = detail[: _MAX_MESSAGE_DETAIL - 3].rstrip() + "..."
    full = f"{base} {detail}".strip() if detail else base
    return {"error_code": resolved_code, "error_class": error_class, "error_message": full}


def _auth_error_code(exc):
    seen = set()
    current = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        response = getattr(current, "response", None)
        if getattr(response, "status_code", None) in _AUTH_STATUS_CODES:
            return ERR_MEDIASERVER_AUTH
        current = getattr(current, "__cause__", None) or getattr(current, "__context__", None)
    return None


def _match_rule(exc):
    for cls in type(exc).__mro__:
        module = getattr(cls, "__module__", "") or ""
        name = cls.__name__
        for rule_name, prefixes, code in _EXCEPTION_RULES:
            if name == rule_name and (prefixes is None or module.startswith(prefixes)):
                return code
    return None


def classify(exc, default_code=UNKNOWN_ERROR_CODE):
    if isinstance(exc, AudioMuseError):
        return exc.code
    auth_code = _auth_error_code(exc)
    if auth_code is not None:
        return auth_code
    matched = _match_rule(exc)
    if matched is not None:
        return matched
    return default_code


def http_status_for_code(code):
    if 1100 <= code < 1200:
        return 502
    if 1000 <= code < 1100:
        return 400
    if 3000 <= code < 3100:
        return 503
    if 4000 <= code < 4100:
        return 503
    return 500


def error_response(code, message=None):
    payload = build(code, message)
    payload["error"] = payload["error_message"]
    return payload, http_status_for_code(payload["error_code"])


class AudioMuseError(Exception):
    def __init__(self, code, message=None, cause=None):
        self.code = code if code in ERROR_REGISTRY else UNKNOWN_ERROR_CODE
        self.error_class = get_error_class(self.code)
        built = build(self.code, message)
        self.error_message = built["error_message"]
        self.cause = cause
        super().__init__(self.error_message)

    def to_dict(self):
        return {
            "error_code": self.code,
            "error_class": self.error_class,
            "error_message": self.error_message,
        }

    def __str__(self):
        return self.error_message


def record(code, message=None, exc=None, logger=None, level=logging.ERROR):
    err = build(code, message)
    log_target = logger if logger is not None else _LOGGER
    log_target.log(
        level,
        "[%s] %s: %s",
        err["error_code"],
        err["error_class"],
        err["error_message"],
        exc_info=exc if exc is not None else False,
    )
    return err


def from_exception(exc, code=None, message=None, logger=None, level=logging.ERROR):
    if isinstance(exc, AudioMuseError):
        err = exc.to_dict()
        log_target = logger if logger is not None else _LOGGER
        log_target.log(
            level,
            "[%s] %s: %s",
            err["error_code"],
            err["error_class"],
            err["error_message"],
            exc_info=exc.cause or exc,
        )
        return err
    resolved = code if code is not None else classify(exc, UNKNOWN_ERROR_CODE)
    if message is not None:
        detail = message
    elif resolved == UNKNOWN_ERROR_CODE:
        detail = None
    else:
        detail = str(exc)
    return record(resolved, detail, exc=exc, logger=logger, level=level)
