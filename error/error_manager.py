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
  module-qualified name matching plus HTTP 401/403 auth detection. Both walk the
  exception chain once, outermost first, the way a traceback prints it: the
  explicit ``__cause__``, else the implicit ``__context__`` unless the raise
  suppressed it (``raise ... from None``). A permanent failure raised ``from`` a
  database or media-server error keeps that error's code, and a deliberately
  detached exception is judged on its own.
* ``task_error_record`` gives a failed task row its structured error: the record
  the row carries, or the generic 9999 one when it carries none.
* ``is_out_of_memory`` recognises memory exhaustion anywhere in the exception
  chain; ``classify`` reports it as a model out-of-memory error when a model
  runtime raised it and as a general out-of-memory error otherwise, so an
  ordinary inference failure is never mistaken for one. A runtime reports an
  allocation failure as a generic RuntimeException whose text is the only
  signal, so that text is matched (the CUDA, cuDNN, cuBLAS, MIOpen and DirectML
  spellings, and OOM as a whole word), but only on exceptions the native
  runtimes raised: an application message may quote a song title.
  ``is_model_out_of_memory`` is true only when the model runtime itself ran out,
  the one case where freeing its session leaves room for a CPU retry.
* ``detail_text`` caps a detail before folding it to one line, so a huge
  exception text is never copied whole; ``build`` produces one-line,
  length-bounded messages and ``build_with_detail`` takes a detail that already
  went through ``detail_text``; ``http_status_for_code`` maps codes to HTTP.
* ``record`` always logs the full trace to a logger (never to the caller).
"""

import logging
import re

from error.error_dictionary import (
    ERROR_REGISTRY,
    UNKNOWN_ERROR_CODE,
    ERR_MEDIASERVER_REFUSED,
    ERR_MEDIASERVER_TIMEOUT,
    ERR_MEDIASERVER_UNREACHABLE,
    ERR_MEDIASERVER_AUTH,
    ERR_DB_CONNECTION,
    ERR_DB_QUERY,
    ERR_MODEL_INFERENCE,
    ERR_MODEL_OUT_OF_MEMORY,
    ERR_OUT_OF_MEMORY,
    ERR_UNKNOWN_SERVER,
    get_error_class,
    get_default_message,
    get_http_status,
)

_LOGGER = logging.getLogger(__name__)

_MAX_MESSAGE_DETAIL = 400
_DETAIL_SCAN_LIMIT = _MAX_MESSAGE_DETAIL * 4

_AUTH_STATUS_CODES = (401, 403)

# (class_name, module_prefixes, code). module_prefixes is None to match the name
# in any module (used for names unique to this app), or a tuple of import-path
# prefixes to restrict the match. The restriction stops unrelated libraries that
# reuse a common class name (e.g. psycopg2.OperationalError, builtin
# BrokenPipeError) from stealing a media-server or database code.
_EXCEPTION_RULES = (
    ("LyrionAPIError", None, ERR_MEDIASERVER_UNREACHABLE),
    ("UnknownServerError", ("app_server_context",), ERR_UNKNOWN_SERVER),
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
    ("NoSuchFile", ("onnxruntime",), ERR_MODEL_INFERENCE),
    ("InvalidProtobuf", ("onnxruntime",), ERR_MODEL_INFERENCE),
    ("NotImplemented", ("onnxruntime",), ERR_MODEL_INFERENCE),
)

_MODEL_RUNTIME_MODULES = ("onnxruntime",)
_NATIVE_MEMORY_MODULES = _MODEL_RUNTIME_MODULES + ("cupy", "cuml", "numpy")
_OUT_OF_MEMORY_MARKERS = (
    "failed to allocate memory",
    "bfcarena",
    "out of memory",
    "out_of_memory",
    "outofmemory",
    "alloc_failed",
    "allocfailed",
    "std::bad_alloc",
)
_OOM_WORD = re.compile(r"\boom\b")


def _one_line(text):
    return " ".join(str(text).split())


def detail_text(message):
    if not message:
        return ""
    text = str(message)
    clipped = len(text) > _DETAIL_SCAN_LIMIT
    detail = _one_line(text[:_DETAIL_SCAN_LIMIT])
    if clipped or len(detail) > _MAX_MESSAGE_DETAIL:
        detail = detail[: _MAX_MESSAGE_DETAIL - 3].rstrip() + "..."
    return detail


def build_with_detail(code, detail):
    resolved_code = code if code in ERROR_REGISTRY else UNKNOWN_ERROR_CODE
    error_class = get_error_class(resolved_code)
    base = get_default_message(resolved_code)
    full = f"{base} {detail}" if detail and resolved_code != UNKNOWN_ERROR_CODE else base
    return {"error_code": resolved_code, "error_class": error_class, "error_message": full}


def build(code, message=None):
    return build_with_detail(code, detail_text(message))


def _exception_chain(exc):
    seen = set()
    links = []
    current = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        links.append(current)
        cause = getattr(current, "__cause__", None)
        if cause is None and not getattr(current, "__suppress_context__", False):
            cause = getattr(current, "__context__", None)
        current = cause
    return links


def _auth_error_code(links):
    for link in links:
        response = getattr(link, "response", None)
        if getattr(response, "status_code", None) in _AUTH_STATUS_CODES:
            return ERR_MEDIASERVER_AUTH
    return None


def _module_of(exc):
    return getattr(type(exc), "__module__", "") or ""


def _is_memory_exhaustion(link):
    if isinstance(link, MemoryError):
        return True
    if not _module_of(link).startswith(_NATIVE_MEMORY_MODULES):
        return False
    if "OutOfMemory" in type(link).__name__:
        return True
    text = str(link)[:_DETAIL_SCAN_LIMIT].lower()
    return any(marker in text for marker in _OUT_OF_MEMORY_MARKERS) or bool(
        _OOM_WORD.search(text)
    )


def is_out_of_memory(exc):
    return any(_is_memory_exhaustion(link) for link in _exception_chain(exc))


def is_model_out_of_memory(exc):
    return any(
        _is_memory_exhaustion(link) and _module_of(link).startswith(_MODEL_RUNTIME_MODULES)
        for link in _exception_chain(exc)
    )


def _out_of_memory_code(links):
    if not any(_is_memory_exhaustion(link) for link in links):
        return None
    if any(_module_of(link).startswith(_MODEL_RUNTIME_MODULES) for link in links):
        return ERR_MODEL_OUT_OF_MEMORY
    return ERR_OUT_OF_MEMORY


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
    links = _exception_chain(exc)
    auth_code = _auth_error_code(links)
    if auth_code is not None:
        return auth_code
    memory_code = _out_of_memory_code(links)
    if memory_code is not None:
        return memory_code
    for link in links:
        if isinstance(link, AudioMuseError):
            return link.code
        matched = _match_rule(link)
        if matched is not None:
            return matched
    return default_code


def http_status_for_code(code):
    explicit = get_http_status(code)
    if explicit is not None:
        return explicit
    if 1100 <= code < 1200:
        return 502
    if 1200 <= code < 1300:
        return 409
    if 1000 <= code < 1100:
        return 400
    if 3000 <= code < 3100:
        return 503
    if 4000 <= code < 4100:
        return 503
    return 500


def task_error_record(details):
    existing = details.get("error") if isinstance(details, dict) else None
    if isinstance(existing, dict) and "error_code" in existing:
        if "error_message" in existing:
            return existing
        return build(existing["error_code"])
    return build(UNKNOWN_ERROR_CODE)


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
