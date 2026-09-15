# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The one way a Flask route answers with an error.

Every error a route returns, a rejected request as much as a caught exception,
is built here from a registry code, so the body always carries
``error_code``, ``error_class`` and ``error_message`` next to the legacy
``error`` text the pages already read, and the HTTP status always comes from
the code unless the route states its own. ``error_manager`` stays free of Flask
so the worker can import it; this module is the Flask-facing half. It imports
the rest of the package inside its functions: every blueprint imports it, and
an eager import here would add a level to the import chain of every route module.

Main Features:
* ``json_error`` answers with a code, an optional detail (the ``error`` alias,
  folded to one line and capped like the message, so a short route message is
  unchanged and a tool's stderr never reaches the page whole) and any extra keys
  the page expects, ``message`` included for the pages that read that key.
* ``json_exception`` classifies a caught exception first, so a database,
  media-server or model failure answers with its own code and registry message
  instead of the route's generic ones. An exception that stays on the route's
  own code answers exactly like ``json_error``: the route detail is folded into
  ``error_message`` too, which is the text the pages show. It never puts the
  exception text in the body: the caller gets only the registry code and its
  predefined message, and the container log gets everything. The exception is
  logged exactly once, with the request method and path: when the route already
  logged it (``logger.exception``, which ``app_logging.LogSanitizingFilter``
  marks) it is not repeated, otherwise it is logged here, as a warning for a
  rejected request (below 500) and as an error for a failure. That holds for
  every branch, a coded ``AudioMuseError`` and a caught Werkzeug error included.
  A route catching ``ValueError`` answers ``json_exception(exc, <code whose
  message explains the rejection>)``, never ``json_error(code, str(exc))``.
* An extra ``message=None`` is filled with the answer text, for the pages and
  clients that read ``message`` next to ``error``.
* A Werkzeug HTTP error a route's own ``except Exception`` caught (the 415 or
  400 ``request.get_json()`` raises for a body that is not JSON) keeps its
  status and request code instead of turning into the route's 500, and still
  carries the route's extra keys (``results: []``), which the page reads.
* ``http_status``, when a route passes it, is the status for everything but a
  Werkzeug HTTP error, a classified exception included: it is the route's
  contract with its caller (a retryable 503, say). It is not named ``status``
  because several bodies carry a ``status`` key of their own.
* ``json_http_exception`` turns a Werkzeug HTTP error (a routing 404, a 405, a
  413 from the request size cap) into the same JSON body on API paths, carrying
  the request code the registry names for that status, while keeping
  Werkzeug's own response, so its headers (Allow on a 405, WWW-Authenticate on
  a 401) survive; an error without a code answers 500, never a JSON error body
  with 200. An ``abort(response)`` carries a response of its own, which is
  returned untouched.
* ``JSON_ERROR_PATH_PREFIXES`` / ``wants_json_error`` name the paths whose
  callers read JSON: the global handlers and the auth barrier answer them the
  same way.
"""

import logging

from flask import has_request_context, json, jsonify, request

logger = logging.getLogger(__name__)

JSON_ERROR_PATH_PREFIXES = ("/api/", "/chat/api/", "/external/")


def _body(payload, alias, extra):
    body = {**payload, "error": alias}
    body.update(extra)
    if "message" in body and body["message"] is None:
        body["message"] = alias
    return body


def _log_once(exc, code, status):
    from app_logging import mark_logged, was_logged

    if was_logged(exc):
        return
    where = f"{request.method} {request.path}" if has_request_context() else "a call"
    if status < 500:
        logger.warning("Rejected %s with error code %s", where, code, exc_info=exc)
    else:
        logger.error("Failed %s with error code %s", where, code, exc_info=exc)
    mark_logged(exc)


def _respond(payload, alias, http_status, extra):
    from error.error_manager import http_status_for_code

    if http_status is None:
        http_status = http_status_for_code(payload["error_code"])
    return jsonify(_body(payload, alias, extra)), http_status


def _detailed(code, detail, http_status, extra):
    from error.error_manager import build_with_detail, detail_text

    text = detail_text(detail)
    payload = build_with_detail(code, text)
    return _respond(payload, text or payload["error_message"], http_status, extra)


def json_error(code, detail=None, http_status=None, **extra):
    return _detailed(code, detail, http_status, extra)


def json_exception(exc, default_code, detail=None, http_status=None, **extra):
    from werkzeug.exceptions import HTTPException

    from error.error_manager import AudioMuseError, build, classify

    if isinstance(exc, HTTPException):
        response = json_http_exception(exc, **extra)
        _log_once(exc, getattr(exc, "code", None), response.status_code)
        return response, response.status_code
    if isinstance(exc, AudioMuseError):
        code = exc.code
        payload = exc.to_dict()
        response = _respond(payload, payload["error_message"], http_status, extra)
    else:
        code = classify(exc, default_code)
        if code == default_code:
            response = _detailed(code, detail, http_status, extra)
        else:
            payload = build(code)
            response = _respond(payload, payload["error_message"], http_status, extra)
    _log_once(exc, code, response[1])
    return response


def wants_json_error(path):
    return str(path or "").startswith(JSON_ERROR_PATH_PREFIXES)


def json_http_exception(err, **extra):
    from error import error_dictionary as codes
    from error.error_manager import build

    if getattr(err, "response", None) is not None:
        return err.response
    status = getattr(err, "code", None) or 500
    if status >= 500:
        payload = build(codes.UNKNOWN_ERROR_CODE)
        alias = payload["error_message"]
    else:
        description = getattr(err, "description", None)
        payload = build(codes.request_code_for_status(status), description)
        alias = description or payload["error_message"]
    response = err.get_response()
    response.status_code = status
    response.set_data(json.dumps(_body(payload, alias, extra)))
    response.content_type = "application/json"
    return response
