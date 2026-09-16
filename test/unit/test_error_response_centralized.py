# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""No route answers with a hand-rolled error body.

Every error response goes through error.responses (json_error, json_exception),
so the page always receives the structured error code next to the legacy
``error`` text. Hand-written ``jsonify({'error': ...}), 500`` bodies drifted
into more than three hundred call sites once: none carried a code, and the
caught exceptions behind them were never classified, so a database outage read
as "Internal error". This scan keeps that from growing back.

Main Features:
* The repo walk prunes virtualenvs, models and build trees in place, so the scan
  reads only the project's own sources
* A ``jsonify(...)`` body (whatever it holds, a plain variable included), or a
  bare dict with an ``error`` key, answered with a literal 4xx/5xx status fails,
  as a tuple or through ``make_response``
* The same body answered with a computed status fails too, since that status is
  an error path in disguise; the two routes whose computed status is a success
  one are named in a minimal allow-list
* A ``jsonify`` of a dict literal holding a non-null ``error`` key with no status
  at all fails: an error answered as an implicit 200 carries no code either. A
  success body that reports a failed probe states its 200 explicitly
* An ``except`` block never answers with its exception's text: ``str(exc)``, an
  f-string or ``exc.args`` handed to ``json_error`` / ``json_exception`` fails
  (CodeQL py/stack-trace-exposure). The route passes the exception itself to
  ``json_exception``: the caller gets the registry code and message, the log
  gets the full traceback
* The scanner is checked against small sources, so each rule really fires
"""

import ast
import os
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

EXCLUDED_DIRS = {
    ".git", ".venv", ".venv-windows", "node_modules", "__pycache__", "build",
    "dist", "pginstall", "native-build", "test", "screenshot", "model",
}

ALLOWED_FILES = {Path("error") / "responses.py"}

ALLOWED_COMPUTED_STATUS = {
    (Path("app_music_servers.py"), "_restart_partial_failure"),
    (Path("app_chat.py"), "chat_playlist_api"),
}


def _python_files():
    for dirpath, dirnames, filenames in os.walk(REPO_ROOT):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIRS and not d.startswith(".")]
        for filename in filenames:
            if not filename.endswith(".py"):
                continue
            path = Path(dirpath) / filename
            relative = path.relative_to(REPO_ROOT)
            if relative not in ALLOWED_FILES:
                yield path, relative


def _call_name(node):
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    return func.attr if isinstance(func, ast.Attribute) else None


def _is_jsonify(node):
    return _call_name(node) == "jsonify"


def _dict_carries_error(node):
    if not isinstance(node, ast.Dict):
        return False
    for key, value in zip(node.keys, node.values):
        if isinstance(key, ast.Constant) and key.value == "error":
            return not (isinstance(value, ast.Constant) and value.value is None)
    return False


def _jsonify_carries_error(call):
    if any(_dict_carries_error(arg) for arg in call.args):
        return True
    return any(keyword.arg == "error" for keyword in call.keywords)


def _walk(node, function="<module>"):
    for child in ast.iter_child_nodes(node):
        is_function = isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
        name = child.name if is_function else function
        yield child, name
        yield from _walk(child, name)


def _body_and_status(node):
    if isinstance(node, ast.Tuple) and len(node.elts) in (2, 3):
        body, status = node.elts[0], node.elts[1]
    elif _call_name(node) == "make_response" and len(node.args) >= 2:
        body, status = node.args[0], node.args[1]
    else:
        return None
    if isinstance(status, (ast.Dict, ast.List)):
        return None
    if _is_jsonify(body) or _dict_carries_error(body):
        return body, status
    return None


def _is_error_status(status, relative, function):
    if isinstance(status, ast.Constant):
        return isinstance(status.value, int) and status.value >= 400
    return (relative, function) not in ALLOWED_COMPUTED_STATUS


def _violations(tree, relative):
    answered = set()
    unanswered = []
    for node, function in _walk(tree):
        pair = _body_and_status(node)
        if pair is not None:
            body, status = pair
            answered.add(id(body))
            if _is_error_status(status, relative, function):
                yield node.lineno
        elif _is_jsonify(node):
            unanswered.append(node)
    for call in unanswered:
        if id(call) not in answered and _jsonify_carries_error(call):
            yield call.lineno


def _scan(source, relative=Path("app_example.py")):
    return list(_violations(ast.parse(textwrap.dedent(source)), relative))


_RESPONSE_HELPERS = {"json_error", "json_exception"}
_CURATED_EXCEPTION_ATTRIBUTES = {"code", "user_message", "status_code"}


def _mentions(node, name):
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return node.value.id == name and node.attr not in _CURATED_EXCEPTION_ATTRIBUTES
    if isinstance(node, ast.Name):
        return node.id == name
    return any(_mentions(child, name) for child in ast.iter_child_nodes(node))


_PAYLOAD_KEYS = {"error", "message", "detail"}
_LOG_CALLS = {"debug", "info", "warning", "error", "exception", "critical", "log", "write"}


def _is_text_of(node, names):
    if isinstance(node, ast.Call) and _call_name(node) in ("str", "repr") and node.args:
        return any(_mentions(node.args[0], name) for name in names)
    if isinstance(node, ast.JoinedStr):
        return any(
            isinstance(part, ast.FormattedValue) and any(_mentions(part.value, name) for name in names)
            for part in node.values
        )
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return node.value.id in names and node.attr not in _CURATED_EXCEPTION_ATTRIBUTES
    if isinstance(node, ast.Name):
        return node.id in names
    return False


def _carries_text(node, names):
    if _call_name(node) in _LOG_CALLS:
        return False
    if isinstance(node, ast.Attribute) and node.attr in _CURATED_EXCEPTION_ATTRIBUTES:
        return False
    if _is_text_of(node, names):
        return True
    return any(_carries_text(child, names) for child in ast.iter_child_nodes(node))


def _tainted_names(handler):
    names = {handler.name}
    for node in ast.walk(handler):
        if isinstance(node, ast.Assign) and _carries_text(node.value, names):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
    return names


def _answers_http(relative):
    return (len(relative.parts) == 1 and relative.name.startswith("app")) or relative == Path(
        "plugin", "blueprint.py"
    )


def _exception_text_leaks(tree, relative=Path("app_example.py")):
    route_module = _answers_http(relative)
    for handler in ast.walk(tree):
        if not isinstance(handler, ast.ExceptHandler) or not handler.name:
            continue
        names = _tainted_names(handler)
        for node in ast.walk(handler):
            if _call_name(node) in _RESPONSE_HELPERS | {"jsonify"}:
                args = list(node.args)
                if _call_name(node) == "json_exception" and args:
                    if isinstance(args[0], ast.Name) and args[0].id == handler.name:
                        args = args[1:]
                values = args + [keyword.value for keyword in node.keywords]
                if any(_carries_text(value, names) for value in values):
                    yield node.lineno
            elif route_module and isinstance(node, ast.Dict):
                for key, value in zip(node.keys, node.values):
                    if isinstance(key, ast.Constant) and key.value in _PAYLOAD_KEYS \
                            and _carries_text(value, names):
                        yield node.lineno


def _leaks(source):
    return list(_exception_text_leaks(ast.parse(textwrap.dedent(source))))


def test_the_walk_never_enters_an_excluded_directory():
    for _path, relative in _python_files():
        assert not set(relative.parts[:-1]) & EXCLUDED_DIRS, relative


def test_the_scanner_flags_every_hand_rolled_error_shape():
    flagged = {
        "literal status": "def r():\n    return jsonify({'error': 'x'}), 500\n",
        "variable body": "def r():\n    return jsonify(body), 404\n",
        "attribute jsonify": "def r():\n    return flask.jsonify(body), 400\n",
        "headers tuple": "def r():\n    return jsonify(body), 503, {'Retry-After': '5'}\n",
        "make_response": "def r():\n    return make_response(jsonify(body), 409)\n",
        "computed status": "def r():\n    return jsonify(body), status\n",
        "computed error dict": "def r():\n    return jsonify({'error': e}), code\n",
        "bare dict": "def r():\n    return {'error': 'x'}, 500\n",
        "implicit 200": "def r():\n    return jsonify({'error': 'x'})\n",
        "keyword error": "def r():\n    return jsonify(error='x')\n",
        "headers only": "def r():\n    return jsonify({'error': 'x'}), {'X': '1'}\n",
    }
    for label, source in flagged.items():
        assert _scan(source), label


def test_the_scanner_leaves_success_bodies_alone():
    allowed = {
        "success": "def r():\n    return jsonify(body), 200\n",
        "implicit success": "def r():\n    return jsonify(body)\n",
        "null error": "def r():\n    return jsonify({'error': None, 'ok': True})\n",
        "explicit 200 probe": "def r():\n    return jsonify({'ok': False, 'error': 'x'}), 200\n",
        "created": "def r():\n    return make_response(jsonify(body), 201)\n",
    }
    for label, source in allowed.items():
        assert not _scan(source), label


def test_only_the_named_function_may_answer_with_a_computed_status():
    source = """
    def _restart_partial_failure(body, status_code=200):
        return jsonify(body), status_code

    def another_route():
        return jsonify(body), status_code
    """

    assert _scan(source, Path("app_music_servers.py")) == [6]
    assert len(_scan(source, Path("app_other.py"))) == 2


def test_no_route_hand_rolls_an_error_response():
    found = []
    for path, relative in _python_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        found.extend(f"{relative}:{line}" for line in _violations(tree, relative))

    assert not found, (
        "Answer errors with error.responses.json_error(code, detail, **extra) or "
        "json_exception(exc, default_code, detail, **extra) so the page receives the "
        "structured error code: " + ", ".join(sorted(found))
    )


def test_the_leak_scanner_flags_exception_text_and_allows_the_exception_itself():
    flagged = {
        "str": "try:\n    x()\nexcept ValueError as exc:\n    json_error(1003, str(exc))\n",
        "f-string": "try:\n    x()\nexcept ValueError as exc:\n    json_error(1003, f'Error: {exc}', message=f'{exc}')\n",
        "args": "try:\n    x()\nexcept ValueError as exc:\n    json_exception(exc, 3003, exc.args[0])\n",
        "detail keyword": "try:\n    x()\nexcept ValueError as exc:\n    json_error(1003, detail=repr(exc))\n",
        "through a variable": "try:\n    x()\nexcept ValueError as exc:\n    msg = str(exc)\n    json_error(1003, msg)\n",
        "payload dict": "try:\n    x()\nexcept ValueError as exc:\n    errors.append({'repo': r, 'error': str(exc)})\n",
    }
    allowed = {
        "exception itself": "try:\n    x()\nexcept ValueError as exc:\n    json_exception(exc, 1003)\n",
        "own code": "try:\n    x()\nexcept AudioMuseError as ae:\n    json_exception(ae, ae.code)\n",
        "curated message": "try:\n    x()\nexcept TaskAlreadyRunning as exc:\n    json_error(1201, exc.user_message, http_status=exc.status_code)\n",
        "logged text": "try:\n    x()\nexcept ValueError as exc:\n    logger.warning('failed: %s', str(exc))\n    json_exception(exc, 1003)\n",
    }
    for label, source in flagged.items():
        assert _leaks(source), label
    for label, source in allowed.items():
        assert not _leaks(source), label


def test_no_route_answers_with_the_text_of_a_caught_exception():
    found = []
    for path, relative in _python_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        found.extend(f"{relative}:{line}" for line in _exception_text_leaks(tree, relative))

    assert not found, (
        "A caught exception's text can name files, shapes, hosts or URLs with their tokens "
        "(CodeQL py/stack-trace-exposure). Answer with json_exception(exc, code): the "
        "body gets only the registry code and message and the log gets the full "
        "traceback: " + ", ".join(sorted(found))
    )
