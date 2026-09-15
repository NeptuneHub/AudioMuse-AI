# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""The Flask-facing error responses every route answers with.

A route that rejects a request or catches an exception must still hand the page
the structured error (code, class, message) next to the legacy ``error`` text
the page already reads, with the status the code maps to.

Main Features:
* json_error keeps the route's text verbatim as the alias and adds extra keys,
  ``message`` included, with the code's HTTP status
* json_exception classifies the exception, so a database outage answers 4001/503
  instead of the route's generic 500, and never leaks the exception text
* An exception left on the route's own code answers like json_error: the route
  text is folded into error_message, the text the pages render
* An explicit http_status is the route's contract and wins in every case
* json_http_exception gives an API 404/405 the JSON body while keeping
  Werkzeug's own headers, such as Allow on a 405, with the request code the
  registry names for that status
"""

import pytest
from flask import Flask

from error import error_dictionary as ed
from error.responses import json_error, json_exception, json_http_exception


@pytest.fixture
def app_context():
    with Flask(__name__).app_context():
        yield


def _psycopg2_operational_error():
    return type('OperationalError', (Exception,), {'__module__': 'psycopg2'})('server closed')


def test_json_error_keeps_the_route_text_and_its_extra_keys(app_context):
    response, status = json_error(
        ed.ERR_INVALID_REQUEST, 'Query cannot be empty', results=[], message='kept'
    )
    body = response.get_json()

    assert status == 400
    assert body['error'] == 'Query cannot be empty'
    assert body['error_code'] == ed.ERR_INVALID_REQUEST
    assert body['error_message'].endswith('Query cannot be empty')
    assert body['results'] == [] and body['message'] == 'kept'


def test_every_request_code_answers_with_its_own_status(app_context):
    expected = {
        ed.ERR_INVALID_REQUEST: 400, ed.ERR_UNAUTHORIZED: 401, ed.ERR_FORBIDDEN: 403,
        ed.ERR_NOT_FOUND: 404, ed.ERR_CONFLICT: 409, ed.ERR_GONE: 410,
        ed.ERR_PAYLOAD_TOO_LARGE: 413, ed.ERR_TASK_IN_PROGRESS: 409,
    }
    for code, status in expected.items():
        assert json_error(code, 'x')[1] == status, code


def test_a_database_outage_answers_with_the_database_code_not_the_route_one(app_context):
    response, status = json_exception(
        _psycopg2_operational_error(), ed.ERR_SEARCH_FAILED, 'An error occurred during search.',
    )
    body = response.get_json()

    assert status == 503
    assert body['error_code'] == ed.ERR_DB_CONNECTION
    assert body['error_message'] == ed.get_default_message(ed.ERR_DB_CONNECTION)
    assert body['error'] == body['error_message']
    assert 'server closed' not in body['error']
    assert 'during search' not in str(body), (
        'the route text describes the route failure, not the database outage the code names'
    )


def test_an_unclassified_exception_keeps_the_route_text_and_code(app_context):
    response, status = json_exception(
        KeyError('secret internal key'), ed.ERR_SEARCH_FAILED, 'An error occurred during search.',
    )
    body = response.get_json()

    assert status == 500
    assert body['error_code'] == ed.ERR_SEARCH_FAILED
    assert body['error'] == 'An error occurred during search.'
    assert 'secret internal key' not in str(body)


def test_an_unclassified_exception_shows_the_route_text_where_the_page_reads_it(app_context):
    detail = 'Artist not in the index.\nTry another name.'
    exception_body = json_exception(ValueError('x'), ed.ERR_SEARCH_FAILED, detail)[0].get_json()
    error_body = json_error(ed.ERR_SEARCH_FAILED, detail)[0].get_json()

    assert exception_body['error_message'] == (
        ed.get_default_message(ed.ERR_SEARCH_FAILED) + ' Artist not in the index. Try another name.'
    ), 'apiErrorText renders error_message, so the route text must be in it'
    assert exception_body == error_body


def test_an_unclassified_exception_without_detail_answers_the_registry_message(app_context):
    body = json_exception(ValueError('x'), ed.ERR_SEARCH_FAILED)[0].get_json()

    assert body['error_message'] == ed.get_default_message(ed.ERR_SEARCH_FAILED)
    assert body['error'] == body['error_message']


def test_a_coded_error_answers_with_its_own_record(app_context):
    from error.error_manager import AudioMuseError

    exc = AudioMuseError(ed.ERR_INDEX_EMPTY, 'no embeddings')
    response, status = json_exception(exc, ed.ERR_SEARCH_FAILED, 'route text')
    body = response.get_json()

    assert status == 503
    assert body['error_code'] == ed.ERR_INDEX_EMPTY
    assert body['error'] == body['error_message'] == exc.error_message
    assert 'route text' not in str(body)


def test_a_huge_detail_reaches_the_body_capped_and_on_one_line(app_context):
    body = json_error(ed.ERR_INVALID_REQUEST, 'line\n' * 100_000)[0].get_json()

    assert '\n' not in body['error'] and body['error'].endswith('...')
    assert body['error_message'].endswith(body['error'])


def test_a_werkzeug_error_caught_by_a_route_keeps_its_own_status(app_context):
    from werkzeug.exceptions import UnsupportedMediaType

    response, status = json_exception(
        UnsupportedMediaType(), ed.ERR_SEARCH_FAILED, 'An internal error occurred.'
    )
    body = response.get_json()

    assert status == 415 and response.status_code == 415, (
        'request.get_json() raises 415 for a non-JSON body inside the route try; the '
        'route catch-all used to turn that client mistake into a 500 search error'
    )
    assert body['error_code'] == ed.ERR_INVALID_REQUEST


def test_an_explicit_http_status_wins_even_for_a_classified_exception(app_context):
    _response, status = json_exception(
        _psycopg2_operational_error(), ed.ERR_PROVIDER_MIGRATION_FAILED, 'retry shortly',
        http_status=200,
    )

    assert status == 200


def test_an_api_405_keeps_the_allow_header(app_context):
    from werkzeug.exceptions import MethodNotAllowed

    response = json_http_exception(MethodNotAllowed(valid_methods=['GET', 'HEAD']))

    assert response.status_code == 405
    assert 'GET' in response.headers['Allow']
    assert response.get_json()['error_code'] == ed.ERR_INVALID_REQUEST


def test_an_api_404_is_json_with_the_not_found_code(app_context):
    from werkzeug.exceptions import NotFound

    response = json_http_exception(NotFound())

    assert response.status_code == 404
    assert response.content_type == 'application/json'
    assert response.get_json()['error_code'] == ed.ERR_NOT_FOUND


def test_every_api_http_error_takes_the_request_code_the_registry_names(app_context):
    from werkzeug.exceptions import BadRequest, Conflict, Forbidden, Gone, RequestEntityTooLarge
    from werkzeug.exceptions import Unauthorized

    expected = {
        BadRequest: ed.ERR_INVALID_REQUEST, Unauthorized: ed.ERR_UNAUTHORIZED,
        Forbidden: ed.ERR_FORBIDDEN, Conflict: ed.ERR_CONFLICT, Gone: ed.ERR_GONE,
        RequestEntityTooLarge: ed.ERR_PAYLOAD_TOO_LARGE,
    }
    for exception_class, code in expected.items():
        response = json_http_exception(exception_class())
        assert response.get_json()['error_code'] == code, exception_class
        assert response.status_code == ed.get_http_status(code), exception_class


def test_an_api_500_hides_its_description(app_context):
    from werkzeug.exceptions import InternalServerError

    response = json_http_exception(InternalServerError(description='secret stack text'))

    assert response.status_code == 500
    assert response.get_json()['error_code'] == ed.UNKNOWN_ERROR_CODE
    assert 'secret' not in response.get_data(as_text=True)


def test_an_http_exception_carrying_its_own_response_is_returned_untouched(app_context):
    from flask import Response
    from werkzeug.exceptions import HTTPException

    custom = HTTPException(response=Response('custom body', status=418))

    assert json_http_exception(custom) is custom
