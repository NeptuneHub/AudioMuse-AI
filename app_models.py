# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only

"""Authenticated model coverage for non-admin clients.

Main Features:
* Returns enablement and global counts/percentages for all setup-wizard models.
* Includes local coverage of the selected server, the default when none is passed.
* Keeps version reporting separate and avoids model warmup side effects.
"""

import logging

from flask import Blueprint, jsonify, request

import app_server_context
from error.error_dictionary import ERR_INVALID_REQUEST, ERR_SEARCH_FAILED
from error.responses import json_exception
from tasks.model_coverage import get_model_coverage

logger = logging.getLogger(__name__)
models_bp = Blueprint('models_bp', __name__)


@models_bp.after_app_request
def model_metadata_no_store(response):
    if request.endpoint in ('models_bp.models_api', 'version_api'):
        response.headers['Cache-Control'] = 'no-store'
    return response


@models_bp.route('/api/models', methods=['GET'])
def models_api():
    """
    Model enablement and catalogue coverage for any authenticated user.
    ---
    tags:
      - Models
    summary: Global and server-local model coverage.
    description: |
      Returns musicnn (MusiCNN), clap (DCLAP), lyrics and neural-fingerprint.
      Each model has enabled, global coverage and the local coverage of one
      server: the one selected with server or server_id, or the default server
      when none is supplied. The resolved id is echoed as server_id. Both are
      absent only when no music server is configured. Counts describe indexed
      tracks; percentage uses the whole relevant catalogue, including for Lyrics.
      Unknown counts/percentages are null. Empty catalogues have zero percent.
      Does not load models or search indexes. See docs/model-coverage-api.md.
    parameters:
      - name: server_id
        in: query
        required: false
        schema:
          type: string
        description: Server ID or name. Omitted or empty selects the default server.
      - name: server
        in: query
        required: false
        schema:
          type: string
        description: Alias for server_id; takes precedence when both are supplied.
    responses:
      200:
        description: All four models, including disabled models.
        content:
          application/json:
            schema:
              type: object
              required: [models]
              properties:
                server_id:
                  type: string
                models:
                  type: object
                  required: [musicnn, clap, lyrics, neural-fingerprint]
                  additionalProperties:
                    type: object
                    required: [enabled, global]
                    properties:
                      enabled:
                        type: boolean
                      global:
                        type: object
                        required: [count, total, percentage]
                        properties:
                          count:
                            type: integer
                            minimum: 0
                            nullable: true
                          total:
                            type: integer
                            minimum: 0
                          percentage:
                            type: number
                            minimum: 0
                            maximum: 100
                            nullable: true
                      local:
                        type: object
                        required: [count, total, percentage]
                        properties:
                          count:
                            type: integer
                            minimum: 0
                            nullable: true
                          total:
                            type: integer
                            minimum: 0
                          percentage:
                            type: number
                            minimum: 0
                            maximum: 100
                            nullable: true
      400:
        description: Unknown server selection.
      401:
        description: Authentication required under the existing policy.
      403:
        description: Initial setup required under the existing policy.
      500:
        description: Coverage could not be determined; no internal detail is exposed.
    """
    try:
        try:
            server_id, is_default = app_server_context.selected_server_scope()
        except ValueError as exc:
            return json_exception(exc, ERR_INVALID_REQUEST)
        return jsonify(get_model_coverage(server_id, include_legacy=is_default))
    except Exception as exc:
        return json_exception(exc, ERR_SEARCH_FAILED, 'Could not determine model coverage.')
