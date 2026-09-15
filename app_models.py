# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only

"""Authenticated model coverage for non-admin clients.

Main Features:
* Returns enablement and global counts/percentages for all setup-wizard models.
* Includes local coverage only when a server is explicitly supplied.
* Keeps version reporting separate and avoids model warmup side effects.
"""

import logging

from flask import Blueprint, jsonify, request

import app_server_context
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
    summary: Global and optional server-local model coverage.
    description: |
      Returns musicnn (MusiCNN), clap (DCLAP), lyrics and neural-fingerprint.
      Each model has enabled and global coverage; local coverage and server_id
      appear only when server or server_id is supplied. Counts describe indexed
      tracks; percentage uses the whole relevant catalogue, including for Lyrics.
      Unknown counts/percentages are null. Empty catalogues have zero percent.
      Does not load models or search indexes. See docs/model-coverage-api.md.
    parameters:
      - name: server_id
        in: query
        required: false
        schema:
          type: string
        description: Server ID or name. Omission returns global coverage only.
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
        description: Invalid or empty explicit server selection.
      401:
        description: Authentication required under the existing policy.
      403:
        description: Initial setup required under the existing policy.
      500:
        description: Coverage could not be determined; no internal detail is exposed.
    """
    try:
        try:
            server_id = app_server_context.resolve_request_server_id()
            if server_id is None and any(key in request.args for key in ('server', 'server_id')):
                return jsonify({'error': 'Invalid server selection.'}), 400
        except ValueError:
            return jsonify({'error': 'Invalid server selection.'}), 400
        return jsonify(get_model_coverage(server_id))
    except Exception:
        logger.exception('Could not determine model coverage')
        return jsonify({'error': 'Could not determine model coverage.'}), 500
