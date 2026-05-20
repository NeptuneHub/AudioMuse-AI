"""Smart Listening Sessions blueprint."""

import logging

from flask import Blueprint, jsonify, render_template, request

from tasks.smart_session_builder import (
    SMART_SESSION_DEFAULT_LENGTH,
    SMART_SESSION_MAX_LENGTH,
    SmartSessionValidationError,
    build_smart_session_preview,
    export_smart_session_playlist,
    get_smart_session_capabilities,
)

logger = logging.getLogger(__name__)

smart_sessions_bp = Blueprint('smart_sessions_bp', __name__, template_folder='../templates')


@smart_sessions_bp.route('/smart_sessions', methods=['GET'])
def smart_sessions_page():
    """Smart Listening Sessions UI page."""
    from config import APP_VERSION

    return render_template(
        'smart_sessions.html',
        title='AudioMuse-AI - Smart Listening Sessions',
        active='smart_sessions',
        app_version=APP_VERSION,
        default_length=SMART_SESSION_DEFAULT_LENGTH,
        max_length=SMART_SESSION_MAX_LENGTH,
    )


@smart_sessions_bp.route('/api/smart_sessions/capabilities', methods=['GET'])
def smart_sessions_capabilities_api():
    """Return available Smart Listening Sessions sources and limits."""
    return jsonify(get_smart_session_capabilities())


@smart_sessions_bp.route('/api/smart_sessions/preview', methods=['POST'])
def smart_sessions_preview_api():
    """Validate and preview a Smart Listening Session."""
    try:
        preview = build_smart_session_preview(request.get_json(silent=True))
        return jsonify(preview), 200
    except SmartSessionValidationError as exc:
        return jsonify({"error": str(exc), "tracks": [], "warnings": []}), 400
    except Exception:
        logger.exception("Smart Listening Sessions preview failed")
        return jsonify({"error": "An internal error occurred while building the session.", "tracks": []}), 500


@smart_sessions_bp.route('/api/smart_sessions/export', methods=['POST'])
def smart_sessions_export_api():
    """Create a media-server playlist from an approved Smart Session queue."""
    try:
        result = export_smart_session_playlist(request.get_json(silent=True))
        return jsonify(result), 201
    except SmartSessionValidationError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        logger.exception("Smart Listening Sessions export failed")
        return jsonify({"error": "An error occurred while creating the playlist on the media server."}), 500