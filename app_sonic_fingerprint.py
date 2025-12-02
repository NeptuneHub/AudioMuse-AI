# app_sonic_fingerprint.py
from flask import Blueprint, jsonify, request, render_template
import logging

from tasks.sonic_fingerprint_manager import generate_sonic_fingerprint
from tasks.mediaserver import resolve_emby_jellyfin_user # Import the new resolver function
from config import MEDIASERVER_TYPE, JELLYFIN_URL, JELLYFIN_USER_ID, JELLYFIN_TOKEN, NAVIDROME_USER, NAVIDROME_PASSWORD # Import configs

logger = logging.getLogger(__name__)

# Create a blueprint for the new feature
sonic_fingerprint_bp = Blueprint('sonic_fingerprint_bp', __name__, template_folder='../templates')

@sonic_fingerprint_bp.route('/sonic_fingerprint', methods=['GET'])
def sonic_fingerprint_page():
    """
    Serves the frontend page for the Sonic Fingerprint feature.
    ---
    tags:
      - UI
    responses:
      200:
        description: HTML content of the Sonic Fingerprint page.
        content:
          text/html:
            schema:
              type: string
    """
    try:
        # The default user info will now be fetched by an API call from the frontend
        return render_template('sonic_fingerprint.html', mediaserver_type=MEDIASERVER_TYPE, title = 'AudioMuse-AI - Sonic Fingerprint', active='sonic_fingerprint')
    except Exception as e:
         logger.error(f"Error rendering sonic_fingerprint.html: {e}", exc_info=True)
         return "Sonic Fingerprint page not implemented yet. Use the API at /api/sonic_fingerprint/generate"

@sonic_fingerprint_bp.route('/api/config/defaults', methods=['GET'])
def get_media_server_defaults():
    """
    Provides default credentials from the server configuration based on the media server type.
    This is intended for trusted network environments to pre-populate frontend forms.
    ---
    tags:
      - Configuration
    responses:
      200:
        description: A JSON object with default credentials for the configured media server.
        content:
          application/json:
            schema:
              type: object
    """
    # MODIFIED: Removed the security credentials from the response.
    # We only return the user ID/username to pre-fill forms, but not the tokens/passwords.
    if MEDIASERVER_TYPE == 'jellyfin':
        return jsonify({
            "default_user_id": JELLYFIN_USER_ID,
        })
    elif MEDIASERVER_TYPE == 'navidrome':
        return jsonify({
            "default_user": NAVIDROME_USER,
        })
    return jsonify({})


@sonic_fingerprint_bp.route('/api/sonic_fingerprint/generate', methods=['GET', 'POST'])
def generate_sonic_fingerprint_endpoint():
    """
    Generates a sonic fingerprint based on a user's listening habits.
    Accepts both GET and POST requests for backward compatibility.
    ---
    tags:
      - Sonic Fingerprint
    parameters:
      - name: n
        in: query
        type: integer
        required: false
        description: (For GET requests) The number of results to return.
      - name: jellyfin_user_identifier
        in: query
        type: string
        required: false
        description: (For GET requests) The Jellyfin Username or User ID.
      - name: jellyfin_token
        in: query
        type: string
        required: false
        description: (For GET requests) The Jellyfin API Token.
      - name: navidrome_user
        in: query
        type: string
        required: false
        description: (For GET requests) The Navidrome username.
      - name: navidrome_password
        in: query
        type: string
        required: false
        description: (For GET requests) The Navidrome password.
    requestBody:
      description: For POST requests, provide parameters in the JSON body.
      required: false
      content:
        application/json:
          schema:
            type: object
            properties:
              n:
                type: integer
                description: The number of results to return.
              jellyfin_user_identifier:
                type: string
                description: The Jellyfin Username or User ID.
              jellyfin_token:
                type: string
                description: The Jellyfin API Token.
              navidrome_user:
                type: string
                description: The Navidrome username.
              navidrome_password:
                type: string
                description: The Navidrome password.
    responses:
      200:
        description: A list of recommended tracks based on the sonic fingerprint.
      400:
        description: Bad Request - Missing credentials or invalid payload.
      500:
        description: Server error during generation.
    """
    # Local import to prevent circular dependency
    from app_helper import get_score_data_by_ids

    try:
        if request.method == 'POST':
            data = request.get_json()
            if not data:
                return jsonify({"error": "Invalid JSON payload"}), 400
        else:  # GET request
            data = request.args

        num_results = data.get('n')
        if num_results is not None:
            try:
                num_results = int(num_results)
            except (ValueError, TypeError):
                return jsonify({"error": "Parameter 'n' must be a valid integer."}), 400
        
        user_creds = {}
        if MEDIASERVER_TYPE == 'jellyfin':
            user_identifier = data.get('jellyfin_user_identifier')
            if not user_identifier:
                return jsonify({"error": "Jellyfin User Identifier is required."}), 400

            token = data.get('jellyfin_token') or JELLYFIN_TOKEN
            
            if not token:
                return jsonify({"error": "Jellyfin API Token is required. Please provide one or set it in the server configuration."}), 400

            logger.info(f"Resolving Jellyfin user identifier: '{user_identifier}'")
            resolved_user_id = resolve_emby_jellyfin_user(user_identifier, token)
            if not resolved_user_id:
                return jsonify({"error": f"Could not resolve Jellyfin user '{user_identifier}'."}), 400
            
            logger.info(f"Resolved Jellyfin user ID: '{resolved_user_id}'")
            user_creds['user_id'] = resolved_user_id
            user_creds['token'] = token

        elif MEDIASERVER_TYPE == 'navidrome':
            user_creds['user'] = data.get('navidrome_user') or NAVIDROME_USER
            user_creds['password'] = data.get('navidrome_password') or NAVIDROME_PASSWORD
            if not user_creds['user'] or not user_creds['password']:
                return jsonify({"error": "Navidrome username and password are required. Please provide them or set them in the server configuration."}), 400
        
        fingerprint_results = generate_sonic_fingerprint(
            num_neighbors=num_results,
            user_creds=user_creds
        )

        if not fingerprint_results:
            return jsonify([])

        result_ids = [r['item_id'] for r in fingerprint_results]
        details_list = get_score_data_by_ids(result_ids)
        
        details_map = {d['item_id']: d for d in details_list}
        distance_map = {r['item_id']: r['distance'] for r in fingerprint_results}

        final_results = []
        for res_id in result_ids:
            if res_id in details_map:
                track_info = details_map[res_id]

                # Construct result object
                result_obj = {
                    "item_id": track_info['item_id'],
                    "title": track_info['title'],
                    "author": track_info['author'], # Legacy/Best Artist
                    "song_artist": track_info.get('song_artist') or track_info['author'], # Prefer explicit Song Artist, fallback to author
                    "album_artist": track_info.get('album_artist'),
                    "album": track_info.get('album'),
                    "distance": distance_map[res_id]
                }

                # Construct direct play URL if using Jellyfin
                if MEDIASERVER_TYPE == 'jellyfin':
                     # The token used for the request (either passed or default)
                    current_token = user_creds.get('token')
                    if current_token:
                         # Universal audio stream URL
                         # Note: using 'universal' endpoint is often safer for direct play across devices,
                         # or /Audio/{Id}/stream. We'll use the stream endpoint with static=true
                         # which often redirects to the file or transcode if needed.
                         # Adding api_key is standard for Jellyfin API auth in URL.
                         stream_url = f"{JELLYFIN_URL}/Audio/{track_info['item_id']}/stream?api_key={current_token}&static=true"
                         result_obj['stream_url'] = stream_url

                final_results.append(result_obj)

        return jsonify(final_results)
    except Exception as e:
        logger.error(f"Error in sonic_fingerprint endpoint: {e}", exc_info=True)
        return jsonify({"error": "An unexpected error occurred while generating the sonic fingerprint."}), 500
