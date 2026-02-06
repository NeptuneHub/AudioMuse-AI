# app_setup.py
"""
Setup Wizard API for AudioMuse-AI

This module provides the backend API for the setup wizard and provider configuration.
It handles:
- Initial setup detection
- Provider configuration (add, update, delete, test)
- Application settings management
- Multi-provider mode enablement
"""

import logging
import json
from datetime import datetime
from flask import Blueprint, jsonify, request, render_template, redirect, url_for, g
from functools import wraps

from app_helper import get_db, detect_music_path_prefix
from tasks.mediaserver import (
    get_available_provider_types,
    get_provider_info,
    test_provider_connection,
    get_sample_tracks_from_provider,
    get_libraries_for_provider,
    PROVIDER_TYPES
)
import config

logger = logging.getLogger(__name__)

setup_bp = Blueprint('setup', __name__)


# ##############################################################################
# HELPER FUNCTIONS
# ##############################################################################

def get_setting(key, default=None):
    """Get a setting value from the database."""
    db = get_db()
    with db.cursor() as cur:
        cur.execute("SELECT value FROM app_settings WHERE key = %s", (key,))
        row = cur.fetchone()
        if row:
            return row[0]
        return default


def set_setting(key, value, category=None, description=None):
    """Set a setting value in the database."""
    db = get_db()
    with db.cursor() as cur:
        cur.execute("""
            INSERT INTO app_settings (key, value, category, description, updated_at)
            VALUES (%s, %s, %s, %s, NOW())
            ON CONFLICT (key) DO UPDATE SET
                value = EXCLUDED.value,
                category = COALESCE(EXCLUDED.category, app_settings.category),
                description = COALESCE(EXCLUDED.description, app_settings.description),
                updated_at = NOW()
        """, (key, json.dumps(value), category, description))
        db.commit()


def get_all_settings():
    """Get all settings grouped by category."""
    db = get_db()
    with db.cursor() as cur:
        cur.execute("SELECT key, value, category, description FROM app_settings ORDER BY category, key")
        rows = cur.fetchall()
        settings = {}
        for row in rows:
            key, value, category, description = row
            # Handle None category - use 'general' as default
            category = category or 'general'
            if category not in settings:
                settings[category] = {}
            settings[category][key] = {
                'value': value,
                'description': description
            }
        return settings


def is_setup_completed():
    """Check if initial setup has been completed."""
    result = get_setting('setup_completed')
    return result is True or result == 'true' or result == True


def is_multi_provider_enabled():
    """Check if multi-provider mode is enabled."""
    result = get_setting('multi_provider_enabled')
    return result is True or result == 'true' or result == True


# ##############################################################################
# PROVIDER MANAGEMENT
# ##############################################################################

def get_providers(enabled_only=False):
    """Get all configured providers.

    Args:
        enabled_only: If True, only return enabled providers
    """
    db = get_db()
    with db.cursor() as cur:
        if enabled_only:
            cur.execute("""
                SELECT id, provider_type, name, config, enabled, priority, created_at, updated_at
                FROM provider
                WHERE enabled = TRUE
                ORDER BY priority DESC, created_at ASC
            """)
        else:
            cur.execute("""
                SELECT id, provider_type, name, config, enabled, priority, created_at, updated_at
                FROM provider
                ORDER BY priority DESC, created_at ASC
            """)
        rows = cur.fetchall()
        providers = []
        for row in rows:
            provider = {
                'id': row[0],
                'provider_type': row[1],
                'name': row[2],
                'config': row[3],  # JSONB is automatically parsed
                'enabled': row[4],
                'priority': row[5],
                'created_at': row[6].isoformat() if row[6] else None,
                'updated_at': row[7].isoformat() if row[7] else None,
            }
            # Don't expose sensitive config values
            if provider['config']:
                safe_config = {}
                for k, v in provider['config'].items():
                    if k in ('password', 'token', 'api_key'):
                        safe_config[k] = '********' if v else None
                    else:
                        safe_config[k] = v
                provider['config_display'] = safe_config
            providers.append(provider)
        return providers


def get_provider_by_id(provider_id):
    """Get a provider by ID."""
    db = get_db()
    with db.cursor() as cur:
        cur.execute("""
            SELECT id, provider_type, name, config, enabled, priority
            FROM provider WHERE id = %s
        """, (provider_id,))
        row = cur.fetchone()
        if row:
            return {
                'id': row[0],
                'provider_type': row[1],
                'name': row[2],
                'config': row[3],
                'enabled': row[4],
                'priority': row[5],
            }
        return None


def add_provider(provider_type, name, config_data, enabled=True, priority=0):
    """Add a new provider configuration."""
    db = get_db()
    with db.cursor() as cur:
        cur.execute("""
            INSERT INTO provider (provider_type, name, config, enabled, priority)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
        """, (provider_type, name, json.dumps(config_data), enabled, priority))
        provider_id = cur.fetchone()[0]
        db.commit()
        return provider_id


def update_provider(provider_id, name=None, config_data=None, enabled=None, priority=None):
    """Update an existing provider configuration."""
    db = get_db()
    updates = []
    values = []

    if name is not None:
        updates.append("name = %s")
        values.append(name)
    if config_data is not None:
        updates.append("config = %s")
        values.append(json.dumps(config_data))
    if enabled is not None:
        updates.append("enabled = %s")
        values.append(enabled)
    if priority is not None:
        updates.append("priority = %s")
        values.append(priority)

    if not updates:
        return False

    updates.append("updated_at = NOW()")
    values.append(provider_id)

    with db.cursor() as cur:
        cur.execute(f"""
            UPDATE provider SET {', '.join(updates)}
            WHERE id = %s
        """, values)
        db.commit()
        return cur.rowcount > 0


def delete_provider(provider_id):
    """Delete a provider configuration."""
    db = get_db()
    with db.cursor() as cur:
        cur.execute("DELETE FROM provider WHERE id = %s", (provider_id,))
        db.commit()
        return cur.rowcount > 0


# ##############################################################################
# PROVIDER CONFIG VALIDATION
# ##############################################################################

PROVIDER_SCHEMAS = {
    'jellyfin': {
        'required': ['url', 'user_id', 'token'],
        'optional': ['music_path_prefix', 'music_libraries'],
    },
    'navidrome': {
        'required': ['url', 'user', 'password'],
        'optional': ['music_path_prefix', 'music_libraries'],
    },
    'lyrion': {
        'required': ['url'],
        'optional': ['music_path_prefix', 'music_libraries'],
    },
    'emby': {
        'required': ['url', 'user_id', 'token'],
        'optional': ['music_path_prefix', 'music_libraries'],
    },
    'localfiles': {
        'required': ['music_directory'],
        'optional': ['supported_formats', 'scan_subdirectories', 'playlist_directory', 'music_path_prefix'],
    },
}


def validate_provider_config(provider_type: str, config_data: dict) -> tuple:
    """
    Validate provider configuration data.

    Args:
        provider_type: Type of provider (jellyfin, navidrome, etc.)
        config_data: Configuration dictionary to validate

    Returns:
        Tuple of (is_valid: bool, errors: list[str])
    """
    errors = []

    if provider_type not in PROVIDER_SCHEMAS:
        return False, [f"Unknown provider type: {provider_type}"]

    schema = PROVIDER_SCHEMAS[provider_type]

    # Check required fields
    for field in schema['required']:
        if not config_data.get(field):
            errors.append(f"Missing required field: {field}")

    # Validate URL fields
    url_fields = ['url']
    for field in url_fields:
        if field in config_data and config_data[field]:
            url = config_data[field]
            if not url.startswith(('http://', 'https://')):
                errors.append(f"{field} must start with http:// or https://")

    # Validate music_directory for localfiles
    if provider_type == 'localfiles' and config_data.get('music_directory'):
        import os
        music_dir = config_data['music_directory']
        if not os.path.isabs(music_dir):
            errors.append("music_directory must be an absolute path")

    return len(errors) == 0, errors


def create_default_provider_from_env():
    """
    Create a default provider from environment variables if no providers exist.
    This enables backward compatibility with existing installations.
    """
    existing = get_providers()
    if existing:
        return None  # Providers already exist

    provider_type = config.MEDIASERVER_TYPE
    if provider_type not in PROVIDER_TYPES:
        logger.warning(f"Unknown provider type from env: {provider_type}")
        return None

    # Build config from environment variables
    config_data = {}

    if provider_type == 'jellyfin':
        config_data = {
            'url': config.JELLYFIN_URL,
            'user_id': config.JELLYFIN_USER_ID,
            'token': config.JELLYFIN_TOKEN,
        }
    elif provider_type == 'navidrome':
        config_data = {
            'url': config.NAVIDROME_URL,
            'user': config.NAVIDROME_USER,
            'password': config.NAVIDROME_PASSWORD,
        }
    elif provider_type == 'lyrion':
        config_data = {
            'url': config.LYRION_URL,
        }
    elif provider_type == 'emby':
        config_data = {
            'url': config.EMBY_URL,
            'user_id': config.EMBY_USER_ID,
            'token': config.EMBY_TOKEN,
        }
    elif provider_type == 'localfiles':
        config_data = {
            'music_directory': config.LOCALFILES_MUSIC_DIRECTORY,
            'supported_formats': config.LOCALFILES_FORMATS,
            'scan_subdirectories': config.LOCALFILES_SCAN_SUBDIRS,
            'playlist_directory': config.LOCALFILES_PLAYLIST_DIR,
        }

    name = f"{PROVIDER_TYPES[provider_type]['name']} (Default)"
    provider_id = add_provider(provider_type, name, config_data, enabled=True, priority=100)
    logger.info(f"Created default provider from environment: {provider_type} (id={provider_id})")
    return provider_id


# ##############################################################################
# API ENDPOINTS
# ##############################################################################

@setup_bp.route('/setup')
def setup_page():
    """Render the setup wizard page."""
    return render_template('setup.html', title='AudioMuse-AI - Setup', active='setup')


@setup_bp.route('/settings')
def settings_page():
    """Render the settings page."""
    return render_template('settings.html', title='AudioMuse-AI - Settings', active='settings')


@setup_bp.route('/api/setup/status', methods=['GET'])
def get_setup_status():
    """
    Get the current setup status.
    ---
    tags:
      - Setup
    responses:
      200:
        description: Setup status information
    """
    completed = is_setup_completed()
    multi_provider = is_multi_provider_enabled()
    providers = get_providers()

    # Check if we need to create default provider from env
    if not providers:
        create_default_provider_from_env()
        providers = get_providers()

    return jsonify({
        'setup_completed': completed,
        'multi_provider_enabled': multi_provider,
        'provider_count': len(providers),
        'providers': providers,
        'current_mediaserver_type': config.MEDIASERVER_TYPE,
        'app_version': config.APP_VERSION,
    })


@setup_bp.route('/api/setup/providers/types', methods=['GET'])
def get_provider_types():
    """
    Get available provider types with their configuration fields.
    ---
    tags:
      - Setup
    responses:
      200:
        description: List of provider types
    """
    types = get_available_provider_types()
    result = []
    for ptype, info in types.items():
        provider_info = get_provider_info(ptype)
        result.append({
            'type': ptype,
            'name': info['name'],
            'description': info['description'],
            'supports_user_auth': info['supports_user_auth'],
            'supports_play_history': info['supports_play_history'],
            'config_fields': provider_info.get('config_fields', []) if provider_info else [],
        })
    return jsonify(result)


@setup_bp.route('/api/setup/providers', methods=['GET'])
def list_providers():
    """
    List all configured providers.
    ---
    tags:
      - Setup
    responses:
      200:
        description: List of providers
    """
    providers = get_providers()
    return jsonify(providers)


@setup_bp.route('/api/setup/providers', methods=['POST'])
def create_provider():
    """
    Add a new provider configuration.
    ---
    tags:
      - Setup
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            properties:
              provider_type:
                type: string
              name:
                type: string
              config:
                type: object
              enabled:
                type: boolean
              priority:
                type: integer
    responses:
      201:
        description: Provider created
      400:
        description: Invalid request
    """
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    provider_type = data.get('provider_type')
    name = data.get('name')
    config_data = data.get('config', {})
    enabled = data.get('enabled', True)
    priority = data.get('priority', 0)

    if not provider_type:
        return jsonify({'error': 'provider_type is required'}), 400
    if not name:
        return jsonify({'error': 'name is required'}), 400
    if provider_type not in PROVIDER_TYPES:
        return jsonify({'error': f'Unknown provider type: {provider_type}'}), 400

    # Validate provider configuration
    is_valid, validation_errors = validate_provider_config(provider_type, config_data)
    if not is_valid:
        return jsonify({'error': 'Validation failed', 'details': validation_errors}), 400

    try:
        # Check if provider of this type already exists - upsert to prevent duplicates
        existing_providers = get_all_providers()
        existing = next((p for p in existing_providers if p['provider_type'] == provider_type), None)

        if existing:
            # Update existing provider instead of creating duplicate
            update_provider(existing['id'], name=name, config_data=config_data, enabled=enabled, priority=priority)
            logger.info(f"Updated existing provider {existing['id']} ({provider_type}) instead of creating duplicate")
            return jsonify({'id': existing['id'], 'message': 'Provider updated', 'was_update': True}), 200

        provider_id = add_provider(provider_type, name, config_data, enabled, priority)
        return jsonify({'id': provider_id, 'message': 'Provider created'}), 201
    except Exception as e:
        logger.error(f"Error creating provider: {e}")
        return jsonify({'error': str(e)}), 500


@setup_bp.route('/api/setup/providers/<int:provider_id>', methods=['PUT'])
def update_provider_endpoint(provider_id):
    """
    Update an existing provider configuration.
    ---
    tags:
      - Setup
    parameters:
      - name: provider_id
        in: path
        required: true
        schema:
          type: integer
    responses:
      200:
        description: Provider updated
      404:
        description: Provider not found
    """
    provider = get_provider_by_id(provider_id)
    if not provider:
        return jsonify({'error': 'Provider not found'}), 404

    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    # Merge config if partial update
    config_data = data.get('config')
    if config_data and isinstance(config_data, dict):
        # Don't allow updating password fields with '********'
        for key in list(config_data.keys()):
            if config_data[key] == '********':
                config_data[key] = provider['config'].get(key)

        # Validate the merged config
        merged_config = {**provider.get('config', {}), **config_data}
        is_valid, validation_errors = validate_provider_config(provider['provider_type'], merged_config)
        if not is_valid:
            return jsonify({'error': 'Validation failed', 'details': validation_errors}), 400

    success = update_provider(
        provider_id,
        name=data.get('name'),
        config_data=config_data,
        enabled=data.get('enabled'),
        priority=data.get('priority')
    )

    if success:
        return jsonify({'message': 'Provider updated'})
    return jsonify({'error': 'Update failed'}), 500


@setup_bp.route('/api/setup/providers/<int:provider_id>', methods=['DELETE'])
def delete_provider_endpoint(provider_id):
    """
    Delete a provider configuration.
    ---
    tags:
      - Setup
    parameters:
      - name: provider_id
        in: path
        required: true
        schema:
          type: integer
    responses:
      200:
        description: Provider deleted
      404:
        description: Provider not found
    """
    success = delete_provider(provider_id)
    if success:
        return jsonify({'message': 'Provider deleted'})
    return jsonify({'error': 'Provider not found'}), 404


@setup_bp.route('/api/setup/providers/<int:provider_id>/test', methods=['POST'])
def test_provider_endpoint(provider_id):
    """
    Test connection to a provider.
    ---
    tags:
      - Setup
    parameters:
      - name: provider_id
        in: path
        required: true
        schema:
          type: integer
    responses:
      200:
        description: Connection test result
    """
    provider = get_provider_by_id(provider_id)
    if not provider:
        return jsonify({'error': 'Provider not found'}), 404

    success, message = test_provider_connection(
        provider['provider_type'],
        provider['config']
    )

    return jsonify({
        'success': success,
        'message': message,
        'provider_id': provider_id,
        'provider_type': provider['provider_type'],
    })


@setup_bp.route('/api/setup/providers/test', methods=['POST'])
def test_provider_config():
    """
    Test connection with provided configuration (without saving).
    Also detects the music_path_prefix by comparing sample tracks with existing data.
    ---
    tags:
      - Setup
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            properties:
              provider_type:
                type: string
              config:
                type: object
              detect_prefix:
                type: boolean
                description: Whether to auto-detect music_path_prefix (default true)
              existing_sample_tracks:
                type: object
                description: Dict of provider_type -> list of tracks from previously tested providers
    responses:
      200:
        description: Connection test result with optional prefix detection
    """
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    provider_type = data.get('provider_type')
    config_data = data.get('config', {})
    detect_prefix = data.get('detect_prefix', True)
    existing_sample_tracks = data.get('existing_sample_tracks', {})

    if not provider_type:
        return jsonify({'error': 'provider_type is required'}), 400

    success, message = test_provider_connection(provider_type, config_data)

    result = {
        'success': success,
        'message': message,
        'provider_type': provider_type,
    }

    # If connection succeeded and prefix detection is enabled, try to detect prefix
    if success and detect_prefix:
        try:
            # Fetch sample tracks from the new provider
            sample_tracks = get_sample_tracks_from_provider(provider_type, config_data, limit=50)

            if sample_tracks:
                # Return sample tracks so frontend can cache them for subsequent provider tests
                result['sample_tracks'] = sample_tracks

                # Detect prefix by comparing with existing tracks (DB + cached tracks from previously tested providers)
                prefix_result = detect_music_path_prefix(sample_tracks, extra_sample_tracks=existing_sample_tracks)
                result['prefix_detection'] = prefix_result

                # If we detected a prefix with any matches, suggest it for auto-fill
                if prefix_result.get('matches_found', 0) > 0:
                    result['suggested_prefix'] = prefix_result.get('detected_prefix', '')
                    if prefix_result.get('confidence') in ('high', 'medium'):
                        result['message'] += f" Detected path prefix: '{prefix_result.get('detected_prefix', '')}' ({prefix_result.get('confidence')} confidence)"
                elif not prefix_result.get('had_existing_tracks', True):
                    # No existing tracks at all - this is truly the first provider
                    result['prefix_detection']['message'] = 'No existing tracks to compare with (first provider setup)'
                # If had_existing_tracks is True but matches_found is 0, keep the original message
                # ("No matching tracks found between providers") which is more accurate
            else:
                result['prefix_detection'] = {
                    'detected_prefix': '',
                    'confidence': 'none',
                    'message': 'Could not fetch sample tracks for comparison'
                }
        except Exception as e:
            logger.warning(f"Prefix detection failed for {provider_type}: {e}")
            result['prefix_detection'] = {
                'detected_prefix': '',
                'confidence': 'none',
                'message': f'Prefix detection failed: {str(e)}'
            }

    return jsonify(result)


@setup_bp.route('/api/setup/providers/libraries', methods=['POST'])
def get_provider_libraries():
    """
    Fetch available music libraries for a provider.
    Called by frontend after successful connection test.
    ---
    tags:
      - Setup
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            properties:
              provider_type:
                type: string
              config:
                type: object
    responses:
      200:
        description: List of available music libraries
    """
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    provider_type = data.get('provider_type')
    config_data = data.get('config', {})

    if not provider_type:
        return jsonify({'error': 'provider_type is required'}), 400

    try:
        libraries = get_libraries_for_provider(provider_type, config_data)
        return jsonify({'libraries': libraries})
    except Exception as e:
        logger.error(f"Error fetching libraries for {provider_type}: {e}")
        return jsonify({'error': str(e), 'libraries': []}), 500


@setup_bp.route('/api/setup/settings', methods=['GET'])
def get_settings():
    """
    Get all application settings.
    ---
    tags:
      - Setup
    responses:
      200:
        description: All settings grouped by category
    """
    settings = get_all_settings()
    return jsonify(settings)


@setup_bp.route('/api/setup/settings', methods=['PUT'])
def update_settings():
    """
    Update application settings.
    ---
    tags:
      - Setup
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            additionalProperties: true
    responses:
      200:
        description: Settings updated
    """
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    for key, value in data.items():
        set_setting(key, value)

    return jsonify({'message': 'Settings updated'})


@setup_bp.route('/api/setup/complete', methods=['POST'])
def complete_setup():
    """
    Mark the setup as complete.
    ---
    tags:
      - Setup
    responses:
      200:
        description: Setup marked as complete
    """
    set_setting('setup_completed', True, 'system', 'Whether the setup wizard has been completed')
    set_setting('setup_version', config.APP_VERSION, 'system', 'Version of the setup wizard last completed')
    return jsonify({'message': 'Setup completed', 'setup_completed': True})


@setup_bp.route('/api/setup/multi-provider', methods=['POST'])
def enable_multi_provider():
    """
    Enable or disable multi-provider mode.
    ---
    tags:
      - Setup
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            properties:
              enabled:
                type: boolean
    responses:
      200:
        description: Multi-provider mode updated
    """
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    enabled = data.get('enabled', False)
    set_setting('multi_provider_enabled', enabled, 'providers', 'Whether multi-provider mode is enabled')

    return jsonify({
        'message': f"Multi-provider mode {'enabled' if enabled else 'disabled'}",
        'multi_provider_enabled': enabled
    })


@setup_bp.route('/api/setup/primary-provider', methods=['PUT'])
def set_primary_provider():
    """
    Set the primary provider for playlist creation.
    ---
    tags:
      - Setup
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            properties:
              provider_id:
                type: integer
    responses:
      200:
        description: Primary provider set
    """
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    provider_id = data.get('provider_id')
    if provider_id is not None:
        provider = get_provider_by_id(provider_id)
        if not provider:
            return jsonify({'error': 'Provider not found'}), 404

    set_setting('primary_provider_id', provider_id, 'providers', 'ID of the primary provider for playlist creation')

    return jsonify({
        'message': 'Primary provider set',
        'primary_provider_id': provider_id
    })


@setup_bp.route('/api/setup/server-info', methods=['GET'])
def get_server_info():
    """
    Get server connection information for configuring remote workers.
    ---
    tags:
      - Setup
    responses:
      200:
        description: Server connection information
    """
    import socket
    import os
    import subprocess

    # Try to get the server's IP address
    try:
        # Get the hostname and try to resolve it
        hostname = socket.gethostname()
        host_ip = socket.gethostbyname(hostname)
        # If we get a loopback address, try to get a better one
        if host_ip.startswith('127.'):
            # Try to connect to a public DNS to get our real IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                s.connect(('8.8.8.8', 80))
                host_ip = s.getsockname()[0]
            except Exception:
                host_ip = hostname  # Fall back to hostname
            finally:
                s.close()
    except Exception:
        host_ip = 'localhost'

    # Detect GPU availability
    gpu_available = False
    gpu_name = None

    # Method 1: Check if onnxruntime-gpu CUDA provider is available
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        if 'CUDAExecutionProvider' in providers:
            gpu_available = True
    except Exception:
        pass

    # Method 2: Try nvidia-smi for GPU name (if available)
    if gpu_available:
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                gpu_name = result.stdout.strip().split('\n')[0]  # First GPU
        except Exception:
            pass

    return jsonify({
        'host': host_ip,
        'hostname': socket.gethostname() if hasattr(socket, 'gethostname') else 'unknown',
        'redis_port': os.environ.get('REDIS_PORT', '6379'),
        'postgres_port': os.environ.get('POSTGRES_PORT', '5432'),
        'postgres_host': os.environ.get('POSTGRES_HOST', 'postgres'),
        'redis_url': os.environ.get('REDIS_URL', 'redis://redis:6379/0'),
        'gpu_available': gpu_available,
        'gpu_name': gpu_name,
    })


@setup_bp.route('/api/setup/browse-directories', methods=['GET'])
def browse_directories():
    """
    Browse directories on the server for file path selection.
    ---
    tags:
      - Setup
    parameters:
      - name: path
        in: query
        required: false
        description: Directory path to list. Defaults to root paths.
        schema:
          type: string
    responses:
      200:
        description: List of directories
    """
    import os

    requested_path = request.args.get('path', '')

    # Security: prevent path traversal
    if '..' in requested_path:
        return jsonify({'error': 'Invalid path'}), 400

    directories = []

    if not requested_path:
        # Return common root paths for Docker/Linux systems
        root_paths = ['/music', '/data', '/media', '/mnt', '/home']
        for path in root_paths:
            if os.path.isdir(path):
                directories.append({
                    'name': path,
                    'path': path,
                    'is_root': True
                })
        # Also check if there are any mounted volumes at root
        try:
            for item in os.listdir('/'):
                full_path = f'/{item}'
                if os.path.isdir(full_path) and item not in ['proc', 'sys', 'dev', 'run', 'tmp', 'var', 'etc', 'usr', 'bin', 'sbin', 'lib', 'lib64', 'boot', 'root']:
                    if full_path not in [d['path'] for d in directories]:
                        directories.append({
                            'name': item,
                            'path': full_path,
                            'is_root': True
                        })
        except PermissionError:
            pass
    else:
        # List contents of the requested path
        try:
            if os.path.isdir(requested_path):
                for item in sorted(os.listdir(requested_path)):
                    full_path = os.path.join(requested_path, item)
                    if os.path.isdir(full_path):
                        # Check if directory is accessible
                        try:
                            os.listdir(full_path)
                            accessible = True
                        except PermissionError:
                            accessible = False

                        directories.append({
                            'name': item,
                            'path': full_path,
                            'accessible': accessible
                        })
        except PermissionError:
            return jsonify({'error': 'Permission denied'}), 403
        except FileNotFoundError:
            return jsonify({'error': 'Path not found'}), 404

    return jsonify({
        'current_path': requested_path or '/',
        'parent_path': os.path.dirname(requested_path) if requested_path else None,
        'directories': directories
    })
