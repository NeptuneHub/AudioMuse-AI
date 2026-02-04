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

from app_helper import get_db
from tasks.mediaserver import (
    get_available_provider_types,
    get_provider_info,
    test_provider_connection,
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

def get_providers():
    """Get all configured providers."""
    db = get_db()
    with db.cursor() as cur:
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
    elif provider_type == 'mpd':
        config_data = {
            'host': config.MPD_HOST,
            'port': config.MPD_PORT,
            'password': config.MPD_PASSWORD,
            'music_directory': config.MPD_MUSIC_DIRECTORY,
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

    try:
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
        description: Connection test result
    """
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    provider_type = data.get('provider_type')
    config_data = data.get('config', {})

    if not provider_type:
        return jsonify({'error': 'provider_type is required'}), 400

    success, message = test_provider_connection(provider_type, config_data)

    return jsonify({
        'success': success,
        'message': message,
        'provider_type': provider_type,
    })


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
