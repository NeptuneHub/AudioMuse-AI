import json
import re
from flask import request, jsonify, render_template, make_response, after_this_request
import config
from app import app, setup_manager, is_bootstrap_mode, refresh_auth_state
import restart_manager

BASIC_SERVER_FIELDS = [
    "MEDIASERVER_TYPE",
    "JELLYFIN_URL",
    "JELLYFIN_USER_ID",
    "JELLYFIN_TOKEN",
    "NAVIDROME_URL",
    "NAVIDROME_USER",
    "NAVIDROME_PASSWORD",
    "LYRION_URL",
    "EMBY_URL",
    "EMBY_USER_ID",
    "EMBY_TOKEN",
]

AUTH_FIELDS = ["AUTH_ENABLED", "AUDIOMUSE_USER", "AUDIOMUSE_PASSWORD", "API_TOKEN", "JWT_SECRET"]
SECRET_FIELDS = {"AUDIOMUSE_PASSWORD", "API_TOKEN", "JELLYFIN_TOKEN", "EMBY_TOKEN", "NAVIDROME_PASSWORD", "JWT_SECRET"}
BASIC_FIELDS = set(BASIC_SERVER_FIELDS + AUTH_FIELDS)
CONNECTION_FIELDS = {
    'DATABASE_URL',
    'POSTGRES_USER',
    'POSTGRES_PASSWORD',
    'POSTGRES_HOST',
    'POSTGRES_PORT',
    'POSTGRES_DB',
    'REDIS_URL'
}

HIDDEN_ADVANCED_FIELDS = {
    'MOOD_LABELS',
    'APP_VERSION',
}

def should_show_advanced(name):
    if name in HIDDEN_ADVANCED_FIELDS or name in CONNECTION_FIELDS:
        return False
    if name.startswith('POSTGRES_') or name.startswith('REDIS_'):
        return False
    if re.match(r'.*_STATS$', name):
        return False
    if re.match(r'.*_PATH$', name):
        return False
    return True

@app.route('/setup')
def setup_page():
    return render_template('setup.html', title='AudioMuse-AI Setup', active='setup')

@app.route('/api/setup', methods=['GET', 'POST'])
def setup_api():
    if request.method == 'GET':
        all_fields = setup_manager.get_all_fields(config)
        basic_fields = []
        advanced_fields = []
        for f in all_fields:
            if f['name'] in SECRET_FIELDS or f['name'].endswith('_API_KEY'):
                f['secret'] = True
                f['has_value'] = bool(f.get('value'))
                f['value'] = ''
            else:
                f['secret'] = False
                f['has_value'] = bool(f.get('overridden', False))

            if f['name'] in CONNECTION_FIELDS and not f.get('overridden', False):
                f['value'] = ''
                f['has_value'] = False

            if f['name'] in BASIC_FIELDS:
                basic_fields.append(f)
            elif should_show_advanced(f['name']):
                advanced_fields.append(f)

        return jsonify({
            'basic_fields': basic_fields,
            'advanced_fields': advanced_fields,
            'setup_saved': setup_manager.is_setup_complete(config),
        })

    data = request.get_json(silent=True) or {}
    config_values = data.get('config')
    if not isinstance(config_values, dict):
        return jsonify({'error': 'Missing config data'}), 400

    filtered_values = {}
    for key, value in config_values.items():
        if not isinstance(key, str) or not key.isupper():
            continue
        filtered_values[key] = value

    if not filtered_values:
        return jsonify({'error': 'No valid configuration values were provided'}), 400

    try:
        was_bootstrap = is_bootstrap_mode()
        new_server_type = filtered_values.get('MEDIASERVER_TYPE', config.MEDIASERVER_TYPE)
        if new_server_type != config.MEDIASERVER_TYPE:
            obsolete_fields = []
            if new_server_type == 'jellyfin':
                obsolete_fields = ['NAVIDROME_URL', 'NAVIDROME_USER', 'NAVIDROME_PASSWORD', 'LYRION_URL', 'EMBY_URL', 'EMBY_USER_ID', 'EMBY_TOKEN']
            elif new_server_type == 'navidrome':
                obsolete_fields = ['JELLYFIN_URL', 'JELLYFIN_USER_ID', 'JELLYFIN_TOKEN', 'LYRION_URL', 'EMBY_URL', 'EMBY_USER_ID', 'EMBY_TOKEN']
            elif new_server_type == 'lyrion':
                obsolete_fields = ['JELLYFIN_URL', 'JELLYFIN_USER_ID', 'JELLYFIN_TOKEN', 'NAVIDROME_URL', 'NAVIDROME_USER', 'NAVIDROME_PASSWORD', 'EMBY_URL', 'EMBY_USER_ID', 'EMBY_TOKEN']
            elif new_server_type == 'emby':
                obsolete_fields = ['JELLYFIN_URL', 'JELLYFIN_USER_ID', 'JELLYFIN_TOKEN', 'NAVIDROME_URL', 'NAVIDROME_USER', 'NAVIDROME_PASSWORD', 'LYRION_URL']
            if obsolete_fields:
                setup_manager.delete_config_values(obsolete_fields)
        setup_manager.save_config_values(filtered_values)
        config.refresh_config()
        refresh_auth_state()
        restart_signal_sent = restart_manager.publish_restart_request()
        restart_requested = restart_signal_sent
        require_login = was_bootstrap and not is_bootstrap_mode()
    except Exception as exc:
        app.logger.error('Setup save failed: %s', exc, exc_info=True)
        return jsonify({'error': 'Unable to save configuration. Check the server log for details.'}), 500

    response = make_response(jsonify({
        'status': 'ok',
        'saved_keys': list(filtered_values.keys()),
        'require_login': config.AUTH_ENABLED,
        'restart_requested': restart_requested,
    }), 200)

    @after_this_request
    def schedule_restart(response):
        if restart_requested:
            restart_manager.schedule_flask_restart()
        return response

    if config.AUTH_ENABLED:
        response.delete_cookie('audiomuse_jwt', samesite='Strict', path='/')
    return response
