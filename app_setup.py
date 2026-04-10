import json
import re
from flask import request, jsonify, render_template
import config
from app import app, setup_manager

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

AUTH_FIELDS = ["AUTH_ENABLED", "AUDIOMUSE_USER", "AUDIOMUSE_PASSWORD", "API_TOKEN"]
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
    return render_template('setup.html', title='AudioMuse-AI Setup')

@app.route('/api/setup', methods=['GET', 'POST'])
def setup_api():
    if request.method == 'GET':
        all_fields = setup_manager.get_all_fields(config)
        basic_fields = [f for f in all_fields if f['name'] in BASIC_FIELDS]
        advanced_fields = [f for f in all_fields if f['name'] not in BASIC_FIELDS and should_show_advanced(f['name'])]
        return jsonify({
            'basic_fields': basic_fields,
            'advanced_fields': advanced_fields,
            'setup_saved': setup_manager.is_setup_saved(),
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
        setup_manager.save_config_values(filtered_values)
        config.refresh_config()
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500

    return jsonify({'status': 'ok', 'saved_keys': list(filtered_values.keys())})
