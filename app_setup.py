import json
import re
from flask import request, jsonify, render_template, make_response, after_this_request
import config
from app import app, setup_manager, is_bootstrap_mode, refresh_auth_state
import restart_manager
import tasks.mediaserver as mediaserver

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
    'TEMP_DIR',
    'CLAP_AUDIO_FMAX',
    'CLAP_AUDIO_FMIN',
    'CLAP_AUDIO_HOP_LENGTH',
    'CLAP_AUDIO_MEL_TRANSPOSE',
    'CLAP_AUDIO_N_FFT',
    'CLAP_AUDIO_N_MELS',
    'CLAP_CATEGORY_WEIGHTS',
    'CLAP_CATEGORY_WEIGHTS_DEFAULT',
    'CLAP_AUDIO_EMBEDDING_DIMENSION',
    'CLAP_OTHER_FEATURES_REDIS_KEY',
    'INDEX_NAME',
    'MINIBATCH_KMEANS_PROCESSING_BATCH_SIZE',
    'MOOD_CENTROIDS_FILE',
    'MPD_HOST',
    'MPD_MUSIC_DIRECTORY',
    'MPD_PASSWORD',
    'MPD_PORT',
    'MULAN_CATEGORY_WEIGHTS',
    'MULAN_CATEGORY_WEIGHTS_DEFAULT',
    'MULAN_EMBEDDING_DIMENSION',
    'MULAN_ENABLED',
    'MULAN_MODEL_DIR',
    'MULAN_TEXT_SEARCH_WARMUP_DURATION',
    'MULAN_TOP_QUERIES_COUNT',
    'OTHER_FEATURE_LABELS',
    'STRATIFIED_GENRES',
    'TEMPO_MAX_BPM',
    'TEMPO_MIN_BPM',
    'USE_MINIBATCH_KMEANS',
    'JWT_SECRET',
}

TEST_CONFIG_KEYS = set(BASIC_SERVER_FIELDS + ['MUSIC_LIBRARIES'])


def _merge_test_config(filtered_values):
    test_config = {}
    for key in TEST_CONFIG_KEYS:
        if key in filtered_values:
            value = filtered_values[key]
            if key in SECRET_FIELDS and value == '********':
                test_config[key] = getattr(config, key, '')
            else:
                test_config[key] = value
        else:
            test_config[key] = getattr(config, key, '')
    if 'MEDIASERVER_TYPE' in test_config and isinstance(test_config['MEDIASERVER_TYPE'], str):
        test_config['MEDIASERVER_TYPE'] = test_config['MEDIASERVER_TYPE'].lower()
    return test_config


def _patch_config_for_test(test_config):
    original_config = {}
    for key, value in test_config.items():
        original_config[key] = getattr(config, key, None)
        setattr(config, key, value)
    return original_config


def _restore_config(original_config):
    for key, value in original_config.items():
        setattr(config, key, value)


def _test_media_server_connection(filtered_values):
    test_config = _merge_test_config(filtered_values)
    original_config = _patch_config_for_test(test_config)
    try:
        media_type = test_config.get('MEDIASERVER_TYPE', 'jellyfin')
        probe_limit = getattr(config, 'PROBE_TOP_PLAYED_LIMIT', 1)
        items = mediaserver.get_top_played_songs(probe_limit)
        if not items:
            raise ValueError(f'Possible problem in connecting to {media_type.capitalize()}. No top-played songs were returned')
        return {
            'type': media_type,
            'probe_count': len(items),
            'probe_limit_hit': probe_limit and len(items) >= probe_limit,
        }
    except Exception as exc:
        raise ValueError(str(exc) or 'Media server connection test failed.') from exc
    finally:
        _restore_config(original_config)


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

    is_test_connection = bool(data.get('test_connection', False))
    if not filtered_values and not is_test_connection:
        return jsonify({'error': 'No valid configuration values were provided'}), 400

    try:
        if is_test_connection:
            result = _test_media_server_connection(filtered_values)
            return jsonify({
                'status': 'ok',
                'test_connection': True,
                'media_server': result['type'],
                'probe_count': result['probe_count'],
                'probe_limit_hit': result.get('probe_limit_hit', False),
            }), 200

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
        if is_test_connection:
            return jsonify({'error': 'Unable to get top player song. Check the server log for details.'}), 500
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
