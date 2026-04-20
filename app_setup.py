import json
import re
import types
from flask import request, jsonify, render_template, make_response, after_this_request
import config
from app import app, setup_manager
from app_helper import check_setup_needed
import restart_manager
import tasks.mediaserver as mediaserver

BASIC_SERVER_FIELDS = ["MEDIASERVER_TYPE"] + [
    field
    for fields in config.MEDIASERVER_FIELDS_BY_TYPE.values()
    for field in fields
]

AUTH_FIELDS = ["AUTH_ENABLED", "AUDIOMUSE_USER", "AUDIOMUSE_PASSWORD", "API_TOKEN", "JWT_SECRET"]
SECRET_FIELDS = {"AUDIOMUSE_PASSWORD", "API_TOKEN", "JELLYFIN_TOKEN", "EMBY_TOKEN", "NAVIDROME_PASSWORD", "JWT_SECRET", "AI_CHAT_DB_USER_PASSWORD"}
BASIC_FIELDS = set(BASIC_SERVER_FIELDS + AUTH_FIELDS)

HIDDEN_ADVANCED_FIELDS = {
    'AI_CHAT_DB_USER_NAME',
    'DATABASE_URL',
    'POSTGRES_USER',
    'POSTGRES_PASSWORD',
    'POSTGRES_HOST',
    'POSTGRES_PORT',
    'POSTGRES_DB',
    'REDIS_URL',
    'MEDIASERVER_FIELDS_BY_TYPE',
    'MEDIASERVER_OBSOLETE_FIELDS_BY_TYPE',
    'SETUP_BOOTSTRAP_EXCLUDED_KEYS',
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


def _normalize_config_value(key, value):
    if isinstance(value, str) and hasattr(config, key):
        default_value = getattr(config, key)
        if isinstance(default_value, bool):
            normalized = value.strip().lower()
            if normalized in ('1', 'true', 'yes', 'on'):
                return True
            if normalized in ('0', 'false', 'no', 'off'):
                return False
    return value


def _merge_test_config(filtered_values):
    test_config = {}
    for key in TEST_CONFIG_KEYS:
        if key in filtered_values:
            value = filtered_values[key]
            if key in SECRET_FIELDS and value == '********':
                test_config[key] = getattr(config, key, '')
            else:
                test_config[key] = _normalize_config_value(key, value)
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
    if name in HIDDEN_ADVANCED_FIELDS:
        return False
    if name.startswith('POSTGRES_') or name.startswith('REDIS_'):
        return False
    if re.match(r'.*_STATS$', name):
        return False
    if re.match(r'.*_PATH$', name):
        return False
    return True


def _get_allowed_setup_keys():
    allowed_keys = set()
    for f in setup_manager.get_all_fields(config):
        if f['name'] in BASIC_FIELDS or should_show_advanced(f['name']):
            allowed_keys.add(f['name'])
    return allowed_keys


def _has_admin_user():
    """Return True if at least one admin exists in audiomuse_users."""
    try:
        from app_helper import count_admin_users
        return count_admin_users() > 0
    except Exception as exc:
        app.logger.error(
            'Failed to determine whether an admin exists during setup page render: %s',
            exc,
            exc_info=True,
        )
        return False

@app.route('/setup')
def setup_page():
    return render_template('setup.html', title='AudioMuse-AI - Setup Wizard', active='setup')

@app.route('/api/setup', methods=['GET', 'POST'])
def setup_api():
    if request.method == 'GET':
        all_fields = setup_manager.get_all_fields(config)
        # Determine which media server fields belong to non-active types
        # so their values are hidden from the UI.
        active_server_type = getattr(config, 'MEDIASERVER_TYPE', '').strip().lower()
        inactive_server_fields = set()
        for stype, sfields in config.MEDIASERVER_FIELDS_BY_TYPE.items():
            if stype != active_server_type:
                inactive_server_fields.update(sfields)

        basic_fields = []
        advanced_fields = []
        for f in all_fields:
            if f['name'] in SECRET_FIELDS or f['name'].endswith('_API_KEY'):
                f['secret'] = True
                f['has_value'] = bool(f.get('value')) and f['name'] not in inactive_server_fields
                f['value'] = ''
            else:
                f['secret'] = False
                f['has_value'] = bool(f.get('overridden', False))

            # Blank out values for non-active server fields
            if f['name'] in inactive_server_fields:
                f['value'] = ''
                f['has_value'] = False
                f['overridden'] = False

            if f['name'] in BASIC_FIELDS:
                basic_fields.append(f)
            elif should_show_advanced(f['name']):
                advanced_fields.append(f)

        return jsonify({
            'basic_fields': basic_fields,
            'advanced_fields': advanced_fields,
            'setup_saved': not check_setup_needed(),
            'has_admin_user': _has_admin_user(),
        })

    data = request.get_json(silent=True) or {}
    config_values = data.get('config')
    if not isinstance(config_values, dict):
        return jsonify({'error': 'Missing config data'}), 400

    allowed_setup_keys = _get_allowed_setup_keys()
    filtered_values = {}
    for key, value in config_values.items():
        if not isinstance(key, str) or not key.isupper() or key not in allowed_setup_keys:
            continue
        filtered_values[key] = _normalize_config_value(key, value)

    is_test_connection = bool(data.get('test_connection', False))
    if not filtered_values and not is_test_connection:
        return jsonify({'error': 'No valid configuration values were provided'}), 400

    if not is_test_connection:
        for key, value in filtered_values.items():
            if (key in SECRET_FIELDS or key.endswith('_API_KEY')) and value == '********':
                return jsonify({'error': 'Placeholder secret values are not accepted on save. Enter the real secret or leave the field blank.'}), 400

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

        new_server_type = filtered_values.get('MEDIASERVER_TYPE', config.MEDIASERVER_TYPE)
        if isinstance(new_server_type, str):
            new_server_type = new_server_type.strip().lower()
        obsolete_fields = config.MEDIASERVER_OBSOLETE_FIELDS_BY_TYPE.get(new_server_type, [])

        auth_val = filtered_values.get('AUTH_ENABLED')
        auth_being_disabled = (auth_val is False or
            (isinstance(auth_val, str) and auth_val.strip().lower() in ('false', '0', 'no', 'off')))

        # The setup form collects the install-time admin via AUDIOMUSE_USER /
        # AUDIOMUSE_PASSWORD, but we store admins in audiomuse_users, not in
        # app_config. Pop them so they are never written to app_config.
        new_admin_user = filtered_values.pop('AUDIOMUSE_USER', None)
        new_admin_password = filtered_values.pop('AUDIOMUSE_PASSWORD', None)
        if isinstance(new_admin_user, str):
            new_admin_user = new_admin_user.strip()
        if new_admin_password == '********':
            new_admin_password = None

        # Once an admin exists in audiomuse_users, the setup wizard is no
        # longer allowed to touch admin credentials - users must be managed
        # from the Users page. Silently drop any admin fields the form
        # submitted; they were hidden in the UI but defense-in-depth.
        if _has_admin_user():
            new_admin_user = None
            new_admin_password = None

        # --- Pre-validate: simulate the post-save config state BEFORE touching the DB ---
        simulated = types.SimpleNamespace()
        for _name in vars(config):
            if _name.isupper() and not _name.startswith('_'):
                setattr(simulated, _name, getattr(config, _name))
        for key in obsolete_fields:
            setattr(simulated, key, '')
        for key, value in filtered_values.items():
            setattr(simulated, key, value)

        if not setup_manager._is_valid_server_config(simulated):
            return jsonify({'error': 'Cannot save: media server configuration is incomplete.'}), 400

        # If auth will remain enabled we need an admin after the save. That
        # admin must either already exist in audiomuse_users or be provided
        # via the form (new_admin_user + new_admin_password).
        from app_helper import count_admin_users, upsert_admin_user, get_db
        auth_will_be_enabled = not auth_being_disabled
        if isinstance(simulated.AUTH_ENABLED, str):
            auth_will_be_enabled = simulated.AUTH_ENABLED.strip().lower() == 'true'
        else:
            auth_will_be_enabled = bool(simulated.AUTH_ENABLED)
        if auth_will_be_enabled:
            try:
                existing_admins = count_admin_users()
            except Exception as exc:
                app.logger.error(
                    'Failed to count admin users during setup save: %s',
                    exc,
                    exc_info=True,
                )
                return jsonify({'error': 'Database error while verifying admin count.'}), 500
            provided_admin = bool(new_admin_user and new_admin_password)
            if existing_admins <= 0 and not provided_admin:
                return jsonify({'error': 'Cannot save: auth is enabled but no admin account was provided.'}), 400

        # Validation passed - apply changes to the database
        if obsolete_fields:
            setup_manager.delete_config_values(obsolete_fields)
        if auth_being_disabled:
            setup_manager.delete_config_values(['API_TOKEN', 'JWT_SECRET'])
            # Wipe all user accounts so disabling auth fully resets user
            # state. Re-enabling auth requires re-creating them.
            try:
                db = get_db()
                with db.cursor() as cur:
                    cur.execute("DELETE FROM audiomuse_users")
                db.commit()
            except Exception as exc:
                app.logger.error('Failed to clear audiomuse_users on auth disable: %s', exc, exc_info=True)
        elif new_admin_user and new_admin_password:
            try:
                if count_admin_users() > 0:
                    return jsonify({'error': 'Cannot save: an admin account already exists.'}), 400
            except Exception as exc:
                app.logger.error(
                    'Unable to verify existing admin accounts before setup save: %s',
                    exc,
                    exc_info=True,
                )
                return jsonify({'error': 'Unable to verify existing admin accounts. Check the server log and try again later.'}), 500
            ok, err = upsert_admin_user(new_admin_user, new_admin_password)
            if not ok:
                return jsonify({'error': err or 'Failed to save admin account.'}), 400

        setup_manager.save_config_values(filtered_values)
        config.refresh_config()

        restart_manager.publish_restart_request()
        restart_requested = True
    except Exception as exc:
        app.logger.error('Setup save failed: %s', exc, exc_info=True)
        if is_test_connection:
            return jsonify({'error': 'Unable to get top player song. Check the server log for details.'}), 500
        return jsonify({'error': 'Unable to save configuration. Check the server log for details.'}), 500

    response = make_response(jsonify({
        'status': 'ok',
        'saved_keys': list(filtered_values.keys()),
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
