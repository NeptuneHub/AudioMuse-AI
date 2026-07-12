# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Persist first-run setup and runtime config overrides in the database.

Backs the setup wizard by storing config values in the app_config table so they
survive restarts and take precedence over environment defaults from config.py.

Main Features:
* Reads and writes typed config overrides (casting stored strings back to the
  default's type) and can bootstrap the table from valid environment config
  when it is empty.
* Hashes secrets with Argon2, skips re-hashing values already hashed, treats
  placeholder values as unset, and reports whether server/auth setup is complete.
"""

import os
import json
import logging
import psycopg2
from argon2 import PasswordHasher
from argon2 import exceptions as argon2_exceptions
from psycopg2.extras import RealDictCursor
from urllib.parse import quote

DEFAULT_CONFIG_TABLE = "app_config"
BASIC_SERVER_FIELDS = {
    'MEDIASERVER_TYPE',
    'JELLYFIN_URL',
    'JELLYFIN_USER_ID',
    'JELLYFIN_TOKEN',
    'NAVIDROME_URL',
    'NAVIDROME_USER',
    'NAVIDROME_PASSWORD',
    'LYRION_URL',
    'EMBY_URL',
    'EMBY_USER_ID',
    'EMBY_TOKEN',
    'PLEX_URL',
    'PLEX_TOKEN',
}
AUTH_FIELDS = {'AUTH_ENABLED', 'AUDIOMUSE_USER', 'AUDIOMUSE_PASSWORD', 'API_TOKEN'}


class SetupManager:
    def __init__(self, database_url=None):
        self.database_url = database_url or self._get_database_url()
        self.logger = logging.getLogger(__name__)
        self._password_hasher = PasswordHasher()

    def _get_database_url(self):
        env_url = os.environ.get("DATABASE_URL")
        if env_url:
            return env_url

        import sys

        if getattr(sys, "frozen", False):
            return None

        import config

        user = os.environ.get("POSTGRES_USER") or config.POSTGRES_USER
        password = os.environ.get("POSTGRES_PASSWORD") or config.POSTGRES_PASSWORD
        host = os.environ.get("POSTGRES_HOST") or config.POSTGRES_HOST
        port = os.environ.get("POSTGRES_PORT") or config.POSTGRES_PORT
        db = os.environ.get("POSTGRES_DB") or config.POSTGRES_DB
        user_escaped = quote(user, safe='')
        password_escaped = quote(password, safe='')
        return f"postgresql://{user_escaped}:{password_escaped}@{host}:{port}/{db}"

    def get_connection(self):
        if not self.database_url:
            raise RuntimeError("DATABASE_URL is not configured")
        return psycopg2.connect(self.database_url, connect_timeout=30)

    def ensure_table(self):
        if self.database_url is None:
            return
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT pg_advisory_lock(726354821)")
                    try:
                        cur.execute(
                            "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = %s)",
                            (DEFAULT_CONFIG_TABLE,),
                        )
                        if not cur.fetchone()[0]:
                            cur.execute(f"""
                                CREATE TABLE {DEFAULT_CONFIG_TABLE} (
                                    key TEXT PRIMARY KEY,
                                    value TEXT NOT NULL,
                                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                                )
                            """)
                    finally:
                        cur.execute("SELECT pg_advisory_unlock(726354821)")
                conn.commit()
        except Exception:
            self.logger.warning("Could not ensure setup config table", exc_info=True)

    def config_table_exists(self):
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = %s)",
                        (DEFAULT_CONFIG_TABLE,),
                    )
                    return bool(cur.fetchone()[0])
        except Exception:
            self.logger.warning("Unable to determine app_config table existence", exc_info=True)
            return False

    def get_raw_overrides(self, ensure_table=True):
        if self.database_url is None:
            return {}
        try:
            if ensure_table:
                self.ensure_table()
            elif not self.config_table_exists():
                return {}
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(f"SELECT key, value FROM {DEFAULT_CONFIG_TABLE}")
                    return {row["key"]: row["value"] for row in cur.fetchall()}
        except Exception:
            self.logger.warning("Unable to read setup config overrides from DB", exc_info=True)
            return {}

    def is_config_table_empty(self):
        try:
            self.ensure_table()
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(f"SELECT EXISTS (SELECT 1 FROM {DEFAULT_CONFIG_TABLE})")
                    return not cur.fetchone()[0]
        except Exception:
            self.logger.warning("Unable to determine app_config state", exc_info=True)
            return True

    def get_default_music_server(self):
        """The music_servers default row, the single source of truth for the
        media-server settings config projects onto its globals. None when the
        table does not exist yet (pre-migration boot) or on any read problem,
        so importing config can never fail because of the registry."""
        if self.database_url is None:
            return None
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT to_regclass('public.music_servers') IS NOT NULL")
                    if not cur.fetchone()[0]:
                        return None
                    cur.execute(
                        "SELECT server_type, creds, music_libraries FROM music_servers "
                        "WHERE is_default LIMIT 1"
                    )
                    row = cur.fetchone()
            if not row:
                return None
            creds = row[1]
            if isinstance(creds, str):
                try:
                    creds = json.loads(creds)
                except ValueError:
                    creds = {}
            return {
                'server_type': row[0],
                'creds': creds or {},
                'music_libraries': row[2] or '',
            }
        except Exception:
            self.logger.warning("Unable to read default music server from registry", exc_info=True)
            return None

    def _looks_like_placeholder(self, value):
        if not isinstance(value, str):
            return False
        normalized = value.strip().lower()
        if not normalized:
            return True
        placeholders = [
            'your_',
            'your default',
            'your-default',
            'no-key-needed',
            'your-gemini-api-key-here',
            'your_jellyfin_url',
            'your_navidrome_url',
            'your_lyrion_url',
            'your_navidrome_user',
            'your_navidrome_password',
            'your_default_user_id',
            'your_default_token',
            'http://your_jellyfin_server',
            'http://your-navidrome-server',
            'http://your-lyrion-server',
        ]
        for placeholder in placeholders:
            if placeholder in normalized:
                return True
        return False

    def _get_env_config_values(self, config_module):
        values = {}
        excluded_keys = getattr(config_module, 'SETUP_BOOTSTRAP_EXCLUDED_KEYS', set())
        for name, default_value in sorted(vars(config_module).items()):
            if not name.isupper() or name.startswith('_'):
                continue
            if name in excluded_keys:
                continue
            values[name] = default_value
        return values

    def _is_valid_string(self, value):
        if not isinstance(value, str):
            return False
        stripped = value.strip()
        if not stripped:
            return False
        return not self._looks_like_placeholder(value)

    @property
    def server_required_fields(self):
        import config

        return config.MEDIASERVER_FIELDS_BY_TYPE

    def _is_valid_server_config(self, config_module):
        media_type = getattr(config_module, 'MEDIASERVER_TYPE', '').strip().lower()
        if media_type not in self.server_required_fields:
            return False
        return all(
            self._is_valid_string(getattr(config_module, field, ''))
            for field in self.server_required_fields[media_type]
        )

    def _is_valid_auth_config(self, config_module):
        enabled = getattr(config_module, 'AUTH_ENABLED', True)
        if isinstance(enabled, str):
            enabled = enabled.strip().lower() == 'true'
        if not enabled:
            return True

        return all(
            self._is_valid_string(getattr(config_module, field, ''))
            for field in ['AUDIOMUSE_USER', 'AUDIOMUSE_PASSWORD']
        )

    def is_valid_env_config(self, config_module):
        return self._is_valid_server_config(config_module) and self._is_valid_auth_config(
            config_module
        )

    def bootstrap_env_config_if_empty(self, config_module):
        if not self.is_config_table_empty():
            return False
        if not self.is_valid_env_config(config_module):
            return False
        values = self._get_env_config_values(config_module)
        self.save_config_values(values)
        return True

    def _is_argon2_password_hash(self, value):
        return isinstance(value, str) and value.startswith('$argon2')

    def cast_value(self, default_value, stored_value):
        if isinstance(default_value, bool):
            return str(stored_value).strip().lower() in ("1", "true", "yes", "on")
        if isinstance(default_value, int):
            try:
                return int(stored_value)
            except ValueError:
                return default_value
        if isinstance(default_value, float):
            try:
                return float(stored_value)
            except ValueError:
                return default_value
        if isinstance(default_value, (list, dict)):
            try:
                return json.loads(stored_value)
            except Exception:
                return default_value
        return stored_value

    def format_value(self, value):
        if isinstance(value, (list, dict)):
            return json.dumps(value)
        return str(value)

    def save_config_values(self, values):
        if not isinstance(values, dict):
            raise TypeError("Expected a dictionary of config values")
        try:
            import config as _config

            media_keys = getattr(_config, 'MEDIASERVER_CONFIG_KEYS', frozenset())
        except Exception:
            media_keys = frozenset()
        values = {k: v for k, v in values.items() if k not in media_keys}
        if not values:
            return
        self.ensure_table()
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    for key, value in values.items():
                        if (
                            key == 'AUDIOMUSE_PASSWORD'
                            and isinstance(value, str)
                            and value
                            and value != '********'
                            and not self._is_argon2_password_hash(value)
                        ):
                            try:
                                value = self._password_hasher.hash(value)
                            except argon2_exceptions.HashingError:
                                self.logger.exception(
                                    "Unable to hash AUDIOMUSE_PASSWORD"
                                )
                                raise
                        cur.execute(
                            f"INSERT INTO {DEFAULT_CONFIG_TABLE} (key, value) VALUES (%s, %s) "
                            f"ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = CURRENT_TIMESTAMP",
                            (key, self.format_value(value)),
                        )
                conn.commit()
        except Exception:
            self.logger.warning("Unable to save setup config values", exc_info=True)
            raise
        try:
            import config

            if hasattr(config, 'refresh_config'):
                config.refresh_config()
        except Exception as exc:
            self.logger.warning(f'Failed to refresh config after saving values: {exc}')

    def delete_config_values(self, keys):
        if not keys:
            return
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f"DELETE FROM {DEFAULT_CONFIG_TABLE} WHERE key = ANY(%s)", (list(keys),)
                    )
                conn.commit()
        except Exception:
            self.logger.warning("Unable to delete setup config values", exc_info=True)
            raise

    def is_setup_complete(self, config_module):
        return self.is_valid_env_config(config_module)

    def get_all_fields(self, config_module):
        raw = self.get_raw_overrides()
        fields = []
        for name, default_value in sorted(vars(config_module).items()):
            if not name.isupper() or name.startswith("_"):
                continue
            value = raw.get(name, None)
            if value is not None:
                value = self.cast_value(default_value, value)
                overridden = True
            else:
                value = default_value
                overridden = False
            fields.append(
                {
                    "name": name,
                    "default": self.format_value(default_value),
                    "value": self.format_value(value),
                    "type": type(default_value).__name__,
                    "overridden": overridden,
                }
            )
        return fields


setup_manager = SetupManager()
