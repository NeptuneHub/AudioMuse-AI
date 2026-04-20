import os
import json
import logging
import psycopg2
from argon2 import PasswordHasher
from argon2 import exceptions as argon2_exceptions
import config
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
    'EMBY_TOKEN'
}
AUTH_FIELDS = {
    'AUTH_ENABLED',
    'AUDIOMUSE_USER',
    'AUDIOMUSE_PASSWORD',
    'API_TOKEN'
}

class SetupManager:
    def __init__(self, database_url=None):
        self.database_url = database_url or self._get_database_url()
        self.logger = logging.getLogger(__name__)
        self._password_hasher = PasswordHasher()

    def _get_database_url(self):
        env_url = os.environ.get("DATABASE_URL")
        if env_url:
            return env_url

        user = os.environ.get("POSTGRES_USER", "audiomuse")
        password = os.environ.get("POSTGRES_PASSWORD", "audiomusepassword")
        host = os.environ.get("POSTGRES_HOST", "postgres-service.playlist")
        port = os.environ.get("POSTGRES_PORT", "5432")
        db = os.environ.get("POSTGRES_DB", "audiomusedb")
        user_escaped = quote(user, safe='')
        password_escaped = quote(password, safe='')
        return f"postgresql://{user_escaped}:{password_escaped}@{host}:{port}/{db}"

    def get_connection(self):
        if not self.database_url:
            raise RuntimeError("DATABASE_URL is not configured")
        return psycopg2.connect(self.database_url, connect_timeout=30)

    def ensure_table(self):
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # Serialize concurrent app_config/table creation across
                    # multiple processes (Flask workers, RQ workers, setup_manager).
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
        except Exception as exc:
            self.logger.warning(f"Could not ensure setup config table: {exc}")

    def config_table_exists(self):
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = %s)",
                        (DEFAULT_CONFIG_TABLE,),
                    )
                    return bool(cur.fetchone()[0])
        except Exception as exc:
            self.logger.warning(f"Unable to determine app_config table existence: {exc}")
            return False

    def get_raw_overrides(self, ensure_table=True):
        try:
            if ensure_table:
                self.ensure_table()
            elif not self.config_table_exists():
                return {}
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(f"SELECT key, value FROM {DEFAULT_CONFIG_TABLE}")
                    return {row["key"]: row["value"] for row in cur.fetchall()}
        except Exception as exc:
            self.logger.warning(f"Unable to read setup config overrides from DB: {exc}")
            return {}

    def is_config_table_empty(self):
        try:
            self.ensure_table()
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(f"SELECT EXISTS (SELECT 1 FROM {DEFAULT_CONFIG_TABLE})")
                    return not cur.fetchone()[0]
        except Exception as exc:
            self.logger.warning(f"Unable to determine app_config state: {exc}")
            return True

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
            'http://your-lyrion-server'
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

    SERVER_REQUIRED_FIELDS = config.MEDIASERVER_FIELDS_BY_TYPE

    def _is_valid_server_config(self, config_module):
        media_type = getattr(config_module, 'MEDIASERVER_TYPE', '').strip().lower()
        if media_type not in self.SERVER_REQUIRED_FIELDS:
            return False
        return all(
            self._is_valid_string(getattr(config_module, field, ''))
            for field in self.SERVER_REQUIRED_FIELDS[media_type]
        )

    def _is_valid_auth_config(self, config_module):
        enabled = getattr(config_module, 'AUTH_ENABLED', True)
        if isinstance(enabled, str):
            enabled = enabled.strip().lower() == 'true'
        if not enabled:
            return True

        # When auth is enabled, only username/password are mandatory.
        # API_TOKEN is optional and does not control setup completeness.
        return all(
            self._is_valid_string(getattr(config_module, field, ''))
            for field in ['AUDIOMUSE_USER', 'AUDIOMUSE_PASSWORD']
        )

    def is_valid_env_config(self, config_module):
        return (
            self._is_valid_server_config(config_module)
            and self._is_valid_auth_config(config_module)
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
            raise ValueError("Expected a dictionary of config values")
        self.ensure_table()
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    for key, value in values.items():
                        if key == 'AUDIOMUSE_PASSWORD' and isinstance(value, str) and value and value != '********' and not self._is_argon2_password_hash(value):
                            try:
                                value = self._password_hasher.hash(value)
                            except argon2_exceptions.HashingError as exc:
                                self.logger.error(f"Unable to hash AUDIOMUSE_PASSWORD: {exc}", exc_info=True)
                                raise
                        cur.execute(
                            f"INSERT INTO {DEFAULT_CONFIG_TABLE} (key, value) VALUES (%s, %s) "
                            f"ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = CURRENT_TIMESTAMP",
                            (key, self.format_value(value)),
                        )
                conn.commit()
        except Exception as exc:
            self.logger.warning(f"Unable to save setup config values: {exc}")
            raise
        try:
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
                        f"DELETE FROM {DEFAULT_CONFIG_TABLE} WHERE key = ANY(%s)",
                        (list(keys),)
                    )
                conn.commit()
        except Exception as exc:
            self.logger.warning(f"Unable to delete setup config values: {exc}")
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
            fields.append({
                "name": name,
                "default": self.format_value(default_value),
                "value": self.format_value(value),
                "type": type(default_value).__name__,
                "overridden": overridden,
            })
        return fields
