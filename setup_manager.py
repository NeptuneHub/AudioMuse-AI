import os
import json
import logging
import psycopg2
from psycopg2.extras import RealDictCursor

DEFAULT_CONFIG_TABLE = "app_config"

class SetupManager:
    def __init__(self, database_url=None):
        self.database_url = database_url or self._get_database_url()
        self.logger = logging.getLogger(__name__)

    def _get_database_url(self):
        env_url = os.environ.get("DATABASE_URL")
        if env_url:
            return env_url

        user = os.environ.get("POSTGRES_USER", "audiomuse")
        password = os.environ.get("POSTGRES_PASSWORD", "audiomusepassword")
        host = os.environ.get("POSTGRES_HOST", "postgres-service.playlist")
        port = os.environ.get("POSTGRES_PORT", "5432")
        db = os.environ.get("POSTGRES_DB", "audiomusedb")
        from urllib.parse import quote
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
                    cur.execute(f"""
                        CREATE TABLE IF NOT EXISTS {DEFAULT_CONFIG_TABLE} (
                            key TEXT PRIMARY KEY,
                            value TEXT NOT NULL,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                conn.commit()
        except Exception as exc:
            self.logger.warning(f"Could not ensure setup config table: {exc}")

    def get_raw_overrides(self):
        try:
            self.ensure_table()
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(f"SELECT key, value FROM {DEFAULT_CONFIG_TABLE}")
                    return {row["key"]: row["value"] for row in cur.fetchall()}
        except Exception as exc:
            self.logger.warning(f"Unable to read setup config overrides from DB: {exc}")
            return {}

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
                        cur.execute(
                            f"INSERT INTO {DEFAULT_CONFIG_TABLE} (key, value) VALUES (%s, %s) "
                            f"ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = CURRENT_TIMESTAMP",
                            (key, self.format_value(value)),
                        )
                conn.commit()
        except Exception as exc:
            self.logger.warning(f"Unable to save setup config values: {exc}")
            raise

    def is_setup_saved(self):
        overrides = self.get_raw_overrides()
        return bool(overrides)

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
