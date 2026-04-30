import importlib
import sys
import types

from env_utils import get_env


def test_get_env_prefers_file_value(tmp_path, monkeypatch):
    secret_file = tmp_path / "openai_api_key"
    secret_file.write_text("from-file\n", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")
    monkeypatch.setenv("OPENAI_API_KEY_FILE", str(secret_file))

    assert get_env("OPENAI_API_KEY", "") == "from-file"


def test_get_env_falls_back_to_env_when_file_missing(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")
    monkeypatch.setenv("OPENAI_API_KEY_FILE", "/does/not/exist")

    assert get_env("OPENAI_API_KEY", "") == "from-env"


def test_get_env_logs_warning_when_file_read_fails(monkeypatch, caplog):
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")
    monkeypatch.setenv("OPENAI_API_KEY_FILE", "/does/not/exist")

    with caplog.at_level("WARNING"):
        assert get_env("OPENAI_API_KEY", "") == "from-env"

    assert "Failed to read OPENAI_API_KEY_FILE" in caplog.text


def test_config_reads_secret_values_from_file(monkeypatch, tmp_path):
    jwt_secret_file = tmp_path / "jwt_secret"
    jwt_secret_file.write_text("jwt-from-file\n", encoding="utf-8")
    pg_password_file = tmp_path / "postgres_password"
    pg_password_file.write_text("postgres-from-file\n", encoding="utf-8")
    pg_db_file = tmp_path / "postgres_db"
    pg_db_file.write_text("postgres-db-from-file\n", encoding="utf-8")
    pg_user_file = tmp_path / "postgres_user"
    pg_user_file.write_text("postgres-user-from-file\n", encoding="utf-8")
    jellyfin_user_id_file = tmp_path / "jellyfin_user_id"
    jellyfin_user_id_file.write_text("jellyfin-user-from-file\n", encoding="utf-8")
    audiomuse_user_file = tmp_path / "audiomuse_user"
    audiomuse_user_file.write_text("admin-from-file\n", encoding="utf-8")

    monkeypatch.setenv("JWT_SECRET_FILE", str(jwt_secret_file))
    monkeypatch.setenv("POSTGRES_DB_FILE", str(pg_db_file))
    monkeypatch.setenv("POSTGRES_USER_FILE", str(pg_user_file))
    monkeypatch.setenv("POSTGRES_PASSWORD_FILE", str(pg_password_file))
    monkeypatch.setenv("JELLYFIN_USER_ID_FILE", str(jellyfin_user_id_file))
    monkeypatch.setenv("AUDIOMUSE_USER_FILE", str(audiomuse_user_file))
    monkeypatch.delenv("JWT_SECRET", raising=False)
    monkeypatch.delenv("POSTGRES_DB", raising=False)
    monkeypatch.delenv("POSTGRES_USER", raising=False)
    monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)
    monkeypatch.delenv("JELLYFIN_USER_ID", raising=False)
    monkeypatch.delenv("AUDIOMUSE_USER", raising=False)

    fake_setup_manager = types.ModuleType("tasks.setup_manager")

    class DummySetupManager:
        def ensure_table(self):
            return None

        def get_raw_overrides(self, ensure_table=True):
            return {}

        def cast_value(self, current_value, new_value):
            return new_value

        def config_table_exists(self):
            return False

    fake_setup_manager.SetupManager = DummySetupManager
    original_config = sys.modules.get("config")
    original_setup_manager = sys.modules.get("tasks.setup_manager")
    monkeypatch.setitem(sys.modules, "tasks.setup_manager", fake_setup_manager)
    sys.modules.pop("config", None)

    try:
        config = importlib.import_module("config")

        assert config.JWT_SECRET == "jwt-from-file"
        assert config.POSTGRES_DB == "postgres-db-from-file"
        assert config.POSTGRES_USER == "postgres-user-from-file"
        assert config.POSTGRES_PASSWORD == "postgres-from-file"
        assert config.JELLYFIN_USER_ID == "jellyfin-user-from-file"
        assert config.AUDIOMUSE_USER == "admin-from-file"
        assert "postgres-from-file" in config.DATABASE_URL
    finally:
        sys.modules.pop("config", None)
        if original_config is not None:
            sys.modules["config"] = original_config
        if original_setup_manager is not None:
            sys.modules["tasks.setup_manager"] = original_setup_manager
        else:
            sys.modules.pop("tasks.setup_manager", None)
