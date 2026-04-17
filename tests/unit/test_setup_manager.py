"""Unit tests for tasks/setup_manager.py

Tests pure validation / transformation logic — no database mocks needed.
"""
import json
import os
import types
import pytest
from unittest.mock import MagicMock, patch

from tasks.setup_manager import SetupManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mgr(database_url="postgresql://test:test@localhost:5432/testdb"):
    """Create a SetupManager with a fake database URL (no real DB needed)."""
    return SetupManager(database_url=database_url)


def _cfg(**attrs):
    """Return a SimpleNamespace that behaves like a config module."""
    return types.SimpleNamespace(**attrs)


# ============================================================================
# _looks_like_placeholder
# ============================================================================

class TestLooksLikePlaceholder:

    def setup_method(self):
        self.mgr = _mgr()

    def test_empty_string(self):
        assert self.mgr._looks_like_placeholder("") is True

    def test_whitespace_only(self):
        assert self.mgr._looks_like_placeholder("   ") is True

    def test_none_returns_false(self):
        assert self.mgr._looks_like_placeholder(None) is False

    def test_integer_returns_false(self):
        assert self.mgr._looks_like_placeholder(42) is False

    def test_your_prefix_variations(self):
        assert self.mgr._looks_like_placeholder("your_jellyfin_url") is True
        assert self.mgr._looks_like_placeholder("Your Default Token") is True
        assert self.mgr._looks_like_placeholder("YOUR_NAVIDROME_USER") is True

    def test_no_key_needed(self):
        assert self.mgr._looks_like_placeholder("no-key-needed") is True

    def test_url_placeholders(self):
        assert self.mgr._looks_like_placeholder("http://your_jellyfin_server") is True
        assert self.mgr._looks_like_placeholder("http://your-navidrome-server") is True
        assert self.mgr._looks_like_placeholder("http://your-lyrion-server") is True
        assert self.mgr._looks_like_placeholder("your-gemini-api-key-here") is True

    def test_real_values_not_detected(self):
        assert self.mgr._looks_like_placeholder("http://192.168.1.100:8096") is False
        assert self.mgr._looks_like_placeholder("abc123def456") is False
        assert self.mgr._looks_like_placeholder("admin") is False
        assert self.mgr._looks_like_placeholder("my-api-key-abc") is False

    def test_placeholder_embedded_in_longer_string(self):
        assert self.mgr._looks_like_placeholder("prefix-your_value-suffix") is True

    def test_case_insensitivity(self):
        assert self.mgr._looks_like_placeholder("YOUR_JELLYFIN_URL") is True
        assert self.mgr._looks_like_placeholder("No-Key-Needed") is True


# ============================================================================
# _is_valid_string
# ============================================================================

class TestIsValidString:

    def setup_method(self):
        self.mgr = _mgr()

    def test_valid_string(self):
        assert self.mgr._is_valid_string("http://localhost:8096") is True

    def test_empty_string(self):
        assert self.mgr._is_valid_string("") is False

    def test_whitespace_only(self):
        assert self.mgr._is_valid_string("   ") is False

    def test_placeholder(self):
        assert self.mgr._is_valid_string("your_jellyfin_url") is False

    def test_non_string_types(self):
        assert self.mgr._is_valid_string(42) is False
        assert self.mgr._is_valid_string(None) is False
        assert self.mgr._is_valid_string(True) is False
        assert self.mgr._is_valid_string([]) is False


# ============================================================================
# _is_argon2_password_hash
# ============================================================================

class TestIsArgon2PasswordHash:

    def setup_method(self):
        self.mgr = _mgr()

    def test_argon2id_hash(self):
        assert self.mgr._is_argon2_password_hash("$argon2id$v=19$m=65536,t=3,p=4$abc") is True

    def test_argon2i_hash(self):
        assert self.mgr._is_argon2_password_hash("$argon2i$v=19$m=4096,t=3,p=1$abc") is True

    def test_plain_password(self):
        assert self.mgr._is_argon2_password_hash("mysecretpassword") is False

    def test_bcrypt_hash_not_matched(self):
        assert self.mgr._is_argon2_password_hash("$2b$12$abcdef") is False

    def test_non_string(self):
        assert self.mgr._is_argon2_password_hash(None) is False
        assert self.mgr._is_argon2_password_hash(123) is False


# ============================================================================
# cast_value — type coercion from stored DB strings
# ============================================================================

class TestCastValue:

    def setup_method(self):
        self.mgr = _mgr()

    # --- bool ---
    @pytest.mark.parametrize("stored", ["true", "True", "TRUE", "1", "yes", "on"])
    def test_bool_truthy(self, stored):
        assert self.mgr.cast_value(True, stored) is True

    @pytest.mark.parametrize("stored", ["false", "False", "0", "no", "off", "", "random"])
    def test_bool_falsy(self, stored):
        assert self.mgr.cast_value(False, stored) is False

    # --- int ---
    def test_int_valid(self):
        assert self.mgr.cast_value(0, "42") == 42

    def test_int_negative(self):
        assert self.mgr.cast_value(0, "-7") == -7

    def test_int_invalid_returns_default(self):
        assert self.mgr.cast_value(10, "not_a_number") == 10

    # --- float ---
    def test_float_valid(self):
        assert self.mgr.cast_value(0.0, "3.14") == pytest.approx(3.14)

    def test_float_invalid_returns_default(self):
        assert self.mgr.cast_value(1.5, "xyz") == 1.5

    # --- list / dict ---
    def test_list(self):
        assert self.mgr.cast_value([], '[1, 2, 3]') == [1, 2, 3]

    def test_dict(self):
        assert self.mgr.cast_value({}, '{"a": 1}') == {"a": 1}

    def test_json_invalid_returns_default(self):
        assert self.mgr.cast_value([1, 2], "not json") == [1, 2]

    def test_nested_json(self):
        assert self.mgr.cast_value({}, '{"a": {"b": [1]}}') == {"a": {"b": [1]}}

    # --- string passthrough ---
    def test_string_passthrough(self):
        assert self.mgr.cast_value("default", "override") == "override"

    def test_empty_string_passthrough(self):
        assert self.mgr.cast_value("default", "") == ""


# ============================================================================
# format_value — serialization for DB storage
# ============================================================================

class TestFormatValue:

    def setup_method(self):
        self.mgr = _mgr()

    def test_string(self):
        assert self.mgr.format_value("hello") == "hello"

    def test_int(self):
        assert self.mgr.format_value(42) == "42"

    def test_bool(self):
        assert self.mgr.format_value(True) == "True"

    def test_list(self):
        assert self.mgr.format_value([1, 2]) == json.dumps([1, 2])

    def test_dict(self):
        assert self.mgr.format_value({"a": 1}) == json.dumps({"a": 1})

    def test_float(self):
        assert self.mgr.format_value(3.14) == "3.14"

    def test_empty_list(self):
        assert self.mgr.format_value([]) == "[]"

    def test_empty_dict(self):
        assert self.mgr.format_value({}) == "{}"


# ============================================================================
# cast_value ↔ format_value round-trip
# ============================================================================

class TestCastFormatRoundTrip:

    def setup_method(self):
        self.mgr = _mgr()

    @pytest.mark.parametrize("original", [42, 3.14, True, False, "hello", [1, 2], {"k": "v"}])
    def test_roundtrip(self, original):
        """format then cast should recover the original value."""
        formatted = self.mgr.format_value(original)
        recovered = self.mgr.cast_value(original, formatted)
        assert recovered == original


# ============================================================================
# _get_database_url
# ============================================================================

class TestGetDatabaseUrl:

    def test_uses_database_url_env(self):
        with patch.dict("os.environ", {"DATABASE_URL": "postgresql://u:p@h:5/d"}, clear=False):
            url = _mgr(database_url=None)._get_database_url()
            assert url == "postgresql://u:p@h:5/d"

    def test_builds_from_components(self):
        env = {
            "POSTGRES_USER": "myuser",
            "POSTGRES_PASSWORD": "mypass",
            "POSTGRES_HOST": "myhost",
            "POSTGRES_PORT": "5433",
            "POSTGRES_DB": "mydb",
        }
        old = os.environ.pop("DATABASE_URL", None)
        try:
            with patch.dict("os.environ", env, clear=False):
                mgr = SetupManager.__new__(SetupManager)
                url = mgr._get_database_url()
                assert "myuser" in url
                assert "mypass" in url
                assert "myhost" in url
                assert "5433" in url
                assert "mydb" in url
        finally:
            if old is not None:
                os.environ["DATABASE_URL"] = old

    def test_special_chars_escaped(self):
        env = {
            "POSTGRES_USER": "user@domain",
            "POSTGRES_PASSWORD": "p@ss:word",
            "POSTGRES_HOST": "host",
            "POSTGRES_PORT": "5432",
            "POSTGRES_DB": "db",
        }
        old = os.environ.pop("DATABASE_URL", None)
        try:
            with patch.dict("os.environ", env, clear=False):
                mgr = SetupManager.__new__(SetupManager)
                url = mgr._get_database_url()
                assert "user%40domain" in url
                assert "p%40ss%3Aword" in url
        finally:
            if old is not None:
                os.environ["DATABASE_URL"] = old


# ============================================================================
# _is_valid_server_config — all four media server types
# ============================================================================

class TestIsValidServerConfig:

    def setup_method(self):
        self.mgr = _mgr()

    def test_valid_jellyfin(self):
        cfg = _cfg(
            MEDIASERVER_TYPE="jellyfin",
            JELLYFIN_URL="http://localhost:8096",
            JELLYFIN_USER_ID="uid123",
            JELLYFIN_TOKEN="tok456",
        )
        assert self.mgr._is_valid_server_config(cfg) is True

    def test_valid_navidrome(self):
        cfg = _cfg(
            MEDIASERVER_TYPE="navidrome",
            NAVIDROME_URL="http://localhost:4533",
            NAVIDROME_USER="admin",
            NAVIDROME_PASSWORD="secret",
        )
        assert self.mgr._is_valid_server_config(cfg) is True

    def test_valid_lyrion(self):
        cfg = _cfg(MEDIASERVER_TYPE="lyrion", LYRION_URL="http://localhost:9000")
        assert self.mgr._is_valid_server_config(cfg) is True

    def test_valid_emby(self):
        cfg = _cfg(
            MEDIASERVER_TYPE="emby",
            EMBY_URL="http://localhost:8096",
            EMBY_USER_ID="uid789",
            EMBY_TOKEN="tok012",
        )
        assert self.mgr._is_valid_server_config(cfg) is True

    def test_unknown_server_type(self):
        assert self.mgr._is_valid_server_config(_cfg(MEDIASERVER_TYPE="plex")) is False

    def test_empty_server_type(self):
        assert self.mgr._is_valid_server_config(_cfg(MEDIASERVER_TYPE="")) is False

    def test_missing_required_field_empty(self):
        cfg = _cfg(
            MEDIASERVER_TYPE="jellyfin",
            JELLYFIN_URL="http://localhost:8096",
            JELLYFIN_USER_ID="uid",
            JELLYFIN_TOKEN="",
        )
        assert self.mgr._is_valid_server_config(cfg) is False

    def test_missing_required_field_absent(self):
        """Field not set at all on the config module (getattr returns '')."""
        cfg = _cfg(MEDIASERVER_TYPE="navidrome", NAVIDROME_URL="http://localhost:4533")
        # NAVIDROME_USER and NAVIDROME_PASSWORD are missing
        assert self.mgr._is_valid_server_config(cfg) is False

    def test_placeholder_in_required_field(self):
        cfg = _cfg(
            MEDIASERVER_TYPE="jellyfin",
            JELLYFIN_URL="http://your_jellyfin_server",
            JELLYFIN_USER_ID="your_default_user_id",
            JELLYFIN_TOKEN="your_default_token",
        )
        assert self.mgr._is_valid_server_config(cfg) is False

    def test_case_insensitive_server_type(self):
        cfg = _cfg(
            MEDIASERVER_TYPE="JELLYFIN",
            JELLYFIN_URL="http://localhost:8096",
            JELLYFIN_USER_ID="uid",
            JELLYFIN_TOKEN="tok",
        )
        assert self.mgr._is_valid_server_config(cfg) is True

    def test_mixed_case_server_type(self):
        cfg = _cfg(
            MEDIASERVER_TYPE="Navidrome",
            NAVIDROME_URL="http://localhost:4533",
            NAVIDROME_USER="u",
            NAVIDROME_PASSWORD="p",
        )
        assert self.mgr._is_valid_server_config(cfg) is True

    def test_whitespace_only_field_invalid(self):
        cfg = _cfg(
            MEDIASERVER_TYPE="lyrion",
            LYRION_URL="   ",
        )
        assert self.mgr._is_valid_server_config(cfg) is False


# ============================================================================
# _is_valid_auth_config
# ============================================================================

class TestIsValidAuthConfig:

    def setup_method(self):
        self.mgr = _mgr()

    def test_auth_disabled_bool_false(self):
        assert self.mgr._is_valid_auth_config(_cfg(AUTH_ENABLED=False)) is True

    def test_auth_disabled_string_false(self):
        assert self.mgr._is_valid_auth_config(_cfg(AUTH_ENABLED="false")) is True

    def test_auth_enabled_with_credentials(self):
        cfg = _cfg(AUTH_ENABLED=True, AUDIOMUSE_USER="admin", AUDIOMUSE_PASSWORD="secret")
        assert self.mgr._is_valid_auth_config(cfg) is True

    def test_auth_enabled_missing_user(self):
        cfg = _cfg(AUTH_ENABLED=True, AUDIOMUSE_USER="", AUDIOMUSE_PASSWORD="secret")
        assert self.mgr._is_valid_auth_config(cfg) is False

    def test_auth_enabled_missing_password(self):
        cfg = _cfg(AUTH_ENABLED=True, AUDIOMUSE_USER="admin", AUDIOMUSE_PASSWORD="")
        assert self.mgr._is_valid_auth_config(cfg) is False

    def test_auth_enabled_both_missing(self):
        cfg = _cfg(AUTH_ENABLED=True, AUDIOMUSE_USER="", AUDIOMUSE_PASSWORD="")
        assert self.mgr._is_valid_auth_config(cfg) is False

    def test_auth_enabled_string_true(self):
        cfg = _cfg(AUTH_ENABLED="true", AUDIOMUSE_USER="admin", AUDIOMUSE_PASSWORD="pass")
        assert self.mgr._is_valid_auth_config(cfg) is True

    def test_auth_enabled_placeholder_user(self):
        cfg = _cfg(AUTH_ENABLED=True, AUDIOMUSE_USER="your_default_user", AUDIOMUSE_PASSWORD="secret")
        assert self.mgr._is_valid_auth_config(cfg) is False

    def test_api_token_not_required(self):
        """API_TOKEN is optional — setup is valid without it."""
        cfg = _cfg(AUTH_ENABLED=True, AUDIOMUSE_USER="admin", AUDIOMUSE_PASSWORD="secret")
        assert self.mgr._is_valid_auth_config(cfg) is True

    def test_auth_not_set_defaults_to_enabled(self):
        """If AUTH_ENABLED is absent, defaults to True → needs credentials."""
        cfg = _cfg(AUDIOMUSE_USER="", AUDIOMUSE_PASSWORD="")
        assert self.mgr._is_valid_auth_config(cfg) is False


# ============================================================================
# is_valid_env_config — combined server + auth
# ============================================================================

class TestIsValidEnvConfig:

    def setup_method(self):
        self.mgr = _mgr()

    def test_valid_complete_config(self):
        cfg = _cfg(
            MEDIASERVER_TYPE="jellyfin",
            JELLYFIN_URL="http://localhost:8096",
            JELLYFIN_USER_ID="uid",
            JELLYFIN_TOKEN="tok",
            AUTH_ENABLED=False,
        )
        assert self.mgr.is_valid_env_config(cfg) is True

    def test_both_valid_with_auth(self):
        cfg = _cfg(
            MEDIASERVER_TYPE="navidrome",
            NAVIDROME_URL="http://localhost:4533",
            NAVIDROME_USER="admin",
            NAVIDROME_PASSWORD="secret",
            AUTH_ENABLED=True,
            AUDIOMUSE_USER="admin",
            AUDIOMUSE_PASSWORD="pass",
        )
        assert self.mgr.is_valid_env_config(cfg) is True

    def test_invalid_server_valid_auth(self):
        cfg = _cfg(MEDIASERVER_TYPE="unknown", AUTH_ENABLED=False)
        assert self.mgr.is_valid_env_config(cfg) is False

    def test_valid_server_invalid_auth(self):
        cfg = _cfg(
            MEDIASERVER_TYPE="lyrion",
            LYRION_URL="http://localhost:9000",
            AUTH_ENABLED=True,
            AUDIOMUSE_USER="",
            AUDIOMUSE_PASSWORD="",
        )
        assert self.mgr.is_valid_env_config(cfg) is False

    def test_both_invalid(self):
        cfg = _cfg(
            MEDIASERVER_TYPE="plex",
            AUTH_ENABLED=True,
            AUDIOMUSE_USER="",
            AUDIOMUSE_PASSWORD="",
        )
        assert self.mgr.is_valid_env_config(cfg) is False

    def test_lyrion_minimal(self):
        """Lyrion only needs URL + auth disabled — smallest valid config."""
        cfg = _cfg(MEDIASERVER_TYPE="lyrion", LYRION_URL="http://x:9000", AUTH_ENABLED=False)
        assert self.mgr.is_valid_env_config(cfg) is True


# ============================================================================
# is_setup_complete (alias for is_valid_env_config)
# ============================================================================

class TestIsSetupComplete:

    def test_delegates_to_is_valid_env_config(self):
        mgr = _mgr()
        good = _cfg(MEDIASERVER_TYPE="lyrion", LYRION_URL="http://x:9000", AUTH_ENABLED=False)
        assert mgr.is_setup_complete(good) is True

        bad = _cfg(MEDIASERVER_TYPE="unknown")
        assert mgr.is_setup_complete(bad) is False


# ============================================================================
# _get_env_config_values — config extraction
# ============================================================================

class TestGetEnvConfigValues:

    def setup_method(self):
        self.mgr = _mgr()

    def test_excludes_private_and_lowercase(self):
        cfg = _cfg(FOO="bar", _PRIVATE="hidden", lowercase="ignored")
        values = self.mgr._get_env_config_values(cfg)
        assert "FOO" in values
        assert "_PRIVATE" not in values
        assert "lowercase" not in values

    def test_excludes_bootstrap_keys(self):
        cfg = _cfg(
            DATABASE_URL="postgres://...",
            JELLYFIN_URL="http://localhost",
            SETUP_BOOTSTRAP_EXCLUDED_KEYS={"DATABASE_URL"},
        )
        values = self.mgr._get_env_config_values(cfg)
        assert "DATABASE_URL" not in values
        assert "JELLYFIN_URL" in values

    def test_values_are_sorted_by_key(self):
        cfg = _cfg(ZEBRA="z", APPLE="a", MANGO="m")
        keys = list(self.mgr._get_env_config_values(cfg).keys())
        assert keys == sorted(keys)

    def test_preserves_value_types(self):
        cfg = _cfg(NUM=42, FLAG=True, NAME="test")
        values = self.mgr._get_env_config_values(cfg)
        assert values["NUM"] == 42
        assert values["FLAG"] is True
        assert values["NAME"] == "test"

    def test_empty_module(self):
        cfg = _cfg()
        assert self.mgr._get_env_config_values(cfg) == {}

    def test_no_excluded_keys_attr(self):
        """Works when config has no SETUP_BOOTSTRAP_EXCLUDED_KEYS at all."""
        cfg = _cfg(MY_KEY="val")
        values = self.mgr._get_env_config_values(cfg)
        assert "MY_KEY" in values


# ============================================================================
# get_connection — RuntimeError guard
# ============================================================================

class TestGetConnection:

    def test_raises_if_no_database_url(self):
        mgr = SetupManager.__new__(SetupManager)
        mgr.database_url = None
        mgr.logger = MagicMock()
        with pytest.raises(RuntimeError, match="DATABASE_URL is not configured"):
            mgr.get_connection()


# ============================================================================
# save_config_values — input validation (no DB needed)
# ============================================================================

class TestSaveConfigValuesValidation:

    def test_rejects_non_dict(self):
        mgr = _mgr()
        with pytest.raises(ValueError, match="Expected a dictionary"):
            mgr.save_config_values("not a dict")

    def test_rejects_list(self):
        mgr = _mgr()
        with pytest.raises(ValueError, match="Expected a dictionary"):
            mgr.save_config_values([("key", "val")])

    def test_rejects_none(self):
        mgr = _mgr()
        with pytest.raises(ValueError, match="Expected a dictionary"):
            mgr.save_config_values(None)


# ============================================================================
# delete_config_values — noop guard (no DB needed)
# ============================================================================

class TestDeleteConfigValuesNoop:

    @patch("tasks.setup_manager.SetupManager.get_connection")
    def test_noop_for_empty_keys(self, mock_get_conn):
        _mgr().delete_config_values([])
        mock_get_conn.assert_not_called()


# ============================================================================
# Scenario: every media server with placeholder fields should fail validation
# ============================================================================

class TestPlaceholderFieldsRejectAllServers:
    """Ensure default env placeholders never pass validation for any type."""

    def setup_method(self):
        self.mgr = _mgr()

    def test_jellyfin_defaults(self):
        cfg = _cfg(
            MEDIASERVER_TYPE="jellyfin",
            JELLYFIN_URL="",
            JELLYFIN_USER_ID="",
            JELLYFIN_TOKEN="",
        )
        assert self.mgr.is_valid_env_config(cfg) is False

    def test_navidrome_defaults(self):
        cfg = _cfg(
            MEDIASERVER_TYPE="navidrome",
            NAVIDROME_URL="",
            NAVIDROME_USER="",
            NAVIDROME_PASSWORD="",
        )
        assert self.mgr.is_valid_env_config(cfg) is False

    def test_emby_defaults(self):
        cfg = _cfg(
            MEDIASERVER_TYPE="emby",
            EMBY_URL="",
            EMBY_USER_ID="",
            EMBY_TOKEN="",
        )
        assert self.mgr.is_valid_env_config(cfg) is False

    def test_lyrion_defaults(self):
        cfg = _cfg(MEDIASERVER_TYPE="lyrion", LYRION_URL="")
        assert self.mgr.is_valid_env_config(cfg) is False


# ============================================================================
# Scenario: switching server type — only the new type's fields matter
# ============================================================================

class TestServerTypeSwitching:

    def setup_method(self):
        self.mgr = _mgr()

    def test_jellyfin_fields_ignored_when_type_is_navidrome(self):
        cfg = _cfg(
            MEDIASERVER_TYPE="navidrome",
            NAVIDROME_URL="http://localhost:4533",
            NAVIDROME_USER="admin",
            NAVIDROME_PASSWORD="secret",
            # Leftover jellyfin fields — should not matter
            JELLYFIN_URL="",
            JELLYFIN_USER_ID="",
            JELLYFIN_TOKEN="",
            AUTH_ENABLED=False,
        )
        assert self.mgr.is_valid_env_config(cfg) is True

    def test_navidrome_fields_ignored_when_type_is_lyrion(self):
        cfg = _cfg(
            MEDIASERVER_TYPE="lyrion",
            LYRION_URL="http://localhost:9000",
            NAVIDROME_URL="",
            NAVIDROME_USER="",
            NAVIDROME_PASSWORD="",
            AUTH_ENABLED=False,
        )
        assert self.mgr.is_valid_env_config(cfg) is True


# ============================================================================
# Scenario: auth transitions
# ============================================================================

class TestAuthTransitions:

    def setup_method(self):
        self.mgr = _mgr()

    def _valid_server(self):
        return dict(MEDIASERVER_TYPE="lyrion", LYRION_URL="http://x:9000")

    def test_enable_auth_requires_credentials(self):
        cfg = _cfg(**self._valid_server(), AUTH_ENABLED=True, AUDIOMUSE_USER="", AUDIOMUSE_PASSWORD="")
        assert self.mgr.is_valid_env_config(cfg) is False

    def test_disable_auth_clears_requirement(self):
        cfg = _cfg(**self._valid_server(), AUTH_ENABLED=False, AUDIOMUSE_USER="", AUDIOMUSE_PASSWORD="")
        assert self.mgr.is_valid_env_config(cfg) is True

    def test_argon2_hash_is_valid_password(self):
        """An already-hashed password still passes _is_valid_string."""
        cfg = _cfg(
            **self._valid_server(),
            AUTH_ENABLED=True,
            AUDIOMUSE_USER="admin",
            AUDIOMUSE_PASSWORD="$argon2id$v=19$m=65536,t=3,p=4$abc",
        )
        assert self.mgr.is_valid_env_config(cfg) is True


# ============================================================================
# MODULE-LEVEL CONSTANTS
# ============================================================================

class TestModuleConstants:

    def test_basic_server_fields_is_set(self):
        from tasks.setup_manager import BASIC_SERVER_FIELDS
        assert isinstance(BASIC_SERVER_FIELDS, set)
        assert 'MEDIASERVER_TYPE' in BASIC_SERVER_FIELDS
        assert 'JELLYFIN_URL' in BASIC_SERVER_FIELDS

    def test_auth_fields_is_set(self):
        from tasks.setup_manager import AUTH_FIELDS
        assert isinstance(AUTH_FIELDS, set)
        assert 'AUTH_ENABLED' in AUTH_FIELDS
        assert 'AUDIOMUSE_USER' in AUTH_FIELDS
        assert 'AUDIOMUSE_PASSWORD' in AUTH_FIELDS
        assert 'API_TOKEN' in AUTH_FIELDS

    def test_server_required_fields_matches_config(self):
        """SetupManager.SERVER_REQUIRED_FIELDS should reference config.MEDIASERVER_FIELDS_BY_TYPE."""
        import config
        mgr = _mgr()
        assert mgr.SERVER_REQUIRED_FIELDS is config.MEDIASERVER_FIELDS_BY_TYPE
