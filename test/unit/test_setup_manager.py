# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Setup-manager config value parsing, validation and placeholder detection.

Covers the helpers that read stored setup values: placeholder/validity checks,
argon2 hash detection and the cast/format round trip used by the config store.

Main Features:
* Placeholder and empty/whitespace strings are detected while real values pass
* Argon2id/argon2i hashes are recognized and bcrypt/plain are not
* cast_value coerces bool/int/float/list/dict and falls back on invalid JSON
* cast and format round-trip preserves the original value; the connection always
  comes from config.DATABASE_URL and a DATABASE_URL env var is ignored
* Startup pruning deletes retired keys without touching valid config rows
* Importing tasks.setup_manager before config keeps the DB-override init working
"""

import json
import os
import subprocess
import sys
import types
from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch

from tasks.setup_manager import SetupManager


def _mgr(database_url="postgresql://test:test@localhost:5432/testdb"):
    return SetupManager(database_url=database_url)


def _cfg(**attrs):
    return types.SimpleNamespace(**attrs)


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


class TestCastValue:
    def setup_method(self):
        self.mgr = _mgr()

    @pytest.mark.parametrize("stored", ["true", "True", "TRUE", "1", "yes", "on"])
    def test_bool_truthy(self, stored):
        assert self.mgr.cast_value(True, stored) is True

    @pytest.mark.parametrize("stored", ["false", "False", "0", "no", "off", "", "random"])
    def test_bool_falsy(self, stored):
        assert self.mgr.cast_value(False, stored) is False

    def test_int_valid(self):
        assert self.mgr.cast_value(0, "42") == 42

    def test_int_negative(self):
        assert self.mgr.cast_value(0, "-7") == -7

    def test_int_invalid_returns_default(self):
        assert self.mgr.cast_value(10, "not_a_number") == 10

    def test_float_valid(self):
        assert self.mgr.cast_value(0.0, "3.14") == pytest.approx(3.14)

    def test_float_invalid_returns_default(self):
        assert self.mgr.cast_value(1.5, "xyz") == 1.5

    def test_list(self):
        assert self.mgr.cast_value([], '[1, 2, 3]') == [1, 2, 3]

    def test_dict(self):
        assert self.mgr.cast_value({}, '{"a": 1}') == {"a": 1}

    def test_json_invalid_returns_default(self):
        assert self.mgr.cast_value([1, 2], "not json") == [1, 2]

    def test_nested_json(self):
        assert self.mgr.cast_value({}, '{"a": {"b": [1]}}') == {"a": {"b": [1]}}

    def test_string_passthrough(self):
        assert self.mgr.cast_value("default", "override") == "override"

    def test_empty_string_passthrough(self):
        assert self.mgr.cast_value("default", "") == ""


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


class TestCastFormatRoundTrip:
    def setup_method(self):
        self.mgr = _mgr()

    @pytest.mark.parametrize("original", [42, 3.14, True, False, "hello", [1, 2], {"k": "v"}])
    def test_roundtrip(self, original):
        formatted = self.mgr.format_value(original)
        recovered = self.mgr.cast_value(original, formatted)
        assert recovered == original


class TestGetDatabaseUrl:
    def test_returns_the_config_database_url(self):
        import config

        assert _mgr(database_url=None)._get_database_url() == config.DATABASE_URL

    def test_database_url_env_var_is_ignored(self):
        with patch.dict("os.environ", {"DATABASE_URL": "postgresql://u:p@h:5/d"}, clear=False):
            url = _mgr(database_url=None)._get_database_url()
        assert url != "postgresql://u:p@h:5/d"

    def test_empty_url_argument_falls_back_to_the_derived_url(self):
        import config

        assert SetupManager(database_url="").database_url == config.DATABASE_URL

    def test_url_is_resolved_on_first_use_not_at_construction(self):
        import config

        mgr = SetupManager()
        assert mgr._database_url_resolved is False
        assert mgr.database_url == config.DATABASE_URL
        assert mgr._database_url_resolved is True

    def test_explicit_url_is_kept_and_never_resolved(self):
        mgr = SetupManager(database_url="postgresql://explicit:pw@host:5432/db")
        assert mgr._database_url_resolved is True
        assert mgr.database_url == "postgresql://explicit:pw@host:5432/db"


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
        cfg = _cfg(MEDIASERVER_TYPE="navidrome", NAVIDROME_URL="http://localhost:4533")
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
        cfg = _cfg(
            AUTH_ENABLED=True, AUDIOMUSE_USER="your_default_user", AUDIOMUSE_PASSWORD="secret"
        )
        assert self.mgr._is_valid_auth_config(cfg) is False

    def test_api_token_not_required(self):
        cfg = _cfg(AUTH_ENABLED=True, AUDIOMUSE_USER="admin", AUDIOMUSE_PASSWORD="secret")
        assert self.mgr._is_valid_auth_config(cfg) is True

    def test_auth_not_set_defaults_to_enabled(self):
        cfg = _cfg(AUDIOMUSE_USER="", AUDIOMUSE_PASSWORD="")
        assert self.mgr._is_valid_auth_config(cfg) is False


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
        cfg = _cfg(MEDIASERVER_TYPE="lyrion", LYRION_URL="http://x:9000", AUTH_ENABLED=False)
        assert self.mgr.is_valid_env_config(cfg) is True


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

    def test_excludes_media_server_registry_keys(self):
        cfg = _cfg(
            JELLYFIN_URL="http://localhost",
            MAX_DISTANCE=0.5,
            MEDIASERVER_CONFIG_KEYS={"JELLYFIN_URL"},
        )
        values = self.mgr._get_env_config_values(cfg)
        assert "JELLYFIN_URL" not in values
        assert "MAX_DISTANCE" in values
        assert "MEDIASERVER_CONFIG_KEYS" not in values

    def test_excludes_persistence_metadata(self):
        cfg = _cfg(
            MAX_DISTANCE=0.5,
            SETUP_BOOTSTRAP_EXCLUDED_KEYS=set(),
        )
        values = self.mgr._get_env_config_values(cfg)
        assert "SETUP_BOOTSTRAP_EXCLUDED_KEYS" not in values
        assert "MAX_DISTANCE" in values

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
        cfg = _cfg(MY_KEY="val")
        values = self.mgr._get_env_config_values(cfg)
        assert "MY_KEY" in values


class TestGetConnection:
    def test_raises_if_no_database_url(self):
        mgr = SetupManager.__new__(SetupManager)
        mgr.database_url = None
        mgr.logger = MagicMock()
        with pytest.raises(RuntimeError, match="DATABASE_URL is not configured"):
            mgr.get_connection()


class TestSaveConfigValuesValidation:
    def test_rejects_non_dict(self):
        mgr = _mgr()
        with pytest.raises(TypeError, match="Expected a dictionary"):
            mgr.save_config_values("not a dict")

    def test_rejects_list(self):
        mgr = _mgr()
        with pytest.raises(TypeError, match="Expected a dictionary"):
            mgr.save_config_values([("key", "val")])

    def test_rejects_none(self):
        mgr = _mgr()
        with pytest.raises(TypeError, match="Expected a dictionary"):
            mgr.save_config_values(None)


class TestDeleteConfigValuesNoop:
    @patch("tasks.setup_manager.SetupManager.get_connection")
    def test_noop_for_empty_keys(self, mock_get_conn):
        _mgr().delete_config_values([])
        mock_get_conn.assert_not_called()


class TestPruneObsoleteConfigValues:
    def setup_method(self):
        self.mgr = _mgr()
        self.connection = MagicMock()
        self.connection.__enter__.return_value = self.connection
        self.cursor = MagicMock()
        self.connection.cursor.return_value.__enter__.return_value = self.cursor

    def _install_connection(self):
        self.mgr.ensure_table = MagicMock()
        self.mgr.get_connection = MagicMock(return_value=self.connection)

    def test_deletes_only_keys_absent_from_persistable_config(self):
        self._install_connection()
        self.cursor.fetchall.return_value = [("RETIRED_PARAMETER",), ("OLD_NAME",)]
        cfg = _cfg(
            ACTIVE_PARAMETER=42,
            OTHER_ACTIVE=True,
            DATABASE_URL="postgres://internal",
            SETUP_BOOTSTRAP_EXCLUDED_KEYS={"DATABASE_URL"},
        )

        removed = self.mgr.prune_obsolete_config_values(cfg)

        assert removed == ["OLD_NAME", "RETIRED_PARAMETER"]
        sql, params = self.cursor.execute.call_args.args
        assert "DELETE FROM app_config" in sql
        assert "WHERE NOT (key = ANY(%s))" in sql
        assert "RETURNING key" in sql
        assert params == (
            [
                "ACTIVE_PARAMETER",
                "AUDIOMUSE_PASSWORD",
                "AUDIOMUSE_USER",
                "OTHER_ACTIVE",
            ],
        )
        self.connection.commit.assert_called_once_with()

    def test_preserves_declared_runtime_keys_without_treating_them_as_parameters(self):
        self._install_connection()
        self.cursor.fetchall.return_value = [("RETIRED_PARAMETER",)]
        cfg = _cfg(
            ACTIVE_PARAMETER=42,
            APP_CONFIG_RUNTIME_KEYS={"PLUGIN_REPOS", "PLUGIN_CATALOG_CACHE"},
            SETUP_BOOTSTRAP_EXCLUDED_KEYS={"APP_CONFIG_RUNTIME_KEYS"},
        )

        removed = self.mgr.prune_obsolete_config_values(cfg)

        assert removed == ["RETIRED_PARAMETER"]
        _sql, params = self.cursor.execute.call_args.args
        assert params == (
            [
                "ACTIVE_PARAMETER",
                "AUDIOMUSE_PASSWORD",
                "AUDIOMUSE_USER",
                "PLUGIN_CATALOG_CACHE",
                "PLUGIN_REPOS",
            ],
        )
        assert "APP_CONFIG_RUNTIME_KEYS" not in params[0]

    def test_valid_rows_are_not_updated_when_nothing_is_obsolete(self):
        self._install_connection()
        self.cursor.fetchall.return_value = []

        removed = self.mgr.prune_obsolete_config_values(_cfg(ACTIVE_PARAMETER=42))

        assert removed == []
        sql = self.cursor.execute.call_args.args[0]
        assert "UPDATE" not in sql.upper()
        assert "INSERT" not in sql.upper()
        self.connection.commit.assert_called_once_with()

    def test_legacy_admin_bridge_rows_survive_the_prune(self):
        self._install_connection()
        self.cursor.fetchall.return_value = []

        self.mgr.prune_obsolete_config_values(_cfg(ACTIVE_PARAMETER=42))

        _sql, params = self.cursor.execute.call_args.args
        assert "AUDIOMUSE_USER" in params[0]
        assert "AUDIOMUSE_PASSWORD" in params[0]

    def test_legacy_top_n_rows_are_dropped_not_migrated(self):
        self._install_connection()
        self.cursor.fetchall.return_value = [
            ("MIN_CLUSTERING_TOP",), ("TOP_N_PLAYLISTS",),
        ]

        removed = self.mgr.prune_obsolete_config_values(
            _cfg(TOP_N_CLUSTERING_PLAYLIST=10)
        )

        assert removed == ["MIN_CLUSTERING_TOP", "TOP_N_PLAYLISTS"]
        executed = [call.args[0] for call in self.cursor.execute.call_args_list]
        assert not any("UPDATE" in sql.upper() for sql in executed)

    def test_refuses_to_delete_everything_when_config_has_no_valid_keys(self):
        self.mgr.ensure_table = MagicMock()
        self.mgr.get_connection = MagicMock()

        with pytest.raises(RuntimeError, match="exposes no persistable parameters"):
            self.mgr.prune_obsolete_config_values(_cfg(lowercase="ignored"))

        self.mgr.ensure_table.assert_not_called()
        self.mgr.get_connection.assert_not_called()


def test_flask_startup_prunes_config_after_schema_init_and_before_bootstrap():
    app_source = (Path(__file__).parents[2] / "app.py").read_text(encoding="utf-8")

    init_position = app_source.index("        init_db()")
    prune_position = app_source.index(
        "        setup_manager.prune_obsolete_config_values(config)"
    )
    bootstrap_position = app_source.index(
        "        setup_manager.bootstrap_env_config_if_empty(config)"
    )

    assert init_position < prune_position < bootstrap_position


class TestPlaceholderFieldsRejectAllServers:
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


class TestServerTypeSwitching:
    def setup_method(self):
        self.mgr = _mgr()

    def test_jellyfin_fields_ignored_when_type_is_navidrome(self):
        cfg = _cfg(
            MEDIASERVER_TYPE="navidrome",
            NAVIDROME_URL="http://localhost:4533",
            NAVIDROME_USER="admin",
            NAVIDROME_PASSWORD="secret",
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


class TestAuthTransitions:
    def setup_method(self):
        self.mgr = _mgr()

    def _valid_server(self):
        return dict(MEDIASERVER_TYPE="lyrion", LYRION_URL="http://x:9000")

    def test_enable_auth_requires_credentials(self):
        cfg = _cfg(
            **self._valid_server(), AUTH_ENABLED=True, AUDIOMUSE_USER="", AUDIOMUSE_PASSWORD=""
        )
        assert self.mgr.is_valid_env_config(cfg) is False

    def test_disable_auth_clears_requirement(self):
        cfg = _cfg(
            **self._valid_server(), AUTH_ENABLED=False, AUDIOMUSE_USER="", AUDIOMUSE_PASSWORD=""
        )
        assert self.mgr.is_valid_env_config(cfg) is True

    def test_argon2_hash_is_valid_password(self):
        cfg = _cfg(
            **self._valid_server(),
            AUTH_ENABLED=True,
            AUDIOMUSE_USER="admin",
            AUDIOMUSE_PASSWORD="$argon2id$v=19$m=65536,t=3,p=4$abc",
        )
        assert self.mgr.is_valid_env_config(cfg) is True


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
        import config

        mgr = _mgr()
        assert mgr.server_required_fields == config.MEDIASERVER_FIELDS_BY_TYPE


class TestImportOrderIndependence:
    def test_setup_manager_imports_before_config_without_breaking_overrides(self):
        code = (
            "import sys\n"
            "import tasks.setup_manager\n"
            "assert 'config' not in sys.modules, "
            "'tasks.setup_manager must not import config at module level'\n"
            "import config\n"
            "assert config.DB_OVERRIDES_LOADED, "
            "'DB-override init was skipped: circular import regression'\n"
        )
        repo_root = os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
        )
        env = dict(os.environ)
        env['DATABASE_URL'] = 'postgresql://u:p@127.0.0.1:1/db'
        proc = subprocess.run(
            [sys.executable, '-c', code],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 0, proc.stderr


class TestWorkerConfigHydration:
    def test_a_raising_projection_pass_leaves_both_flags_false(self, monkeypatch):
        import config

        broken = types.ModuleType('tasks.setup_manager')

        def unreachable(*args, **kwargs):
            raise RuntimeError('database is not up yet')

        broken.SetupManager = unreachable
        monkeypatch.setitem(sys.modules, 'tasks.setup_manager', broken)
        monkeypatch.setattr(config, 'DB_OVERRIDES_LOADED', config.DB_OVERRIDES_LOADED)
        monkeypatch.setattr(config, 'DB_DEFAULT_SERVER_PROJECTED', config.DB_DEFAULT_SERVER_PROJECTED)
        config._apply_db_overrides()

        assert config.DB_OVERRIDES_LOADED is False
        assert config.DB_DEFAULT_SERVER_PROJECTED is False

    def test_the_no_default_server_warning_fires_once_per_process_not_per_job(self, monkeypatch, caplog):
        import logging

        import config
        import tasks.setup_manager as sm

        monkeypatch.setattr(config, 'DB_DEFAULT_SERVER_PROJECTED', False)
        monkeypatch.setattr(config, 'refresh_config', lambda: None)
        monkeypatch.setattr(sm, '_WARNED_NO_DEFAULT_SERVER', False)

        with caplog.at_level(logging.WARNING, logger='tasks.setup_manager'):
            assert sm.hydrate_worker_config() is False
            assert sm.hydrate_worker_config() is False

        warnings = [r for r in caplog.records if 'no default music server' in r.message]
        assert len(warnings) == 1

    def test_refresh_config_is_always_a_real_reload_never_a_stub(self, monkeypatch):
        import importlib

        import config

        reloaded = []
        monkeypatch.setattr(importlib, 'reload', reloaded.append)
        config.refresh_config()

        assert reloaded == [config]

    def test_projecting_the_default_row_marks_the_database_as_reached(self, monkeypatch):
        import config

        class Reachable:
            def config_table_exists(self):
                return False

            def get_default_music_server(self):
                return {
                    'server_type': 'jellyfin',
                    'creds': {'url': 'http://jf:8096', 'user_id': 'uid', 'token': 'tok'},
                    'music_libraries': '',
                }

        fake = types.ModuleType('tasks.setup_manager')
        fake.SetupManager = Reachable
        monkeypatch.setitem(sys.modules, 'tasks.setup_manager', fake)
        monkeypatch.setenv('AUDIOMUSE_ROLE', 'worker')
        for name in ('MEDIASERVER_TYPE', 'MUSIC_LIBRARIES', 'JELLYFIN_URL',
                     'JELLYFIN_USER_ID', 'JELLYFIN_TOKEN', 'HEADERS',
                     'DB_OVERRIDES_LOADED', 'DB_DEFAULT_SERVER_PROJECTED'):
            monkeypatch.setattr(config, name, getattr(config, name), raising=False)
        config._apply_db_overrides()

        assert config.DB_DEFAULT_SERVER_PROJECTED is True
        assert config.DB_OVERRIDES_LOADED is True
        assert config.JELLYFIN_URL == 'http://jf:8096'

    def test_a_swallowed_database_failure_does_not_mark_the_database_as_reached(self, monkeypatch):
        import config

        class Unreachable:
            def config_table_exists(self):
                return False

            def get_default_music_server(self):
                return None

        fake = types.ModuleType('tasks.setup_manager')
        fake.SetupManager = Unreachable
        monkeypatch.setitem(sys.modules, 'tasks.setup_manager', fake)
        monkeypatch.setenv('AUDIOMUSE_ROLE', 'worker')
        for name in ('HEADERS', 'DB_OVERRIDES_LOADED', 'DB_DEFAULT_SERVER_PROJECTED'):
            monkeypatch.setattr(config, name, getattr(config, name), raising=False)
        config._apply_db_overrides()

        assert config.DB_DEFAULT_SERVER_PROJECTED is False
        assert config.DB_OVERRIDES_LOADED is True

    def test_hydrate_skips_the_reload_when_the_import_already_projected_the_default_server(
        self, monkeypatch
    ):
        import config
        from tasks.setup_manager import hydrate_worker_config

        monkeypatch.setattr(config, 'DB_OVERRIDES_LOADED', True)
        monkeypatch.setattr(config, 'DB_DEFAULT_SERVER_PROJECTED', True)
        reloads = []
        monkeypatch.setattr(config, 'refresh_config', lambda: reloads.append(True))

        assert hydrate_worker_config() is True
        assert reloads == []

    def test_hydrate_reloads_when_the_import_time_pass_never_reached_the_database(self, monkeypatch):
        import config
        from tasks.setup_manager import hydrate_worker_config

        monkeypatch.setattr(config, 'DB_OVERRIDES_LOADED', True)
        monkeypatch.setattr(config, 'DB_DEFAULT_SERVER_PROJECTED', False)
        reloads = []

        def arrives():
            reloads.append(True)
            monkeypatch.setattr(config, 'DB_DEFAULT_SERVER_PROJECTED', True)

        monkeypatch.setattr(config, 'refresh_config', arrives)

        assert hydrate_worker_config() is True
        assert reloads == [True]

    def test_hydrate_reports_false_when_the_reload_still_cannot_reach_the_database(self, monkeypatch):
        import config
        from tasks.setup_manager import hydrate_worker_config

        monkeypatch.setattr(config, 'DB_DEFAULT_SERVER_PROJECTED', False)

        def still_down():
            raise RuntimeError('database is not up yet')

        monkeypatch.setattr(config, 'refresh_config', still_down)

        assert hydrate_worker_config() is False

    def test_hydrate_reports_false_when_the_reload_finds_no_default_server(self, monkeypatch):
        import config
        from tasks.setup_manager import hydrate_worker_config

        monkeypatch.setattr(config, 'DB_DEFAULT_SERVER_PROJECTED', False)
        monkeypatch.setattr(config, 'refresh_config', lambda: None)

        assert hydrate_worker_config() is False
