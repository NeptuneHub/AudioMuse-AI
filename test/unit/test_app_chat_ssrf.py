import sys
import types
import logging
from unittest.mock import patch

import pytest
from flask import Flask


def _ensure_flasgger():
    try:
        import flasgger  # noqa: F401

        return
    except ImportError:
        pass
    fake = types.ModuleType('flasgger')

    def swag_from(*a, **k):
        def deco(f):
            return f

        return deco

    fake.swag_from = swag_from
    fake.Swagger = lambda *a, **k: None
    sys.modules['flasgger'] = fake


@pytest.fixture
def app_chat_mod():
    _ensure_flasgger()
    import app_chat

    return app_chat


@pytest.fixture
def client(app_chat_mod):
    app = Flask(__name__)
    app.register_blueprint(app_chat_mod.chat_bp)
    app.config['TESTING'] = True
    return app.test_client()


def _install_fakes(planner_calls):
    fake_tools = types.ModuleType('tasks.ai.tools')
    fake_tools.get_mcp_tools = lambda: [{'name': 'search_database'}]

    def _planner(*a, **k):
        planner_calls.append((a, k))
        if False:
            yield
        return {'error': 'stub'}

    fake_planner = types.ModuleType('tasks.ai.planner')
    fake_planner.plan_and_execute_once = _planner

    fake_mcp = types.ModuleType('tasks.mcp_helper')
    fake_mcp.get_library_context = lambda: {'total_songs': 0}

    stubs = {
        'tasks.ai.tools': fake_tools,
        'tasks.ai.planner': fake_planner,
        'tasks.mcp_helper': fake_mcp,
    }
    for parent in ('tasks', 'tasks.ai'):
        if parent not in sys.modules:
            stubs[parent] = types.ModuleType(parent)
    return stubs


class TestChatEndpointUrlAcceptance:
    def test_lan_url_reaches_planner(self, client):
        planner_calls = []
        stubs = _install_fakes(planner_calls)
        with patch.dict(sys.modules, stubs):
            resp = client.post(
                '/api/chatPlaylist',
                json={
                    'userInput': 'songs',
                    'ai_provider': 'OLLAMA',
                    'ollama_server_url': 'http://192.168.1.10:11434',
                },
            )
        assert resp.status_code != 400
        assert planner_calls

    def test_default_url_when_field_omitted_not_blocked(self, client):
        planner_calls = []
        stubs = _install_fakes(planner_calls)
        with patch.dict(sys.modules, stubs):
            resp = client.post(
                '/api/chatPlaylist', json={'userInput': 'songs', 'ai_provider': 'OLLAMA'}
            )
        assert resp.status_code != 400
        assert planner_calls


class TestChatLogMasking:
    def test_api_keys_masked_in_debug_log(self, client, caplog):
        planner_calls = []
        stubs = _install_fakes(planner_calls)
        with patch.dict(sys.modules, stubs), caplog.at_level(logging.DEBUG, logger='app_chat'):
            client.post(
                '/api/chatPlaylist',
                json={
                    'userInput': 'songs',
                    'ai_provider': 'NONE',
                    'gemini_api_key': 'gm-SECRET-123',
                    'openai_api_key': 'oa-SECRET-456',
                    'mistral_api_key': 'ms-SECRET-789',
                },
            )
        debug_text = "\n".join(r.getMessage() for r in caplog.records)
        assert 'API-KEY' in debug_text
        assert 'gm-SECRET-123' not in debug_text
        assert 'oa-SECRET-456' not in debug_text
        assert 'ms-SECRET-789' not in debug_text
