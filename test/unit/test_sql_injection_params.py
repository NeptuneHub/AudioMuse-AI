import importlib.util
import os
import sys
import types
from unittest.mock import MagicMock, patch

from flask import Flask


EVIL = "x'; DROP TABLE score; --"


def _repo_path(*parts):
    root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
    return os.path.join(root, *parts)


def _import_app_external():
    if 'app_external' in sys.modules:
        return sys.modules['app_external']
    fake_vm = types.ModuleType('tasks.ivf_manager')
    fake_vm.search_tracks_unified = MagicMock(return_value=[])
    stubs = {'tasks.ivf_manager': fake_vm}
    if 'tasks' not in sys.modules:
        stubs['tasks'] = types.ModuleType('tasks')
    with patch.dict(sys.modules, stubs):
        import app_external
    return app_external


def _import_app_waveform():
    import app_waveform
    return app_waveform


def _recording_db():
    cur = MagicMock()
    cur.fetchone.return_value = None
    cur.fetchall.return_value = []
    cur.__enter__ = lambda self: self
    cur.__exit__ = lambda self, *a: None
    db = MagicMock()
    db.cursor.return_value = cur
    return db, cur


class TestItemIdEndpointsParameterized:
    def test_score_endpoint_passes_id_as_param(self):
        import app_helper
        ext = _import_app_external()
        app = Flask(__name__)
        app.register_blueprint(ext.external_bp)
        app.config['TESTING'] = True
        db, cur = _recording_db()
        with patch.object(app_helper, 'get_db', return_value=db):
            resp = app.test_client().get('/get_score', query_string={'id': EVIL})
        assert resp.status_code == 404
        sql = cur.execute.call_args[0][0]
        params = cur.execute.call_args[0][1]
        assert '%s' in sql
        assert params == (EVIL,)
        assert EVIL not in sql

    def test_embedding_endpoint_passes_id_as_param(self):
        import app_helper
        ext = _import_app_external()
        app = Flask(__name__)
        app.register_blueprint(ext.external_bp)
        app.config['TESTING'] = True
        db, cur = _recording_db()
        with patch.object(app_helper, 'get_db', return_value=db):
            resp = app.test_client().get('/get_embedding', query_string={'id': EVIL})
        assert resp.status_code == 404
        sql = cur.execute.call_args[0][0]
        params = cur.execute.call_args[0][1]
        assert '%s' in sql
        assert params == (EVIL,)
        assert EVIL not in sql

    def test_waveform_endpoint_passes_item_id_as_param(self):
        wf = _import_app_waveform()
        app = Flask(__name__)
        app.register_blueprint(wf.waveform_bp)
        app.config['TESTING'] = True
        db, cur = _recording_db()
        with patch.object(wf, 'get_db', return_value=db):
            resp = app.test_client().get('/api/waveform', query_string={'item_id': EVIL})
        assert resp.status_code == 404
        sql = cur.execute.call_args[0][0]
        params = cur.execute.call_args[0][1]
        assert '%s' in sql
        assert params == (EVIL,)
        assert EVIL not in sql


class TestToolImplInClauseParameterized:
    def _load_tool_impl(self):
        if 'tasks.ai.tool_impl' in sys.modules:
            return sys.modules['tasks.ai.tool_impl']
        fake_mcp = types.ModuleType('tasks.mcp_helper')
        fake_mcp.get_db_connection = MagicMock()
        stubs = {'tasks.mcp_helper': fake_mcp}
        for parent in ('tasks', 'tasks.ai'):
            if parent not in sys.modules:
                stubs[parent] = types.ModuleType(parent)
        with patch.dict(sys.modules, stubs):
            spec = importlib.util.spec_from_file_location(
                'tasks.ai.tool_impl', _repo_path('tasks', 'ai', 'tool_impl.py'))
            mod = importlib.util.module_from_spec(spec)
            sys.modules['tasks.ai.tool_impl'] = mod
            spec.loader.exec_module(mod)
        return mod

    def test_song_similarity_in_clause_is_parameterized(self):
        tool_impl = self._load_tool_impl()
        seed_id = 'seed-1'
        similar_ids = ['id-a', 'id-b']

        cur = MagicMock()
        cur.__enter__ = lambda self: self
        cur.__exit__ = lambda self, *a: None
        cur.fetchone.return_value = {
            'item_id': seed_id, 'title': 'T', 'author': 'A', 'album': ''}
        cur.fetchall.return_value = [
            {'item_id': 'id-a', 'title': 'Ta', 'author': 'Aa', 'album': ''},
            {'item_id': 'id-b', 'title': 'Tb', 'author': 'Ab', 'album': ''},
        ]
        conn = MagicMock()
        conn.cursor.return_value = cur

        fake_vm = types.ModuleType('tasks.ivf_manager')
        fake_vm.find_nearest_neighbors_by_id = MagicMock(
            return_value=[{'item_id': s} for s in similar_ids + [seed_id]])
        vm_stubs = {'tasks.ivf_manager': fake_vm}
        if 'tasks' not in sys.modules:
            vm_stubs['tasks'] = types.ModuleType('tasks')

        with patch.object(tool_impl, 'get_db_connection', return_value=conn), \
                patch.dict(sys.modules, vm_stubs):
            result = tool_impl._song_similarity_api_sync('Song', 'Artist', 2)

        assert 'songs' in result
        in_calls = [c for c in cur.execute.call_args_list if 'IN (' in c[0][0]]
        assert len(in_calls) == 1
        sql = in_calls[0][0][0]
        params = in_calls[0][0][1]
        assert sql.count('%s') == len(similar_ids)
        assert list(params) == similar_ids
        assert 'id-a' not in sql
        assert 'id-b' not in sql
