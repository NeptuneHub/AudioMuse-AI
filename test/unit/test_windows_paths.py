import os
from unittest.mock import patch

from test.unit.conftest import _import_module


def _load_paths():
    return _import_module('windows.paths', 'native-build/windows/paths.py')


def _norm(path):
    return path.replace('\\', '/')


class TestAppSupportDir:
    def test_space_free_localappdata_used_directly(self, monkeypatch):
        mod = _load_paths()
        monkeypatch.setenv('LOCALAPPDATA', 'C:\\Users\\tester\\AppData\\Local')
        with patch('os.makedirs') as mk:
            result = mod.app_support_dir()
        assert _norm(result) == 'C:/Users/tester/AppData/Local/AudioMuse-AI'
        assert 'AudioMuse-AI' in result
        assert ' ' not in result
        mk.assert_called_once_with(result, exist_ok=True)

    def test_localappdata_with_space_falls_back_to_programdata(self, monkeypatch):
        mod = _load_paths()
        monkeypatch.setenv('LOCALAPPDATA', 'C:\\Users\\John Doe\\AppData\\Local')
        monkeypatch.delenv('PROGRAMDATA', raising=False)
        with patch('os.makedirs') as mk:
            result = mod.app_support_dir()
        assert _norm(result) == 'C:/ProgramData/AudioMuse-AI'
        assert ' ' not in result
        mk.assert_called_once_with(result, exist_ok=True)

    def test_programdata_env_overrides_fallback_root(self, monkeypatch):
        mod = _load_paths()
        monkeypatch.setenv('LOCALAPPDATA', 'C:\\Users\\John Doe\\AppData\\Local')
        monkeypatch.setenv('PROGRAMDATA', 'D:\\SharedData')
        with patch('os.makedirs') as mk:
            result = mod.app_support_dir()
        assert _norm(result) == 'D:/SharedData/AudioMuse-AI'
        mk.assert_called_once_with(result, exist_ok=True)

    def test_missing_localappdata_uses_home_profile(self, monkeypatch):
        mod = _load_paths()
        monkeypatch.delenv('LOCALAPPDATA', raising=False)
        with patch('os.path.expanduser', return_value='/home/tester') as exp, \
                patch('os.makedirs') as mk:
            result = mod.app_support_dir()
        exp.assert_called_once_with('~')
        expected = os.path.join('/home/tester', 'AppData', 'Local', 'AudioMuse-AI')
        assert result == expected
        assert 'AudioMuse-AI' in result
        mk.assert_called_once_with(result, exist_ok=True)

    def test_missing_localappdata_home_with_space_falls_back(self, monkeypatch):
        mod = _load_paths()
        monkeypatch.delenv('LOCALAPPDATA', raising=False)
        monkeypatch.delenv('PROGRAMDATA', raising=False)
        with patch('os.path.expanduser', return_value='/home/John Doe'), \
                patch('os.makedirs') as mk:
            result = mod.app_support_dir()
        assert _norm(result) == 'C:/ProgramData/AudioMuse-AI'
        assert ' ' not in result
        mk.assert_called_once_with(result, exist_ok=True)
