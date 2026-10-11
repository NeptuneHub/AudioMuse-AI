# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Model space ids of the P2P node (p2p.spaces).

Two servers exchange a model's results only when their space ids match, so an
id must change with every weight file and parameter that changes the result
and must not change with anything else. The tests use small stand-in model
files in a temporary directory.

Main Features:
* a space id is 32 bytes, stable for the same files and parameters, and
  changes with a file's content or any parameter
* a missing model file turns that space off (None) and says so in the log
* CLAP includes its external .data file when present; neural includes the
  local codebook id; lyrics includes the GTE tokenizer and the axis texts
* the lyrics axes and GTE default paths come from tasks.lyrics_constants, the
  same objects the lyrics package uses, without importing the lyrics package
  (which would load librosa) and without reading any .py source (a frozen
  build has none)
* compute_spaces returns all four ids and the local codebook id
"""

import builtins
import logging
import os
import subprocess
import sys

import numpy as np
import pytest

import config
from lyrics import gte_onnx, lyrics_transcriber
from p2p import spaces
from tasks import lyrics_constants, nfp_format

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _file(tmp_path, name, content):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return str(path)


@pytest.fixture
def models(tmp_path, monkeypatch):
    spaces._DIGESTS.clear()
    book = np.random.default_rng(0).standard_normal(
        (nfp_format.PQ_SUBSPACES, nfp_format.PQ_CENTROIDS, nfp_format.PQ_SUBDIM)).astype(np.float32)
    codebook = tmp_path / 'pq.npz'
    np.savez(codebook, codebook=book)
    paths = {
        'EMBEDDING_MODEL_PATH': _file(tmp_path, 'musicnn.onnx', b'musicnn weights'),
        'CLAP_AUDIO_MODEL_PATH': _file(tmp_path, 'clap.onnx', b'clap weights'),
        'NEURAL_FINGERPRINT_MODEL_PATH': _file(tmp_path, 'neural.onnx', b'neural weights'),
        'NEURAL_FINGERPRINT_CODEBOOK_PATH': str(codebook),
    }
    for key, value in paths.items():
        monkeypatch.setattr(config, key, value)
    monkeypatch.setenv('LYRICS_GTE_ONNX_PATH', _file(tmp_path, 'gte.onnx', b'gte weights'))
    tokenizer_dir = tmp_path / 'gte'
    _file(tmp_path, 'gte/tokenizer.json', b'{"model": "stand-in"}')
    monkeypatch.setenv('LYRICS_GTE_TOKENIZER_DIR', str(tokenizer_dir))
    yield tmp_path, nfp_format.codebook_id(book)
    spaces._DIGESTS.clear()


class TestSpaceIds:
    def test_an_id_is_32_bytes_and_stable(self, tmp_path):
        path = _file(tmp_path, 'model.onnx', b'weights')

        first = spaces.space_id('clap', [path], {'n_mels': 128})
        second = spaces.space_id('clap', [path], {'n_mels': 128})

        assert isinstance(first, bytes) and len(first) == 32
        assert first == second

    def test_a_parameter_change_changes_the_id(self, tmp_path):
        path = _file(tmp_path, 'model.onnx', b'weights')

        assert spaces.space_id('clap', [path], {'n_mels': 128}) != spaces.space_id('clap', [path], {'n_mels': 64})

    def test_a_weight_change_changes_the_id(self, tmp_path):
        first = spaces.space_id('clap', [_file(tmp_path, 'a.onnx', b'weights one')], {})
        second = spaces.space_id('clap', [_file(tmp_path, 'b.onnx', b'weights two')], {})

        assert first != second

    def test_the_file_name_does_not_matter(self, tmp_path):
        first = spaces.space_id('clap', [_file(tmp_path, 'a.onnx', b'same weights')], {})
        second = spaces.space_id('clap', [_file(tmp_path, 'b.onnx', b'same weights')], {})

        assert first == second

    def test_the_model_name_is_part_of_the_id(self, tmp_path):
        path = _file(tmp_path, 'a.onnx', b'weights')

        assert spaces.space_id('clap', [path], {}) != spaces.space_id('neural', [path], {})

    def test_a_missing_file_turns_the_space_off(self, tmp_path, caplog):
        with caplog.at_level(logging.WARNING, logger='p2p.spaces'):
            assert spaces.space_id('clap', [str(tmp_path / 'absent.onnx')], {}) is None
        assert 'missing' in caplog.text


class TestModelSpaces:
    def test_compute_spaces_returns_all_four_and_the_codebook(self, models):
        _tmp, book_id = models

        found, crc = spaces.compute_spaces()

        assert crc == book_id
        assert sorted(found) == ['c', 'l', 'm', 'n']
        assert all(isinstance(value, bytes) and len(value) == 32 for value in found.values())
        assert len(set(found.values())) == 4

    def test_the_clap_data_file_is_part_of_the_clap_space(self, models, monkeypatch):
        tmp_path, _book = models
        before = spaces.clap_space()
        _file(tmp_path, 'clap.onnx.data', b'external tensors')

        assert spaces.clap_space() != before

    def test_a_clap_setting_changes_the_clap_space(self, models, monkeypatch):
        before = spaces.clap_space()
        monkeypatch.setattr(config, 'CLAP_AUDIO_HOP_LENGTH', config.CLAP_AUDIO_HOP_LENGTH + 1)

        assert spaces.clap_space() != before

    def test_the_gte_token_limit_changes_the_lyrics_space(self, models, monkeypatch):
        before = spaces.lyrics_space()
        monkeypatch.setattr(config, 'LYRICS_GTE_MAX_TOKENS', config.LYRICS_GTE_MAX_TOKENS + 1)

        assert spaces.lyrics_space() != before

    def test_the_tokenizer_is_part_of_the_lyrics_space(self, models):
        tmp_path, _book = models
        before = spaces.lyrics_space()
        _file(tmp_path, 'gte/tokenizer.json', b'{"model": "another"}')

        assert spaces.lyrics_space() != before

    def test_another_codebook_changes_the_neural_space(self, models):
        _tmp, book_id = models

        assert spaces.neural_space(book_id) != spaces.neural_space(book_id + 1)
        assert spaces.neural_space(None) is None

    def test_a_missing_codebook_turns_neural_sharing_off(self, models, monkeypatch, tmp_path):
        monkeypatch.setattr(config, 'NEURAL_FINGERPRINT_CODEBOOK_PATH', str(tmp_path / 'absent.npz'))

        found, crc = spaces.compute_spaces()

        assert crc is None and found['n'] is None
        assert found['m'] is not None

    def test_a_missing_musicnn_model_turns_matching_off(self, models, monkeypatch, tmp_path):
        monkeypatch.setattr(config, 'EMBEDDING_MODEL_PATH', str(tmp_path / 'absent.onnx'))

        assert spaces.compute_spaces()[0]['m'] is None

    def test_an_axis_text_change_changes_the_lyrics_space(self, models, monkeypatch):
        before = spaces.lyrics_space()
        axes = {name: dict(axis) for name, axis in lyrics_constants.MUSIC_ANALYSIS_AXES.items()}
        first = next(iter(axes))
        axes[first]['description'] = axes[first]['description'] + ' (changed)'
        monkeypatch.setattr(lyrics_constants, 'MUSIC_ANALYSIS_AXES', axes)

        assert spaces.lyrics_space() != before

    def test_the_gte_defaults_apply_without_the_environment_overrides(self, models, monkeypatch, tmp_path):
        onnx = _file(tmp_path, 'default/gte.onnx', b'gte weights')
        _file(tmp_path, 'default/gte/tokenizer.json', b'{"model": "stand-in"}')
        monkeypatch.delenv('LYRICS_GTE_ONNX_PATH')
        monkeypatch.delenv('LYRICS_GTE_TOKENIZER_DIR')
        monkeypatch.setattr(lyrics_constants, 'GTE_DEFAULT_ONNX_PATH', onnx)
        monkeypatch.setattr(lyrics_constants, 'GTE_DEFAULT_TOKENIZER_DIR', str(tmp_path / 'default' / 'gte'))

        assert spaces.lyrics_space() is not None
        monkeypatch.setattr(lyrics_constants, 'GTE_DEFAULT_ONNX_PATH', str(tmp_path / 'absent.onnx'))
        assert spaces.lyrics_space() is None


class TestLyricsConstants:
    def test_the_lyrics_package_uses_the_same_axes(self):
        axes = lyrics_constants.MUSIC_ANALYSIS_AXES

        assert lyrics_transcriber.MUSIC_ANALYSIS_AXES is axes
        assert [len(axis['labels']) for axis in axes.values()] == [6, 6, 6, 5, 4]
        assert all(isinstance(axis['description'], str) for axis in axes.values())

    def test_the_gte_runtime_uses_the_same_paths_and_overrides(self, monkeypatch):
        assert gte_onnx._DEFAULT_ONNX_PATH == lyrics_constants.GTE_DEFAULT_ONNX_PATH
        assert gte_onnx._DEFAULT_TOKENIZER_DIR == lyrics_constants.GTE_DEFAULT_TOKENIZER_DIR
        assert lyrics_constants.GTE_DEFAULT_ONNX_PATH.endswith('.onnx')

        monkeypatch.setenv(lyrics_constants.GTE_ONNX_PATH_ENV, '/models/gte.onnx')
        monkeypatch.setenv(lyrics_constants.GTE_TOKENIZER_DIR_ENV, '/models/gte')

        assert gte_onnx._resolve_onnx_path() == '/models/gte.onnx'
        assert gte_onnx._resolve_tokenizer_dir() == '/models/gte'

    def test_the_lyrics_space_never_opens_a_python_source(self, models, monkeypatch):
        opened = []
        real_open = builtins.open

        def watching(path, *args, **kwargs):
            opened.append(str(path))
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr(builtins, 'open', watching)

        assert spaces.lyrics_space() is not None
        assert opened and not [path for path in opened if path.endswith('.py')]

    def test_reading_the_spaces_never_imports_the_lyrics_package(self):
        code = (
            "import sys\n"
            "from p2p import spaces\n"
            "spaces.lyrics_space()\n"
            "print('lyrics' in sys.modules, 'librosa' in sys.modules)\n"
        )
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                                cwd=REPO_ROOT, timeout=120)

        assert result.returncode == 0, result.stderr
        assert result.stdout.strip().splitlines()[-1] == 'False False'
