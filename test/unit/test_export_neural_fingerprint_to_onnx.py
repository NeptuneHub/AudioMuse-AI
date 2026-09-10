# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Unit tests for scripts/onnx_export/export_neural_fingerprint_to_onnx.py.

Main Features:
* The checkpoint config the export reads must describe the same front end the
  runtime computes, otherwise the exported graph would get the wrong patches.
* The script stays importable and answers --help without TensorFlow installed.
"""

import importlib.util
import os
import sys

import pytest

import config
from tasks import neural_fingerprint

_SCRIPT = os.path.join(
    os.path.dirname(config.__file__), 'scripts', 'onnx_export', 'export_neural_fingerprint_to_onnx.py'
)
_SAMPLE_CONFIG = """MODEL:
  ARCHITECTURE:
    BN: layer_norm2d
    EMB_SZ: 128
  AUDIO:
    FS: 8000
    SEGMENT_DUR: 1.0
  INPUT:
    F_MAX: 4000.0
    F_MIN: 160.0
    N_MELS: 256
    SCALE: true
    STFT_HOP: 256
    STFT_WIN: 1024
    DYNAMIC_RANGE: 80
TRAIN:
  AUDIO:
    SEGMENT_HOP_DUR: 0.5
"""


@pytest.fixture(scope='module')
def export_module():
    spec = importlib.util.spec_from_file_location('export_neural_fingerprint_to_onnx', _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_checkpoint_config_matches_the_runtime_front_end(export_module):
    cfg = export_module.parse_checkpoint_config(_SAMPLE_CONFIG)

    assert cfg['emb_sz'] == neural_fingerprint.DIM
    assert cfg['norm'] == 'layer_norm2d'
    assert cfg['n_mels'] == neural_fingerprint.N_MELS
    assert cfg['sample_rate'] == neural_fingerprint.SAMPLE_RATE
    assert cfg['stft_win'] == neural_fingerprint.N_FFT
    assert cfg['stft_hop'] == neural_fingerprint.HOP_LENGTH
    assert int(cfg['segment_seconds'] * cfg['sample_rate']) == neural_fingerprint.SEGMENT_SAMPLES
    assert export_module.DEFAULT_FRAMES == neural_fingerprint.SEGMENT_FRAMES


def test_missing_config_key_is_a_loud_error(export_module):
    broken = _SAMPLE_CONFIG.replace('EMB_SZ: 128', 'EMBED: 128')
    with pytest.raises(SystemExit, match='EMB_SZ'):
        export_module.parse_checkpoint_config(broken)


def test_default_output_lands_in_the_git_ignored_model_directory(export_module):
    default = os.path.normcase(os.path.abspath(export_module.default_output()))
    expected = os.path.join(os.path.dirname(config.__file__), 'model', 'neural_fingerprint.onnx')

    assert default == os.path.normcase(os.path.abspath(expected))


def test_help_needs_no_tensorflow(export_module):
    with pytest.raises(SystemExit) as raised:
        export_module.main(['--help'])

    assert raised.value.code == 0
    assert 'tensorflow' not in sys.modules or 'tf2onnx' not in sys.modules
